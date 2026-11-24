"""
SegMamba3D 训练器
================
对应模型：SegMamba3D（nnunetv2/training/network/SegMamba3D.py）

训练超参对齐官方 ge-xing/SegMamba ``3_train.py``（BraTSTrainer）::
  - SGD lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
  - PolyLR exponent=0.9，无 warmup（按 epoch 步进；官方按 iter 步进，
    因 nnU-Net 在 on_train_epoch_end 调 step，故 max_steps=num_epochs）
  - max_epoch=1000, iterations/epoch=250
  - patch_size / batch_size：跟 nnU-Net plans，但各维向上对齐到 16 的倍数
    （SegMamba 固定 4×stride-2；非整除会在 UnetrUpBlock cat 处崩）
  - 损失：CrossEntropy（与官方一致；非 nnU-Net 默认 Dice+CE）
  - 无深度监督；无 grad clip（官方 training_step 无裁剪）

用法：
  nnUNetv2_train 1251 3d_fullres 0 -tr nnUNetTrainerSegMamba3D
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.network.SegMamba3D import SegMamba3D
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


def _snap_patch_divisible(patch_size: list, factor: int = 16) -> list:
    """Ceil each dim to a multiple of ``factor`` (min ``factor``). Same idea as UNETR trainer."""
    out = []
    for d in patch_size:
        d = int(d)
        if d < factor or d % factor != 0:
            d = max(factor, ((d + factor - 1) // factor) * factor)
        out.append(d)
    return out


class _SegMambaCELoss(nn.Module):
    """官方 3_train.py：``nn.CrossEntropyLoss()``；兼容 nnU-Net (B,1,*) target。"""

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, (list, tuple)):
            target = target[0]
        if target.ndim == net_output.ndim:
            target = target[:, 0]
        return self.ce(net_output, target.long())


class nnUNetTrainerSegMamba3D(nnUNetTrainerNoDeepSupervision):
    """SegMamba3D：优化器/调度/损失对齐官方 SegMamba ``3_train.py``；patch 用 plans。"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_output_channels = self.label_manager.num_segmentation_heads

        # SegMamba encoder: 4× stride-2 → each dim must be divisible by 16
        # (e.g. 56→28→14→7→3; upsample 3→6 vs skip 7 → cat crash).
        original_patch = list(self.configuration_manager.patch_size)
        self.img_size = _snap_patch_divisible(original_patch, factor=16)
        self.configuration_manager.configuration["patch_size"] = self.img_size
        self._update_plans_with_segmamba_config(self.img_size, configuration)

        self.arch_init_kwargs = {
            "patch_size": self.img_size,
            "spatial_dims": 3,
        }
        if self.img_size != original_patch:
            self.print_to_log_file(
                f"SegMamba3D patch snapped {original_patch} → {self.img_size} "
                f"(need %16==0 for 4× downsample)"
            )
        self.print_to_log_file(
            f"SegMamba3D patch_size={self.img_size}, spatial_dims=3 (from plans, snapped)"
        )

        self.grad_scaler = None
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.current_epoch = 0
        self.enable_deep_supervision = False
        self.save_every = 50

        # 官方 BraTSTrainer
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.momentum = 0.99
        self.lr_exponent = 0.9

    def _update_plans_with_segmamba_config(
        self, patch_size: list, configuration: str
    ):
        """写入内存 plans（随输出目录 plans.json 保存）；同步 dataloader 用的 patch_size。"""
        try:
            plans = self.plans_manager.plans
            if "configurations" not in plans or configuration not in plans["configurations"]:
                return
            cfg = plans["configurations"][configuration]
            cfg["patch_size"] = list(patch_size)
            arch = cfg.setdefault("architecture", {})
            arch_kwargs = arch.get("arch_kwargs") or arch.get("network_arch_init_kwargs") or {}
            arch["arch_kwargs"] = arch_kwargs
            arch_kwargs["patch_size"] = list(patch_size)
            arch_kwargs["spatial_dims"] = 3
            self.print_to_log_file(
                f"Plans arch_kwargs synced (output dir): "
                f"patch_size={patch_size}, batch_size={cfg.get('batch_size')}"
            )
        except Exception as e:
            self.print_to_log_file(f"Warning: could not update plans: {e}")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        # 与 nnUNetTrainerSegMambaV2 相同：按 epoch step，max_steps=num_epochs
        lr_scheduler = PolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            max_steps=self.num_epochs,
            exponent=self.lr_exponent,
        )
        self.print_to_log_file(
            f"SegMamba3D optimizer=SGD(official) lr={self.initial_lr}, "
            f"wd={self.weight_decay}, momentum={self.momentum}, nesterov=True | "
            f"PolyLR(max_steps={self.num_epochs} epochs, exp={self.lr_exponent}, no warmup) | "
            f"loss=CrossEntropy"
        )
        return optimizer, lr_scheduler

    def _build_loss(self):
        if self.label_manager.has_regions:
            self.print_to_log_file(
                "SegMamba3D: regions mode → keep nnU-Net DC+BCE "
                "(official CE assumes discrete labels)"
            )
            return super()._build_loss()
        return _SegMambaCELoss()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        # 官方无 grad clip
        self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)
            del data
            l = self.loss(output, target)

            axes = [0] + list(range(2, output.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(
                    output.shape, device=output.device, dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    target = target.clone()
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target[:, -1:]
                    target = target[:, :-1]
            else:
                mask = None

            if not self.label_manager.has_regions and target.ndim == output.ndim:
                # one-hot for metric
                tgt = target[:, 0] if target.shape[1] == 1 else target.argmax(1)
                target_oh = torch.zeros_like(predicted_segmentation_onehot)
                target_oh.scatter_(1, tgt[:, None].long(), 1)
                target = target_oh

            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot, target, axes=axes, mask=mask
            )

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"lr: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_train_epoch_end(self, train_outputs: list):
        super().on_train_epoch_end(train_outputs)
        self.lr_scheduler.step(self.current_epoch + 1)

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        return SegMamba3D(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            hidden_size=768,
            spatial_dims=3,
            deep_supervision=False,
        )

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            arch_init_kwargs = dict(
                self.configuration_manager.network_arch_init_kwargs
            )
            arch_init_kwargs["patch_size"] = self.img_size
            arch_init_kwargs["spatial_dims"] = 3

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "self.initialize() was called more than once. This should not happen."
            )

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False
        if hasattr(self.network, "deep_supervision"):
            self.network.deep_supervision = False
