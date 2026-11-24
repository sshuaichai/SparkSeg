"""
UMamba 训练器基类
=================
对应模型：UMamba（nnunetv2/training/network/UMamba.py）
论文参考：arXiv:2401.04722  https://arxiv.org/abs/2401.04722
         "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation"

架构变体：
  UMambaBot  (variant='bot') — Mamba 仅置于瓶颈层，编码器为纯残差卷积
  UMambaEnc  (variant='enc') — Mamba 置于编码器每个阶段

训练配置（与官方 UMamba 保持一致）：
  优化器  : AdamW，lr=1e-4，weight_decay=3e-5，eps=1e-4
             （eps=1e-4 防止 Mamba SSM fp16 下溢）
  LR 调度 : PolyLR decay，exponent=0.9，每 epoch 更新
  Epochs  : 1000（nnUNet 标准），原文约 500–1000 epoch
  损失    : DC + CE（nnUNet 默认，与原文 Dice+CE 一致）
  深度监督: 支持开启/关闭（UMamba decoder 原生支持）
  梯度裁剪: max_norm=12（与 nnUNet 标准配置一致）

子类通过覆盖 UMAMBA_VARIANT 和 ENABLE_DEEP_SUPERVISION 指定行为：
  nnUNetTrainerUMambaBotDS   → variant='bot', deep_supervision=True
  nnUNetTrainerUMambaBotNoDS → variant='bot', deep_supervision=False
  nnUNetTrainerUMambaEncDS   → variant='enc', deep_supervision=True
  nnUNetTrainerUMambaEncNoDS → variant='enc', deep_supervision=False
"""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.network.UMamba import (
    UMamba,
    UMambaConfig,
    get_umamba_config,
    create_umamba_from_config,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainerUMambaBase(nnUNetTrainer):
    """
    UMamba 训练器基类。

    训练配置：
      - AdamW，lr=1e-4，weight_decay=3e-5，eps=1e-4
      - PolyLR decay，exponent=0.9，每 epoch 更新
      - 梯度裁剪：max_norm=12
      - 损失：DC + CE（nnUNet 默认）
      - 深度监督：由子类 ENABLE_DEEP_SUPERVISION 控制

    子类需覆盖：
      UMAMBA_VARIANT         : 'bot' 或 'enc'
      ENABLE_DEEP_SUPERVISION: True  或 False
    """

    # 子类覆盖这两个属性
    UMAMBA_VARIANT: str = 'enc'          # 'bot' | 'enc'
    ENABLE_DEEP_SUPERVISION: bool = True

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
        self.img_size = self.configuration_manager.patch_size

        self.umamba_variant = self.UMAMBA_VARIANT
        self.enable_deep_supervision = self.ENABLE_DEEP_SUPERVISION

        # 将配置信息写入 plans，供 initialize() 读取
        self._update_plans_with_umamba_config(
            self.img_size, self.umamba_variant, configuration
        )

        self.arch_init_kwargs = {
            "patch_size": self.img_size,
            "umamba_variant": self.umamba_variant,
        }
        self.print_to_log_file(
            f"UMamba variant='{self.umamba_variant}', "
            f"deep_supervision={self.enable_deep_supervision}, "
            f"patch_size={self.img_size}"
        )

        # -------- 训练超参数 --------
        self.grad_scaler = None          # Mamba SSM 要求 fp32，不使用 AMP GradScaler
        self.num_epochs = 200
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.current_epoch = 0
        self.save_every = 50

        self.initial_lr = 1e-4
        self.weight_decay = 3e-5

    # ------------------------------------------------------------------
    # Plans 持久化
    # ------------------------------------------------------------------
    def _update_plans_with_umamba_config(
        self, new_patch_size: list, variant: str, configuration: str
    ):
        """仅更新内存中的 plans（会随训练保存到输出目录）。不写回预处理目录，避免不同 Trainer 互相覆盖。"""
        try:
            plans = self.plans_manager.plans
            if "configurations" not in plans or configuration not in plans["configurations"]:
                return
            cfg = plans["configurations"][configuration]
            arch = cfg.setdefault("architecture", {})
            arch_kwargs = arch.setdefault("network_arch_init_kwargs", {})
            cfg["patch_size"] = new_patch_size
            cfg["umamba_variant"] = variant
            arch_kwargs["patch_size"] = new_patch_size
            arch_kwargs["umamba_variant"] = variant
            # Persist plans strides so infer rebuild (without trainer.initialize)
            # does not fall back to isotropic pools (ACDC D=10 → SW 32 vs 10).
            try:
                strides = self.configuration_manager.pool_op_kernel_sizes
                arch_kwargs["pool_op_kernel_sizes"] = [list(s) for s in strides]
            except (KeyError, TypeError, AttributeError):
                pass
            try:
                ks = (
                    self.configuration_manager.configuration.get("architecture", {})
                    .get("arch_kwargs", {})
                    .get("kernel_sizes")
                )
                if ks is not None:
                    arch_kwargs["conv_kernel_sizes"] = [list(k) for k in ks]
            except (KeyError, TypeError, AttributeError):
                pass
            self.print_to_log_file(
                f"Plans updated in memory (output dir only): patch_size={new_patch_size}, umamba_variant='{variant}'"
            )
        except Exception as e:
            self.print_to_log_file(f"Warning: could not update plans: {e}")

    # ------------------------------------------------------------------
    # 优化器 & LR 调度
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """
        AdamW + PolyLR decay（每 epoch 更新）。

        eps=1e-4 防止混合精度下 Mamba SSM 数值下溢。
        weight_decay=3e-5 参照原版 UMamba 官方实现。
        """
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4,
        )
        lr_scheduler = PolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            max_steps=self.num_epochs,
            exponent=0.9,
        )
        self.print_to_log_file(
            f"UMamba-{self.umamba_variant.upper()} optimizer: "
            f"AdamW lr={self.initial_lr}, wd={self.weight_decay}, eps=1e-4 | "
            f"PolyLR(max_steps={self.num_epochs}, exp=0.9)"
        )
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------
    # 训练步骤
    # ------------------------------------------------------------------
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
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # 验证步骤
    # ------------------------------------------------------------------
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

            # 深度监督：取最高分辨率输出用于 loss 和 metric 计算
            if isinstance(output, (list, tuple)):
                output = output[0]

            l = self.loss(output, target if not isinstance(target, list) else target[0])

            axes = [0] + list(range(2, output.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (
                    (torch.sigmoid(output) > 0.5).long()
                )
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(
                    output.shape, device=output.device, dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            target_val = target if not isinstance(target, list) else target[0]

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target_val != self.label_manager.ignore_label).float()
                    target_val[target_val == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target_val[:, -1:]
                    target_val = target_val[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot, target_val, axes=axes, mask=mask
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

    # ------------------------------------------------------------------
    # LR 调度：每 epoch 更新一次
    # ------------------------------------------------------------------
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
        self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # 网络构建
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        根据 arch_init_kwargs 构建 UMamba 模型。

        arch_init_kwargs 必须包含：
          - patch_size      : list/tuple，如 [128, 128, 128]（决定 spatial_dims）
          - umamba_variant  : str，'bot' 或 'enc'

        可选覆盖（透传给 UMamba.__init__）：
          - n_stages, base_num_features, max_num_features
          - n_conv_per_stage, n_conv_per_stage_decoder
          - mamba_d_state, mamba_d_conv, mamba_expand
          - conv_bias
        """
        patch_size = arch_init_kwargs.get("patch_size")
        if patch_size is None:
            raise ValueError("patch_size is required in arch_init_kwargs for UMamba")

        variant = arch_init_kwargs.get("umamba_variant", "enc")
        if variant not in ("bot", "enc"):
            raise ValueError(
                f"umamba_variant must be 'bot' or 'enc', got '{variant}'"
            )

        spatial_dims = len(patch_size)

        # 允许 arch_init_kwargs 中的额外字段覆盖默认值
        # pool_op_kernel_sizes 必须与 plans 一致，否则各向异性数据（如 ACDC D=10）会维度不匹配
        override_keys = {
            "n_stages", "base_num_features", "max_num_features",
            "n_conv_per_stage", "n_conv_per_stage_decoder",
            "pool_op_kernel_sizes", "conv_kernel_sizes",
            "mamba_d_state", "mamba_d_conv", "mamba_expand",
            "conv_bias",
        }
        extra_kwargs = {
            k: v for k, v in arch_init_kwargs.items()
            if k in override_keys
        }

        model = UMamba(
            spatial_dims=spatial_dims,
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            patch_size=tuple(patch_size),
            variant=variant,
            deep_supervision=enable_deep_supervision,
            **extra_kwargs,
        )
        return model

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            arch_init_kwargs = dict(
                self.configuration_manager.network_arch_init_kwargs
            )
            arch_init_kwargs["patch_size"] = self.configuration_manager.patch_size
            arch_init_kwargs["umamba_variant"] = self.umamba_variant
            # 使用 plans 的 strides，否则各向异性 patch（如 ACDC D=10）会 skip 维度不匹配
            try:
                arch_init_kwargs["pool_op_kernel_sizes"] = list(
                    self.configuration_manager.pool_op_kernel_sizes
                )
            except (KeyError, TypeError):
                pass  # UMamba 将使用默认 isotropic strides

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
            self.dataset_class = infer_dataset_class(
                self.preprocessed_dataset_folder
            )
            self.was_initialized = True
        else:
            raise RuntimeError(
                "self.initialize() was called more than once. "
                "This should not happen."
            )

    # ------------------------------------------------------------------
    # 深度监督开关
    # ------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        """切换网络深度监督，并同步内部标志。"""
        self.enable_deep_supervision = enabled
        net = self.network
        # 解包 DDP 或 torch.compile 包装
        if hasattr(net, "module"):
            net = net.module
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        # UMamba → self.model → UMambaBot / UMambaEnc → decoder.deep_supervision
        inner = getattr(net, "model", net)
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "deep_supervision"):
            inner.decoder.deep_supervision = enabled

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def get_variant(self) -> str:
        return self.umamba_variant


class nnUNetTrainerUMambaBotDS(nnUNetTrainerUMambaBase):
    """UMambaBot + deep supervision."""

    UMAMBA_VARIANT: str = "bot"
    ENABLE_DEEP_SUPERVISION: bool = True

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("nnUNetTrainerUMambaBotDS: UMambaBot with deep supervision enabled.")


class nnUNetTrainerUMambaBotNoDS(nnUNetTrainerUMambaBase):
    """UMambaBot, single output head."""

    UMAMBA_VARIANT: str = "bot"
    ENABLE_DEEP_SUPERVISION: bool = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("nnUNetTrainerUMambaBotNoDS: UMambaBot without deep supervision.")
        self.num_epochs = 200
        self.weight_decay = 3e-5
        self.initial_lr = 1e-4
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0
        self.hyperparameter_search_space = None
        self.enable_deep_supervision = False


class nnUNetTrainerUMambaEncDS(nnUNetTrainerUMambaBase):
    """UMambaEnc + deep supervision."""

    UMAMBA_VARIANT: str = "enc"
    ENABLE_DEEP_SUPERVISION: bool = True

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file(
            "nnUNetTrainerUMambaEncDS: UMambaEnc with deep supervision enabled. "
            "(Recommended for highest accuracy)"
        )


class nnUNetTrainerUMambaEncNoDS(nnUNetTrainerUMambaBase):
    """UMambaEnc, single output head."""

    UMAMBA_VARIANT: str = "enc"
    ENABLE_DEEP_SUPERVISION: bool = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("nnUNetTrainerUMambaEncNoDS: UMambaEnc without deep supervision.")
