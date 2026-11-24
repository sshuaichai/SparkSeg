"""Official Beckschen 3D-TransUNet trainers (nnUNetv2 wrapper).

Defaults follow ``3D-TransUNet/configs/Brats/decoder_only.yaml`` (tumor-oriented
recommended config in the paper). Encoder-only is also provided.

Network = ``Generic_TransUNet_max_ppbp`` from the vendored official clone — not the
old StackedConv hybrid (see ``TransUNet3D_LegacyHybrid.py``).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network.TransUNet3D import (
    HungarianMatcher3D,
    build_official_transunet3d,
    compute_loss_hungarian,
    get_variant_model_params,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


def _ensure_official_path():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "network" / "3D-TransUNet"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _plans_pool_and_kernels(configuration_manager: ConfigurationManager):
    """Map nnUNetv2 strides/kernels → official nnU-Net v1 pool/conv lists."""
    strides = [tuple(int(x) for x in s) for s in configuration_manager.pool_op_kernel_sizes]
    # nnUNetv2 often starts with identity stride; official pool list does not.
    if len(strides) > 1 and all(x == 1 for x in strides[0]):
        pool = strides[1:]
    else:
        pool = strides

    cfg = configuration_manager.configuration
    if "conv_kernel_sizes" in cfg:
        conv_ks = [tuple(int(x) for x in s) for s in cfg["conv_kernel_sizes"]]
    else:
        arch = cfg.get("architecture", {}).get("arch_kwargs", {})
        ks = arch.get("kernel_sizes")
        if ks is None:
            conv_ks = [(3, 3, 3)] * (len(pool) + 1)
        else:
            conv_ks = [tuple(int(x) for x in s) for s in ks]
            # If kernels still include the identity-stage entry, keep all (S = P+1).
            if len(conv_ks) == len(strides):
                pass
            elif len(conv_ks) == len(pool) + 1:
                pass
            elif len(conv_ks) > len(pool) + 1:
                conv_ks = conv_ks[-(len(pool) + 1) :]

    base = 32
    if "UNet_base_num_features" in cfg:
        base = int(cfg["UNet_base_num_features"])
    else:
        feats = cfg.get("architecture", {}).get("arch_kwargs", {}).get("features_per_stage")
        if feats:
            base = int(feats[0])
    return pool, conv_ks, base


class nnUNetTrainerTransUNet3DBase(nnUNetTrainer):
    """Official Generic_TransUNet_max_ppbp + BraTS yaml training defaults.

    可被子类覆盖的训练超参（对齐 ``configs/Brats/decoder_only.yaml``）::

        NUM_EPOCHS=1000, WARMUP_EPOCHS=10, INITIAL_LR=3e-4, WEIGHT_DECAY=1e-4
        OPTIM_NAME=adamw, LR_SCHEDULE=warmup_cosine, MAX_LOSS_CAL=v1
        NUM_ITERATIONS_PER_EPOCH=250, NUM_VAL_ITERATIONS_PER_EPOCH=50

    BraTS regions（如 Dataset1251）必须把 dataloader 的 multi-hot ``target``
    原样交给 Hungarian（官方 ``500Region`` 分支）；禁止 argmax+scatter。
    """

    TRANSUNET_VARIANT: str = "decoder_only"

    # --- train hyperparams (nnU-Net schedule; official BraTS yaml used 125@8GPU) ---
    NUM_EPOCHS: int = 200
    WARMUP_EPOCHS: int = 10
    INITIAL_LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    OPTIM_NAME: str = "adamw"
    LR_SCHEDULE: str = "warmup_cosine"  # LinearWarmupCosineAnnealingLR
    MAX_LOSS_CAL: str = "v1"  # hungarian aux weight: "v1" equal / "exp" 1/2^i
    NUM_ITERATIONS_PER_EPOCH: int = 250
    NUM_VAL_ITERATIONS_PER_EPOCH: int = 50

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = float(self.INITIAL_LR)
        self.weight_decay = float(self.WEIGHT_DECAY)
        self.num_epochs = int(self.NUM_EPOCHS)
        self.warmup_epochs = int(self.WARMUP_EPOCHS)
        self.optim_name = str(self.OPTIM_NAME)
        self.lr_schedule = str(self.LR_SCHEDULE)
        self.max_loss_cal = str(self.MAX_LOSS_CAL)
        self.num_iterations_per_epoch = int(self.NUM_ITERATIONS_PER_EPOCH)
        self.num_val_iterations_per_epoch = int(self.NUM_VAL_ITERATIONS_PER_EPOCH)
        self.enable_deep_supervision = self.TRANSUNET_VARIANT == "encoder_only"
        self._is_decoder_variant = self.TRANSUNET_VARIANT in (
            "decoder_only",
            "encoder_plus_decoder",
        )
        self.print_to_log_file(
            f"Official 3D-TransUNet variant={self.TRANSUNET_VARIANT} | "
            f"epochs={self.num_epochs} warmup={self.warmup_epochs} "
            f"iter/epoch={self.num_iterations_per_epoch} | "
            f"optim={self.optim_name} lr={self.initial_lr} wd={self.weight_decay} "
            f"schedule={self.lr_schedule} | max_loss_cal={self.max_loss_cal}"
        )

    def initialize(self):
        super().initialize()
        lm = self.label_manager
        self.print_to_log_file(
            f"TransUNet3D labels: has_regions={lm.has_regions} "
            f"num_heads={lm.num_segmentation_heads} "
            f"all_labels={getattr(lm, 'all_labels', None)}"
        )

    def configure_optimizers(self):
        _ensure_official_path()
        if self.optim_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        try:
            from nn_transunet.optimizers.lr_scheduler import (
                LinearWarmupCosineAnnealingLR,
            )

            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.num_epochs,
                eta_min=1e-6,
            )
        except Exception as e:
            self.print_to_log_file(
                f"fallback CosineAnnealingLR (official warmup import failed: {e})"
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=1e-6
            )
        return optimizer, lr_scheduler

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        _ = plans_manager
        pool, conv_ks, base = _plans_pool_and_kernels(configuration_manager)
        variant = cls.TRANSUNET_VARIANT
        do_ds = bool(enable_deep_supervision) if variant == "encoder_only" else False
        return build_official_transunet3d(
            variant=variant,
            num_input_channels=int(num_input_channels),
            num_classes=int(num_output_channels),
            patch_size=list(configuration_manager.patch_size),
            pool_op_kernel_sizes=pool,
            conv_kernel_sizes=conv_ks,
            base_num_features=base,
            deep_supervision=do_ds,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        want = self.TRANSUNET_VARIANT == "encoder_only" and enabled
        self.enable_deep_supervision = want
        if hasattr(self.network, "do_ds"):
            self.network.do_ds = want
        if hasattr(self.network, "_deep_supervision"):
            self.network._deep_supervision = want

    def _target_to_onehot(self, target, num_classes: int) -> torch.Tensor:
        """Match official ``nnUNetTrainerV2_DDP.compute_loss`` Region vs label paths."""
        if isinstance(target, (tuple, list)):
            target = target[0]
        # BraTS / regions: dataloader already yields multi-hot (B, R, D, H, W).
        if self.label_manager.has_regions:
            if target.ndim != 5 or target.shape[1] != num_classes:
                raise RuntimeError(
                    f"region target expected (B,{num_classes},D,H,W), got {tuple(target.shape)}"
                )
            return target.float()
        # Standard exclusive labels → one-hot
        if target.ndim == 5 and target.shape[1] == 1:
            t = target[:, 0]
        elif target.ndim == 4:
            t = target
        else:
            t = target[:, 0] if target.shape[1] == 1 else target.argmax(1)
        target_onehot = torch.zeros(
            (t.shape[0], num_classes, *t.shape[1:]),
            device=t.device,
            dtype=torch.float32,
        )
        target_onehot.scatter_(1, t.long().unsqueeze(1), 1.0)
        return target_onehot

    def _hungarian_loss(self, output, target) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(target, (tuple, list)):
            target = target[0]

        params = get_variant_model_params(self.TRANSUNET_VARIANT)
        is_max_ds = bool(params.get("is_max_ds", False))
        num_classes = int(self.label_manager.num_segmentation_heads)

        aux_outputs = output.get("aux_outputs", [])
        len_ds = 1 + len(aux_outputs)
        if self.max_loss_cal == "exp":
            w = np.array([1 / (2 ** i) for i in range(len_ds)], dtype=np.float64)
            w = w / w.sum()
        else:
            w = [1.0] * len_ds

        target_onehot = self._target_to_onehot(target, num_classes)
        target_sum = target_onehot.flatten(2).sum(dim=2)
        targets = []
        for b in range(len(target_onehot)):
            tgt_mask = target_onehot[b][target_sum[b] > 0]
            tgt_label = torch.nonzero(target_sum[b] > 0).squeeze(1)
            targets.append({"labels": tgt_label, "masks": tgt_mask})

        cost_weight = [2.0, 5.0, 5.0]
        matcher = HungarianMatcher3D(
            cost_class=cost_weight[0],
            cost_mask=cost_weight[1],
            cost_dice=cost_weight[2],
        )
        outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}
        loss_list = [
            w[0]
            * compute_loss_hungarian(
                outputs_without_aux,
                targets,
                0,
                matcher,
                num_classes,
                cost_weight=cost_weight,
            )
        ]
        if is_max_ds and aux_outputs:
            for i, aux in enumerate(aux_outputs[::-1]):
                loss_list.append(
                    w[i + 1]
                    * compute_loss_hungarian(
                        aux,
                        targets,
                        i + 1,
                        matcher,
                        num_classes,
                        cost_weight=cost_weight,
                    )
                )
        if self.max_loss_cal == "v1" and len(loss_list) > 1:
            return (loss_list[0] + sum(loss_list[1:]) / len(loss_list[1:])) / 2
        return sum(loss_list)

    @staticmethod
    def _dict_output_to_soft_maps(output) -> torch.Tensor:
        """Official semantic_inference: soft class maps (not raw logits).

        Always drop the last no-object logit (``softmax(...)[..., :-1]``), same as
        ``nnUNetTrainerV2_DDP.run_online_evaluation``. Do **not** compare against
        ``pred_masks.shape[1]`` (that is ``num_queries``, not ``num_classes``).
        """
        if not isinstance(output, dict):
            if isinstance(output, (list, tuple)):
                return output[0]
            return output
        mask_cls = output["pred_logits"]
        mask_pred = output["pred_masks"]
        cls = mask_cls.softmax(-1)[..., :-1]
        return torch.einsum("bqc,bqdhw->bcdhw", cls, mask_pred.sigmoid())

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            if self._is_decoder_variant:
                l = self._hungarian_loss(output, target)
            else:
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            if self._is_decoder_variant:
                l = self._hungarian_loss(output, target)
                # Already soft maps ∈ ~[0,1]; do NOT sigmoid again (official online eval).
                output_for_metrics = self._dict_output_to_soft_maps(output)
            else:
                l = self.loss(output, target)
                output_for_metrics = output[0] if isinstance(output, (list, tuple)) else output

        axes = [0] + list(range(2, output_for_metrics.ndim))
        if self.label_manager.has_regions:
            predicted = (output_for_metrics > 0.5).long()
        else:
            predicted = output_for_metrics.argmax(1)[:, None]
            predicted = torch.zeros_like(output_for_metrics).scatter_(1, predicted, 1)

        if isinstance(target, list):
            target = target[0]
        if self.label_manager.has_regions:
            if target.ndim == 4:
                raise RuntimeError(f"region target expected 5D, got {tuple(target.shape)}")
            target = target.float()
        else:
            if target.ndim == 4:
                target = target[:, None]
            target_oh = torch.zeros_like(output_for_metrics)
            target_oh.scatter_(1, target.long(), 1)
            target = target_oh

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted, target, axes=axes)
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


class nnUNetTrainerTransUNet3D(nnUNetTrainerTransUNet3DBase):
    """Default = official BraTS ``decoder_only.yaml`` (recommended for tumors)."""

    TRANSUNET_VARIANT = "decoder_only"


class nnUNetTrainerTransUNet3D_DecoderOnly(nnUNetTrainerTransUNet3D):
    """Explicit alias of default BraTS decoder_only."""


class nnUNetTrainerTransUNet3D_EncoderOnly(nnUNetTrainerTransUNet3DBase):
    """Official BraTS ``encoder_only.yaml``."""

    TRANSUNET_VARIANT = "encoder_only"


class nnUNetTrainerTransUNet3D_EncoderPlusDecoder(nnUNetTrainerTransUNet3DBase):
    """Official BraTS /Synapse / ACDC ``encoder_plus_decoder.yaml``."""

    TRANSUNET_VARIANT = "encoder_plus_decoder"


# Compat names: map old R50-* trainers → official encoder_only (bottleneck ViT).
class nnUNetTrainerTransUNet3DR50ViTB16(nnUNetTrainerTransUNet3D_EncoderOnly):
    """Compat → official encoder_only (old hybrid name kept for CLI)."""


class nnUNetTrainerTransUNet3DR50ViTL16(nnUNetTrainerTransUNet3D_EncoderOnly):
    """Compat → official encoder_only."""
