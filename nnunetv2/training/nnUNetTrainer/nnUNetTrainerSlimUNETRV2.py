from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.network.SlimUNETRV2 import create_slimunetrv2_from_config
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


class _WarmupCosineLRScheduler:
    """Linear warmup then cosine decay; step(current_epoch) like PolyLRScheduler."""

    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        warmup_steps: int = 50,
        eta_min: float = 0.0,
    ):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.warmup_steps = max(1, int(warmup_steps))
        self.eta_min = float(eta_min)
        self.ctr = 0

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.initial_lr * (current_step + 1) / self.warmup_steps
        else:
            decay_steps = max(1, self.max_steps - self.warmup_steps)
            progress = min(max((current_step - self.warmup_steps) / decay_steps, 0.0), 1.0)
            # Standard cosine: eta_min + 0.5*(lr-eta_min)*(1+cos(pi*progress))
            new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
                1.0 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(new_lr)


class nnUNetTrainerSlimUNETRV2Base(nnUNetTrainerNoDeepSupervision):
    """
    SlimUNETRV2 trainer base — aligned with official deepang-ai SlimUNETRV2_*/nnUNetTrainer
    defaults (SGD lr=1e-2, wd=3e-5, momentum=0.99 nesterov, PolyLR exp=0.9, 1000 epochs).

    Intentional local deviation: train in fp32 (grad_scaler=None). Official nnU-Net AMP
    often yields NaN with mamba selective scan; same remedy as nnUNetTrainerSegMamba3D.

    Requires the vendored mamba_ssm fork (bimamba_type='v3', nslices).

    Subclasses override MODEL_CONFIG_NAME / TARGET_PATCH_SIZE and optionally optimizer fields.
    """

    MODEL_CONFIG_NAME = "slim_96"
    TARGET_PATCH_SIZE = (96, 96, 96)
    # "poly" | "warmup_cosine"
    LR_SCHEDULE = "poly"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        # Disable AMP: selective scan / Mamba in fp16 often yields NaN from step 0.
        self.grad_scaler = None

        # Official SlimUNETRV2_*Trainer inherits nnUNetTrainer defaults:
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.optimizer_name = "SGD"
        self.momentum = 0.99
        self.lr_exponent = 0.9
        self.warmup_epochs = 50
        self.lr_eta_min = 0.0

        if hasattr(self.configuration_manager, "configuration"):
            self.configuration_manager.configuration["patch_size"] = tuple(
                self.TARGET_PATCH_SIZE
            )

        self.print_to_log_file(
            "SlimUNETRV2: fp32 training (no autocast / GradScaler) for Mamba stability; "
            f"config={self.MODEL_CONFIG_NAME}, patch={self.TARGET_PATCH_SIZE}."
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False
        if (
            hasattr(self, "network")
            and self.network is not None
            and hasattr(self.network, "deep_supervision")
        ):
            self.network.deep_supervision = False

    @classmethod
    def build_network_architecture(
        cls,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return create_slimunetrv2_from_config(
            cls.MODEL_CONFIG_NAME,
            in_channels=num_input_channels,
            num_classes=num_output_channels,
        )

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                eps=1e-4,
            )

        max_steps = self.num_epochs
        if self.LR_SCHEDULE == "warmup_cosine":
            lr_scheduler = _WarmupCosineLRScheduler(
                optimizer,
                initial_lr=self.initial_lr,
                max_steps=max_steps,
                warmup_steps=self.warmup_epochs,
                eta_min=self.lr_eta_min,
            )
            sched_msg = (
                f"warmup_cosine warmup={self.warmup_epochs}, eta_min={self.lr_eta_min}"
            )
        else:
            lr_scheduler = PolyLRScheduler(
                optimizer,
                initial_lr=self.initial_lr,
                max_steps=max_steps,
                exponent=self.lr_exponent,
            )
            sched_msg = f"poly max_steps={max_steps} (per-epoch), poly_exp={self.lr_exponent}"

        self.print_to_log_file(
            f"SlimUNETRV2 optimizer={self.optimizer_name}, lr={self.initial_lr}, "
            f"wd={self.weight_decay}, {sched_msg}"
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"lr: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

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

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

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
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

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


# ---------------------------------------------------------------------------
# Official-aligned patch presets (≈ SlimUNETRV2_96/128Trainer + nnUNet defaults)
# ---------------------------------------------------------------------------


class nnUNetTrainerSlimUNETRV2_96(nnUNetTrainerSlimUNETRV2Base):
    """Official 96³ path: kernel_sizes=[2,2,2,2], SGD/1e-2/1000ep/Poly."""

    MODEL_CONFIG_NAME = "slim_96"
    TARGET_PATCH_SIZE = (96, 96, 96)


class nnUNetTrainerSlimUNETRV2_128(nnUNetTrainerSlimUNETRV2Base):
    """Official 128³ path: kernel_sizes=[4,2,2,2], SGD/1e-2/1000ep/Poly."""

    MODEL_CONFIG_NAME = "slim_128"
    TARGET_PATCH_SIZE = (128, 128, 128)


class nnUNetTrainerSlimUNETRV2(nnUNetTrainerSlimUNETRV2_96):
    """Compatibility alias → official-aligned 96³."""


# ---------------------------------------------------------------------------
# Dataset-recommended trainers (paper IV.B mapped onto nnU-Net + DC+CE)
# ---------------------------------------------------------------------------


class nnUNetTrainerSlimUNETRV2_BraTS(nnUNetTrainerSlimUNETRV2_128):
    """
    Recommended for BraTS 2021 / BraTS 2023 (3d_fullres).

    Paper IV.B BraTS23: SGD, poly from 0.01, 1000 epochs; patch 128³ (BraTS21).
    Matches official SlimUNETRV2_128Trainer + nnU-Net SGD defaults.
    Loss remains nnU-Net DC+CE (paper III.D dice+focal / IV.B CE are inconsistent).
    """


class nnUNetTrainerSlimUNETRV2_MMWHS(nnUNetTrainerSlimUNETRV2_96):
    """
    Recommended for MM-WHS CT/MRI (3d_fullres).

    Paper IV.B: AdamW lr=0.001, weight_decay=0.4, linear warmup + cosine, patch 96³.
    Warmup length defaults to 50 epochs (paper does not specify); override self.warmup_epochs if needed.
    """

    LR_SCHEDULE = "warmup_cosine"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.optimizer_name = "AdamW"
        self.initial_lr = 1e-3
        self.weight_decay = 0.4
        self.num_epochs = 1000
        self.warmup_epochs = 50
        self.lr_eta_min = 0.0
        self.print_to_log_file(
            "SlimUNETRV2 MM-WHS recipe: AdamW lr=1e-3, wd=0.4, warmup+cosine "
            f"(warmup_epochs={self.warmup_epochs})."
        )


# ---------------------------------------------------------------------------
# Previous local recipe (AdamW / 200ep) — keep for short smoke runs
# ---------------------------------------------------------------------------


class nnUNetTrainerSlimUNETRV2_LegacyAdamW(nnUNetTrainerSlimUNETRV2Base):
    """Pre-alignment local defaults: AdamW 1e-3, wd=1e-5, 200 epochs, Poly. Default patch 96³."""

    MODEL_CONFIG_NAME = "slim_96"
    TARGET_PATCH_SIZE = (96, 96, 96)

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.optimizer_name = "AdamW"
        self.initial_lr = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 200
        self.print_to_log_file(
            "SlimUNETRV2 LegacyAdamW: AdamW lr=1e-3, wd=1e-5, 200ep (not official)."
        )


class nnUNetTrainerSlimUNETRV2_LegacyAdamW_96(nnUNetTrainerSlimUNETRV2_LegacyAdamW):
    MODEL_CONFIG_NAME = "slim_96"
    TARGET_PATCH_SIZE = (96, 96, 96)


class nnUNetTrainerSlimUNETRV2_LegacyAdamW_128(nnUNetTrainerSlimUNETRV2_LegacyAdamW):
    MODEL_CONFIG_NAME = "slim_128"
    TARGET_PATCH_SIZE = (128, 128, 128)
