from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.network.SegMambaV2 import create_segmambav2_from_config
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


def _snap_patch_divisible(patch_size: list, factor: int = 16) -> list:
    out = []
    for d in patch_size:
        d = int(d)
        if d < factor or d % factor != 0:
            d = max(factor, ((d + factor - 1) // factor) * factor)
        out.append(d)
    return out


class nnUNetTrainerSegMambaV2(nnUNetTrainerNoDeepSupervision):
    """
    SegMambaV2 trainer base class.

    Subclasses can override MODEL_CONFIG_NAME to switch model presets.

    默认优化超参与 SegMamba-V2-main/brats23/3_train.py（BraTSTrainer）一致：
      SGD lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True, poly 调度 exponent=0.9, max_epoch=1000。
    nnU-Net 仍使用 plans 的 batch_size / patch（各维对齐到 16 的倍数）；损失为 Dice+CE。
    """

    MODEL_CONFIG_NAME = "default"

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
        # 与 brats23/3_train.py 对齐（light_training BraTSTrainer）
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.optimizer_name = "SGD"  # 与官方一致；可改为 "AdamW" 做对照
        self.momentum = 0.99
        self.lr_exponent = 0.9

        # V2：4× stride-2 → %16==0（Synapse 56→64；否则 UnetrUpBlock cat 崩，8 vs 7）
        original_patch = list(self.configuration_manager.patch_size)
        new_patch = _snap_patch_divisible(original_patch, factor=16)
        self.configuration_manager.configuration["patch_size"] = new_patch
        try:
            self.plans_manager.plans["configurations"][configuration]["patch_size"] = new_patch
        except Exception:
            pass
        self.print_to_log_file(
            f"SegMambaV2 patch_size={new_patch}"
            + (f" (snapped from {original_patch})" if new_patch != original_patch else "")
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False
        if hasattr(self, "network") and self.network is not None and hasattr(self.network, "deep_supervision"):
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
        # Must be classmethod (like parent staticmethod): predict calls
        # TrainerClass.build_network_architecture(...) with no instance.
        model = create_segmambav2_from_config(
            cls.MODEL_CONFIG_NAME,
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
        )
        return model

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

        # on_train_epoch_end 里按「epoch」调用 lr_scheduler.step(self.current_epoch + 1)，
        # 故 poly 的 max_steps 须为 num_epochs（若用 iteration 步进需改 step 与 max_steps 一致）。
        max_steps = self.num_epochs
        lr_scheduler = PolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            max_steps=max_steps,
            exponent=self.lr_exponent,
        )
        self.print_to_log_file(
            f"SegMambaV2 optimizer={self.optimizer_name}, lr={self.initial_lr}, "
            f"wd={self.weight_decay}, poly_max_steps(epochs)={max_steps}, poly_exp={self.lr_exponent}"
        )
        return optimizer, lr_scheduler

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
        # PolyLR is stepped once per epoch in this trainer.
        self.lr_scheduler.step(self.current_epoch + 1)


class nnUNetTrainerSegMambaV2Default(nnUNetTrainerSegMambaV2):
    MODEL_CONFIG_NAME = "default"


class nnUNetTrainerSegMambaV2BraTS(nnUNetTrainerSegMambaV2):
    MODEL_CONFIG_NAME = "brats"


class nnUNetTrainerSegMambaV2Testing(nnUNetTrainerSegMambaV2):
    MODEL_CONFIG_NAME = "testing"
