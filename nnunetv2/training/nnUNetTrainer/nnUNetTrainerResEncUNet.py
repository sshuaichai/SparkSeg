"""ResidualEncoderUNet on the plans already selected (default ``nnUNetPlans``).

Does not load ``nnUNetResEncUNetLPlans`` / ``nnUNetResEncUNetMPlans``. Patch size,
batch size, spacing, stage count, features, kernels and strides stay the current
configuration. The block schedule is the official ResEnc one: encoder
``(1, 3, 4, 6, 6, 6, …)`` truncated to ``n_stages``, decoder one conv per stage.

Training length and optimizer are the stock nnU-Net recipe (also what ResEnc
presets use): 1000 epochs, 250 / 50 iterations, SGD + poly LR. ``DEMO_EPOCHS`` /
``DEMO_ITERS`` still override those after ``__init__``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

_RESENC_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
# ResEncUNetPlanner.UNet_blocks_per_stage_encoder. M/L/XL share this list.
_ENCODER_BLOCKS = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)


def resenc_blocks_per_stage(n_stages: int) -> list[int]:
    """Official encoder depth, cut to this plans' stage count. S6 → (1, 3, 4, 6, 6, 6)."""
    n = int(n_stages)
    if n <= len(_ENCODER_BLOCKS):
        return list(_ENCODER_BLOCKS[:n])
    return list(_ENCODER_BLOCKS) + [6] * (n - len(_ENCODER_BLOCKS))


class nnUNetTrainerResEncUNet(nnUNetTrainer):
    """``ResidualEncoderUNet`` on the current plans, with the stock training recipe."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler: LRScheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        kwargs = dict(configuration_manager.network_arch_init_kwargs)
        n_stages = int(kwargs["n_stages"])
        # Plain plans name the conv count n_conv_per_stage. ResidualEncoderUNet
        # takes n_blocks_per_stage instead, and its decoder is one conv per stage.
        kwargs.pop("n_conv_per_stage", None)
        kwargs["n_blocks_per_stage"] = resenc_blocks_per_stage(n_stages)
        kwargs["n_conv_per_stage_decoder"] = [1] * (n_stages - 1)
        return get_network_from_plans(
            _RESENC_CLASS,
            kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
