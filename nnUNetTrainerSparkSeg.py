"""Dataset-specific nnU-Net trainers for SparkSeg.

Available trainers
------------------
MRI:
``nnUNetTrainerSparkSegACDC``
    Dataset100, residual encoder, 200 epochs, no ``stock_he`` initialisation.
``nnUNetTrainerSparkSegBraTS2021``
    Dataset1251, plain U-Net encoder, 1000 epochs, no ``stock_he`` initialisation.

CT:
``nnUNetTrainerSparkSegSynapse``
    Dataset180, residual encoder, 1000 epochs, ``stock_he`` initialisation.
``nnUNetTrainerSparkSegKiTS23``
    Dataset489, residual encoder, 1000 epochs, ``stock_he`` initialisation.

Patch size and batch size come from each dataset's nnU-Net plans.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Type

import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .network import (
    DatasetPreset,
    build_sparkseg,
    resolve_preset,
)
from nnunetv2.utilities.helpers import dummy_context


def _unwrap_net(module: torch.nn.Module) -> torch.nn.Module:
    module = getattr(module, "module", module)
    return getattr(module, "_orig_mod", module)


def _prior_loss(
    prior_probs: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    has_regions: bool,
    smooth: float = 1e-5,
) -> torch.Tensor:
    prior = prior_probs.float()
    tgt = target.float()
    if tgt.shape[2:] != prior.shape[2:]:
        tgt = F.interpolate(tgt, size=prior.shape[2:], mode="nearest")
    if has_regions or (
        tgt.ndim == 5 and tgt.shape[1] == int(num_classes) and tgt.shape[1] > 1
    ):
        one_hot = tgt.clamp(0.0, 1.0)
    else:
        if tgt.ndim == 5 and tgt.shape[1] == 1:
            labels = tgt[:, 0].long()
        elif tgt.ndim == 4:
            labels = tgt.long()
        else:
            labels = tgt[:, 0].long()
        one_hot = F.one_hot(labels, int(num_classes)).permute(0, 4, 1, 2, 3)
        one_hot = one_hot.contiguous().float()

    prior = prior.clamp(1e-7, 1.0 - 1e-7)
    bce = F.binary_cross_entropy(prior, one_hot, reduction="mean")
    dims = (0, 2, 3, 4)
    intersection = (prior * one_hot).sum(dim=dims)
    denominator = (prior + one_hot).sum(dim=dims)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    if int(num_classes) > 1 and not has_regions:
        dice = 1.0 - dice[1:].mean()
    else:
        dice = 1.0 - dice.mean()
    return bce + dice


def _foreground_prior_bce(
    prior: torch.Tensor,
    target: torch.Tensor,
    *,
    has_regions: bool,
) -> torch.Tensor:
    probability = prior.float()
    if probability.shape[1] > 1:
        foreground = probability[:, 1:].sum(dim=1, keepdim=True)
        foreground = foreground.clamp(1e-7, 1.0 - 1e-7)
    else:
        foreground = probability.clamp(1e-7, 1.0 - 1e-7)

    tgt = target.float()
    if tgt.ndim == 5 and tgt.shape[1] > 1:
        if has_regions:
            gt_foreground = tgt[:, :-1].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        else:
            gt_foreground = (tgt[:, 1:].sum(dim=1, keepdim=True) > 0.5).float()
    else:
        labels = tgt[:, 0] if tgt.ndim == 5 else tgt
        gt_foreground = (labels > 0).float().unsqueeze(1)
    if gt_foreground.shape[2:] != foreground.shape[2:]:
        gt_foreground = F.interpolate(
            gt_foreground, size=foreground.shape[2:], mode="nearest"
        )
    return F.binary_cross_entropy(foreground, gt_foreground, reduction="mean")


class _SparkSegDatasetTrainer(nnUNetTrainer):
    """Train SparkSeg with the loss and validation behavior used by the manuscript."""

    PRESET: str = ""
    NUM_EPOCHS: int = 1000
    NUM_ITERATIONS_PER_EPOCH: int = 250
    NUM_VAL_ITERATIONS_PER_EPOCH: int = 50
    INITIAL_LR: float = 1e-2
    WEIGHT_DECAY: float = 3e-5
    OVERSAMPLE_FOREGROUND_PERCENT: float = 0.33
    PRIOR_LOSS_WEIGHT: float = 0.05
    PRIOR_AUX_LOSS_WEIGHT: float = 0.02
    MAX_GRAD_NORM: float = 12.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = int(self.NUM_EPOCHS)
        self.num_iterations_per_epoch = int(self.NUM_ITERATIONS_PER_EPOCH)
        self.num_val_iterations_per_epoch = int(self.NUM_VAL_ITERATIONS_PER_EPOCH)
        self.initial_lr = float(self.INITIAL_LR)
        self.weight_decay = float(self.WEIGHT_DECAY)
        self.oversample_foreground_percent = float(self.OVERSAMPLE_FOREGROUND_PERCENT)
        self.prior_loss_weight = float(self.PRIOR_LOSS_WEIGHT)
        self.prior_aux_loss_weight = float(self.PRIOR_AUX_LOSS_WEIGHT)
        self.max_grad_norm = float(self.MAX_GRAD_NORM)

    @classmethod
    def preset(cls) -> DatasetPreset:
        resolved = resolve_preset(cls.PRESET)
        assert resolved is not None
        return resolved

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        _ = plans_manager
        arch = configuration_manager.network_arch_init_kwargs
        strides = arch.get("strides") if isinstance(arch, dict) else None
        return build_sparkseg(
            int(num_input_channels),
            int(num_output_channels),
            plan_strides=strides,
            preset=cls.PRESET,
            enable_deep_supervision=bool(enable_deep_supervision),
        )

    def initialize(self) -> None:
        super().initialize()
        network = _unwrap_net(self.network)
        self.sparkseg_config = dict(getattr(network, "model_config", {}))
        preset = self.preset()
        self.print_to_log_file(
            f"SparkSeg trainer={self.__class__.__name__} modality={preset.modality} "
            f"dataset=Dataset{preset.dataset_id:03d} host={preset.host} init={preset.init} "
            f"epochs={self.num_epochs} "
            f"K={tuple(self.sparkseg_config.get('anchor_k_schedule', preset.anchor_k_schedule))}"
        )

    @staticmethod
    def _unpack(output):
        if isinstance(output, tuple) and len(output) == 2:
            return output
        return output, None

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        start = time.time()
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            segmentation, prior = self._unpack(self.network(data))
            segmentation_loss = self.loss(segmentation, target)
        total_loss = segmentation_loss
        deepest_target = target[-1] if isinstance(target, list) else target
        if prior is not None:
            context = autocast(self.device.type, enabled=False) if self.device.type == "cuda" else dummy_context()
            with context:
                prior_loss = _prior_loss(
                    prior,
                    deepest_target,
                    self.label_manager.num_segmentation_heads,
                    has_regions=bool(self.label_manager.has_regions),
                )
            total_loss = total_loss + self.prior_loss_weight * prior_loss
        if self.prior_aux_loss_weight > 0:
            network = _unwrap_net(self.network)
            prior_maps = getattr(network, "_last_stage_p_effs", None) or []
            if prior_maps:
                context = autocast(self.device.type, enabled=False) if self.device.type == "cuda" else dummy_context()
                with context:
                    auxiliary_loss = sum(
                        _foreground_prior_bce(
                            prior_map,
                            deepest_target,
                            has_regions=bool(self.label_manager.has_regions),
                        )
                        for prior_map in prior_maps
                    ) / float(len(prior_maps))
                total_loss = (
                    total_loss
                    + self.prior_aux_loss_weight * auxiliary_loss
                )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        output = {
            "loss": total_loss.detach().cpu().numpy(),
            "step_time_s": time.time() - start,
        }
        if self.device.type == "cuda":
            output["vram_peak_gb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            )
        return output

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            segmentation, _ = self._unpack(self.network(data))
            del data
            loss = self.loss(segmentation, target)

        if self.enable_deep_supervision:
            output, evaluation_target = segmentation[0], target[0]
        else:
            output, evaluation_target = segmentation, target
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted = (torch.sigmoid(output) > 0.5).long()
        else:
            output_labels = output.argmax(1)[:, None]
            predicted = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
            predicted.scatter_(1, output_labels, 1)
            del output_labels

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (
                    evaluation_target != self.label_manager.ignore_label
                ).float()
                evaluation_target[
                    evaluation_target == self.label_manager.ignore_label
                ] = 0
            else:
                if evaluation_target.dtype == torch.bool:
                    mask = (~evaluation_target[:, -1:]).float()
                else:
                    mask = 1 - evaluation_target[:, -1:]
                evaluation_target = evaluation_target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted, evaluation_target, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        result = {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
        if self.device.type == "cuda":
            result["vram_peak_gb"] = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            )
        return result


class nnUNetTrainerSparkSegACDC(_SparkSegDatasetTrainer):
    """SparkSeg with the reported ACDC (MRI) 3d_fullres configuration."""

    PRESET = "acdc"
    NUM_EPOCHS = 200


class nnUNetTrainerSparkSegSynapse(_SparkSegDatasetTrainer):
    """SparkSeg with the reported Synapse/BTCV (CT) 3d_fullres configuration."""

    PRESET = "synapse"


class nnUNetTrainerSparkSegBraTS2021(_SparkSegDatasetTrainer):
    """SparkSeg with the reported BraTS2021 (MRI) 3d_fullres configuration."""

    PRESET = "brats"


class nnUNetTrainerSparkSegKiTS23(_SparkSegDatasetTrainer):
    """SparkSeg with the reported KiTS23 (CT) 3d_fullres configuration."""

    PRESET = "kits23"


TRAINER_CLASSES: Tuple[Type[_SparkSegDatasetTrainer], ...] = (
    nnUNetTrainerSparkSegACDC,
    nnUNetTrainerSparkSegSynapse,
    nnUNetTrainerSparkSegBraTS2021,
    nnUNetTrainerSparkSegKiTS23,
)
TRAINER_BY_PRESET: Dict[str, Type[_SparkSegDatasetTrainer]] = {
    cls.PRESET: cls for cls in TRAINER_CLASSES
}

__all__ = [
    "nnUNetTrainerSparkSegACDC",
    "nnUNetTrainerSparkSegSynapse",
    "nnUNetTrainerSparkSegBraTS2021",
    "nnUNetTrainerSparkSegKiTS23",
    "TRAINER_CLASSES",
    "TRAINER_BY_PRESET",
]
