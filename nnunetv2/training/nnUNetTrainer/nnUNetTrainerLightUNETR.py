"""
nnU-Net trainers for :class:`nnunetv2.training.network.light_unetr.LightUNETR`
and :class:`nnunetv2.training.network.light_unetr.LightUNETRLarge`
(ported from ``Light-UNETR-main/fullysup`` + ``code_cse``; paper TPAMI 2025).

Dataset-recommended trainers (official / paper defaults mapped to nnU-Net)::

    # LA / Pancreas / BraTS2019 — code_cse train_supervised.py (SGD)
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_LA
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_Pancreas
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_BraTS2019

    # AbdomenCT-1K / HNC / MSD Task01 — fullysup + paper IV.B (AdamW)
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_AbdomenCT1k
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_LargeAbdomen
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_HNC
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_MSDTask01

Generic::

    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR
    nnUNetv2_train ID 3d_fullres 0 -tr nnUNetTrainerLightUNETR_Large

Notes:

  - Loss remains nnU-Net DC+CE (official fullysup uses Focal+Dice sigmoid for multi-label organs).
  - CSE semi-supervised pipeline is not ported; SSL LA/Pancreas/BraTS trainers use the
    supervised ``code_cse/train_supervised.py`` optimizer recipe only.
  - **FP32** (``grad_scaler=None``). Batch override: ``NNUNET_LIGHTUNETR_BATCH_SIZE``.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.network.light_unetr import (
    LIGHTUNETR_ABLATION_TRAINER_KEYS,
    LIGHTUNETR_ABLATION_VARIANTS,
    LIGHTUNETR_ABDOMENCT1K_DEFAULTS,
    LIGHTUNETR_LARGE_ABDOMEN_DEFAULTS,
    LIGHTUNETR_LARGE_DEFAULTS,
    LIGHTUNETR_NET_DEFAULTS,
    build_light_unetr,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """MONAI-style scheduler (from Light-UNETR ``fullysup/src/optimizer.py``)."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(self.warmup_epochs - 1, 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
                )
            )
            for base_lr in self.base_lrs
        ]


def _lightunetr_ablation_presets() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in LIGHTUNETR_ABLATION_TRAINER_KEYS:
        d = dict(LIGHTUNETR_NET_DEFAULTS)
        d["lightunetr_ablation"] = key
        d.update(LIGHTUNETR_ABLATION_VARIANTS[key])
        out[key] = d
    return out


LIGHTUNETR_ABLATION_PRESETS_BASE: Dict[str, Dict[str, Any]] = _lightunetr_ablation_presets()


class nnUNetTrainerLightUNETRBase(nnUNetTrainerNoDeepSupervision):
    ARCHITECTURE_NAME = "Light-UNETR"
    # Predictor prefers (plans_manager, configuration_manager, ...) so patch_size is available.
    BUILD_NETWORK_USES_CONFIGURATION_MANAGER = True
    MODEL_DEFAULTS: Dict[str, Any] = dict(LIGHTUNETR_NET_DEFAULTS)

    NUM_EPOCHS = 1000
    NUM_ITERATIONS_PER_EPOCH = 250
    NUM_VAL_ITERATIONS_PER_EPOCH = 50
    OVERSAMPLE_FOREGROUND_PERCENT = 0.33
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 5e-3
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-7
    MAX_GRAD_NORM = 1.0
    EXPERIMENT_SEED_BASE = 12345
    DETERMINISTIC_TRAINING = True
    # "adamw" (fullysup) | "sgd" (code_cse train_supervised)
    OPTIMIZER_NAME = "adamw"
    SGD_MOMENTUM = 0.9
    # If set, force configuration patch_size (dataset official crop).
    TARGET_PATCH_SIZE: Tuple[int, int, int] | None = None

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.grad_scaler = None
        self.was_initialized = False
        self.num_epochs = int(self.NUM_EPOCHS)
        self.oversample_foreground_percent = float(self.OVERSAMPLE_FOREGROUND_PERCENT)
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = int(self.NUM_ITERATIONS_PER_EPOCH)
        self.num_val_iterations_per_epoch = int(self.NUM_VAL_ITERATIONS_PER_EPOCH)
        self.initial_lr = float(self.INITIAL_LR)
        self.weight_decay = float(self.WEIGHT_DECAY)
        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.warmup_epochs = int(self.WARMUP_EPOCHS)
        self.min_lr = float(self.MIN_LR)
        self.max_grad_norm = float(self.MAX_GRAD_NORM)
        self.optimizer_name = str(self.OPTIMIZER_NAME)
        self.sgd_momentum = float(self.SGD_MOMENTUM)

        if self.TARGET_PATCH_SIZE is not None and hasattr(self.configuration_manager, "configuration"):
            self.configuration_manager.configuration["patch_size"] = tuple(self.TARGET_PATCH_SIZE)

        self.print_to_log_file(
            f"{self.ARCHITECTURE_NAME}: fp32; opt={self.optimizer_name}; epochs={self.num_epochs} "
            f"warmup={self.warmup_epochs} lr={self.initial_lr} wd={self.weight_decay}"
            + (f" patch={self.TARGET_PATCH_SIZE}" if self.TARGET_PATCH_SIZE is not None else "")
        )

    def _do_i_compile(self) -> bool:
        return False

    @staticmethod
    def _register_ddp_grad_stride_hooks(module: nn.Module) -> None:
        def _force_contiguous_layout(grad: torch.Tensor) -> torch.Tensor:
            return grad.clone(memory_format=torch.contiguous_format)

        for p in module.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 5 and p.shape[1] == 1 and tuple(p.shape[2:]) == (3, 3, 3):
                p.register_hook(_force_contiguous_layout)

    @classmethod
    def _resolve_spatial_patch_size(cls, arch_init_kwargs: dict | None = None) -> Tuple[int, int, int] | None:
        """Patch (D,H,W) for auto_embedding_dim — must match training plans."""
        ps = None
        # Trainer-forced crop (dataset recipes) wins over plans when set.
        if getattr(cls, "TARGET_PATCH_SIZE", None) is not None:
            ps = cls.TARGET_PATCH_SIZE
        if ps is None and arch_init_kwargs:
            ps = arch_init_kwargs.get("patch_size") or arch_init_kwargs.get("spatial_patch_size")
        if ps is None:
            env_ps = os.environ.get("NNUNET_LIGHTUNETR_PATCH_SIZE", "").strip()
            if env_ps:
                parts = [p for p in env_ps.replace("x", ",").split(",") if p.strip()]
                if len(parts) == 3:
                    ps = parts
        if ps is None:
            return None
        out = tuple(int(x) for x in ps)
        if len(out) != 3:
            raise ValueError(f"spatial patch must be length-3 (D,H,W), got {out}")
        return out  # type: ignore[return-value]

    @classmethod
    def _assemble_light_unetr(
        cls,
        num_input_channels: int,
        num_output_channels: int,
        *,
        spatial_patch: Tuple[int, int, int] | None,
    ) -> nn.Module:
        md: Dict[str, Any] = dict(cls.MODEL_DEFAULTS)
        env_abl = os.environ.get("NNUNET_LIGHTUNETR_ABLATION", "").strip()
        if env_abl:
            md["lightunetr_ablation"] = env_abl
        if md.get("lightunetr_ablation") not in (None, "", "none", "full"):
            key = str(md["lightunetr_ablation"]).strip().lower()
            if key not in LIGHTUNETR_ABLATION_VARIANTS:
                raise ValueError(
                    f"unknown lightunetr_ablation {key!r}; valid: {sorted(LIGHTUNETR_ABLATION_VARIANTS.keys())}"
                )
        if spatial_patch is not None:
            md["spatial_patch_size"] = tuple(int(x) for x in spatial_patch)
        elif md.get("auto_embedding_dim", True) and not md.get("spatial_patch_size"):
            raise ValueError(
                "LightUNETR (base) with auto_embedding_dim=True needs spatial patch (D,H,W). "
                "Predictor should inject plans patch_size into arch_init_kwargs; or set "
                "TARGET_PATCH_SIZE on the trainer; or export NNUNET_LIGHTUNETR_PATCH_SIZE=D,H,W "
                "(must match training plans 3d_fullres patch_size)."
            )
        return build_light_unetr(int(num_input_channels), int(num_output_channels), model_defaults=md)

    @classmethod
    def build_network_architecture(cls, *args, **kwargs) -> nn.Module:
        """Build LightUNETR for train/predict (new + legacy predictor signatures).

        New:  ``(plans_manager, configuration_manager, in_ch, out_ch, enable_ds=False)``
        Old:  ``(arch_name, arch_kwargs, req_import, in_ch, out_ch, enable_ds=False)``
        """
        enable_ds = kwargs.pop("enable_deep_supervision", True)
        _ = enable_ds, kwargs

        # New API: first arg is PlansManager (not a class-name string)
        if args and not isinstance(args[0], str):
            configuration_manager = args[1]
            num_input_channels = int(args[2])
            num_output_channels = int(args[3])
            if len(args) > 4:
                _ = args[4]
            ps = cls._resolve_spatial_patch_size(
                {"patch_size": list(configuration_manager.patch_size)}
            )
            return cls._assemble_light_unetr(
                num_input_channels, num_output_channels, spatial_patch=ps
            )

        # Legacy API
        if len(args) < 5:
            raise TypeError(
                "build_network_architecture expected "
                "(plans_manager, configuration_manager, in_ch, out_ch) or "
                "(arch_name, arch_kwargs, req_import, in_ch, out_ch)"
            )
        arch_init_kwargs = dict(args[1] or {})
        num_input_channels = int(args[3])
        num_output_channels = int(args[4])
        ps = cls._resolve_spatial_patch_size(arch_init_kwargs)
        return cls._assemble_light_unetr(
            num_input_channels, num_output_channels, spatial_patch=ps
        )

    def _env_batch_size_override(self) -> tuple[str, str]:
        for k in ("NNUNET_LIGHTUNETR_BATCH_SIZE", "NNUNET_SAMA_BATCH_SIZE", "NNUNET_INNOV_GCR_BATCH_SIZE"):
            v = os.environ.get(k, "").strip()
            if v:
                return v, k
        return "", ""

    def _maybe_cap_batch_size_for_memory(self) -> None:
        cfg = self.configuration_manager.configuration
        current_bs = int(cfg.get("batch_size", self.configuration_manager.batch_size))
        env_bs, env_key = self._env_batch_size_override()
        target_bs = current_bs
        reason = ""
        if env_bs:
            target_bs = max(int(env_bs), 1)
            reason = f"from {env_key}"
        else:
            patch = list(self.configuration_manager.patch_size)
            voxels = int(np.prod(patch))
            if voxels >= 1_500_000 and current_bs > 1:
                target_bs = 1
                reason = f"auto-cap very large patch ({patch})"
            elif voxels >= 800_000 and current_bs > 2:
                target_bs = 2
                reason = f"auto-cap large patch ({patch})"

        if target_bs != current_bs:
            cfg["batch_size"] = target_bs
            self.print_to_log_file(f"[Light-UNETR] Adjusted per-GPU batch_size: {current_bs} -> {target_bs} ({reason})")

    def _effective_seed(self) -> int:
        base = int(getattr(self, "EXPERIMENT_SEED_BASE", 12345))
        try:
            fold_idx = int(self.fold)
        except (TypeError, ValueError):
            fold_idx = 0
        if fold_idx < 0:
            fold_idx = 0
        return base + fold_idx

    def _setup_experiment_randomness(self) -> None:
        seed = self._effective_seed()
        deterministic = bool(getattr(self, "DETERMINISTIC_TRAINING", True))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = False

        self.print_to_log_file(
            f"[Light-UNETR][repro] seed={seed} deterministic={deterministic} "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
        )

    def initialize(self) -> None:
        self._setup_experiment_randomness()
        if self.was_initialized:
            raise RuntimeError("Trainer already initialized")

        self._maybe_cap_batch_size_for_memory()
        self._set_batch_size_and_oversample()
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )

        md: Dict[str, Any] = dict(self.MODEL_DEFAULTS)
        env_abl = os.environ.get("NNUNET_LIGHTUNETR_ABLATION", "").strip()
        if env_abl:
            md["lightunetr_ablation"] = env_abl
            self.print_to_log_file(f"[Light-UNETR] NNUNET_LIGHTUNETR_ABLATION={env_abl!r}")
        if md.get("lightunetr_ablation") not in (None, "", "none", "full"):
            key = str(md["lightunetr_ablation"]).strip().lower()
            if key not in LIGHTUNETR_ABLATION_VARIANTS:
                raise ValueError(
                    f"unknown lightunetr_ablation {key!r}; valid: {sorted(LIGHTUNETR_ABLATION_VARIANTS.keys())}"
                )
            self.print_to_log_file(f"[Light-UNETR] lightunetr_ablation={key!r} -> {LIGHTUNETR_ABLATION_VARIANTS[key]}")

        md["spatial_patch_size"] = tuple(int(x) for x in self.configuration_manager.patch_size)
        self.print_to_log_file(
            f"[Light-UNETR][arch] variant={md.get('lightunetr_variant')} in/out="
            f"{self.num_input_channels}/{self.label_manager.num_segmentation_heads} patch={md['spatial_patch_size']}"
        )

        self.network = build_light_unetr(
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            model_defaults=md,
        ).to(self.device)

        if self._do_i_compile():
            self.network = torch.compile(self.network)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self._register_ddp_grad_stride_hooks(self.network)
            from torch.nn.parallel import DistributedDataParallel as DDP

            raw_find = os.environ.get("NNUNET_LIGHTUNETR_DDP_FIND_UNUSED", "").strip().lower()
            if not raw_find:
                raw_find = os.environ.get("NNUNET_SAMA_DDP_FIND_UNUSED", "").strip().lower()
            if not raw_find:
                raw_find = os.environ.get("NNUNET_INNOV_GCR_DDP_FIND_UNUSED", "").strip().lower()
            if raw_find in {"0", "false", "no", "n", "off"}:
                ddp_find_unused = False
            elif raw_find in {"1", "true", "yes", "y", "on"}:
                ddp_find_unused = True
            else:
                ddp_find_unused = True
            self.print_to_log_file(f"[DDP] find_unused_parameters={ddp_find_unused}")
            self.network = DDP(
                self.network,
                device_ids=[self.local_rank],
                find_unused_parameters=ddp_find_unused,
                gradient_as_bucket_view=False,
            )

        self.loss = self._build_loss()
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "sgd":
            # code_cse/train_supervised.py defaults: SGD momentum=0.9, wd=1e-4, cosine
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=False,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=self.min_lr
            )
            self.print_to_log_file(
                f"Light-UNETR SGD lr={self.initial_lr} mom={self.sgd_momentum} "
                f"wd={self.weight_decay} CosineAnnealingLR T_max={self.num_epochs}"
            )
        else:
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                betas=(self.beta1, self.beta2),
            )
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.num_epochs,
                warmup_start_lr=0.0,
                eta_min=self.min_lr,
            )
            self.print_to_log_file(
                f"Light-UNETR AdamW lr={self.initial_lr} betas=({self.beta1},{self.beta2}) "
                f"wd={self.weight_decay} warmup={self.warmup_epochs} cosine"
            )
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with torch.no_grad():
            output = self.network(data)
            l = self.loss(output, target)
            axes = [0] + list(range(2, output.ndim))
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target != self.label_manager.ignore_label).float()
                    target[target == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target[:, -1:]
                    target = target[:, :-1]
            else:
                mask = None
            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]
            return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        from nnunetv2.utilities.collate_outputs import collate_outputs
        from torch import distributed as dist

        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)
        if self.is_ddp:
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            fps = [None for _ in range(world_size)]
            fns = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            dist.all_gather_object(fps, fp)
            dist.all_gather_object(fns, fn)
            tp = np.vstack([i[None] for i in tps]).sum(0)
            fp = np.vstack([i[None] for i in fps]).sum(0)
            fn = np.vstack([i[None] for i in fns]).sum(0)
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])
        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)


class nnUNetTrainerLightUNETR(nnUNetTrainerLightUNETRBase):
    """Base ``LightUNETR``; AdamW lr=1e-3 / wd=5e-3 / warmup=5 / 1000ep (``config_abdomen_lightunetr.yml``)."""

    ARCHITECTURE_NAME = "Light-UNETR"
    MODEL_DEFAULTS = dict(LIGHTUNETR_NET_DEFAULTS)


class nnUNetTrainerLightUNETR_Large(nnUNetTrainerLightUNETRBase):
    """``LightUNETRLarge``; AdamW lr=1e-4 / 400ep (``config_abdomen_lightunetr_large.yml``)."""

    ARCHITECTURE_NAME = "Light-UNETR-Large"
    MODEL_DEFAULTS = dict(LIGHTUNETR_LARGE_DEFAULTS)
    INITIAL_LR = 1e-4
    NUM_EPOCHS = 1000
    WEIGHT_DECAY = 5e-3


# ---------------------------------------------------------------------------
# Dataset trainers — paper IV + official fullysup / code_cse
# ---------------------------------------------------------------------------


class _LightUNETR_CodeCSE_SGD(nnUNetTrainerLightUNETRBase):
    """
    Shared recipe from ``code_cse/train_supervised.py`` (LA / Pancreas / BraTS fully-sup upper bound):
    SGD lr=0.01, momentum=0.9, wd=1e-4, cosine; ~30k iters → 120 epochs × 250 iters.
    """

    ARCHITECTURE_NAME = "Light-UNETR-code_cse-SGD"
    MODEL_DEFAULTS = dict(LIGHTUNETR_NET_DEFAULTS)
    OPTIMIZER_NAME = "sgd"
    INITIAL_LR = 1e-2
    WEIGHT_DECAY = 1e-4
    SGD_MOMENTUM = 0.9
    WARMUP_EPOCHS = 0
    MIN_LR = 0.0
    NUM_EPOCHS = 120
    NUM_ITERATIONS_PER_EPOCH = 250


class nnUNetTrainerLightUNETR_LA(_LightUNETR_CodeCSE_SGD):
    """LA (Atrial Segmentation): patch 112×112×80, base Light-UNETR, code_cse SGD recipe."""

    ARCHITECTURE_NAME = "Light-UNETR-LA"
    TARGET_PATCH_SIZE = (112, 112, 80)


class nnUNetTrainerLightUNETR_Pancreas(_LightUNETR_CodeCSE_SGD):
    """Pancreas-CT: patch 96³, base Light-UNETR, code_cse SGD recipe."""

    ARCHITECTURE_NAME = "Light-UNETR-Pancreas"
    TARGET_PATCH_SIZE = (96, 96, 96)


class nnUNetTrainerLightUNETR_BraTS2019(_LightUNETR_CodeCSE_SGD):
    """BraTS 2019 (WT / FLAIR-style): patch 96³, base Light-UNETR, code_cse SGD recipe."""

    ARCHITECTURE_NAME = "Light-UNETR-BraTS2019"
    TARGET_PATCH_SIZE = (96, 96, 96)


class nnUNetTrainerLightUNETR_AbdomenCT1k(nnUNetTrainerLightUNETRBase):
    """
    AbdomenCT-1K base: ``config_abdomen_lightunetr.yml`` + ``main_abdomenct1k_lightunetr.py``.
    AdamW lr=1e-3, wd=5e-3, warmup=5, 1000ep, patch 96³, legacy pads + embedding_dim=27.
    """

    ARCHITECTURE_NAME = "Light-UNETR-AbdomenCT1k"
    MODEL_DEFAULTS = dict(LIGHTUNETR_ABDOMENCT1K_DEFAULTS)
    TARGET_PATCH_SIZE = (96, 96, 96)
    INITIAL_LR = 1e-3
    WEIGHT_DECAY = 5e-3
    WARMUP_EPOCHS = 5
    NUM_EPOCHS = 1000


class nnUNetTrainerLightUNETR_LargeAbdomen(nnUNetTrainerLightUNETRBase):
    """
    AbdomenCT-1K Light-UNETR-L: ``config_abdomen_lightunetr_large.yml``.
    AdamW lr=1e-4, wd=5e-3, warmup=5, 400ep, patch 96³, legacy pads.
    (Official also uses batch=1 + grad_accum=4; set ``NNUNET_LIGHTUNETR_BATCH_SIZE=1``.)
    """

    ARCHITECTURE_NAME = "Light-UNETR-Large-Abdomen"
    MODEL_DEFAULTS = dict(LIGHTUNETR_LARGE_ABDOMEN_DEFAULTS)
    TARGET_PATCH_SIZE = (96, 96, 96)
    INITIAL_LR = 1e-4
    NUM_EPOCHS = 400
    WEIGHT_DECAY = 5e-3
    WARMUP_EPOCHS = 5


class nnUNetTrainerLightUNETR_HNC(nnUNetTrainerLightUNETRBase):
    """
    HNC Tumor / HeadNeck: ``main_headneck.py`` runtime overrides (not yml alone).
    Base Light-UNETR, AdamW lr=2e-4, wd=5e-3, warmup=5, 1000ep, patch 96³.
    """

    ARCHITECTURE_NAME = "Light-UNETR-HNC"
    MODEL_DEFAULTS = dict(LIGHTUNETR_NET_DEFAULTS)
    TARGET_PATCH_SIZE = (96, 96, 96)
    INITIAL_LR = 2e-4
    WEIGHT_DECAY = 5e-3
    WARMUP_EPOCHS = 5
    NUM_EPOCHS = 1000


# Aliases for paper naming
nnUNetTrainerLightUNETR_HNCTumor = nnUNetTrainerLightUNETR_HNC
nnUNetTrainerLightUNETR_HeadNeck = nnUNetTrainerLightUNETR_HNC


class nnUNetTrainerLightUNETR_MSDTask01(nnUNetTrainerLightUNETRBase):
    """
    MSD Task01 (brain tumor): paper IV.B fully-supervised Light-UNETR-L —
    AdamW lr=1e-4, batch 1, channels {32,60,96,150,198}; patch 96³ (Table IV FLOPs).
    Uses large abdomen-style pads off by default (isotropic brain crops).
    """

    ARCHITECTURE_NAME = "Light-UNETR-MSDTask01"
    MODEL_DEFAULTS = dict(LIGHTUNETR_LARGE_DEFAULTS)
    TARGET_PATCH_SIZE = (96, 96, 96)
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 5e-3
    WARMUP_EPOCHS = 5
    NUM_EPOCHS = 400


_ABL_CLASS_SUFFIX: Dict[str, str] = {
    "attn_scale": "AblAttnScale",
    "legacy_pads": "AblLegacyPads",
}

_ABL_PURPOSE_DOC: Dict[str, str] = {
    "attn_scale": "LIDRA: enable sequence-length-aware attention scaling (``lidra_attn_scale=True``).",
    "legacy_pads": "Use fixed ``embedding_dim=27`` + AbdomenCT-1k-style encoder/decoder spatial pads.",
}


def _preset_rows() -> Tuple[Tuple[str, Dict[str, Dict[str, Any]], str], ...]:
    large_presets = {
        k: {**LIGHTUNETR_LARGE_DEFAULTS, **LIGHTUNETR_ABLATION_VARIANTS[k], "lightunetr_ablation": k}
        for k in LIGHTUNETR_ABLATION_TRAINER_KEYS
    }
    return (
        ("", LIGHTUNETR_ABLATION_PRESETS_BASE, ""),
        ("Large_", large_presets, "Large-"),
    )


def _register_lightunetr_ablation_trainers() -> None:
    import sys

    module = sys.modules[__name__]
    for name_mid, preset_map, arch_mid in _preset_rows():
        for abl_key in LIGHTUNETR_ABLATION_TRAINER_KEYS:
            if abl_key == "full":
                continue
            suffix = _ABL_CLASS_SUFFIX[abl_key]
            cls_name = f"nnUNetTrainerLightUNETR_{name_mid}{suffix}"
            arch_name = f"{arch_mid}Light-UNETR-Abl-{abl_key}"
            doc = _ABL_PURPOSE_DOC[abl_key] + f"\n\n``lightunetr_ablation='{abl_key}'``."
            cls: Type[nnUNetTrainerLightUNETRBase] = type(
                cls_name,
                (nnUNetTrainerLightUNETRBase,),
                {"__doc__": doc, "ARCHITECTURE_NAME": arch_name, "MODEL_DEFAULTS": dict(preset_map[abl_key])},
            )
            setattr(module, cls_name, cls)


_register_lightunetr_ablation_trainers()
