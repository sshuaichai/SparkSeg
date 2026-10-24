"""SparkSeg configuration: the clean operating point and the per-dataset presets.

Kept separate from the model (:mod:`SparkSeg.network.model`) so the model file stays
about the model.  Import either way::

    from SparkSeg import CLEAN_CONFIG, DATASET_PRESETS, preset_kwargs
    from SparkSeg.network.config import DATASET_PRESETS

``CLEAN_CONFIG`` is the single operating point of the manuscript (no ablation
switches). The dataset trainers consume this configuration directly.

``DATASET_PRESETS`` = the dataset-specific settings reported in Methods §2.5.
``DatasetPreset.init`` records the
initialization arm that produced the reported number: ``stock_he`` = He(1e-2)
post-init plus zero-last-BN before each residual add (product default,
``nnUNetTrainer*_StockHe``), ``no_stock_he`` = that step skipped
(``nnUNetTrainer*_NoStockHe``, the paper-period runs)::

    modality  dataset   id    host       patch / batch    epochs  init         mean DSC
    MRI       ACDC      100   resenc     224×256×10 / 7   200     no_stock_he  91.98
    MRI       BraTS21   1251  plainconv  128×128×128 / 2  1000    no_stock_he  91.73
    CT        Synapse   180   resenc     56×192×224 / 2   1000    stock_he     85.94
    CT        KiTS23    489   resenc     128×128×128 / 2  1000    stock_he     85.66

Those runs pre-date the current K pyramid (they used a tighter 64/32/32/32 budget
and an older code snapshot), so the column is provenance of the endpoint rather
than a controlled single-variable comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

ENCODER_P = "plainconv"
ENCODER_R = "resenc"

#: Single operating point of the paper; consumed by every SparkSeg trainer.
CLEAN_CONFIG: Dict[str, Any] = {
    "base_num_features": 32, "num_pool": 5, "max_num_features": 320,
    "conv_per_stage": 2, "conv_per_stage_decoder": 2, "enable_deep_supervision": True,
    "guide_stages": (1, 2, 3, 4), "local_only_stages": (0,),
    "axial_kernel": 5, "energy_fallback_lambda": 0.25,
    "window_size": 4, "window_heads": 4, "window_use_shift": True,
    "anchor_deep_k_max": 16, "anchor_heads": 4, "anchor_drop": 0.0,
    "gate_bias_init": -2.5, "topk_jitter": 1e-2,
    "encoder_type": ENCODER_P, "res_n_blocks": None,
}

#: Behaviours hard-wired in this clean build (mirrored into ``model_config``).
PRODUCT_FIXED: Dict[str, Any] = {
    "enable_window_mhsa": True, "ba_residual_on_base": True,
    "enable_decoder_attn_gate": True, "enable_prior": True,
    "enable_bottleneck_anchor": False, "enable_read_on_guide": True,
    "enable_read_on_local_only": False,
}

#: Matched 3d fullres recipe (Methods): SGD, Nesterov 0.99, polynomial decay,
#: 250 train / 50 val iterations per epoch, λ_prior 0.05 / λ_prioraux 0.02 (Eq. 8).
TRAIN_DEFAULTS: Dict[str, Any] = {
    "num_iterations_per_epoch": 250, "num_val_iterations_per_epoch": 50,
    "initial_lr": 1e-2, "nesterov_momentum": 0.99, "weight_decay": 3e-5,
    "poly_decay_exponent": 0.9, "oversample_foreground_percent": 0.33,
    "max_grad_norm": 12.0, "prior_loss_weight": 0.05, "prior_aux_loss_weight": 0.02,
}

RESENC_BLOCKS_S6: Tuple[int, ...] = (1, 3, 4, 6, 6, 6)
ANCHOR_K_SCHEDULE: Tuple[int, int, int, int] = (128, 64, 32, 32)


@dataclass(frozen=True)
class DatasetPreset:
    """Reported configuration of one benchmark (manuscript Methods §2.5)."""

    name: str
    modality: str
    dataset_id: int
    raw_name: str
    fold: str
    trainer: str
    host: str
    patch_size: Tuple[int, int, int]
    batch_size: int
    epochs: int
    init: str  # "stock_he" | "no_stock_he": arm behind the reported number
    reported: Dict[str, float] = field(default_factory=dict)
    anchor_k_schedule: Tuple[int, int, int, int] = ANCHOR_K_SCHEDULE


# modality, name, id, raw folder, fold, trainer, host, patch, batch, epochs,
# init arm, (dsc, hd95, train GB, params M, flops G, latency s), aliases.
_ROWS = (
    ("MRI", "acdc", 100, "Dataset100_ACDC", "0", "nnUNetTrainerSparkSegACDC",
     ENCODER_R, (224, 256, 10), 7, 200, "no_stock_he",
     (91.98, 1.08, 15.0, 111.1, 440.4, 0.087),
     ("100", "dataset100", "dataset100_acdc")),
    ("MRI", "brats", 1251, "Dataset1251_BraTS2023GLI", "0",
     "nnUNetTrainerSparkSegBraTS2021", ENCODER_P, (128, 128, 128), 2, 1000,
     "no_stock_he",
     (91.73, 2.57, 11.3, 34.8, 575.1, 0.356),
     ("brats2021", "brats2023", "1251", "dataset1251", "dataset1251_brats2023gli")),
    ("CT", "synapse", 180, "Dataset180_Synapse", "all",
     "nnUNetTrainerSparkSegSynapse", ENCODER_R, (56, 192, 224), 2, 1000,
     "stock_he",
     (85.94, 11.21, 13.7, 111.1, 1089.4, 7.523),
     ("btcv", "180", "dataset180", "dataset180_synapse")),
    ("CT", "kits23", 489, "Dataset489_KiTS23", "0",
     "nnUNetTrainerSparkSegKiTS23", ENCODER_R, (128, 128, 128), 2, 1000,
     "stock_he",
     (85.66, 13.41, 17.2, 111.1, 920.0, 9.074),
     ("kits", "Dataset489_KiTS23", "489", "dataset489", "dataset489_kits23")),
)
_REPORTED = ("mean_dsc", "mean_hd95", "peak_train_gb", "params_m", "flops_g", "latency_s")

DATASET_PRESETS: Dict[str, DatasetPreset] = {
    r[1]: DatasetPreset(name=r[1], modality=r[0], dataset_id=r[2], raw_name=r[3],
                        fold=r[4], trainer=r[5], host=r[6],
                        patch_size=r[7], batch_size=r[8], epochs=r[9], init=r[10],
                        reported=dict(zip(_REPORTED, r[11])))
    for r in _ROWS
}

#: Ways a caller may name a benchmark (key, dataset id, nnU-Net folder name).
PRESET_ALIASES: Dict[str, str] = {a: r[1] for r in _ROWS for a in (r[1], *r[12])}


def resolve_preset(name: Union[str, int, None]) -> Optional[DatasetPreset]:
    """Resolve a preset from a key, nnU-Net id or folder name (``None`` → ``None``)."""
    if name is None:
        return None
    key = str(name).strip().lower().replace("-", "_")
    key = PRESET_ALIASES.get(key, key)
    if key not in DATASET_PRESETS:
        raise KeyError(
            f"unknown dataset preset {name!r}; known: {sorted(DATASET_PRESETS)}, "
            f"aliases: {sorted(PRESET_ALIASES)}"
        )
    return DATASET_PRESETS[key]


def clean_kwargs(**over: Any) -> Dict[str, Any]:
    """Network kwargs of the clean (paper-default) configuration."""
    cfg = {**CLEAN_CONFIG, **PRODUCT_FIXED, "stock_he": True}
    cfg.update(over)
    return cfg


def preset_kwargs(preset: Union[str, int, DatasetPreset], **over: Any) -> Dict[str, Any]:
    """Clean-config kwargs specialised to one benchmark (host + init arm + plan)."""
    p = preset if isinstance(preset, DatasetPreset) else resolve_preset(preset)
    assert p is not None
    cfg = clean_kwargs(encoder_type=p.host, stock_he=(p.init == "stock_he"))
    if p.host == ENCODER_R:
        cfg["res_n_blocks"] = RESENC_BLOCKS_S6
    cfg.update(anchor_k_schedule=tuple(p.anchor_k_schedule), dataset_id=int(p.dataset_id),
               patch_size=tuple(p.patch_size), batch_size=int(p.batch_size),
               epochs=int(p.epochs))
    cfg.update(over)
    return cfg


def describe_preset(dataset: Union[str, int, DatasetPreset]) -> str:
    """One-line summary of a preset, including the init arm behind its number."""
    p = resolve_preset(dataset)
    assert p is not None
    rep = ", ".join(f"{k}={v}" for k, v in p.reported.items())
    return (
        f"{p.name} (Dataset{p.dataset_id:03d}, {p.modality}) host={p.host} "
        f"fold={p.fold} patch={tuple(p.patch_size)} batch={p.batch_size} epochs={p.epochs} "
        f"init={p.init} K={tuple(p.anchor_k_schedule)} | reported: {rep}"
    )


def verify_clean_config(strict: bool = False) -> Dict[str, Any]:
    """Validate the SparkSeg budget schedule against the clean configuration."""
    from .utils import num_tokens_for_stage

    schedule = tuple(
        num_tokens_for_stage(
            stage,
            deep_stage=max(CLEAN_CONFIG["guide_stages"]),
            deep_k=int(CLEAN_CONFIG["anchor_deep_k_max"]),
        )
        for stage in CLEAN_CONFIG["guide_stages"]
    )
    mismatch = {}
    if schedule != ANCHOR_K_SCHEDULE:
        mismatch["anchor_k_schedule"] = (ANCHOR_K_SCHEDULE, schedule)
    report: Dict[str, Any] = {
        "checked": True,
        "mismatch": mismatch,
        "reason": "ok" if not mismatch else f"{len(mismatch)} key(s) differ",
    }
    if mismatch and strict:
        raise RuntimeError(f"invalid SparkSeg configuration: {mismatch}")
    return report


__all__ = [
    "ENCODER_P", "ENCODER_R",
    "CLEAN_CONFIG", "PRODUCT_FIXED", "TRAIN_DEFAULTS",
    "RESENC_BLOCKS_S6", "ANCHOR_K_SCHEDULE",
    "DatasetPreset", "DATASET_PRESETS", "PRESET_ALIASES",
    "resolve_preset", "clean_kwargs", "preset_kwargs",
    "describe_preset", "verify_clean_config",
]
