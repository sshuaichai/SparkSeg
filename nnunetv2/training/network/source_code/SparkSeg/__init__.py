"""SparkSeg network implementation."""

from __future__ import annotations

from .blocks import (
    AttentionGate,
    MSBA,
    PGTS,
    PriorHead,
    SparkSegBlock,
    SparkSegDecoder,
    TAE,
    WinMHSA3D,
)
from .config import (
    ANCHOR_K_SCHEDULE,
    CLEAN_CONFIG,
    DATASET_PRESETS,
    ENCODER_P,
    ENCODER_R,
    PRESET_ALIASES,
    PRODUCT_FIXED,
    RESENC_BLOCKS_S6,
    TRAIN_DEFAULTS,
    DatasetPreset,
    clean_kwargs,
    describe_preset,
    preset_kwargs,
    resolve_preset,
    verify_clean_config,
)
from .model import SparkSeg, build_sparkseg, build_sparkseg_for
from .utils import (
    delta_energy_per_sample,
    num_tokens_for_stage,
    topk_coords_from_map,
)

__all__ = [
    "SparkSeg",
    "SparkSegBlock",
    "SparkSegDecoder",
    "TAE",
    "PriorHead",
    "PGTS",
    "MSBA",
    "WinMHSA3D",
    "AttentionGate",
    "build_sparkseg",
    "build_sparkseg_for",
    "delta_energy_per_sample",
    "topk_coords_from_map",
    "num_tokens_for_stage",
    "CLEAN_CONFIG",
    "PRODUCT_FIXED",
    "TRAIN_DEFAULTS",
    "RESENC_BLOCKS_S6",
    "ANCHOR_K_SCHEDULE",
    "DatasetPreset",
    "DATASET_PRESETS",
    "PRESET_ALIASES",
    "resolve_preset",
    "clean_kwargs",
    "preset_kwargs",
    "describe_preset",
    "verify_clean_config",
    "ENCODER_P",
    "ENCODER_R",
]
