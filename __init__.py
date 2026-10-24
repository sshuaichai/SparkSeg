"""SparkSeg — sparse prior-guided anchor attention for 3D CT and MRI segmentation.

Public API::

    from SparkSeg import build_sparkseg, DATASET_PRESETS, describe_preset

    net = build_sparkseg(in_channels, num_classes, plan_strides=strides,
                         preset="acdc")     # or "synapse" / "brats" / "kits23"

Network code lives under :mod:`SparkSeg.network`. Dataset-specific nnU-Net trainers
live in :mod:`SparkSeg.nnUNetTrainerSparkSeg`.
"""

from __future__ import annotations

from .network import (  # noqa: F401
    ANCHOR_K_SCHEDULE,
    CLEAN_CONFIG,
    DATASET_PRESETS,
    ENCODER_P,
    ENCODER_R,
    PRESET_ALIASES,
    PRODUCT_FIXED,
    RESENC_BLOCKS_S6,
    TRAIN_DEFAULTS,
    SparkSeg,
    AttentionGate,
    DatasetPreset,
    MSBA,
    PGTS,
    PriorHead,
    SparkSegBlock,
    SparkSegDecoder,
    TAE,
    WinMHSA3D,
    build_sparkseg,
    build_sparkseg_for,
    clean_kwargs,
    delta_energy_per_sample,
    describe_preset,
    num_tokens_for_stage,
    preset_kwargs,
    resolve_preset,
    topk_coords_from_map,
    verify_clean_config,
)

__version__ = "1.3.0"

__all__ = [
    # network
    "SparkSeg", "SparkSegBlock", "SparkSegDecoder", "TAE", "PriorHead",
    "PGTS", "MSBA", "WinMHSA3D", "AttentionGate",
    "build_sparkseg", "build_sparkseg_for",
    "delta_energy_per_sample", "topk_coords_from_map", "num_tokens_for_stage",
    # configuration
    "CLEAN_CONFIG", "PRODUCT_FIXED", "TRAIN_DEFAULTS",
    "RESENC_BLOCKS_S6", "ANCHOR_K_SCHEDULE",
    "DatasetPreset", "DATASET_PRESETS", "PRESET_ALIASES",
    "resolve_preset", "clean_kwargs", "preset_kwargs",
    "describe_preset", "verify_clean_config",
    "ENCODER_P", "ENCODER_R",
    "__version__",
]
