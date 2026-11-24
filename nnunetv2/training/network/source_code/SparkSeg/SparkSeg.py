"""Canonical import path for the SparkSeg network.

The implementation is split across sibling modules in
``nnunetv2.training.network.source_code.SparkSeg``: ``model.py`` assembles the
host, ``blocks.py`` defines the paper components, and ``config.py`` holds the
dataset presets.

    from nnunetv2.training.network.source_code.SparkSeg import build_sparkseg

    net = build_sparkseg(
        in_channels,
        num_classes,
        plan_strides=strides,
        preset="acdc",
    )
"""

from __future__ import annotations

from . import (
    ANCHOR_K_SCHEDULE,
    CLEAN_CONFIG,
    DATASET_PRESETS,
    ENCODER_P,
    ENCODER_R,
    PRESET_ALIASES,
    PRODUCT_FIXED,
    RESENC_BLOCKS_S6,
    TRAIN_DEFAULTS,
    AttentionGate,
    DatasetPreset,
    MSBA,
    PGTS,
    PriorHead,
    SparkSeg,
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
