"""Beckschen 3D-TransUNet — thin facade over the official clone.

Source of truth (do not re-implement here)::
  ``nnunetv2/training/network/source_code/transunet3d/upstream/nn_transunet/networks/transunet3d_model.py``
  class ``Generic_TransUNet_max_ppbp``
  configs: ``source_code/transunet3d/upstream/configs/Brats/{decoder_only,encoder_only,encoder_plus_decoder}.yaml``

Paper: arXiv 2310.07781 / Beckschen/3D-TransUNet
Legacy mis-repro: ``TransUNet3D_LegacyHybrid.py`` (2D-TransUNet-style StackedConv+ViT) — not official.

# =============================================================================
# 架构流程图 (Architecture Flow) — Generic_TransUNet_max_ppbp
# =============================================================================
#
# Notation (this repo): P = num_pool = len(pool_op_kernel_sizes); S = P+1 encoder stages.
# BraTS / nnU-Net 3d_fullres typical: P5 → S6 = E0..E5 (base_num_features=32, ×2/stage, cap 320).
# crop_size hint (yaml): 128³. convolutional_upsampling=True (official initialize_network).
#
# Shared CNN U-Net trunk (always on):
#
#   Input (B, C, D, H, W)     e.g. BraTS C=4, crop 128³
#            |
#            v
#   +---------------------------+
#   |  CNN Encoder (context)    |  conv_blocks_context[0..P]
#   |  E0..E{P-1}: StackedConv  |---- skips[0..P-1] -------------------+
#   |  (+ MaxPool / conv-pool)  |                                      |
#   |  E_P (bottleneck CNN)     |                                      |
#   +---------------------------+                                      |
#            |                                                         |
#            |  [A] bottleneck ViT  iff is_max_bottleneck_transformer  |
#            |      Transformer(R50-ViT-B_16 layout) + conv_more 1×1   |
#            |      vit_depth=12 (yaml); tokens @ feat_size = crop/2^P |
#            v                                                         |
#   +---------------------------+                                      |
#   |  CNN Decoder (localize)   |  tu↑ + cat(skip) + conv_blocks_loc   |
#   |  D0..D{P-1}               | <------------------------------------+
#   |  collect ds_feats:        |  ds_feats[0]=bot, [1..]=after each D
#   |    [bot, d0, d1, ..., d*] |
#   +---------------------------+
#            |
#            +--[B] classic seg heads (encoder_only / is_max=False) ----+
#            |     do_ds → multi-scale logits; else full-res only       |
#            |                                                          |
#            +--[C] Max PPB+ Mask2Former decoder (is_max=True) ---------+
#                  max_ms_idxs=[-4,-3,-2] → 3-level pixel feats          |
#                  input_proj → max_hidden_dim=192                      |
#                  MultiScaleMaskedTransformerDecoder3d                 |
#                    num_queries=20, max_dec_layers=3, masked attn      |
#                  → dict{pred_logits (B,Q,C[+1]), pred_masks (B,Q,DHW) |
#                     + aux_outputs×(dec_layers) if is_max_ds}          |
#                  train: HungarianMatcher3D + compute_loss_hungarian   |
#                  (max_loss_cal='v1' in BraTS yaml)
#
# =============================================================================
# 三变体开关 (BraTS yaml → flags) — 同一主干，插拔 [A]/[C]
# =============================================================================
#
#   Variant                 [A] bot ViT   [C] Max decoder   DS / loss
#   ---------------------   -----------   ---------------   -----------------
#   decoder_only            OFF           ON (is_max)       DS off; Hungarian
#   encoder_only            ON            OFF (is_max=F)    DS on; CE+Dice
#   encoder_plus_decoder    ON            ON                DS off; Hungarian
#
# Tumor (BraTS) paper default = decoder_only. Organ / multi-class often encoder_only.
#
# ASCII — decoder_only (BraTS recommended):
#
#   Enc CNN ──► Bot CNN ──► Dec CNN + skips ──► Max PPB+ (queries) ──► masks
#                 (no ViT)
#
# ASCII — encoder_only:
#
#   Enc CNN ──► Bot CNN+ViT ──► Dec CNN + skips ──► 1×1 seg heads (+DS)
#                 (no Max)
#
# ASCII — encoder_plus_decoder:
#
#   Enc CNN ──► Bot CNN+ViT ──► Dec CNN + skips ──► Max PPB+ ──► masks
#
# =============================================================================
# 与易混模型的边界
# =============================================================================
# - 2D TransUNet (MICCAI'21 / arXiv:2102.04306): 2D R50-ViT + CUP; ≠ this 3D paper.
# - TransUNet3D_LegacyHybrid.py: homemade StackedConv+ViT; do not cite as official.
# - This facade only re-exports + BraTS yaml params + build_official_transunet3d().
#
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_OFFICIAL_ROOT = Path(__file__).resolve().parent / "source_code" / "transunet3d" / "upstream"
if str(_OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_ROOT))

from nn_transunet.networks.transunet3d_model import (  # noqa: E402
    Generic_TransUNet_max_ppbp,
    InitWeights_He,
    HungarianMatcher3D,
    compute_loss_hungarian,
    softmax_helper,
)

# ---------------------------------------------------------------------------
# Official BraTS yaml model_params (configs/Brats/*.yaml)
# ---------------------------------------------------------------------------

# Flags map to flowchart [A]/[C] above. BraTS tumor default = decoder_only.
BRATS_DECODER_ONLY_PARAMS: Dict[str, Any] = {
    "is_masked_attn": True,
    "max_dec_layers": 3,
    "is_max_bottleneck_transformer": False,  # [A] off
    "max_msda": "",
    "is_max_ms": True,
    "max_ms_idxs": [-4, -3, -2],
    "max_hidden_dim": 192,
    "mw": 1.0,
    "is_max_ds": True,
    "is_masking": True,
    "is_max_hungarian": True,
    "num_queries": 20,
    "is_max_cls": True,
    "is_mhsa_float32": True,
    # is_max defaults True when model name contains 'max'
}

BRATS_ENCODER_ONLY_PARAMS: Dict[str, Any] = {
    "is_max": False,  # [C] off → classic seg heads
    "is_max_hungarian": False,
    "is_masked_attn": True,
    "max_dec_layers": 3,
    "is_max_bottleneck_transformer": True,  # [A] on
    "vit_depth": 12,
    "max_msda": "",
    "is_max_ms": True,
    "max_ms_idxs": [-4, -3, -2],
    "max_hidden_dim": 192,
    "mw": 1.0,
    "is_max_ds": True,
    "is_masking": True,
    "num_queries": 20,
    "is_max_cls": True,
    "is_mhsa_float32": True,
    # Official yaml sets is_vit_pretrain=True; disable unless weights exist.
    "is_vit_pretrain": False,
    "vit_layer_scale": True,
}

BRATS_ENCODER_PLUS_DECODER_PARAMS: Dict[str, Any] = {
    **BRATS_DECODER_ONLY_PARAMS,  # [C] on
    "is_max_bottleneck_transformer": True,  # [A] on

    "vit_depth": 12,
    "is_vit_pretrain": False,
    "vit_layer_scale": True,
}

VARIANT_PARAMS = {
    "decoder_only": BRATS_DECODER_ONLY_PARAMS,
    "encoder_only": BRATS_ENCODER_ONLY_PARAMS,
    "encoder_plus_decoder": BRATS_ENCODER_PLUS_DECODER_PARAMS,
}

# Training hyper-params from configs/Brats/*.yaml (shared across variants).
# Trainer class attrs on nnUNetTrainerTransUNet3DBase mirror these; keep in sync.
BRATS_TRAIN_DEFAULTS: Dict[str, Any] = {
    "initial_lr": 3e-4,
    "optim_name": "adamw",
    "weight_decay": 1e-4,
    "lrschedule": "warmup_cosine",
    "warmup_epochs": 10,
    "max_num_epochs": 1000,
    "max_loss_cal": "v1",
    "num_iterations_per_epoch": 250,
    "num_val_iterations_per_epoch": 50,
    "crop_size_hint": (128, 128, 128),
}


def get_variant_model_params(variant: str) -> Dict[str, Any]:
    key = str(variant).lower().strip().replace("-", "_")
    if key not in VARIANT_PARAMS:
        raise ValueError(
            f"unknown 3D-TransUNet variant {variant!r}; "
            f"choose one of {list(VARIANT_PARAMS)}"
        )
    return dict(VARIANT_PARAMS[key])


def build_official_transunet3d(
    *,
    variant: str,
    num_input_channels: int,
    num_classes: int,
    patch_size: Sequence[int],
    pool_op_kernel_sizes: Sequence[Sequence[int]],
    conv_kernel_sizes: Optional[Sequence[Sequence[int]]] = None,
    base_num_features: int = 32,
    conv_per_stage: int = 2,
    deep_supervision: Optional[bool] = None,
    model_params_override: Optional[Dict[str, Any]] = None,
) -> Generic_TransUNet_max_ppbp:
    """Build ``Generic_TransUNet_max_ppbp`` exactly as official train/inference do."""
    params = get_variant_model_params(variant)
    if model_params_override:
        params.update(model_params_override)

    raw_pool = [tuple(int(x) for x in s) for s in pool_op_kernel_sizes]
    dropped_identity = len(raw_pool) > 1 and all(int(x) == 1 for x in raw_pool[0])
    pool = raw_pool[1:] if dropped_identity else raw_pool
    num_pool = len(pool)
    if conv_kernel_sizes is None:
        conv_ks = [(3, 3, 3)] * (num_pool + 1)
    else:
        conv_ks = [tuple(int(x) for x in s) for s in conv_kernel_sizes]
        if dropped_identity and len(conv_ks) == len(raw_pool):
            conv_ks = conv_ks[1:]
        if len(conv_ks) < num_pool + 1:
            conv_ks = list(conv_ks) + [(3, 3, 3)] * (num_pool + 1 - len(conv_ks))
        elif len(conv_ks) > num_pool + 1:
            conv_ks = conv_ks[: num_pool + 1]

    # Official BraTS decoder_only: disable_ds=True; encoder_only: disable_ds=False
    if deep_supervision is None:
        deep_supervision = variant.lower().replace("-", "_") == "encoder_only"

    do_ds = bool(deep_supervision)
    # Official initialize_network: convolutional_upsampling=True (unless is_fam)
    convolutional_upsampling = True

    net = Generic_TransUNet_max_ppbp(
        int(num_input_channels),
        int(base_num_features),
        int(num_classes),
        num_pool,
        int(conv_per_stage),
        2,
        nn.Conv3d,
        nn.InstanceNorm3d,
        {"eps": 1e-5, "affine": True},
        nn.Dropout3d,
        {"p": 0, "inplace": True},
        nn.LeakyReLU,
        {"negative_slope": 1e-2, "inplace": True},
        do_ds,
        False,
        lambda x: x,
        InitWeights_He(1e-2),
        pool,
        conv_ks,
        False,
        True,
        convolutional_upsampling=convolutional_upsampling,
        patch_size=list(int(x) for x in patch_size),
        **params,
    )
    net.inference_apply_nonlin = softmax_helper
    # Diagnose ViT pos-embed size (must match checkpoint). ACDC 3d_fullres default:
    # patch=[224,256,10] + anisotropic pools → feat=[7,8,5] → 280 tokens.
    if getattr(net, "is_max_bottleneck_transformer", False):
        meta = {
            "patch_size": list(int(x) for x in patch_size),
            "pool": [list(p) for p in pool],
            "feat_size": list(getattr(net, "_transunet_feat_size", ())),
            "num_tokens": int(getattr(net, "_transunet_num_tokens", -1)),
            "variant": variant,
        }
        net._transunet_build_meta = meta
        print(
            f"[TransUNet3D] build {variant}: patch={meta['patch_size']} "
            f"pool={meta['pool']} feat={meta['feat_size']} tokens={meta['num_tokens']}"
        )
    return net


def _pool_without_identity(pool_op_kernel_sizes: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    raw = [tuple(int(x) for x in s) for s in pool_op_kernel_sizes]
    if len(raw) > 1 and all(int(x) == 1 for x in raw[0]):
        return raw[1:]
    return raw


def expected_transunet_tokens(
    patch_size: Sequence[int], pool_op_kernel_sizes: Sequence[Sequence[int]]
) -> Tuple[List[int], int]:
    """ViT bottleneck tokens for TransUNet3D (matches ``Generic_TransUNet_max_ppbp``)."""
    import numpy as np

    pool = _pool_without_identity(pool_op_kernel_sizes)
    prod = np.prod(np.array(pool), axis=0).astype(np.float64)
    npa = np.log2(prod).astype(np.int64)
    feat = [int(patch_size[i]) // (2 ** int(npa[i])) for i in range(3)]
    return feat, int(feat[0] * feat[1] * feat[2])


def align_configuration_patch_to_transunet_checkpoint(
    configuration_manager, state_dict: dict
) -> Optional[List[int]]:
    """If plans patch yields wrong ViT token count vs checkpoint, fix ``patch_size`` in-place.

    Common ACDC failure: results ``plans.json`` has D padded to 32 (tokens=896) while the
    checkpoint was trained with D=10 (tokens=280). Sliding-window and ViT pos-embed must
    use the training patch; rewrite ``configuration['patch_size']`` when uniquely recoverable.
    """
    key = "transformer.embeddings.position_embeddings"
    if key not in state_dict:
        return None
    n_ckpt = int(state_dict[key].shape[1])
    patch = [int(x) for x in configuration_manager.patch_size]
    strides = configuration_manager.pool_op_kernel_sizes
    feat, n_plans = expected_transunet_tokens(patch, strides)
    if n_plans == n_ckpt and min(feat) >= 1:
        return None

    import numpy as np

    pool = _pool_without_identity(strides)
    npa = np.log2(np.prod(np.array(pool), axis=0).astype(np.float64)).astype(np.int64)
    # Prefer correcting the smallest axis first (ACDC depth).
    for axis in sorted(range(3), key=lambda i: patch[i]):
        if feat[axis] < 1:
            continue
        other = n_plans // feat[axis]
        if other < 1 or n_ckpt % other != 0:
            continue
        new_feat_ax = n_ckpt // other
        new_patch = list(patch)
        new_patch[axis] = int(new_feat_ax) * (2 ** int(npa[axis]))
        feat2, n2 = expected_transunet_tokens(new_patch, strides)
        if n2 == n_ckpt and min(feat2) >= 1:
            print(
                f"[TransUNet3D] align patch_size {patch} → {new_patch} "
                f"(plans tokens={n_plans} feat={feat} → ckpt tokens={n_ckpt} feat={feat2})"
            )
            configuration_manager.configuration["patch_size"] = new_patch
            return new_patch

    raise RuntimeError(
        f"TransUNet3D ViT pos-embed mismatch: checkpoint tokens={n_ckpt}, "
        f"plans patch={patch} → feat={feat} tokens={n_plans}. "
        f"Fix plans.json patch_size to match training (ACDC default [224,256,10] → 280)."
    )


class TransUNet3DPredictWrapper(nn.Module):
    """nnU-Net predict/TTA expects a Tensor; Mask2Former heads return a dict at train time.

    Only used on the predict path (not training): semantic_inference → soft class maps.
    ``load_state_dict`` accepts bare checkpoint keys (no ``net.`` prefix) because
    ``predict_logits_from_preprocessed_data`` reloads fold weights onto ``self.network``.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, dict) and "pred_logits" in out and "pred_masks" in out:
            cls = out["pred_logits"].softmax(-1)[..., :-1]
            return torch.einsum("bqc,bqdhw->bcdhw", cls, out["pred_masks"].sigmoid())
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        if state_dict and next(iter(state_dict)).startswith("net."):
            return super().load_state_dict(state_dict, strict=strict)
        return self.net.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)


def wrap_transunet3d_for_nnunet_predict(network: nn.Module) -> nn.Module:
    if isinstance(network, TransUNet3DPredictWrapper):
        return network
    return TransUNet3DPredictWrapper(network)


# Public alias used by trainers / docs
TransUNet3D = Generic_TransUNet_max_ppbp

__all__ = [
    "TransUNet3D",
    "Generic_TransUNet_max_ppbp",
    "InitWeights_He",
    "HungarianMatcher3D",
    "compute_loss_hungarian",
    "softmax_helper",
    "build_official_transunet3d",
    "get_variant_model_params",
    "align_configuration_patch_to_transunet_checkpoint",
    "expected_transunet_tokens",
    "TransUNet3DPredictWrapper",
    "wrap_transunet3d_for_nnunet_predict",
    "VARIANT_PARAMS",
    "BRATS_DECODER_ONLY_PARAMS",
    "BRATS_ENCODER_ONLY_PARAMS",
    "BRATS_ENCODER_PLUS_DECODER_PARAMS",
    "BRATS_TRAIN_DEFAULTS",
    "_OFFICIAL_ROOT",
]
