"""Auxiliary helpers for SparkSeg (not part of the architecture).

Host geometry, encoder construction, the Top-K budget schedule, spatial alignment,
Top-K score utilities and the post-build initialisation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.network_initialization import InitWeights_He

from .config import RESENC_BLOCKS_S6


def _get_stage_channels(
    base_num_features: int, num_pool: int, max_num_features: Optional[int],
) -> List[int]:
    chans: List[int] = []
    for i in range(int(num_pool) + 1):
        c = int(base_num_features) * (2 ** i)
        if max_num_features is not None:
            c = min(c, int(max_num_features))
        chans.append(int(c))
    return chans


def _hist_ch_for_guide_stage(
    stage_i: int,
    enhanced_prev: set,
    features: Sequence[int],
) -> int:
    """Width of the previous enhanced stage, or the stage's own width if there is none.

    ``guide_stages`` may have gaps (e.g. ``(2, 4)``); at runtime ``hist`` still comes from
    the most recent guide/local output below the current stage.
    """
    prev_enh = [j for j in enhanced_prev if j < int(stage_i)]
    if prev_enh:
        return int(features[max(prev_enh)])
    return int(features[int(stage_i)])


def _normalize_stride3(stride: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int]:
    if isinstance(stride, int):
        return (int(stride),) * 3
    if len(stride) != 3:
        raise ValueError(f"stride must be int or length-3, got {stride!r}")
    return tuple(int(x) for x in stride)


def encoder_strides_from_plan(
    plan_strides: Optional[Sequence[Sequence[int]]], num_pool: int,
) -> Tuple[Tuple[int, int, int], ...]:
    n = int(num_pool) + 1
    if plan_strides is None:
        return tuple(
            (1, 1, 1) if i == 0 else (2, 2, 2) if i < n - 1 else (1, 2, 2)
            for i in range(n)
        )
    rows = [_normalize_stride3(r) for r in plan_strides]
    if len(rows) < n:
        raise ValueError(f"plan strides length {len(rows)} < {n}")
    return tuple(rows[:n])


def _default_resenc_blocks_per_stage(n_stages: int) -> List[int]:
    """ResEnc-L depths (1, 3, 4, 6, 6, 6), padded with the deepest value past S6."""
    t = list(RESENC_BLOCKS_S6)
    return [t[i] if i < len(t) else t[-1] for i in range(max(1, int(n_stages)))]


def _resolve_res_n_blocks(
    encoder_type: str,
    n_stages: int,
    res_n_blocks: Optional[Sequence[int]],
) -> List[int]:
    if res_n_blocks is not None:
        blocks = [max(1, int(x)) for x in res_n_blocks]
        if len(blocks) != int(n_stages):
            raise ValueError(
                f"res_n_blocks length {len(blocks)} != n_stages={n_stages} "
                f"(encoder_type={encoder_type!r})"
            )
        return blocks
    if encoder_type == "resenc":
        return _default_resenc_blocks_per_stage(n_stages)
    raise ValueError(f"unexpected encoder_type for res blocks: {encoder_type!r}")


def _build_encoder(
    encoder_type: str,
    *,
    in_channels: int,
    n_stages: int,
    features: Sequence[int],
    strides: Sequence[Sequence[int]],
    conv_per_stage: int,
    res_n_blocks: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Construct PlainConvEncoder or ResidualEncoder (BasicBlockD).

    ``encoder_type``: ``plainconv`` | ``resenc``.
    """
    enc = str(encoder_type).lower().strip()
    kernel_sizes = [(3, 3, 3)] * int(n_stages)
    stride_list = [tuple(int(x) for x in s) for s in strides]
    norm_kw = dict(
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    if enc == "plainconv":
        from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
            PlainConvEncoder,
        )

        return PlainConvEncoder(
            int(in_channels),
            int(n_stages),
            list(features),
            nn.Conv3d,
            kernel_sizes,
            stride_list,
            [int(conv_per_stage)] * int(n_stages),
            conv_bias=True,
            return_skips=True,
            nonlin_first=False,
            **norm_kw,
        )
    if enc == "resenc":
        blocks = _resolve_res_n_blocks(enc, int(n_stages), res_n_blocks)
        from dynamic_network_architectures.building_blocks.residual import BasicBlockD
        from dynamic_network_architectures.building_blocks.residual_encoders import (
            ResidualEncoder,
        )

        return ResidualEncoder(
            int(in_channels),
            int(n_stages),
            list(features),
            nn.Conv3d,
            kernel_sizes,
            stride_list,
            n_blocks_per_stage=blocks,
            conv_bias=False,
            return_skips=True,
            stem_channels=int(features[0]),
            block=BasicBlockD,
            **norm_kw,
        )
    raise ValueError(
        f"encoder_type must be one of plainconv|resenc, got {encoder_type!r}"
    )


def num_tokens_for_stage(stage: int, *, deep_stage: int = 4, deep_k: int = 16) -> int:
    """Top-K budget per guided stage: ``deep_k`` at ``deep_stage``, ×2 per stage shallower,
    floored at 32 and capped at 512 (Eq. 4–6).

    ``deep_k = 16`` gives the paper schedule **128/64/32/32**; the Table 7 sweep is
    ``deep_k`` = 8 / 16 / 32.
    """
    k = int(deep_k) * 2 ** max(0, int(deep_stage) - int(stage))
    return min(512, max(32, k))


def align_spatial3d(
    feat: torch.Tensor, size: Tuple[int, int, int],
) -> torch.Tensor:
    """Align spatially: pure downsampling uses ``adaptive_avg_pool3d``, otherwise trilinear."""
    tgt = (int(size[0]), int(size[1]), int(size[2]))
    src = (int(feat.shape[2]), int(feat.shape[3]), int(feat.shape[4]))
    if src == tgt:
        return feat
    if all(a >= b for a, b in zip(src, tgt)):
        return F.adaptive_avg_pool3d(feat, tgt)
    return F.interpolate(feat, size=tgt, mode="trilinear", align_corners=False)


def delta_energy_per_sample(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Residual-energy map ``e = mean_c(Δ²) / max_{D,H,W}(·)`` (Eq. 2), used by the prior.

    The max runs over spatial dims (2, 3, 4) only, so nothing leaks across the batch.
    """
    d2 = delta.pow(2).mean(dim=1, keepdim=True)
    return d2 / (d2.amax(dim=(2, 3, 4), keepdim=True) + float(eps))


def topk_coords_from_map(importance: torch.Tensor, k: int) -> torch.Tensor:
    """``[B, 1, D, H, W] -> idx [B, K, 3]`` in C-order flattening; do not call on a
    permuted (non-contiguous) score map."""
    b, _, d, h, w = importance.shape
    flat = importance.view(b, -1)
    k = min(int(k), flat.shape[1])
    _, topk = torch.topk(flat, k, dim=1)
    hw = h * w
    zd = topk // hw
    rem = topk % hw
    return torch.stack([zd, rem // w, rem % w], dim=-1).long()


def apply_topk_jitter(score: torch.Tensor, jitter: float) -> torch.Tensor:
    """Relative jitter ``jitter * per-sample std`` to break near-ties (default 1e-2)."""
    j = float(jitter)
    if j <= 0:
        return score
    dims = tuple(range(1, score.ndim))
    scale = score.detach().std(dim=dims, keepdim=True).clamp_min(1e-6)
    return score + torch.rand_like(score) * (j * scale)


def _init_sigmoid_gate_closed(last_layer: nn.Module, bias: float = -2.5) -> None:
    """Zero the last Linear/Conv and set a constant bias, so Sigmoid ≈ σ(bias) (≈0.075)."""
    nn.init.zeros_(last_layer.weight)
    if getattr(last_layer, "bias", None) is not None:
        nn.init.constant_(last_layer.bias, bias)


def _stock_he_init(module: nn.Module) -> None:
    """Post-build init of the product/host stack: He(1e-2) + zero the last BN before
    each residual add.  Gated by ``stock_he`` (the ``_StockHe`` / ``_NoStockHe`` arms);
    the zero-initialised gates live outside the touched submodules and stay zero."""
    try:
        from dynamic_network_architectures.initialization.weight_init import (
            init_last_bn_before_add_to_0,
        )
    except Exception:  # pragma: no cover - optional dependency
        init_last_bn_before_add_to_0 = None  # type: ignore[assignment]
    module.apply(InitWeights_He(1e-2))
    if init_last_bn_before_add_to_0 is not None:
        module.apply(init_last_bn_before_add_to_0)


__all__ = [
    # host geometry / encoder
    "_get_stage_channels",
    "_hist_ch_for_guide_stage",
    "_normalize_stride3",
    "encoder_strides_from_plan",
    "_default_resenc_blocks_per_stage",
    "_resolve_res_n_blocks",
    "_build_encoder",
    # Top-K budget
    "num_tokens_for_stage",
    # score / spatial utilities
    "align_spatial3d",
    "delta_energy_per_sample",
    "topk_coords_from_map",
    "apply_topk_jitter",
    # initialization
    "_init_sigmoid_gate_closed",
    "_stock_he_init",
]
