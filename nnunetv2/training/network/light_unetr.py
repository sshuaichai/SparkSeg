"""
Light-UNETR — nnU-Net–friendly port of
``training/network/source_code/light_unetr/upstream/fullysup/src/models/lightunetr.py``
and ``.../lightunetr_large.py``.

The reference **base** encoder uses a learnable positional parameter shaped
``(1, embedding_dim, embed_dim)`` where ``embedding_dim`` must equal the bottleneck
token count ``W * H * Z``. For variable nnU-Net patch sizes, set
``auto_embedding_dim=True`` (default) and provide ``spatial_patch_size`` so
:func:`build_light_unetr` can infer ``embedding_dim`` before constructing weights.

Detailed architecture flow (LightUNETR / LightUNETRLarge):

Input: ``(B, C, D, H, W)``
   |
   v
+------------------------------------------------------+
| Overlap patch embedding stem                         |
| - cascaded conv projection (stride-2 then refine)    |
| - output tokens/features at reduced spatial scale    |
+------------------------------------------------------+
   |
   v
+------------------------------------------------------+
| Encoder hierarchy (4 transformer-like stages)        |
| E0: LightUNETRBlock x blocks[0] ---- skip0 -------+  |
| E1: LightUNETRBlock x blocks[1] ---- skip1 ----+   |  |
| E2: LightUNETRBlock x blocks[2] ---- skip2 -+  |   |  |
| E3: LightUNETRBlock x blocks[3] ---- emb    |  |   |  |
|                                              |  |   |  |
| Block internals: depthwise conv + LiDRA +    |  |   |  |
| CGLU + channel attention + residual mixing    |  |   |  |
+-----------------------------------------------|--|---|--+
   |                                             |  |   |
   v                                             |  |   |
+-----------------------------------------------+  |   |
| Decoder hierarchy (multi-scale recover)          |   |
| D2: recover/upsample + fuse skip2  <------------+   |
| D1: recover/upsample + fuse skip1  <----------------+
| D0: recover/upsample + fuse skip0  <-------------------+
| (large variant uses wider stages and deep stacks)      |
+------------------------------------------------------+
   |
   v
+------------------------------------------------------+
| Segmentation head (1x1 Conv3d -> classes)            |
+------------------------------------------------------+
   |
   v
Segmentation logits

See ``nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightUNETR`` for trainers.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        expansion = 4
        self.identity = nn.Conv3d(channels, channels * expansion // 2, kernel_size=1, bias=False)
        self.gating = nn.Conv3d(channels, channels * expansion // 2, kernel_size=1, bias=False)
        self.act = nn.Sigmoid()
        self.recover = nn.Conv3d(channels * expansion // 2, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.identity(x) * self.act(self.gating(x))
        x = self.recover(x)
        return x


class SpatialRecover(nn.Module):
    def __init__(self, channels: int, r: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=r, stride=r, groups=channels),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightDimensionReductiveAttention(nn.Module):
    def __init__(self, channels: int, r: int, heads: int, attn_scale: bool = False):
        super().__init__()
        assert channels % (3 * heads) == 0, f"channels={channels} not divisible by 3*heads={3*heads}"
        self.head_dim = channels // (3 * heads)
        self.scale = self.head_dim**-0.5
        self.attn_scale = attn_scale
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        self.qkv = nn.Conv3d(channels // 3, (channels // 3) * 3, kernel_size=1, bias=False)
        g = max(1, channels // 6)
        self.high_freq = nn.Sequential(
            nn.Conv3d(channels // 3, (channels // 2) // 3, kernel_size=1, bias=False),
            nn.BatchNorm3d((channels // 2) // 3),
            nn.GELU(),
            nn.Conv3d((channels // 2) // 3, channels // 3, kernel_size=3, padding=1, groups=g, bias=False),
            nn.GELU(),
        )
        self.high_freq3 = nn.Sequential(
            nn.Conv3d(channels // 3, (channels // 2) // 3, kernel_size=1, bias=False),
            nn.BatchNorm3d((channels // 2) // 3),
            nn.GELU(),
            nn.Conv3d((channels // 2) // 3, channels // 3, kernel_size=5, padding=2, groups=g, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sparse_sampler(x)
        b, c, h, w, z = x.shape
        x_low, x_high, x_high2 = x.split([c // 3, c // 3, c // 3], dim=1)
        q, k, v = (
            self.qkv(x_low)
            .view(b, self.num_heads, -1, h * w * z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        if self.attn_scale:
            scale = 1.0 / math.sqrt(self.head_dim * (h * w * z))
            attn = (q.transpose(-1, -2) @ k) * scale
            attn = attn.softmax(dim=-1)
        else:
            attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x_low = (v @ attn.transpose(-2, -1)).view(b, -1, h, w, z)
        x_high = self.high_freq(x_high)
        x_high2 = self.high_freq3(x_high2)
        return torch.cat([x_low, x_high, x_high2], dim=1)


class OverlapPatchEmbedding2(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, stride: int):
        super().__init__()
        _ = stride
        self.proj1 = nn.Conv3d(in_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=embed_dim // 4)
        self.proj2 = nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=embed_dim // 2)
        self.proj3 = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.proj1(x)))
        x = self.act(self.norm2(self.proj2(x)))
        return self.norm3(self.proj3(x))


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class LightUNETRBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        r: int,
        heads: int,
        lidra_attn_scale: bool = False,
        use_lidra: bool = True,
    ):
        super().__init__()
        self.use_lidra = bool(use_lidra)
        self.conv0 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        if self.use_lidra:
            self.lidr1 = LightweightDimensionReductiveAttention(channels, r, heads, attn_scale=lidra_attn_scale)
            self.rec1 = SpatialRecover(channels, r)
        self.cglu1 = CGLU(channels)
        if self.use_lidra:
            self.lidr2 = LightweightDimensionReductiveAttention(channels, r, heads, attn_scale=lidra_attn_scale)
            self.rec2 = SpatialRecover(channels, r)
        self.cglu2 = CGLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x) + x
        if self.use_lidra:
            x = self.rec1(self.lidr1(x)) + x
        x = self.cglu1(x) + x
        if self.use_lidra:
            x = self.rec2(self.lidr2(x)) + x
        x = self.cglu2(x) + x
        return x


class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, r: int):
        super().__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.se = SELayer(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.norm(self.depth_wise(x)))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 384,
        embedding_dim: int = 27,
        channels: Tuple[int, int, int] = (48, 96, 240),
        blocks: Tuple[int, int, int, int] = (1, 2, 3, 2),
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        r: Tuple[int, int, int, int] = (4, 2, 2, 1),
        dropout: float = 0.3,
        legacy_abdomen_spatial_pads: bool = False,
        lidra_attn_scale: bool = False,
    ):
        super().__init__()
        self.legacy_abdomen_spatial_pads = legacy_abdomen_spatial_pads
        self.DWconv1 = OverlapPatchEmbedding2(in_channels=in_channels, embed_dim=channels[0], stride=2)
        self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=2)
        self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=2)
        self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=2)

        def _stack(n: int, ch: int, rr: int, hd: int) -> nn.Sequential:
            return nn.Sequential(
                *[LightUNETRBlock(channels=ch, r=rr, heads=hd, lidra_attn_scale=lidra_attn_scale) for _ in range(n)]
            )

        self.block1 = _stack(blocks[0], channels[0], r[0], heads[0])
        self.block2 = _stack(blocks[1], channels[1], r[1], heads[1])
        self.block3 = _stack(blocks[2], channels[2], r[2], heads[2])
        self.block4 = _stack(blocks[3], embed_dim, r[3], heads[3])

        self.position_embeddings = nn.Parameter(torch.zeros(1, embedding_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[int, int, int, int, int]]:
        hidden_states_out: List[torch.Tensor] = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        if self.legacy_abdomen_spatial_pads and tuple(x.shape[2:]) == (7, 7, 5):
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        b, c, w, h, z = x.shape
        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, hidden_states_out, (b, c, w, h, z)


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, r: int):
        super().__init__()
        self.transposed = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.se = SELayer(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.norm(self.transposed(x)))


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        embed_dim: int = 384,
        channels: Tuple[int, int, int] = (48, 96, 240),
        blocks: Tuple[int, int, int, int] = (1, 2, 3, 2),
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        r: Tuple[int, int, int, int] = (4, 2, 2, 1),
        dropout: float = 0.3,
        legacy_abdomen_spatial_pads: bool = False,
        lidra_attn_scale: bool = False,
    ):
        super().__init__()
        _ = dropout
        self.legacy_abdomen_spatial_pads = legacy_abdomen_spatial_pads
        self.SegHead = nn.ConvTranspose3d(channels[0], out_channels, kernel_size=4, stride=4)
        self.TSconv3 = TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=2)
        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=2)
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=2)

        def _stack(n: int, ch: int, rr: int, hd: int) -> nn.Sequential:
            return nn.Sequential(
                *[LightUNETRBlock(channels=ch, r=rr, heads=hd, lidra_attn_scale=lidra_attn_scale) for _ in range(n)]
            )

        self.block1 = _stack(blocks[0], channels[0], r[0], heads[0])
        self.block2 = _stack(blocks[1], channels[1], r[1], heads[1])
        self.block3 = _stack(blocks[2], channels[2], r[2], heads[2])
        self.block4 = _stack(blocks[3], embed_dim, r[3], heads[3])

    def forward(
        self, x: torch.Tensor, hidden_states_out: List[torch.Tensor], x_shape: Tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        b, c, w, h, z = x_shape
        x = x.reshape(b, c, w, h, z)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + hidden_states_out[2]
        x = self.block3(x)
        if self.legacy_abdomen_spatial_pads and tuple(x.shape[2:]) == (8, 8, 6):
            x = F.pad(x, (0, -1, 0, -1, 0, -1))
        x = self.TSconv2(x)
        x = x + hidden_states_out[1]
        x = self.block2(x)
        x = self.TSconv3(x)
        x = x + hidden_states_out[0]
        x = self.block1(x)
        return self.SegHead(x)


class LightUNETR(nn.Module):
    """Reference ``LightUNETR`` (fully supervised abdomen / generic)."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        embed_dim: int = 96,
        embedding_dim: int = 27,
        channels: Tuple[int, int, int] = (24, 48, 60),
        blocks: Tuple[int, int, int, int] = (1, 2, 3, 2),
        heads: Tuple[int, int, int, int] = (1, 2, 4, 4),
        r: Tuple[int, int, int, int] = (4, 2, 2, 1),
        dropout: float = 0.3,
        legacy_abdomen_spatial_pads: bool = False,
        lidra_attn_scale: bool = False,
    ):
        super().__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
            legacy_abdomen_spatial_pads=legacy_abdomen_spatial_pads,
            lidra_attn_scale=lidra_attn_scale,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
            legacy_abdomen_spatial_pads=legacy_abdomen_spatial_pads,
            lidra_attn_scale=lidra_attn_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        emb, skips, shape_t = self.Encoder(x)
        out = self.Decoder(emb, skips, shape_t)
        if out.shape[2:] != spatial:
            out = F.interpolate(out, size=spatial, mode="trilinear", align_corners=False)
        return out


# --- LightUNETRLarge (from ``lightunetr_large.py``) ---
class LightUNETRBlockLarge(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_r: int, kernel_size: int, r: int, heads: int):
        super().__init__()
        _ = out_channels, exp_r, kernel_size
        channels = in_channels
        self.conv0 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.lidr1 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec1 = SpatialRecover(channels, r)
        self.cglu1 = CGLU(channels)
        self.lidr2 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec2 = SpatialRecover(channels, r)
        self.cglu2 = CGLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x) + x
        x = self.rec1(self.lidr1(x)) + x
        x = self.cglu1(x) + x
        x = self.rec2(self.lidr2(x)) + x
        x = self.cglu2(x) + x
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        n_groups: Optional[int] = None,
        r: int = 0,
        heads: int = 0,
    ):
        super().__init__()
        _ = out_channels, r, heads
        self.conv_block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels if n_groups is None else n_groups,
            ),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r * in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_r: int = 4, kernel_size: int = 7):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.res_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_r: int = 4, kernel_size: int = 7):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels
            ),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.res_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_block(x)
        x1 = F.pad(x1, (1, 0, 1, 0, 1, 0))
        res = F.pad(self.res_conv(x), (1, 0, 1, 0, 1, 0))
        return x1 + res


class LightUNETRLarge(nn.Module):
    """Wider encoder–decoder from ``lightunetr_large.py`` (head/neck & large abdomen configs)."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        embedding_dim: int = 27,
        exp_r: int = 2,
        block_cfg: Optional[List[str]] = None,
        channels_cfg: Optional[List[int]] = None,
        block_counts: Optional[List[int]] = None,
        head_counts: Optional[List[int]] = None,
        r_ratios: Optional[Tuple[int, int, int]] = None,
        legacy_abdomen_spatial_pads: bool = False,
    ):
        super().__init__()
        _ = embedding_dim
        if block_cfg is None:
            block_cfg = ["c", "c", "l", "l", "l", "l", "l", "c", "c"]
        if channels_cfg is None:
            channels_cfg = [32, 60, 96, 150, 198, 150, 96, 60, 32]
        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        if head_counts is None:
            # Match official fullysup/src/models/lightunetr_large.py (9 entries).
            # Local had an extra 2 at index 6 → stage-6 heads 2 instead of 1.
            head_counts = [0, 0, 1, 2, 2, 2, 1, 0, 0]
        if r_ratios is None:
            r_ratios = (4, 2, 2)
        self.legacy_abdomen_spatial_pads = legacy_abdomen_spatial_pads

        block_mapper = {"c": ConvBlock, "l": LightUNETRBlockLarge}
        enc_kernel_size = dec_kernel_size = 3
        exp_r_list = [exp_r for _ in range(len(block_counts))]

        self.stem = nn.Conv3d(in_channels, channels_cfg[0], kernel_size=1)

        def stage_heads(stage: int) -> int:
            mapping = {0: 2, 1: 2, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 2, 8: 2}
            idx = mapping.get(stage, 2)
            return head_counts[idx] if idx < len(head_counts) else head_counts[-1]

        def stage_r(stage: int) -> int:
            if stage in (0, 1, 2, 7, 8):
                return r_ratios[0]
            if stage in (3, 5, 6):
                return r_ratios[1]
            return r_ratios[2]

        def make_seq(stage: int, ksz: int) -> nn.Sequential:
            blk_cls = block_mapper[block_cfg[stage]]
            seq = [
                blk_cls(
                    in_channels=channels_cfg[stage],
                    out_channels=channels_cfg[stage],
                    exp_r=exp_r_list[stage],
                    kernel_size=ksz,
                    r=stage_r(stage),
                    heads=stage_heads(stage),
                )
                for _ in range(block_counts[stage])
            ]
            return nn.Sequential(*seq)

        self.enc_block_0 = make_seq(0, 3)
        for i in range(4):
            setattr(
                self,
                f"down_{i}",
                DownBlock(
                    in_channels=channels_cfg[i],
                    out_channels=channels_cfg[i + 1],
                    exp_r=exp_r_list[i + 1],
                    kernel_size=3 if i == 0 else enc_kernel_size,
                ),
            )
            if i < 3:
                setattr(self, f"enc_block_{i + 1}", make_seq(i + 1, enc_kernel_size))

        self.bottleneck = make_seq(4, dec_kernel_size)

        for i in range(4):
            setattr(
                self,
                f"up_{3 - i}",
                UpBlock(
                    in_channels=channels_cfg[4 + i],
                    out_channels=channels_cfg[5 + i],
                    exp_r=exp_r_list[5 + i],
                    kernel_size=dec_kernel_size if i < 3 else 3,
                ),
            )
            dec_stage = 5 + i
            setattr(
                self,
                f"dec_block_{3 - i}",
                make_seq(dec_stage, dec_kernel_size if i < 3 else 3),
            )

        self.out_0 = nn.ConvTranspose3d(channels_cfg[8], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        x = self.enc_block_0(x)
        skips.append(x)
        for i in range(3):
            x = getattr(self, f"down_{i}")(x)
            x = getattr(self, f"enc_block_{i + 1}")(x)
            skips.append(x)
        x = self.down_3(skips[-1])
        if self.legacy_abdomen_spatial_pads and tuple(x.shape[2:]) == (7, 7, 5):
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
        x = self.bottleneck(x)
        if self.legacy_abdomen_spatial_pads and tuple(x.shape[2:]) == (8, 8, 6):
            x = F.pad(x, (0, -1, 0, -1, 0, -1))
        for i in range(3, -1, -1):
            x_up = getattr(self, f"up_{i}")(x)
            x = getattr(self, f"dec_block_{i}")(skips[i] + x_up)
        out = self.out_0(x)
        if out.shape[2:] != spatial:
            out = F.interpolate(out, size=spatial, mode="trilinear", align_corners=False)
        return out


def compute_base_embedding_dim(
    spatial_patch_size: Tuple[int, int, int],
    in_channels: int,
    embed_dim: int,
    channels: Tuple[int, int, int],
    blocks: Tuple[int, int, int, int],
    heads: Tuple[int, int, int, int],
    r: Tuple[int, int, int, int],
    dropout: float,
    legacy_abdomen_spatial_pads: bool,
    lidra_attn_scale: bool,
) -> int:
    """Bottleneck token count ``W*H*Z`` after ``DWconv4`` (matches reference pos-embed length)."""
    enc = Encoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        embedding_dim=1,
        channels=channels,
        blocks=blocks,
        heads=heads,
        r=r,
        dropout=dropout,
        legacy_abdomen_spatial_pads=legacy_abdomen_spatial_pads,
        lidra_attn_scale=lidra_attn_scale,
    )
    x = torch.zeros(1, in_channels, *spatial_patch_size)
    with torch.no_grad():
        x = enc.DWconv1(x)
        x = enc.block1(x)
        x = enc.DWconv2(x)
        x = enc.block2(x)
        x = enc.DWconv3(x)
        if enc.legacy_abdomen_spatial_pads and tuple(x.shape[2:]) == (7, 7, 5):
            x = F.pad(x, (0, 1, 0, 1, 0, 1))
        x = enc.block3(x)
        x = enc.DWconv4(x)
        _, _, w, h, z = x.shape
        return int(w * h * z)


LIGHTUNETR_NET_DEFAULTS: Dict[str, Any] = {
    "lightunetr_variant": "base",
    "embed_dim": 96,
    "embedding_dim": 27,
    "channels": (24, 48, 60),
    "blocks": (1, 2, 3, 2),
    "heads": (1, 2, 4, 4),
    "r": (4, 2, 2, 1),
    "dropout": 0.3,
    "legacy_abdomen_spatial_pads": False,
    "lidra_attn_scale": False,
    "auto_embedding_dim": True,
    "spatial_patch_size": None,
}

LIGHTUNETR_ABDOMENCT1K_DEFAULTS: Dict[str, Any] = {
    **LIGHTUNETR_NET_DEFAULTS,
    "legacy_abdomen_spatial_pads": True,
    "embedding_dim": 27,
    "auto_embedding_dim": False,
}

LIGHTUNETR_LARGE_DEFAULTS: Dict[str, Any] = {
    "lightunetr_variant": "large",
    "exp_r": 2,
    "legacy_abdomen_spatial_pads": False,
}

LIGHTUNETR_LARGE_ABDOMEN_DEFAULTS: Dict[str, Any] = {
    **LIGHTUNETR_LARGE_DEFAULTS,
    "legacy_abdomen_spatial_pads": True,
}

LIGHTUNETR_ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "attn_scale": {"lidra_attn_scale": True},
    "legacy_pads": {"legacy_abdomen_spatial_pads": True, "auto_embedding_dim": False, "embedding_dim": 27},
}

LIGHTUNETR_ABLATION_TRAINER_KEYS: Tuple[str, ...] = tuple(LIGHTUNETR_ABLATION_VARIANTS.keys())


def _merge_lightunetr_ablation(md: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(md)
    key = str(out.get("lightunetr_ablation", "full")).strip().lower()
    if key in (None, "", "none"):
        key = "full"
    delta = LIGHTUNETR_ABLATION_VARIANTS.get(key)
    if delta is None:
        raise ValueError(f"unknown lightunetr_ablation {key!r}; valid: {sorted(LIGHTUNETR_ABLATION_VARIANTS.keys())}")
    out.update(delta)
    return out


def build_light_unetr(
    num_input_channels: int,
    num_output_channels: int,
    model_defaults: Optional[Dict[str, Any]] = None,
    *,
    allow_init: bool = True,
    deep_supervision: bool = False,
) -> nn.Module:
    _ = allow_init, deep_supervision
    md = copy.deepcopy(LIGHTUNETR_NET_DEFAULTS)
    if model_defaults:
        md.update(model_defaults)
    md = _merge_lightunetr_ablation(md)

    variant = str(md.get("lightunetr_variant", "base")).strip().lower()
    legacy = bool(md.get("legacy_abdomen_spatial_pads", False))

    if variant == "large":
        return LightUNETRLarge(
            in_channels=int(num_input_channels),
            n_classes=int(num_output_channels),
            embedding_dim=int(md.get("embedding_dim", 27)),
            exp_r=int(md.get("exp_r", 2)),
            legacy_abdomen_spatial_pads=legacy,
        )

    channels = tuple(int(x) for x in md["channels"])
    blocks = tuple(int(x) for x in md["blocks"])
    heads = tuple(int(x) for x in md["heads"])
    r = tuple(int(x) for x in md["r"])

    embed_dim = int(md["embed_dim"])
    dropout = float(md["dropout"])
    lidra_attn_scale = bool(md.get("lidra_attn_scale", False))

    embedding_dim = md.get("embedding_dim")
    if md.get("auto_embedding_dim", True):
        sp = md.get("spatial_patch_size")
        if sp is None or len(sp) != 3:
            raise ValueError(
                "LightUNETR (base) with auto_embedding_dim=True requires model_defaults['spatial_patch_size'] "
                "as a length-3 tuple (D, H, W) matching nnU-Net patch."
            )
        sp_t = (int(sp[0]), int(sp[1]), int(sp[2]))
        embedding_dim = compute_base_embedding_dim(
            sp_t,
            int(num_input_channels),
            embed_dim,
            channels,
            blocks,
            heads,
            r,
            dropout,
            legacy,
            lidra_attn_scale,
        )
    embedding_dim = int(embedding_dim)

    return LightUNETR(
        in_channels=int(num_input_channels),
        out_channels=int(num_output_channels),
        embed_dim=embed_dim,
        embedding_dim=embedding_dim,
        channels=channels,
        blocks=blocks,
        heads=heads,
        r=r,
        dropout=dropout,
        legacy_abdomen_spatial_pads=legacy,
        lidra_attn_scale=lidra_attn_scale,
    )


__all__ = [
    "LightUNETR",
    "LightUNETRLarge",
    "build_light_unetr",
    "compute_base_embedding_dim",
    "LIGHTUNETR_NET_DEFAULTS",
    "LIGHTUNETR_LARGE_DEFAULTS",
    "LIGHTUNETR_ABDOMENCT1K_DEFAULTS",
    "LIGHTUNETR_LARGE_ABDOMEN_DEFAULTS",
    "LIGHTUNETR_ABLATION_VARIANTS",
    "LIGHTUNETR_ABLATION_TRAINER_KEYS",
]
