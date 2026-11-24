"""SwinUnet network definition.

Visual architecture diagram (default: num_pool=5, conv_per_stage=2):
Architecture overview:
  - PatchEmbed -> Swin stages -> Swin decoder with skip fusion
  - shifted window attention in a U-Net symmetric structure

Source and citation:
  - Source: https://arxiv.org/abs/2105.05537
  - Date: 2021
  - Venue: preprint
  - Paper: Swin-Unet (arXiv:2105.05537)
  - Copyright: Microsoft Corporation - MIT License

3D support:
  - The original Swin-Unet is 2D; this implementation supports 2D/3D via spatial_dims.

Input (B, C, D, H, W)
    |
    v
+-----------------------------+
|         Encoder             |
| PatchEmbed + Swin stages    |
+-----------------------------+
    |
    v
+-----------------------------+
|         Bottleneck          |
| shifted window attention    |
+-----------------------------+
    |
    v
+-----------------------------+
|          Decoder            |
| Up + skip fusion            |
+-----------------------------+
    |
    v
+-----------------------------+
|      Segmentation Head      |
| 1x1 Conv -> classes         |
+-----------------------------+
"""

from __future__ import annotations

import copy
import itertools
import logging
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

logger = logging.getLogger(__name__)

__all__ = [
    "SwinUnet",
    "SwinTransformerSys",
    "BasicLayer",
    "BasicLayer_up",
    "SwinTransformerBlock",
    "WindowAttention",
    "PatchMerging",
    "PatchExpand",
    "PatchEmbed",
]


def ensure_tuple_rep(tup, dim):
    """Ensure the input is a tuple of length dim."""
    if isinstance(tup, (list, tuple)):
        if len(tup) == dim:
            return tuple(tup)
        raise ValueError(f"Sequence length {len(tup)} != expected {dim}")
    return tuple([tup] * dim)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization."""
    with torch.no_grad():
        l = (1.0 + torch.erf(torch.tensor((a - mean) / std / np.sqrt(2.0)))) / 2.0
        u = (1.0 + torch.erf(torch.tensor((b - mean) / std / np.sqrt(2.0)))) / 2.0
        tensor.uniform_(2 * l.item() - 1, 2 * u.item() - 1)
        tensor.erfinv_()
        tensor.mul_(std * np.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP block for Swin Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: Sequence[int]):
    """
    Partition image into non-overlapping windows.
    Supports both 2D and 3D inputs.

    Args:
        x: input tensor with shape (B, *spatial_dims, C)
        window_size: window size for each spatial dimension

    Returns:
        windows: (num_windows*B, window_volume, C)
    """
    if len(x.shape) == 4:  # 2D: (B, H, W, C)
        B, H, W, C = x.shape
        x = x.view(
            B,
            H // window_size[0], window_size[0],
            W // window_size[1], window_size[1],
            C
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size[0] * window_size[1], C)
    else:  # 3D: (B, D, H, W, C)
        B, D, H, W, C = x.shape
        x = x.view(
            B,
            D // window_size[0], window_size[0],
            H // window_size[1], window_size[1],
            W // window_size[2], window_size[2],
            C
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows


def window_reverse(windows, window_size: Sequence[int], dims: Sequence[int]):
    """
    Reverse window partition.

    Args:
        windows: (num_windows*B, window_volume, C)
        window_size: window size for each spatial dimension
        dims: spatial dimensions (H, W) or (D, H, W)

    Returns:
        x: (B, *spatial_dims, C)
    """
    if len(dims) == 2:  # 2D
        H, W = dims
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        x = windows.view(
            B,
            H // window_size[0], W // window_size[1],
            window_size[0], window_size[1],
            -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    else:  # 3D
        D, H, W = dims
        B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
        x = windows.view(
            B,
            D // window_size[0], H // window_size[1], W // window_size[2],
            window_size[0], window_size[1], window_size[2],
            -1
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Adjust window size based on input size."""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size, device):
    """Compute attention mask for shifted window attention."""
    if len(dims) == 2:
        H, W = dims
        img_mask = torch.zeros((1, H, W, 1), device=device)
        slices = []
        for dim_size, w_size, s_size in zip([H, W], window_size, shift_size):
            slices.append([
                slice(0, -w_size),
                slice(-w_size, -s_size) if s_size > 0 else slice(-w_size, None),
                slice(-s_size, None) if s_size > 0 else slice(0, 0)
            ])
        cnt = 0
        for h in slices[0]:
            for w in slices[1]:
                if h.start == h.stop or w.start == w.stop:
                    continue
                img_mask[:, h, w, :] = cnt
                cnt += 1
    else:  # 3D
        D, H, W = dims
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        for d in [slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)]:
            for h in [slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)]:
                for w in [slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)]:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position bias.
    Supports both 2D and 3D.
    """

    def __init__(
        self,
        dim: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.spatial_dims = len(window_size)

        # Relative position bias table
        if self.spatial_dims == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        else:  # 3D
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                    num_heads
                )
            )
            coords_d = torch.arange(window_size[0])
            coords_h = torch.arange(window_size[1])
            coords_w = torch.arange(window_size[2])
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 2] += window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block supporting 2D and 3D.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Sequence[int],
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, list] = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = list(window_size)
        self.shift_size = list(shift_size)
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = len(input_resolution)

        # Adjust window size if input is smaller
        for i in range(self.spatial_dims):
            if self.input_resolution[i] <= self.window_size[i]:
                self.shift_size[i] = 0
                self.window_size[i] = self.input_resolution[i]

        self.window_size = tuple(self.window_size)
        self.shift_size = tuple(self.shift_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        x = self.norm1(x)

        if self.spatial_dims == 2:
            B, H, W, C = x.shape
            window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

            pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            _, Hp, Wp, _ = x.shape
            dims = [Hp, Wp]

            if any(s > 0 for s in shift_size):
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None

        else:  # 3D
            B, D, H, W, C = x.shape
            window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

            pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d))
            _, Dp, Hp, Wp, _ = x.shape
            dims = [Dp, Hp, Wp]

            if any(s > 0 for s in shift_size):
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if self.spatial_dims == 2:
            if any(s > 0 for s in shift_size):
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
            else:
                x = shifted_x
            if pad_b > 0 or pad_r > 0:
                x = x[:, :H, :W, :].contiguous()
        else:
            if any(s > 0 for s in shift_size):
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                x = shifted_x
            if pad_d > 0 or pad_b > 0 or pad_r > 0:
                x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        if self.spatial_dims == 2:
            B, L, C = x.shape
            H, W = self.input_resolution
            x = x.view(B, H, W, C)
        else:
            B, L, C = x.shape
            D, H, W = self.input_resolution
            x = x.view(B, D, H, W, C)

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)

        x = shortcut + self.drop_path(x)

        if self.spatial_dims == 2:
            x = x.view(B, H * W, C)
        else:
            x = x.view(B, D * H * W, C)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer supporting 2D and 3D."""

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)
        else:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, L, C = x.shape

        if self.spatial_dims == 2:
            H, W = self.input_resolution
            x = x.view(B, H, W, C)
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
        else:
            D, H, W = self.input_resolution
            x = x.view(B, D, H, W, C)
            pad_d = (2 - D % 2) % 2
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

            x = torch.cat([
                x[:, i::2, j::2, k::2, :]
                for i, j, k in itertools.product(range(2), range(2), range(2))
            ], -1)

        x = x.view(B, -1, x.shape[-1])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    """Patch Expanding Layer supporting 2D and 3D."""

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.spatial_dims = spatial_dims
        
        # 输出通道数为 dim // dim_scale
        self.output_dim = dim // dim_scale

        if spatial_dims == 2:
            # 2D: 上采样 2x2，需要 4 倍的特征来填充
            # 输入 dim，输出 dim // 2，需要 (dim // 2) * 4 = 2 * dim
            self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        else:
            # 3D: 上采样 2x2x2，需要 8 倍的特征来填充
            # 输入 dim，输出 dim // 2，需要 (dim // 2) * 8 = 4 * dim
            self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        B, L, C = x.shape

        if self.spatial_dims == 2:
            H, W = self.input_resolution
            x = x.view(B, H, W, C)
            x = self.expand(x)  # B, H, W, 2*C
            C_expanded = x.shape[-1]
            # reshape to B, H, W, 2, 2, C//2
            x = x.view(B, H, W, 2, 2, C_expanded // 4)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C_expanded // 4)
            x = x.view(B, -1, C_expanded // 4)
        else:
            D, H, W = self.input_resolution
            x = x.view(B, D, H, W, C)
            x = self.expand(x)  # B, D, H, W, 4*C
            C_expanded = x.shape[-1]
            # reshape to B, D, H, W, 2, 2, 2, C//2
            x = x.view(B, D, H, W, 2, 2, 2, C_expanded // 8)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D * 2, H * 2, W * 2, C_expanded // 8)
            x = x.view(B, -1, C_expanded // 8)

        x = self.norm(x)
        return x


class FinalPatchExpand(nn.Module):
    """Final Patch Expanding Layer with configurable upsampling."""

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim: int,
        patch_size: Sequence[int],
        norm_layer: nn.Module = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims

        scale = int(np.prod(patch_size))
        self.expand = nn.Linear(dim, dim * scale, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, L, C = x.shape

        if self.spatial_dims == 2:
            H, W = self.input_resolution
            x = x.view(B, H, W, C)
            x = self.expand(x)
            C_new = x.shape[-1]
            p1, p2 = self.patch_size
            x = x.view(B, H, W, p1, p2, C_new // (p1 * p2))
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * p1, W * p2, -1)
            x = x.view(B, -1, x.shape[-1])
        else:
            D, H, W = self.input_resolution
            x = x.view(B, D, H, W, C)
            x = self.expand(x)
            C_new = x.shape[-1]
            p1, p2, p3 = self.patch_size
            x = x.view(B, D, H, W, p1, p2, p3, C_new // (p1 * p2 * p3))
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D * p1, H * p2, W * p3, -1)
            x = x.view(B, -1, x.shape[-1])

        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """Basic Swin Transformer layer for encoder."""

    def __init__(
        self,
        dim: int,
        input_resolution: Sequence[int],
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, list] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=tuple(0 for _ in window_size) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, spatial_dims=spatial_dims
            )
        else:
            self.downsample = None

    def forward(self, x):
        # Compute attention mask for shifted window attention
        if self.spatial_dims == 2:
            H, W = self.input_resolution
            window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
            Hp = int(np.ceil(H / window_size[0])) * window_size[0]
            Wp = int(np.ceil(W / window_size[1])) * window_size[1]
            attn_mask = compute_mask([Hp, Wp], window_size, shift_size, x.device)
        else:
            D, H, W = self.input_resolution
            window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
            Dp = int(np.ceil(D / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            attn_mask = compute_mask([Dp, Hp, Wp], window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    """Basic Swin Transformer layer for decoder."""

    def __init__(
        self,
        dim: int,
        input_resolution: Sequence[int],
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, list] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        upsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=tuple(0 for _ in window_size) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer, spatial_dims=spatial_dims
            )
        else:
            self.upsample = None

    def forward(self, x):
        if self.spatial_dims == 2:
            H, W = self.input_resolution
            window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
            Hp = int(np.ceil(H / window_size[0])) * window_size[0]
            Wp = int(np.ceil(W / window_size[1])) * window_size[1]
            attn_mask = compute_mask([Hp, Wp], window_size, shift_size, x.device)
        else:
            D, H, W = self.input_resolution
            window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
            Dp = int(np.ceil(D / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            attn_mask = compute_mask([Dp, Hp, Wp], window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding supporting 2D and 3D."""

    def __init__(
        self,
        img_size: Sequence[int] = (224, 224),
        patch_size: Sequence[int] = (4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.patches_resolution = (
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
                img_size[2] // patch_size[2]
            )
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        if self.spatial_dims == 2:
            x = x.flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        else:
            x = x.flatten(2).transpose(1, 2)  # B, Pd*Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerSys(nn.Module):
    """
    Swin Transformer for semantic segmentation (U-Net like architecture).
    Supports both 2D and 3D.
    """

    def __init__(
        self,
        img_size: Sequence[int] = (224, 224, 224),
        patch_size: Sequence[int] = (4, 4, 4),
        in_chans: int = 1,
        num_classes: int = 2,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        depths_decoder: Sequence[int] = (1, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            spatial_dims=spatial_dims,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if spatial_dims == 2:
                layer_resolution = (
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                )
            else:
                layer_resolution = (
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                    patches_resolution[2] // (2 ** i_layer),
                )

            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=layer_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                spatial_dims=spatial_dims,
            )
            self.layers.append(layer)

        # Build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        # 计算每一层的实际特征图尺寸（考虑下采样）
        self.layer_resolutions = []
        current_resolution = list(patches_resolution)
        for i_layer in range(self.num_layers):
            self.layer_resolutions.append(tuple(current_resolution))
            if i_layer < self.num_layers - 1:
                # 下采样时分辨率减半
                current_resolution = [r // 2 for r in current_resolution]

        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )

            if spatial_dims == 2:
                layer_resolution = (
                    patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                )
            else:
                layer_resolution = (
                    patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    patches_resolution[2] // (2 ** (self.num_layers - 1 - i_layer)),
                )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=layer_resolution,
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                    spatial_dims=spatial_dims,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=layer_resolution,
                    depth=depths_decoder[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:(self.num_layers - 1 - i_layer)]):
                        sum(depths[:(self.num_layers - 1 - i_layer) + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    spatial_dims=spatial_dims,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        # Final upsampling and output
        self.up = FinalPatchExpand(
            input_resolution=patches_resolution,
            dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer,
            spatial_dims=spatial_dims,
        )

        if spatial_dims == 2:
            self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)
        else:
            self.output = nn.Conv3d(embed_dim, num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Encoder forward pass."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        """Decoder forward pass with skip connections."""
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip = x_downsample[self.num_layers - 1 - inx]
                # 确保 x 和 skip 的序列长度匹配
                B, L_x, C_x = x.shape
                B, L_skip, C_skip = skip.shape
                
                if L_x != L_skip:
                    # 如果长度不匹配，需要调整
                    # 计算目标分辨率
                    if self.spatial_dims == 2:
                        target_res = self.layer_resolutions[self.num_layers - 1 - inx]
                        target_L = target_res[0] * target_res[1]
                    else:
                        target_res = self.layer_resolutions[self.num_layers - 1 - inx]
                        target_L = target_res[0] * target_res[1] * target_res[2]
                    
                    # 调整 x 的长度以匹配 skip
                    if L_x > L_skip:
                        x = x[:, :L_skip, :]
                    elif L_x < L_skip:
                        # 这种情况不应该发生，但为了安全起见
                        skip = skip[:, :L_x, :]
                
                x = torch.cat([x, skip], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_final(self, x):
        """Final upsampling to original resolution."""
        x = self.up(x)
        B = x.shape[0]

        if self.spatial_dims == 2:
            H, W = self.img_size
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, C, H, W
        else:
            D, H, W = self.img_size
            x = x.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)  # B, C, D, H, W

        x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_final(x)
        return x


class SwinUnet(nn.Module):
    """
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.
    Supports both 2D and 3D medical images.

    Args:
        img_size: input image size (H, W) for 2D or (D, H, W) for 3D
        in_channels: number of input channels
        out_channels: number of output channels
        patch_size: patch size for each dimension
        embed_dim: embedding dimension
        depths: number of layers in each stage
        depths_decoder: number of layers in each decoder stage
        num_heads: number of attention heads
        window_size: window size for each dimension
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: if True, add a learnable bias to query, key, value
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        norm_layer: normalization layer
        ape: if True, add absolute position embedding
        patch_norm: if True, add normalization after patch embedding
        use_checkpoint: whether to use checkpointing to save memory
        spatial_dims: number of spatial dimensions (2 or 3)
        pretrained_path: path to pretrained weights
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = 224,
        in_channels: int = 1,
        out_channels: int = 2,
        patch_size: Union[Sequence[int], int] = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        depths_decoder: Sequence[int] = (1, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Union[Sequence[int], int] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure tuple representations
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=depths_decoder,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        if pretrained_path is not None:
            self.load_from(pretrained_path)

    def forward(self, x):
        return self.swin_unet(x)

    def load_from(self, pretrained_path: str):
        """Load pretrained weights."""
        if pretrained_path is not None:
            print(f"Loading pretrained weights from: {pretrained_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=True)

            if "model" not in pretrained_dict:
                print("Loading pretrained model by splitting keys...")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"Deleting key: {k}")
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                print(f"Load result: {msg}")
                return

            pretrained_dict = pretrained_dict["model"]
            print("Loading pretrained Swin encoder weights...")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            # Copy encoder weights to decoder
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

            # Remove mismatched keys
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(f"Deleting: {k}; pretrain shape: {full_dict[k].shape}; model shape: {model_dict[k].shape}")
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            print(f"Load result: {msg}")
        else:
            print("No pretrained weights provided.")
