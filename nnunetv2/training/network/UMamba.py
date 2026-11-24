"""UMamba network definition.

Reference-style **U‑Net residual encoder–decoder** with **selective state-space
(Mamba / ``mamba_ssm``) mixing**. Top-level API: :class:`UMamba` dispatches to
:class:`UMambaBot` (``variant='bot'``) or :class:`UMambaEnc` (``variant='enc'``).
Supports **2D** (``Conv2d``) and **3D** (``Conv3d``) via ``spatial_dims``.

----------------------------------------------------------------------
Variant summary (see inline comment block below ``__all__`` for trainers)
----------------------------------------------------------------------
- **UMambaBot**: pure residual CNN encoder; **one** :class:`MambaLayer` on the
  **deepest skip** (bottleneck) before decoding. Lighter / faster.
- **UMambaEnc**: :class:`ResidualMambaEncoder` — after **each** encoder stage,
  a :class:`MambaLayer` refines features. Uses ``patch_size`` to decide
  **patch-token** vs **channel-token** Mamba routing per stage (when spatial
  volume ≤ channel count → channel tokens). Stronger global mixing, closer to
  typical UMamba-Enc setups.

----------------------------------------------------------------------
Architecture flow (same boxed ASCII style as ``samu_net.py`` lines 146–200;
example ``n_stages = 6``: ``skip5`` seeds decoding; ``skip4`` … ``skip0``
feed horizontal fusions analogous to SAMU-Net's ``skip`` ladder.)
----------------------------------------------------------------------

Input ``(B, C, H, W)`` or ``(B, C, D, H, W)`` (matches ``spatial_dims``)
   |
   v
+-------------------------------+
| Stem (+ residual downsamp)    |
+-------------------------------+
   |
   v
+-------------------------------+
|   Residual CNN encoder        |
| E0: pool + BasicBlockD xK     |---- skip0 -------------------------------+
| E1: pool + BasicBlockD xK     |---- skip1 --------------------------+    |
| E2: pool + BasicBlockD xK     |---- skip2 ---------------------+    |    |
| E3: pool + BasicBlockD xK     |---- skip3 ---------------+     |    |    |
| E4: pool + BasicBlockD (K~)   |---- skip4 ----------+    |     |    |    |
| E5: bottleneck spatial        |---- skip5 ----------+    |     |    |    |    |
| (Enc: + MambaLayer per stage) |                   |    |     |    |    |
+-------------------------------+                   |    |     |    |    |
   |                                                |    |     |    |    |
   v                                                |    |     |    |    |
+-------------------------------+                   |    |     |    |    |
| MambaLayer @ skip5 (Bot only) |                   |    |     |    |    |
| (Enc skips this block)        |                   |    |     |    |    |
+-------------------------------+                   |    |     |    |    |
   |                                                |    |     |    |    |
   v                                                |    |     |    |    |
+---------------------------------------------------|----|-----|----|----+
|                UNetResDecoder (5)                |    |     |    |    |
| D0: Up + concat(skip4)      <--------------------+    |     |    |    |
| D1: Up + concat(skip3)      <-------------------------+     |    |    |
| D2: Up + concat(skip2)      <-------------------------------+    |    |
| D3: Up + concat(skip1)      <------------------------------------+    |
| D4: Up + concat(skip0)      <-----------------------------------------+
+---------------------------------------------------+
   |
   v
+-------------------------------+
| Seg head(s): Conv 1x1         |
| ``deep_supervision`` / finest |
+-------------------------------+
   |
   v
Output: single logits tensor, or a **list** of scales; ``forward`` reverses
           ordering so finest resolution is typically index ``0``.
           (**Other ``n_stages``:** same ladder idea with ``skip0`` …
           ``skip{N-1}`` and ``N-1`` decoder steps.)

**Dependency**: ``mamba_ssm.Mamba`` must be importable; otherwise
:class:`MambaLayer` raises on construction.

"""


import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.amp import autocast

try:
    from nnunetv2.training.network.mamba_import import get_mamba_class

    get_mamba_class(mode="uni")  # probe only
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD as _BasicBlockD
    _DNA_AVAILABLE = True
except ImportError:
    _DNA_AVAILABLE = False
    _BasicBlockD = None

__all__ = [
    "UMamba",
    "UMambaBot",
    "UMambaEnc",
    "UMambaConfig",
    "get_umamba_config",
    "create_umamba_from_config",
]


# =============================================================================
# Configuration

# 模型结构
# 变体	variant='bot' UMambaBot	variant='enc' UMambaEnc
# 编码器	纯残差卷积 (UNetResEncoder)	残差卷积 + 每阶段一个 MambaLayer
# Mamba 位置	仅瓶颈层 (bottleneck)	编码器每个阶段
# 特点	轻量、更快	全局依赖建模更强

# 训练器类名	变体	深度监督	适用场景
# nnUNetTrainerUMambaBotDS	Bot	✅	显存有限，兼顾精度
# nnUNetTrainerUMambaBotNoDS	Bot	❌	与 TransUNet/MambaUNet 公平对比
# nnUNetTrainerUMambaEncDS	Enc	✅	推荐，最接近原论文设置
# nnUNetTrainerUMambaEncNoDS	Enc	❌	单头推理部署 / 消融实验
# =============================================================================

@dataclass
class MambaConfig:
    """Mamba SSM hyper-parameters."""
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class UMambaConfig:
    """Complete UMamba configuration (mirrors TransUNet3DConfig style)."""
    # General
    spatial_dims: int = 3                        # 2 or 3
    in_channels: int = 1
    num_classes: int = 2
    patch_size: Tuple[int, ...] = (128, 128, 128)

    # Encoder / decoder
    n_stages: int = 6
    base_num_features: int = 32
    max_num_features: int = 320
    n_conv_per_stage: Union[int, List[int]] = 2
    n_conv_per_stage_decoder: Union[int, List[int]] = 2
    feat_map_mul_on_downscale: int = 2

    # Kernel / stride sizes (None → auto-derived)
    conv_kernel_sizes: Optional[List[Tuple[int, ...]]] = None
    pool_op_kernel_sizes: Optional[List[Tuple[int, ...]]] = None

    # Conv options
    conv_bias: bool = True
    norm_op_kwargs: dict = field(default_factory=lambda: {'eps': 1e-5, 'affine': True})
    nonlin_kwargs: dict = field(default_factory=lambda: {'inplace': True})

    # Mamba options
    mamba: MambaConfig = field(default_factory=MambaConfig)

    # Training
    deep_supervision: bool = True

    # Architecture variant
    variant: str = 'enc'   # 'enc' → UMambaEnc, 'bot' → UMambaBot

    def __post_init__(self):
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)

        if self.conv_kernel_sizes is None:
            k = (3,) * self.spatial_dims
            self.conv_kernel_sizes = [k] * self.n_stages

        if self.pool_op_kernel_sizes is None:
            # first stage stride=1 (no downsampling), remaining stride=2
            s1 = (1,) * self.spatial_dims
            s2 = (2,) * self.spatial_dims
            self.pool_op_kernel_sizes = [s1] + [s2] * (self.n_stages - 1)

        if self.mamba is None:
            self.mamba = MambaConfig()


def get_umamba_config(variant: str = 'enc', spatial_dims: int = 3) -> UMambaConfig:
    """Return a default UMamba config."""
    return UMambaConfig(
        variant=variant,
        spatial_dims=spatial_dims,
    )


def create_umamba_from_config(config: UMambaConfig) -> 'UMamba':
    """Instantiate a UMamba model from a UMambaConfig."""
    return UMamba.from_config(config)


# =============================================================================
# Utility: Weight Initialisation
# =============================================================================

class InitWeights_He:
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


# =============================================================================
# Building Blocks
# =============================================================================

class UpsampleLayer(nn.Module):
    """Nearest-neighbour upsample + 1×1 projection."""

    def __init__(self, conv_op, input_channels: int, output_channels: int,
                 pool_op_kernel_size, mode: str = 'nearest'):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        return self.conv(x)


class BasicResBlock(nn.Module):
    """Two-layer residual block (Conv-Norm-Act × 2) with optional 1×1 shortcut."""

    def __init__(self, conv_op, input_channels: int, output_channels: int,
                 norm_op, norm_op_kwargs: dict,
                 kernel_size=3, padding=1, stride=1,
                 use_1x1conv: bool = False,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        self.conv1 = conv_op(input_channels, output_channels,
                             kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels,
                             kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        self.conv3 = conv_op(input_channels, output_channels,
                             kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = y + x
        return self.act2(y)


if _DNA_AVAILABLE:
    # Use the official dynamic_network_architectures implementation directly
    BasicBlockD = _BasicBlockD
else:
    class BasicBlockD(nn.Module):
        """
        Fallback residual block matching dynamic_network_architectures.BasicBlockD.
        Used only when the dna library is not installed.
        Order: Conv → Norm → Act (post-norm, matching original UMamba).
        """

        def __init__(self, conv_op, input_channels: int, output_channels: int,
                     kernel_size, stride=1, conv_bias: bool = False,
                     norm_op=None, norm_op_kwargs=None,
                     nonlin=nn.LeakyReLU, nonlin_kwargs=None):
            super().__init__()
            if nonlin_kwargs is None:
                nonlin_kwargs = {'inplace': True}
            if norm_op_kwargs is None:
                norm_op_kwargs = {}
            padding = [k // 2 for k in kernel_size] if isinstance(kernel_size, (list, tuple)) else kernel_size // 2

            self.conv1 = conv_op(input_channels, output_channels,
                                 kernel_size, stride=stride, padding=padding, bias=conv_bias)
            self.norm1 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
            self.act1 = nonlin(**nonlin_kwargs)

            self.conv2 = conv_op(output_channels, output_channels,
                                 kernel_size, padding=padding, bias=conv_bias)
            self.norm2 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
            self.act2 = nonlin(**nonlin_kwargs)

            self.shortcut = (nn.Identity() if input_channels == output_channels and stride == 1
                             else conv_op(input_channels, output_channels,
                                          kernel_size=1, stride=stride, bias=conv_bias))

        def forward(self, x):
            y = self.act1(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return self.act2(y + self.shortcut(x))


# =============================================================================
# Mamba Layer
# =============================================================================

class MambaLayer(nn.Module):
    """
    Wraps mamba_ssm.Mamba for spatial feature maps.

    Two tokenisation modes:
      - patch_token  (default): spatial positions as tokens, channels as features
      - channel_token: channels as tokens, spatial positions as features
        (used when feature-map volume is smaller than channel count)
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, channel_token: bool = False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install it with: pip install mamba-ssm")
        self.dim = dim
        self.channel_token = channel_token
        self.norm = nn.LayerNorm(dim)
        from nnunetv2.training.network.mamba_import import get_mamba_class

        MambaCls = get_mamba_class(mode="uni")
        self.mamba = MambaCls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    # ---- patch-token forward: tokens = H*W(*D), features = C ----
    def _forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        n_tokens = int(np.prod(img_dims))
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)   # (B, n, C)
        x_out = self.mamba(self.norm(x_flat))
        return x_out.transpose(-1, -2).reshape(B, C, *img_dims)

    # ---- channel-token forward: tokens = C, features = H*W(*D) ----
    def _forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)                                   # (B, C, spatial)
        x_out = self.mamba(self.norm(x_flat))
        return x_out.reshape(B, C, *img_dims)

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        if self.channel_token:
            return self._forward_channel_token(x)
        return self._forward_patch_token(x)


# =============================================================================
# Encoders
# =============================================================================

def _make_stage(conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                conv_bias, in_ch, out_ch, kernel_size, padding, stride,
                n_blocks):
    """Build a single encoder stage: BasicResBlock + (n_blocks-1) × BasicBlockD."""
    layers = [
        BasicResBlock(
            conv_op=conv_op, input_channels=in_ch, output_channels=out_ch,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            kernel_size=kernel_size, padding=padding, stride=stride,
            use_1x1conv=True, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
        )
    ]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlockD(
            conv_op=conv_op, input_channels=out_ch, output_channels=out_ch,
            kernel_size=kernel_size, stride=1, conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
        ))
    return nn.Sequential(*layers)


def _to_list_stride(stride, spatial_dims: int):
    """Normalise a stride entry to a list of length spatial_dims (mirrors maybe_convert_scalar_to_list)."""
    if isinstance(stride, (int, float)):
        return [int(stride)] * spatial_dims
    return [int(s) for s in stride]


class _BaseEncoder(nn.Module):
    """Shared bookkeeping for both encoder types."""

    def __init__(self, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 conv_bias, kernel_sizes, strides, features_per_stage,
                 n_blocks_per_stage, input_channels):
        super().__init__()
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.output_channels = features_per_stage
        # Normalise strides to lists (mirrors maybe_convert_scalar_to_list from dna)
        spatial_dims = 3 if conv_op in (nn.Conv3d,) else 2
        self.strides = [_to_list_stride(s, spatial_dims) for s in strides]
        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

    def _build_stem(self, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                    conv_bias, in_ch, stem_ch, kernel_size, padding, n_blocks):
        layers = [
            BasicResBlock(
                conv_op=conv_op, input_channels=in_ch, output_channels=stem_ch,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_size, padding=padding,
                stride=1, use_1x1conv=True,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            )
        ]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlockD(
                conv_op=conv_op, input_channels=stem_ch, output_channels=stem_ch,
                kernel_size=kernel_size, stride=1, conv_bias=conv_bias,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            ))
        return nn.Sequential(*layers)


class UNetResEncoder(_BaseEncoder):
    """
    Pure residual encoder (used by UMambaBot).
    No Mamba layers; Mamba is only applied at the bottleneck.
    """

    def __init__(self, input_channels: int, n_stages: int,
                 features_per_stage: List[int], conv_op, kernel_sizes,
                 strides, n_blocks_per_stage: List[int],
                 conv_bias: bool, norm_op, norm_op_kwargs: dict,
                 nonlin, nonlin_kwargs: dict, return_skips: bool = False,
                 stem_channels: int = None, pool_type: str = 'conv'):
        super().__init__(conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                         conv_bias, kernel_sizes, strides, features_per_stage,
                         n_blocks_per_stage, input_channels)
        self.return_skips = return_skips
        stem_ch = features_per_stage[0]

        self.stem = self._build_stem(
            conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            conv_bias, input_channels, stem_ch,
            kernel_sizes[0], self.conv_pad_sizes[0], n_blocks_per_stage[0],
        )

        stages = []
        in_ch = stem_ch
        for s in range(n_stages):
            stages.append(_make_stage(
                conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                conv_bias, in_ch, features_per_stage[s],
                kernel_sizes[s], self.conv_pad_sizes[s], self.strides[s],
                n_blocks_per_stage[s],
            ))
            in_ch = features_per_stage[s]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class ResidualMambaEncoder(_BaseEncoder):
    """
    Residual encoder with a Mamba layer at every stage (used by UMambaEnc).

    At stages where the feature-map volume ≤ channel count, channel-token
    mode is used; otherwise patch-token mode is used.
    """

    def __init__(self, input_size: Tuple[int, ...], input_channels: int,
                 n_stages: int, features_per_stage: List[int], conv_op,
                 kernel_sizes, strides, n_blocks_per_stage: List[int],
                 conv_bias: bool, norm_op, norm_op_kwargs: dict,
                 nonlin, nonlin_kwargs: dict, return_skips: bool = False,
                 stem_channels: int = None, pool_type: str = 'conv',
                 mamba_cfg: MambaConfig = None):
        super().__init__(conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                         conv_bias, kernel_sizes, strides, features_per_stage,
                         n_blocks_per_stage, input_channels)
        if mamba_cfg is None:
            mamba_cfg = MambaConfig()
        self.return_skips = return_skips

        # Use normalised self.strides (set by _BaseEncoder) for feature-map size computation
        do_channel_token = []
        fmap_sizes = []
        cur_size = list(input_size)
        for s in range(n_stages):
            cur_size = [i // j for i, j in zip(cur_size, self.strides[s])]
            fmap_sizes.append(cur_size[:])
            # If any spatial dim underflows to 0, prod is 0; 0 <= C would wrongly
            # enable channel_token and set d_model=0 → mamba_ssm Conv1d groups=0.
            spatial_vol = int(np.prod(cur_size))
            do_channel_token.append(
                spatial_vol > 0 and spatial_vol <= features_per_stage[s]
            )

        stem_ch = features_per_stage[0]
        self.stem = self._build_stem(
            conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            conv_bias, input_channels, stem_ch,
            kernel_sizes[0], self.conv_pad_sizes[0], n_blocks_per_stage[0],
        )

        stages, mamba_layers = [], []
        in_ch = stem_ch
        for s in range(n_stages):
            stages.append(_make_stage(
                conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                conv_bias, in_ch, features_per_stage[s],
                kernel_sizes[s], self.conv_pad_sizes[s], self.strides[s],
                n_blocks_per_stage[s],
            ))
            spatial_vol = int(np.prod(fmap_sizes[s]))
            mamba_dim = (
                spatial_vol if do_channel_token[s] else features_per_stage[s]
            )
            mamba_dim = max(int(mamba_dim), 1)
            mamba_layers.append(MambaLayer(
                dim=mamba_dim,
                d_state=mamba_cfg.d_state,
                d_conv=mamba_cfg.d_conv,
                expand=mamba_cfg.expand,
                channel_token=do_channel_token[s],
            ))
            in_ch = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        ret = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = stage(x)
            x = mamba(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# =============================================================================
# Shared Decoder
# =============================================================================

class UNetResDecoder(nn.Module):
    """
    Residual skip-connection decoder shared by both UMambaBot and UMambaEnc.
    """

    def __init__(self, encoder: _BaseEncoder, num_classes: int,
                 n_conv_per_stage: Union[int, List[int]],
                 deep_supervision: bool):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_enc = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_enc - 1)

        stages, upsample_layers, seg_layers = [], [], []

        for s in range(1, n_stages_enc):
            in_ch_below = encoder.output_channels[-s]
            in_ch_skip = encoder.output_channels[-(s + 1)]
            stride_up = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=in_ch_below,
                output_channels=in_ch_skip,
                pool_op_kernel_size=stride_up,
                mode='nearest',
            ))

            # All decoder stages concatenate the skip connection (original UMamba behaviour)
            stages.append(_make_stage(
                conv_op=encoder.conv_op,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
                conv_bias=encoder.conv_bias,
                in_ch=2 * in_ch_skip, out_ch=in_ch_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                padding=encoder.conv_pad_sizes[-(s + 1)],
                stride=1,
                n_blocks=n_conv_per_stage[s - 1],
            ))
            seg_layers.append(
                encoder.conv_op(in_ch_skip, num_classes, 1, 1, 0, bias=True)
            )

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    @staticmethod
    def _align_skip_spatial(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Encoder stride conv vs. scale_factor upsample are not exact inverses on odd sizes."""
        if x.shape[2:] == skip.shape[2:]:
            return skip
        nd = x.dim() - 2
        if nd == 3:
            return F.interpolate(
                skip, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        if nd == 2:
            return F.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return F.interpolate(skip, size=x.shape[2:], mode="nearest")

    def forward(self, skips: List[torch.Tensor]):
        lres = skips[-1]
        seg_outputs = []
        n = len(self.stages)

        for s in range(n):
            x = self.upsample_layers[s](lres)
            # All stages cat with the corresponding encoder skip (mirrors original)
            skip = self._align_skip_spatial(x, skips[-(s + 2)])
            x = torch.cat((x, skip), dim=1)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == n - 1:
                seg_outputs.append(self.seg_layers[-1](x))
            lres = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# =============================================================================
# Top-level model classes
# =============================================================================

class UMambaBot(nn.Module):
    """
    UMamba-Bot: standard residual U-Net encoder with a single Mamba
    layer inserted at the bottleneck before decoding.
    Supports 2D and 3D via conv_op choice.
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int]],
                 conv_op,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage: Union[int, List[int]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int]],
                 conv_bias: bool = True,
                 norm_op=None,
                 norm_op_kwargs: dict = None,
                 dropout_op=None,
                 dropout_op_kwargs: dict = None,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = True,
                 stem_channels: int = None,
                 mamba_cfg: MambaConfig = None):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if mamba_cfg is None:
            mamba_cfg = MambaConfig()

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        n_blocks = n_conv_per_stage
        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * n_stages
        n_blocks = list(n_blocks)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Thin out deeper stages for efficiency (mirrors original UMamba logic)
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks[s] = 1
        n_conv_dec = list(n_conv_per_stage_decoder)
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_dec[s] = 1

        self.encoder = UNetResEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels,
        )

        self.bottleneck_mamba = MambaLayer(
            dim=features_per_stage[-1],
            d_state=mamba_cfg.d_state,
            d_conv=mamba_cfg.d_conv,
            expand=mamba_cfg.expand,
            channel_token=False,
        )

        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_dec,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips[-1] = self.bottleneck_mamba(skips[-1])
        return self.decoder(skips)


class UMambaEnc(nn.Module):
    """
    UMamba-Enc: residual encoder with Mamba at every encoder stage.
    Supports 2D and 3D via conv_op choice.
    """

    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int]],
                 conv_op,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage: Union[int, List[int]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int]],
                 conv_bias: bool = True,
                 norm_op=None,
                 norm_op_kwargs: dict = None,
                 dropout_op=None,
                 dropout_op_kwargs: dict = None,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = True,
                 stem_channels: int = None,
                 mamba_cfg: MambaConfig = None):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if mamba_cfg is None:
            mamba_cfg = MambaConfig()

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        n_blocks = n_conv_per_stage
        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * n_stages
        n_blocks = list(n_blocks)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks[s] = 1
        n_conv_dec = list(n_conv_per_stage_decoder)
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_dec[s] = 1

        self.encoder = ResidualMambaEncoder(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels,
            mamba_cfg=mamba_cfg,
        )

        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_dec,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)


# =============================================================================
# Unified entry-point: UMamba
# =============================================================================

class UMamba(nn.Module):
    """
    Unified UMamba model supporting:
      - 2D or 3D data (spatial_dims)
      - Two Mamba integration variants: 'bot' or 'enc'

    Architecture overview
    ---------------------
    Encoder  → (Stem + N residual stages) with optional per-stage Mamba
    Bottleneck → single Mamba layer (UMambaBot) or last encoder Mamba (UMambaEnc)
    Decoder  → N−1 upsample + skip-cat + residual stages, optional deep supervision

    Parameters
    ----------
    spatial_dims : int
        2 for 2-D data, 3 for 3-D data.
    in_channels : int
        Number of input image channels.
    num_classes : int
        Number of segmentation classes (output channels).
    patch_size : tuple of int
        Spatial size of one training patch (needed by UMambaEnc to determine
        channel-vs-patch tokenisation). Ignored for UMambaBot.
    n_stages : int
        Number of encoder/decoder resolution stages.
    base_num_features : int
        Feature channels at the first encoder stage. Doubles every stage
        up to max_num_features.
    max_num_features : int
        Upper bound on feature channels.
    n_conv_per_stage : int or list
        Residual blocks per encoder stage.
    n_conv_per_stage_decoder : int or list
        Residual blocks per decoder stage.
    pool_op_kernel_sizes : list of tuple
        Stride for each encoder stage (also used as upsample factor in decoder).
    conv_kernel_sizes : list of tuple
        Conv kernel size for each encoder stage.
    variant : str
        'bot' → UMambaBot (Mamba only at bottleneck)
        'enc' → UMambaEnc (Mamba at every encoder stage)
    deep_supervision : bool
        If True, returns a list of segmentation maps at multiple scales.
    mamba_d_state / mamba_d_conv / mamba_expand : int
        Hyper-parameters forwarded to Mamba SSM.
    """

    def __init__(self,
                 spatial_dims: int = 3,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 patch_size: Tuple[int, ...] = (128, 128, 128),
                 n_stages: int = 6,
                 base_num_features: int = 32,
                 max_num_features: int = 320,
                 n_conv_per_stage: Union[int, List[int]] = 2,
                 n_conv_per_stage_decoder: Union[int, List[int]] = 2,
                 pool_op_kernel_sizes: Optional[List[Tuple[int, ...]]] = None,
                 conv_kernel_sizes: Optional[List[Tuple[int, ...]]] = None,
                 conv_bias: bool = True,
                 norm_op_kwargs: Optional[dict] = None,
                 nonlin_kwargs: Optional[dict] = None,
                 variant: str = 'enc',
                 deep_supervision: bool = True,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if variant not in ('bot', 'enc'):
            raise ValueError(f"variant must be 'bot' or 'enc', got '{variant}'")

        self.spatial_dims = spatial_dims
        self.variant = variant
        self.deep_supervision = deep_supervision
        self.patch_size = tuple(patch_size)

        # Select conv/norm ops based on dimensionality
        if spatial_dims == 3:
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        # Auto-build kernel / stride lists
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3,) * spatial_dims] * n_stages
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = (
                [(1,) * spatial_dims] +
                [(2,) * spatial_dims] * (n_stages - 1)
            )

        # Feature counts per stage
        features_per_stage = [
            min(base_num_features * (2 ** i), max_num_features)
            for i in range(n_stages)
        ]

        mamba_cfg = MambaConfig(d_state=mamba_d_state, d_conv=mamba_d_conv,
                                expand=mamba_expand)

        common_kwargs = dict(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=conv_kernel_sizes,
            strides=pool_op_kernel_sizes,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            mamba_cfg=mamba_cfg,
        )

        if variant == 'bot':
            self.model = UMambaBot(**common_kwargs)
        else:
            self.model = UMambaEnc(
                input_size=self.patch_size,
                **common_kwargs,
            )

        self.apply(InitWeights_He(1e-2))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: UMambaConfig) -> 'UMamba':
        return cls(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            patch_size=config.patch_size,
            n_stages=config.n_stages,
            base_num_features=config.base_num_features,
            max_num_features=config.max_num_features,
            n_conv_per_stage=config.n_conv_per_stage,
            n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
            pool_op_kernel_sizes=config.pool_op_kernel_sizes,
            conv_kernel_sizes=config.conv_kernel_sizes,
            conv_bias=config.conv_bias,
            norm_op_kwargs=config.norm_op_kwargs,
            nonlin_kwargs=config.nonlin_kwargs,
            variant=config.variant,
            deep_supervision=config.deep_supervision,
            mamba_d_state=config.mamba.d_state,
            mamba_d_conv=config.mamba.d_conv,
            mamba_expand=config.mamba.expand,
        )


# =============================================================================
# Convenience factory functions
# =============================================================================

def create_umamba_bot_2d(in_channels: int = 1, num_classes: int = 2,
                         patch_size=(512, 512), **kwargs) -> UMamba:
    """UMamba-Bot for 2-D images."""
    return UMamba(spatial_dims=2, in_channels=in_channels,
                  num_classes=num_classes, patch_size=patch_size,
                  variant='bot', **kwargs)


def create_umamba_bot_3d(in_channels: int = 1, num_classes: int = 2,
                         patch_size=(128, 128, 128), **kwargs) -> UMamba:
    """UMamba-Bot for 3-D volumes."""
    return UMamba(spatial_dims=3, in_channels=in_channels,
                  num_classes=num_classes, patch_size=patch_size,
                  variant='bot', **kwargs)


def create_umamba_enc_2d(in_channels: int = 1, num_classes: int = 2,
                         patch_size=(512, 512), **kwargs) -> UMamba:
    """UMamba-Enc for 2-D images."""
    return UMamba(spatial_dims=2, in_channels=in_channels,
                  num_classes=num_classes, patch_size=patch_size,
                  variant='enc', **kwargs)


def create_umamba_enc_3d(in_channels: int = 1, num_classes: int = 2,
                         patch_size=(128, 128, 128), **kwargs) -> UMamba:
    """UMamba-Enc for 3-D volumes."""
    return UMamba(spatial_dims=3, in_channels=in_channels,
                  num_classes=num_classes, patch_size=patch_size,
                  variant='enc', **kwargs)
