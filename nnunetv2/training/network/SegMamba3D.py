"""SegMamba3D network definition.

Visual architecture diagram (default: feat_size=[48, 96, 192, 384], depths=[2,2,2,2]):
Execution policy:
  - Encoder uses 4 Mamba stages (S0-S3), each stage = downsample (except S0 stem) + GSC + Mamba stack + MLP residual
  - Decoder uses UNETR-style upsampling with multi-scale skips from conv encoders
  - Output head is 1x1 Conv3d (official forward: no skip/output interpolate)

Design distinction:
  - SegMamba3D follows official SegMamba: GSC + BiMamba-v3 (nslices) at each scale
  - Global dependency modeling is performed in Mamba token space (flatten -> norm -> Mamba -> reshape)
  - Skip path combines Mamba encoder outputs with convolutional UNETR decoders for dense prediction
  - Tri-axis scan is SegMamba-V2 only — not used here

Paper/source:
  - Paper: SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation
  - arXiv: https://arxiv.org/abs/2401.13560
  - Code: https://github.com/ge-xing/SegMamba
  - Venue: MICCAI 2024
  - Authors: Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu

Input (B, C, D, H, W)
   |
   v
+-----------------------------------------------+
|          Mamba Encoder (4 stages)             |
| S0: Stem Conv7 s2 (in->48) + StageBlock x d0  |---- outs0 ----+
| S1: Downsample2 (48->96) + StageBlock x d1    |---- outs1 --+ |
| S2: Downsample2 (96->192) + StageBlock x d2   |---- outs2 -+ | |
| S3: Downsample2 (192->384) + StageBlock x d3  |---- outs3 + | | |
+-----------------------------------------------+            | | | |
   |                                                           | | | |
   +--------------------> conv encoder skips ------------------+ | | |
   |                                                             | | |
   v                                                             | | |
+----------------------------------------------------------------|-|-|-+
|                  UNETR-style Decoder                           | | | |
| D3: decoder5(enc_hidden, enc4)  <------------------------------+ | | |
| D2: decoder4(dec3, enc3)        <--------------------------------+ | |
| D1: decoder3(dec2, enc2)        <----------------------------------+ |
| D0: decoder2(dec1, enc1)        <------------------------------------+
| DB: decoder1(dec0) -> out(1x1 conv)                                  |
+-----------------------------------------------------------------------+
   |
   v
+-----------------------------+
|      Segmentation Head      |
| 1x1 Conv3d -> num_classes   |
+-----------------------------+

StageBlock (S0/S1/S2/S3) internal flow:
Input feature (B, C_s, D_s, H_s, W_s)
   |
   v
[GSC]
   |
   v
[Flatten B,N,C -> Norm -> BiMamba-v3(nslices) -> Reshape]  # not tri-axis
   |
   v
[Residual Add]
   |
   v
[InstanceNorm -> MLP(C->2C->C) -> Residual Add]  # applied on outs path
   |
   v
Output feature (same stage shape)

Dataflow note (official):
  outs = vit(x_in);  enc1=encoder1(x_in);  enc2..4 from outs[0..2];  enc_hidden=encoder5(outs[3])
  then UnetrUpBlock fuse — no interpolate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

__all__ = [
    "SegMamba3D",
    "SegMamba3DConfig",
    "get_segmamba3d_config",
    "create_segmamba3d_from_config",
    "get_segmamba3d_from_plans",
]

# Prefer vendored SegMamba mamba fork (see mamba_import.py). Official SegMamba
# uses bimamba_type="v3" + nslices. Escape hatch: SEGMAMBA_FORCE_UNI=1 → uni Mamba.
import os

_FORCE_UNI = os.environ.get("SEGMAMBA_FORCE_UNI", "").lower() in ("1", "true", "t", "yes")

try:
    from nnunetv2.training.network.mamba_import import get_mamba_class

    Mamba = get_mamba_class(mode="uni")
    MAMBA_AVAILABLE = True
except ImportError as _mamba_exc:
    import warnings

    warnings.warn(
        f"SegMamba3D: mamba_ssm unavailable ({_mamba_exc}). "
        "Install matching mamba-ssm/causal-conv1d wheels, and ensure "
        "`import selective_scan_cuda` / `import causal_conv1d_cuda` work "
        "(needed by the SegMamba BiMamba-v3 fork).",
        stacklevel=1,
    )
    MAMBA_AVAILABLE = False
    Mamba = None

try:
    from monai.networks.blocks.dynunet_block import UnetOutBlock
    from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    UnetrBasicBlock = UnetrUpBlock = UnetOutBlock = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SegMamba3DConfig:
    """SegMamba3D configuration."""
    spatial_dims: int = 3
    in_channels: int = 1
    num_classes: int = 2
    depths: List[int] = None
    feat_size: List[int] = None
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6
    hidden_size: int = 768
    norm_name: str = "instance"
    conv_block: bool = True
    res_block: bool = True
    deep_supervision: bool = False

    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 2, 2]
        if self.feat_size is None:
            self.feat_size = [48, 96, 192, 384]


def get_segmamba3d_config(**kwargs) -> SegMamba3DConfig:
    """Get SegMamba3D configuration."""
    return SegMamba3DConfig(**kwargs)


def create_segmamba3d_from_config(config_name: str = "default", **kwargs) -> "SegMamba3D":
    """Create SegMamba3D from config name."""
    config = get_segmamba3d_config()
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return SegMamba3D.from_config(config)


# =============================================================================
# LayerNorm (channels_first for 3D)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None, None, None] * x + self.bias[:, None, None, None, None]
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


# =============================================================================
# MambaLayer (3D)
# =============================================================================

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required for SegMamba3D. Install: pip install mamba-ssm")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Official ge-xing/SegMamba: SegMamba fork + bimamba_type="v3" + nslices.
        from nnunetv2.training.network.mamba_import import get_mamba_class

        MambaCls = get_mamba_class(mode="uni")
        use_uni = _FORCE_UNI
        if not use_uni:
            import inspect

            if "bimamba_type" not in inspect.signature(MambaCls.__init__).parameters:
                import warnings

                warnings.warn(
                    "SegMamba3D: loaded mamba_ssm has no bimamba_type (site-packages "
                    "official build). Falling back to uni Mamba. For official BiMamba-v3, "
                    "sync nnunetv2/training/network/source_code/segmamba/upstream/ onto the server, or set "
                    "SEGMAMBA_FORCE_UNI=1 to silence this.",
                    UserWarning,
                    stacklevel=2,
                )
                use_uni = True
        if use_uni:
            self.mamba = MambaCls(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.mamba = MambaCls(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type="v3",
                nslices=num_slices,
            )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out + x_skip


# =============================================================================
# GSC Block
# =============================================================================

class GSC(nn.Module):
    """Global Semantic Context block."""

    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()
        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()
        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        x2 = self.nonliner3(self.norm3(self.proj3(x)))
        x = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
        return x + x_residual


# =============================================================================
# MlpChannel
# =============================================================================

class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# =============================================================================
# MambaEncoder
# =============================================================================

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=(2, 2, 2, 2), dims=(48, 96, 192, 384),
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        for i in range(4):
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            self.gscs.append(gsc)

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i in range(4):
            layer = nn.InstanceNorm3d(dims[i])
            self.add_module(f'norm{i}', layer)
            self.mlps.append(MlpChannel(dims[i], 2 * dims[i]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


# =============================================================================
# Main Model: SegMamba3D
# =============================================================================

class SegMamba3D(nn.Module):
    """
    SegMamba3D: 3D Medical Image Segmentation with Mamba.
    UNETR-style encoder-decoder with Mamba encoder.
    """

    def __init__(self,
                 in_chans: int = 1,
                 out_chans: int = 2,
                 depths: List[int] = None,
                 feat_size: List[int] = None,
                 drop_path_rate: float = 0.0,
                 layer_scale_init_value: float = 1e-6,
                 hidden_size: int = 768,
                 norm_name: str = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 spatial_dims: int = 3,
                 deep_supervision: bool = False,
                 # nnUNet compatibility
                 input_channels: int = None,
                 num_classes: int = None):
        super().__init__()
        if input_channels is not None:
            in_chans = input_channels
        if num_classes is not None:
            out_chans = num_classes
        if depths is None:
            depths = [2, 2, 2, 2]
        if feat_size is None:
            feat_size = [48, 96, 192, 384]

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.feat_size = feat_size
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision

        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        if MONAI_AVAILABLE:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans,
                out_channels=feat_size[0],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[0],
                out_channels=feat_size[1],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[1],
                out_channels=feat_size[2],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[2],
                out_channels=feat_size[3],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[3],
                out_channels=hidden_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feat_size[3],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[3],
                out_channels=feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[2],
                out_channels=feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[1],
                out_channels=feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[0],
                out_channels=feat_size[0],
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.out = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[0],
                out_channels=out_chans
            )
        else:
            raise ImportError("MONAI is required for SegMamba3D. Install: pip install monai")

    @classmethod
    def from_config(cls, config: SegMamba3DConfig) -> "SegMamba3D":
        return cls(
            in_chans=config.in_channels,
            out_chans=config.num_classes,
            depths=config.depths,
            feat_size=config.feat_size,
            drop_path_rate=config.drop_path_rate,
            layer_scale_init_value=config.layer_scale_init_value,
            hidden_size=config.hidden_size,
            norm_name=config.norm_name,
            conv_block=config.conv_block,
            res_block=config.res_block,
            spatial_dims=config.spatial_dims,
            deep_supervision=config.deep_supervision,
        )

    def forward(self, x_in):
        # Official ge-xing/SegMamba forward (no skip/output interpolate).
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        return self.out(out)


# =============================================================================
# nnUNet Plans Integration
# =============================================================================

def get_segmamba3d_from_plans(plans_manager, dataset_json, configuration_manager,
                              num_input_channels: int, deep_supervision: bool = True):
    """Create SegMamba3D from nnUNet plans."""
    if hasattr(plans_manager, 'get_label_manager'):
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
    else:
        num_classes = configuration_manager.num_classes
    return SegMamba3D(
        in_chans=num_input_channels,
        out_chans=num_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        hidden_size=768,
        spatial_dims=3,
        deep_supervision=deep_supervision,
    )


# =============================================================================
# Factory Functions
# =============================================================================

def create_segmamba3d_default(**kwargs):
    """Create SegMamba3D with default configuration."""
    return create_segmamba3d_from_config("default", **kwargs)
