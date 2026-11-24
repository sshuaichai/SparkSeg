"""SegMambaV2 network definition.

Upstream (verbatim tree, not this adapter): ``source_code/segmamba_v2/upstream`` — ge-xing/SegMamba-V2.

与本代码库中的SegMamba3D相比，SegMambaV2主要进行了三项实用性升级：
1. 每个Mamba层中的多轴Mamba融合：
    在原始布局和两个轴置换布局上运行Mamba，然后进行融合。
2. 阶段式计算重新分配：
    早期阶段保持强大的局部卷积建模，较深阶段使用Mamba。
3. 更强的局部上下文模块：
    GSC风格的局部模块被大核卷积（LargeKernelConv）+ 挤压激励（SE）优化所取代。

简而言之：V2保留了UNETR风格的解码，但将更多能力转移到“大核局部建模 + 选择性深度Mamba长距离建模”上。
严复现：forward 对齐官方 BraTS ``segmambav2.py``（``enc1→vit(enc1)``，无 skip/输出插值）。

Visual architecture diagram (default: depths=[2,2,2,2], feat_size=[48,96,192,384]):

Input (B, C, D, H, W)
   |
   v
+-----------------------------+
| encoder1: UnetrBasicBlock   |  in_chans -> C=48   ---- skip/enc1 --------+
+-----------------------------+                                            |
   |                                                                       |
   v  vit(enc1)  ← official: NOT raw x_in                                  |
+-----------------------------+                                            |
|      MambaEncoderV2         |                                            |
| ds0: Conv3 s2 (48->48)      |                                            |
| Stage0: LargeKernelConv x2  |---- outs0 -> encoder2 ------------------+  |
| ds1: Conv3 s2 (48->96)      |                                         |  |
| Stage1: LargeKernelConv x2  |---- outs1 -> encoder3 ---------------+  |  |
| ds2: Conv3 s2 (96->192)     |                                      |  |  |
| Stage2: LargeKernel + Mamba |---- outs2 -> encoder4 ------------+  |  |  |
| ds3: Conv3 s2 (192->384)    |                                   |  |  |  |
| Stage3: LargeKernel + Mamba |---- outs3 -> encoder5 (hidden)    |  |  |  |
+-----------------------------+                                   |  |  |  |
   |  MambaLayer = tri-axis uni-Mamba (sum 3 layouts + residual)  |  |  |  |
   v                                                              |  |  |  |
+---------------------------------------------------------------|-|--|--|--+
|          Decoder (UnetrUpBlock, no interpolate)               |  |  |  |
| decoder5(enc_hidden, enc4) <----------------------------------+  |  |  |
| decoder4(dec3, enc3)       <-------------------------------------+  |  |
| decoder3(dec2, enc2)       <----------------------------------------+  |
| decoder2(dec1, enc1)       <-------------------------------------------+
| decoder1 -> OutBlock -> classes
+-----------------------------------------------------------------------+

Paper/source: SegMamba-V2 (TMI); code under ``SegMamba-V2-main/.../segmambav2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange

__all__ = [
    "SegMambaV2",
    "SegMambaV2Config",
    "get_segmambav2_config",
    "create_segmambav2_from_config",
    "CONFIGS_SegMambaV2",
]


try:
    from nnunetv2.training.network.mamba_import import get_mamba_class

    get_mamba_class(mode="site")  # plain uni Mamba (not SegMamba BiMamba-v3 fork)
    MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # noqa: BLE001 — probe only
    MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = _e


try:
    from monai.networks.blocks.dynunet_block import UnetOutBlock
    from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
    MONAI_AVAILABLE = True
except Exception:
    MONAI_AVAILABLE = False
    UnetOutBlock = UnetrBasicBlock = UnetrUpBlock = None


@dataclass
class SegMambaV2Config:
    spatial_dims: int = 3
    in_channels: int = 1
    num_classes: int = 2
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    feat_size: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    hidden_size: int = 768
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6
    norm_name: str = "instance"
    conv_block: bool = True
    res_block: bool = True
    deep_supervision: bool = False


def _cfg_default() -> SegMambaV2Config:
    return SegMambaV2Config()


def _cfg_brats() -> SegMambaV2Config:
    return SegMambaV2Config(in_channels=4, num_classes=3)


def _cfg_testing() -> SegMambaV2Config:
    return SegMambaV2Config(
        in_channels=1,
        num_classes=2,
        depths=[1, 1, 1, 1],
        feat_size=[16, 32, 64, 128],
        hidden_size=128,
    )


CONFIGS_SegMambaV2: Dict[str, callable] = {
    "default": _cfg_default,
    "brats": _cfg_brats,
    "testing": _cfg_testing,
}


def get_segmambav2_config(config_name: str = "default") -> SegMambaV2Config:
    if config_name not in CONFIGS_SegMambaV2:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {list(CONFIGS_SegMambaV2.keys())}"
        )
    return CONFIGS_SegMambaV2[config_name]()


def create_segmambav2_from_config(config_name: str = "default", **kwargs) -> "SegMambaV2":
    config = get_segmambav2_config(config_name)
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return SegMambaV2.from_config(config)


class MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from nnunetv2.training.network.mamba_import import get_mamba_class

        try:
            MambaCls = get_mamba_class(mode="site")
        except Exception as exc:
            hint = ""
            if _MAMBA_IMPORT_ERROR is not None:
                hint = f" (module probe: {_MAMBA_IMPORT_ERROR!r})"
            raise ImportError(
                "mamba_ssm site-packages required for SegMambaV2 "
                f"(tri-axis uses uni Mamba, not SegMamba BiMamba fork).{hint}"
            ) from exc
        self.norm = nn.LayerNorm(dim)
        self.mamba = MambaCls(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(bsz, channels, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).reshape(bsz, channels, *img_dims)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        out_x_1 = self._mamba_forward(x)
        out_x_2 = self._mamba_forward(rearrange(x, "b c d w h -> b c w d h"))
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")
        out_x_3 = self._mamba_forward(rearrange(x, "b c d w h -> b c h w d"))
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")
        out = out_x_1 + out_x_2 + out_x_3
        return out + x_skip


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, _, _, _ = x.size()
        y = self.avg_pool(x).view(bsz, channels)
        y = self.fc(y).view(bsz, channels, 1, 1, 1)
        return x * y


class LargeKernelConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deep_path = nn.Sequential(
            ConvBlock(channels, channels, 5, padding=2),
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 3, padding=2, dilation=2),
        )
        self.shortcut_path = nn.Sequential(
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 1, padding=0),
        )
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.deep_path(x) + self.shortcut_path(x) + x
        return self.se(out)


class MambaEncoderV2(nn.Module):
    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        out_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if out_indices is None:
            out_indices = [0, 1, 2, 3]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1)))
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1))
            )

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        for i in range(4):
            self.gscs.append(nn.Sequential(*[LargeKernelConv(dims[i]) for _ in range(depths[i])]))
            if i < 2:
                self.stages.append(nn.Identity())
            else:
                self.stages.append(nn.Sequential(*[MambaLayer(dim=dims[i]) for _ in range(depths[i])]))

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            norm_layer = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f"norm{i_layer}", norm_layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward(self, x: torch.Tensor):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)


class SegMambaV2(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for SegMambaV2. Install: pip install monai")
        if depths is None:
            depths = [2, 2, 2, 2]
        if feat_size is None:
            feat_size = [48, 96, 192, 384]

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.feat_size = feat_size
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision

        self.vit = MambaEncoderV2(depths=depths, dims=feat_size)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

    @classmethod
    def from_config(cls, config: SegMambaV2Config) -> "SegMambaV2":
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

    def forward(self, x_in: torch.Tensor):
        # Official SegMamba-V2 BraTS forward (no skip/output interpolate).
        enc1 = self.encoder1(x_in)
        outs = self.vit(enc1)

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


def create_segmambav2_default(**kwargs) -> SegMambaV2:
    return create_segmambav2_from_config("default", **kwargs)


def create_segmambav2_brats(**kwargs) -> SegMambaV2:
    return create_segmambav2_from_config("brats", **kwargs)


def create_segmambav2_testing(**kwargs) -> SegMambaV2:
    return create_segmambav2_from_config("testing", **kwargs)
