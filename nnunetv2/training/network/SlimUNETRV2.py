
from __future__ import annotations

"""SlimUNETRV2 network definition.

Upstream (verbatim tree, not this adapter): ``source_code/slim_unetrv2/upstream`` — deepang-ai/Slim-UNETRV2.

本代码库中SlimUNETRV2的主要优势：
1. 精简的计算配置：
    默认的阶段深度较浅（例如[1,1,1,1]），在保持多阶段表示的同时减少了内存/浮点运算量。
2. 混合远程+局部建模：
    每个阶段将SFM局部细化与SlimUNETRBlockV2相结合（Mamba序列建模+轻量级qkv交互）。
3. 面向3D切片流水线的解码器稳健性：
    转置卷积层在融合之前对齐跳跃连接的空间大小，减少了多步下采样/上采样导致的形状不匹配问题。
4. 实用的可配置性：
    提供slim_96/slim_128/测试预设以及明确的Mamba参数（d_state、d_conv、expand），以便快速进行面向部署的调优。 
 
Visual architecture diagram (default: depths=[1,1,1,1], dims=[48,96,192,384]):

Input (B, C, D, H, W)
   |
   v
+-----------------------------+
|         Encoder             |
| Stem: Conv3d patch embed    |
| Stage1: SFM x1 + SlimBlock x1 (C=48,  head=1) |
| Stage2: SFM x1 + SlimBlock x1 (C=96,  head=2) |
| Stage3: SFM x1 + SlimBlock x1 (C=192, head=4) |
| Stage4: SFM x1 + SlimBlock x1 (C=384, head=4) |
+-----------------------------+
   |
   v
+-----------------------------+
|      Global/Bottleneck      |
| hidden_downsample: 1 layer  |
+-----------------------------+
   |
   v
+-----------------------------+
|          Decoder            |
| tsconv1 -> fuse(skip4)      |
| tsconv2 -> fuse(skip3)      |
| tsconv3 -> fuse(skip2)      |
| tsconv4 -> fuse(skip1)      |
+-----------------------------+
   |
   v
+-----------------------------+
|      Segmentation Head      |
| ConvTranspose -> classes    |
+-----------------------------+

Paper/source citations are kept in the file header comments.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.utils import get_act_layer

__all__ = [
    "SlimUNETRV2",
    "SlimUNETRV2Config",
    "get_slimunetrv2_config",
    "create_slimunetrv2_from_config",
    "CONFIGS_SlimUNETRV2",
]


def _get_mamba_class():
    """Lazy import BiMamba-capable mamba_ssm (Slim Tri-SSA needs bimamba_type/nslices)."""
    from nnunetv2.training.network.mamba_import import get_mamba_class

    return get_mamba_class(mode="bi")


@dataclass
class SlimMambaConfig:
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class SlimUNETRV2Config:
    in_channels: int = 1
    num_classes: int = 2
    kernel_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_slices_list: List[int] = field(default_factory=lambda: [16, 8, 4, 2])
    depths: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    dims: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    hidden_size: int = 768
    out_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    mamba: SlimMambaConfig = field(default_factory=SlimMambaConfig)


def _cfg_96() -> SlimUNETRV2Config:
    return SlimUNETRV2Config(
        kernel_sizes=[2, 2, 2, 2],
        num_slices_list=[16, 8, 4, 2],
    )


def _cfg_128() -> SlimUNETRV2Config:
    return SlimUNETRV2Config(
        kernel_sizes=[4, 2, 2, 2],
        num_slices_list=[64, 32, 16, 8],
    )


def _cfg_testing() -> SlimUNETRV2Config:
    return SlimUNETRV2Config(
        in_channels=1,
        num_classes=2,
        kernel_sizes=[2, 2, 2, 2],
        num_slices_list=[8, 4, 2, 1],
        depths=[1, 1, 1, 1],
        dims=[16, 32, 64, 128],
        heads=[1, 1, 2, 4],
        hidden_size=256,
    )


CONFIGS_SlimUNETRV2: Dict[str, Callable[[], SlimUNETRV2Config]] = {
    "slim_96": _cfg_96,
    "slim_128": _cfg_128,
    "testing": _cfg_testing,
}


def get_slimunetrv2_config(config_name: str) -> SlimUNETRV2Config:
    if config_name not in CONFIGS_SlimUNETRV2:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {list(CONFIGS_SlimUNETRV2.keys())}"
        )
    return CONFIGS_SlimUNETRV2[config_name]()


def create_slimunetrv2_from_config(config_name: str, **kwargs) -> "SlimUNETRV2":
    cfg = get_slimunetrv2_config(config_name)
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif hasattr(cfg.mamba, k):
            setattr(cfg.mamba, k, v)
    return SlimUNETRV2.from_config(cfg)


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, shallow: bool = True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = get_act_layer(name="gelu" if shallow else "swish")
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SFM(nn.Module):
    def __init__(self, in_channels: int, shallow: bool = True) -> None:
        super().__init__()
        act_name = "gelu" if shallow else "swish"

        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = get_act_layer(name=act_name)

        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = get_act_layer(name=act_name)

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = get_act_layer(name=act_name)

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = get_act_layer(name=act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        return x + x_residual


class SlimUNETRBlockV2(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        head: int = 4,
        num_slices: int = 4,
        step: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.step = step
        self.num_heads = head
        self.head_dim = dim // head
        self.norm = nn.LayerNorm(dim)

        if self.step not in (2, 3):
            self.compression = nn.Conv3d(dim, dim, kernel_size=3, stride=4, padding=0)
            self.upsample = nn.ConvTranspose3d(
                dim, dim, kernel_size=3, stride=4, padding=0, output_padding=1
            )
        elif self.step == 2:
            self.compression = nn.Conv3d(dim, dim, kernel_size=1, stride=2, padding=0)
            self.upsample = nn.ConvTranspose3d(
                dim, dim, kernel_size=4, stride=2, padding=1, output_padding=0
            )

        mamba_kwargs = dict(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )
        Mamba = _get_mamba_class()
        try:
            self.mamba = Mamba(**mamba_kwargs)
        except TypeError as exc:
            # Standard pip mamba-ssm lacks bimamba_type/nslices (Tri-SSA). Do not
            # silently fall back to unidirectional Mamba — that would diverge from
            # the paper / official deepang-ai fork under environment/mamba.
            raise ImportError(
                "SlimUNETRV2 requires the vendored mamba_ssm fork that accepts "
                "bimamba_type='v3' and nslices (see deepang-ai/Slim-UNETRV2 "
                "environment/mamba). The installed mamba_ssm rejected those kwargs; "
                "install/build that fork instead of plain `pip install mamba-ssm`."
            ) from exc
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        if self.step != 3:
            x = self.compression(x)

        bsz, channels = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(bsz, channels, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(bsz, channels, *img_dims)

        bsz, channels, hgt, wdt, dep = x.shape
        q, k, v = (
            self.qkv(out)
            .view(bsz, self.num_heads, -1, hgt * wdt * dep)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        out = (v @ attn.transpose(-2, -1)).view(bsz, -1, hgt, wdt, dep)

        if self.step != 3:
            out = self.upsample(out)

        # stride-(2|4) conv / transposed conv pairs only round-trip for specific spatial sizes; sliding-window
        # tiles (Synapse etc.) often use arbitrary shapes → residual mismatch. Align before skip add.
        if out.shape[2:] != x_skip.shape[2:]:
            out = F.interpolate(
                out, size=x_skip.shape[2:], mode="trilinear", align_corners=False
            )

        return out + x_skip


class SlimMambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        kernel_sizes: Sequence[int] = (4, 2, 2, 2),
        depths: Sequence[int] = (1, 1, 1, 1),
        dims: Sequence[int] = (48, 96, 192, 384),
        num_slices_list: Sequence[int] = (64, 32, 16, 8),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        heads: Sequence[int] = (1, 2, 4, 4),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(
            in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=kernel_sizes[i + 1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.sfms = nn.ModuleList()
        for i in range(4):
            shallow = i <= 1
            self.sfms.append(SFM(dims[i], shallow))

            stage = nn.Sequential(
                *[
                    SlimUNETRBlockV2(
                        dim=dims[i],
                        num_slices=num_slices_list[i],
                        head=heads[i],
                        step=i,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(stage)

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i_layer in range(4):
            self.norms.append(nn.InstanceNorm3d(dims[i_layer]))
            self.mlps.append(
                MlpChannel(dims[i_layer], 2 * dims[i_layer], shallow=(i_layer < 2))
            )

    def forward(self, x: torch.Tensor):
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.sfms[i](x)
            feature_out.append(self.stages[i](x))
            if i in self.out_indices:
                x = self.norms[i](x)
                x = self.mlps[i](x)
        return x, feature_out


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, r: int):
        super().__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.transposed2 = nn.ConvTranspose3d(
            dim_out * 2, dim_out, kernel_size=1, stride=1
        )

    def forward(self, x: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        x = self.transposed1(x)
        if x.shape[2:] != feature.shape[2:]:
            feature = F.interpolate(
                feature, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        x = torch.cat((x, feature), dim=1)
        x = self.transposed2(x)
        x = self.norm(x)
        return x


class SlimUNETRV2(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        out_chans: int = 3,
        kernel_sizes: Sequence[int] = (2, 2, 2, 2),
        num_slices_list: Sequence[int] = (16, 8, 4, 2),
        depths: Sequence[int] = (1, 1, 1, 1),
        dims: Sequence[int] = (48, 96, 192, 384),
        heads: Sequence[int] = (1, 2, 4, 4),
        hidden_size: int = 768,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.encoder = SlimMambaEncoder(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            heads=heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.hidden_downsample = nn.Conv3d(dims[3], hidden_size, kernel_size=2, stride=2)

        self.tsconv1 = TransposedConvLayer(dim_in=hidden_size, dim_out=dims[3], r=2)
        self.tsconv2 = TransposedConvLayer(dim_in=dims[3], dim_out=dims[2], r=kernel_sizes[3])
        self.tsconv3 = TransposedConvLayer(dim_in=dims[2], dim_out=dims[1], r=kernel_sizes[2])
        self.tsconv4 = TransposedConvLayer(dim_in=dims[1], dim_out=dims[0], r=kernel_sizes[1])

        self.seghead = nn.ConvTranspose3d(
            dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )

    @classmethod
    def from_config(cls, config: SlimUNETRV2Config) -> "SlimUNETRV2":
        return cls(
            in_chans=config.in_channels,
            out_chans=config.num_classes,
            kernel_sizes=config.kernel_sizes,
            num_slices_list=config.num_slices_list,
            depths=config.depths,
            dims=config.dims,
            heads=config.heads,
            hidden_size=config.hidden_size,
            out_indices=config.out_indices,
            d_state=config.mamba.d_state,
            d_conv=config.mamba.d_conv,
            expand=config.mamba.expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs, feature_out = self.encoder(x)
        deep_feature = self.hidden_downsample(outs)
        x = self.tsconv1(deep_feature, feature_out[-1])
        x = self.tsconv2(x, feature_out[-2])
        x = self.tsconv3(x, feature_out[-3])
        x = self.tsconv4(x, feature_out[-4])
        x = self.seghead(x)
        return x


def create_slimunetrv2_96(**kwargs) -> SlimUNETRV2:
    return create_slimunetrv2_from_config("slim_96", **kwargs)


def create_slimunetrv2_128(**kwargs) -> SlimUNETRV2:
    return create_slimunetrv2_from_config("slim_128", **kwargs)


def create_slimunetrv2_testing(**kwargs) -> SlimUNETRV2:
    return create_slimunetrv2_from_config("testing", **kwargs)
