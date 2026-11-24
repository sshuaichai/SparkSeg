"""UNETRPP network definition.

**UNETR++** (this implementation): **hybrid encoder** — per-scale **Conv** stem /
downsampling plus **Transformer** stages with **EPA** (Efficient Paired Attention);
**additive** skip fusion in :class:`UnetrUpBlock` (``out = up(x) + skip``), then
:class:`TransformerBlock` or :class:`UnetResBlock` in the decoder path; logits via
:class:`UnetOutBlock`
(optional **deep supervision** via ``do_ds``).

Based on: Shaker et al., *UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation*.

----------------------------------------------------------------------
Architecture flow (boxed ASCII style as ``samu_net.py`` / ``UNETR.py`` diagram;
``UnetrPPEncoder`` returns ``hidden_states[0..3]`` as ``enc1..enc4``; ``encoder1`` is the
parallel full-resolution CNN skip ``cnn``.)
----------------------------------------------------------------------

Input ``(B, C, D, H, W)`` (3D Conv3d stack in this file)
   |
   v
+-------------------------------+
|   UNETR++ hybrid encoder      |
| E0: CNN UnetRes (encoder1)    |---- cnn ---------------------------------+
| S0: stem stride4 + EPA stacks |---- enc1 ---------------------------+    |
| S1: down stride2 + EPA stacks |---- enc2 ----------------------+    |    |
| S2: down stride2 + EPA stacks |---- enc3 ----------------+     |    |    |
| S3: down stride2 + EPA stacks |---- enc4 -----------+    |     |    |    |
| (EPA paired-attn block)       |                   |    |     |    |    |
| E5: proj(enc4) bottleneck z4  |---- z4 -------------+    |     |    |    |    |
+-------------------------------+                   |    |     |    |    |
   |                                                |    |     |    |    |
   v                                                |    |     |    |    |
+---------------------------------------------------|----|-----|----|----+
|                  UNETRPP decoder                 |    |     |    |    |
| D0: UnetrUp + add(enc3)     <--------------------+    |     |    |    |
| D1: UnetrUp + add(enc2)     <-------------------------+     |    |    |
| D2: UnetrUp + add(enc1)     <-------------------------------+    |    |
| D3: UnetrUp + add(cnn)      <------------------------------------+    |
| D4: UnetOutBlock (conv)     <-----------------------------------------+
+---------------------------------------------------+
   |
   v
Output logits (``do_ds=True``: list of **3** heads at matched resolution; else one map)

**Note**: Decoder **adds** skips (not concat). ``D4`` maps to ``out1`` on the finest
``decoder2`` output; when ``do_ds``, heads ``out2(dec1)`` and ``out3(dec2)`` are
interpolated to the input shape together with ``out1(out)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Union, Sequence, Optional
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = ["UNETRPP", "UnetrPP"]


def init_weights(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class EPA(nn.Module):
    """
    Efficient Paired Attention Block
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.EF = nn.Parameter(init_weights(torch.zeros(input_size, proj_size)))

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        ef = self.EF
        if ef.shape[0] != N:
            # Dynamic token alignment for non-canonical patch sizes.
            ef = F.interpolate(
                ef.transpose(0, 1).unsqueeze(0),
                size=N,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)

        k_shared_projected = torch.einsum('bhdn,nk->bhdk', k_shared, ef)
        v_SA_projected = torch.einsum('bhdn,nk->bhdk', v_SA, ef)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA + x_SA


class UnetResBlock(nn.Module):
    """
    Residual block for UNet
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = self._get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size, stride, dropout
        )
        self.conv2 = self._get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size, 1, dropout
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = self._get_conv_layer(
                spatial_dims, in_channels, out_channels, 1, stride, dropout
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def _get_conv_layer(self, spatial_dims, in_channels, out_channels, kernel_size, stride, dropout):
        if spatial_dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                           kernel_size//2 if isinstance(kernel_size, int) else kernel_size[0]//2)
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                           kernel_size//2 if isinstance(kernel_size, int) else kernel_size[0]//2)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with EPA attention
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, 
                           num_heads=num_heads, channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            # Some anisotropic patch settings can make configured input_size very small.
            # Keep interpolation robust even if a legacy checkpoint has zero-length pos_embed.
            if self.pos_embed.shape[1] == 0:
                pos_embed = torch.zeros(
                    (1, x.shape[1], self.pos_embed.shape[2]),
                    dtype=x.dtype,
                    device=x.device,
                )
            if self.pos_embed.shape[1] != x.shape[1]:
                pos_embed = F.interpolate(
                    self.pos_embed.transpose(1, 2),
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2) if self.pos_embed.shape[1] > 0 else pos_embed
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class UnetrPPEncoder(nn.Module):
    """
    UNETR++ Encoder with ConvNet stems and Transformer blocks
    """
    def __init__(self, dims=[32, 64, 128, 256], proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], 
                 num_heads=4, spatial_dims=3, in_channels=4, dropout=0.0, transformer_dropout_rate=0.1, 
                 patch_size=None):
        super().__init__()

        # 根据patch_size动态计算input_size
        if patch_size is not None:
            # 计算每个stage的特征图大小
            # Stage 0: patch_size // 4 (stem layer stride=4)
            # Stage 1: previous_size // 2 (downsample stride=2)
            # Stage 2: previous_size // 2
            # Stage 3: previous_size // 2
            stage_sizes = []
            # Keep stage sizes >= 1 so Transformer positional embeddings never get zero tokens.
            current_size = [max(1, p // 4) for p in patch_size]  # stem layer
            stage_sizes.append(current_size[0] * current_size[1] * current_size[2])
            
            for i in range(3):  # 3个downsample layers
                current_size = [max(1, s // 2) for s in current_size]
                stage_sizes.append(current_size[0] * current_size[1] * current_size[2])
            
            input_size = stage_sizes
        else:
            # 默认input_size，基于128x128x128的patch_size
            input_size = [32*32*32, 16*16*16, 8*8*8, 4*4*4]

        self.downsample_layers = nn.ModuleList()
        
        # Stem layer
        stem_layer = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=in_channels, num_channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        
        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.GroupNorm(num_groups=dims[i], num_channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        # Transformer stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(
                    input_size=input_size[i], 
                    hidden_size=dims[i],
                    proj_size=proj_size[i], 
                    num_heads=num_heads,
                    dropout_rate=transformer_dropout_rate, 
                    pos_embed=True
                ))
            self.stages.append(nn.Sequential(*stage_blocks))

    def forward(self, x):
        hidden_states = []

        for i in range(4):
            # Protect thin axes (for example ACDC z-dim) from stride-conv collapse.
            # For stage 0 we need at least 4 voxels (kernel=4), later stages need 2 (kernel=2).
            min_size = 4 if i == 0 else 2
            d, h, w = x.shape[2:]
            pad_d = max(0, min_size - d)
            pad_h = max(0, min_size - h)
            pad_w = max(0, min_size - w)
            if pad_d or pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                B, C, H, W, D = x.shape
                x = x.reshape(B, H * W * D, C)
            hidden_states.append(x)
        
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    """
    Upsampling block for UNETR++
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
    ):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.transp_conv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size
        )

        # Decoder blocks
        self.decoder_block = nn.ModuleList()

        if conv_decoder:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, 
                           kernel_size=kernel_size, stride=1, norm_name=norm_name)
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(
                    input_size=out_size, 
                    hidden_size=out_channels,
                    proj_size=proj_size, 
                    num_heads=num_heads,
                    dropout_rate=0.1, 
                    pos_embed=True
                ))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        # For odd patch sizes, repeated stride-2 down/up sampling can create
        # one-voxel shape gaps between decoder feature and skip feature.
        if out.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip,
                size=out.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        out = out + skip
        out = self.decoder_block[0](out)
        return out


class UnetOutBlock(nn.Module):
    """
    Output block for segmentation
    """
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, inp):
        return self.conv(inp)


class UNETRPP(nn.Module):
    """
    UNETR++ network for 3D medical image segmentation
    
    Based on: "Shaker et al., UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds=True,
        patch_size=None,  # 添加patch_size参数
    ):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels  
            feature_size: base feature size
            hidden_size: hidden size for transformer
            num_heads: number of attention heads
            pos_embed: position embedding type
            norm_name: normalization type
            dropout_rate: dropout rate
            depths: transformer block depths for each stage
            dims: feature dimensions for each stage
            conv_op: convolution operation type
            do_ds: whether to use deep supervision
            patch_size: input patch size for dynamic sizing
        """
        super().__init__()
        
        if depths is None:
            depths = [3, 3, 3, 3]
        if dims is None:
            dims = [feature_size, feature_size*2, feature_size*4, feature_size*8]
            
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # 动态计算feat_size，基于patch_size
        if patch_size is not None:
            # 计算最后一个stage的特征图大小：patch_size // (4 * 2^3) = patch_size // 32
            self.feat_size = tuple(max(1, p // 32) for p in patch_size)
            
            # 动态计算每个解码器stage的特征图大小
            # decoder5: patch_size // 16 (after 3 downsamples from patch_size // 4)
            # decoder4: patch_size // 8  (after 2 downsamples from patch_size // 4)  
            # decoder3: patch_size // 4  (after 1 downsample from patch_size // 4)
            # decoder2: patch_size // 1  (original size)
            self.decoder_sizes = {
                'decoder5': tuple(max(1, p // 16) for p in patch_size),
                'decoder4': tuple(max(1, p // 8) for p in patch_size),
                'decoder3': tuple(max(1, p // 4) for p in patch_size),
                'decoder2': tuple(p for p in patch_size),
            }
        else:
            self.feat_size = (4, 4, 4)  # 默认值，适用于128x128x128的输入
            self.decoder_sizes = {
                'decoder5': (8, 8, 8),
                'decoder4': (16, 16, 16),
                'decoder3': (32, 32, 32),
                'decoder2': (128, 128, 128),
            }
            
        self.hidden_size = hidden_size

        # Encoder - 传递patch_size参数
        self.unetr_pp_encoder = UnetrPPEncoder(
            dims=dims, depths=depths, num_heads=num_heads, in_channels=in_channels,
            patch_size=patch_size  # 传递patch_size
        )

        # Initial encoder
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        # 修正解码器通道数配置，确保与编码器输出匹配
        # dims = [feature_size, feature_size*2, feature_size*4, feature_size*8]
        # 对应     [16,         32,            64,            128]
        
        # Decoders - 修正通道数配置和out_size
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[3],  # 使用dims[3]而不是feature_size * 16
            out_channels=dims[2], # 使用dims[2]而不是feature_size * 8
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,  
            out_size=self.decoder_sizes['decoder5'][0] * self.decoder_sizes['decoder5'][1] * self.decoder_sizes['decoder5'][2],
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[2],  # 使用dims[2]而不是feature_size * 8
            out_channels=dims[1], # 使用dims[1]而不是feature_size * 4
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.decoder_sizes['decoder4'][0] * self.decoder_sizes['decoder4'][1] * self.decoder_sizes['decoder4'][2],
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[1],  # 使用dims[1]而不是feature_size * 4
            out_channels=dims[0], # 使用dims[0]而不是feature_size * 2
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.decoder_sizes['decoder3'][0] * self.decoder_sizes['decoder3'][1] * self.decoder_sizes['decoder3'][2],
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=dims[0],  # 使用dims[0]而不是feature_size * 2
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=self.decoder_sizes['decoder2'][0] * self.decoder_sizes['decoder2'][1] * self.decoder_sizes['decoder2'][2],
            conv_decoder=True,
        )

        # Output layers - 修正输入通道数
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)  # 使用dims[0]
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=dims[1], out_channels=out_channels)  # 使用dims[1]

    def proj_feat(self, x, hidden_size, feat_size):
        """Project features for decoder - 动态计算特征大小"""
        # x的形状应该是 (B, N, C)，其中N是空间维度的乘积
        B, N, C = x.shape
        
        # 计算实际的空间维度
        total_spatial_elements = N
        
        # 尝试重塑为3D特征图
        # 假设特征图是立方体或接近立方体的形状
        spatial_dim = int(round(total_spatial_elements ** (1/3)))
        
        # 如果不是完美的立方，需要计算实际的维度
        if spatial_dim ** 3 != total_spatial_elements:
            # 计算总的比例因子
            base_h = max(1, int(feat_size[0]))
            base_w = max(1, int(feat_size[1]))
            base_d = max(1, int(feat_size[2]))
            base_volume = max(1, base_h * base_w * base_d)
            scale_factor = (total_spatial_elements / base_volume) ** (1/3)
            
            actual_h = max(1, int(round(base_h * scale_factor)))
            actual_w = max(1, int(round(base_w * scale_factor)))
            actual_d = max(1, int(round(base_d * scale_factor)))
            
            # 确保乘积等于total_spatial_elements
            while actual_h * actual_w * actual_d != total_spatial_elements:
                if actual_h * actual_w * actual_d < total_spatial_elements:
                    actual_d += 1
                else:
                    actual_d -= 1
                    if actual_d < 1:
                        actual_d = 1
                        actual_w -= 1
                        if actual_w < 1:
                            actual_w = 1
                            actual_h -= 1
                
            actual_feat_size = (actual_h, actual_w, actual_d)
        else:
            actual_feat_size = (spatial_dim, spatial_dim, spatial_dim)
        
        # 重塑张量
        x = x.view(B, actual_feat_size[0], actual_feat_size[1], actual_feat_size[2], C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        """Forward pass"""
        input_shape = x_in.shape[2:]
        # Encoder
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # Extract encoder features
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Decoder pathway - 使用动态的proj_feat
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, convBlock)

        # Output
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            logits = [
                F.interpolate(l, size=input_shape, mode="trilinear", align_corners=False)
                if l.shape[2:] != input_shape else l
                for l in logits
            ]
        else:
            logits = self.out1(out)
            if logits.shape[2:] != input_shape:
                logits = F.interpolate(
                    logits, size=input_shape, mode="trilinear", align_corners=False
                )

        return logits


# Alias for backward compatibility
UnetrPP = UNETRPP
