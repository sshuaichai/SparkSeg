"""CoTr network definition (``UResTran3D``).

CNN backbone (:class:`CNNBackbone`) takes **only the last two** feature maps
(**L2, L3**) into a **multi-scale deformable transformer**
(:class:`DeformableTransformer`), then a **U-shaped decoder** with **additive**
skip fusion and optional deep supervision.

Source and citation:
  - Source: https://arxiv.org/abs/2103.03024
  - Date: 2021
  - Venue: MICCAI 2021
  - Paper: CoTr (arXiv:2103.03024)

Only **3D** is supported in this integration (``spatial_dims=3``).

----------------------------------------------------------------------
Architecture flow (boxed ASCII style as ``samu_net.py`` / ``UNet.py``;
backbone depth-9; MSDeform on **L2+L3** only; decoder fusion is ``x + skip``,
not concat.)
----------------------------------------------------------------------

Input ``(B, C, D, H, W)``
   |
   v
+-------------------------------+
| CNN Backbone (ResNet-9)       |
| E0: conv7 stem (64ch)         |---- skip0 -------------------------------+
| E1: layer1 ResBlk x3, 192ch   |---- skip1 --------------------------+    |
| E2: layer2 ResBlk x2, 384ch   |---- skip2 ---------------------+    |    |
| E3: layer3 deepest (384ch)    |---- L3->MSDeform L1 (in)-+     |    |    |
| MSDeform Transformer (6 enc)  |                   |    |     |    |    |
| L2+L3 levels, d_model 384     |                   |    |     |    |    |
+-------------------------------+                   |    |     |    |    |
   |                                                |    |     |    |    |
   v                                                |    |     |    |    |
+---------------------------------------------------|----|-----|----|----+
|       UResTran3D decoder (add skips)             |    |     |    |    |
| D0: TrConv + add(skip2)     <--------------------+    |     |    |    |
| D1: TrConv + add(skip1)     <-------------------------+     |    |    |
| D2: TrConv + add(skip0)     <-------------------------------+    |    |
+---------------------------------------------------+
   |
   v
+-------------------------------+
| Head: Up (1,2,2) + Conv 1x1   |
| deep_sup: ds0, ds1, ds2       |
+-------------------------------+
   |
   v
 logits: ``forward`` returns ``[fine, ds0, ds1, ds2]`` (fine is trilinear
 upsampled to patch size when stride leaves odd axes).

**Note**: ``skip0`` / ``skip1`` are CNN skips from the backbone; ``skip2`` is
the **transformer** branch aligned and added at L2. Multi-scale attention levels
are **two** (``posi_mask``: ``lvl > 1``). Deep supervision taps live at decoder
stages before the final upsample.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy
import math
import warnings
from functools import partial
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from scipy.ndimage.filters import gaussian_filter

__all__ = [
    "CoTr", 
    "get_cotr_config", 
    "create_cotr_from_config", 
    "CONFIGS_CoTr",
    "CoTrConfig"
]


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TransformerConfig:
    """Transformer configuration for CoTr"""
    d_model: int = 384
    nhead: int = 6
    num_encoder_layers: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    activation: str = "gelu"
    num_feature_levels: int = 2
    enc_n_points: int = 4


@dataclass
class CoTrConfig:
    """Complete CoTr configuration"""
    # Model architecture
    spatial_dims: int = 3
    in_channels: int = 1
    num_classes: int = 2
    patch_size: Union[Tuple[int, ...], List[int]] = (96, 96, 96)
    
    # CNN Backbone
    backbone_depth: int = 9
    norm_cfg: str = 'BN'
    activation_cfg: str = 'ReLU'
    weight_std: bool = False
    
    # Transformer parameters
    transformer: TransformerConfig = None
    
    # Training parameters
    deep_supervision: bool = True
    
    def __post_init__(self):
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)


def get_cotr_default_config():
    """Returns the default CoTr configuration."""
    return CoTrConfig(
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        patch_size=(96, 96, 96),
        backbone_depth=9,
        norm_cfg='BN',
        activation_cfg='ReLU',
        weight_std=False,
        transformer=TransformerConfig(
            d_model=384,
            nhead=6,
            num_encoder_layers=6,
            dim_feedforward=1536,
            dropout=0.1,
            activation='gelu',
            num_feature_levels=2,
            enc_n_points=4
        ),
        deep_supervision=True
    )


def get_cotr_light_config():
    """Returns a lighter CoTr configuration for smaller GPUs."""
    return CoTrConfig(
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        patch_size=(64, 64, 64),
        backbone_depth=9,
        norm_cfg='BN',
        activation_cfg='ReLU',
        weight_std=False,
        transformer=TransformerConfig(
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            num_feature_levels=2,
            enc_n_points=4
        ),
        deep_supervision=True
    )


def get_cotr_large_config():
    """Returns a larger CoTr configuration for better performance."""
    return CoTrConfig(
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        patch_size=(128, 128, 128),
        backbone_depth=9,
        norm_cfg='BN',
        activation_cfg='ReLU',
        weight_std=False,
        transformer=TransformerConfig(
            d_model=512,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            num_feature_levels=2,
            enc_n_points=4
        ),
        deep_supervision=True
    )


def get_cotr_testing_config():
    """Returns a minimal CoTr configuration for testing."""
    return CoTrConfig(
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        patch_size=(32, 32, 32),
        backbone_depth=9,
        norm_cfg='BN',
        activation_cfg='ReLU',
        weight_std=False,
        transformer=TransformerConfig(
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            num_feature_levels=2,
            enc_n_points=4
        ),
        deep_supervision=False
    )


# Configuration registry
CONFIGS_CoTr = {
    'CoTr-Default': get_cotr_default_config,
    'CoTr-Light': get_cotr_light_config,
    'CoTr-Large': get_cotr_large_config,
    'testing': get_cotr_testing_config,
}


def get_cotr_config(model_name: str) -> CoTrConfig:
    """
    Get CoTr configuration by model name.
    
    Args:
        model_name: Name of the model configuration
        
    Returns:
        CoTrConfig object
        
    Raises:
        ValueError: If model_name is not found
    """
    if model_name in CONFIGS_CoTr:
        return CONFIGS_CoTr[model_name]()
    
    available_models = list(CONFIGS_CoTr.keys())
    raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")


def create_cotr_from_config(config_name: str, **kwargs) -> 'CoTr':
    """
    Create CoTr model from configuration name.
    
    Args:
        config_name: Name of the configuration
        **kwargs: Additional arguments to override config values
        
    Returns:
        CoTr model instance
    """
    config = get_cotr_config(config_name)
    
    # Override config values with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.transformer, key):
            setattr(config.transformer, key, value)
    
    return CoTr.from_config(config)


# =============================================================================
# Utility Functions
# =============================================================================

def softmax_helper(x):
    return F.softmax(x, 1)


class InitWeights_He:
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


# =============================================================================
# 3D Deformable Attention Core Functions
# =============================================================================

def ms_deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Multi-scale deformable attention core function for 3D data.
    
    Args:
        value: (N, S, M, D) where S is the total number of spatial locations
        value_spatial_shapes: (L, 3) spatial shapes for each level
        sampling_locations: (N, Lq, M, L, P, 3) sampling locations
        attention_weights: (N, Lq, M, L, P) attention weights
        
    Returns:
        output: (N, Lq, M*D)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    for lid_, (T_, H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, T_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:, None, :, :, :]
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), 
            mode='bilinear', padding_mode='zeros', align_corners=False
        )[:, :, 0]
        sampling_value_list.append(sampling_value_l_)
    
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


# =============================================================================
# Multi-Scale Deformable Attention Module
# =============================================================================

class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module for 3D data.
    
    Args:
        d_model: hidden dimension
        n_levels: number of feature levels
        n_heads: number of attention heads
        n_points: number of sampling points per attention head per feature level
    """
    
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin() * thetas.cos(), thetas.sin() * thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 3
        ).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, 
                input_level_start_index, input_padding_mask=None):
        """
        Args:
            query: (N, Length_query, C)
            reference_points: (N, Length_query, n_levels, 3)
            input_flatten: (N, sum(D_l * H_l * W_l), C)
            input_spatial_shapes: (n_levels, 3)
            input_level_start_index: (n_levels,)
            input_padding_mask: (N, sum(D_l * H_l * W_l))
            
        Returns:
            output: (N, Length_query, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 3
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack([
                input_spatial_shapes[..., 0], 
                input_spatial_shapes[..., 2], 
                input_spatial_shapes[..., 1]
            ], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = ms_deform_attn_core_pytorch_3D(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


# =============================================================================
# Position Encoding
# =============================================================================

class PositionEmbeddingSine3D(nn.Module):
    """
    3D Sinusoidal Position Embedding.
    """
    
    def __init__(self, num_pos_feats=None, temperature=10000, normalize=True, scale=None):
        super().__init__()
        if num_pos_feats is None:
            num_pos_feats = [64, 64, 64]
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, d, h, w = x.shape
        mask = torch.zeros(bs, d, h, w, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_d = d_embed[:, :, :, :, None] / dim_td

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


def build_position_encoding(hidden_dim):
    """Build position encoding for the given hidden dimension."""
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]
    
    return PositionEmbeddingSine3D(num_pos_feats=N_steps, normalize=True)


# =============================================================================
# Deformable Transformer
# =============================================================================

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class DeformableTransformerEncoderLayer(nn.Module):
    """Single layer of the Deformable Transformer Encoder."""
    
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # Self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src, 
            spatial_shapes, level_start_index, padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transformer Encoder."""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            ref_d, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij'
            )

            ref_d = ref_d.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)

            ref = torch.stack((ref_d, ref_x, ref_y), -1)  # D W H
            reference_points_list.append(ref)
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformer(nn.Module):
    """
    Deformable Transformer for 3D feature processing.
    """
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, activation="relu",
                 num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        # Prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), 
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, 
            valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        return memory


# =============================================================================
# CNN Backbone with Weight Standardization
# =============================================================================

class Conv3d_wd(nn.Conv3d):
    """3D Convolution with Weight Standardization."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), 
                 padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(
            dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), 
              dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    """3x3x3 convolution with padding."""
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                         padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    """Create normalization layer based on config."""
    if norm_cfg == 'BN':
        return nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        return nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        return nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        return nn.InstanceNorm3d(inplanes, affine=True)
    else:
        raise ValueError(f"Unknown norm_cfg: {norm_cfg}")


def Activation_layer(activation_cfg, inplace=True):
    """Create activation layer based on config."""
    if activation_cfg == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)
    else:
        raise ValueError(f"Unknown activation_cfg: {activation_cfg}")


class Conv3dBlock(nn.Module):
    """Convolution + Normalization + Activation block."""
    
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size,
                 stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super().__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class BackboneResBlock(nn.Module):
    """Residual block for backbone."""
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), 
                 downsample=None, weight_std=False):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)
        return out


class CNNBackbone(nn.Module):
    """
    CNN Backbone for CoTr.
    
    Consists of an initial convolution followed by three residual blocks with downsampling.
    """
    
    arch_settings = {
        9: (BackboneResBlock, (3, 3, 2))
    }

    def __init__(self, depth=9, in_channels=1, norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        super().__init__()

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for backbone')
        
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        
        # Initial convolution
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), 
                               padding=3, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 192, layers[0], stride=(2, 2, 2), 
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 384, layers[1], stride=(2, 2, 2), 
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 384, layers[2], stride=(2, 2, 2), 
                                       norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', 
                    activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(self.inplanes, planes * block.expansion, kernel_size=1, 
                          stride=stride, bias=False, weight_std=weight_std),
                Norm_layer(norm_cfg, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, 
                           stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append(x)

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)

        return out


# =============================================================================
# Decoder Components
# =============================================================================

class DecoderResBlock(nn.Module):
    """Residual block for decoder."""
    
    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super().__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, 
                                    kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, 
                                    kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x
        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual
        return out


class UResTran3D(nn.Module):
    """
    U-shaped Residual Transformer for 3D segmentation.
    
    Combines CNN backbone with Deformable Transformer and U-Net style decoder.
    """
    
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', in_channels=1,
                 num_classes=2, weight_std=False, transformer_config=None):
        super().__init__()

        self.num_classes = num_classes
        
        if transformer_config is None:
            transformer_config = TransformerConfig()

        # Upsampling
        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        # Transpose convolutions
        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(384, 192, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(192, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        # Decoder residual blocks
        self.stage2_de = DecoderResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = DecoderResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = DecoderResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        # Deep supervision outputs
        self.ds2_cls_conv = nn.Conv3d(384, num_classes, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(192, num_classes, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(64, num_classes, kernel_size=1)

        # Final output
        self.cls_conv = nn.Conv3d(64, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

        # CNN Backbone
        self.backbone = CNNBackbone(depth=9, in_channels=in_channels, norm_cfg=norm_cfg, 
                                    activation_cfg=activation_cfg, weight_std=weight_std)
        
        # Position encoding
        self.position_embed = build_position_encoding(transformer_config.d_model)
        
        # Deformable Transformer
        self.encoder_Detrans = DeformableTransformer(
            d_model=transformer_config.d_model,
            nhead=transformer_config.nhead,
            num_encoder_layers=transformer_config.num_encoder_layers,
            dim_feedforward=transformer_config.dim_feedforward,
            dropout=transformer_config.dropout,
            activation=transformer_config.activation,
            num_feature_levels=transformer_config.num_feature_levels,
            enc_n_points=transformer_config.enc_n_points
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def posi_mask(self, x):
        """Generate position embeddings and masks for transformer input."""
        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:  # Only use the last two feature levels
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros(
                    (fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), 
                    dtype=torch.bool, device=fea.device
                ))
        return x_fea, masks, x_posemb

    @staticmethod
    def _align_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align skip connection spatial shape to x (handles odd-size stride mismatches)."""
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode='trilinear', align_corners=False
            )
        return skip

    def forward(self, inputs):
        input_spatial_shape = inputs.shape[2:]
        # CNN Backbone
        x_convs = self.backbone(inputs)
        
        # Prepare features for transformer
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        
        # Deformable Transformer
        x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)

        # Multi-scale transformer output processing
        # x_trans length: D0*H0*W0 + D1*H1*W1
        fea_0_size = x_fea[0].shape[-3] * x_fea[0].shape[-2] * x_fea[0].shape[-1]
        
        x = self.transposeconv_stage2(
            x_trans[:, fea_0_size:].transpose(-1, -2).view(x_convs[-1].shape)
        )
        skip2 = x_trans[:, :fea_0_size].transpose(-1, -2).view(x_convs[-2].shape)
        skip2 = self._align_skip(x, skip2)
        x = x + skip2
        x = self.stage2_de(x)
        ds2 = self.ds2_cls_conv(x)

        x = self.transposeconv_stage1(x)
        skip1 = self._align_skip(x, x_convs[-3])
        x = x + skip1
        x = self.stage1_de(x)
        ds1 = self.ds1_cls_conv(x)

        x = self.transposeconv_stage0(x)
        skip0 = self._align_skip(x, x_convs[-4])
        x = x + skip0
        x = self.stage0_de(x)
        ds0 = self.ds0_cls_conv(x)

        result = self.upsamplex2(x)
        result = self.cls_conv(result)

        # Keep logits aligned with label shape when patch size contains odd/non-divisible axes.
        if result.shape[2:] != input_spatial_shape:
            result = F.interpolate(
                result, size=input_spatial_shape, mode='trilinear', align_corners=False
            )

        return [result, ds0, ds1, ds2]


# =============================================================================
# Main CoTr Model
# =============================================================================

class CoTr(nn.Module):
    """
    CoTr: Efficient 3D Medical Image Segmentation by bridging CNN and Transformer
    
    Paper: https://arxiv.org/abs/2103.03024
    
    This model combines a CNN backbone with Deformable Transformer for efficient
    3D medical image segmentation.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels
        num_classes: Number of output classes
        patch_size: Input patch size (used for configuration)
        norm_cfg: Normalization type ('BN', 'SyncBN', 'GN', 'IN')
        activation_cfg: Activation type ('ReLU', 'LeakyReLU')
        weight_std: Whether to use weight standardization
        deep_supervision: Whether to use deep supervision
        transformer_d_model: Transformer hidden dimension
        transformer_nhead: Number of attention heads
        transformer_num_layers: Number of transformer layers
        transformer_dim_feedforward: FFN hidden dimension
        transformer_dropout: Dropout rate
        transformer_activation: Transformer activation ('relu', 'gelu')
        transformer_num_feature_levels: Number of feature levels for multi-scale
        transformer_enc_n_points: Number of sampling points per level
    """
    
    def __init__(self,
                 spatial_dims: int = 3,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 patch_size: Tuple[int, ...] = (96, 96, 96),
                 norm_cfg: str = 'BN',
                 activation_cfg: str = 'ReLU',
                 weight_std: bool = False,
                 deep_supervision: bool = True,
                 # Transformer parameters
                 transformer_d_model: int = 384,
                 transformer_nhead: int = 6,
                 transformer_num_layers: int = 6,
                 transformer_dim_feedforward: int = 1536,
                 transformer_dropout: float = 0.1,
                 transformer_activation: str = 'gelu',
                 transformer_num_feature_levels: int = 2,
                 transformer_enc_n_points: int = 4):
        
        super().__init__()
        
        if spatial_dims != 3:
            raise ValueError(f"CoTr currently only supports 3D (spatial_dims=3), got {spatial_dims}")
        
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        
        # Create transformer config
        transformer_config = TransformerConfig(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            num_feature_levels=transformer_num_feature_levels,
            enc_n_points=transformer_enc_n_points
        )
        
        # Main network
        self.network = UResTran3D(
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg,
            in_channels=in_channels,
            num_classes=num_classes,
            weight_std=weight_std,
            transformer_config=transformer_config
        )
        
        # Set conv_op and norm_op for compatibility
        if weight_std:
            self.conv_op = Conv3d_wd
        else:
            self.conv_op = nn.Conv3d
        
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        elif norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        elif norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        elif norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        
        self.dropout_op = nn.Dropout3d
        self._deep_supervision = deep_supervision

    @classmethod
    def from_config(cls, config: CoTrConfig) -> 'CoTr':
        """
        Create CoTr model from configuration object.
        
        Args:
            config: CoTrConfig object containing model parameters
            
        Returns:
            CoTr model instance
        """
        return cls(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            patch_size=config.patch_size,
            norm_cfg=config.norm_cfg,
            activation_cfg=config.activation_cfg,
            weight_std=config.weight_std,
            deep_supervision=config.deep_supervision,
            transformer_d_model=config.transformer.d_model,
            transformer_nhead=config.transformer.nhead,
            transformer_num_layers=config.transformer.num_encoder_layers,
            transformer_dim_feedforward=config.transformer.dim_feedforward,
            transformer_dropout=config.transformer.dropout,
            transformer_activation=config.transformer.activation,
            transformer_num_feature_levels=config.transformer.num_feature_levels,
            transformer_enc_n_points=config.transformer.enc_n_points
        )

    def forward(self, x):
        seg_output = self.network(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]
    
    def enable_deep_supervision(self, enabled: bool = True):
        """Enable or disable deep supervision."""
        self._deep_supervision = enabled
        self.do_ds = enabled


# =============================================================================
# Factory Functions for Easy Model Creation
# =============================================================================

def create_cotr_default(**kwargs) -> CoTr:
    """Create a CoTr model with default configuration."""
    return create_cotr_from_config('CoTr-Default', **kwargs)


def create_cotr_light(**kwargs) -> CoTr:
    """Create a lighter CoTr model for smaller GPUs."""
    return create_cotr_from_config('CoTr-Light', **kwargs)


def create_cotr_large(**kwargs) -> CoTr:
    """Create a larger CoTr model for better performance."""
    return create_cotr_from_config('CoTr-Large', **kwargs)


def create_cotr_testing(**kwargs) -> CoTr:
    """Create a minimal CoTr model for testing."""
    return create_cotr_from_config('testing', **kwargs)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test model creation
    print("Testing CoTr model...")
    
    # Test with testing config
    model = create_cotr_testing(in_channels=1, num_classes=14)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        model.do_ds = False
        out = model(x)
        print(f"Output shape (no deep supervision): {out.shape}")
        
        model.do_ds = True
        out = model(x)
        print(f"Output shapes (deep supervision): {[o.shape for o in out]}")
    
    print("Test passed!")
