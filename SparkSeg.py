"""SparkSeg — sparse prior-guided anchor attention for 3D CT and MRI segmentation.

The network itself: it assembles the blocks from :mod:`SparkSeg.blocks` on the host
encoder/decoder, following the manuscript's configuration (Stage 1 TAE only,
Stages 2–5 one SparkSeg Block each, Stage 6 CNN bottleneck + PriorHead).  Helpers live
in :mod:`SparkSeg.utils`, the configuration in :mod:`SparkSeg.config`.

paper ↔ code: SparkSeg · SparkSeg Block · TAE (Eq. 1–2) · PriorHead (Eq. 3) ·
PGTS (Eq. 4–6) · MSBA · WinMHSA3D · AG (Fig. 1c).  Legacy aliases are kept.

Usage::

    from SparkSeg import build_sparkseg
    net = build_sparkseg(4, 4, plan_strides=strides, preset="brats")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .blocks import (
    AttentionGate, MSBA, PGTS, PriorHead, SparkSegBlock, SparkSegDecoder, TAE, WinMHSA3D,
)
from .config import (  # noqa: F401  (re-exported for convenience)
    ANCHOR_K_SCHEDULE, CLEAN_CONFIG, DATASET_PRESETS, ENCODER_L, ENCODER_P, ENCODER_R,
    PRESET_ALIASES, PRODUCT_FIXED, RESENC_BLOCKS_S6, TRAIN_DEFAULTS, DatasetPreset,
    clean_kwargs, describe_preset, preset_kwargs, resolve_preset, verify_clean_config,
)
from .utils import (
    _build_encoder, _default_resenc_blocks_per_stage, _get_stage_channels, _hist_ch_for_guide_stage, _stock_he_init, delta_energy_per_sample, encoder_strides_from_plan, num_tokens_for_stage, topk_coords_from_map,
)

#: Decoder layer kwargs (defined once; the decoder mirrors the encoder strides).
_DECODER_KWARGS: Dict[str, Any] = dict(
    nonlin_first=False, norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None, dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, conv_bias=True,
)

#: Module names inside a SparkSeg Block before the manuscript renaming; used only to
#: keep older checkpoints loadable (see ``SparkSeg.load_state_dict``).
_LEGACY_UNIT_MODULES: Dict[str, str] = {
    "axial": "tae",
    "tan": "pgts",
    "ba": "msba",
    "win_mhsa": "win_mhsa3d",
    "block_attn": "attn",
    "gm_proj": "global_proj",
    "anchor_gate": "write_gate",
}


class SparkSeg(nn.Module):
    """SparkSeg host — dense convolutional encoder–decoder + SparkSeg Blocks.

    Paper **SparkSeg** (Fig. 1a): Stage 1 is TAE-only, Stages 2–5 each host one SparkSeg
    Block, Stage 6 is the CNN bottleneck with PriorHead, and the decoder carries one
    attention gate per level.  ``preset=`` selects the reported host *and* the init arm
    behind that benchmark's number; the geometry still comes from the host plans/strides.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        plan_strides: Optional[Sequence[Sequence[int]]] = None,
        preset: Union[str, int, DatasetPreset, None] = None,
        stock_he: Optional[bool] = None,
        **over: Any,
    ) -> None:
        """Assemble the network from the building blocks above.

        Every default lives once, in ``config.CLEAN_CONFIG``; any config key may be
        overridden as a keyword.  ``plan_strides`` comes from the host plans
        (``configuration_manager.network_arch_init_kwargs["strides"]``) and ``preset``
        selects the manuscript's host/init arm for a benchmark.
        """
        super().__init__()
        self.preset = resolve_preset(preset)
        if self.preset is not None:  # manuscript Table 9: reported host per benchmark
            over.setdefault("encoder_type", self.preset.host)
        cfg = clean_kwargs(**over)
        unknown = set(cfg) - set(CLEAN_CONFIG) - set(PRODUCT_FIXED) - {"stock_he"}
        if unknown:
            raise TypeError(f"unknown SparkSeg config keys: {sorted(unknown)}")

        # None -> the arm that produced the preset's reported number (clean default True).
        self.stock_he = (
            bool(stock_he) if stock_he is not None
            else (True if self.preset is None else self.preset.init == "stock_he")
        )
        self.num_classes = int(out_channels)
        self.enable_deep_supervision = bool(cfg["enable_deep_supervision"])
        self.num_pool = int(cfg["num_pool"])
        n_stages = self.num_pool + 1
        self.encoder_type = str(cfg["encoder_type"]).lower().strip()
        features = _get_stage_channels(
            int(cfg["base_num_features"]), self.num_pool, int(cfg["max_num_features"])
        )
        self.stage_channels = list(features)
        if self.encoder_type in (ENCODER_R, ENCODER_L) and cfg["res_n_blocks"] is None:
            cfg["res_n_blocks"] = _default_resenc_blocks_per_stage(n_stages)

        self.guide_stages = tuple(
            int(i) for i in cfg["guide_stages"] if 0 <= int(i) < n_stages - 1
        )
        guide_set = set(self.guide_stages)
        self.local_only_stages = tuple(
            int(i) for i in cfg["local_only_stages"]
            if 0 <= int(i) < n_stages - 1 and int(i) not in guide_set
        )
        enhanced = guide_set | set(self.local_only_stages)
        anchor_k = {
            i: num_tokens_for_stage(
                i, n_stages,
                shallow_mult=float(cfg["anchor_shallow_mult"]),
                shallow_until=int(cfg["anchor_shallow_until"]),
                k_cap=int(cfg["anchor_k_cap"]),
                deep_k_max=int(cfg["anchor_deep_k_max"]),
                k_floor=int(cfg["anchor_k_floor"]),
                deep_stage=int(max(self.guide_stages)),
            )
            for i in self.guide_stages
        }

        self.encoder = _build_encoder(
            self.encoder_type,
            in_channels=int(in_channels),
            n_stages=n_stages,
            features=features,
            strides=list(encoder_strides_from_plan(plan_strides, self.num_pool)),
            conv_per_stage=int(cfg["conv_per_stage"]),
            res_n_blocks=cfg["res_n_blocks"],
            res_stem_channels=cfg["res_stem_channels"],
        )
        if self.stock_he:  # He touches conv stacks only; unit gates stay zero-init
            _stock_he_init(self.encoder)

        self.units = nn.ModuleDict()
        for i in self.guide_stages:
            self.units[str(i)] = nn.ModuleList([SparkSegBlock(
                int(features[i]), int(out_channels),
                axial_kernel=int(cfg["axial_kernel"]), anchor_k=anchor_k[i],
                anchor_heads=int(cfg["anchor_heads"]), anchor_drop=float(cfg["anchor_drop"]),
                enable_read=True, energy_fallback_lambda=float(cfg["energy_fallback_lambda"]),
                enable_window_mhsa=True, window_size=int(cfg["window_size"]),
                window_heads=int(cfg["window_heads"]),
                window_use_shift=bool(cfg["window_use_shift"]),
                hist_ch=_hist_ch_for_guide_stage(i, enhanced, features),
                topk_jitter=float(cfg["topk_jitter"]),
                gate_bias_init=float(cfg["gate_bias_init"]),
            )])
        for i in self.local_only_stages:
            self.units[str(i)] = nn.ModuleList([SparkSegBlock(
                int(features[i]), int(out_channels), axial_kernel=int(cfg["axial_kernel"]),
                enable_read=False, enable_window_mhsa=False,
                gate_bias_init=float(cfg["gate_bias_init"]),
            )])

        self.prior_head = PriorHead(int(features[-1]), int(out_channels))
        self.decoder = SparkSegDecoder(
            self.encoder, int(out_channels),
            [int(cfg["conv_per_stage_decoder"])] * (n_stages - 1),
            self.enable_deep_supervision, **_DECODER_KWARGS,
        )
        if self.stock_he:
            for name, child in self.decoder.decoder.named_children():
                if name != "encoder":
                    _stock_he_init(child)

        self.model_config: Dict[str, Any] = {
            **cfg,
            "encoder_type": self.encoder_type,
            "guide_stages": self.guide_stages,
            "local_only_stages": self.local_only_stages,
            "anchor_k_schedule": tuple(anchor_k[i] for i in self.guide_stages),
            "stock_he": bool(self.stock_he),
            "preset": self.preset.name if self.preset is not None else None,
            "dataset_id": int(self.preset.dataset_id) if self.preset is not None else None,
        }
        self._last_skips: Optional[List[torch.Tensor]] = None
        self._last_prior: Optional[torch.Tensor] = None
        self._last_stage_p_effs: List[torch.Tensor] = []

    @staticmethod
    def _remap_legacy_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Translate unit keys written before the manuscript naming.

        Two things changed: the block now sits at depth index ``0`` of its unit list
        (``units.<stage>.<name>`` → ``units.<stage>.0.<name>``), and the modules inside a
        block were renamed (``axial`` → ``tae``, ``tan`` → ``pgts``, ``ba`` → ``msba``,
        ``win_mhsa`` → ``win_mhsa3d``, ``block_attn`` → ``attn``, ``gm_proj`` →
        ``global_proj``, ``anchor_gate`` → ``write_gate``).
        """
        out: Dict[str, Any] = {}
        for k, v in state_dict.items():
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "units" and parts[1].isdigit():
                if not parts[2].isdigit():
                    parts.insert(2, "0")
                if len(parts) > 3:
                    parts[3] = _LEGACY_UNIT_MODULES.get(parts[3], parts[3])
                k = ".".join(parts)
            out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a checkpoint, tolerating pre-manuscript keys and obsolete bottleneck keys."""
        obsoleted = ("bottleneck_unit.", "bot_win_mhsa.", "bot_win_gate.")
        sd = {k: v for k, v in self._remap_legacy_keys(state_dict).items()
              if not k.startswith(obsoleted)}
        return super().load_state_dict(sd, strict=strict)

    def forward(self, x: torch.Tensor):
        feats: List[torch.Tensor] = []
        hist: Optional[torch.Tensor] = None
        stage_p_effs: List[torch.Tensor] = []
        z = x
        stem = getattr(self.encoder, "stem", None)
        if stem is not None:
            z = stem(z)
        for i, stage in enumerate(self.encoder.stages):
            z = stage(z)
            if str(i) in self.units:
                p_eff: Optional[torch.Tensor] = None
                for unit in self.units[str(i)]:
                    z, hist, p_eff = unit(z, hist_prev=hist)
                    if self.training and p_eff is not None and unit.enable_read:
                        stage_p_effs.append(p_eff)
            feats.append(z)

        if not self.training:
            seg = self.decoder(feats)
            self._last_skips = feats
            self._last_prior = None
            self._last_stage_p_effs = []
            return seg

        prior = self.prior_head(feats[-1])
        seg = self.decoder(feats)
        self._last_skips = feats
        self._last_prior = prior
        self._last_stage_p_effs = stage_p_effs
        return seg, prior


def build_sparkseg(
    in_channels: int,
    out_channels: int,
    *,
    plan_strides: Optional[Sequence[Sequence[int]]] = None,
    preset: Union[str, int, DatasetPreset, None] = None,
    stock_he: Optional[bool] = None,
    **kwargs: Any,
) -> SparkSeg:
    """Build the SparkSeg host on the default configuration.

    ``plan_strides`` = the host plan's strides
    (``configuration_manager.network_arch_init_kwargs["strides"]``); ``preset`` =
    ``"acdc"`` / ``"synapse"`` / ``"brats"`` (or dataset id / folder name).
    ``stock_he`` defaults to the preset's reported init arm (``True`` without a preset).
    """
    return SparkSeg(
        int(in_channels),
        int(out_channels),
        plan_strides=plan_strides,
        preset=preset,
        stock_he=stock_he,
        **kwargs,
    )


def build_sparkseg_for(
    dataset: Union[str, int, DatasetPreset],
    in_channels: int,
    out_channels: int,
    *,
    plan_strides: Optional[Sequence[Sequence[int]]] = None,
    **over: Any,
) -> SparkSeg:
    """Convenience: default configuration + the reported host/init of one benchmark."""
    return build_sparkseg(
        in_channels, out_channels, plan_strides=plan_strides, preset=dataset, **over
    )


__all__ = [
    # Network (manuscript names)
    "SparkSeg",
    "SparkSegBlock",
    "SparkSegDecoder",
    "TAE",
    "PriorHead",
    "PGTS",
    "MSBA",
    "WinMHSA3D",
    "AttentionGate",
    "build_sparkseg",
    "build_sparkseg_for",
    # Configuration
    "CLEAN_CONFIG",
    "PRODUCT_FIXED",
    "TRAIN_DEFAULTS",
    "RESENC_BLOCKS_S6",
    "ANCHOR_K_SCHEDULE",
    "DatasetPreset",
    "DATASET_PRESETS",
    "PRESET_ALIASES",
    "resolve_preset",
    "clean_kwargs",
    "preset_kwargs",
    "describe_preset",
    "verify_clean_config",
    "ENCODER_P",
    "ENCODER_L",
    "ENCODER_R",
    # Helpers
    "delta_energy_per_sample",
    "topk_coords_from_map",
    "num_tokens_for_stage",
    "_hist_ch_for_guide_stage",
]
