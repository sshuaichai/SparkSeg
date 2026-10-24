"""Smoke tests for SparkSeg configuration, presets, and forward behavior.

Covers configuration consistency, per-dataset resolution, the Top-K budget,
train/eval forward behavior, and legacy checkpoint-key remapping.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from SparkSeg import (
    ANCHOR_K_SCHEDULE,
    DATASET_PRESETS,
    ENCODER_P,
    ENCODER_R,
    SparkSeg,
    build_sparkseg,
    describe_preset,
    num_tokens_for_stage,
    resolve_preset,
    verify_clean_config,
)

# Real 3d_fullres strides of the core 3D benchmarks (S6 hosts).
STRIDES_ISOTROPIC = [[1, 1, 1]] + [[2, 2, 2]] * 5
STRIDES_ACDC = [[1, 1, 1], [2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]


def test_k_schedule_is_the_paper_budget():
    """Eq. 4-6 with S=6, alpha=8, floor 32, cap 512 -> 128/64/32/32."""
    got = tuple(num_tokens_for_stage(i, deep_stage=4, deep_k=16) for i in (1, 2, 3, 4))
    assert got == (128, 64, 32, 32) == tuple(ANCHOR_K_SCHEDULE)
    # Table 7 sweep: deep_k = 8 / 16 / 32 -> 64/32/32/32 · 128/64/32/32 · 256/128/64/32
    sweep = {k: tuple(num_tokens_for_stage(i, deep_stage=4, deep_k=k) for i in (1, 2, 3, 4))
             for k in (8, 16, 32)}
    assert sweep[8] == (64, 32, 32, 32)
    assert sweep[32] == (256, 128, 64, 32)


def test_clean_config_matches_training_stack():
    report = verify_clean_config()
    if not report["checked"]:
        pytest.skip(f"training stack unavailable: {report['reason']}")
    assert report["mismatch"] == {}, report["mismatch"]


def test_presets_resolve_and_pick_reported_host():
    assert resolve_preset("acdc").host == ENCODER_R
    assert resolve_preset(180).host == ENCODER_R
    assert resolve_preset("Dataset1251_BraTS2023GLI").host == ENCODER_P
    assert resolve_preset("brats2021").dataset_id == 1251
    assert resolve_preset("Dataset489_KiTS23").host == ENCODER_R
    assert resolve_preset("kits").dataset_id == 489
    assert resolve_preset(None) is None
    with pytest.raises(KeyError):
        resolve_preset("not-a-dataset")
    # the init arm behind each reported number (Methods §2.5 provenance)
    assert DATASET_PRESETS["acdc"].init == "no_stock_he"
    assert DATASET_PRESETS["synapse"].init == "stock_he"
    assert DATASET_PRESETS["brats"].init == "no_stock_he"
    assert DATASET_PRESETS["kits23"].init == "stock_he"
    for key, p in DATASET_PRESETS.items():
        assert p.anchor_k_schedule == ANCHOR_K_SCHEDULE, key
        assert f"init={p.init}" in describe_preset(key)


def test_stock_he_follows_preset_and_can_be_overridden():
    acdc = build_sparkseg(2, 3, plan_strides=STRIDES_ACDC, preset="acdc")
    assert acdc.stock_he is False and acdc.model_config["stock_he"] is False
    assert build_sparkseg(2, 3, plan_strides=STRIDES_ACDC, preset="acdc",
                          stock_he=True).stock_he is True
    syn = build_sparkseg(2, 3, plan_strides=STRIDES_ISOTROPIC, preset="synapse")
    assert syn.stock_he is True
    assert build_sparkseg(2, 3, plan_strides=STRIDES_ISOTROPIC).stock_he is True


def test_stock_he_changes_the_initial_weights():
    def conv0_std(flag: bool) -> float:
        torch.manual_seed(0)
        net = build_sparkseg(2, 3, plan_strides=STRIDES_ISOTROPIC, stock_he=flag)
        conv = next(m for m in net.encoder.modules() if isinstance(m, torch.nn.Conv3d))
        return float(conv.weight.detach().std())

    he, no_he = conv0_std(True), conv0_std(False)
    assert he != no_he
    assert he > no_he  # He(1e-2) raises the initial conv scale


@pytest.mark.parametrize(
    "preset,strides,shape",
    [
        ("acdc", STRIDES_ACDC, (2, 32, 32, 8)),
        ("brats", STRIDES_ISOTROPIC, (2, 64, 64, 64)),
    ],
)
def test_build_and_forward(preset, strides, shape):
    torch.manual_seed(0)
    net = build_sparkseg(shape[0], 3, plan_strides=strides, preset=preset)
    assert isinstance(net, SparkSeg)
    assert net.model_config["anchor_k_schedule"] == (128, 64, 32, 32)
    assert net.model_config["dataset_id"] == resolve_preset(preset).dataset_id

    net.train()
    seg, prior = net(torch.randn(1, *shape))
    assert isinstance(seg, list) and len(seg) == 5  # deep supervision
    assert prior.shape[:2] == (1, 3)
    assert len(net._last_stage_p_effs) == 4  # one P_eff per guided stage

    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, *shape))
    assert isinstance(out, list) and out[0].shape[:2] == (1, 3)


def test_pre_manuscript_names_are_gone():
    """Only paper names are exported; the pre-manuscript identifiers were removed."""
    import SparkSeg as P
    import SparkSeg.network.blocks as B

    removed = (
        "SPARKUNet", "SPARKUnit", "SPARKDecoder", "AxialDW3D", "TokenAllocationNetwork",
        "LocalSelectiveBlockAttn", "WindowMHSA3D", "DecoderAttentionGate", "build_sparkunet",
        "_BlockAttnMHSA",
    )
    for name in removed:
        assert not hasattr(P, name), name
        assert not hasattr(B, name), name
        assert name not in P.__all__


def test_legacy_checkpoint_keys_are_remapped():
    """Checkpoints whose unit keys carry no depth index still load."""
    # NOTE: the package re-exports the class ``SparkSeg``, which shadows the submodule
    # attribute of the same name; import the module explicitly.
    M = importlib.import_module("SparkSeg.network.model")
    remapped = M.SparkSeg._remap_legacy_keys({
        "units.1.block.weight": 0,                # no depth index -> inserted
        "units.2.0.block.weight": 0,              # already indexed -> untouched
        "units.1.axial.dw_d.weight": 0,
        "units.2.tan.conv.0.weight": 0,
        "units.3.ba.block_attn.q_proj.weight": 0,
        "units.4.ba.anchor_gate.0.weight": 0,
        "units.2.gm_proj.weight": 0,
        "units.1.win_mhsa.qkv.weight": 0,
        "encoder.stem.convs.0.conv.weight": 0,    # untouched
    })
    assert sorted(remapped) == [
        "encoder.stem.convs.0.conv.weight",
        "units.1.0.block.weight",
        "units.1.0.tae.dw_d.weight",
        "units.1.0.win_mhsa3d.qkv.weight",
        "units.2.0.block.weight",
        "units.2.0.global_proj.weight",
        "units.2.0.pgts.conv.0.weight",
        "units.3.0.msba.attn.q_proj.weight",
        "units.4.0.msba.write_gate.0.weight",
    ]
