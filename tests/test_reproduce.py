"""Release packaging and reproduction metadata checks."""

from __future__ import annotations

from pathlib import Path

from SparkSeg.network.config import DATASET_PRESETS
from SparkSeg.release import (
    bundle_one,
    install_archive,
    model_folder,
    package_name,
    upload_files,
    verify_archive,
    verify_installed,
)


def test_presets_have_reproduction_metadata():
    for preset in DATASET_PRESETS.values():
        assert preset.raw_name.startswith(f"Dataset{preset.dataset_id}_")
        assert preset.fold in {"0", "all"}
        assert preset.trainer.startswith("nnUNetTrainerSparkSeg")


def test_readme_trainer_table_matches_presets():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    display = {
        "acdc": "ACDC",
        "brats": "BraTS2021",
        "synapse": "Synapse/BTCV",
        "kits23": "KiTS23",
    }
    host = {"resenc": "residual encoder", "plainconv": "plain U-Net"}
    init = {"stock_he": "`stock_he`", "no_stock_he": "不使用 `stock_he`"}
    for key, preset in DATASET_PRESETS.items():
        row = (
            f"| {preset.modality} | {display[key]} | {preset.dataset_id} | "
            f"`{preset.trainer}` | {host[preset.host]} | {init[preset.init]} | "
            f"{preset.epochs:,} |"
        )
        assert row in readme


def test_release_upload_list():
    assert upload_files() == [
        "sparkseg-acdc-mri-3d_fullres.zip",
        "sparkseg-brats2021-mri-3d_fullres.zip",
        "sparkseg-synapse-ct-3d_fullres.zip",
        "sparkseg-kits23-ct-3d_fullres.zip",
        "SHA256SUMS.txt",
    ]


def test_release_bundle_install_roundtrip(tmp_path):
    preset = DATASET_PRESETS["acdc"]
    results = tmp_path / "results"
    release_dir = tmp_path / "release"
    install_root = tmp_path / "installed"
    folder = model_folder(results, preset)
    (folder / "fold_0").mkdir(parents=True)
    (folder / "plans.json").write_text("{}", encoding="utf-8")
    (folder / "dataset.json").write_text("{}", encoding="utf-8")
    (folder / "splits_final.json").write_text("[]", encoding="utf-8")
    (folder / "fold_0" / "checkpoint_final.pth").write_bytes(b"checkpoint")

    archive = bundle_one("acdc", results, release_dir)
    assert archive.name == package_name(preset)
    manifest = verify_archive(archive)
    assert manifest["trainer"] == preset.trainer
    assert manifest["dataset_id"] == preset.dataset_id
    assert manifest["host"] == preset.host
    assert manifest["init"] == preset.init
    assert manifest["fold"] == preset.fold
    assert manifest["reported_mean_dsc"] == preset.reported["mean_dsc"]
    assert manifest["reported_mean_hd95"] == preset.reported["mean_hd95"]
    assert manifest["inference"]["use_mirroring_tta"] is False

    target = install_archive(archive, install_root)
    assert target == model_folder(install_root, preset)
    assert verify_installed(install_root, ["acdc"])[0]["dataset"] == "acdc"


def test_bundle_accepts_legacy_glr_result_folder(tmp_path):
    preset = DATASET_PRESETS["acdc"]
    dataset_root = tmp_path / "Dataset100_ACDC"
    legacy = (
        dataset_root
        / "nnUNetTrainerGLRResEncConv_NoStockHe__nnUNetPlans__3d_fullres"
    )
    (legacy / "fold_0").mkdir(parents=True)
    (legacy / "plans.json").write_text("{}", encoding="utf-8")
    (legacy / "dataset.json").write_text("{}", encoding="utf-8")
    (legacy / "fold_0" / "checkpoint_final.pth").write_bytes(b"legacy")

    archive = bundle_one(preset, tmp_path, tmp_path / "release")
    manifest = verify_archive(archive)
    assert manifest["source_model_folder"].startswith("nnUNetTrainerGLR")
