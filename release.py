"""Bundle, install, and verify SparkSeg release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .network.config import DATASET_PRESETS, DatasetPreset, resolve_preset

PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
SCHEMA = "sparkseg.release.v1"
DEFAULT_TAG = "v1.3.0"
REPO_URL = "https://github.com/sshuaichai/SparkSeg"
LEGACY_MODEL_FOLDERS = {
    "acdc": (
        "nnUNetTrainerGLRResEncConv_NoStockHe",
        "nnUNetTrainerGLRResEncConv",
    ),
    "brats": (
        "nnUNetTrainerGLRPlainConv_NoStockHe",
        "nnUNetTrainerGLRPlainConv",
    ),
    "synapse": ("nnUNetTrainerGLRResEncConv_StockHe",),
    "kits23": ("nnUNetTrainerGLRResEncConv_StockHe",),
}


def package_name(preset: DatasetPreset) -> str:
    labels = {
        "acdc": "acdc-mri",
        "brats": "brats2021-mri",
        "synapse": "synapse-ct",
        "kits23": "kits23-ct",
    }
    return f"sparkseg-{labels[preset.name]}-3d_fullres.zip"


def upload_files() -> List[str]:
    return [package_name(p) for p in DATASET_PRESETS.values()] + ["SHA256SUMS.txt"]


def model_folder(results_root: Path, preset: DatasetPreset) -> Path:
    return (
        Path(results_root)
        / preset.raw_name
        / f"{preset.trainer}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
    )


def source_model_folder(results_root: Path, preset: DatasetPreset) -> Path:
    """Find current SparkSeg results or the legacy GLR result folder."""
    dataset_root = Path(results_root) / preset.raw_name
    current = model_folder(results_root, preset)
    candidates = [current]
    candidates.extend(
        dataset_root / f"{name}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
        for name in LEGACY_MODEL_FOLDERS.get(preset.name, ())
    )
    for candidate in candidates:
        checkpoint = candidate / f"fold_{preset.fold}" / "checkpoint_final.pth"
        if checkpoint.is_file():
            return candidate
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with zf.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_files(
    folder: Path,
    preset: DatasetPreset,
    checkpoint: str,
    include_best: bool,
) -> Dict[str, Path]:
    split_local = Path(__file__).resolve().parent / "splits" / f"{preset.raw_name}.json"
    split_path = folder / "splits_final.json"
    if not split_path.is_file():
        split_path = split_local

    required = {
        "plans.json": folder / "plans.json",
        "dataset.json": folder / "dataset.json",
        f"fold_{preset.fold}/{checkpoint}": folder / f"fold_{preset.fold}" / checkpoint,
    }
    if split_path.is_file():
        required["splits_final.json"] = split_path
    if include_best:
        required[f"fold_{preset.fold}/checkpoint_best.pth"] = (
            folder / f"fold_{preset.fold}" / "checkpoint_best.pth"
        )

    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing release files:\n" + "\n".join(missing))
    return required


def bundle_one(
    dataset: str | int | DatasetPreset,
    results_root: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str = "checkpoint_final.pth",
    include_best: bool = False,
) -> Path:
    preset = dataset if isinstance(dataset, DatasetPreset) else resolve_preset(dataset)
    assert preset is not None
    folder = source_model_folder(Path(results_root), preset)
    files = _bundle_files(folder, preset, checkpoint, include_best)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / package_name(preset)

    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": preset.name,
        "dataset_id": preset.dataset_id,
        "raw_name": preset.raw_name,
        "modality": preset.modality,
        "host": preset.host,
        "init": preset.init,
        "trainer": preset.trainer,
        "source_model_folder": folder.name,
        "plans_identifier": PLANS_IDENTIFIER,
        "configuration": CONFIGURATION,
        "fold": preset.fold,
        "checkpoint": checkpoint,
        "inference": {
            "tile_step_size": 0.5,
            "use_mirroring_tta": False,
            "gaussian": True,
        },
        "reported_mean_dsc": preset.reported.get("mean_dsc"),
        "reported_mean_hd95": preset.reported.get("mean_hd95"),
        "files": {
            rel: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for rel, path in files.items()
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for rel, source in files.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    compression = (
                        zipfile.ZIP_STORED
                        if path.suffix in {".pth", ".pt"}
                        else zipfile.ZIP_DEFLATED
                    )
                    zf.write(
                        path,
                        path.relative_to(staging).as_posix(),
                        compress_type=compression,
                    )
    return archive


def bundle_all(
    datasets: Iterable[str | int | DatasetPreset],
    results_root: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str = "checkpoint_final.pth",
    include_best: bool = False,
) -> List[Path]:
    archives = [
        bundle_one(
            dataset,
            results_root,
            output_dir,
            checkpoint=checkpoint,
            include_best=include_best,
        )
        for dataset in datasets
    ]
    checksum_path = Path(output_dir) / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in archives),
        encoding="utf-8",
    )
    return archives


def _read_manifest(archive: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        try:
            return json.loads(zf.read("manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise ValueError(f"{archive} does not contain manifest.json") from exc


def verify_archive(archive: str | Path) -> Dict[str, Any]:
    archive = Path(archive)
    manifest = _read_manifest(archive)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')}")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for rel, info in manifest["files"].items():
            if rel not in names:
                raise FileNotFoundError(f"{archive}: missing {rel}")
            digest = _sha256_zip_member(zf, rel)
            if digest != info["sha256"]:
                raise ValueError(f"{archive}: checksum mismatch for {rel}")
    return manifest


def _safe_extract(archive: Path, target: Path, expected: Iterable[str]) -> None:
    allowed = set(expected) | {"manifest.json"}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {name}")
            if name not in allowed:
                raise ValueError(f"Unexpected archive member: {name}")
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def install_archive(
    archive: str | Path,
    results_root: str | Path,
    *,
    force: bool = False,
) -> Path:
    archive = Path(archive)
    manifest = verify_archive(archive)
    preset = resolve_preset(manifest["dataset"])
    assert preset is not None
    if manifest["dataset_id"] != preset.dataset_id:
        raise ValueError(
            f"Dataset id mismatch for {preset.name}: "
            f"{manifest['dataset_id']} != {preset.dataset_id}"
        )

    target = model_folder(Path(results_root), preset)
    if target.exists():
        if not force:
            raise FileExistsError(f"Target already exists: {target}; pass --force to replace it")
        resolved_root = Path(results_root).resolve()
        if resolved_root not in target.resolve().parents:
            raise ValueError(f"Refusing to replace folder outside results root: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        _safe_extract(archive, staging, manifest["files"])
        for rel in manifest["files"]:
            source = staging / rel
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (target / "sparkseg_release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def verify_installed(
    results_root: str | Path,
    datasets: Iterable[str | int | DatasetPreset],
) -> List[Dict[str, Any]]:
    reports = []
    for dataset in datasets:
        preset = dataset if isinstance(dataset, DatasetPreset) else resolve_preset(dataset)
        assert preset is not None
        folder = model_folder(Path(results_root), preset)
        manifest_path = folder / "sparkseg_release_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing installed manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rel, info in manifest["files"].items():
            path = folder / rel
            if not path.is_file():
                raise FileNotFoundError(f"Missing installed file: {path}")
            if sha256_file(path) != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        reports.append({"dataset": preset.name, "folder": str(folder), "files": len(manifest["files"])})
    return reports


def download_release(
    download_dir: str | Path,
    *,
    tag: str = DEFAULT_TAG,
    overwrite: bool = False,
) -> List[Path]:
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in upload_files():
        target = download_dir / filename
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        url = f"{REPO_URL}/releases/download/{tag}/{filename}"
        with urllib.request.urlopen(url) as response, target.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        downloaded.append(target)
    checksum_file = download_dir / "SHA256SUMS.txt"
    expected = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename.strip()] = digest
    for filename, digest in expected.items():
        path = download_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checksum list references missing file: {path}")
        if sha256_file(path) != digest:
            raise ValueError(f"Checksum mismatch: {path}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Print the files to upload to a GitHub release")
    p_list.set_defaults(func=lambda _: print("\n".join(upload_files())))

    p_bundle = sub.add_parser("bundle", help="Create release archives from nnU-Net results")
    p_bundle.add_argument("--datasets", nargs="+", default=list(DATASET_PRESETS))
    p_bundle.add_argument("--results", required=True)
    p_bundle.add_argument("--out", required=True)
    p_bundle.add_argument("--checkpoint", default="checkpoint_final.pth")
    p_bundle.add_argument("--include-best", action="store_true")
    p_bundle.set_defaults(func=lambda a: [print(x) for x in bundle_all(
        a.datasets, a.results, a.out, checkpoint=a.checkpoint, include_best=a.include_best
    )])

    p_install = sub.add_parser("install", help="Install one or more release archives")
    p_install.add_argument("archives", nargs="+")
    p_install.add_argument("--results", required=True)
    p_install.add_argument("--force", action="store_true")
    p_install.set_defaults(func=lambda a: [print(install_archive(
        archive, a.results, force=a.force
    )) for archive in a.archives])

    p_verify = sub.add_parser("verify", help="Verify installed release files")
    p_verify.add_argument("--datasets", nargs="+", default=list(DATASET_PRESETS))
    p_verify.add_argument("--results", required=True)
    p_verify.set_defaults(func=lambda a: [print(row) for row in verify_installed(a.results, a.datasets)])

    p_download = sub.add_parser("download", help="Download release archives")
    p_download.add_argument("--out", required=True)
    p_download.add_argument("--tag", default=DEFAULT_TAG)
    p_download.add_argument("--overwrite", action="store_true")
    p_download.set_defaults(func=lambda a: [
        print(x) for x in download_release(a.out, tag=a.tag, overwrite=a.overwrite)
    ])

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
