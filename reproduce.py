"""Reproduce SparkSeg training, prediction, evaluation, and release installation."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SparkSeg.network.config import DATASET_PRESETS, DatasetPreset, resolve_preset
    from SparkSeg.release import (
        DEFAULT_TAG,
        bundle_all,
        download_release,
        install_archive,
        upload_files,
        verify_installed,
        model_folder,
        source_model_folder,
    )
else:
    from .network.config import DATASET_PRESETS, DatasetPreset, resolve_preset
    from .release import (
        DEFAULT_TAG,
        bundle_all,
        download_release,
        install_archive,
        upload_files,
        verify_installed,
        model_folder,
        source_model_folder,
    )

PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"

EVAL_MODULE = "SparkSeg.nnUNetTrainerSparkSeg_eval"
_REGISTRATION_FILE: Path | None = None


def select_datasets(values: Sequence[str] | None) -> List[DatasetPreset]:
    if not values:
        return list(DATASET_PRESETS.values())
    selected: List[DatasetPreset] = []
    for value in values:
        preset = resolve_preset(value)
        if preset is None:
            raise ValueError("dataset cannot be None")
        if preset not in selected:
            selected.append(preset)
    return selected


def env_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return Path(value).expanduser().resolve()


def raw_dir(preset: DatasetPreset) -> Path:
    return env_root("nnUNet_raw") / preset.raw_name


def preprocessed_dir(preset: DatasetPreset) -> Path:
    return env_root("nnUNet_preprocessed") / preset.raw_name


def result_root(preset: DatasetPreset) -> Path:
    return env_root("nnUNet_results") / preset.raw_name


def trained_model_dir(preset: DatasetPreset) -> Path:
    return model_folder(env_root("nnUNet_results"), preset)


def default_prediction_dir(preset: DatasetPreset) -> Path:
    return raw_dir(preset) / "labelsTs_SparkSeg" / "final_pre"


def _console_script(name: str) -> List[str]:
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        exe_dir / name,
        exe_dir / f"{name}.exe",
        exe_dir / "Scripts" / name,
        exe_dir / "Scripts" / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    raise FileNotFoundError(f"Cannot find console script {name!r} next to {sys.executable}")


def _ensure_trainer_registration() -> Path:
    """Create a temporary nnU-Net trainer registration module."""
    global _REGISTRATION_FILE
    if _REGISTRATION_FILE is not None and _REGISTRATION_FILE.is_file():
        return _REGISTRATION_FILE
    import nnunetv2.training.nnUNetTrainer as trainer_package

    trainer_dir = Path(trainer_package.__file__).resolve().parent
    registration = trainer_dir / "_nnUNetTrainerSparkSeg.py"
    registration.write_text(
        "from SparkSeg.nnUNetTrainerSparkSeg import (\n"
        "    nnUNetTrainerSparkSegACDC,\n"
        "    nnUNetTrainerSparkSegBraTS2021,\n"
        "    nnUNetTrainerSparkSegKiTS23,\n"
        "    nnUNetTrainerSparkSegSynapse,\n"
        ")\n",
        encoding="utf-8",
    )
    _REGISTRATION_FILE = registration
    atexit.register(lambda: registration.unlink(missing_ok=True))
    return registration


def _subprocess_env() -> dict:
    env = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parent.parent)
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(
        [
            package_parent,
            *[
                p
                for p in existing
                if Path(p).resolve() != Path(package_parent).resolve()
            ],
        ]
    )
    _ensure_trainer_registration()
    return env


def run_command(command: Sequence[str], *, dry_run: bool = False) -> None:
    rendered = subprocess.list2cmdline(list(command))
    print(f"\n$ {rendered}")
    if dry_run:
        return
    subprocess.run(list(command), check=True, env=_subprocess_env())


def built_in_split(preset: DatasetPreset) -> Path:
    return Path(__file__).resolve().parent / "splits" / f"{preset.raw_name}.json"


def _copy_split(preset: DatasetPreset, destination: Path, force: bool) -> None:
    source = built_in_split(preset)
    if not source.is_file():
        return
    target = destination / "splits_final.json"
    if target.exists() and not force:
        if target.read_bytes() != source.read_bytes():
            raise FileExistsError(
                f"{target} differs from the bundled split; pass --force-splits to replace it"
            )
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare(datasets: Sequence[DatasetPreset], *, force_splits: bool, dry_run: bool) -> None:
    raw_root = env_root("nnUNet_raw")
    prep_root = env_root("nnUNet_preprocessed")
    _ = raw_root, prep_root
    command = _console_script("nnUNetv2_plan_and_preprocess")
    for preset in datasets:
        raw = raw_dir(preset)
        if not raw.is_dir():
            raise FileNotFoundError(f"Raw dataset folder not found: {raw}")
        if not dry_run:
            _copy_split(preset, raw, force_splits)
        run_command(
            [
                *command,
                "-d",
                str(preset.dataset_id),
                "-c",
                CONFIGURATION,
                "--verify_dataset_integrity",
            ],
            dry_run=dry_run,
        )
        if not dry_run:
            preprocessed_dir(preset).mkdir(parents=True, exist_ok=True)
            _copy_split(preset, preprocessed_dir(preset), force=True)


def train(
    datasets: Sequence[DatasetPreset],
    *,
    num_gpus: int,
    device: str,
    continue_training: bool,
    dry_run: bool,
) -> None:
    command = _console_script("nnUNetv2_train")
    for preset in datasets:
        if not preprocessed_dir(preset).is_dir():
            raise FileNotFoundError(f"Preprocessed dataset not found: {preprocessed_dir(preset)}")
        run_command(
            [
                *command,
                str(preset.dataset_id),
                CONFIGURATION,
                preset.fold,
                "-tr",
                preset.trainer,
                "-p",
                PLANS_IDENTIFIER,
                "-num_gpus",
                str(num_gpus),
                "-device",
                device,
                *(["--c"] if continue_training else []),
            ],
            dry_run=dry_run,
        )


def predict(
    datasets: Sequence[DatasetPreset],
    *,
    checkpoint: str,
    save_probabilities: bool,
    num_processes: int,
    device: str,
    overwrite: bool,
    use_tta: bool,
    dry_run: bool,
) -> None:
    command = _console_script("nnUNetv2_predict")
    for preset in datasets:
        raw = raw_dir(preset)
        input_dir = raw / "imagesTs"
        output_dir = default_prediction_dir(preset)
        model_dir = trained_model_dir(preset)
        checkpoint_path = model_dir / f"fold_{preset.fold}" / checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Test images not found: {input_dir}")
        if output_dir.exists() and any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Prediction folder is not empty: {output_dir}; pass --overwrite"
                )
            if not dry_run:
                shutil.rmtree(output_dir)
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                *command,
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-d",
                str(preset.dataset_id),
                "-p",
                PLANS_IDENTIFIER,
                "-tr",
                preset.trainer,
                "-c",
                CONFIGURATION,
                "-f",
                preset.fold,
                "-chk",
                checkpoint,
                *([] if use_tta else ["--disable_tta"]),
                "-npp",
                str(num_processes),
                "-nps",
                str(num_processes),
                "-device",
                device,
                *(["--save_probabilities"] if save_probabilities else []),
            ],
            dry_run=dry_run,
        )


def evaluate(
    datasets: Sequence[DatasetPreset],
    *,
    prediction_dirs: Sequence[Path] | None,
    force: bool,
    workers: int,
    dry_run: bool,
) -> None:
    explicit_dirs = [Path(p).resolve() for p in prediction_dirs or []]
    if explicit_dirs and len(datasets) != 1:
        raise ValueError("--pred-dirs can only be used with one dataset")

    for preset in datasets:
        pred_dirs = explicit_dirs or [default_prediction_dir(preset)]
        for pred_dir in pred_dirs:
            if not pred_dir.is_dir() and not dry_run:
                raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
        command = [
            sys.executable,
            "-m",
            EVAL_MODULE,
            "--dataset",
            preset.name,
            "--gt",
            str(raw_dir(preset) / "labelsTs"),
            "--pred-dirs",
            *(str(path) for path in pred_dirs),
            "--workers",
            str(workers),
        ]
        if force:
            command.append("--force")
        run_command(command, dry_run=dry_run)


def doctor(datasets: Sequence[DatasetPreset]) -> int:
    failures = 0
    print("Environment")
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        value = os.environ.get(name)
        exists = bool(value and Path(value).is_dir())
        print(f"  {name}: {value or '<unset>'} ({'ok' if exists else 'missing'})")
        failures += int(not exists)

    print("\nPython dependencies")
    for module in ("torch", "SimpleITK", "scipy", "numpy", "batchgeneratorsv2"):
        ok = importlib.util.find_spec(module) is not None
        print(f"  {module}: {'ok' if ok else 'missing'}")
        failures += int(not ok)

    try:
        registration = _ensure_trainer_registration()
        print(f"  trainer registration: ok ({registration.name})")
    except Exception as exc:
        print(f"  trainer registration: failed ({type(exc).__name__}: {exc})")
        failures += 1

    print("\nDatasets")
    for preset in datasets:
        raw = raw_dir(preset)
        prep = preprocessed_dir(preset)
        model = source_model_folder(env_root("nnUNet_results"), preset)
        checkpoint = model / f"fold_{preset.fold}" / "checkpoint_final.pth"
        label_cases = len(list((raw / "labelsTr").glob("*.nii.gz"))) if raw.is_dir() else 0
        test_cases = len(list((raw / "labelsTs").glob("*.nii.gz"))) if raw.is_dir() else 0
        print(
            f"  {preset.name:8s} modality={preset.modality:3s} id={preset.dataset_id:<4d} "
            f"raw={raw.is_dir()} train_labels={label_cases} test_labels={test_cases} "
            f"preprocessed={prep.is_dir()} checkpoint={checkpoint.is_file()}"
        )
        failures += int(not raw.is_dir())
    return 1 if failures else 0


def inspect_checkpoint(path: Path) -> dict:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return {
        "trainer_name": checkpoint.get("trainer_name"),
        "epoch": checkpoint.get("current_epoch"),
        "has_network_weights": "network_weights" in checkpoint,
    }


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="acdc, brats, synapse, kits23, ids, or nnU-Net folder names",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Check data paths, dependencies, and installed checkpoints")
    _add_dataset_args(p_doctor)

    p_prepare = sub.add_parser("prepare", help="Install fixed splits, plan, and preprocess datasets")
    _add_dataset_args(p_prepare)
    p_prepare.add_argument("--force-splits", action="store_true")
    p_prepare.add_argument("--dry-run", action="store_true")

    p_train = sub.add_parser("train", help="Train the selected SparkSeg presets")
    _add_dataset_args(p_train)
    p_train.add_argument("--num-gpus", type=int, default=1)
    p_train.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p_train.add_argument("--continue", dest="continue_training", action="store_true")
    p_train.add_argument("--dry-run", action="store_true")

    p_predict = sub.add_parser("predict", help="Run sliding-window inference on imagesTs")
    _add_dataset_args(p_predict)
    p_predict.add_argument("--checkpoint", default="checkpoint_final.pth")
    p_predict.add_argument("--save-probabilities", action="store_true")
    p_predict.add_argument("--num-processes", type=int, default=1)
    p_predict.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p_predict.add_argument(
        "--use-tta",
        dest="use_tta",
        action="store_true",
        help="Enable nnU-Net mirroring TTA; the paper uses no TTA",
    )
    p_predict.add_argument("--overwrite", action="store_true")
    p_predict.add_argument("--dry-run", action="store_true")

    p_eval = sub.add_parser("evaluate", help="Compute Dice and HD95 on held-out cases")
    _add_dataset_args(p_eval)
    p_eval.add_argument("--pred-dirs", nargs="+", default=None)
    p_eval.add_argument("--workers", type=int, default=1)
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument("--dry-run", action="store_true")

    p_validate = sub.add_parser("validate", help="Alias of evaluate using labelsTs")
    _add_dataset_args(p_validate)
    p_validate.add_argument("--pred-dirs", nargs="+", default=None)
    p_validate.add_argument("--workers", type=int, default=1)
    p_validate.add_argument("--force", action="store_true")
    p_validate.add_argument("--dry-run", action="store_true")

    p_download = sub.add_parser("download", help="Download all release archives")
    p_download.add_argument("--out", required=True)
    p_download.add_argument("--tag", default=DEFAULT_TAG)
    p_download.add_argument("--overwrite", action="store_true")

    p_install = sub.add_parser("install", help="Install release archives into nnUNet_results")
    p_install.add_argument("archives", nargs="+")
    p_install.add_argument("--force", action="store_true")

    p_bundle = sub.add_parser("bundle", help="Create release archives and SHA256SUMS.txt")
    _add_dataset_args(p_bundle)
    p_bundle.add_argument("--out", required=True)
    p_bundle.add_argument("--checkpoint", default="checkpoint_final.pth")
    p_bundle.add_argument("--include-best", action="store_true")

    p_verify = sub.add_parser("verify", help="Verify installed release archives")
    _add_dataset_args(p_verify)
    p_verify.add_argument("--load-checkpoints", action="store_true")

    p_upload = sub.add_parser("upload-list", help="Print the exact files to upload to GitHub Releases")
    p_upload.add_argument("--tag", default=DEFAULT_TAG)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "upload-list":
        print(f"Release tag: {args.tag}")
        for filename in upload_files():
            print(filename)
        return

    datasets = select_datasets(getattr(args, "datasets", None))
    if args.command == "doctor":
        raise SystemExit(doctor(datasets))
    if args.command == "prepare":
        prepare(datasets, force_splits=args.force_splits, dry_run=args.dry_run)
        return
    if args.command == "train":
        train(
            datasets,
            num_gpus=args.num_gpus,
            device=args.device,
            continue_training=args.continue_training,
            dry_run=args.dry_run,
        )
        return
    if args.command == "predict":
        predict(
            datasets,
            checkpoint=args.checkpoint,
            save_probabilities=args.save_probabilities,
            num_processes=args.num_processes,
            device=args.device,
            overwrite=args.overwrite,
            use_tta=args.use_tta,
            dry_run=args.dry_run,
        )
        return
    if args.command in {"evaluate", "validate"}:
        evaluate(
            datasets,
            prediction_dirs=args.pred_dirs,
            force=args.force,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        return
    if args.command == "download":
        for path in download_release(args.out, tag=args.tag, overwrite=args.overwrite):
            print(path)
        return
    if args.command == "install":
        for archive in args.archives:
            print(install_archive(archive, env_root("nnUNet_results"), force=args.force))
        return
    if args.command == "bundle":
        for archive in bundle_all(
            datasets,
            env_root("nnUNet_results"),
            args.out,
            checkpoint=args.checkpoint,
            include_best=args.include_best,
        ):
            print(archive)
        print(Path(args.out) / "SHA256SUMS.txt")
        return
    if args.command == "verify":
        reports = verify_installed(env_root("nnUNet_results"), datasets)
        for report in reports:
            print(report)
            if args.load_checkpoints:
                preset = resolve_preset(report["dataset"])
                assert preset is not None
                checkpoint = (
                    Path(report["folder"])
                    / f"fold_{preset.fold}"
                    / "checkpoint_final.pth"
                )
                print(inspect_checkpoint(checkpoint))
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
