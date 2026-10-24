"""Self-contained Dice and HD95 evaluation for the four SparkSeg benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required for SparkSeg evaluation") from exc

from .network.config import resolve_preset


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    labels: Tuple[int, ...]


@dataclass(frozen=True)
class EvaluationSpec:
    key: str
    metric_prefix: str
    regions: Tuple[Region, ...]
    aggregate: str  # "case_macro" or "region_macro"


SPECS: Dict[str, EvaluationSpec] = {
    "acdc": EvaluationSpec(
        "acdc",
        "metrics_ACDC_testset",
        (
            Region("1", "RV", (1,)),
            Region("2", "MYO", (2,)),
            Region("3", "LV", (3,)),
        ),
        "case_macro",
    ),
    "brats": EvaluationSpec(
        "brats",
        "metrics_BraTS_WT_TC_ET_testset",
        (
            Region("WT", "WT", (1, 2, 3)),
            Region("TC", "TC", (1, 3)),
            Region("ET", "ET", (3,)),
        ),
        "region_macro",
    ),
    "synapse": EvaluationSpec(
        "synapse",
        "metrics_synapse_testset",
        (
            Region("1", "aorta", (1,)),
            Region("2", "gallbladder", (2,)),
            Region("3", "left_kidney", (3,)),
            Region("4", "right_kidney", (4,)),
            Region("5", "liver", (5,)),
            Region("6", "pancreas", (6,)),
            Region("7", "spleen", (7,)),
            Region("8", "stomach", (8,)),
        ),
        "case_macro",
    ),
    "kits23": EvaluationSpec(
        "kits23",
        "metrics_KiTS23_HEC_testset",
        (
            Region("Kidney and Masses", "Kidney and Masses", (1, 2, 3)),
            Region("Kidney Mass", "Kidney Mass", (2, 3)),
            Region("Tumor", "Tumor", (2,)),
        ),
        "region_macro",
    ),
}


def _np_zyx(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def _spacing_zyx(image: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = image.GetSpacing()
    return float(sz), float(sy), float(sx)


def _resample_prediction(prediction: sitk.Image, reference: sitk.Image) -> np.ndarray:
    if (
        prediction.GetSize() == reference.GetSize()
        and prediction.GetSpacing() == reference.GetSpacing()
        and prediction.GetOrigin() == reference.GetOrigin()
        and prediction.GetDirection() == reference.GetDirection()
    ):
        return _np_zyx(prediction)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return _np_zyx(resampler.Execute(prediction))


def _dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred_sum = int(np.count_nonzero(prediction))
    ref_sum = int(np.count_nonzero(reference))
    if pred_sum == 0 and ref_sum == 0:
        return 1.0
    if pred_sum == 0 or ref_sum == 0:
        return 0.0
    intersection = int(np.count_nonzero(prediction & reference))
    return 2.0 * intersection / (pred_sum + ref_sum)


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask)
    if not np.any(eroded):
        return mask
    return mask & ~eroded


def _crop_union(
    prediction: np.ndarray, reference: np.ndarray, margin: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    union = prediction | reference
    coordinates = np.nonzero(union)
    if not coordinates:
        return prediction, reference
    slices = tuple(
        slice(max(0, int(axis.min()) - margin), int(axis.max()) + margin + 1)
        for axis in coordinates
    )
    return prediction[slices], reference[slices]


def _hd95(
    prediction: np.ndarray, reference: np.ndarray, spacing: Sequence[float]
) -> Optional[float]:
    if not np.any(prediction) and not np.any(reference):
        return 0.0
    if not np.any(prediction) or not np.any(reference):
        return None
    prediction, reference = _crop_union(prediction, reference)
    pred_surface = _surface(prediction)
    ref_surface = _surface(reference)
    if not np.any(pred_surface) or not np.any(ref_surface):
        return None
    to_reference = distance_transform_edt(
        ~ref_surface, sampling=tuple(spacing)
    )[pred_surface]
    to_prediction = distance_transform_edt(
        ~pred_surface, sampling=tuple(spacing)
    )[ref_surface]
    distances = np.concatenate([to_reference, to_prediction])
    return float(np.percentile(distances, 95)) if distances.size else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(valid)) if valid else None


def _evaluate_case(
    gt_path: Path,
    pred_path: Path,
    regions: Tuple[Region, ...],
) -> Dict[str, Any]:
    gt_image = sitk.ReadImage(str(gt_path))
    pred_image = sitk.ReadImage(str(pred_path))
    gt = _np_zyx(gt_image)
    pred = _resample_prediction(pred_image, gt_image)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch for {gt_path.name}: {gt.shape} != {pred.shape}")
    spacing = _spacing_zyx(gt_image)

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    dice_values: List[Optional[float]] = []
    hd_values: List[Optional[float]] = []
    for region in regions:
        pred_mask = np.isin(pred, region.labels)
        gt_mask = np.isin(gt, region.labels)
        dice = 100.0 * _dice(pred_mask, gt_mask)
        hd = _hd95(pred_mask, gt_mask, spacing)
        metrics[region.key] = {
            "name": region.name,
            "Dice": round(dice, 4),
            "HD95": round(hd, 4) if hd is not None else None,
        }
        dice_values.append(dice)
        hd_values.append(hd)
    return {
        "case": gt_path.name,
        "metrics": metrics,
        "macro_dice": _mean(dice_values),
        "macro_hd95": _mean(hd_values),
    }


def evaluate_predictions(
    dataset: str,
    gt_dir: str | Path,
    pred_dir: str | Path,
    *,
    workers: int = 1,
) -> Dict[str, Any]:
    preset = resolve_preset(dataset)
    if preset is None or preset.name not in SPECS:
        raise KeyError(f"Unknown evaluation dataset: {dataset}")
    spec = SPECS[preset.name]
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise FileNotFoundError(f"No NIfTI labels found in {gt_dir}")
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

    if workers <= 1:
        cases = [
            _evaluate_case(path, pred_dir / path.name, spec.regions)
            for path in gt_files
        ]
    else:
        cases_by_name = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_case,
                    path,
                    pred_dir / path.name,
                    spec.regions,
                ): path.name
                for path in gt_files
            }
            for future in as_completed(futures):
                result = future.result()
                cases_by_name[result["case"]] = result
        cases = [cases_by_name[path.name] for path in gt_files]

    region_means: Dict[str, Dict[str, Optional[float]]] = {}
    for region in spec.regions:
        region_means[region.key] = {
            "name": region.name,
            "Dice": _mean(case["metrics"][region.key]["Dice"] for case in cases),
            "HD95": _mean(case["metrics"][region.key]["HD95"] for case in cases),
        }

    if spec.aggregate == "case_macro":
        foreground_dice = _mean(case["macro_dice"] for case in cases)
        foreground_hd95 = _mean(case["macro_hd95"] for case in cases)
    else:
        foreground_dice = _mean(region["Dice"] for region in region_means.values())
        foreground_hd95 = _mean(region["HD95"] for region in region_means.values())

    return {
        "meta": {
            "dataset": preset.name,
            "modality": preset.modality,
            "dataset_id": preset.dataset_id,
            "n_cases": len(cases),
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir),
            "regions": {
                region.key: list(region.labels) for region in spec.regions
            },
        },
        "mean": region_means,
        "foreground_mean": {
            "Dice": round(foreground_dice, 4) if foreground_dice is not None else None,
            "HD95": round(foreground_hd95, 4) if foreground_hd95 is not None else None,
        },
        "metric_per_case": cases,
    }


def write_metrics(summary: Dict[str, Any], output_dir: str | Path, dataset: str) -> Tuple[Path, Path]:
    preset = resolve_preset(dataset)
    assert preset is not None
    spec = SPECS[preset.name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{spec.metric_prefix}.json"
    csv_path = output_dir / f"{spec.metric_prefix}.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Region", "Name", "Dice_pct", "HD95_mm"])
        for key, region in summary["mean"].items():
            writer.writerow(
                [key, region["name"], region["Dice"], region["HD95"]]
            )
        writer.writerow(
            [
                "Mean",
                preset.modality,
                summary["foreground_mean"]["Dice"],
                summary["foreground_mean"]["HD95"],
            ]
        )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred-dirs", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preset = resolve_preset(args.dataset)
    assert preset is not None
    prefix = SPECS[preset.name].metric_prefix
    for pred_dir in map(Path, args.pred_dirs):
        json_path = pred_dir / f"{prefix}.json"
        csv_path = pred_dir / f"{prefix}.csv"
        if json_path.is_file() and csv_path.is_file() and not args.force:
            print(f"Skip existing metrics: {pred_dir}")
            continue
        summary = evaluate_predictions(
            preset.name,
            args.gt,
            pred_dir,
            workers=args.workers,
        )
        outputs = write_metrics(summary, pred_dir, preset.name)
        print(
            f"{preset.name}: Dice={summary['foreground_mean']['Dice']} "
            f"HD95={summary['foreground_mean']['HD95']} -> "
            + " | ".join(str(path) for path in outputs)
        )


if __name__ == "__main__":
    main()
