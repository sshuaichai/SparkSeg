"""Regression checks for the released evaluation metrics."""

from __future__ import annotations

import numpy as np

from SparkSeg.nnUNetTrainerSparkSeg_eval import _hd95


def test_hd95_uses_surface_to_surface_distances():
    shape = (64, 64, 64)
    zz, yy, xx = np.indices(shape)
    reference = (
        ((zz - 32) / 20) ** 2
        + ((yy - 32) / 16) ** 2
        + ((xx - 32) / 24) ** 2
        <= 1
    )
    prediction = (
        ((zz - 34) / 18) ** 2
        + ((yy - 30) / 14) ** 2
        + ((xx - 30) / 22) ** 2
        <= 1
    )
    assert np.isclose(
        _hd95(prediction, reference, (1.0, 1.0, 1.0)),
        4.898979485566356,
    )
