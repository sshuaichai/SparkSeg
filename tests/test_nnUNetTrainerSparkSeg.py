"""Checks for the dataset-specific SparkSeg trainer registrations."""

from __future__ import annotations

import os
from pathlib import Path

import SparkSeg.nnUNetTrainerSparkSeg as trainer_module
from SparkSeg.network.config import resolve_preset
from SparkSeg.reproduce import _subprocess_env
from SparkSeg.nnUNetTrainerSparkSeg import (
    TRAINER_BY_PRESET,
    nnUNetTrainerSparkSegACDC,
    nnUNetTrainerSparkSegBraTS2021,
    nnUNetTrainerSparkSegKiTS23,
    nnUNetTrainerSparkSegSynapse,
)


def test_each_manuscript_dataset_has_a_default_trainer():
    expected = {
        "acdc": (nnUNetTrainerSparkSegACDC, 100, 200, "MRI"),
        "synapse": (nnUNetTrainerSparkSegSynapse, 180, 1000, "CT"),
        "brats": (nnUNetTrainerSparkSegBraTS2021, 1251, 1000, "MRI"),
        "kits23": (nnUNetTrainerSparkSegKiTS23, 489, 1000, "CT"),
    }
    assert set(TRAINER_BY_PRESET) == set(expected)
    for preset, (trainer, dataset_id, epochs, modality) in expected.items():
        assert TRAINER_BY_PRESET[preset] is trainer
        assert trainer.PRESET == preset
        assert trainer.NUM_EPOCHS == epochs
        assert resolve_preset(preset).dataset_id == dataset_id
        assert resolve_preset(preset).modality == modality


def test_subprocesses_expose_the_sparkseg_trainer_package(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "other"))
    env = _subprocess_env()
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert any((Path(path) / "SparkSeg" / "nnUNetTrainerSparkSeg.py").is_file() for path in paths)
    assert str(tmp_path / "other") in paths
    assert (Path(paths[0]) / "SparkSeg" / "nnUNetTrainerSparkSeg.py").is_file()


def test_build_routes_the_selected_preset_and_plan_strides(monkeypatch):
    captured = {}

    def fake_build_sparkseg(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(trainer_module, "build_sparkseg", fake_build_sparkseg)
    config_manager = type(
        "ConfigManager",
        (),
        {"network_arch_init_kwargs": {"strides": [[1, 1, 1], [2, 2, 2]]}},
    )()

    trainer_module.nnUNetTrainerSparkSegKiTS23.build_network_architecture(
        None, config_manager, 1, 4, enable_deep_supervision=True
    )

    assert captured["args"] == (1, 4)
    assert captured["kwargs"]["preset"] == "kits23"
    assert captured["kwargs"]["plan_strides"] == [[1, 1, 1], [2, 2, 2]]
    assert captured["kwargs"]["enable_deep_supervision"] is True
