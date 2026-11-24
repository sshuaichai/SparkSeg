"""Compatibility exports for the canonical SparkSeg nnU-Net trainers."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSparkSeg import (  # noqa: F401
    TRAINER_BY_PRESET,
    TRAINER_CLASSES,
    nnUNetTrainerSparkSegACDC,
    nnUNetTrainerSparkSegBraTS2021,
    nnUNetTrainerSparkSegKiTS23,
    nnUNetTrainerSparkSegSynapse,
)

__all__ = [
    "nnUNetTrainerSparkSegACDC",
    "nnUNetTrainerSparkSegSynapse",
    "nnUNetTrainerSparkSegBraTS2021",
    "nnUNetTrainerSparkSegKiTS23",
    "TRAINER_CLASSES",
    "TRAINER_BY_PRESET",
]
