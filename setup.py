"""Setuptools entry point for the combined SparkSeg and nnU-Net source tree."""

from setuptools import find_packages, setup

nnunet_packages = ["nnunetv2"] + [
    f"nnunetv2.{package}" for package in find_packages(where="nnunetv2")
]

setup(
    packages=["SparkSeg", *nnunet_packages],
    package_dir={"SparkSeg": ".", "nnunetv2": "nnunetv2"},
    entry_points={
        "console_scripts": [
            "sparkseg-reproduce = SparkSeg.reproduce:main",
            "nnUNetv2_plan_and_preprocess = "
            "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:"
            "plan_and_preprocess_entry",
            "nnUNetv2_train = nnunetv2.run.run_training:run_training_entry",
            "nnUNetv2_predict = "
            "nnunetv2.inference.predict_from_raw_data:predict_entry_point",
        ]
    },
)
