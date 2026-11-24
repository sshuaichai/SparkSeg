# ✨ SparkSeg

English | [Chinese](README_CN.md)

**SparkSeg: sparse prior-guided anchor attention for 3D CT and MRI segmentation**

SparkSeg keeps the dense convolutional U-Net host and makes long-range context an explicit, ablatable
read/write operation on a fixed Top-K anchor set. A task-conditioned prior decides where to read, a hard
stage-wise Top-K fixes how many anchors are used, and multi-source sparse block attention with gated
residual write-back updates only the selected sites.

![SparkSeg overview](nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig01_overview.png)

## 💡 Highlights

- **Backbone-preserving.** Adds global context to dense 3D CNNs without replacing the backbone.
- **Task-aware anchors.** Anchors track image-specific anatomy while remaining task-relevant.
- **Selective write-back.** Preserves local detail while propagating global evidence.
- **Inspectable routing.** Stage-wise anchor maps reveal where long-range evidence enters the network.
- **Strong results.** Highest mean DSC and lowest reported HD95 on three benchmarks, with KiTS23 transfer.

## 📊 Results

Values are reported as DSC (%) / HD95 (mm). Higher DSC and lower HD95 are
better; SparkSeg improves on the matched nnU-Net control on all four datasets.

| Dataset      | SparkSeg DSC (%) ↑ / HD95 (mm) ↓ | nnU-Net DSC (%) ↑ / HD95 (mm) ↓ | nnU-Net ResEnc DSC (%) ↑ / HD95 (mm) ↓ |
| ------------ | -------------------------------: | ------------------------------: | -------------------------------------: |
| ACDC         | **91.98 / 1.08**                 | 91.17 / 1.11                    | 91.13 / 1.09                           |
| BraTS2021    | **91.73 / 2.57**                 | 91.36 / 2.86                    | 91.51 / 2.60                           |
| Synapse/BTCV | **85.94 / 11.21**                | 84.43 / 11.72                   | 85.54 / 11.91                          |
| KiTS23       | **85.66 / 13.41**                | 85.12 / 18.41                   | 84.36 / 17.78                          |

## 📋 Full method comparison

The following tables report the method comparison (Tables 1–4). The best value in every numerical column is shown in bold, and the proposed configuration is shown in bold in the first column.

### Table 1. BraTS2021 comparison

BraTS2021 held-out test set. (a) DSC (%) and HD95 (mm); (b) compute profile.

**(a) Segmentation**

| Model | DSC WT | DSC TC | DSC ET | DSC Avg | HD95 WT | HD95 TC | HD95 ET | HD95 Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnU-Net [1] | 93.96 | 91.91 | 88.22 | 91.36 | 3.68 | 2.83 | 2.06 | 2.86 |
| nnU-Net ResEnc [22] | **94.05** | 91.92 | 88.57 | 91.51 | 3.33 | 2.54 | 1.93 | 2.60 |
| AttU-Net [9] | 94.02 | 91.59 | 88.20 | 91.27 | 3.29 | 2.54 | 2.17 | 2.67 |
| CoTr [24] | 93.78 | 91.13 | 87.99 | 90.97 | 3.59 | 2.97 | 2.17 | 2.91 |
| LightUNETR-Large [36] | 93.46 | 90.55 | 87.08 | 90.36 | **3.22** | 2.69 | 2.05 | 2.65 |
| SegMamba [16] | 94.04 | 91.35 | 87.57 | 90.99 | 3.73 | 3.08 | 2.20 | 3.00 |
| SegMamba-V2 [17] | 93.89 | 91.44 | 88.30 | 91.21 | 3.80 | 2.73 | 2.49 | 3.01 |
| SlimUNETR-V2 [37] | 92.93 | 91.74 | 86.20 | 90.29 | 3.49 | 2.54 | 2.43 | 2.82 |
| Swin-Unet [7] | 92.97 | 89.85 | 85.80 | 89.54 | 4.25 | 3.14 | 2.37 | 3.25 |
| TransUNet [4] | 93.58 | 91.36 | 88.50 | 91.15 | 3.94 | 2.82 | 1.99 | 2.92 |
| U-Mamba [13] | 93.78 | 91.74 | 88.58 | 91.37 | 3.78 | 2.78 | 2.22 | 2.93 |
| UNETR [5] | 92.97 | 90.18 | 86.67 | 89.94 | 4.18 | 3.85 | 2.96 | 3.66 |
| UNETR++ [38] | 93.54 | 91.69 | 87.59 | 90.94 | 3.93 | 3.06 | 2.40 | 3.13 |
| **SparkSeg** | 93.87 | **92.68** | **88.63** | **91.73** | 3.41 | **2.41** | **1.89** | **2.57** |

**(b) Compute**

| Model | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
| :--- | ---: | ---: | ---: | ---: |
| nnU-Net [1] | ~**6.2** / 1.7 | ~31.2 | ~538.1 | ~0.340 |
| nnU-Net ResEnc [22] | ~11.6 / 2.2 | ~102.4 | ~778.3 | ~0.460 |
| AttU-Net [9] | ~13.6 / 2.4 | ~23.6 | ~586.0 | ~0.311 |
| CoTr [24] | ~12.0 / 3.2 | ~41.9 | ~787.4 | ~0.301 |
| LightUNETR-Large [36] | ~15.7 / 3.8 | **~3.9** | ~129.4 | ~0.544 |
| SegMamba [16] | ~37.2 / 6.9 | ~67.4 | ~1573.0 | ~0.728 |
| SegMamba-V2 [17] | ~18.0 / 3.0 | ~139.2 | ~1982.5 | ~0.595 |
| SlimUNETR-V2 [37] | ~15.9 / **0.3** | ~23.6 | **~41.9** | **~0.138** |
| Swin-Unet [7] | ~9.2 / 3.1 | ~30.6 | ~46.8 | ~0.251 |
| TransUNet [4] | ~13.4 / 4.8 | ~119.0 | ~555.0 | ~0.620 |
| U-Mamba [13] | ~18.8 / 3.4 | ~42.2 | ~975.5 | ~0.384 |
| UNETR [5] | ~8.2 / 2.9 | ~130.8 | ~202.8 | ~0.288 |
| UNETR++ [38] | ~6.5 / 1.2 | ~23.5 | ~87.5 | ~0.265 |
| **SparkSeg** | ~11.3 / 2.5 | ~34.8 | ~575.1 | ~0.356 |

### Table 2. ACDC comparison

ACDC held-out test set. (a) DSC (%) and HD95 (mm); (b) compute profile.

**(a) Segmentation**

| Model | DSC RV | DSC MYO | DSC LV | DSC Avg | HD95 RV | HD95 MYO | HD95 LV | HD95 Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnU-Net [1] | 88.66 | 89.51 | 95.33 | 91.17 | 1.25 | **1.02** | 1.06 | 1.11 |
| nnU-Net ResEnc [22] | 88.70 | 89.27 | 95.41 | 91.13 | 1.22 | **1.02** | 1.03 | 1.09 |
| AttU-Net [9] | 86.60 | 88.15 | 94.72 | 89.83 | 4.04 | 1.04 | 1.04 | 2.04 |
| CoTr [24] | 85.24 | 86.68 | 94.07 | 88.66 | 4.64 | 1.09 | 1.09 | 2.27 |
| SegMamba [16] | 85.79 | 87.95 | 94.39 | 89.37 | 1.73 | 1.14 | 1.07 | 1.31 |
| SegMamba-V2 [17] | 87.94 | 89.10 | 95.22 | 90.76 | 2.58 | 1.13 | 1.04 | 1.58 |
| Swin-Unet [7] | 80.61 | 82.63 | 92.31 | 85.18 | 4.38 | 1.31 | 1.19 | 2.29 |
| TransUNet [4] | 88.73 | 89.07 | 95.24 | 91.01 | 1.29 | 1.03 | **1.02** | 1.12 |
| U-Mamba [13] | 71.23 | 75.30 | 89.09 | 78.54 | 4.64 | 2.07 | 1.89 | 2.87 |
| UNETR [5] | 74.98 | 80.84 | 90.84 | 82.22 | 14.31 | 1.62 | 1.55 | 5.83 |
| UNETR++ [38] | 82.23 | 85.39 | 93.06 | 86.89 | 1.90 | 1.29 | 1.14 | 1.44 |
| **SparkSeg** | **90.24** | **89.90** | **95.80** | **91.98** | **1.19** | **1.02** | **1.02** | **1.08** |

**(b) Compute**

| Model | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
| :--- | ---: | ---: | ---: | ---: |
| nnU-Net [1] | ~**6.2** / **0.6** | ~31.2 | ~203.4 | ~0.075 |
| nnU-Net ResEnc [22] | ~14.5 / 1.2 | ~102.4 | ~374.3 | ~0.055 |
| AttU-Net [9] | ~13.0 / 0.7 | ~23.6 | ~176.9 | **~0.045** |
| CoTr [24] | ~16.8 / 1.9 | ~41.9 | ~310.7 | ~0.112 |
| SegMamba [16] | ~45.5 / 3.1 | ~67.4 | ~678.8 | ~0.117 |
| SegMamba-V2 [17] | ~26.2 / 1.7 | ~139.2 | ~863.6 | ~0.100 |
| Swin-Unet [7] | ~25.5 / 2.3 | ~30.6 | **~40.4** | ~0.212 |
| TransUNet [4] | ~18.1 / 2.6 | ~119.0 | ~232.4 | ~0.161 |
| U-Mamba [13] | ~18.4 / 3.4 | ~42.2 | ~359.3 | ~0.092 |
| UNETR [5] | ~10.5 / 1.4 | ~130.8 | ~85.4 | ~0.095 |
| UNETR++ [38] | ~12.8 / 1.5 | **~23.5** | ~64.1 | ~0.120 |
| **SparkSeg** | ~15.0 / 1.2 | ~111.1 | ~440.4 | ~0.087 |

### Table 3a. Synapse/BTCV DSC

Synapse/BTCV: DSC (%) by organ (12 held-out cases).

| Model | Aor | Gal | LK | RK | Liv | Pan | Spl | Sto | Avg DSC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnU-Net [1] | 92.70 | 66.09 | 87.07 | 87.17 | 96.54 | 74.30 | 92.86 | 78.70 | 84.43 |
| nnU-Net ResEnc [22] | 93.16 | 70.77 | 85.60 | 86.45 | 96.48 | 75.83 | 93.69 | **82.31** | 85.54 |
| AttU-Net [9] | 92.91 | 68.55 | 83.68 | 84.97 | 96.36 | 69.72 | 91.95 | 76.55 | 83.09 |
| CoTr [24] | **93.48** | 67.06 | 86.84 | 86.30 | 96.82 | **78.68** | 92.97 | 80.64 | 85.35 |
| SegMamba [16] | 91.03 | 61.18 | 85.43 | 86.99 | 95.83 | 68.26 | 91.47 | 73.24 | 81.68 |
| SegMamba-V2 [17] | 92.50 | 59.65 | **87.71** | 85.23 | 95.92 | 70.15 | **94.83** | 73.90 | 82.49 |
| Swin-Unet [7] | 83.30 | 54.93 | 79.20 | 74.86 | 92.85 | 49.66 | 85.29 | 68.21 | 73.54 |
| TransUNet [4] | 93.18 | 64.04 | 85.65 | 87.03 | 96.57 | 70.69 | 91.25 | 75.81 | 83.03 |
| U-Mamba [13] | 91.64 | 58.98 | 80.94 | 77.30 | 96.03 | 60.08 | 87.66 | 58.20 | 76.35 |
| UNETR [5] | 84.49 | 63.26 | 78.44 | 78.01 | 93.79 | 51.34 | 80.49 | 56.40 | 73.28 |
| UNETR++ [38] | 91.78 | 58.16 | 86.61 | 85.45 | 95.84 | 64.81 | 91.19 | 74.73 | 81.07 |
| **SparkSeg** | 93.25 | **73.66** | 85.54 | **89.93** | **96.97** | 75.20 | 92.49 | 80.45 | **85.94** |

Aor, aorta; Gal, gallbladder; LK/RK, left/right kidney; Liv, liver; Pan, pancreas; Spl, spleen; Sto, stomach.

### Table 3b. Synapse/BTCV HD95

Synapse/BTCV: HD95 (mm) by organ.

| Model | Aor | Gal | LK | RK | Liv | Pan | Spl | Sto | Avg HD95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnU-Net [1] | 3.12 | 25.26 | 17.50 | 7.26 | 4.29 | 10.02 | 8.14 | 20.04 | 11.72 |
| nnU-Net ResEnc [22] | **1.48** | 14.60 | 18.34 | 8.94 | 7.41 | 7.09 | 21.13 | **17.34** | 11.91 |
| AttU-Net [9] | 4.83 | 23.22 | 7.91 | 24.00 | 3.19 | 10.44 | 8.13 | 32.88 | 14.11 |
| CoTr [24] | 1.49 | 18.45 | 15.91 | 10.15 | 2.16 | **6.40** | 6.95 | 31.51 | 11.29 |
| SegMamba [16] | 8.74 | 31.36 | 17.85 | 6.16 | 15.80 | 9.56 | 29.31 | 33.66 | 18.76 |
| SegMamba-V2 [17] | 8.09 | 22.52 | 15.29 | 22.53 | 18.16 | 16.04 | **6.60** | 32.42 | 17.50 |
| Swin-Unet [7] | 34.78 | 23.13 | 53.22 | 28.03 | 17.86 | 16.43 | 62.78 | 36.69 | 33.92 |
| TransUNet [4] | 1.87 | 25.85 | 18.73 | 8.88 | **1.76** | 11.04 | 13.30 | 30.17 | 13.70 |
| U-Mamba [13] | 11.34 | 23.03 | **4.96** | 10.81 | 4.76 | 28.66 | 11.50 | 24.55 | 14.74 |
| UNETR [5] | 10.91 | 18.71 | 26.95 | 46.51 | 42.48 | 23.24 | 38.98 | 37.76 | 30.55 |
| UNETR++ [38] | 2.43 | 18.04 | 17.31 | 24.51 | 3.50 | 11.42 | 8.97 | 23.78 | 13.61 |
| **SparkSeg** | 1.67 | **13.23** | 18.60 | **3.87** | 2.12 | 9.66 | 21.57 | 19.86 | **11.21** |

### Table 3c. Synapse/BTCV resources

Synapse/BTCV: peak training/inference memory, parameters, FLOPs, and latency.

| Model | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
| :--- | ---: | ---: | ---: | ---: |
| nnU-Net [1] | ~7.9 / 3.1 | ~31.2 | ~619.6 | **~3.290** |
| nnU-Net ResEnc [22] | ~11.7 / 3.5 | ~102.4 | ~916.2 | ~5.785 |
| AttU-Net [9] | ~15.0 / **2.6** | ~23.6 | ~670.4 | ~4.079 |
| CoTr [24] | ~14.5 / 3.6 | ~41.9 | ~865.6 | ~3.818 |
| SegMamba [16] | ~42.3 / 8.7 | ~67.4 | ~2037.2 | ~12.166 |
| SegMamba-V2 [17] | ~23.0 / 3.4 | ~139.2 | ~2591.6 | ~10.181 |
| Swin-Unet [7] | ~13.6 / 4.8 | ~30.6 | **~62.0** | ~5.292 |
| TransUNet [4] | ~16.8 / 6.9 | ~119.0 | ~663.8 | ~9.667 |
| U-Mamba [13] | ~24.1 / 4.7 | ~42.2 | ~1126.8 | ~7.493 |
| UNETR [5] | ~13.0 / 4.2 | ~130.8 | ~256.4 | ~5.476 |
| UNETR++ [38] | ~**7.8** / 3.2 | **~23.5** | ~64.5 | ~9.335 |
| **SparkSeg** | ~13.7 / 2.7 | ~111.1 | ~1089.4 | ~7.523 |

### Table 4. KiTS23 comparison

KiTS23: DSC (%) and HD95 (mm) on 98 held-out cases.

| Model | DSC K&M | DSC KM | DSC T | DSC Avg | HD95 K&M | HD95 KM | HD95 T | HD95 Avg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nnU-Net | 95.89 | 81.56 | 77.92 | 85.12 | 11.02 | 20.49 | 23.73 | 18.41 |
| nnU-Net ResEnc | 95.69 | 81.32 | 76.06 | 84.36 | 8.60 | 19.69 | 25.06 | 17.78 |
| AttU-Net | 94.47 | 77.76 | 72.35 | 81.53 | 13.13 | 26.90 | 31.83 | 23.95 |
| **SparkSeg** | **96.16** | **82.83** | **78.00** | **85.66** | **5.86** | **15.18** | **19.19** | **13.41** |

## 🧠 Available trainers

| Modality | Dataset | ID | Trainer | Host | Initialisation | Epochs |
|---|---|---:|---|---|---|---:|
| MRI | ACDC | 100 | `nnUNetTrainerSparkSegACDC` | residual encoder | no `stock_he` | 200 |
| MRI | BraTS2021 | 1251 | `nnUNetTrainerSparkSegBraTS2021` | plain U-Net | no `stock_he` | 1,000 |
| CT | Synapse/BTCV | 180 | `nnUNetTrainerSparkSegSynapse` | residual encoder | `stock_he` | 1,000 |
| CT | KiTS23 | 489 | `nnUNetTrainerSparkSegKiTS23` | residual encoder | `stock_he` | 1,000 |

The canonical trainers are in
[nnUNetTrainerSparkSeg.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainerSparkSeg.py),
and the evaluator is in [nnUNetTrainerSparkSeg_eval.py](nnUNetTrainerSparkSeg_eval.py).
Model, configuration, utility, paper assets, and fixed splits are in
[nnunetv2/training/network/source_code/SparkSeg](nnunetv2/training/network/source_code/SparkSeg/).

| Dataset | Patch | Batch | Training memory | Params | FLOPs | Latency |
|---|---|---:|---:|---:|---:|---:|
| ACDC | 224×256×10 | 7 | 15.0 GB | 111.1 M | 440.4 G | 0.087 s |
| BraTS2021 | 128×128×128 | 2 | 11.3 GB | 34.8 M | 575.1 G | 0.356 s |
| Synapse/BTCV | 56×192×224 | 2 | 13.7 GB | 111.1 M | 1,089.4 G | 7.523 s |
| KiTS23 | 128×128×128 | 2 | — | — | — | — |

## ⚙️ Installation

The commands below use a Linux or WSL shell. The three data directories are
configured by the user; the scripts read the standard nnU-Net environment
variables:

```text
nnUNet_raw           raw data
nnUNet_preprocessed  preprocessed data
nnUNet_results       training results and checkpoints
```

```bash
cd /path/to/SparkSeg
python -m pip install -e .

export NNUNET_DATA_ROOT=/path/to/nnUNet_data
export nnUNet_raw="$NNUNET_DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_DATA_ROOT/nnUNet_results"

python -m SparkSeg.reproduce doctor --datasets acdc brats synapse kits23
```

Run from this checkout so the bundled, SparkSeg-enabled `nnunetv2/` package is
used. For training and prediction, the reproduction CLI creates temporary trainer
registration files, so no external evaluation or bridge scripts are required.

## 🚀 Training

Training uses `L_seg + 0.05 L_prior + 0.02 L_prioraux`, SGD with learning rate
`1e-2`, Nesterov momentum `0.99`, weight decay `3e-5`, polynomial decay, and
`250/50` training/validation iterations per epoch.

Prepare a new dataset first. For KiTS23:

```bash
python -m SparkSeg.reproduce prepare --datasets kits23
```

Prepare all four datasets with:

```bash
python -m SparkSeg.reproduce prepare --datasets acdc brats synapse kits23
```

Train each dataset with:

```bash
python -m SparkSeg.reproduce train --datasets acdc
python -m SparkSeg.reproduce train --datasets brats
python -m SparkSeg.reproduce train --datasets synapse
python -m SparkSeg.reproduce train --datasets kits23
```

For multi-GPU training:

```bash
python -m SparkSeg.reproduce train --datasets brats --num-gpus 2
```

The trained checkpoint is written to:

```text
$nnUNet_results/<Dataset>/<Trainer>__nnUNetPlans__3d_fullres/fold_<fold>/checkpoint_final.pth
```

## 🔎 Prediction and evaluation

```bash
python -m SparkSeg.reproduce predict --datasets acdc
python -m SparkSeg.reproduce predict --datasets brats
python -m SparkSeg.reproduce predict --datasets synapse
python -m SparkSeg.reproduce predict --datasets kits23

python -m SparkSeg.reproduce evaluate --datasets acdc
python -m SparkSeg.reproduce evaluate --datasets brats
python -m SparkSeg.reproduce evaluate --datasets synapse
python -m SparkSeg.reproduce evaluate --datasets kits23
```

`validate` is an alias of `evaluate`. Predictions and metrics are written under:

```text
$nnUNet_raw/<Dataset>/labelsTs_SparkSeg/final_pre
```

Inference uses no test-time augmentation (TTA).

| Dataset | Metric files |
|---|---|
| ACDC | `metrics_ACDC_testset.json/.csv` |
| BraTS2021 | `metrics_BraTS_WT_TC_ET_testset.json/.csv` |
| Synapse/BTCV | `metrics_synapse_testset.json/.csv` |
| KiTS23 | `metrics_KiTS23_HEC_testset.json/.csv` |

## 📦 Release weights

To download and install the released weights:

```bash
python -m SparkSeg.reproduce download --out ./SparkSeg_releases --tag v1.3.0

python -m SparkSeg.reproduce install \
  ./SparkSeg_releases/sparkseg-acdc-mri-3d_fullres.zip \
  ./SparkSeg_releases/sparkseg-brats2021-mri-3d_fullres.zip \
  ./SparkSeg_releases/sparkseg-synapse-ct-3d_fullres.zip \
  ./SparkSeg_releases/sparkseg-kits23-ct-3d_fullres.zip

python -m SparkSeg.reproduce verify --load-checkpoints
```

Each archive contains `manifest.json`, `plans.json`, `dataset.json`, and
`fold_<fold>/checkpoint_final.pth`, with `splits_final.json` when a fixed split
is used.

## 🐍 Python API

```python
from SparkSeg import build_sparkseg, describe_preset

strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
net = build_sparkseg(4, 4, plan_strides=strides, preset="brats")
print(describe_preset("brats"))
```

Preset names are `acdc`, `brats`/`brats2021`, `synapse`/`btcv`, and
`kits23`/`kits`. Dataset IDs and nnU-Net folder names are also accepted.

## 🗂️ Data and splits

A fixed split file is provided in [splits](nnunetv2/training/network/source_code/SparkSeg/splits/) for every preset. ACDC,
BraTS2021, and KiTS23 train on fold 0. Synapse uses fold `all`: its split file
records the 18 cases in `imagesTr`, which nnU-Net uses for both training and
validation, while the 12 cases in `imagesTs` remain the held-out test set.

| Dataset | Modality | Cases | Training / validation / held-out test | Source |
|---|---|---:|---|---|
| ACDC | MRI | 100 patients | 70 / 10 / 20 patients (140 / 20 / 40 volumes) | [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| BraTS2021 | MRI | 1,251 | 875 / 125 / 251 | [Synapse syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| Synapse/BTCV | CT | 30 | 18 training; 12 held-out test | [Synapse syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| KiTS23 | CT | 489 | 342 / 49 / 98 | [KiTS23 challenge](https://kits-challenge.org/kits23/) |

Preprocessing, plans, topology, and augmentation follow the standard `3d_fullres`
protocol. KiTS23 is distributed under CC BY-NC-SA 4.0.

## 🖼️ Figures

<p align="center">
  <a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig01_overview.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig01_overview.png" width="900" alt="Figure 1"></a>
</p>

<table>
  <tr>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig02_qualitative.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig02_qualitative.png" width="320" alt="Figure 2"></a><br><strong>Figure 2</strong>: Qualitative comparison on ACDC, BraTS2021 and Synapse/BTCV</td>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig03_per_case_dsc.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig03_per_case_dsc.png" width="320" alt="Figure 3"></a><br><strong>Figure 3</strong>: Per-case DSC distributions and mean HD95</td>
  </tr>
  <tr>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig04_resource_profiles.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig04_resource_profiles.png" width="320" alt="Figure 4"></a><br><strong>Figure 4</strong>: Memory, FLOPs and latency profiles</td>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig05_kits23.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig05_kits23.png" width="320" alt="Figure 5"></a><br><strong>Figure 5</strong>: KiTS23 tumour comparison</td>
  </tr>
  <tr>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig06_anchor_visualisation.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig06_anchor_visualisation.png" width="320" alt="Figure 6"></a><br><strong>Figure 6</strong>: Prior-guided Top-K anchors at Stages 2–5</td>
    <td align="center" width="50%"><a href="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig07_selection_quality.png"><img src="nnunetv2/training/network/source_code/SparkSeg/assets/paper/fig07_selection_quality.png" width="320" alt="Figure 7"></a><br><strong>Figure 7</strong>: Anchor-selection quality (Precision@K, Recall@K, Enrichment, BoundaryHit@K)</td>
  </tr>
</table>

## 📄 Citation and licence

```bibtex
@misc{chai2026sparkseg,
  author  = {Chai, Shuaishuai and Jin, Liang and Li, Ming},
  title   = {SparkSeg: sparse prior-guided anchor attention for 3D CT and MRI segmentation},
  year    = {2026},
  url     = {https://github.com/sshuaichai/SparkSeg}
}
```

Released under [Apache-2.0](LICENSE.txt). Training and inference use
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
