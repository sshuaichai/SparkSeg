# SparkSeg

**Sparse prior-guided anchor attention for robust 3D CT and MRI segmentation.**

English | [Chinese](README_CN.md)

<p align="center">
  <img src="assets/paper/fig01_overview.png" alt="Fig. 1 — Overview of the SparkSeg framework" width="900">
</p>

<p align="center"><sub><strong>Fig. 1 | Overview of the SparkSeg framework.</strong> (a) Encoder–decoder spanning Stages 1–5 and the bottleneck, annotated with the per-stage channel counts and spatial sizes. Stage 1 stacks the tri-axial enhancement (TAE) only; Stages 2–5 each stack one SparkSeg Block; <em>hist</em> marks the historical memory passed to the next stage's block; skip connections feed a U-Net decoder, and the backbone is either a plain U-Net or a residual-encoder U-Net. (b) The SparkSeg Block and its sub-modules: TAE, windowed multi-head self-attention (WinMHSA3D) with its regular and shifted passes (Δ<sub>regular</sub>, Δ<sub>shift</sub>) and gate σ<sub>win</sub>, prior-gated token selection (PGTS) with the local prior P<sub>loc</sub>, the effective prior P<sub>eff</sub> and the per-stage hard Top-K budget K<sub>s</sub>, and multi-source sparse block attention (MSBA) with the historical source Hist<sub>s</sub>, the write-back gate σ<sub>ba</sub> and the token update Δ<sub>tokens</sub>. (c) Decoder block with its attention gate. Operators: Δ residual · σ sigmoid · e energy · ⊖ subtraction · ⊕ residual addition · ⊙ multiplication.</sub></p>

SparkSeg keeps a **dense convolutional grid** intact and spends a **fixed token budget** at task-relevant
anchors: a prior-gated ranker selects **hard Top-K** anchors, multi-source sparse block attention writes
context back **only at those anchors**, and tri-axial plus windowed attention supply local detail. The same
block runs on a plain U-Net host and on a residual-encoder U-Net host, so the read/write design can be
measured separately from backbone capacity.

## Headline results

| Benchmark | Host reported | Init arm | Mean DSC (%) ↑ | Mean HD95 (mm) ↓ | Peak train mem | Params | FLOPs | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| ACDC | residual encoder | `no_stock_he` | **91.98** | **1.08** | 15.0 GB | 111.1 M | 440.4 G | 0.087 |
| Synapse/BTCV | residual encoder | `stock_he` | **85.94** | **11.21** | 13.7 GB | 111.1 M | 1,089.4 G | 7.523 |
| BraTS2021 | plain U-Net | `no_stock_he` | **91.73** | **2.57** | 11.3 GB | 34.8 M | 575.1 G | 0.356 |

Every pipeline trains within **11.3–15.0 GB on a single 24 GB GPU**, and on BraTS2021 SparkSeg uses
**2.7× fewer FLOPs** and 3.3× less memory than SegMamba at a higher mean DSC.

## What this package contains

This is the **released network package** that matches the manuscript: the clean configuration, the
per-benchmark presets, the model code, the smoke tests and the figures.

| Included | Not included (by design) |
|---|---|
| `SparkSeg` host, `SparkSegBlock`, `TAE`, `PriorHead`, `PGTS`, `MSBA`, `WinMHSA3D`, `AttentionGate` | nnU-Net **trainers** (wire `build_sparkseg` into your own trainer) |
| Clean configuration (`CLEAN_CONFIG`) + per-dataset presets (`DATASET_PRESETS`) | Preprocessing / experiment planning scripts |
| `verify_clean_config()` + `tests/test_sparkseg.py` | Evaluation scripts |

## Naming: paper ↔ code

| Manuscript | Code object | Notes |
|---|---|---|
| SparkSeg | `SparkSeg` | encoder–decoder host (Fig. 1a) |
| SparkSeg Block | `SparkSegBlock` | one budgeted read/write operator per guided stage (Fig. 1b) |
| TAE — tri-axial enhancement | `TAE` | Δ_orient and the residual-energy probe e (Eq. 1–2) |
| PriorHead — prior synthesis | `PriorHead` | P_loc → P_eff, energy-modulated (Eq. 3) |
| PGTS — prior-gated token selection | `PGTS` | importance ranking + hard Top-K (Eq. 4–6) |
| MSBA — multi-source sparse block attention | `MSBA` | source-wise softmax + gated residual write-back |
| WinMHSA3D — windowed MHSA | `WinMHSA3D` | Δ_regular / Δ_shift, gate σ_win |
| AG — decoder attention gate | `AttentionGate` | spatial soft gate on each skip (Fig. 1c) |

Pre-manuscript names are kept as **exact aliases**: `SPARKUNet`, `SPARKUnit`, `SPARKDecoder`,
`AxialDW3D`, `TokenAllocationNetwork`, `LocalSelectiveBlockAttn`, `WindowMHSA3D`,
`DecoderAttentionGate`, `build_sparkunet`.

## Quickstart

```python
from SparkSeg import build_sparkseg, describe_preset, DATASET_PRESETS

# strides come from the host plans:
#   configuration_manager.network_arch_init_kwargs["strides"]
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

print(describe_preset("brats"))
# brats (Dataset1251) host=plainconv patch=(128, 128, 128) batch=2 epochs=1000
# K=(128, 64, 32, 32) | reported: mean_dsc=91.73, mean_hd95=2.57, ...

net = build_sparkseg(in_channels=4, out_channels=4, plan_strides=strides, preset="brats")
```

`preset` accepts the benchmark key (`"acdc"` / `"synapse"` / `"brats"`), the nnU-Net dataset id
(`100` / `180` / `1251`) or the folder name (`"Dataset1251_BraTS2023GLI"`). Omitting `preset` builds the
clean configuration on the plain U-Net host; `build_sparkseg_for("acdc", ...)` is the same as
`build_sparkseg(..., preset="acdc")`. The canonical network-namespace import is
`from nnunetv2.training.network.SparkSeg import build_sparkseg`.

In train mode the host returns `(seg, prior)` — `prior` feeds the auxiliary prior loss — and exposes each
guided stage's P_eff through `net._last_stage_p_effs`; in eval mode it returns the segmentation output
only, so sliding-window inference works unchanged.

## Clean configuration (paper default)

| Component | Setting |
|---|---|
| Initialization | `stock_he` — He(1e-2) post-init plus zero-last-BN before each residual add (the product default). The two training variants are `nnUNetTrainer*_StockHe` / `*_NoStockHe`; the preset adopts the arm that produced the reported number (see below) |
| Stage mount | Stage 1 (E0): TAE only · Stages 2–5 (E1–E4): one SparkSeg Block each · Stage 6 (E5): CNN bottleneck + PriorHead |
| Anchor budget | hard Top-K 128/64/32/32 (Eq. 4–6: α = 8, K_min = 32, K_max = 512, S = 6) |
| Local window attention | 4×4×4 windows, 4 heads, regular + half-window shift (shift = 2) |
| Gates | σ_ba and σ_win start closed (scalar bias −2.5); decoder AG starts near identity (bias +4) |
| Auxiliary losses | λ_prior = 0.05, λ_prioraux = 0.02 (Eq. 8) |
| Optimiser / schedule | SGD, lr 1e-2, Nesterov 0.99, weight decay 3e-5, polynomial decay; 250 train / 50 validation iterations per epoch |

`CLEAN_CONFIG` in `SparkSeg.py` is the single source of truth, and `verify_clean_config()` re-checks it
key-by-key against the training stack it was frozen from (it reports `{"checked": True, "mismatch": {}}`
when aligned).

## Per-dataset configurations (manuscript Table 9)

| Benchmark | Reported configuration | Init arm | Batch size | Patch size | Epochs | Mean DSC (plain / residual host) |
|---|---|---|---|---|---|---|
| BraTS2021 | Plain U-Net host | `no_stock_he` | 2 | 128×128×128 | 1,000 | 91.73 / 91.50 |
| ACDC | Residual-encoder U-Net host | `no_stock_he` | 7 | 224×256×10 | 200 | 91.18 / 91.98 |
| Synapse | Residual-encoder U-Net host | `stock_he` | 2 | 56×192×224 | 1,000 | 85.73 / 85.94 |

The **init arm** column is provenance, not a preference: it records which initialization variant
produced the reported endpoint (`no_stock_he` = the paper-period runs without the He post-init;
`stock_he` = He(1e-2) + zero-last-BN). All three benchmarks were also trained with the other arm, and
the two arms differ by ≤0.72 DSC — see the note below. `build_sparkseg(..., preset=...)` applies the
arm listed here; pass `stock_he=True`/`False` explicitly to override it.

> Note: the reported numbers are the runs that entered the manuscript. Those runs pre-date the current
> K pyramid (they also used a tighter 64/32/32/32 budget and an older code snapshot), so treat the
> init column as provenance of the endpoint rather than as a controlled single-variable ablation.

The same numbers live in code:

```python
from SparkSeg import DATASET_PRESETS
DATASET_PRESETS["acdc"].host          # 'resenc'
DATASET_PRESETS["acdc"].init          # 'no_stock_he'  (the arm behind 91.98)
DATASET_PRESETS["acdc"].reported      # {'mean_dsc': 91.98, 'mean_hd95': 1.08, ...}
```

## Results

Tables 2–4 reproduce the manuscript's benchmark tables (DSC ↑ %, HD95 ↓ mm; the last four columns are
approximate estimates obtained with one accounting across the three benchmarks; the first two rows are the
control backbones that keep the baseline nnU-Net plan).

### Table 2 | BraTS2021 held-out test set (WT/TC/ET)

| Model | WT | TC | ET | Avg. | WT | TC | ET | Avg. | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nnU-Net | 93.96 | 91.91 | 88.22 | 91.36 | 3.68 | 2.83 | 2.06 | 2.86 | ~6.2 / 1.7 | ~31.2 | ~538.1 | ~0.340 |
| nnU-Net ResEnc | 94.05 | 91.92 | 88.57 | 91.51 | 3.33 | 2.54 | 1.93 | 2.6 | ~11.6 / 2.2 | ~102.4 | ~778.3 | ~0.460 |
| AttUNet | 94.02 | 91.59 | 88.2 | 91.27 | 3.29 | 2.54 | 2.17 | 2.67 | ~13.6 / 2.4 | ~23.6 | ~586 | ~0.311 |
| CoTr | 93.78 | 91.13 | 87.99 | 90.97 | 3.59 | 2.97 | 2.17 | 2.91 | ~12.0 / 3.2 | ~41.9 | ~787.4 | ~0.301 |
| LightUNETR-Large | 93.46 | 90.55 | 87.08 | 90.36 | 3.22 | 2.69 | 2.05 | 2.65 | ~15.7 / 3.8 | ~3.9 | ~129.4 | ~0.544 |
| SegMamba | 94.04 | 91.35 | 87.57 | 90.99 | 3.73 | 3.08 | 2.2 | 3 | ~37.2 / 6.9 | ~67.4 | ~1573 | ~0.728 |
| SegMamba-V2 | 93.89 | 91.44 | 88.3 | 91.21 | 3.8 | 2.73 | 2.49 | 3.01 | ~18.0 / 3.0 | ~139.2 | ~1982.5 | ~0.595 |
| SlimUNETR-V2 | 92.93 | 91.74 | 86.2 | 90.29 | 3.49 | 2.54 | 2.43 | 2.82 | ~15.9 / 0.3 | ~23.6 | ~41.9 | ~0.138 |
| SwinUNET | 92.97 | 89.85 | 85.8 | 89.54 | 4.25 | 3.14 | 2.37 | 3.25 | ~9.2 / 3.1 | ~30.6 | ~46.8 | ~0.251 |
| TransUNet | 93.58 | 91.36 | 88.5 | 91.15 | 3.94 | 2.82 | 1.99 | 2.92 | ~13.4 / 4.8 | ~119.0 | ~555 | ~0.620 |
| Umamba | 93.78 | 91.74 | 88.58 | 91.37 | 3.78 | 2.78 | 2.22 | 2.93 | ~18.8 / 3.4 | ~42.2 | ~975.5 | ~0.384 |
| UNETR | 92.97 | 90.18 | 86.67 | 89.94 | 4.18 | 3.85 | 2.96 | 3.66 | ~8.2 / 2.9 | ~130.8 | ~202.8 | ~0.288 |
| UNETR++ | 93.54 | 91.69 | 87.59 | 90.94 | 3.93 | 3.06 | 2.4 | 3.13 | ~6.5 / 1.2 | ~23.5 | ~87.5 | ~0.265 |
| **SparkSeg** | 93.87 | **92.68** | 88.63 | **91.73** | 3.41 | **2.41** | **1.89** | **2.57** | ~11.3 / 2.5 | ~34.8 | ~575.1 | ~0.356 |

### Table 3 | ACDC held-out test set (RV/MYO/LV)

| Model | RV | MYO | LV | Avg. | RV | MYO | LV | Avg. | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nnU-Net | 88.66 | 89.51 | 95.33 | 91.17 | 1.25 | 1.02 | 1.06 | 1.11 | ~6.2 / 0.6 | ~31.2 | ~203.4 | ~0.075 |
| nnU-Net ResEnc | 88.7 | 89.27 | 95.41 | 91.13 | 1.22 | 1.02 | 1.03 | 1.09 | ~14.5 / 1.2 | ~102.4 | ~374.3 | ~0.055 |
| AttUNet | 86.6 | 88.15 | 94.72 | 89.83 | 4.04 | 1.04 | 1.04 | 2.04 | ~13.0 / 0.7 | ~23.6 | ~176.9 | ~0.045 |
| CoTr | 85.24 | 86.68 | 94.07 | 88.66 | 4.64 | 1.09 | 1.09 | 2.27 | ~16.8 / 1.9 | ~41.9 | ~310.7 | ~0.112 |
| SegMamba | 85.79 | 87.95 | 94.39 | 89.37 | 1.73 | 1.14 | 1.07 | 1.31 | ~45.5 / 3.1 | ~67.4 | ~678.8 | ~0.117 |
| SegMamba-V2 | 87.94 | 89.1 | 95.22 | 90.76 | 2.58 | 1.13 | 1.04 | 1.58 | ~26.2 / 1.7 | ~139.2 | ~863.6 | ~0.100 |
| SwinUNET | 80.61 | 82.63 | 92.31 | 85.18 | 4.38 | 1.31 | 1.19 | 2.29 | ~25.5 / 2.3 | ~30.6 | ~40.4 | ~0.212 |
| TransUNet | 88.73 | 89.07 | 95.24 | 91.01 | 1.29 | 1.03 | 1.02 | 1.12 | ~18.1 / 2.6 | ~119.0 | ~232.4 | ~0.161 |
| Umamba | 71.23 | 75.3 | 89.09 | 78.54 | 4.64 | 2.07 | 1.89 | 2.87 | ~18.4 / 3.4 | ~42.2 | ~359.3 | ~0.092 |
| UNETR | 74.98 | 80.84 | 90.84 | 82.22 | 14.31 | 1.62 | 1.55 | 5.83 | ~10.5 / 1.4 | ~130.8 | ~85.4 | ~0.095 |
| UNETR++ | 82.23 | 85.39 | 93.06 | 86.89 | 1.9 | 1.29 | 1.14 | 1.44 | ~12.8 / 1.5 | ~23.5 | ~64.1 | ~0.120 |
| **SparkSeg** | **90.24** | 89.9 | **95.8** | **91.98** | **1.19** | 1.02 | 1.02 | **1.08** | ~15.0 / 1.2 | ~111.1 | ~440.4 | ~0.087 |

### Table 4 | Synapse 12-case held-out test set (eight organs)

| Model | Aor. | Gal. | LKid. | RKid. | Liv. | Pan. | Spl. | Sto. | Avg. | Aor. | Gal. | LKid. | RKid. | Liv. | Pan. | Spl. | Sto. | Avg. | Mem T/I (GB) | Params (M) | FLOPs (G) | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nnU-Net | 92.7 | 66.09 | 87.07 | 87.17 | 96.54 | 74.3 | 92.86 | 78.7 | 84.43 | 3.12 | 25.26 | 17.5 | 7.26 | 4.29 | 10.02 | 8.14 | 20.04 | 11.72 | ~7.9 / 3.1 | ~31.2 | ~619.6 | ~3.290 |
| nnU-Net ResEnc | 93.16 | 70.77 | 85.6 | 86.45 | 96.48 | 75.83 | 93.69 | 82.31 | 85.54 | 1.48 | 14.6 | 18.34 | 8.94 | 7.41 | 7.09 | 21.13 | 17.34 | 11.91 | ~11.7 / 3.5 | ~102.4 | ~916.2 | ~5.785 |
| AttUNet | 92.91 | 68.55 | 83.68 | 84.97 | 96.36 | 69.72 | 91.95 | 76.55 | 83.09 | 4.83 | 23.22 | 7.91 | 24 | 3.19 | 10.44 | 8.13 | 32.88 | 14.11 | ~15.0 / 2.6 | ~23.6 | ~670.4 | ~4.079 |
| CoTr | 93.48 | 67.06 | 86.84 | 86.3 | 96.82 | 78.68 | 92.97 | 80.64 | 85.35 | 1.49 | 18.45 | 15.91 | 10.15 | 2.16 | 6.4 | 6.95 | 31.51 | 11.29 | ~14.5 / 3.6 | ~41.9 | ~865.6 | ~3.818 |
| SegMamba | 91.03 | 61.18 | 85.43 | 86.99 | 95.83 | 68.26 | 91.47 | 73.24 | 81.68 | 8.74 | 31.36 | 17.85 | 6.16 | 15.8 | 9.56 | 29.31 | 33.66 | 18.76 | ~42.3 / 8.7 | ~67.4 | ~2037.2 | ~12.166 |
| SegMamba-V2 | 92.5 | 59.65 | 87.71 | 85.23 | 95.92 | 70.15 | 94.83 | 73.9 | 82.49 | 8.09 | 22.52 | 15.29 | 22.53 | 18.16 | 16.04 | 6.6 | 32.42 | 17.5 | ~23.0 / 3.4 | ~139.2 | ~2591.6 | ~10.181 |
| SwinUNET | 83.3 | 54.93 | 79.2 | 74.86 | 92.85 | 49.66 | 85.29 | 68.21 | 73.54 | 34.78 | 23.13 | 53.22 | 28.03 | 17.86 | 16.43 | 62.78 | 36.69 | 33.92 | ~13.6 / 4.8 | ~30.6 | ~62.0 | ~5.292 |
| TransUNet | 93.18 | 64.04 | 85.65 | 87.03 | 96.57 | 70.69 | 91.25 | 75.81 | 83.03 | 1.87 | 25.85 | 18.73 | 8.88 | 1.76 | 11.04 | 13.3 | 30.17 | 13.7 | ~16.8 / 6.9 | ~119.0 | ~663.8 | ~9.667 |
| Umamba | 91.64 | 58.98 | 80.94 | 77.3 | 96.03 | 60.08 | 87.66 | 58.2 | 76.35 | 11.34 | 23.03 | 4.96 | 10.81 | 4.76 | 28.66 | 11.5 | 24.55 | 14.74 | ~24.1 / 4.7 | ~42.2 | ~1126.8 | ~7.493 |
| UNETR | 84.49 | 63.26 | 78.44 | 78.01 | 93.79 | 51.34 | 80.49 | 56.4 | 73.28 | 10.91 | 18.71 | 26.95 | 46.51 | 42.48 | 23.24 | 38.98 | 37.76 | 30.55 | ~13.0 / 4.2 | ~130.8 | ~256.4 | ~5.476 |
| UNETR++ | 91.78 | 58.16 | 86.61 | 85.45 | 95.84 | 64.81 | 91.19 | 74.73 | 81.07 | 2.43 | 18.04 | 17.31 | 24.51 | 3.5 | 11.42 | 8.97 | 23.78 | 13.61 | ~7.8 / 3.2 | ~23.5 | ~64.5 | ~9.335 |
| **SparkSeg** | 93.25 | **73.66** | 85.54 | **89.93** | 96.97 | 75.2 | 92.49 | 80.45 | **85.94** | 1.67 | **13.23** | 18.6 | **3.87** | 2.12 | 9.66 | 21.57 | 19.86 | **11.21** | ~13.7 / 2.7 | ~111.1 | ~1089.4 | ~7.523 |

Case-level paired Wilcoxon tests (two-sided, Bonferroni-corrected) confirm the ACDC gain
(p < 1e-6; 34/40 wins) and give 10/12 wins on the 12-case Synapse split (p = 0.034, adjusted 0.068);
on BraTS2021 SparkSeg matches nnU-Net per case (p = 0.90; 120/251) while improving the mean by
+0.36 DSC. Against the host-matched controls the same block adds **+0.85 (ACDC)**, **+0.36 (BraTS2021)**
and **+0.40 (Synapse)** mean DSC.

## Figures

### Fig. 2 | Qualitative overlays on BraTS2021, Synapse and ACDC

<p align="center"><img src="assets/paper/fig02_qualitative.png" alt="Fig. 2 — qualitative overlays" width="820"></p>

<p align="center"><sub>Two rows per dataset; the input image is followed by panels that overlay ground truth and predictions. SparkSeg is shown with the configuration reported for each dataset.</sub></p>

### Fig. 3 | Per-case DSC distributions on the three held-out benchmarks

<p align="center"><img src="assets/paper/fig03_per_case_dsc.png" alt="Fig. 3 — per-case DSC distributions" width="820"></p>

<p align="center"><sub>(a) One row per compared method (SparkSeg first), boxes = median/IQR, points = individual held-out cases (n = 40 ACDC, 251 BraTS2021, 12 Synapse), inset = mean HD95. (b) Per-case DSC by anatomical category for SparkSeg and the two matched controls.</sub></p>

### Fig. 4 | Resource profiles of the compared methods

<p align="center"><img src="assets/paper/fig04_resource_profiles.png" alt="Fig. 4 — resource profiles" width="820"></p>

<p align="center"><sub>Peak training (solid) and one-patch inference (faded) memory, FLOPs and whole-case latency per method and dataset, with each method's DSC annotated on the corresponding bars. Lower is better on all resource axes; SparkSeg is hatched.</sub></p>

### Fig. 5 | Prior-guided Top-K anchors on one BraTS case

<p align="center"><img src="assets/paper/fig05_anchor_visualisation.png" alt="Fig. 5 — prior-guided Top-K anchors" width="820"></p>

<p align="center"><sub>Rows are Stages 2–5; column groups are ScoreOnly, Soft-Pr, Lock-Pr (default), PriorOnly and Random, each with axial, coronal and sagittal views. The heat map is the ranking score that drives the hard Top-K selection.</sub></p>

### Fig. 6 | Anchor-selection quality on the BraTS2021 held-out set (n = 251)

<p align="center"><img src="assets/paper/fig06_selection_quality.png" alt="Fig. 6 — anchor-selection quality" width="820"></p>

<p align="center"><sub>Rows: WT/TC/ET; columns: Precision@K, Recall@K, Enrichment and BoundaryHit@K (Stages 2–5). Markers are case-level means, whiskers are normal-approximation 95% CIs. Random is the uniform-K reference (Enrichment ≈ 1).</sub></p>

Figures are the manuscript's Fig. 1–6, downscaled to 2000 px for the repository; the full-resolution
originals ship with the manuscript files.

## Repository layout

```
SparkSeg/
├── SparkSeg.py                 # the network: host + builders (assembly only)
├── blocks.py                   # TAE, PGTS, MSBA, WinMHSA3D, PriorHead, AG, SparkSegBlock, decoder
├── utils.py                    # host geometry, encoder builder, K budget, score utils, inits
├── config.py                   # CLEAN_CONFIG + DATASET_PRESETS + helpers
├── __init__.py                 # public API (__version__ = 1.0.0)
├── LICENSE.txt                 # Apache-2.0
├── META.json                   # provenance/audit metadata
├── docs/ARCHITECTURE.md        # architecture reference (modules, equations, API, ablations)
├── assets/paper/               # manuscript figures Fig. 1–6
└── tests/test_sparkseg.py      # schedule + clean-config drift + presets + forwards
```

## Tests

```bash
python -m pytest nnunetv2/training/network/source_code/SparkSeg/tests/test_sparkseg.py -q
```

The suite checks that the Top-K schedule equals the paper's 128/64/32/32, that `CLEAN_CONFIG` still matches
the training stack (`verify_clean_config()`), that the presets resolve to the reported hosts, that both
hosts build and run train/eval forwards, and that the legacy aliases remain exact aliases.

## Data

Cloud links below provide **nnU-Net–format** dataset packs; official portals are for citation and
licensing. Follow each dataset's original terms of use.

### ACDC

| Type | Link |
|---|---|
| nnU-Net pack (Baidu Netdisk) | [ACDC](https://pan.baidu.com/s/1UpbyOIFCrYgThEsCaDyAWg?pwd=fr7t) · code `fr7t` |
| nnU-Net pack (Aliyun Drive) | [ACDC](https://www.alipan.com/s/EJPiXceGWZV) |
| Official source | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV

| Type | Link |
|---|---|
| nnU-Net pack (Baidu Netdisk) | [Synapse](https://pan.baidu.com/s/1IvX_5Q1h6QeSDa__gjEX_A?pwd=drsm) · code `drsm` |
| Official source (BTCV / Synapse) | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS 2021 Adult Glioma

The paper and experiments use the **BraTS 2021** publicly labelled training cohort (1,251 cases).
BraTS 2022/2023 Adult Glioma redistributed the same cohort and are **not** BraTS 2025 Lighthouse.

| Type | Link |
|---|---|
| nnU-Net pack (Aliyun Drive) | [Dataset1251_BraTS2021GLI](https://www.alipan.com/s/M7cS2KvaAuK) |
| Official source (BraTS 2021) | [Synapse: syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| Same-cohort redistribute page (2023 challenge) | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
| Kaggle mirror (same cohort) | [part-1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) · [part-2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip) |

## Citation

If this work is relevant to your research, please cite the SparkSeg paper (BibTeX / DOI will be added
here upon acceptance).

## License

[Apache License 2.0](LICENSE.txt). Training and inference build on
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet); the public benchmarks remain subject to their own terms.
