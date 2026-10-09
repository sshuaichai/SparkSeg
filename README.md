# ✨ SparkSeg

**Sparse prior-guided anchor attention for 3D CT and MRI segmentation.** · [Chinese](README_CN.md)

<p align="center"><img src="assets/paper/fig01_overview.png" alt="Fig. 1 — SparkSeg framework" width="900"></p>

```
Input
 └─ Stage 1  TAE                                           → skip 1
     └─ Stage 2  TAE ∥ WinMHSA3D ∥ PGTS → MSBA             → skip 2
         └─ Stage 3  same                                  → skip 3
             └─ Stage 4  same                              → skip 4
                 └─ Stage 5  same                          → skip 5
                     └─ Stage 6  CNN + PriorHead
                         └─ Decoder, one AG per level → deep-supervised logits

hist = previous stage output, reused as an extra attention source at Stages 2–5
```

## 📊 Results

| Dataset | Host | Epochs | Init | Mean DSC (%) | Mean HD95 (mm) | Train mem | Params | FLOPs | Latency |
|---|---|---|---|---|---|---|---|---|
| ACDC | residual encoder | 200 | `no_stock_he` | **91.98** | **1.08** | 15.0 GB | 111.1 M | 440.4 G | 0.087 s |
| Synapse/BTCV | residual encoder | 1,000 | `stock_he` | **85.94** | **11.21** | 13.7 GB | 111.1 M | 1,089.4 G | 7.523 s |
| BraTS2021 | plain U-Net | 1,000 | `no_stock_he` | **91.73** | **2.57** | 11.3 GB | 34.8 M | 575.1 G | 0.356 s |

```
host-matched control → +0.85 ACDC · +0.36 BraTS2021 · +0.40 Synapse (mean DSC)
per-case Wilcoxon    → ACDC p<1e-6 (34/40) · Synapse 10/12 (p=0.034, adj. 0.068) · BraTS p=0.90 (120/251)
BraTS2021 vs SegMamba→ 2.7× fewer FLOPs, 3.3× less memory, higher mean DSC
per-class / full baselines → manuscript Tables 2–4
```

## 🚀 Quickstart

```python
from SparkSeg import build_sparkseg, describe_preset, DATASET_PRESETS

strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]   # from the host plans
net = build_sparkseg(in_channels=4, out_channels=4, plan_strides=strides, preset="brats")

print(describe_preset("brats"))
# brats (Dataset1251) host=plainconv patch=(128,128,128) batch=2 epochs=1000
# init=no_stock_he K=(128,64,32,32) | reported: mean_dsc=91.73, mean_hd95=2.57, ...
```

```
preset            → "acdc" | "synapse" | "brats" (or 100 | 180 | 1251, or the folder name)
build_sparkseg    → host only; wire it into your own trainer's build_network_architecture
train / eval      → (seg, prior) with net._last_stage_p_effs  |  seg (sliding window ready)
```

## ⚙️ Configuration

```
Stage mount     1 TAE · 2–5 one SparkSeg Block each · 6 CNN + PriorHead · decoder AG per level
Anchor budget   hard Top-K 128/64/32/32          (Eq. 4–6: α=8, K_min=32, K_max=512, S=6)
Windows         4×4×4, 4 heads, half-window shift (shift = 2)
Gates           σ_ba / σ_win start closed (bias −2.5); AG starts near identity (bias +4)
Losses          L_seg + 0.05·L_prior + 0.02·L_prioraux     (Eq. 8)
Optimiser       SGD 1e-2, Nesterov 0.99, wd 3e-5, polynomial; 250 train / 50 val iterations
```

`CLEAN_CONFIG` holds these values; `verify_clean_config()` returns `{"checked": True, "mismatch": {}}`.

## 🗂️ Per-dataset presets (manuscript Table 9)

| Dataset | Host | Init | Batch | Patch | Epochs | Mean DSC (plain / residual host) |
|---|---|---|---|---|---|---|
| BraTS2021 | plain U-Net | `no_stock_he` | 2 | 128×128×128 | 1,000 | 91.73 / 91.50 |
| ACDC | residual encoder | `no_stock_he` | 7 | 224×256×10 | 200 | 91.18 / 91.98 |
| Synapse | residual encoder | `stock_he` | 2 | 56×192×224 | 1,000 | 85.73 / 85.94 |

```
init = initialization of the reported run:  stock_he = He(1e-2) + zero-last-BN · no_stock_he = skipped
build_sparkseg(preset=...) applies it;  stock_he=True/False overrides
```

## 🖼️ Figures

Click a thumbnail to open the figure at full resolution (browser zoom works there).

<table>
  <tr>
    <td align="center"><a href="assets/paper/fig02_qualitative.png"><img src="assets/paper/fig02_qualitative.png" alt="Fig. 2" width="300"></a>
    <br><sub><strong>Fig. 2</strong> — qualitative overlays: BraTS2021 / Synapse / ACDC</sub></td>
    <td align="center"><a href="assets/paper/fig03_per_case_dsc.png"><img src="assets/paper/fig03_per_case_dsc.png" alt="Fig. 3" width="300"></a>
    <br><sub><strong>Fig. 3</strong> — per-case DSC (n = 40 / 251 / 12) + per-category panels</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/paper/fig04_resource_profiles.png"><img src="assets/paper/fig04_resource_profiles.png" alt="Fig. 4" width="300"></a>
    <br><sub><strong>Fig. 4</strong> — memory, FLOPs, latency per method per dataset</sub></td>
    <td align="center"><a href="assets/paper/fig05_anchor_visualisation.png"><img src="assets/paper/fig05_anchor_visualisation.png" alt="Fig. 5" width="300"></a>
    <br><sub><strong>Fig. 5</strong> — Top-K anchors on one BraTS case, Stages 2–5</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/paper/fig06_selection_quality.png"><img src="assets/paper/fig06_selection_quality.png" alt="Fig. 6" width="300"></a>
    <br><sub><strong>Fig. 6</strong> — Precision@K, Recall@K, Enrichment, BoundaryHit@K (n = 251)</sub></td>
    <td></td>
  </tr>
</table>

## 📁 Layout

```
SparkSeg/
├── SparkSeg.py     network: host + builders (assembly only)
├── blocks.py       TAE · PGTS · MSBA · WinMHSA3D · PriorHead · AttentionGate · SparkSegBlock · decoder
├── utils.py        host geometry, encoder builder, K budget, score utils, inits
├── config.py       CLEAN_CONFIG · DATASET_PRESETS · helpers
├── assets/paper/   Fig. 1–6
└── tests/          test_sparkseg.py
```

## 📥 Data

nnU-Net-format packs; official portals for citation and licensing. Follow each dataset's terms.

| Dataset | nnU-Net pack | Official source |
|---|---|---|
| ACDC | [Baidu](https://pan.baidu.com/s/1UpbyOIFCrYgThEsCaDyAWg?pwd=fr7t) `fr7t` · [Aliyun](https://www.alipan.com/s/EJPiXceGWZV) | [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| Synapse / BTCV | [Baidu](https://pan.baidu.com/s/1IvX_5Q1h6QeSDa__gjEX_A?pwd=drsm) `drsm` | [Synapse syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| BraTS 2021 | [Aliyun](https://www.alipan.com/s/M7cS2KvaAuK) | [Synapse syn25829067](https://www.synapse.org/Synapse:syn25829067) |

BraTS 2021 (1,251 cases) — the 2022/2023 Adult Glioma releases redistribute the same cohort.

## 📜 Citation · License

Cite the SparkSeg paper (BibTeX / DOI on acceptance). [Apache-2.0](LICENSE.txt); training and inference build on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
