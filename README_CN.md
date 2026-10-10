# ✨ SparkSeg

**面向三维 CT / MRI 的稀疏先验引导锚点注意力分割网络。** - [English](README.md)

<p align="center"><img src="assets/paper/fig01_overview.png" alt="图 1 — SparkSeg 框架" width="900"></p>

```
输入
 └─ Stage 1  TAE                                           → skip 1
     └─ Stage 2  TAE ∥ WinMHSA3D ∥ PGTS → MSBA             → skip 2
         └─ Stage 3  同上                                  → skip 3
             └─ Stage 4  同上                              → skip 4
                 └─ Stage 5  同上                          → skip 5
                     └─ Stage 6  CNN + PriorHead
                         └─ 解码器，每级一个 AG → 深监督 logits

hist = 上一级输出，在 Stage 2–5 作为额外注意力源复用
```

## 📊 结果

| 数据集 | 模型 | Mean DSC (%) | Mean HD95 (mm) |
|---|---|---|---|
| ACDC | nnU-Net *（对照）* | 91.17 | 1.11 |
| ACDC | nnU-Net ResEnc *（对照）* | 91.13 | 1.09 |
| ACDC | TransUNet | 91.01 | 1.12 |
| ACDC | **SparkSeg** | **91.98** | **1.08** |
| Synapse/BTCV | nnU-Net *（对照）* | 84.43 | 11.72 |
| Synapse/BTCV | nnU-Net ResEnc *（对照）* | 85.54 | 11.91 |
| Synapse/BTCV | CoTr | 85.35 | 11.29 |
| Synapse/BTCV | **SparkSeg** | **85.94** | **11.21** |
| BraTS2021 | nnU-Net *（对照）* | 91.36 | 2.86 |
| BraTS2021 | nnU-Net ResEnc *（对照）* | 91.51 | 2.60 |
| BraTS2021 | U-Mamba | 91.37 | 2.93 |
| BraTS2021 | **SparkSeg** | **91.73** | **2.57** |

| SparkSeg 运行 | 主干 | 训练轮次 | 初始化 | 训练显存 | 参数量 | FLOPs | 延迟 |
|---|---|---|---|---|---|---|---|
| ACDC | residual encoder | 200 | `no_stock_he` | 15.0 GB | 111.1 M | 440.4 G | 0.087 s |
| Synapse/BTCV | residual encoder | 1,000 | `stock_he` | 13.7 GB | 111.1 M | 1,089.4 G | 7.523 s |
| BraTS2021 | plain U-Net | 1,000 | `no_stock_he` | 11.3 GB | 34.8 M | 575.1 G | 0.356 s |

```
主干匹配对照 → +0.85 ACDC - +0.36 BraTS2021 - +0.40 Synapse（mean DSC）
逐例 Wilcoxon → ACDC p<1e-6 (34/40) - Synapse 10/12 (p=0.034, 校正 0.068) - BraTS p=0.90 (120/251)
BraTS2021 vs SegMamba → FLOPs 少 2.7x，显存少 3.3x，mean DSC 更高
分类别 / 全基线表 → 稿件 Tables 2–4
```

## 🚀 快速开始

```python
from SparkSeg import build_sparkseg, describe_preset, DATASET_PRESETS

strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]   # 取自 host plans
net = build_sparkseg(in_channels=4, out_channels=4, plan_strides=strides, preset="brats")

print(describe_preset("brats"))
# brats (Dataset1251) host=plainconv patch=(128,128,128) batch=2 epochs=1000
# init=no_stock_he K=(128,64,32,32) | reported: mean_dsc=91.73, mean_hd95=2.57, ...
```

```
preset         → "acdc" | "synapse" | "brats"（或 100 | 180 | 1251、或文件夹名）
build_sparkseg → 只建主机；在你的 trainer 的 build_network_architecture 里调用
train / eval   → (seg, prior) 且 net._last_stage_p_effs  |  仅 seg（可直接滑窗推理）
```

## ⚙️ 配置

```
阶段挂载   1 只挂 TAE - 2–5 各一个 SparkSeg Block - 6 CNN + PriorHead - 解码器每级一个 AG
锚点预算   硬 Top-K 128/64/32/32          （式 4–6：α=8、K_min=32、K_max=512、S=6）
窗注意力   4x4x4、4 heads、半窗位移（shift = 2）
门控       σ_ba / σ_win 起始关闭（bias −2.5）；AG 起始近恒等（bias +4）
损失       L_seg + 0.05*L_prior + 0.02*L_prioraux     （式 8）
优化器     SGD 1e-2、Nesterov 0.99、wd 3e-5、多项式衰减；250 训练 / 50 验证迭代
```

`CLEAN_CONFIG` 保存以上取值；`verify_clean_config()` 返回 `{"checked": True, "mismatch": {}}`。

## 🗂️ 逐数据集预设（稿件 Table 9）

| 数据集 | 主干 | 初始化 | Batch | Patch | Epochs | Mean DSC（plain / residual 主干） |
|---|---|---|---|---|---|---|
| BraTS2021 | plain U-Net | `no_stock_he` | 2 | 128x128x128 | 1,000 | 91.73 / 91.50 |
| ACDC | residual encoder | `no_stock_he` | 7 | 224x256x10 | 200 | 91.18 / 91.98 |
| Synapse | residual encoder | `stock_he` | 2 | 56x192x224 | 1,000 | 85.73 / 85.94 |

```
init = 报告那次训练的初始化：stock_he = He(1e-2) + 末 BN 归零 - no_stock_he = 跳过
build_sparkseg(preset=...) 自动采用；stock_he=True/False 可覆盖
```

## 🖼️ 配图

点小图 → 打开全分辨率原图（浏览器内可继续放大缩小）。

<table>
  <tr>
    <td align="center"><a href="assets/paper/fig02_qualitative.png"><img src="assets/paper/fig02_qualitative.png" alt="图 2" width="300"></a>
    <br><sub><strong>图 2</strong> — 定性对比：BraTS2021 / Synapse / ACDC</sub></td>
    <td align="center"><a href="assets/paper/fig03_per_case_dsc.png"><img src="assets/paper/fig03_per_case_dsc.png" alt="图 3" width="300"></a>
    <br><sub><strong>图 3</strong> — held-out 逐例 DSC（n = 40 / 251 / 12）+ 分解剖类别面板</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/paper/fig04_resource_profiles.png"><img src="assets/paper/fig04_resource_profiles.png" alt="图 4" width="300"></a>
    <br><sub><strong>图 4</strong> — 各方法在各数据集上的显存、FLOPs、延迟</sub></td>
    <td align="center"><a href="assets/paper/fig05_anchor_visualisation.png"><img src="assets/paper/fig05_anchor_visualisation.png" alt="图 5" width="300"></a>
    <br><sub><strong>图 5</strong> — 单个 BraTS 病例的 Top-K 锚点，Stage 2–5</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/paper/fig06_selection_quality.png"><img src="assets/paper/fig06_selection_quality.png" alt="图 6" width="300"></a>
    <br><sub><strong>图 6</strong> — Precision@K、Recall@K、Enrichment、BoundaryHit@K（n = 251）</sub></td>
    <td></td>
  </tr>
</table>

## 📁 目录

```
SparkSeg/
├── SparkSeg.py     网络本体：主机 + 构建函数（只做装配）
├── blocks.py       TAE - PGTS - MSBA - WinMHSA3D - PriorHead - AttentionGate - SparkSegBlock - 解码器
├── utils.py        宿主几何、encoder 构建、K 预算、打分工具、初始化
├── config.py       CLEAN_CONFIG - DATASET_PRESETS - 辅助函数
├── assets/paper/   图 1–6
└── tests/          test_sparkseg.py
```

## 📥 数据

nnU-Net 格式数据包；官方源站用于引用与授权。请遵守各数据集条款。

| 数据集 | nnU-Net 包 | 官方源 |
|---|---|---|
| ACDC | [百度](https://pan.baidu.com/s/1U_HlzeetW2kNKHKFC7-OvA?pwd=yiyk) `yiyk` | [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| Synapse / BTCV | [百度](https://pan.baidu.com/s/1RsphDHFFMrAdtFDjgfaTEg?pwd=4463) `4463` | [Synapse syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| BraTS 2021 | [百度](https://pan.baidu.com/s/1gyC9G7RnbavCt9jmeN8FfQ?pwd=mna8) `mna8` | [Synapse syn25829067](https://www.synapse.org/Synapse:syn25829067) |

BraTS 2021（1,251 例）——2022/2023 Adult Glioma 为同一队列的再分发。

## 📜 引用 - License

引用 SparkSeg 论文（BibTeX / DOI 接受后补充）。[Apache-2.0](LICENSE.txt)；训练与推理基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)。
