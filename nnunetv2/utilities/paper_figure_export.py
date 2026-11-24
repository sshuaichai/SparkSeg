# =============================================================================
# 论文用栅格图导出：PNG 写入 DPI 元数据，并对过小图做高质量放大以满足印刷像素数。
# =============================================================================
from __future__ import annotations

from typing import Optional

import numpy as np

# 常见期刊：栅格图 300 DPI；单栏宽约 3.3–3.5 inch → 约 1000–1050 px；此处默认长边至少 1800 px 留余量。
DEFAULT_EXPORT_DPI = 300
DEFAULT_MIN_EDGE = 1800


def prepare_rgb_uint8_for_paper(rgb: np.ndarray, min_edge: int) -> np.ndarray:
    """
    uint8 H×W×3；若 max(H,W) < min_edge，则按比例 LANCZOS 放大（min_edge<=0 时不放大）。
    """
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if min_edge <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if max(h, w) >= min_edge:
        return rgb
    scale = min_edge / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS))
    except Exception:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None, ...]
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        return (t[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


def save_raster_png_paper(
    path: str,
    rgb: np.ndarray,
    *,
    dpi: int = DEFAULT_EXPORT_DPI,
    min_edge: int = DEFAULT_MIN_EDGE,
) -> None:
    """保存 RGB PNG，写入 dpi 元数据；可选将长边放大至至少 min_edge 像素。"""
    out = prepare_rgb_uint8_for_paper(rgb, min_edge)
    try:
        from PIL import Image

        Image.fromarray(out).save(path, dpi=(dpi, dpi), compress_level=3)
    except Exception:
        import matplotlib.pyplot as plt

        plt.imsave(path, out)
