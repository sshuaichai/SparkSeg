#!/usr/bin/env bash
# AutoDL / existing nnUNetv2 env — do NOT recreate conda or pip-install nnunet v1.
# Official Beckschen/3D-TransUNet is vendored at:
#   nnunetv2/training/network/3D-TransUNet/
# Used by: nnUNetTrainerTransUNet3D* → TransUNet3D.py → Generic_TransUNet_max_ppbp
#
# Hard deps for our wrapper (decoder_only default; is_masked_attn=True):
#   fvcore, ml_collections, einops, pyyaml, scipy
# detectron2 is NOT required (point_sample_3d is local; d2util is vendored).
set -euo pipefail

NNUNET_ROOT="${NNUNET_ROOT:-/root/nnUnetv2}"
VENDOR="$NNUNET_ROOT/nnunetv2/training/network/3D-TransUNet"

echo "[3D-TransUNet] pip deps..."
pip install -q fvcore ml_collections einops pyyaml scipy

if [[ ! -f "$VENDOR/nn_transunet/networks/transunet3d_model.py" ]]; then
  echo "[3D-TransUNet] vendored tree missing — download zip (ghfast first)..."
  TMP="${TMPDIR:-/tmp}/3D-TransUNet-main.zip"
  DEST="$(dirname "$VENDOR")"
  curl -L --fail -o "$TMP" \
    "https://ghfast.top/https://github.com/Beckschen/3D-TransUNet/archive/refs/heads/main.zip" \
    || curl -L --fail -o "$TMP" \
    "https://github.com/Beckschen/3D-TransUNet/archive/refs/heads/main.zip"
  unzip -qo "$TMP" -d "$DEST"
  rm -rf "$VENDOR"
  mv "$DEST/3D-TransUNet-main" "$VENDOR"
fi

export PYTHONPATH="${NNUNET_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
echo "[3D-TransUNet] smoke import (decoder_only)..."
python - <<'PY'
from nnunetv2.training.network.TransUNet3D import build_official_transunet3d
import torch

m = build_official_transunet3d(
    variant="decoder_only",
    num_input_channels=1,
    num_classes=9,
    patch_size=(64, 128, 128),
    pool_op_kernel_sizes=[(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)],
    conv_kernel_sizes=[(3, 3, 3)] * 6,
    base_num_features=32,
)
x = torch.randn(1, 1, 64, 128, 128)
with torch.no_grad():
    y = m(x)
keys = sorted(y.keys()) if isinstance(y, dict) else type(y)
print("ok", type(m).__name__, keys)
PY

echo "[3D-TransUNet] done."
echo "  nnUNetv2_train 180 3d_fullres 0 -tr nnUNetTrainerTransUNet3D"
