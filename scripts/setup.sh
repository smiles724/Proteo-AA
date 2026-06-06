#!/bin/bash
# Set up PXDesign-train: init submodules + apply PXDesign patch.
#
# Layout after this script (Protenix and PXDesign live inside PXDesign-train
# as git submodules pinned to known-good commits):
#
#   PXDesign-train/
#   ├── Protenix/         (bytedance/Protenix @ c3bfc36, v2.0.0)        — submodule
#   ├── PXDesign/         (bytedance/PXDesign @ f78844 + embedders patch) — submodule + patch
#   ├── pxdesign_train/   (this package)
#   ├── patches/
#   ├── scripts/
#   └── tests/
#
# Usage:
#   git clone --recursive https://github.com/guanlueli/PXDesign-train.git
#   cd PXDesign-train
#   bash scripts/setup.sh
#
# If you forgot --recursive on the clone:
#   git submodule update --init --recursive
#   bash scripts/setup.sh
#
# Then run the smoke test:
#   LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." \
#     python scripts/smoke_test_gpu.py \
#       --cif PXDesign/examples/5o45.cif --binder_chain B --crop_size 200 --device cuda

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$ROOT"

# ---- initialize submodules if needed ----
if [ ! -f "Protenix/protenix/version.py" ] || [ ! -f "PXDesign/setup.py" ]; then
    echo "[submodule] initializing"
    git submodule update --init --recursive
fi

# ---- apply PXDesign patches ----
echo "[patch] applying patches/*.patch to PXDesign submodule"
for p in "$ROOT"/patches/*.patch; do
    [ -e "$p" ] || continue
    if (cd "$ROOT/PXDesign" && git apply --check "$p" 2>/dev/null); then
        (cd "$ROOT/PXDesign" && git apply "$p")
        echo "  applied: $(basename "$p")"
    elif (cd "$ROOT/PXDesign" && git apply --check --reverse "$p" 2>/dev/null); then
        echo "  already applied: $(basename "$p")"
    else
        echo "  WARNING: cannot apply $(basename "$p") (manual fix needed)"
    fi
done

echo ""
echo "Setup complete. Try:"
echo "  cd $ROOT"
echo "  LAYERNORM_TYPE=torch PYTHONPATH=\"Protenix:PXDesign:.\" \\"
echo "    python scripts/smoke_test_gpu.py \\"
echo "      --cif PXDesign/examples/5o45.cif --binder_chain B --crop_size 200 --device cuda"
