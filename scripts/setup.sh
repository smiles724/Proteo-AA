#!/bin/bash
# Initialize submodules and apply the one permitted source patch.
#
# Layout after this script:
#
#   Proteo-AA-pxdesign-fampnn-pack/
#   ├── PXDesign/   bytedance/PXDesign  @ f788441 + embedders patch  (backbone)
#   ├── Protenix/   bytedance/Protenix  @ c3bfc36                    (backbone deps)
#   ├── fampnn/     richardshuai/fampnn @ aaf788b, pristine          (side chain)
#   │               its weights/ ship in-repo -- no separate download
#   └── pxf/        this package
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f "PXDesign/setup.py" ] || [ ! -f "Protenix/protenix/version.py" ] \
   || [ ! -f "fampnn/setup.py" ]; then
    echo "[submodule] initializing"
    git submodule update --init --recursive
fi

echo "[patch] applying patches/*.patch to PXDesign"
for patch in "$ROOT"/patches/*.patch; do
    [ -e "$patch" ] || continue
    if (cd "$ROOT/PXDesign" && git apply --check "$patch" 2>/dev/null); then
        (cd "$ROOT/PXDesign" && git apply "$patch")
        echo "  applied: $(basename "$patch")"
    elif (cd "$ROOT/PXDesign" && git apply --check --reverse "$patch" 2>/dev/null); then
        echo "  already applied: $(basename "$patch")"
    else
        echo "  WARNING: cannot apply $(basename "$patch")"
    fi
done

echo ""
echo "[check] pinned revisions and weights"
PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn" python -c "
from pxf import provenance
for name, record in provenance.runtime_sources().items():
    print(f'  {name:9s} {record[\"revision\"][:10]}  patched={record[\"patched\"]}')
print(f'  fampnn weights: {provenance.fampnn_checkpoint()}')
"

echo ""
echo "Next:"
echo "  PYTHONPATH=\"\$PWD:\$PWD/PXDesign:\$PWD/Protenix:\$PWD/fampnn\" python -m pytest tests/ -q"
echo "  python scripts/pack.py --pdb-dir fampnn/data/casp15/pdbs --out packed/"
