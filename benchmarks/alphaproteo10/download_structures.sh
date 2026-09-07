#!/usr/bin/env bash
# Fetch the ten AlphaProteo binder-design target structures from the RCSB.
#
# PDB IDs come from AlphaProteo Table S1 (arXiv:2409.08022, p38). Nothing here
# is derived or guessed -- if a download 404s, the ID is wrong in the configs
# too and both should be fixed together.
#
#   ./benchmarks/alphaproteo10/download_structures.sh
#
# mmCIF rather than PDB on purpose: several of these targets are multi-chain
# with chain IDs the legacy PDB format cannot always round-trip (TrkA's chain
# X, VEGF-A's V/W), and PXDesign's own example ships a .cif.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/structures}"
mkdir -p "$DEST"

# id     target      (kept alongside so a failure names something recognisable)
IDS=(
  "2wh6   BHRF1"
  "6m0j   SC2RBD"
  "3di3   IL-7RA"
  "5o45   PD-L1"
  "1www   TrkA"
  "4zxb   Insulin"
  "5vli   H1"
  "1bj1   VEGF-A"
  "4hsa   IL-17A"
  "1tnf   TNFa"
)

fail=0
for row in "${IDS[@]}"; do
  read -r id target <<<"$row"
  out="$DEST/${id}.cif"
  if [[ -s "$out" ]]; then
    printf '  %-6s %-9s already present\n' "$id" "$target"
    continue
  fi
  if curl -fsSL --max-time 120 "https://files.rcsb.org/download/${id}.cif" -o "$out"; then
    printf '  %-6s %-9s %s\n' "$id" "$target" "$(wc -c <"$out" | tr -d ' ') bytes"
  else
    printf '  %-6s %-9s DOWNLOAD FAILED\n' "$id" "$target" >&2
    rm -f "$out"
    fail=1
  fi
done

if (( fail )); then
  echo "one or more downloads failed; see above" >&2
  exit 1
fi
echo "all ten structures in $DEST"
