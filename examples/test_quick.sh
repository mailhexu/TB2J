#!/bin/bash
# Quick test for both modes with small k-mesh (for debugging)
# Uses 1x1x1 k-mesh and nz=2 for faster execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REFS_DIR="$PROJECT_DIR/../Refs/DMFT/LaMnO3_DMFT_data"

echo "========================================================================"
echo "Quick test: DMFT --static-sigma mode (1x1x1, nz=2)"
echo "========================================================================"

cd "$PROJECT_DIR"

.venv/bin/python -m TB2J.scripts.dmft2J \
    --dmft_file="$REFS_DIR/sig.inp" \
    --path="$REFS_DIR" \
    --prefix=wannier90 \
    --spin-channels=+1,-1,-1,+1 \
    --magnetic-elements=Mn \
    --nspin=1 \
    --mode=dmft \
    --static-sigma \
    --kmesh 1 1 1 \
    --nz 2 \
    --output_path="$PROJECT_DIR/../.tmp/LaMnO3_dmft_quick"

echo ""
echo "Results:"
grep -A 5 "^Mn" "$PROJECT_DIR/../.tmp/LaMnO3_dmft_quick/exchange.out" | head -8

echo ""
echo "========================================================================"
echo "Quick test: static-hamiltonian mode (1x1x1, nz=2)"
echo "========================================================================"

.venv/bin/python -m TB2J.scripts.dmft2J \
    --dmft_file="$REFS_DIR/sig.inp" \
    --path="$REFS_DIR" \
    --prefix=wannier90 \
    --spin-channels=+1,-1,-1,+1 \
    --magnetic-elements=Mn \
    --mode=static-hamiltonian \
    --kmesh 1 1 1 \
    --nz 2 \
    --output_path="$PROJECT_DIR/../.tmp/LaMnO3_static_quick"

echo ""
echo "Results:"
grep -A 5 "^Mn" "$PROJECT_DIR/../.tmp/LaMnO3_static_quick/exchange.out" | head -8
