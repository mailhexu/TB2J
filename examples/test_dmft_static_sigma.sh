#!/bin/bash
# Test DMFT --static-sigma mode for LaMnO3
# Uses static self-energy Σ(∞)-Vdc from sig.inp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REFS_DIR="$PROJECT_DIR/../Refs/DMFT/LaMnO3_DMFT_data"
OUTPUT_DIR="$PROJECT_DIR/LaMnO3_dmft_static_sigma_test"

echo "========================================================================"
echo "Running DMFT --static-sigma mode for LaMnO3"
echo "========================================================================"
echo ""
echo "Input:"
echo "  sig.inp: $REFS_DIR/sig.inp"
echo "  Wannier90 files: $REFS_DIR/wannier90_*"
echo ""
echo "Output:"
echo "  $OUTPUT_DIR"
echo ""

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
    --kmesh 5 5 5 \
    --nz 40 \
    --output_path="$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Results:"
echo "========================================================================"
echo ""
echo "Expected (from target):"
echo "  Mn1: charge=4.8622, spin=-3.9596 μB"
echo "  Mn2: charge=4.8622, spin=+3.9595 μB"
echo "  Mn3: charge=4.8622, spin=+3.9596 μB"
echo "  Mn4: charge=4.8622, spin=-3.9596 μB"
echo ""
echo "Actual:"
grep -A 5 "^Mn" "$OUTPUT_DIR/exchange.out" | head -8
echo ""
echo "J (in-plane):"
grep "J_iso:" "$OUTPUT_DIR/exchange.out" | head -5
