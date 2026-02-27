#!/bin/bash
# Magnon band structure with custom spin configuration via CLI
#
# This example shows how to specify magnetic moments directly via CLI.
# Values are in μB (Bohr magneton). For spin S, use m = 2S μB.
#
# Example: Two spins of 3 μB along +z (S=3/2 each)
#   --spin-conf 0 0 3 0 0 3
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_with_spin_conf.sh

set -e

if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    exit 1
fi

# Two spins: 3 μB along +z each (S=3/2)
TB2J_magnon.py --bands \
    --path TB2J_results \
    --kpath GMKG \
    --npoints 200 \
    --no-DMI \
    --no-Jani \
    --spin-conf 0 0 3 0 0 3 \
    --output magnon_bands_spin_conf.png \
    --show

echo ""
echo "Band structure with custom spin config saved to: magnon_bands_spin_conf.png"
