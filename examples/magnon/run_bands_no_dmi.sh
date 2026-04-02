#!/bin/bash
# Magnon band structure without DMI and anisotropic exchange (isotropic only)
#
# This example shows how to calculate magnon bands using only the isotropic
# Heisenberg exchange interaction, which is useful for understanding the 
# basic magnetic coupling without spin-orbit effects.
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_bands_no_dmi.sh

set -e

if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    exit 1
fi

TB2J_magnon.py --bands \
    --path TB2J_results \
    --kpath GMKG \
    --npoints 200 \
    --no-DMI \
    --no-Jani \
    --output magnon_bands_no_dmi.png \
    --show

echo ""
echo "Band structure (isotropic only) saved to: magnon_bands_no_dmi.png"
