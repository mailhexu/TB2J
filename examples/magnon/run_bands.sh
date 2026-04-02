#!/bin/bash
# Magnon band structure calculation with default parameters
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_bands.sh

set -e

# Check if TB2J_results exists
if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    echo "Please run TB2J calculation first or specify path with --path"
    exit 1
fi

# Run magnon band calculation
TB2J_magnon.py --bands \
    --path TB2J_results \
    --kpath GMKG \
    --npoints 200 \
    --output magnon_bands.png \
    --show

echo ""
echo "Band structure saved to: magnon_bands.png"
echo "Data saved to: magnon_bands.json"
