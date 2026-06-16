#!/bin/bash
# Magnon band structure calculation with custom q-points
#
# This example shows how to specify custom q-points from the command line.
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_bands_custom_qpoints.sh

set -e

# Check if TB2J_results exists
if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    echo "Please run TB2J calculation first or specify path with --path"
    exit 1
fi

# Run magnon band calculation with custom q-points
# Format: --qpoints "name1:x1,y1,z1,name2:x2,y2,z2,..."
TB2J_magnon.py --bands \
    --path TB2J_results \
    --kpath GXMRG \
    --npoints 200 \
    --qpoints "G:0,0,0,X:0.5,0,0,M:0.5,0.5,0,R:0.5,0.5,0.5" \
    --output magnon_bands_custom.png \
    --show

echo ""
echo "Band structure saved to: magnon_bands_custom.png"
echo "Data saved to: magnon_bands_custom.json"
