#!/bin/bash
# Magnon DOS calculation
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_dos.sh

set -e

if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    exit 1
fi

TB2J_magnon.py --dos \
    --path TB2J_results \
    --kmesh 20 20 20 \
    --width 0.001 \
    --npts 401 \
    --no-DMI \
    --no-Jani \
    --output magnon_dos.png \
    --show

echo ""
echo "DOS saved to: magnon_dos.png"
echo "Data saved to: magnon_dos.json"
