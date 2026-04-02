#!/bin/bash
# Magnon band structure using TOML configuration file
#
# Usage:
#   cd TB2J
#   source .venv/bin/activate
#   ./examples/magnon/run_with_config.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "TB2J_results" ]; then
    echo "Error: TB2J_results directory not found"
    exit 1
fi

TB2J_magnon.py --bands \
    --config "$SCRIPT_DIR/config.toml" \
    --show

echo ""
echo "Band structure saved to: magnon_bands.png"
