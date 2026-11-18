#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}/../.."

python -m TB2J.magnon.magnon_cli \
    --bands \
    --path TB2J_results \
    --kpath GMKG \
    --band-output magnon_sc_bz.png
