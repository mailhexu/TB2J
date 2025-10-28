#!/bin/bash

# Bash script examples for running spin-phonon coupling calculations
# Using the path: /home_phythema/hexu/spinphon/2025-10-02_newdata/k555q555

PROJECT_PATH="/home_phythema/hexu/spinphon/2025-10-02_newdata/k555q555"
PYTHON_MODULE="TB2J.Oiju_epw2"

# Example 1: Single displacement mode using CLI
python -m $PYTHON_MODULE \
    --path "$PROJECT_PATH" \
    --idisp 3 \
    --rcut 8.0 \
    --efermi 10.67 \
    --magnetic-elements Mn \
    --kmesh 5 5 5 \
    --output-path "results_single_mode"

# Example 2: Multiple displacement modes using CLI
python -m $PYTHON_MODULE \
    --path "$PROJECT_PATH" \
    --idisp 3 6 7 10 12 \
    --rcut 8.0 \
    --efermi 10.67 \
    --magnetic-elements Mn \
    --kmesh 5 5 5 \
    --output-path "results_multiple_modes"

# Example 3: Using TOML configuration file
cat > calc_config.toml << EOF
path = "$PROJECT_PATH"
colinear = true
posfile = "POSCAR"
prefix_up = "wannier90.up"
prefix_dn = "wannier90.dn"
epw_path = "$PROJECT_PATH"
epw_prefix_up = "system_up"
epw_prefix_dn = "system_dn"
Ru = [0, 0, 0]
Rcut = 8.0
efermi = 10.67
magnetic_elements = ["Mn"]
kmesh = [5, 5, 5]
emin = -7.33
emax = 0.0
nz = 50
np = 1
description = "TOML configuration example"
output_path = "results_toml_config"
idisp_list = [3, 6, 7, 10, 12]
EOF

python -m $PYTHON_MODULE --config calc_config.toml

# Example 4: Single displacement mode with TOML
cat > single_mode_config.toml << EOF
path = "$PROJECT_PATH"
colinear = true
posfile = "POSCAR"
prefix_up = "wannier90.up"
prefix_dn = "wannier90.dn"
epw_path = "$PROJECT_PATH"
epw_prefix_up = "system_up"
epw_prefix_dn = "system_dn"
Ru = [0, 0, 0]
Rcut = 8.0
efermi = 10.67
magnetic_elements = ["Mn"]
kmesh = [5, 5, 5]
emin = -7.33
emax = 0.0
nz = 50
np = 1
description = "Single mode TOML example"
output_path = "results_single_toml"
idisp = 6
EOF

python -m $PYTHON_MODULE --config single_mode_config.toml

echo "All calculations completed!"