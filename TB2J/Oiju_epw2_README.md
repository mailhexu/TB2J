# Oiju_epw2.py: Spin-Phonon Coupling with Multiple Interfaces

This module provides functionality to calculate spin-phonon coupling parameters using EPW (Electron-Phonon Wannier) data. It combines Wannier90 electronic structure data with EPW electron-phonon matrix elements to compute the coupling between spin and lattice degrees of freedom.

## New Features

This enhanced version includes three interfaces for running calculations:

1. **Multiple idisp support**: Function to handle multiple displacement patterns
2. **Command Line Interface (CLI)**: Direct command-line usage
3. **TOML configuration file**: Batch processing with configuration files

## Usage

### 1. Multiple IDisp Function

Use the `gen_exchange_Oiju_epw_multiple()` function to calculate multiple displacement patterns:

```python
from TB2J.Oiju_epw2 import gen_exchange_Oiju_epw_multiple

# Calculate for specific idisp values
gen_exchange_Oiju_epw_multiple(
    path="./wannier_calc",
    idisp_list=[3, 6, 7, 10],
    Rcut=8.0,
    efermi=10.67,
    magnetic_elements=["Mn"],
    output_path="results"
)
```

### 2. Command Line Interface

Run calculations directly from the command line:

```bash
# Single idisp
python -m TB2J.Oiju_epw2 --path ./wannier_calc --idisp 3 --rcut 8.0

# Multiple idisp values
python -m TB2J.Oiju_epw2 --path ./wannier_calc --idisp 3 6 7 --rcut 8.0

# With custom parameters
python -m TB2J.Oiju_epw2 \
    --path ~/projects/wannier90 \
    --posfile POSCAR \
    --epw-path ~/projects/epw \
    --idisp 3 6 7 10 12 \
    --rcut 8.0 \
    --efermi 10.67 \
    --magnetic-elements Mn Fe \
    --kmesh 5 5 5 \
    --output-path my_results
```

#### CLI Options

- `--config, -c`: Path to TOML configuration file
- `--path, -p`: Path to Wannier90 calculation directory
- `--posfile`: Structure file name (default: POSCAR)
- `--prefix-up`: Spin-up Wannier90 prefix (default: wannier90.up)
- `--prefix-dn`: Spin-down Wannier90 prefix (default: wannier90.dn)
- `--epw-path`: EPW calculation directory path
- `--epw-prefix-up`: Spin-up EPW prefix (default: SrMnO3_up)
- `--epw-prefix-dn`: Spin-down EPW prefix (default: SrMnO3_dn)
- `--idisp`: One or more displacement mode indices
- `--rcut`: Interaction cutoff radius
- `--efermi`: Fermi energy in eV (default: 3.0)
- `--magnetic-elements`: List of magnetic elements
- `--kmesh`: k-point mesh dimensions (default: 5 5 5)
- `--emin`: Minimum energy for integration (default: -12.0)
- `--emax`: Maximum energy for integration (default: 0.0)
- `--nz`: Number of energy points (default: 50)
- `--np`: Number of processors (default: 1)
- `--output-path`: Output directory path

### 3. TOML Configuration File

Create a TOML configuration file for complex calculations:

```bash
python -m TB2J.Oiju_epw2 --config my_config.toml
```

#### Example TOML Configuration

```toml
# Basic calculation parameters
path = "~/projects/wannier90"
colinear = true
posfile = "POSCAR"

# Wannier90 file prefixes
prefix_up = "wannier90.up"
prefix_dn = "wannier90.dn"

# EPW calculation parameters
epw_path = "~/projects/epw"
epw_prefix_up = "system_up"
epw_prefix_dn = "system_dn"

# Calculation settings
Ru = [0, 0, 0]
Rcut = 8.0
efermi = 10.67
magnetic_elements = ["Mn"]

# k-point mesh and energy integration
kmesh = [5, 5, 5]
emin = -7.33
emax = 0.0
nz = 70
np = 1

# Output settings
description = "Spin-phonon coupling calculation"
output_path = "TB2J_results"

# Multiple displacement modes
idisp_list = [3, 6, 7, 10, 12]
```

## Function Reference

### `gen_exchange_Oiju_epw_multiple()`

Calculate spin-phonon coupling for multiple displacement patterns.

**Parameters:**
- `idisp_list`: List of displacement mode indices (default: [3, 6, 7])
- All other parameters same as `gen_exchange_Oiju_epw()`

**Returns:**
- Creates separate output directories for each idisp value

### `run_from_toml(config_file)`

Run calculations from a TOML configuration file.

**Parameters:**
- `config_file`: Path to TOML configuration file

### `create_cli()`

Create and return the argument parser for the CLI interface.

## Output Structure

When using multiple idisp values, the module creates separate output directories:

```
output_path_idisp_3/
├── exchange.out
├── Jij.txt
└── ...

output_path_idisp_6/
├── exchange.out
├── Jij.txt
└── ...

output_path_idisp_7/
├── exchange.out
├── Jij.txt
└── ...
```

## Dependencies

- `toml`: For parsing configuration files
- `argparse`: For CLI interface
- `ase`: For structure file reading
- TB2J modules: `epwparser`, `exchange_pert2`, `myTB`, `utils`

## Examples

See `example_config.toml` for a complete configuration file example.