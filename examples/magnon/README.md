# Magnon Band Structure Examples

This directory contains examples for calculating magnon band structures and DOS from TB2J results.

## Python Scripts

| File | Description |
|------|-------------|
| `magnon_bands.py` | Basic band structure calculation |
| `magnon_bands_no_dmi.py` | Band structure without DMI and anisotropic exchange |
| `magnon_bands_custom_qpoints.py` | Band structure with custom q-points |
| `magnon_dos.py` | DOS calculation |

## Shell Scripts

| File | Description |
|------|-------------|
| `run_bands.sh` | Band structure via CLI |
| `run_bands_no_dmi.sh` | Band structure (isotropic only) via CLI |
| `run_bands_custom_qpoints.sh` | Band structure with custom q-points via CLI |
| `run_dos.sh` | DOS via CLI |
| `run_with_config.sh` | Using TOML configuration file |
| `run_with_spin_conf.sh` | Custom spin configuration via CLI |

## Configuration Files

| File | Description |
|------|-------------|
| `config.toml` | Basic TOML configuration |
| `config_spin.toml` | TOML config with custom spin configuration |
| `config_custom_qpoints.toml` | TOML config with custom q-points |

## Usage

### Python Scripts

```bash
cd TB2J
source .venv/bin/activate
python examples/magnon/magnon_bands.py
```

### Shell Scripts

```bash
cd TB2J
source .venv/bin/activate
./examples/magnon/run_bands.sh
```

### With Custom Data

Copy your `TB2J_results` directory to the TB2J root or specify the path:

```bash
TB2J_magnon.py --bands --path /path/to/TB2J_results
```

## Custom Q-Points

By default, TB2J uses ASE's built-in special k-points for standard crystal structures. However, you can specify custom q-points with your own names and coordinates.

### Python API

```python
from TB2J.magnon.magnon_parameters import MagnonParameters

# Define custom q-points as a dictionary
custom_qpoints = {
    "G": [0.0, 0.0, 0.0],    # Gamma point
    "X": [0.5, 0.0, 0.0],    # X point
    "M": [0.5, 0.5, 0.0],    # M point
    "R": [0.5, 0.5, 0.5],    # R point
}

params = MagnonParameters(
    path="TB2J_results",
    kpath="GXMRG",           # Path connecting the q-points
    qpoints=custom_qpoints,  # Your custom q-points
)
```

### Command Line

Use the `--qpoints` option with format `name:x,y,z`:

```bash
TB2J_magnon.py --bands \
    --kpath GXMRG \
    --qpoints "G:0,0,0,X:0.5,0,0,M:0.5,0.5,0,R:0.5,0.5,0.5"
```

### TOML Configuration

In your TOML file:

```toml
kpath = "GXMRG"

[qpoints]
G = [0.0, 0.0, 0.0]
X = [0.5, 0.0, 0.0]
M = [0.5, 0.5, 0.0]
R = [0.5, 0.5, 0.5]
```

## Requirements

- TB2J results directory (e.g., `TB2J_results/` with `TB2J.pickle`)
