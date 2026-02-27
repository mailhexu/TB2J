# Magnon Band Structure Examples

This directory contains examples for calculating magnon band structures and DOS from TB2J results.

## Python Scripts

| File | Description |
|------|-------------|
| `magnon_bands.py` | Basic band structure calculation |
| `magnon_bands_no_dmi.py` | Band structure without DMI and anisotropic exchange |
| `magnon_dos.py` | DOS calculation |

## Shell Scripts

| File | Description |
|------|-------------|
| `run_bands.sh` | Band structure via CLI |
| `run_bands_no_dmi.sh` | Band structure (isotropic only) via CLI |
| `run_dos.sh` | DOS via CLI |
| `run_with_config.sh` | Using TOML configuration file |
| `run_with_spin_conf.sh` | Custom spin configuration via CLI |

## Configuration Files

| File | Description |
|------|-------------|
| `config.toml` | Basic TOML configuration |
| `config_spin.toml` | TOML config with custom spin configuration |

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

## Requirements

- TB2J results directory (e.g., `TB2J_results/` with `TB2J.pickle`)
