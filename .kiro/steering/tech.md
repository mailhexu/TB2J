# TB2J Technology Stack

## Build System
- Uses setuptools for packaging and distribution
- Modern pyproject.toml configuration with fallback setup.py
- Entry points defined for console scripts in both files

## Core Dependencies
- **Python**: >=3.6 (tested on 3.6+)
- **NumPy**: <2.0 (core numerical operations)
- **SciPy**: Scientific computing
- **ASE**: >=3.19 (Atomic Simulation Environment)
- **matplotlib**: Plotting and visualization
- **tqdm**: Progress bars
- **pathos**: Parallel processing
- **HamiltonIO**: >=0.2.4 (Hamiltonian I/O operations)
- **packaging**: >=20.0 (Version handling)
- **pre-commit**: Code quality automation
- **sympair**: >0.1.0 (Symmetry operations)
- **tomli**: >=2.0.0 + **tomli-w**: >=1.0.0 (TOML file handling)

## Optional Dependencies
- **sisl**: >=0.9.0 + netcdf4 (for Siesta interface)
- **lawaf**: ==0.2.3 (for LAWAF interface)

## Code Quality Tools
- **Ruff**: Linting and formatting (configured in .ruff.toml)
  - Line length: 88 characters
  - Target Python 3.8+
  - Double quotes for strings, 4-space indentation
- **pre-commit**: Automated code quality checks
  - Runs ruff linter with import sorting
  - Runs ruff formatter

## Common Commands
```bash
# Installation
pip install TB2J

# Development installation
pip install -e .

# Run linting
ruff check --fix --extend-select I .

# Run formatting  
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## Main Entry Points
- `wann2J`: Wannier90 interface
- `siesta2J`: Siesta interface  
- `abacus2J`: ABACUS interface
- `TB2J_magnon`: Magnon calculations
- `TB2J_plot_magnon_bands`: Magnon band plotting
- `TB2J_plot_magnon_dos`: Magnon density of states plotting
- `TB2J_rotate`: Rotate magnetic structures
- `TB2J_merge`: Merge exchange parameters
- `TB2J_downfold`: Downfolding utilities
- `TB2J_eigen`: Eigenvalue calculations
- `lawaf2J.py`: LAWAF interface
- `TB2J_symmetrize.py`: Symmetrize exchange parameters