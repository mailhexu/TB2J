# TB2J Project Structure

## Root Directory Layout
```
TB2J/                    # Main package directory
├── interfaces/          # DFT code interfaces (Wannier90, Siesta, ABACUS)
├── io_exchange/         # Exchange parameter I/O formats
├── magnon/              # Magnon band structure calculations
├── mathutils/           # Mathematical utilities and algorithms
├── spinham/             # Spin Hamiltonian handling
├── wannier/             # Wannier function parsers
└── downfold/            # Downfolding utilities

scripts/                 # Command-line entry points
examples/                # Usage examples and test cases
docs/                    # Documentation source files
```

## Key Module Organization

### Core Physics Modules
- `exchange.py`, `exchange_*.py`: Exchange interaction calculations
- `green.py`: Green's function methods
- `MAE.py`, `MAEGreen.py`: Magnetic anisotropy energy
- `anisotropy.py`: Anisotropic exchange interactions

### Interface Modules (`TB2J/interfaces/`)
- `wannier90_interface.py`: Wannier90 integration
- `siesta_interface.py`: Siesta DFT code interface
- `abacus/`: ABACUS DFT code interface (subpackage)
- `manager.py`: Interface management

### I/O Modules (`TB2J/io_exchange/`)
- `io_*.py`: Output formatters for different spin dynamics codes
- Supports: MULTIBINIT, UppASD, Vampire, TomASD

### Utilities
- `mathutils/`: Mathematical operations, k-point handling, Fermi functions
- `spinham/`: Spin Hamiltonian construction and manipulation
- `utils.py`: General utility functions

## Naming Conventions
- Script files: `TB2J_*.py` or `*2J.py` pattern
- Interface modules: `*_interface.py` or `*_wrapper.py`
- I/O modules: `io_*.py`
- Test files: `test_*.py`

## File Extensions
- `.py`: Python source files
- `.pickle`: Serialized TB2J results
- `.out`, `.txt`: Text output files
- `.xml`: Exchange parameter XML format