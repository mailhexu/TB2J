# TB2J Changelog

## Summary of changes since v0.9.0 (March 22, 2024)

This document summarizes the changes made to TB2J between version releases based on git history analysis. The current version in `pyproject.toml` is `0.9.12.16`.

**Note**: The ReleaseNotes.md mentions v0.10.0 (September 1, 2024) and v0.11.0 (October 10, 2024), but these appear to be planned releases based on features rather than actual version numbers in the codebase, which continues with 0.9.x versioning.

---

## Version 0.9.12.16 (November 2025 - Current)

### Major Features
- **HamiltonIO Integration**: Major refactoring to use HamiltonIO (v0.2.6+) package for parsing DFT Hamiltonians
  - Separation of DFT interface code into external package
  - Improved interface management with `TB2J/interfaces/manager.py`
  - Support for ABACUS, SIESTA, Wannier90, and LaWaF interfaces

- **MAE and Anisotropy Calculations**: 
  - MAE (Magnetic Anisotropy Energy) computation with ABACUS and SIESTA interfaces
  - Single-ion anisotropy (SIA) calculations
  - Split SOC (Spin-Orbit Coupling) workflow implementation
  - MAE computation using Green's function method (`MAEGreen.py`)
  - Orbital decomposition of MAE
  - IBZ (Irreducible Brillouin Zone) k-points support for MAE calculations
  - z-to-x angle scan for MAE

- **Magnon Band Structure**: 
  - Full implementation of magnon band structure from linear spin wave theory
  - New magnon DOS (Density of States) plotting capabilities
  - `TB2J_magnon2.py` and `TB2J_plot_magnon_bands.py` scripts
  - `TB2J_plot_magnon_dos.py` with customization options
  - Support for custom spin configurations (`--spin_conf_file` option)
  - Magnon3 module with improved API

- **Downfolding Method Improvements**:
  - Projected Wannier Function (PWF) method for downfolding
  - Ligand correction to exchange based on Wannier function method
  - Lowdin orthogonalization method improvements
  - Support for DMI and anisotropic J in downfolding
  - `--method` option to select downfolding method (pwf/lowdin)

- **LaWaF Interface**:
  - New interface to LaWaF (Lattice Wannier Functions) package
  - `lawaf2J.py` script for LaWaF to J conversion
  - Integration with downfolding workflow

### Interface Enhancements

#### ABACUS Interface
- Non-collinear spin calculation support
- Orbital decomposition improvements
- MAE calculation support
- SOC (Spin-Orbit Coupling) handling
- Allow computing anisotropic J and DMI with fewer calculations
- `--index_magnetic_atoms` parameter for detailed magnetic atom selection
- Fix for duplicate spin basis in SOC calculations

#### SIESTA Interface  
- MAE calculation support
- Improved orbital decomposition
- Allow computing anisotropic J and DMI with fewer calculations
- `--orth` option for orthogonalization
- Synthetic atom support (atomic number > 200)
- Optional reading of number of electrons from siesta.nc

#### Wannier90 Interface
- Improved interface structure
- Better basis assignment using Cartesian distances
- Support for `--index_magnetic_atoms` parameter

### Symmetry Features
- `TB2J_symmetrize.py` script for symmetrizing exchange according to crystal symmetry
- Note: spin order not considered in symmetrization
- `--Jonly` option to symmetrize only J tensor

### Output Format Improvements
- **Vampire Format**: 
  - Fixed tensor/tensorial convention
  - Added factor of 2 to exchange values
  - Added `material[id]:unit-cell-category` requirement for Vampire v5+
  - Complex number handling fixes

- **espins Format**: New I/O format support for espins
- MAE output improvements with atom-pair decomposition matrix
- Removed confusing atom-wise MAE from output

### Performance Optimizations
- Vectorization of Pauli matrix operations
- Green's function computation optimizations
- Memory optimization: reduced density matrix computation from Green's function
- Parallelization improvements with `--np` option (changed from `-np`)
- CFR (Continued Fraction Representation) method for Green's function
- Contour integration refactoring with improved weight handling
- Legendre integration path as fallback for stability

### Bug Fixes
- Fixed magnetic moment y-component error in `exchange.out` (introduced in v0.9.0)
- Fixed command line function name errors (v0.9.12.5)
- Fixed magnon band CLI issues
- Fixed downfold with Lowdin method for non-collinear cases
- Fixed MAE coefficient in MAEGreen
- Fixed vampire output format for complex numbers
- Fixed rotation of Hamiltonian in spin rotation
- Fixed occupation calculations for split SOC
- Fixed Fermi energy computation from number of electrons
- Fixed overlap matrix handling in charge and magnetic moment calculations

### Code Quality and Infrastructure
- **Python 3.13 Compatibility**: Updated numpy dependencies and syntax
- **Type System**: Improved J-tensor dtype handling
- **Version Management**: `__version__` now adapted from `pyproject.toml`
- **Testing**: Git submodule setup for automated tests
- **Pre-commit Hooks**: Added pre-commit configuration
- **Ruff Integration**: Code formatting and linting with Ruff
- **Dependencies**: 
  - Made sisl, netcdf4, and lawaf optional dependencies
  - Updated to HamiltonIO >= 0.2.7
  - Updated to sympair > 0.1.1
  - tomli >= 2.0.0, tomli-w >= 1.0.0
- **Documentation**: Improved documentation for downfolding, MAE, rotation, and merge methods

### Scripts and Tools
- `TB2J_magnon_dos.py`: Magnon DOS plotting
- `TB2J_plot_magnon_bands.py`: Magnon band plotting CLI
- `TB2J_plot_magnon_dos.py`: Enhanced DOS plotting with options
- `TB2J_symmetrize.py`: Exchange symmetrization
- `lawaf2J.py`: LaWaF interface
- Improved TB2J_downfold.py with method selection
- Improved TB2J_merge.py for non-collinear systems

### Removed Features
- Removed `myTB` usage
- Removed `requirements.txt` (migrated to pyproject.toml)
- Removed J' and B from output (confusing and often not useful)

### Development and Workflow
- Improved merge workflow for non-collinear systems
- Spin rotation capabilities with density matrix
- Created rotated structures for non-collinear systems
- Support for spin-phonon workflow
- ExchangeIO object for magnon computations

---

## Key Milestones Referenced in ReleaseNotes.md

### "v0.11.0" (October 10, 2024 - mentioned in ReleaseNotes)
**Main Feature**: Symmetrization of exchange parameters
- TB2J_symmetrize.py script implementation
- Crystal symmetry-based exchange symmetrization
- Note: No actual version tag or pyproject.toml version 0.11.0 exists

### "v0.10.0" (September 1, 2024 - mentioned in ReleaseNotes)  
**Main Features**: ABACUS and SIESTA enhancements
- Improved orbital decomposition for ABACUS interface
- Computing anisotropic J and DMI with fewer calculations (3 or more calculations no longer required)
- MAE and single-ion anisotropy support for ABACUS and SIESTA

### v0.9.0 (March 22, 2024 - last entry before gap)
- Improved merge method for anisotropic exchange and DMI (Thanks to Andres Tellez Mora)
- Non-collinear merging algorithm improvements

---

## Statistics
- **Total commits since v0.9.0**: ~276 commits
- **Time period**: March 22, 2024 - November 3, 2025
- **Major contributors**: Xu He, Andres Tellez Mora, Aldo Romero, and others
- **Lines changed**: Extensive refactoring across multiple modules

---

## Migration Notes for Users

### Breaking Changes
- `myTB` module removed - use HamiltonIO instead
- Some command-line option changes: `-np` → `--np`
- Vampire output format changes (factor of 2, convention changes)
- J' and B removed from standard output

### New Dependencies
- HamiltonIO >= 0.2.7 (required)
- sympair > 0.1.1 (required)
- sisl >= 0.9.0 (optional, for SIESTA)
- netcdf4 (optional, for SIESTA)
- lawaf >= 0.2.3 (optional, for downfolding)

### Recommended Actions
1. Update HamiltonIO to latest version
2. Review Vampire output files if using Vampire
3. Check MAE calculations with new Green's function method
4. Explore new magnon band structure features
5. Try symmetrization for systems with crystal symmetry
6. Use `--index_magnetic_atoms` for fine-grained magnetic atom selection

---

## Future Development (v1.0.0-alpha mentioned in ReleaseNotes.md)
The ReleaseNotes.md indicates ongoing development toward v1.0.0-alpha with:
- Further MAE and SIA improvements
- Complete HamiltonIO integration
- Enhanced downfolding with ligand correction
- Full magnon band structure implementation
- `--index_magnetic_atoms` parameter refinements
