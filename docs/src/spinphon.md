# Reading EPW Output Files

This document describes how to read and work with Electron-Phonon Wannier (EPW) output files using TB2J.

## File Format Overview

EPW calculations generate several output files that need to be read together:

1. `crystal.fmt`: Contains crystal structure information
2. `epwdata.fmt`: Contains basic dimensions and Fermi energy
3. `WSVecDeg.dat`: Contains Wigner-Seitz vectors and degeneracies
4. `.epmatwp`: Binary file containing the electron-phonon matrix elements

## Converting EPW Output to NetCDF

For easier handling of the EPW matrix elements, TB2J provides a script to convert the binary `.epmatwp` file to NetCDF format. To use it:

```bash
python3 convert_epw_to_netcdf.py --path PATH --prefix PREFIX [--ncfile NCFILE]
```

Arguments:
- `--path`: Directory containing the EPW data (default: current directory)
- `--prefix`: Prefix of the EPW data files
- `--ncfile`: Name of the output netcdf file (default: 'epmat.nc')

Example:
```bash
python3 convert_epw_to_netcdf.py --path ./epw_calc --prefix sic --ncfile epmat.nc
```

## Reading EPW Data Using the Python API

TB2J provides the `Epmat` class for handling EPW data. Here's how to use it:

```python
from TB2J.epwparser import Epmat

# Initialize and read data
ep = Epmat()
ep.read(path='./epw_calc', prefix='sic', epmat_ncfile='epmat.nc')

# Get matrix elements for specific mode and R vectors
matrix = ep.get_epmat_Rv_from_RgRk(imode=0, Rg=(0,0,0), Rk=(0,0,0))
```

### Working with Single Phonon Modes

For analyzing individual phonon modes, use the `EpmatOneMode` class:

```python
from TB2J.epwparser import EpmatOneMode

# Get data for a specific mode
ep1mode = EpmatOneMode(ep, imode=3)

# Get matrix elements for specific R vectors
matrix = ep1mode.get_epmat_RgRk(Rg=(0,0,0), Rk=(0,1,0))

# Include time-reversal symmetry
matrix_avg = ep1mode.get_epmat_RgRk(Rg=(0,0,0), Rk=(0,1,0), avg=True)
```

## Data Structure

The EPW matrix elements are stored with the following dimensions:

- `nRg`: Number of R vectors in g-space
- `nmodes`: Number of phonon modes
- `nRk`: Number of R vectors in k-space
- `nwann`: Number of Wannier functions

The matrix elements in the NetCDF file are stored in two variables:
- `epmat_real`: Real part of the matrix elements
- `epmat_imag`: Imaginary part of the matrix elements

Each has dimensions `(nRg, nmodes, nRk, nwann, nwann)`.

## Units

The matrix elements are automatically converted from Rydberg/Bohr to atomic units when read from the NetCDF file.

## Crystal Structure Information

The crystal structure information can be read from the `crystal.fmt` file using:

```python
from TB2J.epwparser import read_crystal_fmt

crystal = read_crystal_fmt('crystal.fmt')
```

This provides access to:
- Number of atoms (`natom`)
- Number of modes (`nmode`)
- Number of electrons (`nelect`)
- Lattice vectors (`at`, `bg`)
- Unit cell volume (`omega`)
- Lattice parameter (`alat`)
- Atomic positions (`tau`)
- Atomic masses (`amass`)
- Atomic types (`ityp`)
- Wannier function centers (`w_centers`)

## Calculating Spin-Phonon Coupling

TB2J provides functionality to calculate spin-phonon coupling parameters by combining Wannier90 electronic structure with EPW electron-phonon matrix elements. This is done using the `gen_exchange_Oiju_epw` function:

```python
from TB2J.Oiju_epw2 import gen_exchange_Oiju_epw

gen_exchange_Oiju_epw(
    path="./wannier90_calc",          # Path to Wannier90 calculation
    colinear=True,                    # Whether system is collinear magnetic
    posfile='POSCAR',                 # Structure file name
    prefix_up='wannier90.up',         # Prefix for spin-up files
    prefix_dn='wannier90.dn',         # Prefix for spin-down files
    epw_path='./epw_calc',            # Path to EPW calculation
    epw_prefix='system',              # Prefix for EPW files
    idisp=0,                          # Atomic displacement index
    Ru=(0, 0, 0),                     # R vector for unit cell
    efermi=3.0,                       # Fermi energy (eV)
    magnetic_elements=['Fe'],         # Magnetic elements
    kmesh=[5, 5, 5],                 # k-point mesh
    emin=-12.0,                      # Minimum energy for integration
    emax=0.0,                        # Maximum energy for integration
    nz=50,                           # Number of energy points
    output_path='TB2J_results'       # Output directory
)
```

### Key Parameters

- `path`: Directory containing Wannier90 calculation files
- `epw_path`: Directory containing EPW calculation files
- `idisp`: Index of the atomic displacement pattern to analyze
- `Ru`: R vector specifying the unit cell
- `magnetic_elements`: List of magnetic elements in the system
- `kmesh`: Dimensions of the k-point mesh for integration
- `emin`, `emax`: Energy range for integration
- `nz`: Number of energy points for integration
- `output_path`: Directory where results will be written

### Example Usage

Here's a complete example for calculating spin-phonon coupling in SrMnO3:

```python
from TB2J.Oiju_epw2 import gen_exchange_Oiju_epw

# Calculate coupling for displacement pattern 3
gen_exchange_Oiju_epw(
    path="./W90",
    colinear=True,
    posfile='SrMnO3.scf.pwi',
    prefix_up="SrMnO3_up",
    prefix_dn="SrMnO3_down",
    epw_path='./epw',
    epw_prefix='SrMnO3',
    idisp=3,
    Ru=(0, 0, 0),
    Rcut=8,
    efermi=10.67,
    magnetic_elements=['Mn'],
    kmesh=[5, 5, 5],
    emin=-7.34,
    emax=0.0,
    nz=70,
    output_path="spinphon_results"
)
```

The results will be written to the specified output directory, containing the calculated spin-phonon coupling parameters for the selected atomic displacement pattern. The calculation combines the electronic structure from Wannier90 with the electron-phonon matrix elements from EPW to determine how atomic displacements couple to the magnetic degrees of freedom.
