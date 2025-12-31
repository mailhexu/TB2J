# TB2J Command Reference

## Common Arguments

All calculation scripts (`wann2J.py`, `siesta2J.py`, `abacus2J.py`) share a common set of arguments.

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--elements` | List[str] | None | Elements to be considered as magnetic. Examples: `--elements Fe Ni`, `--elements Fe_d`. If specified with orbitals (e.g. `Fe_d`), only those orbitals are used. |
| `--efermi` | float | None | **Required**. Fermi energy in eV. Must match the DFT calculation. |
| `--kmesh` | int int int | 5 5 5 | Monkhorst-Pack k-point mesh (kx ky kz). |
| `--spinor` | flag | False | Enable for non-collinear/spin-orbit coupling calculations. |
| `--index_magnetic_atoms` | List[int] | None | Indices of magnetic atoms in the unit cell (1-based index). If specified, overrides `--elements` for atom selection. |

### Energy Integration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--emin` | float | -14.0 | Lower bound of energy integration (relative to Fermi energy) in eV. |
| `--emax` | float | 0.0 | Upper bound of energy integration (relative to Fermi energy) in eV. **Typically 0.0 to integrate up to the Fermi level.** |
| `--nz` | int | 100 | Number of points for the semi-circle contour integration. |
| `--ne` | float | None | Number of electrons. If provided, `efermi` is adjusted to match this number. |

### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_path` | str | `TB2J_results` | Directory where results will be saved. |
| `--cutoff` | float | 1e-5 | Minimum absolute value of J (in eV) to write to output. |
| `--description` | str | "Calculated with TB2J." | Description string added to the `exchange.xml` output. |
| `--write_dm` | flag | False | Write density matrix. |

### Computational Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--np` | int | 1 | Number of CPU cores for parallel processing. |
| `--use_cache` | flag | False | Use disk caching for wavefunctions/Hamiltonians to save RAM. |
| `--rcut` | float | None | Cutoff distance (Angstroms) for calculating interactions. If None, all commensurate R-vectors are calculated. |
| `--exclude_orbs`| List[int]| [] | Indices of Wannier functions (0-based) to exclude from magnetic sites. |
| `--orb_decomposition`| flag | False | Perform orbital decomposition in non-collinear mode. |

### Advanced / Debug Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--orth` | flag | False | Use Lowdin orthogonalization (testing only). |
| `--ibz` | flag | False | Use irreducible k-points (only for computing total MAE). |
| `--mae_angles`| List[float]| 0 0 0 | Angles for computing MAE. |

## Tool-Specific Arguments

### wann2J.py (Wannier90)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--path` | `./` | Path to Wannier90 files. |
| `--posfile` | `POSCAR` | Structure file (POSCAR, .xyz, etc.). Defaults to `POSCAR`. |
| `--prefix_spinor` | `wannier90` | Prefix for spinor wannier files (e.g. `prefix.win`, `prefix.eig`). |
| `--prefix_up` | `wannier90.up` | Prefix for spin-up files (collinear). |
| `--prefix_down` | `wannier90.dn` | Prefix for spin-down files (collinear). |
| `--groupby` | `spin` | Ordering of orbitals in spinor mode: `spin` (orb1_up, orb1_dn...) or `orbital`. |
| `--wannier_type` | `Wannier90` | Type of Wannier function (`Wannier90` or `banddownfolder`). |

### siesta2J.py (Siesta)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fdf_fname` | `./` | Path to the Siesta `.fdf` input file. |
| `--fname` | `exchange.xml` | Name of the output XML file. |
| `--split_soc` | False | If True, reads SOC part from a separate file. |

### abacus2J.py (Abacus)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--path` | `./` | Path containing Abacus calculation. |
| `--suffix` | `abacus` | Suffix of the output directory (`OUT.suffix`). |

## Rotate and Merge Tools

Tools for calculating anisotropy from collinear codes using the "Rotate and Merge" method.

### `TB2J_rotate.py`
Generates rotated crystal structures.

| Parameter | Description |
|-----------|-------------|
| `input_file` | The input structure file (e.g. `POSCAR`). |
| `--ftype` | File type (e.g. `vasp`, `xyz`). |
| `--noncollinear` | Generate 6 structures (for full J tensor) instead of 3 (for diagonal terms only). |

### `TB2J_merge.py`
Merges results from rotated calculations.

**Usage:**
```bash
TB2J_merge.py <path_to_rotated_1> <path_to_rotated_2> ... <path_to_unrotated>
```
The last argument is the reference (unrotated) structure's result directory.

## Output Files

The results are typically found in the directory specified by `--output_path` (default: `TB2J_results`).

### Primary Output

- **`exchange.xml`**: The master output file containing:
  - System structure (lattice, atoms).
  - Exchange parameters ($J_{ij}$) for all pairs.
  - DMI vectors ($D_{ij}$) if applicable.
  - Anisotropic exchange tensors ($J^{ani}_{ij}$).
  - Orbital decomposition data (if requested).

### Code-Specific Formats

TB2J automatically generates input files for other spin dynamics/simulation codes:

- **Multibinit** (`TB2J_results/Multibinit/`):
  - `exchange.xml` (Multibinit format)
  - `reduced_basis.xml`

- **Vampire** (`TB2J_results/Vampire/`):
  - `vampire.mat`: Material file.
  - `vampire.UCF`: Unit cell file.

- **Tom's ASD** (`TB2J_results/TomASD/`):
  - Input files compatible with Tom's ASD code.
