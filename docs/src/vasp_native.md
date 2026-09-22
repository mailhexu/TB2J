# VASP native PAW interface

TB2J can compute magnetic exchange directly from a VASP run through a
**native PAW export**: a patched VASP writes a compact binary file
(`tb2j_native.bin`) containing the PAW projector coefficients (`CPROJ`),
eigenvalues, occupations and on-site `CDIJ` matrices.  TB2J reads this file
and computes the exchange Green function — no Wannierisation is needed.

This is the VASP counterpart of the ABINIT `savetb2j` NetCDF interface
(see `abinit_savetb2j_schema.rst`).

## Requirements

- VASP 6.4.1 (or compatible MPI build, `vasp_std`) with the
  **TB2J export patch** applied and rebuilt.
- TB2J installed (`pip install -e /path/to/TB2J`).

## Installing the patch

```bash
cd /path/to/vasp.6.4.1
bash /path/to/VASP_TB2J_patch/patch_vasp6.4.1/apply_patch.sh
make std
```

The patch adds a self-contained module `tb2j_export.F` and hooks the
export into `main.F`.

## Running VASP

**No INCAR parameter is needed** to enable the export.  The exporter runs
automatically after the SCF when:

- the SCF has converged (`INFO%LSTOP`),
- the calculation is collinear spin-polarised (`ISPIN ≤ 2`, `NCDIJ ≤ 2`).

Keep `ISYM` at its default (or set `ISYM = 1`): with symmetry active the
patch forces the allocation of the full-BZ mapping, and the export
automatically uses the symmetry-reduced **v6** format (IBZ k-points plus a
self-contained IBZ→full-BZ expansion plan).  Without symmetry a full-BZ
**v5** stream is written (complex `CPROJ`/`CDIJ`, per-type element labels,
VASP `QTOT` metric).  On success VASP prints e.g.

```
 TB2J: native PAW v6 IBZ export written to tb2j_native.bin (IBZ=10, BZ=64)
```

## Computing the exchange

```bash
vasp_native2J.py --input tb2j_native.bin --elements Fe --Rcut 6.0 --nz 80
```

| Option | Default | Meaning |
|---|---|---|
| `--input` | (required) | `tb2j_native.bin` (v4/v5 full-BZ or v6 IBZ stream) |
| `--output_path` | `TB2J_results_vasp` | output directory |
| `--elements` | all | magnetic elements, e.g. `Fe` or `Mn Fe` |
| `--index_magnetic_atoms` | all | indices of magnetic atoms (alternative to `--elements`) |
| `--Rcut` | 10.0 | spin-pair distance cutoff (Å) |
| `--nz` | 80 | continued-fraction poles |
| `--smearing` | 0.05 | CFR smearing (eV) |

The results are written to the output directory (`exchange.out`,
`TB2J.pickle`, …) in the standard TB2J format, ready for
`TB2J_magnon.py`, `TB2J_plot_exchange.py` or the symmetrizer
(see {doc}`symmetrization`).

## Example

A worked ferromagnetic FeO (PBE+U, primitive cell) example — INCAR, POSCAR,
KPOINTS and the reference `exchange.out` — ships with the patch package
(`examples/VASP_FeO_U5`).  Its INCAR contains **no** TB2J-specific setting:
the export is triggered by the patched binary itself.
