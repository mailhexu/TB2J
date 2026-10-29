### Symmetrization of the exchange parameters 

The exchange parameters obtained from the output of TB2J may not preserve the symmetry of the crystal. This can be attributed to two reasons:

1. The magnetic state breaks the symmetry of the crystal. As TB2J perturbs this magnetic state, the J values should reflect the symmetry of the magnetic state.

2. Numerical noise can be present in the calculation, either during the DFT stage, the Wannierization procedure, or the post-processing within TB2J. This numerical noise should ideally be very small. If it is not, the calculation procedure should be optimized. 

In some cases, it is necessary for the exchange parameters to strictly adhere to the symmetry of the crystal structure. To achieve this, you can use the script `TB2J_symmetrize.py` to symmetrize the exchange parameters.

:Warning: By default the script considers the symmetry of the crystal only and does not take the magnetic moment into account. Use the `--magnetic` option to symmetrize with the magnetic space group built from the magnetic moments stored with the exchange parameters.

The `TB2J_symmetrize.py` script utilizes the symmetry of the crystal to identify symmetrically equivalent atom pairs and the corresponding symmetry operators. It then averages the J values over these symmetrically equivalent atom pairs. The full pair tensor is symmetrized, that is the isotropic exchange, the Dzyaloshinskii-Moriya interaction and the anisotropic exchange, as well as the single-ion anisotropy tensors when they are present. The symmetry operations are detected with [spglib](https://spglib.readthedocs.io/) and include the translational parts of the operations (lattice centering, glide and screw translations).

Here is the usage information for the script:

```
usage: TB2J_symmetrize.py [-h] [-i INPATH] [-o OUTPATH] [-s SYMPREC]
                           [--Jonly] [--magnetic] [--zero-tol ZERO_TOL]

Symmetrize exchange parameters (isotropic exchange, DMI, anisotropic exchange
and single-ion anisotropy) with the exact space-group operations of the
structure (rotations and translations). With --magnetic, the magnetic space
group built from the stored magnetic moments (spinat) is used instead.

options:
  -h, --help            show this help message and exit
  -i INPATH, --inpath INPATH
                        input path to the exchange parameters
  -o OUTPATH, --outpath OUTPATH
                        output path to the symmetrized exchange parameters
  -s SYMPREC, --symprec SYMPREC
                        precision for symmetry detection. The default value is 1e-5 Angstrom
  --Jonly               symmetrize only the exchange parameters and discard the DMI,
                        anisotropic exchange and single-ion anisotropy
  --magnetic            use the magnetic space group (from the stored spinat) instead
                        of the crystallographic space group
  --zero-tol ZERO_TOL  components of the symmetrized channels smaller than this are
                        snapped to exactly zero. The default value is 1e-8
```

An example:

```bash
TB2J_symmetrize.py -i TB2J_results -o TB2J_symmetrized -s 1e-4
```

The script can read the data from TB2J\_results and write the symmetrized exchange parameters to a new directory TB2J\_symmetrized. The precistion for detecting the symmetry is 1e-4 angstrom. 

The symmetrization can also be applied to an existing set of exchange parameters with a reference structure, for instance to restore the symmetry of an ideal parent structure, with the `TB2J_edit.py symmetrize` command:

```bash
TB2J_edit.py symmetrize -i TB2J_results/TB2J.pickle -S structure.cif -o modified_results
```

If transferring reference symmetry produces a non-integer bond image,
`TB2J_edit.py symmetrize` raises `ValueError` rather than silently skipping
the image. Check that the stored and reference structures share the cell
and have consistent positions and symmetry tolerance.

### Editable symmetry-shell exchange

Use `TB2J_shell generate` to create a symmetry-aware editable file from a
structure or a trusted TB2J pickle, then `TB2J_shell export` to write standard
results. See [symmetry-shell exchange files](exchange_shells.md) for the format,
Python interface, and supported solver-output omissions.
