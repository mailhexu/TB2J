# Fe3 spiral import oracle

Producer: SIESTA `5.4.2-23-g549d9de29`, executable
`siesta_spinor/_build/build_fresh/Src/siesta`; sisl `0.16.2` for ingestion.
`SHA256SUMS` pins the committed FDF, pseudo, HSX, EIG, KP, stdout and reduced
NPZ; `producer.json` pins the executable and relevant source identities.

The complete producer artifacts are included. Run SIESTA with stdin from
`fe3_spiral.fdf` in a fresh directory containing `Fe_nc.psf`. Reduce with
`make_fixture_from_siesta(hsx_path, eig_path, kp_path, (0,0,0.25), 24, 300,
"fe3")` and `numpy.savez_compressed`; those count/temperature arguments are
cross-checked by `fe3_spiral.out` (`Total number of electrons: 24.000000`,
`Electronic Temperature = 300.0000 K`). The pseudo comes from the pinned
SIESTA checkout's `Tests/Pseudos/Fe_nc.psf`.

Input: Fe3 chain, 12×12×10.932 Bohr cell, 3.644 Bohr spacing, PBE, SZP,
200 Ry mesh, noncollinear spin, reciprocal-lattice pitch (0,0,1/4), four
k-points. **DM-only convergence**, iteration 37: dDmax 5.3e-5,
dHmax 0.0573583027 eV. This is an import convention reference, not a claim
of a stationary physical spiral. The independent gapped fixture is O2.

The HSX is version 2, **double precision**, 27 orbitals, 45 supercell images,
2,607 stored entries. Preserve the 1,320 nonzero-H entries with exactly
zero overlap; they are not removable structural zeros. Filtering by
nonzero overlap caused the initial 2.8e-4 eV discrepancy and is covered
by a regression test.

Observed on the lossless fixture:
- All recorded finite EIG bands: maximum absolute difference 7.01449e-7 eV.
  KP Cartesian momenta are printed to six decimals; remaining rounding is
  bounded by the fixed 1e-5 eV absolute budget (no relative slack).
- Hermiticity: maximum assembled H defect below 3e-11 eV.
- Fermi–Dirac count at printed EIG Ef and 300 K: 24.0000038213;
  fixed absolute budget 1e-4 electrons (no relative slack).

Conventions follow `diag2kspiral.F90:312-351`: both H and S use q/2 sector
phases, with the **full orbital-center displacement**, not only the cell
image. Off-diagonals use the same stored pair with H12=H3−iH4 and
H21=H3+iH4. sisl converts the raw CSC orientation and spin packing;
retain its orientation with its corresponding displacement, rather than
independently undoing only the imaginary-component sign. Undo its Fermi
shift with the HSX header's exact Ef, not the rounded EIG value.

[Vector derivation](../../../docs/sympy/siesta_spiral_vector.md) asserts
skew-cell reciprocal conversion and the full textured adjoint relation;
the actual stored mixing-partner defect is 8.882e-15 eV.
