# SIESTA WFSX exchange reference fixtures

This directory contains small, complete producer outputs for independent WFSX checks. `fixture_driver.py verify --root .` checks hashes, metadata, complete state IDs, spin/k-point/weight/energy conventions, EIG agreement, and generalized eigenpair residuals using sisl directly. It is not a WFSX importer and does not reconstruct an ordinary Hamiltonian from WFSX.

## Producer and reproduction

- SIESTA version: `5.2.0-46-gfc8b48876`; executable SHA-256: `865ab6f1db3619ec53033e5d2eba40174f6b106360196ce6329bbc7872ec9dbf`.
- SIESTA executable observed at: `/home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta`.
- Reported build: x86_64; GNU-15.2.0; `-fallow-argument-mismatch -O3 -march=native`; MPI, one MPI rank.
- Source checkout observed separately from the binary version: `f564b96d5fd1aaa8e7c190d9040eae39c9c07f8e` at `/home/hexu/projects/siesta_dev/siesta`.
- The binary's version string and the source checkout HEAD are recorded separately; the checkout HEAD is not claimed as the binary's exact build commit.
- sisl ingestion API version: `0.16.2`.
- All runs used `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no MPI launcher was used (one rank).
- Reproduce in an empty destination with the exact executable and source checkout:
  `rtk env PYTHONPATH=/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j:/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio /home/hexu/projects/myenvs/mydev/bin/python /home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/fixture_driver.py generate --siesta /home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta --source-root /home/hexu/projects/siesta_dev/siesta --output /path/to/empty-output`.
  The generator refuses to overwrite a non-empty output path, runs scalar, collinear (one run for both exports), and noncollinear inputs, then writes metadata and SHA256SUMS.

## Physics, convergence, and export semantics

All cases use the same O2 chain geometry (7 × 7 × 4 Å cell; O positions (2.895, 3.5, 2.0) and (4.105, 3.5, 2.0) Å), PBE/SZP, 150 Ry mesh cutoff, 100 K Fermi-Dirac occupations, and a 1 × 1 × 5 z mesh. The retained `.ORB_INDX` files are unmodified same-run SIESTA outputs; they carry the supercell range needed by sisl/HamiltonIO geometry loading.

Every SCF converged in nine iterations by the FreeE criterion only (`SCF.FreeE.Tolerance = 1e-8 eV`). No stricter H/DM stationarity or torque claim is made. Magnetic moments below are the final SIESTA moments, not merely initialization values.

The magnetic nspin=2 FDF has `TimeReversalSymmetryForKpoints true`, `COOP.Write T`, and an explicit `WaveFuncKPoints` list at 0, ±0.2, ±0.4. In the single converged run, `o2.fullBZ.WFSX` is the three-point SCF folded export (`scf-folded`, weights 0.2/0.4/0.4) and `o2.selected.WFSX` is the five-point explicit list (`selected-list`, raw WFSX weights 1.0 each, not integration weights). The two fixture metadata records share one `producer_run_id` and the exact same `.HSX`, `.EIG`, `.KP`, `.ORB_INDX`, `.out`, and FDF ordinary references. The fold map is explicit in the full-list metadata.

The scalar nspin=1 fixture declares real-Hamiltonian, no-SOC, no-orbital-field, and no-spin-spiral symmetry. The magnetic nspin=2 fixture declares the same scalar symmetry conditions while retaining its two spin channels. The nspin=4 fixture is a full noncollinear spinor with no scalar reduction and no SOC; it is not relabeled as SOC nspin=8.

## Independent checks observed

sisl 0.16.2 reads orbital order from same-run `HSX.read_basis()` and `WFSX.read_basis()`, then reads states through `sisl.get_sile(path, lattice=geom.lattice).read_sizes()` and `yield_eigenstate()`. WFSX record indices are zero-based, coefficient arrays are complex64, energies are float64 and absolute eV, and `info['k']` is fractional reciprocal-lattice coordinates after one lattice conversion. `o2.EIG` values read by sisl are relative to E_F; subtracting the exact EIG Fermi record from WFSX energies matches all EIG states. Metadata retains every state-ID vector, spin record, basis/site order, raw/folded k weights, and source dtype.

Direct sisl H/S generalized-eigenpair checks use same-run HSX `H(k)`, `S(k)` and WFSX vectors in orbital gauge. The verifier requires max residual ≤1e-5 eV and max S-orthogonality defect ≤1e-5. The observed maxima are stored per fixture in `fixture.json`; no WFSX-based H/S reconstruction is used.

WFSX metadata coordinates and canonical full-to-reduced mapping use wrapped componentwise absolute tolerance 1e-8 and reject ambiguous representatives. Same-run `.KP` coordinates are printed to six significant digits, so only WFSX-to-`.KP` lookup uses `KP_PRINTING_TOL = 5e-7`; the maximum measured fractional discrepancy in retained fixtures is 1.832899072e-7. This print-quantization tolerance does not apply to WFSX metadata or canonical fold-map comparisons.

The ordinary-parser values below are dated 2026-10-30 observations for this retained reference bundle and the cited TW/HW source commits. `fixture_driver.py generate` does not recompute ordinary TB2J parser comparisons; it runs SIESTA and independent direct-sisl checks. `TB2J.__file__` was `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/TB2J/__init__.py` (TW baseline 33c5624), and `HamiltonIO.__file__` was `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio/HamiltonIO/__init__.py`.

With retained same-run `.ORB_INDX`, scalar and nspin=2 ordinary spectra on approved HW baseline 829fc0f agreed with WFSX to 2.339746174584434e-9 eV and 3.371603440882609e-9 eV, respectively. Historical nspin=4 behavior on HW baseline 829fc0f was `IndexError: index 5 is out of bounds for axis 3 with size 5` in `convert_sisl_to_spinorham` for HSX CSR shape (18, 54, 5).

After separate user authorization, the four-component noncollinear H/S parser fix was committed as e806c29 based on review base 0e1551c. Independent review and successful qualified checked finish passed; the two-axis test-hook cleanup review was also approved, with the seven-line numerical branch unchanged. On that unchanged numerical branch, actual ordinary operator smoke for nspin=1, nspin=2, and nspin=4 at fractional k_z = 0, ±0.2, ±0.4 matched native parser H/S with maximum absolute H difference 3.469446951953614e-18 eV and S difference 4.235164736271502e-22. The nspin=4 WFSX-versus-ordinary full-spectrum comparison covered 180 states with maximum difference 3.7032847899354238e-9 eV. These are retained-bundle observations, not freshly computed by `generate`. The repair is limited to four-component noncollinear H/S parsing; no SOC or HSX-offset change was made.

## Pseudopotential source and redistribution note

`O.psf` is byte-identical to the already-existing TB2J O2 test fixture and to `siesta/Tests/Pseudos/O.psf` (SHA-256 `d3ecdc324e91f272a7d1b9d7747cd6f08c81899c37d0500256a0f2fc61291c71`). The upstream `Tests/Pseudos/README` says only that the directory contains pseudopotential files for SIESTA tests. No separate per-pseudopotential license was found, so this note makes no license/redistribution claim based solely on SIESTA's source-code GPL. The fixture retains the exact pre-existing test asset without modification.

`SHA256SUMS` pins the retained inputs, producer outputs, metadata, and this provenance record. Oracle H/S files are validation-only; they are not production-mode metadata inputs.

- `scalar_nspin1`: run `20261030T073812044051Z-scalar_nspin1`, nspin=1, EF=-5.03963234 eV, N=12.000000, moment=(0.00000, 0.00000, 0.00000) μB, 9 FreeE iterations; 5 WFSX k points, 5 k/spin groups, 18 unit-cell orbitals, coefficient dtype complex64, max residual 1.42571959e-06 eV, max S defect 1.06022185e-07.
- `collinear_nspin2_full`: run `20261030T073812044051Z-collinear_nspin2`, nspin=2, EF=-4.76597097 eV, N=12.000000, moment=(0.00000, 0.00000, 2.00000) μB, 9 FreeE iterations; 5 WFSX k points, 10 k/spin groups, 18 unit-cell orbitals, coefficient dtype complex64, max residual 1.48673462e-06 eV, max S defect 1.00262786e-07.
- `collinear_nspin2_reduced`: run `20261030T073812044051Z-collinear_nspin2`, nspin=2, EF=-4.76597097 eV, N=12.000000, moment=(0.00000, 0.00000, 2.00000) μB, 9 FreeE iterations; 3 WFSX k points, 6 k/spin groups, 18 unit-cell orbitals, coefficient dtype complex64, max residual 1.48673462e-06 eV, max S defect 1.00262786e-07.
- `noncollinear_nspin4_full`: run `20261030T073812044051Z-noncollinear_nspin4_full`, nspin=4, EF=-4.76596733 eV, N=12.000000, moment=(2.00000, 0.00000, -0.00000) μB, 9 FreeE iterations; 5 WFSX k points, 5 k/spin groups, 18 unit-cell orbitals, coefficient dtype complex64, max residual 2.36345147e-06 eV, max S defect 1.01989404e-07.
