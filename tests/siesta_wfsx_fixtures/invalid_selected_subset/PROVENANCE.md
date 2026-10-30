# Actual SIESTA exports with incomplete selected-k coverage

These are negative-test WFSX files produced by SIESTA for Story 002. They are intentionally not `fixture.json` cases and do not replace or modify any of the four positive fixtures. Both inputs copy the retained magnetic nspin=2 FDF, preserving its geometry, pseudopotential, PBE/SZP basis, 1×1×5 SCF grid, symmetry, temperature, SCF settings, and output options; only the explicit `%block WaveFuncKPoints` records differ. `O.psf` is not duplicated here: its source is `../O.psf` relative to this directory (SHA-256 `d3ecdc324e91f272a7d1b9d7747cd6f08c81899c37d0500256a0f2fc61291c71`). The producer runs staged that exact pseudopotential into each disposable run directory.

## Producer and reference identity

- Governing record: `tb2j/siesta-wavefunction-backed-exchange-without-electronic-diagonali/story/002`.
- Governed scratch run: `20261030T081704Z-siesta-wfsx-fixtures-71c7a3685f0020e11662be9a520e6774`.
- Original magnetic reference run: `20261030T073812044051Z-collinear_nspin2`.
- Producer: `/home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta`, SIESTA `5.2.0-46-gfc8b48876`, executable SHA-256 `865ab6f1db3619ec53033e5d2eba40174f6b106360196ce6329bbc7872ec9dbf`, GNU-15.2.0, flags `-fallow-argument-mismatch -O3 -march=native`, one MPI rank. Both commands used `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
- Original FDF: `../collinear_nspin2_shared/o2_collinear.fdf`, SHA-256 `c275c7fd7b1b0871b6b8477ed96d71b06b0ffabbac275047f8d849db47ae7456`.

Commands (run from the indicated governed-scratch directories):

```sh
# cwd: <scratch>/missing
rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta < o2_missing_selected.fdf > o2.out 2>&1

# cwd: <scratch>/duplicate
rtk env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta < o2_duplicate_selected.fdf > o2.out 2>&1
```

Both producer processes exited successfully. Their output reports 12 electrons, convergence by the FreeE criterion, total energy `-868.179053` eV, and printed Fermi energy `-4.765971` eV. sisl reads exact E_F `-4.76597097` eV from each EIG. For both runs, `.HSX`, `.EIG`, `.KP`, and `.ORB_INDX` are byte-identical to the original same-SCF reference outputs, establishing exact unchanged ordinary H/S, eigenvalue/Fermi, integration-mesh, and geometry/basis files:

| File | SHA-256 (original and both negative runs) |
|---|---|
| `o2.HSX` | `0e77294db9317922e0f7f53084031a445485a37c902bf5e07ae665ed5ef433b2` |
| `o2.EIG` | `0e374255a6184507f3fc7a29dcdf9c625128d6146e632e0dd87dd953e4d2ccbc` |
| `o2.KP` | `aa622acabf9589c6a3915f256f092b901f9c2d37115ddd880d51e9cf4b6897ea` |
| `o2.ORB_INDX` | `634964d31d2bb5a9cb8c057cc5c549800223834e635bc902e7cdf7ffc831cd76` |

## Retained actual exports

`sisl 0.16.2` read each WFSX with the retained same-run HSX geometry via `sisl.get_sile(path, lattice=geom.lattice).read_sizes()` and `yield_eigenstate()`.

- `negative.selected.WFSX` (`negative` run's `o2.selected.WFSX`), SHA-256 `1783ed7f78bacc96f52079af6ed519b8b1c04133039e475854aec25d3baec996`: WFSX header `(nspin=2, no_u=18, nk=3, gamma_only=false)`. Records contain only `(0,0,0)`, `(0,0,+0.2)`, and `(0,0,+0.4)`: the two negative k points are omitted. All six k/spin records have raw weight 1.0, 18 states, zero-based IDs 0–17, and complex64 coefficients.
- `duplicate.selected.WFSX` (`duplicate` run's `o2.selected.WFSX`), SHA-256 `447c50dea65a7eacda77d8562116f583467a7b3a4a025e997f7b7fd0427567dc`: WFSX header `(nspin=2, no_u=18, nk=5, gamma_only=false)`. The actual records preserve `(0,0,0)`, `(0,0,+0.2)`, `(0,0,+0.2)`, `(0,0,+0.4)`, and `(0,0,-0.4)`, duplicating `+0.2` and omitting `-0.2`. All ten k/spin records have raw weight 1.0, 18 states, zero-based IDs 0–17, and complex64 coefficients.

## Retained input hashes

- `o2_missing_selected.fdf`: SHA-256 `07a2540f7821f9b3028c6f03a14f63d68d7d972007c949853109a8514692baf3`.
- `o2_duplicate_selected.fdf`: SHA-256 `42fb22691ea5e4d0453bbc7b41d0e160252fbd6e2e1f44b2730845faef6e5fbf`.
