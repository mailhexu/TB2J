# Gapped O2 molecular-chain import oracle

Producer: SIESTA `5.4.2-23-g549d9de29`; ingestion: sisl `0.16.2`.
`producer.json` identifies the executable and relevant source hashes;
`SHA256SUMS` pins all required raw inputs/outputs and the reduced NPZ.
`O.psf` is the exact upstream `Tests/Pseudos/O.psf` used in the run.

Geometry: O2 bond 1.210 Å along x, periodic spacing 4 Å along z and 7 Å
transverse vacuum. PBE/SZP, 150 Ry mesh, noncollinear spin, reciprocal
pitch (0,0,1/4), four Cartesian KP samples. There are 18 orbitals and 900
stored HSX entries; 576 nonzero H entries are on nonzero cell images,
so this is not an electronically disconnected-molecule q=0 test.

`o2_all_states.out` records 12 electrons, 100 K, and **free-energy-only
convergence** in nine iterations with residual dDmax 6.1e-5 and
`max |H_out-H_in| = 0.0005719325 eV`. Total spin moment at that iteration:
(1.99995, 0.00338, 0.00000) μB. No stationary-torque claim is made.
Earlier attempts with strict H/DM criteria did not converge; they are
not treated as successful physics validation.

The input deliberately disables `diag2kspiral`'s zero-occupation eigenvector
pruning using `DM.OccupancyTolerance -1e-12`. In the audited producer's
loop (`diag2kspiral.F90:185-198`), `abs(qo)>occtol` then retains every
state. Exactly zero-weight density contributions remain zero. This is a
recording configuration, not a change to occupations or H. The complete
36-band `.EIG` is finite and comes from the native spiral diagonalizer.
The ordinary SIESTA `.bands` path calls `diag2k` (no spiral pitch), so it
is expressly **not** used as a spiral reference.

Observed checks:
- Independent native EIG gap: **2.51112651 eV**, between bands 12 and 13
  across the recorded k mesh; printed Ef is strictly inside the gap.
- Every occupied and unoccupied band: max absolute deviation
  **3.89539e-7 eV** after lossless import (budget 1e-5 eV, relative slack 0).
- Imported occupation count: **12.000000000**, at recorded Ef and 100 K
  (absolute budget 1e-4 electrons, relative slack 0).

Reproduce in a fresh directory containing `O.psf`:
run SIESTA with stdin from `o2_spiral.fdf`, then call
`make_fixture_from_siesta("o2.HSX", "o2.EIG", "o2.KP", (0,0,0.25), 12,
100, "o2")` and save its returned arrays with `numpy.savez_compressed`.
This fixture certifies import conventions and a gapped spectrum only;
native-versus-supercell equivalence and torque-state reporting belong to
story 004.
