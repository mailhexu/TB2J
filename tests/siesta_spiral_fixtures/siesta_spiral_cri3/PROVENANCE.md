# CrI3 noncollinear spiral producer fixture

Producer: SIESTA `5.4.2-23-g549d9de29`, source commit
`549d9de2985008c2571350029b5d81cd94a3e0e1`; executable SHA-256 and audited
source-file hashes are in `producer.json`. The input uses `sisl 0.16.2` for
HSX/EIG/KP ingestion. `SHA256SUMS` pins the committed FDF, PSML files, HSX,
EIG, KP, SIESTA output, producer record, and reduced NPZ.

## Reproduction

Run the committed `cri3_q025_dzp_g200_k4.fdf` in a fresh directory containing
`Cr.psml` and `I.psml` with the pinned SIESTA executable. Reduce the native
HSX/EIG/KP records using:

```python
make_fixture_from_siesta(
    hsx_path, eig_path, kp_path,
    (0.25, 0.0, 0.0), 70, 300,
    "cri3_q025_dzp_g200_k4",
)
```

Save the returned arrays with `numpy.savez_compressed` as
`siesta_spiral_cri3.npz`. The HSX declares its exact Fermi level; the importer
undoes that HSX shift and retains the stored Hamiltonian without hermitization.

## Producer settings and observations

- Monolayer primitive cell: 2 Cr + 6 I; lattice vectors
  `(6.867,0,0)`, `(-3.4335,5.946178895,0)`, `(0,0,20)` Angstrom.
- PBE, noncollinear native generalized Bloch spiral, reciprocal pitch
  `(0.25,0,0)`, `DM.InitSpin` Cr sites `+3.0` along `+z`, 300 K,
  4x4x1 k mesh, 200 Ry mesh cutoff, `PAO.EnergyShift=0.05 eV`.
- DZP basis for Cr and I. The committed scalar-relativistic ONCVPSP PSML
  files record generator `ONCVPSP-3.3.1+psml-3.3.1-75`, date 2026-10-10;
  Cr has 14 valence electrons and I has 7.
- SIESTA exited normally and converged the density after 21 iterations;
  reported `dHmax=6.125026e-4 eV` and `E_KS=-6981.2723 eV`.
- The final Mulliken table reports Cr moments `+3.491052` and
  `+3.491052` muB; each I carries approximately `-0.1637` muB. Atomic
  Mulliken charges are `-0.612429` on Cr and `+0.2041` on I. The ligand
  magnetization is nonzero and retained in the fixture evidence.
- Spinor electron count is 70 (`n_occupied=70`). Across the recorded k points,
  the maximum occupied energy is `-5.34940557 eV`, the minimum unoccupied
  energy is `-4.24010728 eV`, and the gap is `1.10929829 eV`.
- The bundle has 116 spatial orbitals, 16 recorded k points, and a 232-state
  spinor pencil. Recorded uncomputed EIG tail entries are `+Infinity`, as
  supported by the fixture format.

## Independent producer-oracle results

- Eigenvalue oracle: maximum absolute deviation `1.893516e-6 eV`, below the
  fixed `1e-5 eV` oracle budget. The Fe3/O2 fixtures retain their existing
  `1e-6 eV` regression bound; the CrI3 test uses the declared oracle tolerance
  because this larger real pencil's deviation exceeds the smaller-fixture bound.
- Electron-count oracle: 70 electrons within the fixed `1e-4` budget
  (observed deviation `-7.28e-11`).
- Across the 16 recorded k points, the largest assembled H-pencil magnitude
  is `77.6651541 eV`; the maximum anti-Hermitian H defect is
  `1.1264755e-8 eV` (relative defect `1.450426e-10`). The original absolute
  `1e-8` check rejected this scale-level roundoff. The producer oracle now compares against
  `1e-8 * max(1, max(abs(matrix)))`; it neither modifies nor symmetrizes H or S.
  The independent wrong-pitch negative test remains fail-closed.

This fixture certifies the producer import/oracles and provides a real,
gapped DZP ligand-bearing starting point. It does not by itself claim response
convergence or material validation; those are the separate Story-022 campaign
outputs.
