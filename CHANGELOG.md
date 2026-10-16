# Changelog

## Unreleased

- SIESTA spiral certified exports now bind AC-009 to the complete derivative
  records (`seam_payload_sha256` for dH/dR+dS/dR and
  `derivative_payload_sha256` for dV/dR), including geometry, units, gauges
  and declarations. Partial or missing digest evidence is refused.
- AC-008 verdicts carry the certified `vertex_sha256` and
  `source_bundle_sha256`; +U exports require both to match their response.
  Unknown operator identity requires an explicit `requires_plusu` declaration
  rather than silently assuming a non-+U calculation. Raw-response-only
  exports continue to omit exchange parameters.
- Spiral provenance preserves the revision and independently records
  `code_sha256` over the relevant implementation modules. Ensemble audits
  compare the producer's `declaration` as well as numerical fields.

- All projector interfaces (GPAW, ABINIT NC/PAW/savetb2j, VASP
  native/split-SOC, QE) now bound the real-space pair list by the
  k-mesh supercell instead of extending it purely from `Rcut`: the
  Bloch-sum Green function is exactly periodic under supercell
  translations, so cells beyond the supercell alias cells inside it
  and listing both double-counts those pairs. `_R_grid_for_cutoff`
  infers the mesh divisions from the (IBZ or full-BZ) k-points and
  restricts R to a symmetric alias-free fundamental window; for even
  divisions the mutually congruent +-n/2 classes are excluded to keep
  the kernel's R -> -R closure without duplicates. A warning is
  printed whenever `Rcut` reaches beyond the window (the k mesh is
  then too coarse for that `Rcut`).

Unreleased

### SIESTA spiral import oracle

- Assemble 3D rotating-frame H/S pencils using full orbital-center
  displacements and q/2 phases. Lossless HSX reduction retains Hamiltonian
  entries with zero overlap and undoes sisl's exact HSX Fermi shift.
- Record Fe3 and gapped O2 native SIESTA fixtures, raw provenance/hashes,
  independent eigenvalues and occupation counts. Oracle checks reject
  non-Hermitian pencils and corrupted convention metadata before solving.
  These certify import only, not stationary response or supercell equivalence.

### SIESTA spiral producer gate and magnetic vertex

- Verified a four-cell same-density folding identity for the SIESTA GBT
  pencil (spinor q/2 phases, coset-wrapped bonds) against a 16-atom
  supercell at machine precision; the independent SCFs retain an
  InitSpin-defined helix but differ in their magnetic amplitudes. The
  SIESTA producer does not demonstrate q-controlled texture retention.
- Added the Oroszlány total-H magnetic spin-split vertex, with explicit
  site-orbital partitioning, Pauli sign convention, source H/S/pitch
  provenance and certificate round-trip integrity. Spatial-XC
  reconstruction remains an optional audit, not a production dependency.

### Magnon reference updates

- Recompute J(q=0) from current exchange/reference arrays in Hq; the
  unkeyed cache could violate the Goldstone mode after moment, rotation-axis,
  propagation-vector or exchange updates. Fourier vectorization is retained.

### QE noncollinear spinor export (NC1.1)

- Added QE NC1.1 noncollinear KB spinor export/reader with nine
  native frozen-magnetic-source response blocks, PAW one-center
  magnetic source rotation, complex beta fields, explicit
  source-tangent vertices and x/y/z rank-nine CLI. Legacy NC1.0
  is scalar-only; NC spinor smooth-source runs now have per-species
  admission and native parity gates. A conditioned Hermitian
  beta-Gram inverse acts only on smooth fields, not dual Green
  coefficients or native nonlocal response.
- bccFe same-UPF SOC-off G3 agrees within 0.00395 meV (US) and
  0.00109 meV (PAW). Fully relativistic US and PAW x/y/z
  produce nearest-neighbor-only (`Rcut=2.6 Å`) rank-nine
  tensors with Jiso 17.71744/14.77750 meV and repeated-row
  spreads 8.37e-9/2.09e-8 eV below the 5e-8-eV gate.
  The PAW run needs `conv_thr=1e-12 Ry` in all three
  SCF references; `1e-8 Ry` fails even the first shell.
  Both 6-Å tensors fail the far-shell gate and yield
  **no** extended-range output. Unphysical self and
  out-of-cutoff pairs are excluded before the merge.
  GPAW LDA/PW92 x/y/z references themselves fail their
  unchanged 1e-8-eV first-shell repeated-row gate even
  after tighter SCF; no GPAW full tensor or matched
  QE–GPAW quantitative claim is made.

### QE NC spinor smooth-source extension

- QE NC1.1 now writes and TB2J strictly reads magnetic NC KB spinor
  sites through their beta-projected smooth Pauli field. Native
  nonlocal magnetic and nine-response NC slabs must be measured zero;
  x/y quarter-turns, tilted rotations, finite-difference tangents,
  retained-Gram reconstruction and site-local native Pauli action
  fail closed. A real Fe NC + O US mixed dump reads with separate
  site sources, but its O ligand has an induced magnetic source.
  Because no real zero-source augmented ligand reference passed,
  mixed NC+US/PAW spinor production exchange now refuses;
  research-only strict reading remains available. US/PAW replayed
  from fixed wavefunctions remains byte-identical, and existing
  first-shell US/PAW rank-nine outputs still pass.
- A campaign-only QE ``TB2J_NC_GRAM_DIAG`` sidecar records **per-k**
  complex beta-Gram modes and their Gram-normalized band/occupation/
  Fermi-window spectral weights without changing the NC1.1 binary
  ABI. The official PseudoDojo FR NC Fe 2³ pilot carries 20.19%
  of beta-projected occupied weight in ten discarded modes
  (42.80% in ±0.5 eV of the Fermi level). A numerical eigengap
  is insufficient: TB2J refuses SO-bearing and Gram-truncated NC
  production tensors, including SOC-off runs with the same FR UPF.
  On independent x/y/z 12³/48-band full-BZ FR references,
  every k discards ten of 34 modes at `rcond=1e-6`;
  these carry 20.03% of beta-projected occupied weight and
  43.14% in the ±0.5 eV Fermi window. At `rcond=1e-5`
  and `1e-3`, 13 and 16 modes carry 24.22% and 32.42%
  of occupied weight. Across all three research-only raw
  legs and six first-neighbor vectors, the full-cutoff
  component bracket shifts by at most 0.0273347 meV.
  This does not bound the lost-state physics; the real
  three-reference CLI explicitly refused a FR tensor.
  Below the first gap, `rcond=1e-10` retains all 34
  modes but gives a 3.4746e8-eV operator with
  13.7–16.9-eV Hermiticity defects; the native tangent
  refuses all three legs without manufacturing a value.
  The matched same-FR-UPF SOC-off 12³ x/y/z family also
  discards 43.1286% of Fermi-window weight and its
  public tensor path refuses. Its research-only
  symmetric transverse off-diagonal raw legs at R=(1,0,0)
  are ~1e-9 meV versus ~1.78e-4 meV SOC-on.
  Neither is certified physical anisotropy.
  Scalar NC spinor exchange passes the matched-band same-UPF
  G3 anchor: Jxx=Jyy=14.7910404 meV versus collinear
  14.7932825 meV at 12³ (48 spinor versus 24/spin bands),
  maximum difference 0.0023220 meV below the 0.01 meV gate.

- VASP split-SOC exports now require `LWRITE_TB2J = .TRUE.` in each
  leg's INCAR. The new VASP patch switch defaults to `.FALSE.`; without
  it, neither native nor CSO export is written.

### Split-SOC documentation and examples

- Documented the ABINIT NC split-SOC workflow: a new page covering the
  abinao WFK → `abinit.nc_pao_hs` v2 / `abinao.nc_soc_ks` v1 production
  chain, the SHA-256 hash join, PAO dualization `B = S^-1 C`, units
  (eV sidecar / Hartree-on-disk PAO_HS), the full refusal catalog, the
  single-reference transverse-block scope and the gate set (SOC-off
  anchor, always-on rank-9 merge invariance gate, FR-032 tangent
  projection gate).  Added the sidecar
  on-disk contract to `abinit_savetb2j_schema.rst` and a real runnable
  example `examples/projector_green/abinit_nc_i2_split_soc.py`; aligned
  with the restored strict rank-9 contract (FR-032 default = the 1e-2 eV
  spec value on CLI and API, never derived from the merge's own spread;
  merge_consistency_atol back to the 1e-6 eV refusal default; the
  example reports a refused merge cleanly and exits nonzero).  Smoked on
  the I2 fixture: anchor passes (1.7e-15 relative), merge refuses at
  2.495e-2 eV vs 1e-6 exactly per the story adjudication, per-leg
  artifacts preserved.
- Documented the per-backend split-SOC workflows: a shared strength-zero
  overview in `projector_green.rst`, a new GPAW page with the CLI reference,
  MAE comparison how-to and provenance contract, and a new ABINIT PAW page
  covering the schema-1.1 `soc_pauli` recipe, loader orientation, driver
  usage, `spinat` sign rules and physical anchors.
- Added `examples/projector_green/abinit_paw_fe_split_soc.py`: runs the
  three-leg PAW driver on a real schema-1.1 export and closes the lam=0
  SOC-off anchor against the collinear `delta_total` exchange, rejecting
  vacuous Gamma-only baselines.

### Split-SOC raw rank-nine core cutover

- The split-SOC kernel now emits, per leg, only the *measured* transverse
  `2x2` block (`J_leg`, zero-masked on the leg's `n` row/column with the
  residual recorded) of its right-handed `(u, v, n)` triad; per-leg scalar
  `Jiso`/DMI/Jani from a single reference are no longer produced anywhere.
  This is the ADR-9 tangent rank-nine contract of the split-soc-ks spec.
- New merge `TB2J.split_soc_kernel.merge_transverse_legs`: three x/y/z
  transverse legs give 12 constraints for the 9 entries of the raw real
  lattice tensor (each diagonal measured twice); exact least-squares solve,
  then `Jtensor.decompose_J_tensor` (Levi-Civita DMI).  The legacy scalar
  `TB2J.io_merge` averaging is retired from every split-SOC path (its
  independent scalar/traceless decomposition biases anisotropy: a true
  `diag(1,2,3)` returns `diag(7/6,2,17/6)` through that route).
- Two invariance gates fail closed on every merge: design-matrix rank must
  be 9 (three independent reference axes), and every repeated diagonal row
  must agree within a per-backend `consistency_atol` (GPAW driver 5e-5 eV,
  ABINIT PAW driver 1e-4 eV, ABINIT NC driver 1e-6 eV).  Diagnostics record
  `min_rank`, `max_repeat_deviation`, `max_transverse_mask_residual` and
  `max_reciprocity_residual`.
- Per-leg artifacts are now raw `split_soc_leg.npz` blocks plus
  `split_soc_provenance.json`; merged results are rank-nine SpinIO outputs
  with the merge diagnostics embedded in the provenance.
- Real ABINIT PAW fcc Ni fixture (`a = 3.52` Å, schema 1.1, 28 bands)
  passes the full rank-nine merge: rank 9 on every pair, repeated-diagonal
  agreement 9.0e-8 eV (gate 1e-4 eV), mask residual 7.8e-19 eV, reciprocity
  3.7e-19 eV, merged nn `Jiso` 0.209 meV.  Its 26→28 band-window study is
  *not* converged (4.7e-5 at 1e-6) and stays flagged.
- Noncertification recorded: on the real NC iodine-dimer fixture the
  repeated-diagonal rows disagree by up to 2.5e-2 eV (~25 meV) against the
  1e-6 eV NC gate, so `merge_transverse_legs` refuses and the driver aborts
  before any merged tensor or tangent report exists — no exchange number
  from that fixture is quotable.  The SOC-off anchor still passes
  (4.0e-15 max relative deviation over 12 R-pairs).  The earlier
  pre-cutover "tangent gate passes at 7.1 meV" recording described the
  retired scalar merge and is superseded.
- Documented the VASP split-SOC workflow (`split_soc_vasp.rst`, registered
  in the toctree): **three independent strength-0 SAXIS runs, one per
  x/y/z leg** (`--leg x=RUN_DIR` x3 on `vasp_split_soc2J.py`, with
  `--lam`/`--mode`/`--merge_consistency_atol`/`--no-band-window-study`);
  one run determines only its own transverse plane.  `tb2j_cso.bin`
  v1-v3 contract with **v3 now current** (patch commit `0f9f073`: runtime
  `FELECT`/`INVMC2`/`AUTOA` plus per-ion `POTAE_XCUPDATED`, gated on a
  committed native export; bit-identical patch-vs-installer files, no
  sidecar left behind on native `OPEN` failure), the `E_soc` identity
  oracle, per-leg run-identity + COCC pairing gates, all-atom `W_SO` vs
  magnetic-only vertices, per-leg `split_soc_leg.npz` with schema-2.0
  provenance, and explicit non-claims: merged DMI/Jani await the FR
  full-exchange cross-validation (story-013 cross-code gate).  Real-FeO gate: merged
  rank-nine, repeated-diagonal spread 2.9e-5 eV, nn `Jiso` 7.3836 meV vs
  SOC-off 7.3856 meV.

### GPAW split-SOC exchange and MAE

- Added `gpaw_split_soc2J.py`: one old-API collinear no-SOC GPAW checkpoint
  produces three frozen-density second-variational x/y/z exchange legs,
  merged into the raw rank-nine `SpinIO` tensor, with exact GPAW band-energy
  MAE and a tolerance-reported second-order contour comparison. The shared
  KS-band kernel uses all-atom SOC with magnetic-only exchange vertices.

### ABINIT PAW split-SOC projector exchange

- Read schema-1.1 ``soc_pauli`` in Hartree using the full projector-and-spin
  transpose required by ABINIT cprj; validate all-atom, frozen-density
  provenance and preserve the normalized spinor block in NetCDF round-trips.
- Added a three-leg PAW KS-band SOC driver: balanced up/down band prefixes,
  all-atom SOC and explicitly magnetic-only signed vertices, tensor rotation
  and merge, real window-convergence diagnostics in each leg and merged
  ``SpinIO`` artifacts. Insertion derivatives cannot be written as exchange.
- Real Fe 2x2x2 full-BZ fixture verifies a nonzero 34-meV SOC-off exchange
  shell against schema 1.0; iodine 5p pins complex matrix-element orientation.
  The Fe 22→24 window study reports a ~2.25e-4 change and is not certified
  converged at 1e-6.
- Matched native Fe spinor wavefunction solves at lambda=0 and 0.005
  use the same fixed density; the exact loaded-component consumer path
  reproduces all 8×24 eigenvalue responses to 0.886 meV maximum and
  0.065 meV RMS. Physical Fe ``spinat`` uses the opposite sign of its
  up-minus-down PAW potential trace, not the potential sign itself.


### VASP split-SOC adapter (ADR-7, story 011) — tangent cutover

- `TB2J/interfaces/vasp_split_soc.py` migrated from the legacy
  A-channel/io_merge flow to the shared tangent contract: each collinear
  strength-0 run is one psi-gauge magnetic reference (states are
  SAXIS-frame eigenstates, vertices `Delta sigma_z`, `W_SO` the
  state-space matrix in the same frame); the shared kernel
  (`TB2J.split_soc_kernel.compute_ks_split_soc_exchange`) measures each
  reference's transverse 2x2 (`J_leg` masked on the `n` row/column,
  `mask_residual` longitudinal-spurion diagnostic).  `TB2J.io_merge` is
  no longer used anywhere in this workflow; the merged decomposition
  comes from the rank-nine raw-tensor solve
  (`merge_transverse_legs`, `rotate_transverse_leg`, 12 transverse
  constraints -> exact least squares -> `decompose_J_tensor`).
- The driver consumes **three independent strength-0 references**
  (`leg_artifacts` mapping `x`/`y`/`z` to that run's `tb2j_native.bin`
  + `tb2j_cso.bin`; the dump SAXIS must be parallel to its tag axis).
  One run determines only the transverse plane of its own reference, so
  a rank-nine lattice tensor requires the x/y/z SAXIS campaign (the
  retained FeO collinear_x/y/z layout).  Real-fixture gate: the FeO
  campaign merges at rank 9 with repeated-diagonal spread 2.9e-5 eV,
  reciprocity residual 1.3e-15, inversion-symmetric-pair DMI 9.7e-10 eV,
  and merged NN `Jiso = 8.0381 meV` vs the SOC-off collinear
  `8.0402 meV` (reproducible numbers; the earlier 7.3836 recording was withdrawn); ADR-8 band-window study persisted
  per leg.
- COCC reconstruction convention pinned to the real FeO dump
  (`CRHODE(LP,L) = conj(CPROJ(LP)) CPROJ(L)`, fast_aug order); the
  driver fails fast when `tb2j_cso.bin` and `tb2j_native.bin` come from
  different runs (COCC-vs-CPROJ integrity gate).
- `tb2j_cso.bin` reader synced to the story-010 v1-v3 loader: v2
  band/k provenance (ispin, nkpts, nbands, nb_tot, efermi, native
  writer identity, vkpt/wtkpt) drives a dump-vs-native identity gate in
  `_check_consistency` (same-run enforcement, IBZ k-count comparison);
  k weights must sum to 1 within 1e-8; v3 per-ion XC-updated reference
  potential (`potae_xcr` + constants) is consumed transparently.
  Real-dump behavior gates: Ni nisoc v2 E_soc oracle (-0.08188974 eV vs
  OUTCAR -0.0818897) and FeO collinear_z v2 dump/native identity.
- Shared kernel upgraded in step with `wt-split-soc` (tangent pin):
  `TB2J/split_soc_kernel.py` and `TB2J/projector_green.py` carry the
  physical tangent vertices, `J_leg` leg frames, raw-tensor merge, and
  frame-validation helpers; their test suites
  (`test_split_soc_kernel.py`, `test_spinor_tangent_green.py`) pin the
  contracts on this branch.  The GPAW exporter writer pin migrates with
  the exporter story and is intentionally not carried here.
- New CLI `vasp_split_soc2J.py`: `--leg x=RUN_DIR` (repeat for x/y/z;
  `TAG=native_path:cso_path` also accepted), `--elements`/
  `--index_magnetic_atoms`, `--lam`, `--mode`,
  `--merge_consistency_atol`, `--no-band-window-study`; writes
  per-leg `split_soc_leg.npz` + provenance, the merged TB2J results,
  and `split_soc_provenance.json` (schema 2.0, raw_rank_nine
  diagnostics, per-leg O maps with `O e_z = SAXIS`).

### Packaging

- `pypao` is now optional. Install `TB2J[pypao]` to declare the pypao
  integration dependency.

### Validation foundation (Epic 010) and restored E2E baseline (Epic 011)

- The canonical `SpinIO` result is now the primary scientific validation
  contract; full-text `exchange.out` body comparison is retired in favor of a
  layered oracle (schema -> toleranced quantities -> physical invariants).
  Shared helpers live in `tests/utils/spinio_checks.py`.
- E2E cases are now plain pytest functions; the legacy `metadata.toml`/`runner`
  discovery harness is being retired as cases migrate (`tests/tests/test_e2e_*.py`).
- Registered pytest markers (`tier1/2/3`, `default/slow/gpu/ecosystem`); the
  default `pytest` run deselects `slow`/`gpu`/`ecosystem`. Missing optional
  dependencies/data produce explicit, reasoned skips.
- The default CPU import path no longer pulls in `TB2J.gpu`/JAX: the GPU
  exchange classes in `interfaces/manager.py` and `interfaces/siesta_interface.py`
  are now imported lazily (guarded by `use_gpu`). JAX remains optional.
- CI (`.github/workflows/python-app.yml`) now triggers on `main`/`develop` and
  runs `ruff check` + `ruff format --check` + `pytest` (default profile),
  replacing the dead `master`-triggered flake8 + hardcoded-example path.
  `pyproject.toml` is the single ruff config source (the shadowing `.ruff.toml`
  was removed); pre-existing lint debt was cleared.
- Restored the scientific E2E baseline: Wannier90 SrMnO3 (collinear), Wannier90
  CrI3 (SOC x/y/z merge), and SIESTA CrI3 (collinear) now pass as `SpinIO`-oracle
  tests. The SIESTA `spin=None` xfail is resolved by current HamiltonIO/sisl.

### New supported-interface E2E workflows (Epic 012)

- Added governed public-interface E2E workflows for ABACUS (bcc Fe collinear)
  and SIESTA (bcc Fe collinear), validated through the layered oracle.
  SPR-KKR RuO2 import + magnon bridge remain covered by `test_sprkkr*.py`, and
  exchange editing/supercell by `test_exchange_supercell.py`.
  ... (data curation into the `tests/data` submodule is in progress)


### Wannier90 Wigner-Seitz weights (ws-weights epic)

- `WannierManager` now records which Wigner-Seitz interpolation scheme was
  auto-detected (scheme 1 = global `ndegen`, scheme 2 = per-orbital `_wsvec.dat`)
  in its output description, along with a migration note.
- **Breaking (correctness)**: because HamiltonIO's `WannierHam.gen_ham` now
  correctly divides by `ndegen(R)` (and applies `_wsvec.dat` when present),
  Wannier90-derived exchange constants **will differ from previous TB2J
  versions**. The new values are correct. Re-run any Wannier90-based TB2J
  calculation to update results.
- No `WannierManager` API change; wsvec is auto-detected from the Wannier90
  output directory. To force scheme 1 for A/B comparison, temporarily move
  `{prefix}_wsvec.dat` aside.
