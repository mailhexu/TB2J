# Changelog

## Unreleased

### Split-SOC documentation and examples

- Documented the ABINIT NC split-SOC workflow: a new page covering the
  abinao WFK → `abinit.nc_pao_hs` v2 / `abinao.nc_soc_ks` v1 production
  chain, the SHA-256 hash join, PAO dualization `B = S^-1 C`, units
  (eV sidecar / Hartree-on-disk PAO_HS), the full refusal catalog, the
  single-reference transverse-block scope and the gate set (SOC-off
  anchor, always-on rank-9 merge invariance gate, FR-032 tangent
  projection gate).  Added the sidecar
  on-disk contract to `abinit_savetb2j_schema.rst` and a real runnable
  example `examples/projector_green/abinit_nc_i2_split_soc.py`.
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
