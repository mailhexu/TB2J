# Changelog

## Unreleased

### Split-SOC documentation and examples

- Documented the ABINIT NC split-SOC workflow: a new page covering the
  abinao WFK → `abinit.nc_pao_hs` v2 / `abinao.nc_soc_ks` v1 production
  chain, the SHA-256 hash join, PAO dualization `B = S^-1 C`, units
  (eV sidecar / Hartree-on-disk PAO_HS), the full refusal catalog, the
  single-reference transverse-block scope and the FR-032 tangent
  projection gate (iodine I2: 7.114 meV vs 10 meV tolerance,
  projection-only, not a full-tensor proof).  Added the sidecar
  on-disk contract to `abinit_savetb2j_schema.rst` and a real runnable
  example `examples/projector_green/abinit_nc_i2_split_soc.py` (smoked
  on the I2 fixture: anchor 1.7e-15 relative over 28 pairs).
- Documented the per-backend split-SOC workflows: a shared strength-zero
  overview in `projector_green.rst`, a new GPAW page with the CLI reference,
  MAE comparison how-to and provenance contract, and a new ABINIT PAW page
  covering the schema-1.1 `soc_pauli` recipe, loader orientation, driver
  usage, `spinat` sign rules and physical anchors.
- Added `examples/projector_green/abinit_paw_fe_split_soc.py`: runs the
  three-leg PAW driver on a real schema-1.1 export and closes the lam=0
  SOC-off anchor against the collinear `delta_total` exchange, rejecting
  vacuous Gamma-only baselines.

### GPAW split-SOC exchange and MAE

- Added `gpaw_split_soc2J.py`: one old-API collinear no-SOC GPAW checkpoint
  produces three frozen-density second-variational x/y/z exchange legs,
  rotated and merged `SpinIO` tensors, exact GPAW band-energy MAE and a
  tolerance-reported second-order contour comparison. The shared KS-band
  kernel uses all-atom SOC with magnetic-only exchange vertices.

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
