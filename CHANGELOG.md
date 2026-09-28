# Changelog

## Unreleased

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
  and merged NN `Jiso = 7.3836 meV` vs the SOC-off collinear
  `7.3856 meV` (SOC shift 2.0e-6 eV); ADR-8 band-window study persisted
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
