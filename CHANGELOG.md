# Changelog

## Unreleased

### Non-orthogonal SIESTA spin-lattice coupling
- **Fixed invalid exchange values from the CFR contour in the spin-lattice path.** The full-spectrum Green's function required by the exact pencil derivative keeps unoccupied poles above E_F; the finite-temperature CFR pole contour is only valid for windowed spectra and diverges with them present (bcc Fe benchmark: J_1 = -298.8 meV under CFR versus 18.7 meV semicircle and 17.6-18.2 meV for the standard windowed two-channel path, matching the archived SIESTA tutorial). The spin-lattice exchange now defaults to and enforces the semicircle (or legendre) contour, and pins the user's ``emin`` instead of the per-model auto-gap so the contour stays identical between displaced finite-difference runs. Earlier bccFe certification numbers produced under CFR are retracted.

- Added the SIESTA FC spin-lattice entrypoint and executable finite-difference comparison. Raw FC derivatives and sisl Hamiltonians are aligned to the same fixed reference-Fermi gauge.
- Added explicit periodic image-summed response alongside the localized home-cell convention. Full-spectrum pencil inverses retain the virtual states required by `G(dH-z dS)G`.
- Fixed localized magnetic-vertex derivatives at translated bond endpoints and included internal Green-function support beyond the output pair radius. Replaced the old duplicated-endpoint FD oracle with independent displaced-supercell exchange checks.
- The bccFe examples record finite-difference errors and numerical convergence settings; pickle units are eV and eV/Angstrom, not meV.
- Fixed cached spin-lattice initialization by reopening the stored eigenvector and overlap arrays before preparing scalar-spin views.
- Streamed collinear k-space contractions and scalar-spin Fourier vertices avoid full doubled zero-spin arrays. Finite-difference sweeps retain only serialized per-bond data between reference/plus/minus solves.
- Fixed the dense-mesh cached preparation peak: collinear eigensystems and overlaps now go directly into compact disk-backed spin blocks instead of allocating full spinor H/S/eigenvector k-stacks before spilling.
- Validated the bccFe development case with declared rough convergence (5% +0.5 meV/Ang), unchanged strict independent-FD checks (1% +0.1 meV/Ang), and serialized cubic-axis symmetry. Precision convergence is explicitly not claimed.

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
