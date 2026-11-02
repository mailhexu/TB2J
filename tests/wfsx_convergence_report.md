# SIESTA WFSX mesh convergence (N/2N/3N)

Generated: 2026-11-02T15:30:11Z

- TB2J: `5b5fde3e11e07985a19f2820c3a29554dab01ec5 (dirty: True)` at `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j`
- SIESTA executable: `/home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta` (sha256 `865ab6f1db36`)
- SIESTA source root: `/home/hexu/projects/siesta_dev/siesta`; base mesh 5; meshes `5x5x5`, `10x10x10`, `15x15x15`
- Invocation root: `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/convergence-final/convergence-20261102T152858475463Z` (fresh child per invocation; prior evidence preserved)

## Method

- Producer: tests/siesta_wfsx_fixtures/measurement_producer.py:generate_measurement_case (bounded measurement extension of the SHA256SUMS-pinned Story 002 producer driver, reusing the existing fixture_driver helpers; unchanged O2 SZP/PBE/100K physical template, repeat_z=1; only the k-grid and the exact template fields named in changed_template_fields differ)
- Pair runner: tests/siesta_wfsx_fixtures/run_benchmarks.py:run_matched_pair (shared benchmark slice; no duplicated measurement pipeline)
- green_mode `default`; cache_policy `warm` (explicit). passed explicitly to run_matched_pair; the runner owns the cold/warm mechanics (per-input POSIX_FADV_DONTNEED cold advice where available; warm = immediately repeated fresh process on the same inputs) and records them per run
- Integration: run_matched_pair defaults: magnetic O, Rcut 4.3 A, nz 8, nproc 1, smearing 100 kB, complete_spectrum True, metadata EF shift 0; kmesh from the measurement mesh; explicit full requested fractional points with uniform weights
- Alignment: successive refinements are compared on the exactly aligned full-tensor R/site domain (all keys, including zero blocks); unequal full R/site key sets refuse the comparison
- Thresholds (approved Architecture, unchanged): max |change| <= 0.01 meV AND max relative change <= 0.01 where |reference| > 0.1 meV, on BOTH successive refinements.
- Tensor unit: meV, verified against tensors_meta.<mode>.unit before any statistic is computed; none; thresholds are evaluated directly in the serialized meV values.
- converged iff BOTH successive refinements meet BOTH thresholds
- each step is judged on its own measured change; monotonic decrease is never assumed
- the relative criterion counts only elements with \|reference\| > 0.1 meV; an active count of zero is an explicit vacuous determination, never a NaN pass
- any non-finite tensor element fails its step
- the same-mesh ordinary/WFSX equivalence proof is carried separately and does not enter the refinement verdict

## Meshes

| requested k-mesh | requested k-points | R-list mesh (odd-coerced, hs/wfsx) | SCF iterations | Fermi (eV) | template sha | pseudo sha |
|---|---|---|---|---|---|---|
| 5x5x5 | 125 | 5x5x5 / 5x5x5 | 9 | -4.76597 | `c275c7fd7b1b` | `d3ecdc324e91` |
| 10x10x10 | 1000 | 11x11x11 / 11x11x11 | 9 | -4.76597 | `c275c7fd7b1b` | `d3ecdc324e91` |
| 15x15x15 | 3375 | 15x15x15 / 15x15x15 | 9 | -4.76597 | `c275c7fd7b1b` | `d3ecdc324e91` |

Requested quadrature points are the explicit full requested fractional list from each measurement manifest (never odd-coerced); the odd R-list mesh is the exchange machinery's separate axis and is reported per run.

## Successive refinement

| step | aligned pairs (elements) | keys only coarse/fine | max abs change (meV) | max relative change | relative active count | verdict |
|---|---|---|---|---|---|---|
| 5x5x5 -> 10x10x10 | 12 (108) | 0/0 | 5.31701e-06 | 3.12175e-08 | 6 | PASS |
| 10x10x10 -> 15x15x15 | 12 (108) | 0/0 | 2.83703e-06 | 1.66569e-08 | 6 | PASS |

Each step is judged on its own measured change; no monotonic improvement is assumed.

## Same-mesh equivalence (separate proof)

Ordinary/WFSX equivalence at each fixed mesh, recorded by run_matched_pair; it does not enter the refinement verdict above.

| requested k-mesh | same_mesh.pass | observations |
|---|---|---|
| 5x5x5 | PASS | `{"green_mode": "default", "cache_policy": "warm", "tensor": {"abs_tolerance": 0.001, "rel_tolerance": 0.0001, "unit": "meV", "reference_records": 12, "comparison_records": 12, "missing_keys": [], "missing_key_count": 0, "extra_keys": [], "extra_key_count": 0, "duplicate_reference_keys": [], "dupl...` |
| 10x10x10 | PASS | `{"green_mode": "default", "cache_policy": "warm", "tensor": {"abs_tolerance": 0.001, "rel_tolerance": 0.0001, "unit": "meV", "reference_records": 12, "comparison_records": 12, "missing_keys": [], "missing_key_count": 0, "extra_keys": [], "extra_key_count": 0, "duplicate_reference_keys": [], "dupl...` |
| 15x15x15 | PASS | `{"green_mode": "default", "cache_policy": "warm", "tensor": {"abs_tolerance": 0.001, "rel_tolerance": 0.0001, "unit": "meV", "reference_records": 12, "comparison_records": 12, "missing_keys": [], "missing_key_count": 0, "extra_keys": [], "extra_key_count": 0, "duplicate_reference_keys": [], "dupl...` |

## Verdict

**CONVERGED.** Both successive refinements meet the approved thresholds (max \|change\| <= 0.01 meV and max relative change <= 0.01 where \|reference\| > 0.1 meV).
