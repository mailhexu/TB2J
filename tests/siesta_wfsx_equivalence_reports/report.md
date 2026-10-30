# SIESTA WFSX full-array equivalence report (story 008)

- Story: `tb2j/siesta-wavefunction-backed-exchange-without-electronic-diagonali/story/008`
- Generated: 2026-10-30T19:25:16+00:00 UTC (wall 41.4 s)
- Command: `python tests/test_siesta_wfsx_equivalence.py --work-root /home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T182102Z-siesta-wfsx-equivalence-46ff67c1de4b4ca29cd0181a77b3489b/scratch/acceptance-final --output-json /home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_equivalence_reports/observations.json --output-md /home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_equivalence_reports/report.md --native-trace`
- TB2J worktree: `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j` head `e246fdb77e732c501f03a19b42e9452352c095af` (baseline e246fdb, match: True)
- HamiltonIO worktree: `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio` head `da9881f77082e9516759e84381242ed3e2a02bfa` (baseline da9881f, match: True)
- Report driver: `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/test_siesta_wfsx_equivalence.py` sha256 `a385785636b44914143d322333816618d3f15211949436b9ceed5c79256d8264` (tracked: False, clean: False) - uncommitted report infrastructure, not a commit
- Identity citations: `memnotes/Projects/TB2J/mathdocs/siesta-wfsx-spectral.md`

**Overall: PASS** - 20 cases, 826 array rows, 0 array failures, 0 guard failures, 0 key-set failures, trace rows 8 (0 failed).

Warnings:
- report driver is not tracked in git yet; recorded by content hash only (expected)

## Thresholds (unchanged Architecture table)

| quantity group | atol | rtol | unit |
| --- | --- | --- | --- |
| H_S_H0_Delta_HR0 | 2e-05 | 1e-05 | eV (H), dimensionless (S) |
| green_functions | 2e-05 | 0.0001 | 1/eV |
| populations_charges_spins | 0.0001 | 1e-05 | electrons |
| exchange_tensors | 1e-06 | 0.0001 | eV |

## Case results

| case | green mode | arrays | worst ratio | array fails | guard fails | errors | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collinear/full | default | 47 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/internal_fold | default | 93 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/representative | default | 93 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/shuffle_permute_wrap | default | 47 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/gamma_only | default | 31 | 0.1908 | 0 | 0 | 0 | PASS |
| collinear/nonclosed_shifted | default | 39 | 0.1124 | 0 | 0 | 0 | PASS |
| nc4/full | default | 31 | 0.08289 | 0 | 0 | 0 | PASS |
| nc4/shuffle_permute_wrap | default | 31 | 0.08289 | 0 | 0 | 0 | PASS |
| collinear/half_boundary | default | 1 | 4.443e-14 | 0 | 0 | 0 | PASS |
| nc4/spinor_reduction_refused | default | 0 | - | 0 | 0 | 0 | PASS |
| collinear/full | contravariant | 47 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/internal_fold | contravariant | 93 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/representative | contravariant | 93 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/shuffle_permute_wrap | contravariant | 47 | 0.06959 | 0 | 0 | 0 | PASS |
| collinear/gamma_only | contravariant | 31 | 0.1908 | 0 | 0 | 0 | PASS |
| collinear/nonclosed_shifted | contravariant | 39 | 0.1124 | 0 | 0 | 0 | PASS |
| nc4/full | contravariant | 31 | 0.08289 | 0 | 0 | 0 | PASS |
| nc4/shuffle_permute_wrap | contravariant | 31 | 0.08289 | 0 | 0 | 0 | PASS |
| collinear/half_boundary | contravariant | 1 | 4.443e-14 | 0 | 0 | 0 | PASS |
| nc4/spinor_reduction_refused | contravariant | 0 | - | 0 | 0 | 0 | PASS |


## Entry trace matrix

- Source: `test_siesta_wfsx_no_hs_execution.trace_entry_cases` under `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T182102Z-siesta-wfsx-equivalence-46ff67c1de4b4ca29cd0181a77b3489b/scratch/acceptance-final/equivalence-p6zvt79k/trace_matrix` (native_trace=True, method `/bin/strace -f -s 200 -e trace=open,openat,openat2`), rows 8/8

| entry | fixture | green mode | impl | native | failed checks | result |
| --- | --- | --- | --- | --- | --- | --- |
| console | collinear_nspin2_reduced | default | installed-script | opens=4348, forbidden=0 | 0 | PASS |
| console | collinear_nspin2_reduced | contravariant | installed-script | opens=4345, forbidden=0 | 0 | PASS |
| python | collinear_nspin2_reduced | default | gen_exchange_siesta | opens=4340, forbidden=0 | 0 | PASS |
| python | collinear_nspin2_reduced | contravariant | gen_exchange_siesta | opens=4340, forbidden=0 | 0 | PASS |
| console | noncollinear_nspin4_full | default | installed-script | opens=4290, forbidden=0 | 0 | PASS |
| console | noncollinear_nspin4_full | contravariant | installed-script | opens=4290, forbidden=0 | 0 | PASS |
| python | noncollinear_nspin4_full | default | gen_exchange_siesta | opens=4285, forbidden=0 | 0 | PASS |
| python | noncollinear_nspin4_full | contravariant | gen_exchange_siesta | opens=4285, forbidden=0 | 0 | PASS |

## Scope and provenance limits

- Only Green functions and operators are compared across independent diagonalizations; eigenvectors and partial spectra are never compared.
- H0 is the requested-mesh arithmetic (mesh-periodized) average; no continuum H(R=0) identification is claimed [memnotes/Projects/TB2J/mathdocs/siesta-wfsx-spectral.md#mesh_onsite_alias].
- float32 WFSX source coefficients bound the recoverable accuracy [memnotes/Projects/TB2J/mathdocs/siesta-wfsx-spectral.md#float32_conditioning_report]; the LU reciprocal-condition guard (refusal below 1e-10) can reject a fixture.
- The ordinary oracle is the story-002 same-run H/S parser (SislParser, complete_spectrum=True); the recorded 23a9903 HSX-offset question remains a governance item outside this report and no ordinary-path code was changed here.
- collinear/half_boundary mutates only k-point headers on a copy (coordinate-only synthetic boundary inherited from HamiltonIO tests/test_wfsx.py::test_nonzero_self_inverse_coordinate_is_not_doubled); no H/S operator equivalence is claimed at the synthetic coordinate.
- default and contravariant Green modes are each compared against the same-mode oracle only; they are never asserted mutually equal.
- Gamma-only and non-closed shifted subsets are exercised on the magnetic collinear representative; the nspin=4 spinor case is exercised on the full grid and the shuffled/permuted/wrapped request. Scalar reduction of spinor exchange must be refused (UnsupportedSpinMode) and is checked as a refusal, never attempted.
- This report driver is uncommitted test infrastructure; git HEADs and the driver content hash are recorded and are not claimed as commits.
- The trace matrix is produced by tests/test_siesta_wfsx_no_hs_execution.py::trace_entry_cases; native open tracing only with --native-trace (requires /bin/strace and the installed siesta2J.py console script); permanent tests never require ptrace.
- No production source, threshold, fixture or golden data was modified by this command; fixtures are consumed read-only and all mutations happen on copies under the caller-provided work root.
