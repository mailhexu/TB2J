# QE Hubbard joint frame: exact trace and normalization contracts

The authority is [`qe_hubbard_joint_frame.py`](qe_hubbard_joint_frame.py).
Run it with the `mydev` environment. This report describes algebraic
contracts for the approved QE DFT+U interface; it does not certify a native
QE exporter, a pseudopotential family, or physical exchange accuracy.

On 2026-10-26, ordinary mydev execution completed all four named checks.
The run used exact symbolic/rational arithmetic and no QE runtime data.

## Assumptions and units

- Each source is separable in its own actual column frame. XC and native
  Hubbard frames need not coincide, be orthogonal, or be linearly independent.
- Site ownership labels the coefficient block, not the spatial support of a
  column. A full atomic filter can mix columns owned by different sites.
- The two resolvents and the two site frames are independent complex matrices.
  Resolvents at a complex energy are not assumed Hermitian.
- Stored rows have the convention `C[n,p] = <p|n>` used by
  `TB2J/projector_green.py:1180-1182,1343-1351`. A transpose of the stored rows,
  rather than their adjoint, occurs on the left side of the spectral sum.
- Atomic normalization uses the existing full per-k Hermitian channel filter.
  Its inverse, SVD, Lowdin, and Tikhonov choices remain defined by
  `TB2J/projector_green.py:1221-1312`; this script does not introduce a
  second numerical filter implementation or loosen its conditioning rules.
- Native Hubbard columns already contain their native generalized-overlap
  action and any requested native orthogonalization. No atomic exchange-frame
  filter, second generalized overlap, or joint Gram inverse acts on them.
- Coefficients and channel filters use the normalized computational-basis
  convention. Potential blocks have energy units and Green functions inverse
  energy units. The traced two-vertex/two-Green kernel is dimensionless before
  energy integration. No Ry/eV scale is fitted by these identities.

## Executable checks

### `joint_trace`

The exact rectangular cyclic-trace identity is checked with independent
symbolic complex entries, two physical dimensions and three channels at each
site. Each site block contains a two-channel XC block and a separate Hubbard
block. The joint Gram determinant is identically zero. The trace still equals
the direct physical-space operator trace; no inverse occurs in this identity.
Finite dimensions make the cyclic reordering legitimate. This assertion does
not make an inherited finite-projector XC approximation complete in full PW
space.

### `mixed_terms`

An exact complex example has two sites, full intersite atomic normalization,
two native Hubbard channels, and four joint channels in three physical
dimensions. Independent symbolic source strengths keep XC-XC, XC-U, U-XC,
and U-U contractions identifiable. Every contribution is nonzero. The full
joint trace equals the direct operator trace, while adding the two
isolated-source traces omits the two nonzero mixed terms.

### `row_filter_transpose`

A generic symbolic Hermitian filter, including a complex off-diagonal entry,
acts on the original XC coefficient rows only. The script checks both-sided
XC-XC dressing, one-sided XC-U and U-XC dressing, and an unchanged U-U block.
Negative controls distinguish the required transpose from multiplication by
the untransposed filter, repeated atomic filtering, and filtering native U.
This fixes the row/conjugation convention before any consumer cutover.

### `fourier_filter`

Two distinct full per-k atomic filters, including a positive semidefinite
rank-truncated example, are applied before concatenation and the Fourier
sum. The coefficient route matches blockwise spectral dressing. Applying the
first k-point's filter after the sum does not match. The example includes
complex Fourier phases and native Hubbard channels that remain unchanged.

## Native evidence still required

These assertions do not measure QE `wfcU`, native occupations, or native
rotations. Stories 002-003 require independent normalized full-PW action
trials, including directions outside the beta span, and independent native
occupation/source/rotation comparisons. Budgets remain relative action
`<=1e-10`, analytic-zero absolute action `<=1e-12 Ry`, derivative `<=1e-6`
with finite-difference step convergence, and all existing NC1.1 gates.
Later runtime Stories must reuse the existing filter implementation, retain
all mixed Green blocks, and set canonical `overlap_k=None` only after the
original atomic frame has been normalized once. Original atomic Gram and
filter controls remain provenance. Existing no-U behavior and independent
FR-NC/mixed-family production refusals are not changed by this contract.
