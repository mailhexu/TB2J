# Multi-site finite-grid transform and basis-phase authority (Story 009)

Executable authority: `qspace_multisite_transform.py` (this directory).
Extends, without weakening, the one-site authorities
`qspace-finite-grid-transform` and `qspace-unrenormalized-kernel`.

## Model class (declared boundary)

Real orbital gauge, collinear spin-diagonal channels, no SOC, complete
uniform Gamma-centered odd mesh closed under k -> -k - q, orthonormal
basis, cell-gauge Hamiltonian h(k) = sum_R exp(i k.R) h(R) (basis-position
phases live in the hopping values, never as extra projection factors).
Shifted/even meshes, SOC, noncollinear references, nonorthogonal bases and
the position-twist (q.tau generalized-Bloch) convention are outside.

## S1 — inverse-transform phase (the failing-before defect)

With the production real-space Green convention
`G(R) = (1/N) sum_k exp(-2 pi i R.k) G(k)` (`TBGreen.get_GR`,
`k2Rfactor = -2 pi i`), the real-space ordered-pair kernel

    T_ij(R) = tr[D_i G^up_ij(R) D_j G^down_ji(-R)]

satisfies, as rational functions of the energy,

    T_ij(R) = (1/N) sum_q exp(+2 pi i R.q) K1_ij(q),   K1_ij(q) = sum_k tr[D_i G^up_ij(k) D_j G^down_ji(k+q)] / N.

For the single-channel K1 kernel, the MINUS phase yields `T_ij(-R) =
T_ji(R)`.  The production driver transforms the channel-AVERAGED kernel
(K1 + K2)/2 with the per-q imaginary part, which under the former minus
phase produced the SYMMETRIZATION `[T_ij(R) + T_ji(R)]/2`: every
inter-sublattice R != 0 entry was the mean of the correct and transposed
values (deviation half the transposition gap) while every R-even entry
(diagonals, R = 0) was exact — invisible for the approved one-site
evidence, where J(R) = J(-R).

Verified result `multisite_plus_phase_identity = 0` (difference of the two
sides over the ordered pairs (A,B), (B,A) and R = 0, 1, 2) and
`multisite_minus_phase_counterexample = 25605388728125/994469785538463`
(the exact nonzero T - (minus-phase transform) value of the model below).

## S2 — imaginary-extraction placement

The per-q imaginary part does not commute with the phase transform:

    sum_q e^{+i q R} Im K(q)  !=  Im sum_q e^{+i q R} K(q)

on asymmetric inter-sublattice pairs.  The production `integrate` extracted
`np.imag` per q per pair BEFORE the transform, destroying the odd-in-q
component the R != 0 pairs need.
`imag_extraction_order_difference = 25730058728125/1988939571076926`.
For the even one-site kernel (Im part odd in q, Re part even) both orders
agree — the one-site evidence is unchanged.

## S3 — channel-average mesh-sum premise

The coded symmetric kernel (K1 + K2)/2 transforms to the same
imaginary-projected real-space pair as the single-channel K1 on the
complete mesh (`channel_average_mesh_sum = 0`), extending the approved
one-site reciprocity claim of `qspace-unrenormalized-kernel` to
inter-sublattice pairs.  The complex integrands differ at fixed (q, E);
only the Im-projected mesh sum is pinned.

## S4 — cell-gauge basis-phase premise

A constant onsite gauge U = diag(1, e^{i phi}) applied to both channels
leaves the ordered-pair kernel exactly invariant
(`cell_gauge_block_projection = 0`): the orbital-block projection
(`Gk_atom`) carries no position phase and needs none.  All basis-position
dependence sits in the hopping values.  A k-dependent position twist (the
q.tau generalized-Bloch convention) is a different model class, outside
this authority.

## Model

Two inequivalent sublattices A, B per cell; onsite ea = eb = 1/10,
splittings dA = 7/20, dB = 1/4, intra-cell hopping 2/25, asymmetric
inter-cell hoppings tR = 1/20, tL = 3/100 (eV); three-point mesh
{1, omega, omega*} with omega = (-1 + i sqrt(3))/2.  Up channel occupied,
down channel empty; evaluations at E = -1/2 (inside the up band, below
mu = 0).  Genericity: dA != dB and tR != tL make the inter-sublattice
response R-asymmetric, which is what exposes both defects.

## Production consequences (implemented in TB2J/exchange_qspace.py)

1. `q_to_r` transforms the COMPLEX per-q kernel integrals (`Jq_c`, `Kq_c`,
   `Jnorm_c`) with the PLUS phase and extracts `np.imag` after the
   transform, with the 1/nqpt factor unchanged.
2. `integrate` retains the complex per-q contour integrals alongside the
   real per-q responses `Jq`/`Kq`/`Xq` (unchanged semantics: exact for
   even kernels, per-q diagnostics otherwise; `calculate_raw()` output
   unchanged).
3. `bruno_renormalize` builds `Jnorm_c = Jq_c + (B C^+ B^T)/2` so the
   Bruno-renormalized table follows the same transform.

## Oracle (independent)

`tests/test_qspace_multisite_oracle.py` reproduces the full pair table of
the two-route comparison (real-space `ExchangeCL2` vs reciprocal-space
`ExchangeCLQspace`) on this model class.  Failing-before record: with both
defects active the inter-sublattice entries disagreed by 8.2e-4 eV
(exact transposition signature); after the corrections the routes agree to
1.7e-16.  Reciprocity J_ij(R) = J_ji(-R) asserted on the corrected table.

## Human Review

Pending (Story 009 review).  The one-site approvals of the two extended
authorities are unaffected: their declared domains exclude
multi-sublattice orbital phases, which this authority now covers.
