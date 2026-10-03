# Generalized (Frozen-Band) Pitch Slope

Companion script: `pitch_slope_generalized.py` (assertion-checked; run
in the `mydev` environment). Status: **all assertions pass**
(2026-09-25).

Story 001 (spiral-first-order-response), NFR-005, FR-009. Companion to
`planar_su2_frame.md` and `local_gradient_nonorthogonal.md`.

## Observable and units

For fractional-reciprocal components $q_\alpha$ ($\alpha=x,y,z$), fixed
primitive k mesh, fixed occupations, unchanged fields:

$$
\frac{\partial E}{\partial q_\alpha}=\sum_{\mathbf k}w_{\mathbf k}
\sum_n f_{n\mathbf k}\,
c_{n\mathbf k}^{\dagger}
\Bigl(\frac{\partial H}{\partial q_\alpha}
-\varepsilon_{n\mathbf k}\frac{\partial S}{\partial q_\alpha}\Bigr)
c_{n\mathbf k}
\quad
\left[\frac{\mathrm{eV}}{\text{primitive cell}\cdot q_\alpha}\right],
$$

with $c^{\dagger}Sc=1$. Every class dressing carries

$$
\Delta_{\mu\nu}(\mathbf R)=2\pi\,\mathbf q\cdot\mathbf R+\alpha_\nu-\alpha_\mu,
\qquad
\frac{\partial\Delta}{\partial q_\alpha}
=2\pi\bigl(R_\alpha+\tau_{\nu,\alpha}-\tau_{\mu,\alpha}\bigr),
\qquad
\frac{\partial U}{\partial q_\alpha}
=-\frac{i}{2}\sigma_y U(\Delta)\frac{\partial\Delta}{\partial q_\alpha},
$$

i.e. BOTH vertices move with the pitch: the twist vertex $2\pi
R_\alpha$ and the intra-cell vertex $2\pi(\tau_{\nu,\alpha}-\tau_{\mu,\alpha})$.
Hopping **and** overlap are dressed identically, so
$\partial_q S$ is generically nonzero — the representation itself moves
under a pitch change (unlike the local $\beta/\delta$ gradient, where
$\partial S=0$). The half-shift of the mesh is HELD FIXED across the
$\pm h$ legs (parity of $\mathrm{round}(qN)$ does not change).

## Checks and observed values

| Check | Content | Observed |
|-------|---------|----------|
| [A] symbolic | scalar normalized pencil $(\mathrm dH-\varepsilon\,\mathrm dS)/S=\mathrm d\varepsilon/\mathrm dq$ identically; rational 2x2 matrix pencil with the S-normalized eigenvector | exact |
| [B] analytic derivative | $(\mathrm dH,\mathrm dS)$ each $(3,2n,2n)$ vs element-wise FD of the assembled pencil; $|\mathrm dS|$ materially nonzero; tau-ablation (equal $\tau$) pins both vertices | $\le1\times10^{-8}$ |
| [C] slope vs FD | slope vector (3,) vs fixed-mesh frozen-occupation central FD per component | max dev $4.2\times10^{-9}$ |
| [C] negative control | dropping $-\varepsilon\,\mathrm dS/\mathrm dq$ errs by $2.6\times10^{-2}$ | fails as it must |
| [D] lab cross-check | exact chain rule through the spiral angles; intra-cell components (y, z) == folded to $7.1\times10^{-15}$ | pass |

## The twist component and the finite-ring seam (important)

For the twist component $q_x$ the explicit $N$-cell ring and the folded
primitive mesh are the same system ONLY on the commensurate grid: off
it, the ring's wrap bond carries a phase mismatch $2\pi\,\delta N$ that
the folded mesh does not. The two representations' twist slopes differ
at $O(1)$ for every finite $N$ (observed: folded $0.1114$ vs lab ring
$0.0122$ eV per fractional $q_x$ on the test ring). The folded value is
the normative primitive-cell slope (FR-009, NFR-002 — "an explicit lab
supercell remains a validation oracle only"); the lab-ring twist slope
is **not** an oracle for it. The intra-cell components ($q_y$, $q_z$
here) act identically in both representations and agree to machine
precision — that is the cross-frame validation.

## Implementation consequences

1. `q_derivative` returns $(\mathrm dH,\mathrm dS)$ with shape
   $(3,2n,2n)$, component order $(x,y,z)$ fractional reciprocal, at
   fixed k; the $-\varepsilon\,\partial S$ term is part of the
   definition, not an option, whenever the overlap is dressed.
2. Fixed mesh across the FD/validation legs; state the mesh convention
   in every result (kpts, weights, half-shift).
3. The scalar projection is a path derivative only when the caller
   supplies the direction; the vector is the primary result.
4. Matched-q SCF slopes (FR-010) are a different observable: see
   `constrained_free_energy.md` for the functional the difference must
   use.
