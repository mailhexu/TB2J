# Nonstationary Ward Identities on a Three-Site Oracle

Companion script: `ward_identities_three_site.py` (assertion-checked;
run in the `mydev` environment). Status: **all assertions pass**
(2026-09-25).

Story 001 (spiral-first-order-response), TEST-002, FR-006, FR-007.
Companion to `spiral_state_mft.md` (stationary gates) — this report is
the gradient-corrected, NON-stationary generalization.

## Pair model (classical, pair-once)

With planar moments
$\mathbf e_i=(\sin(\Theta_i+\beta_i)\cos\delta_i,\ \sin\delta_i,\
\cos(\Theta_i+\beta_i)\cos\delta_i)$ and
$E=-\sum_{i<j}J_{ij}\mathbf e_i\cdot\mathbf e_j$, at
$\beta=\delta=0$:

$$
g_i^{\beta}=\sum_{j\ne i}J_{ij}\sin(\Theta_i-\Theta_j),\qquad
g_i^{\delta}=0,
$$
$$
C^{\delta\delta}_{ij}=-J_{ij},\qquad
C^{\beta\beta}_{ij}=-J_{ij}\cos(\Theta_i-\Theta_j),\qquad
C^{\delta\beta}_{ij}=0\quad(i\ne j),
$$

diagonals from $\cos(\Theta_i-\Theta_j)$ sums. The nonstationary
(gradient-corrected) Ward identities:

$$
C^{\beta\beta}\mathbf 1=0,\qquad
C^{\delta\delta}\sin\boldsymbol\Theta=\cos\boldsymbol\Theta\odot\mathbf g^{\beta},
\qquad
C^{\delta\delta}\cos\boldsymbol\Theta=-\sin\boldsymbol\Theta\odot\mathbf g^{\beta}.
$$

Only at a stationary reference ($\mathbf g=0$) do the out-of-plane
vectors become Hessian zero modes; the existing stationary gates are
that special case and must not be applied when $\mathbf g\ne0$.

## Gradient inverse problem and ordered pairs

The map $J\mapsto\mathbf g^{\beta}$ for three sites has **rank 2**
($\sum_i g_i^{\beta}=0$ identically): one configuration's gradients
cannot recover three pair couplings — the null direction is
$(1,1,1)$. TB2J's `SpinIO.exchange_Jdict` stores ordered pairs, each
HALF the pair-once value:

$$
g_i^{\beta}=-2\sum_{j\ne i}J^{\rm ord}_{ij}\sin(\Theta_i-\Theta_j)
=-\sum_{j\ne i}C^{\delta\delta}_{ij}\sin(\Theta_i-\Theta_j).
$$

An inversion-symmetric $J_1$-$J_2$ chain has zero local torque at every
pitch while $\mathrm dE/\mathrm dq$ can be nonzero (at $q=\pi/3$,
$J_1=1$, $J_2=0$: $\mathrm dE/\mathrm dq=\sqrt3/2$): a global boundary
twist is not a local rotation.

## Electronic oracle (per-site lab coordinates)

A three-site electronic ring with spin-scalar hopping and co-rotating
fields $B_f(\cos\Theta\,\sigma_z+\sin\Theta\,\sigma_x)$ is globally
SU(2) invariant. The functional-general gradient-corrected scalar Ward
identities, with the frozen-FD curvature (V2 tip-back included) and
per-site gradients in the $(a,\mu)$ lab coordinates:

$$
R_y:\ C^{\beta\beta}\mathbf 1=0,
$$
$$
R_x:\ \sum_j g^{\delta}_j\cos\Theta_j=0
\quad(\text{first order}),\qquad
\cos\boldsymbol\Theta^{T}C^{\delta\delta}\cos\boldsymbol\Theta
+\sum_j g^{\beta}_j\sin\Theta_j\cos\Theta_j=0 ,
$$
$$
R_z:\ \sin\boldsymbol\Theta^{T}C^{\delta\delta}\sin\boldsymbol\Theta
-\sum_j g^{\beta}_j\sin\Theta_j\cos\Theta_j=0 ,
$$

from the generator decompositions $v(R_x)_j=-\cos\Theta_j\,\delta_j$,
$a(R_x)_j=\sin\Theta_j\cos\Theta_j\,\beta_j$;
$v(R_z)_j=\sin\Theta_j\,\delta_j$,
$a(R_z)_j=-\sin\Theta_j\cos\Theta_j\,\beta_j$. These live in the
PER-SITE lab coordinates: the generators' amplitudes vary within each
orbital family, so the folded per-orbital coordinates (shared by all
cell replicas) cannot express them.

## Checks and observed values

| Check | Content | Observed |
|-------|---------|----------|
| [A] symbolic pair model | gradient formula, $g^{\delta}=0$, C blocks, gradient-corrected Ward identities, J-fit rank 2, ordered-pair factor 2, flat-chain zero torque with nonzero $\mathrm dE/\mathrm dq$ | exact |
| [B] classical FD oracle | FD gradient vs symbolic (dev $2.3\times10^{-10}$); FD C blocks ($\le10^{-6}$); Ward residuals $\le10^{-8}$ | pass |
| [C] electronic ring | $C^{\beta\beta}\mathbf 1$ residual $1.2\times10^{-8}$; $R_x$ first-order $\delta$ Ward $8.8\times10^{-17}$; $R_x$/$R_z$ scalar Ward $1.4\times10^{-9}$/$5.8\times10^{-9}$; splitting $=B_{\rm local}$ exact | pass |

The FR-007 pair-once prediction
$g_i=-\sum_{j\ne i}C^{\delta\delta}_{ij}\sin(\Theta_i-\Theta_j)$
is exact for the classical pair model checked in [A]/[B]. This script
does not compute that prediction for the generic electronic ring in [C];
the production comparison is separately gated on pairwise-isotropic
assumptions.

## Implementation consequences

1. Nonstationary diagnostics evaluate Ward residuals against the
   gradient-corrected forms above; a nonzero gradient turns the old
   zero-mode gates into these identities instead of a failure.
2. Curvature blocks for Ward checks must include the V2 tip-back
   diagonal ($-B_f\langle\sigma_z\rangle$ per rotation squared on
   $C^{\beta\beta}$/$C^{\delta\delta}$ diagonals); two-kernel-only
   constructions violate $C^{\beta\beta}\mathbf 1=0$.
3. The gradient-corrected identities constrain per-SITE quantities;
   folded per-orbital gradients/curvatures must be unfolded to the
   $(a,\mu)$ lab coordinates (full SU(2) inverse Bloch transform)
   before testing them.
4. The first-order $R_x$ identity $\sum_j g^{\delta}_j\cos\Theta_j=0$
   is an additional check on the delta channel for time-reversal-broken
   (complex-hopping) references.
