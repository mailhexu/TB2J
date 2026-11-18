# Magnetic Force Theorem About a GBT Spin Spiral

Companion script: `spiral_state_mft.py` (assertion-checked; run in the
`mydev` environment). Status: **all assertions pass** (2026-09-24).

## Purpose

This derivation treats a fixed, converged, torque-free spin spiral as
the magnetic reference. It does **not** infer exchange by scanning the
energy over different spiral wavevectors. Instead, it follows the
ordinary collinear LKAG construction: rotate individual moments
infinitesimally, evaluate the frozen-potential second-order response,
and map that local curvature to exchange tensors. The reference Green
function is the full spinor resolvent of the GBT Hamiltonian.

## Local-frame perturbations

For an in-plane reference moment

$$
 \mathbf n_a=(\sin\Theta_a,0,\cos\Theta_a),\qquad
 \mathcal B_a=B(\cos\Theta_a\sigma_z+\sin\Theta_a\sigma_x),
$$

use two transverse coordinates: an out-of-plane tilt $\delta_a$ and an
in-plane phase rotation $\beta_a$. The rotated local field is

$$
 \mathcal B_a(\delta_a,\beta_a)=B\left[
 (\cos(\Theta_a+\beta_a)\sigma_z+
  \sin(\Theta_a+\beta_a)\sigma_x)\cos\delta_a
 +\sigma_y\sin\delta_a\right].
$$

The script verifies symbolically that

$$
 V_{1a}=B\left[\delta_a\sigma_y+
 \beta_a(-\sin\Theta_a\sigma_z+\cos\Theta_a\sigma_x)\right],
 \qquad
 V_{2a}=-\frac{\delta_a^2+\beta_a^2}{2}\mathcal B_a.
$$

## Magnetic-force-theorem curvature

With occupations frozen at the spiral reference, the eigenbasis form is

$$
 E^{(2)}=\operatorname{Tr}(V_2\rho_q)+\frac12\sum_{n\ne m}
 \frac{f_n-f_m}{\varepsilon_n-\varepsilon_m}|(V_1)_{nm}|^2.
$$

The antisymmetrized quotient is essential for symmetry-protected
spiral degeneracies. Pairs with equal energy and occupation are handled
as one degenerate subspace; their first-order splitting cancels from
the four-point curvature. The equivalent production expression is the
full-spinor contour trace

$$
 E^{(2)}=\operatorname{Tr}(V_2\rho_q)-\frac1\pi\operatorname{Im}
 \oint f(z)\operatorname{Tr}[V_1G_q(z)V_1G_q(z)]\,dz.
$$

On three random six-site rings, both local transverse channels agree
with frozen-occupation finite differences to relative error
$4.24\times10^{-6}$ or better.

## Heisenberg mapping and signs

For a pairwise energy
$E_{ab}=-J_{ab}\mathbf e_a\cdot\mathbf e_b$, SymPy verifies

$$
 C^{\delta\delta}_{ab}=-J_{ab},\qquad
 C^{\beta\beta}_{ab}=-J_{ab}\cos(\Theta_a-\Theta_b),\qquad
 C^{\delta\beta}_{ab}=0\quad (a\ne b).
$$

Therefore

$$
 J^{\rm spiral}_{ab}=-C^{\delta\delta}_{ab},
$$

and the correct pairwise in-plane diagnostic is

$$
 \Delta_{\rm NH}=\left\|C^{\beta\beta}+J^{\rm spiral}\circ
 \cos\Theta\right\|,
$$

with diagonals fixed by the zero-mode sum rule. The plus sign in this
norm is required because the pairwise baseline is
$C^{\beta\beta}=-J\circ\cos\Theta$.

## Torque and Goldstone gates

The in-plane uniform mode is a global $R_y$ rotation and obeys
$C^{\beta\beta}\mathbf1=0$ at every reference. The out-of-plane
$\cos\Theta_a$ and $\sin\Theta_a$ modes become exact Goldstone modes
only when the reference is torque-free. The numerical checks resolve
these modes at $10^{-6}$ scale for the commensurate torque-balanced
reference and detect $10^{-2}$ violations for a non-torque-free
reference. A production calculator must reject or clearly flag the
latter.

## Collinear anchor

At $q=0$, the two transverse blocks coincide:

$$
 C^{\delta\delta}=C^{\beta\beta}=M,\qquad C^{\delta\beta}=0.
$$

The script verifies this within $10^{-6}$ and recovers
$J_{ab}=-M_{ab}$ off diagonal. This closes the numerical reduction to
the collinear response matrix already matched to the LKAG dimer kernel
in `spiral_force_theorem_J.py`.

## Spiral-frame magnon gate

For an isotropic flat screw, the extracted exchange must reproduce the
Tóth--Lake result

$$
 \omega(k)=S\sqrt{A_kC_k},\quad
 A_k=J(q)-\frac{J(k+q)+J(k-q)}2,\quad C_k=J(q)-J(k).
$$

The script verifies symbolically the zeros at $k=0$ and $k=\pm q$ and
the ferromagnetic limit $q\to0$.

## Implementation consequences

1. The reference is a converged per-$q$ TBUpy spiral, not a collinear
   potential reused across $q$.
2. Rotate $V_U$ and its occupation matrix unitarily with the local
   frames; preserve the run's double-counting convention.
3. Use unfolded real-space blocks of the full spinor $G_q$ in the
   existing `ExchangeNCL`/`ExchangePert2` four-component tensor path.
4. Emit torque norms, Goldstone residuals, the $q=0$ anchor, and
   $\Delta_{\rm NH}$ with every result.
5. Treat an $E(q)$ scan only as a secondary consistency check.
