# Spinor Projector-Green Exchange — Sympy Derivation Report

Story 001 of the SOC spinor projector-Green spec
(`Projects/TB2J/specs/soc-spinor-projector-green`).

Script: `spinor_projector_green.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-22).

## Pinned exchange tensor object

$$
J^{\alpha\beta}_{ij}(E) = -\mathrm{Tr}\!\left[
\sigma_\alpha \Delta_i\; G_{ij}(E)\; \sigma_\beta \Delta_j\; G_{ji}(E)
\right]
$$

- Trace over spinor (2) and projector indices.
- $\Delta_i$: site spin-splitting operator, Hermitian, $a\,\mathbb{1} + \mathbf{d}\cdot\boldsymbol{\sigma}$ (real coefficients), acting in the projector space of its site.
- $G_{ij}$: spinor Green blocks, $2\times2$ in spin $\otimes$ projector space.
- Overall sign chosen so positive $J$ = ferromagnetic, consistent with the collinear kernel.

## Collinear limit (anchor to the validated kernel)

With $\Delta_i = z_i\sigma_z$, $\Delta_j = z_j\sigma_z$, block-diagonal
$G=\mathrm{diag}(g_\uparrow,g_\downarrow)$, $H=G_{ji}=\mathrm{diag}(h_\uparrow,h_\downarrow)$:

$$
J^{xx} = J^{yy} = z_i z_j\,(g_\uparrow h_\downarrow + g_\downarrow h_\uparrow)
\quad\text{(cross-channel)}
$$

- The cross-channel sum is **exactly** the two channels of the validated collinear kernel `Tr[Δ_i G↑_ij Δ_j G↓_ji]` + (↑↔↓) partner (`projector_green.py::projector_exchange_trace`).
- $J^{zz} = -z_i z_j (g_\uparrow h_\uparrow + g_\downarrow h_\downarrow)$: same-channel piece, spin-conserving, excluded by the contour/imaginary-part prescription of the physical exchange.
- $J^{xz} = 0$ identically; $J^{xy} = \mathrm{i}\,z_i z_j (g_\downarrow h_\uparrow - g_\uparrow h_\downarrow)$ and $J^{yx}=-J^{xy}$; these vanish for physical collinear states (FM: $g_\uparrow=g_\downarrow$, $h_\uparrow=h_\downarrow$; AFM cross-sublattice pairs give equal products), so **collinear systems carry no DMI and no anisotropy**, and

$$
J_\mathrm{iso} = \tfrac{1}{2}(J^{xx}+J^{yy}) = z_i z_j\,(g_\uparrow h_\downarrow + g_\downarrow h_\uparrow)
$$

reduces the spinor formula to the collinear kernel result.

## Conjugation structure and the real-part convention

$\mathrm{conj}(\mathrm{Tr}[A\,G\,B\,H]) = \mathrm{Tr}[\bar A\,\bar G\,\bar B\,\bar H]$ (elementwise conjugation; $\sigma_y$ makes the operators complex). Per-component traces are generally complex even for $H=G^\dagger$; the physical pair tensor takes the **real part** — precisely the existing `TB2J.Jtensor.decompose_J_tensor` convention (`Jtensor = Jtensor.real`) — with the imaginary remainder odd under the $(i\leftrightarrow j,\ R\leftrightarrow -R)$ pairing.

## Tensor decomposition (ExchangeNCL conventions, asserted round-trip)

For the real $3\times3$ tensor $J$ (`TB2J/Jtensor.py`):

$$
J_\mathrm{iso} = \tfrac{1}{3}\mathrm{Tr}\,J,\qquad
\mathbf{D} = \tfrac{J - J^T}{2}\big|_{(1,2),(2,0),(0,1)},\qquad
J_\mathrm{ani} = \tfrac{J+J^T}{2} - J_\mathrm{iso}\,\mathbb{1}
$$

with $J_\mathrm{ani}$ symmetric and traceless, and
`combine_J_tensor(J_iso, D, J_ani) == J` (round trip asserted symbolically).

## Verification

- Pauli product/anticommutation identities asserted symbolically.
- All collinear closed forms asserted symbolically.
- Conjugation identity asserted on exact random rational substitutions (1e-12).
- Decomposition identities asserted symbolically on a general real 3x3 tensor.
- Full tensor cross-checked symbolically vs an independent numpy implementation on random complex inputs (1e-12), plus a lambdified diagonal-limit spot check.

## Consequences for implementation (story 002)

1. Kernel entry point: compute $J^{\alpha\beta}(E)$ for $\alpha,\beta\in\{x,y,z\}$ from spinor Green blocks and 2x2 site operators, take Re, decompose via `TB2J.Jtensor.decompose_J_tensor`.
2. Collinear ($nspinor=1$) data feeds the same formula with $\Delta = z\,\sigma_z$ and diagonal $G$; $J^{xx}=J^{yy}$ reproduce the existing channel sum — the bitwise-stability requirement should instead pin the *existing* collinear path unchanged and route spinor data through the new path.
3. The $J^{zz}$ same-channel piece is excluded by prescription; implement the contour with the standard imaginary-part/contour scheme as in the collinear kernel.


## Correction (2026-09-23): Pauli placement and channel mapping

The originally pinned arrangement `J^{ab} = -Tr[(sigma_a Delta_i) G_ij
(sigma_b Delta_j) G_ji]/(4 pi)` is **identically zero** for
block-diagonal (collinear) G: the spin-flip factors make the two
operator products nilpotent.  The Pauli matrices must decompose the
Green function, not multiply Delta.  The implementation now uses the
ExchangeNCL channel matrix

    A^{uv} = Tr[Delta_i G^(u)_ij Delta_j G^(v)_ji] / pi,
    G^(u) = T^u (x) sigma_u  (Pauli components of the spinor G block)

with the mapping `J_iso = Im(A00 - Axx - Ayy - Azz)/8`,
`DMI_i = Re(A0i - Ai0)/8`, `Jani[i,j] = Im(A^{ij}(R) + A^{ij}(-R))/8`.
The identity `(T^0)^2 - (T^z)^2 = G_up G_down` makes the A00-Azz
contraction the algebraic cross-channel (both operator orderings,
hence the extra factor 2 absorbed in the 1/8), so the collinear
reduction reproduces the collinear kernel
`Im integral Tr[Delta G_up Delta G_down]/(4 pi)` exactly.  The same
kernel defect class included a conjugation error in the spinor overlap
dressing (`S^-1 G (S^-1)^T` instead of `S^-1 G S^-1`), which broke the
cubic symmetry of symmetry-equivalent exchange vectors.  Validation:
bccFe at the validated projector-exchange settings reproduces the
collinear reference shell-by-shell (TB2J commit 5281218).
