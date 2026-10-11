# Spinor Projector-Green Exchange — Sympy Derivation Report

Story 001 of the SOC spinor projector-Green spec
(`Projects/TB2J/specs/soc-spinor-projector-green`).

Script: `spinor_projector_green.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-22).

> **SUPERSEDED (2026-09-28).** The exchange-tensor object pinned here and the
> 2026-09-23 correction at the bottom of this document are **replaced** by the
> physical tangent-vertex derivation: `spinor_tangent_vertex_green.py` / `.md`
> (story 002). The false-premise claim and the defective A-channel mapping are
> documented there as negative controls. The collinear cross-channel algebra
> below remains valid and is the anchor the tangent vertex reduces onto.
> A single-reference one-shot replay is a **transverse projection only**
> (rank 4 of 9) — not a full tensor.

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

## Consequences for implementation (story 002) — [SUPERSEDED 2026-09-28]

The items below describe the original Pauli-left path and are **superseded**
by `spinor_tangent_vertex_green.md`. Kept only with corrections for the
record:

1. ~~Kernel entry point: compute $J^{\alpha\beta}(E)$ ... take Re~~ ->
   kernel entry point is the tangent vertex on the **full spinor G**
   (`magnetic_tangent_vertices` / `spinor_tangent_pair_matrix` /
   `spinor_tangent_trace`), with `J^{ab} = Im contour K^{ab}/(2 pi)` — not
   `Re`, and no `TB2J.Jtensor.decompose_J_tensor` on per-leg data.
2. ~~Collinear data feeds the same formula with $\Delta = z\sigma_z$ ...~~ ->
   the collinear path is retained unchanged for bitwise stability; the
   tangent path reduces onto it exactly (`J^{xx} = J^{yy} = J_cl`), with
   sign-free `|Delta|` vertices and no `s_i s_j` in the tangent path.
3. ~~The $J^{zz}$ same-channel piece is excluded by prescription~~ ->
   **false for the old vertex**: `Jzz_old = -z_i z_j (g_up h_up + g_dn
   h_dn) != 0` survives any prescription (spurious `Jani_zz` class). The
   tangent vertex has `K^{zz} = 0` exactly.


## Superseded (2026-09-28): the pinned object above and the 2026-09-23 "correction" are replaced

**The 2026-09-23 correction recorded below was false in both premise and
prescription.** Its factual claim — that `-Tr[(sigma_a Delta_i) G_ij
(sigma_b Delta_j) G_ji]` is "identically zero" for block-diagonal collinear
G — is wrong: that arrangement yields exactly the LKAG cross-channel
`z_i z_j (g_up h_dn + g_dn h_up)` (asserted in `spinor_tangent_vertex_green.py`).
Its replacement mapping, the G-Pauli channel reconstruction
`A^{uv} = Tr[Delta_i G^(u)_ij Delta_j G^(v)_ji]/pi` with
`J_iso = Im(A00-Axx-Ayy-Azz)/8`, `DMI_i = Re(A0i-Ai0)/8`,
`Jani = Im(A^{ij}(R)+A^{ij}(-R))/8`, is the actual defect:

- `A^{0i} - A^{i0} = 0` **identically** for collinear `Delta = z sigma_z`,
  for arbitrary (even fully SOC spin-mixed) G components — the mapping
  cannot produce DMI from collinear legs;
- the `Azz` contraction carries a longitudinal self-pair residue — the
  spurious FeO `Jani_zz = -114` meV;
- the antisymmetric assembly `0.5*(dmi[:,None]-dmi[None,:])` is not the TB2J
  Levi-Civita D tensor;
- the claimed conjugation fix `S^-1 G S^-1` is unrelated to this defect class
  and is not endorsed here.

**The correct object is the physical local magnetic rotation vertex on the
full spinor G**, pinned and assertion-checked in
`spinor_tangent_vertex_green.py` / `.md` (story 002):

$$
V_i^a = -\frac{i}{2}\big[(\mathbf{n}_i\times\mathbf{t}_a)\cdot\boldsymbol{\sigma},\, H_{\mathrm{mag},i}\big],\quad
H_{\mathrm{mag},i} = M_i/2,\quad
K_{ij}^{ab} = \mathrm{Tr}[V_i^a G_{ij} V_j^b G_{ji}],\quad
J^{ab} = \mathrm{Im}\oint K^{ab}\mathrm{d}z/(2\pi)
$$

with `Kzz = 0` exactly, collinear reduction `J^{xx} = J^{yy} = J_cl` exact
(sign-free `|Delta|` vertices, no `s_i s_j`), the two-site finite-angle
anchor `J = E''/2 = bt/(8(b+t))` and the DMI chiral anchor
`D_z = bt sin(phi)/(8(b+t))`. **A single-reference one-shot replay is a
transverse projection only (rank 4 of 9): `Jiso`, `D_x`, `D_y`, `Jani` and
the longitudinal entries are not claimable without genuine x/y/z magnetic
reference legs.**

What survives of the present document: the Pauli identities, the collinear
cross-channel algebra (it is the kernel anchor the tangent vertex reduces
onto), the conjugation-structure discussion for that algebra, and the
`TB2J.Jtensor` decomposition identities. The exchange-tensor pin in the
header and the story-002 items below are superseded by the tangent-vertex
derivation.

## Correction (2026-09-23): Pauli placement and channel mapping — [RETRACTED 2026-09-28, see above]

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
