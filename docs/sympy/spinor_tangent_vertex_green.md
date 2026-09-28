# Spinor Magnetic Tangent-Vertex Exchange — Sympy Derivation Report

Story 002 of the SOC spinor projector-Green spec
(`Projects/TB2J/specs/soc-spinor-projector-green`). This report **supersedes
the exchange-tensor pin of `spinor_projector_green.md` and the 2026-09-23
"correction" recorded there**, which is false in premise and defective in its
replacement mapping (negative controls below).

Script: `spinor_tangent_vertex_green.py` (assertion-checked; run in `mydev`
from the worktree root). Status: **all assertions pass** (2026-09-28, mydev
SymPy + numpy + `TB2J.mycfr.CFR`).

## Pinned object: physical local magnetic rotation vertex on the full spinor G

For each magnetic site, with the stored magnetic `spinor_operator`
`M_i = Delta_i (n_i . sigma)` (`Delta_i` the signed splitting, `n_i` the
*physical moment direction* = M-vector direction), `H_mag,i = M_i/2`, and
Cartesian tangents `t_a`:

$$
V_i^a \;=\; -\frac{i}{2}\,\big[(\mathbf{n}_i\times\mathbf{t}_a)\cdot\boldsymbol{\sigma},\; H_{\mathrm{mag},i}\big],
\qquad
K_{ij,R}^{ab}(z) \;=\; \mathrm{Tr}\!\left[V_i^a\, G_{ij}(R,z)\, V_j^b\, G_{ji}(-R,z)\right]
$$

$$
J^{ab} \;=\; \frac{1}{2\pi}\,\mathrm{Im}\;\oint_{\mathcal C} K^{ab}(z)\,\mathrm{d}z
$$

- SOC is **never rotated**: the vertex rotates only `H_mag`; the vertex
  algebra is asserted for general unit `n`.
- Exact vertex identity (asserted): `V(t) = (Delta_i/2) sigma.t` for **any
  tangent `t` perpendicular to `n`**; for Cartesian-indexed vertices
  `V^a = (Delta_i/2)(e_a - (n.e_a) n).sigma` (the normal component is the
  null rotation).  On axis-aligned legs (`n = ±z` cyclically) this is the
  magnitude form `V^a = |Delta_i| sigma_a/2 (a = x,y)`, `V^z = 0` — the sign
  of `Delta_i` **never enters the vertex**, and no `s_i s_j` factor appears
  anywhere in the tangent path (the propagator carries the moment direction;
  asserted on the AFM dimer, where the kernel's `s_i s_j = -1` is reproduced
  by the sign-free vertex + AFM propagator).
- `K` contracts the **unprojected, untransposed, full complex spinor blocks**
  — no Pauli decomposition of `G`, no channel reconstruction.

## Contour normalization (sign and factor pinned by the two-site anchor)

`J^{ab} = Im contour K^{ab} dz/(2 pi)` with the standard occupied contour
(retarded poles below `E_F`, `Im contour f = -pi sum Res(f)`; numerically the
house `TB2J.mycfr.CFR` evaluation). Sign pinned: positive `J` =
ferromagnetic.

Collinear reduction is **exact** (asserted as contour-object identity and
numerically to 1e-7 on the dimer):

$$
J^{xx}_{\mathrm{tan}} = J^{yy}_{\mathrm{tan}} = J_{\mathrm{cl}}
= s_i s_j\; \mathrm{Im}\;\oint \frac{\Delta_i\Delta_j\, G^{\uparrow}_{ij}\, G^{\downarrow}_{ji}}{4\pi}\,\mathrm{d}z
$$

(the tangent trace contains both conjugate channels; their doubled Im turns
the vertex `1/4` into the kernel's `/(4 pi)` against the tangent `/(2 pi)`),
and

$$
K^{zz} = 0 \quad\text{exactly — no longitudinal spurion.}
$$

## Two-site finite-angle energy anchor

`H_ii = -b n_i.sigma`, `H_jj = -b n_j.sigma`, `H_ij = t 1` (`b, t > 0`), one
occupied band. All steps are executable assertions:

| object | exact result |
| --- | --- |
| char. polynomial (rel. angle `theta`) | `(z^2-b^2-t^2)^2 - 2 b^2 t^2 (1+cos theta)` |
| occupied energy | `E(theta) = -sqrt(b^2 + t^2 + 2bt cos(theta/2))` |
| curvature | `E''(0) = bt/(4(b+t))` |
| ordered-bond exchange | `J = E''/2 = bt/(8(b+t))` |
| residue | `Res_{z=-b-t}(G^up_ij G^dn_ji) = -t/(8 b (b+t))` |
| contour (FM) | `J_cl = J_tan = -pi Delta^2 Res/(4pi) = +bt/(8(b+t))` |
| contour (AFM, `H_jj = +b sigma_z`) | `J = -b^2 t^2/(4 (b^2+t^2)^{3/2}) < 0` |

Numeric CFR corroboration (`b=1.3, t=0.9`, `E_F = -b`): `J^xx = +0.06647727 =
bt/(8(b+t))` to 1e-7, kernel-object equality on the same contour, AFM sign
and magnitude (cold contour `kT = 0.01` eV; the AFM occupied band sits only
`B-b = 0.28` eV deep, and only the simple-pole coefficient of the double pole
enters the contour — asserted symbolically).

## DMI chiral phase anchor

Gauged bond `H_ij = t exp(+i phi sigma_z/2)`, `H_ji = H_ij^dagger` (proper
diagonal spin flux; a scalar bond phase is a trivial gauge and carries no
chirality):

$$
\frac{J^{xy}-J^{yx}}{2} = D_z = +\frac{bt\,\sin\phi}{8(b+t)},\qquad
J^{xy} = -J^{yx}\ \text{(pure chiral)},\qquad
J^{xx}(\phi) = J_{\mathrm{cl}}\cos\phi
$$

all asserted symbolically (exact residues at `z = -b-t` of the inverted 4x4)
and numerically (CFR to 1e-7), plus an independent exact-eigenvalue check of
the 4x4 dimer: tilting `n_i -> z + alpha x`, `n_j -> z + beta y` gives

$$
\left.\frac{\partial^2 E}{\partial\alpha\,\partial\beta}\right|_0 = -2 D_z
\quad(\text{verified to } 10^{-7}).
$$

## Reciprocity and SOC insertion

- Pair reversal **before integration**, all nine Cartesian components
  (cyclicity, asserted with general complex blocks):
  `K_ij,R^{ab} = K_ji,-R^{ba}`, hence `J_ij(R) = J_ji(-R)^T` on the raw tensor.
- SOC enters the propagator only, on **both legs** (asserted symbolically and
  by 1e-8 finite difference):
  `dK/dlambda = Tr[V_i G0 W G0|_ij V_j G_ji] + Tr[V_i G_ij V_j G0 W G0|_ji]`.

## Three-leg raw rank-9 reconstruction — and the one-shot limitation

Right-handed tangent frames `(u, v, n)`: leg `z -> (x,y)`, leg `x -> (y,z)`,
leg `y -> (z,x)`. Each genuine magnetic reference leg measures its 2x2
transverse block; the 12 measured entries determine the 9 raw lattice-tensor
entries (design matrix rank 9 = symmetric 6 + antisymmetric 3; single leg
rank 4):

- leg `z`: `{Jxx, Jyy, Jxy, Jyx}`; leg `x`: `{Jyy, Jzz, Jyz, Jzy}`;
  leg `y`: `{Jzz, Jxx, Jzx, Jxz}`; repeated diagonals pass a consistency
  gate and are averaged.
- Then, per `TB2J` Levi-Civita conventions:
  `Jiso = tr J/3`, `D = (Jyz-Jzy, Jzx-Jxz, Jxy-Jyx)/2`,
  `Jani = (J+J^T)/2 - Jiso I`; asserted round trip on a generic
  all-channels tensor.

> **One-shot replay is a transverse projection, NOT a full tensor.** A single
> collinear reference (e.g. `z`) recovers exactly `{Jxx, Jyy, Jxy, Jyx}`
> (rank 4 of 9). `Jzz`, `Jzx`, `Jxz`, `Jyz`, `Jzy`, `Jiso`, `D_x`, `D_y` and
> the full `Jani` are **not determined** — the script constructs two distinct
> true tensors sharing the leg-`z` block. One-shot vs three-leg gates must
> compare only the measured transverse projection; a one-shot "full tensor"
> claim is incorrect by construction.

Core merge contract (agreed with the kernel worker): `io_merge` is bypassed;
`merge_transverse_legs(legs)` (split_soc_kernel.py) solves the raw 9-entry
tensor per pair from the 12 transverse entries, averages the gated diagonals,
checks reciprocity `J_ij(R) = J_ji(-R)^T`, and emits one decomposed result.
Per-leg output stays `{"J_leg": 3x3 float zero-masked on n, "frame": {...},
"K_ijR": integrated 3x3 complex}` — **never** per-leg `Jiso/dmi/jani` keys.

## Negative controls (the 2026-09-23 correction, replaced)

1. **False premise.** The correction claimed `-Tr[(sigma_a Delta_i) G_ij
   (sigma_b Delta_j) G_ji]` is "identically zero" for block-diagonal
   collinear G. It is not: `Jxx_old = Jyy_old = z_i z_j (g_up h_dn + g_dn
   h_up)` — exactly the LKAG cross-channel (asserted).
2. **The A-channel mapping is DMI-dead.** For `Delta = z sigma_z` the
   ExchangeNCL mapping `A^{uv} = Tr[Delta_i G^(u)_ij Delta_j G^(v)_ji]/pi`
   has `A^{0i} - A^{i0} = 0` **identically** — the spinor trace
   `Tr[sigma_z sigma_u sigma_z sigma_v] = ±2 delta_uv` vanishes for the
   antisymmetric combination for *any* complex projector content `T^u`, i.e.
   even with full SOC spin-mixing in G (asserted). Observed FeO symptom:
   exactly-zero DMI.
3. **Longitudinal self-pair residue.** The same channel family carries
   `Azz` same-channel contractions; the old Pauli-left vertex additionally
   has `Jzz_old = -z_i z_j (g_up h_up + g_dn h_dn) != 0`, which no contour
   prescription removes — the defect class behind the spurious FeO
   `Jani_zz = -114` meV. The tangent vertex has `Kzz = 0` exactly.
4. **`io_merge` route is structurally incapable here.**
   `io_merge.Merger.merge_Jani` lstsq-rebuilds a matrix from **six symmetric
   parameters** (`[[J0,J3,J5],[J3,J1,J4],[J5,J4,J2]]`) — an antisymmetric
   component cannot be represented by construction; `merge_DMI` projects each
   leg's `D` onto transverse rows `u`, discarding the sole observable `D_n`
   of a collinear leg. The design-review execution recorded the resulting
   bias as `diag(1,2,3) -> (7/6, 2, 17/6)` for per-leg tangent data through
   the independently-averaged scalar/traceless route. Hence the
   `merge_transverse_legs` raw solve above.
5. **`0.5*(dmi[:,None] - dmi[None,:])`** (previous assembly in
   `spinor_channels_to_exchange_tensor`) is not the TB2J Levi-Civita
   embedding `[[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]]`.

## Implementation seams (core worker; cited for docs coordination)

- `TB2J/projector_green.py`: `magnetic_tangent_vertices(operator_block)`
  (dict `n`/`magnitude`/`vertices`, `V^a = -(i/4)[kron((n x t_a).sigma,
  I_orb), M_dense]`), `spinor_tangent_pair_matrix(v_i, g_ij, v_j, g_ji)` ->
  `(3,3)` complex `K^{ab}`, `spinor_tangent_trace(...)` -> raw `K_ijR`
  (the `/(2 pi)` happens after contour integration). The old
  `spinor_pair_channels` / `spinor_channels_to_exchange_tensor` are deleted.
- `TB2J/split_soc_kernel.py`: `merge_transverse_legs(legs)` raw rank-9 solve.
- The unchanged collinear path (`projector_exchange_trace`, `/(4*pi)` per
  energy, `Im` over the contour, `s = sign(Tr Delta)`) is retained for
  bitwise stability; the tangent path reduces onto it exactly.

## Verification summary

Pauli identities; vertex commutator form vs numeric spinor-rotation generator
(1e-8); tangent-kernel closed forms; contour-object equality with the
validated collinear kernel; charpoly/curvature/residue anchor chain
(symbolic, exact); chiral anchor symbolic + CFR (1e-7) + exact-eigen mixed
derivative (1e-7); reciprocity (symbolic, 9 components); insertion topology
(symbolic + 1e-8 FD); three-leg rank-9 reconstruction with consistency gate,
round-trip decomposition and explicit one-shot counterexample; symbolic-vs-
numpy trace cross-check (1e-10); CFR corroboration FM/AFM/gauged (1e-7..1e-5).
