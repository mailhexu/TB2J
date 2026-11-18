# Production magnetic vertex from the total-H spin splitting (story 005)

Executable authority: [siesta_spiral_vertex.py](siesta_spiral_vertex.py),
following [siesta_spiral_vector.py](siesta_spiral_vector.py) and
[siesta_spiral_pencil.py](siesta_spiral_pencil.py). All assertions run on a
minimal 2-orbital 2-site symbolic model (sublattices A = site 0, B = site 1;
one intersite bond plus its $(1,0,-R)$ partner plus two onsite entries).

## 1. Pauli decomposition of the stored spinor block

The stored lab-frame components pack into the Hermitian block

$$
W(\mu,\nu,R)=\begin{pmatrix}h_1 & h_3-i h_4\\ h_3+i h_4 & h_2\end{pmatrix},
$$

decomposed as $W = h_0\sigma_0 + \mathbf h\cdot\boldsymbol\sigma$ with

$$
h_0=\tfrac{h_1+h_2}{2},\qquad h_x=h_3,\quad h_y=+h_4,\quad
h_z=\tfrac{h_1-h_2}{2}.
$$

The sign of $h_y$ is pinned by the packing convention: the 12 element of
$h_x\sigma_x+h_y\sigma_y$ is $h_x - i h_y$, and the stored 12 element is
$h_3 - i h_4$, hence $h_y = h_4$. $h_0$ is the scalar (spin-independent)
part; $\mathbf h = (h_x,h_y,h_z)$ is the **production magnetic vertex**:
the spin-dependent part of the saved total Hamiltonian (Oroszlány
convention, no XC reconstruction).

## 2. Partner consistency (bundle `validate()`)

`SpiralBundle.validate()` requires the $(\nu,\mu,-R)$ partner to exist and
pins the spin-diagonal components:

$$
h_1(\nu,\mu,-R)=h_1(\mu,\nu,R),\qquad h_2(\nu,\mu,-R)=h_2(\mu,\nu,R).
$$

The mixing components carry the converged-spiral texture (fixture-certified
by story 003, [siesta_spiral_vector.md](siesta_spiral_vector.md)):

$$
h_{12}(\nu,\mu,-R) = e^{-i\,\mathbf q\cdot\mathbf x_{ij}}\,h_{12}(\mu,\nu,R),
\qquad h_{12}=h_3-i h_4 .
$$

At the shadow $\mathbf q\cdot\mathbf x_{ij}=2\pi n$ this reduces to
*identical* components, not the plain real-space adjoint pairing. The
negative control in the script shows the untextured $h_4'=-h_4$ partner
**breaks** pencil Hermiticity for $\mathbf q\cdot\mathbf x\neq 0$. With the
textured partner the assembled $H(\mathbf k)$ is exactly Hermitian
(symbolic).

## 3. Local SU(2) rotations and the weighted declared partition (vertex 2.0)

With $U_p = \exp(-i\theta_p\,\sigma_{\hat n}/2)$, the block-level identity
(symbolic for axes $z$ and $x$, exact rational for a generic unit axis via
Rodrigues, and numeric for a random axis) is

$$
U_p\,(\mathbf h\cdot\boldsymbol\sigma)\,U_p^\dagger
= \bigl(R_{\hat n}(\theta_p)\,\mathbf h\bigr)\cdot\boldsymbol\sigma ,
\qquad
U_p\,\sigma_0\,U_p^\dagger = \sigma_0 .
$$

**Declared partition (vertex 2.0).** Each magnetic block $(\mu,\nu,R)$ has
*distinct endpoint-owner sites* $P = \{\mathrm{own}(\mu), \mathrm{own}(\nu)\}$
(the sites owning the row and the column orbital). The rotation is allocated
additively over $P$,

$$
W \;\mapsto\; W' = \sum_{p\in P} w_p\, U_p W U_p^\dagger
= W + \sum_{p\in P} w_p\bigl(U_p W U_p^\dagger - W\bigr),
\qquad w_p = \frac{1}{|P|},
$$

each contribution conjugated by the **full** $U_p$ (two-sided, same $U$);
the scalar part $h_0$ and the overlap $S$ are untouched. Same-site blocks
($|P| = 1$) carry $w = 1$ and get the full conjugation. Cross-site (shared)
blocks carry $w = 1/2$ per endpoint site: **one site's rotation changes only
its weighted half**, not the whole block. The old full-incidence reading
(every entry incident to A conjugated by $U_A W U_A^\dagger$, and the shared
bond double-conjugated when both endpoint sites rotate) is false; it is
removed and rejected by negative control below.

Consequences (all asserted):

- trace preservation: $h_0$ invariant, the rotated magnetic part stays
  traceless — the scalar part and $H_{\rm scalar}(\mathbf k)$ are exactly
  invariant; only the vertex rotates.
- $S(\mathbf k)$ is never conjugated: its lab blocks are $\delta_{ab}$
  (spin scalars) and the implementation applies the allocation to the vertex
  only.
- **Global covariance:** under a common $U_p = U$ the weighted
  contributions of a shared block sum to $U W U^\dagger$ exactly — the
  joint equal-angle action transforms each block *once*. Asserted
  symbolically per block, and at pencil level at $\mathbf q = 0$ against
  $\mathcal W H(\mathbf k)\mathcal W^\dagger$ with nonzero intersite $B$
  ($\mathcal W=\mathrm{diag}(U,U)$; max defect $<10^{-12}$ on the model).
- **Second vertex:** on a shared block the same-site second vertex carries
  $w = 1/2$ (not $w^2$, not unweighted), and the mixed cross-site second
  vertex vanishes identically,
  $\partial^2 W'/\partial\theta_A\,\partial\theta_B = 0$. Negative control:
  the double conjugation $U_B U_A W U_A^\dagger U_B^\dagger$ carries a
  spurious mixed second vertex and is rejected.
- Inventory on the model: onsite $(0,0)$ owned by A and $(1,1)$ owned by B
  ($w = 1$); the intersite pair $(0,1)$ and its $(1,0)$ partner is shared by
  both endpoint sites ($w = 1/2$ each). Adding a B rotation adds exactly
  $w_B (U_B W U_B^\dagger - W)$ to every B-owned half (onsite B: $w_B = 1$;
  shared bond pair: $w_B = 1/2$); A-owned blocks are untouched (partition
  locality).
- Negative control: a **one-sided** row- or column-only application
  ($U_A\sigma_0 \neq \sigma_0$, $\mathrm{Tr}(U_A\,\mathbf h\cdot\sigma)\neq0$)
  mixes the scalar part in and is *not* the production convention;
  such convention differences belong to audit story 007.

## 4. Rotating-frame statement and the $q/2$ gauge

The vertex is the k-space operator

$$
B(\mathbf k) = H(\mathbf k) - H_{\rm scalar}(\mathbf k),
\qquad
H_{\rm scalar}(\mathbf k)=\sum_{\text{entries}}
h_0\,\bigl(d_{11}\ket{\uparrow}\bra{\uparrow}
            + d_{22}\ket{\downarrow}\bra{\downarrow}\bigr),
$$

$$
d_{11}=e^{-i(\mathbf k+\mathbf q/2)\cdot\mathbf x_{ij}},\qquad
d_{22}=e^{-i(\mathbf k-\mathbf q/2)\cdot\mathbf x_{ij}},
$$

with the magnetic components $h_x,h_y,h_z$ placed by the **same** sector
phases as the Hamiltonian path ($d_{11}$ on the up row / 12 element,
$d_{22}$ on the down row / 21 element). The $q/2$ gauge is therefore
*carried*, not re-derived: the phases attach to row-spin components, so at
$\mathbf q\neq 0$ they do not commute with a spin rotation on spin-mixing
blocks; the rotating frame is defined by using this identical placement for
$H$, $H_{\rm scalar}$ and $B$. Pinned checks:

- at $\mathbf q\to 0$, $B(\mathbf k)$ reduces to the collinear vertex
  Fourier sum $\sum (\mathbf h\cdot\sigma)\,e^{-i\mathbf k\cdot\mathbf x}$;
- at $\mathbf q=0$ (scalar phases), the weighted partition under a common
  $U$ ($w = 1/2 + 1/2$ on shared bonds, nonzero intersite $B$ included)
  equals the global conjugation
  $\mathcal W H(\mathbf k)\mathcal W^\dagger$, $\mathcal W=\mathrm{diag}(U,U)$;
- at $\mathbf q\neq0$ the same comparison differs by $O(q)$
  (max defect $9.98\times10^{-2}$ on the model): the phase placement *is*
  the gauge declaration. $B(\mathbf k)$ remains exactly Hermitian under the
  partition rotation (symbolic $\theta$).

## 5. Certificate fields (provenance, per HundJp practice)

The implementation must emit and round-trip:

| field | content |
|---|---|
| `vertex_version` | `2.0` (serialized v1.0 vertices reject at the gate) |
| `vertex_convention` | `oroszlany_total_h_spin_split` |
| `vertex_convention_value` | spin-dependent part of saved total H; no XC reconstruction; $h_x=h_3$, $h_y=h_4$, $h_z=(h_1-h_2)/2$ |
| `partition_declaration` | weighted site contributions, w=1/count of distinct endpoint owners, each conjugated by full U_site; scalar h0 and S untouched |
| `packing_convention` | $H_{12}=h_3-i h_4$ |
| `sector_phase_convention` | $d_{11}=e^{-i(\mathbf k+\mathbf q/2)\cdot x}$, $d_{22}=e^{-i(\mathbf k-\mathbf q/2)\cdot x}$ |
| `projector_convention` | coefficient form $A^\sim=S^{-1}A$; $\mathrm{Tr}\,A=\mathrm{Tr}(S A^\sim)$ |
| `metric_convention` | generalized problem $H c=\varepsilon S c$ with the assembled spin-dependent $S(\mathbf k)$ |
| `allocation` | DZP channels retained |

Run: `python docs/sympy/siesta_spiral_vertex.py` in the `mydev` environment
(exit 0, `ALL ASSERTIONS GREEN`).
