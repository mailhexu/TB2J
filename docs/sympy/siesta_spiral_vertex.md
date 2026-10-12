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

## 3. Local SU(2) rotation on sublattice A

With $U_A = \exp(-i\theta\,\sigma_{\hat n}/2)$, the block-level identity
(symbolic for axes $z$ and $x$, exact rational for a generic unit axis via
Rodrigues, and numeric for a random axis) is

$$
U_A\,(\mathbf h\cdot\boldsymbol\sigma)\,U_A^\dagger
= \bigl(R_{\hat n}(\theta)\,\mathbf h\bigr)\cdot\boldsymbol\sigma ,
\qquad
U_A\,\sigma_0\,U_A^\dagger = \sigma_0 .
$$

**Declared partition (ADR-004).** Every sparse entry $(\mu,\nu,R)$ with
$\mu\in A$ or $\nu\in A$ *carries A's rotation*, applied two-sided with the
same $U_A$:

$$
W \mapsto U_A\,W\,U_A^\dagger
= h_0\sigma_0 + U_A(\mathbf h\cdot\sigma)U_A^\dagger .
$$

Consequences (all asserted):

- trace preservation: $h_0$ invariant, the rotated magnetic part stays
  traceless — the scalar part and $H_{\rm scalar}(\mathbf k)$ are exactly
  invariant; only the vertex rotates.
- $S(\mathbf k)$ is never conjugated: its lab blocks are $\delta_{ab}$
  (spin scalars) and the implementation applies the rotation to the vertex
  only.
- Inventory on the model: onsite A $(0,0)$ conjugated $U_A W U_A^\dagger$;
  intersite $(0,1)$ ("row in A") and $(1,0)$ ("column in A") carry the
  rotation; $(1,1)$ untouched. Adding a B-sublattice rotation changes only
  the $(1,1)$ entry (partition locality).
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
- at $\mathbf q=0$ (scalar phases), entrywise two-sided rotation of a
  partition covering all entries equals the global conjugation
  $\mathcal W H(\mathbf k)\mathcal W^\dagger$, $\mathcal W=\mathrm{diag}(U,U)$;
- at $\mathbf q\neq0$ the same comparison differs by $O(q)$
  (max defect $9.98\times10^{-2}$ on the model): the phase placement *is*
  the gauge declaration. $B(\mathbf k)$ remains exactly Hermitian under the
  partition rotation (symbolic $\theta$).

## 5. Certificate fields (provenance, per HundJp practice)

The implementation must emit and round-trip:

| field | content |
|---|---|
| `vertex_convention` | `oroszlany_total_h_spin_split` |
| `vertex_convention_value` | spin-dependent part of saved total H; no XC reconstruction; $h_x=h_3$, $h_y=h_4$, $h_z=(h_1-h_2)/2$ |
| `partition_declaration` | sublattices $A$; every entry with $\mu\in A$ or $\nu\in A$ carries $A$'s rotation (two-sided $U_A W U_A^\dagger$); others untouched |
| `packing_convention` | $H_{12}=h_3-i h_4$ |
| `sector_phase_convention` | $d_{11}=e^{-i(\mathbf k+\mathbf q/2)\cdot x}$, $d_{22}=e^{-i(\mathbf k-\mathbf q/2)\cdot x}$ |
| `projector_convention` | coefficient form $A^\sim=S^{-1}A$; $\mathrm{Tr}\,A=\mathrm{Tr}(S A^\sim)$ |
| `metric_convention` | generalized problem $H c=\varepsilon S c$ with the assembled spin-dependent $S(\mathbf k)$ |
| `allocation` | DZP channels retained |

Run: `python docs/sympy/siesta_spiral_vertex.py` in the `mydev` environment
(exit 0, `ALL ASSERTIONS GREEN`).
