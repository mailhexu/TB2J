# 04 — Space-group symmetrization of exchange tensors: exact orbit projection

Companion script: [`04_spacegroup_symmetrization.py`](04_spacegroup_symmetrization.py)
(all assertions in this report are executed there; 88 `PASS` lines).

This note derives, with machine-checked symbolic algebra, the symmetrization of the TB2J
exchange data by space-group (and magnetic space-group) operations, replacing the
distance+tag heuristic by exact orbit projection.  Source convention
(`docs/src/convention.rst`):

$$
E \;=\; -\sum_i \mathbf S_i^T K_i \mathbf S_i
\;-\; \sum_{i\neq j}\Bigl[J^{\mathrm{iso}}_{ij}\,\mathbf S_i\!\cdot\!\mathbf S_j
\;+\; \mathbf S_i\,J^{\mathrm{ani}}_{ij}\,\mathbf S_j
\;+\; \mathbf D_{ij}\!\cdot\!(\mathbf S_i\times\mathbf S_j)\Bigr],
$$

with every ordered pair $(ij$ and $ji)$ stored separately and spins normalized to 1
(positive $J$ = ferromagnetic).

Run with:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
python docs/sympy/04_spacegroup_symmetrization.py
```

---

## 1. Assumptions, notation, and the pair tensor

- **Pair tensor.**  Collect the three bilinear channels of an ordered bond $(i,j,R)$ into

$$
\Gamma_{ij}(R) \;=\; J^{\mathrm{iso}}_{ij}\,\mathbb 1 \;+\; J^{\mathrm{ani}}_{ij}
\;+\; A(\mathbf D_{ij}),
\qquad
A^{\alpha\beta}(\mathbf D) \;=\; \sum_\gamma \varepsilon^{\alpha\beta\gamma} D^\gamma ,
$$

  so that $\mathbf S_i^T \Gamma_{ij}(R)\, \mathbf S_j
  = J^{\mathrm{iso}}\,\mathbf S_i\!\cdot\!\mathbf S_j
  + \mathbf S_i J^{\mathrm{ani}} \mathbf S_j + \mathbf D\cdot(\mathbf S_i\times\mathbf S_j)$.
  **Verified** (script §1): $A(\mathbf D)$ is exactly skew-symmetric and the index
  computation $\mathbf S_i^T A(\mathbf D)\mathbf S_j - \mathbf D\cdot(\mathbf S_i\times\mathbf S_j)=0$
  holds identically in the six spin components and three DMI components.

- **Channel decomposition** (unique):

$$
J^{\mathrm{iso}} = \tfrac13\,\mathrm{tr}\,\Gamma,\qquad
J^{\mathrm{ani}} = \mathrm{sym}\,\Gamma - J^{\mathrm{iso}}\mathbb 1
\ \ (\text{symmetric, traceless}),\qquad
D^\gamma = \tfrac12\sum_{\alpha\beta}\varepsilon^{\gamma\alpha\beta}\,
\mathrm{skew}(\Gamma)^{\alpha\beta}.
$$

  **Verified**: $\Gamma = J^{\mathrm{iso}}\mathbb 1 + J^{\mathrm{ani}} + A(\mathbf D)$ is an
  exact round trip for a generic 9-entry matrix, and $A(\mathbf D)=\mathrm{skew}\,\Gamma$.

- **Reversal identity.**  The stored data of a pair and its reverse satisfy

$$
\Gamma_{ji}(-R) \;=\; \Gamma_{ij}(R)^T ,
$$

  which in channel language means $J^{\mathrm{iso}}$ and $J^{\mathrm{ani}}$ are even under
  reversal while $\mathbf D$ is odd: $\mathbf D_{ji}(-R) = -\mathbf D_{ij}(R)$.  **Verified**
  symbolically on $\Gamma^T$ for a generic $\Gamma$.  Consistency of the double counting:
  $-\mathbf S_i^T\Gamma\mathbf S_j - \mathbf S_j^T\Gamma^T\mathbf S_i = -2\,\mathbf S_i^T\Gamma\mathbf S_j$
  (scalar transpose identity), reproducing the three-channel form term by term.

## 2. Action of a space-group operation $g=\{W|t\}$

- **Site map.**  $g$ maps the atom at fractional $x$ to the atom whose position is
  $Wx + t$ (mod 1, within symprec); $W$ is an *integer* $3\times3$ matrix in the
  fractional (lattice) basis and $t$ a fractional translation.
- **Bond map.**  For a stored key $(i,j,R)$ with fractional bond vector
  $d = x_j + R - x_i$:

$$
d \;\longmapsto\; d' = W d, \qquad
R' \;=\; d' - (x_{j'} - x_{i'}) \;=\; W R + n_j - n_i ,
$$

  where the site images $Wx_i + t = x_{i'} + n_i$, $Wx_j + t = x_{j'} + n_j$ are reduced to
  the stored atoms with integer lattice shifts $n_i, n_j$.  $R'$ is an integer vector
  automatically.  **Verified** symbolically for generic integer $W$, generic fractional
  positions and translation, and reproduced by the concrete key machinery of the script
  (check 2.8, hexagonal 60° rotation with a half translation: $(0,1;\,2,-1,3)\to(1,0;\,1,1,4)$).

- **Tensor action.**  With the cartesian rotation

$$
W_c \;=\; \mathcal A\, W\, \mathcal A^{-1}
\qquad(\mathcal A = \text{cell matrix}),
$$

  a stored tensor transforms as

$$
\boxed{\;\Gamma' \;=\; W_c\,\Gamma\, W_c^T\;}
\qquad\qquad
K' \;=\; W_c\,K\,W_c^T \quad\text{(single-site rank-2)}.
$$

  **Verified** (script §2): the pair energy is invariant,
  $\mathbf S_{i'}^T\Gamma'\mathbf S_{j'} = \mathbf S_i^T\Gamma\mathbf S_j$, for a generic
  symbolic rotation $R_z(\theta)$ (det $=+1$), the $xz$ mirror (det $=-1$) and inversion
  (det $=-1$), because $W_c^TW_c=\mathbb 1$.  **Grounding** (checks 2.6–2.7): for the
  hexagonal cell $\mathcal A = [[a,a/2,0],[0,a\sqrt3/2,0],[0,0,c]]$ and the integer 60°
  rotation $W = [[0,-1,0],[1,1,0],[0,0,1]]$ ($a_1\mapsto a_2$), $W_c$ is exactly orthogonal
  with $\det W_c = +1$ — integer fractional $W$ are genuine cartesian rotations.

## 3. DMI as an axial vector: $D' = \det(W_c)\,W_c\,D$

- **Master identity** (script check 3.1, *general* invertible symbolic $W$):

$$
W\,A(\mathbf D)\,W^T \;=\; A\bigl(\det W\; W^{-T}\mathbf D\bigr).
$$

- On the orthogonal group ($W^{-T}=W$) this becomes the axial-vector law

$$
\boxed{\;A\bigl(\det W\; W\,\mathbf D\bigr) \;=\; W\,A(\mathbf D)\,W^T\;}
\qquad\Longrightarrow\qquad
\mathbf D' \;=\; \det(W_c)\,W_c\,\mathbf D .
$$

  **Verified** (script §3) for full Euler-angle parametrizations of *both* connected
  components of $O(3)$ — proper ($\det=+1$) and improper ($\det W\,\mathbf D$ with
  $\det=-1$) — with symbolic angles, for the explicit substitution cases (mirror:
  $A(-W\mathbf D)=WA(\mathbf D)W^T$; inversion: $A(\mathbf D)$ matrix-invariant while
  $\mathbf D\mapsto-\mathbf D$), and numerically for seeded random orthogonal matrices.
  The contraction identity behind it,
  $\varepsilon^{\gamma\mu\nu}W_{\alpha\gamma}W_{\beta\mu}
  = \det(W)\,\varepsilon^{\alpha\beta\delta}(W^{-1})^{\delta\nu}$, is what the entry-wise
  symbolic check establishes.  The tensor law $\Gamma' = W_c\Gamma W_c^T$ therefore
  already *contains* the axial transformation of the DMI channel.

## 4. Time reversal cancels: primed ops act identically on stored tensors

A primed (time-reversed) magnetic operation acts on spins as $\mathbf S \mapsto -W_c\mathbf S$.
For any bilinear term (generic $\Gamma$, generic $W_c$, symbolic $\sigma=\pm1$; script §4):

$$
(\sigma W_c\mathbf S_i)^T\,\Gamma'\,(\sigma W_c\mathbf S_j) - \mathbf S_i^T\Gamma\,\mathbf S_j
\;=\; (\sigma^2-1)\,\mathbf S_i^T\Gamma\,\mathbf S_j \;=\; 0
\quad\text{for } \sigma=\pm1,
$$

and identically for the single-site term
$(\sigma W_c\mathbf S)^T K'(\sigma W_c\mathbf S) = \mathbf S^T K \mathbf S$.  Both spin flips
cancel.  Consequently:

> **Priming has no effect on the tensor map.**  $\Gamma' = W_c\Gamma W_c^T$ contains no
> $\sigma$.  A primed operation differs from its unprimed counterpart *only* through the
> site map (which orbits of stored keys it relates).  Time reversal therefore only
> **enlarges the set of site maps** — exactly the magnetic-space-group effect exploited
> in §6e.

## 5. Projection: Reynolds operator, $P^2=P$, exact round trip

Store the data as a field $\{\,(i,j,R)\mapsto\Gamma_{ij}(R)\,\}$ closed under the group.
Each $g$ (and each $g$ composed with reversal) transports the field by
$(T_g\Gamma)_{g\cdot k} = W_c\,\Gamma_k\,W_c^T$ (transposed for the reversal factor).  The
symmetrized field is the group average

$$
\bar\Gamma \;=\; P\,\Gamma, \qquad
P \;=\; \frac{1}{|G|}\sum_{g\in G} T_g ,
$$

with $G$ the group generated by the symmetry operations **plus reversal**
($|G| = 2\times$ the number of space-group ops; reversal commutes with everything).

**Verified** (script §5) with concrete exact groups on a 2-site bond along $z$
(generic 9-entry $\Gamma$ at $(0,1;\mathbf 0)$, its transpose at the reversed key):

- **C2v** $=\{E,C_{2z},\sigma_{xz},\sigma_{yz}\}$ ($|G|=8$ with reversal): the averaged
  field is invariant under every operation, $P^2 = P$ holds *identically in the generic
  entries* (averaging twice $=$ once), the channel decomposition round-trips exactly, and
  $\bar{\mathbf D} = 0$ with $\bar J^{\mathrm{ani}}$ forced diagonal.  (mm2 has only
  sign-flip rotations, so the diagonal entries of $J^{\mathrm{ani}}$ remain free —
  uniaxiality needs a 3-/4-fold axis, cf. §6f.)
- **D2d** $=\{E,2S_4,C_{2z},2C_2',2\sigma_d\}$ ($|G|=16$): same properties, and
  $\bar J^{\mathrm{ani}} = \mathrm{diag}(a,a,b)$ (uniaxial, $S_4$ mixes $xx\leftrightarrow yy$),
  $\bar{\mathbf D}=0$.

The decomposition is applied per stored key afterwards:
$J^{\mathrm{iso}}=\mathrm{tr}\,\bar\Gamma/3$, $\bar J^{\mathrm{ani}} = \mathrm{sym}\,\bar\Gamma -
J^{\mathrm{iso}}\mathbb 1$, $\bar D^\gamma = \frac12\varepsilon^{\gamma\alpha\beta}\mathrm{skew}(\bar\Gamma)^{\alpha\beta}$.
(Implementation note: the numerical symmetrizer additionally snaps $|{\rm value}| <$ zero-tol,
default $10^{-8}$ meV, to exactly $0$; the symbolic derivation below proves the zeros are
*exact*, so the snap only removes round-off.)

## 6. Moriya rules as exact symbolic orbit averages

Each case below is a full orbit average of a generic $\Gamma$ over the group generated by
the listed operation(s) plus reversal, followed by the exact channel decomposition
(script §6; bond along $z$, sites at $z=\tfrac14,\tfrac34$, midpoint $\tfrac12$).

| case | operation(s) | forced to zero | survives |
|---|---|---|---|
| **a** | inversion at bond midpoint $\bar 1$ | $\mathbf D=0$ | $J^{\mathrm{iso}}$; $J^{\mathrm{ani}}$ = any symmetric traceless (fully unconstrained) |
| **b** | mirror plane **containing** the bond ($\sigma_{xz}$) | $D_x, D_z$; $\mathrm{ani}_{xy},\mathrm{ani}_{zy}$ | $D_y\parallel$ mirror normal ($\perp$ bond); $\mathrm{ani}_{xz}$, diagonals |
| **c** | mirror plane **perpendicular** to the bond ($\sigma_{xy}$) | $D_z$ ($\mathbf D\parallel$ bond dies); $\mathrm{ani}_{xz},\mathrm{ani}_{yz}$ | $D_x,D_y$: $\mathbf D$ lies **in the mirror plane** ($\perp$ bond) |
| **d** | 2-fold axis **perpendicular** to the bond ($C_{2x}$) | $D_x$ ($\mathbf D\parallel C_2$ axis dies); $\mathrm{ani}_{xy},\mathrm{ani}_{xz}$ | $D_y,D_z$: $\mathbf D\perp C_2$ axis |
| **e** | **primed half-translation** $\theta T_{1/2}$ (AFM) | $\mathbf D = 0$; $\mathrm{skew}\,\bar\Gamma = 0$ | $J^{\mathrm{iso}}$, $J^{\mathrm{ani}}$ intact ($\bar\Gamma$ symmetric) |
| **f** | $C_3$ about the bond axis | $D_x,D_y$; all off-diagonal $\mathrm{ani}$ | $D_z\parallel$ bond; $\bar J^{\mathrm{ani}}=\mathrm{diag}(a,a,b)$ uniaxial |
| **g** | pure translation $\{1|T\}$ | — | $\Gamma_{ij}(R+T)=\Gamma_{ij}(R)$: consistency only |

Mechanistic remarks, all machine-checked:

- **(a)** The inversion exchanges the two sites, so its constraint
  $\Gamma^T = W\Gamma W^T$ combines with the reversal identity to kill the skew part
  exactly; the symmetric part is untouched (check 6a.6).
- **(c)/(d)** These operations also **exchange the sites** (checks 6c.0, 6d.0): the
  transformed tensor lands on the reversed key, and only *after* combining with
  $\Gamma_{ji}=\Gamma_{ij}^T$ does the constraint act on $\Gamma_{ij}$ itself.  The result
  kills exactly one DMI component each — the bond-parallel component under the
  perpendicular mirror, the axis-parallel component under the perpendicular 2-fold axis.
- **(e)** $W=\mathbb 1$, $t=\tfrac12$: the tensor action is the identity
  ($\Gamma'=\Gamma$), but the operation exchanges the two AFM sublattices *only because it
  is primed* (the unprimed $T_{1/2}$ does not map an AFM spin configuration onto itself).
  The orbit average collapses to $\bar\Gamma = \tfrac12(\Gamma+\Gamma^T)$: **DMI is
  forbidden outright** by the magnetic half-translation while the symmetric channels
  survive.  This is the flagship magnetic-space-group effect: priming changes no tensor
  action (§4) yet adds the site map that produces the constraint.
- **(f)** With reversal in the group the average also symmetrizes, and the 3-fold
  rotation leaves precisely the uniaxial form $\mathrm{diag}(a,a,b)$ and $D\parallel z$.
- **(g)** For $W=\mathbb 1$ the transported tensor equals the original one
  ($\Gamma'=\Gamma$; checks 6g.2–6g.3): translations carry **no rotational information**
  and act only as consistency constraints ($\Gamma$ constant on a translation orbit).  All
  channel information enters through operations with $W\neq\mathbb 1$.  (In a magnetic-cell
  description, keys are reduced mod the cell, so $R+T$ is the same stored key.)

### Relation to Moriya's original rules

The computed patterns reproduce, component by component, the five rules of
T. Moriya, *Anisotropic Superexchange Interaction and Weak Ferromagnetism*,
Phys. Rev. **120**, 91 (1960) (rules quoted verbatim, e.g., in Fert, Chshiev, Thiaville &
Yang, arXiv:2305.02163 (2023)); $A$, $B$ the two sites, $C$ the bond midpoint:

1. inversion center at $C$ $\Rightarrow D=0$ — case **a**;
2. mirror plane perpendicular to $AB$ through $C$ $\Rightarrow \mathbf D\parallel$ mirror
   plane (i.e. $D\perp AB$) — case **c**;
3. mirror plane containing $A$ and $B$ $\Rightarrow \mathbf D\perp$ mirror plane — case **b**;
4. two-fold axis perpendicular to $AB$ through $C$ $\Rightarrow \mathbf D\perp C_2$ axis — case **d**;
5. $n$-fold axis ($n\ge2$) along $AB$ $\Rightarrow \mathbf D\parallel AB$ — case **f**.

> **Caution.**  The shorthand frequently found in the literature — "a mirror plane
> perpendicular to the bond kills the DMI" / "a 2-fold axis perpendicular to the bond
> kills the DMI" — is a **misquote** of Moriya's rules 2 and 4.  The exact orbit averages
> (checks 6c.3, 6d.3) confirm the original statement: those operations forbid exactly one
> DMI component each (the bond-parallel / axis-parallel one), and force $\mathbf D=0$ only
> in combination with further operations (e.g. the two perpendicular mirrors of C2v, §5).

## 7. Numerical validation (script §7)

Seeded (`numpy.random.default_rng(20240517)`, house seed) random batteries confirm every
symbolic result numerically:

- random tensors: $\mathbf S^TA(\mathbf D)\mathbf S = \mathbf D\cdot(\mathbf S\times\mathbf S)$,
  decomposition round trip, reversal $\mathbf D\mapsto-\mathbf D$;
- for each case a–f: random non-invariant fields $\to$ numeric projection $\to$ field
  invariance under every operation with deviations $\le 2.2\times10^{-16}$, exact round
  trip, forced-zero channels below $10^{-10}$ while surviving components stay $O(1)$
  (e.g. $|D|=0.20$–$1.15$ depending on case);
- random orthogonal matrices of both det signs: $\mathbf D' = \det(W)\,W\mathbf D$ and the
  primed cancellation $\mathbf S\mapsto-W\mathbf S$ to machine precision.

## 8. Summary of the contract consumed by the implementation

$$
\Gamma_{ij}(R)=J^{\mathrm{iso}}\mathbb 1+J^{\mathrm{ani}}+A(\mathbf D),\quad
A^{\alpha\beta}=\varepsilon^{\alpha\beta\gamma}D^\gamma,\quad
\Gamma_{ji}(-R)=\Gamma_{ij}^T,
$$
$$
g=\{W|t\}:\ x\mapsto Wx+t,\quad d\mapsto Wd,\quad R'=WR+n_j-n_i,\quad
W_c=\mathcal A W\mathcal A^{-1},
$$
$$
\Gamma'=W_c\Gamma W_c^T,\qquad K'=W_cKW_c^T,\qquad
\mathbf D'=\det(W_c)\,W_c\,\mathbf D,
$$
$$
\text{primed ops: identical tensor action, enlarged site-map orbits},\qquad
P=\tfrac1{|G|}\sum_gT_g,\quad P^2=P,
$$
$$
J^{\mathrm{iso}}=\tfrac{\mathrm{tr}\,\bar\Gamma}{3},\quad
J^{\mathrm{ani}}=\mathrm{sym}\,\bar\Gamma-J^{\mathrm{iso}}\mathbb 1,\quad
\bar D^\gamma=\tfrac12\varepsilon^{\gamma\alpha\beta}\mathrm{skew}(\bar\Gamma)^{\alpha\beta}.
$$

Every identity in this box is asserted symbolically (and, where applicable, numerically)
by the companion script.
