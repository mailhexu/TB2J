# Green's Function of the Spin-Spiral Hamiltonian and Its Magnetic Force
Theorem Application — Derivation Report

Script: `spiral_green_function_mft.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-24).
Serves the spin-spiral MFT spec (`Projects/TB2J/specs/spin-spiral-mft`) and
the paper `papers/TB2J_spinspiral_GBT`.

## 1. Motivation

The planned TB2J feature extracts Heisenberg exchange constants from a
tight-binding (TB) mean-field model by the spin-spiral route: instead
of the Green-function (LKAG) exchange computed from the collinear
Hamiltonian, the exchange is read from the energy of flat spin spirals
obtained with the generalized Bloch theorem. For this to be more than
a heuristic energy scan, the spiral energies must be obtained with the
**magnetic force theorem** (MFT), and the exchange extraction must be
provably the same object as the established Green-function result.

That proof lives in the Green's function. The MFT energy difference is
a statement about the spectral response of the frozen density matrix,
which is a statement about the **Green's function of the spin-spiral
Hamiltonian** — evaluated in the folded (generalized-Bloch)
representation that keeps the computation on a single primitive cell.
The two earlier derivation scripts supply the pieces around this
middle step: `spin_spiral_generalized_bloch.py` derives the spiral
Hamiltonian and its folded blocks, and `spiral_force_theorem_J.py` maps
spiral energies onto the exchange dictionary. This report derives the
missing middle — the resolvent of the spiral Hamiltonian, the exact
relations that let site-resolved quantities be read off it, the contour
form of the force theorem, and the identification of the spiral
curvature with the Green-function exchange kernel. Every claim here is
checked by assertion in the companion script.

## 2. Context

The inputs and the surrounding workflow, for orientation:

- **Electronic structure.** A converged *collinear* mean-field
  calculation in TBUpy (Hubbard-corrected Wannier or model
  Hamiltonians), saved as a `.tbupy.nc` result: the effective
  Hamiltonians and overlap in real space, the local Hubbard
  potential, the density matrix, and the Fermi level. TB2J reads this
  through its `TBUpyManager` interface.
- **Magnetic state.** A reference state whose spins point along one
  axis (ferromagnetic, or layered antiferromagnetic with sublattice
  signs). The spiral is created by rotating each site's spin
  direction by an angle $\theta_{a\mu}=2\pi\,\mathbf q\cdot(a+\tau_\mu)+\phi_\mu$ (a single flat spiral with propagation vector $\mathbf q$ and sublattice phase offsets $\phi_\mu$).
- **Spiral Hamiltonian.** A rotated copy of the collinear
  Hamiltonian. For spins rotating about the same axis as the reference
  magnetization, this rotation leaves the local exchange fields
  untouched and appears entirely as spin-dependent phases on the
  hopping (script 1). Because a single flat spiral is an exact
  symmetry of the crystal, the spiral Hamiltonian is represented
  exactly in the *folded* form: one primitive cell, a spinor matrix
  $H_q(\mathbf k)$, and a nonorthogonal pencil
  $(H_q,S_q)$ when the basis is not orthogonal.
- **Magnetic force theorem.** The MFT evaluates the total-energy
  difference between two magnetic states — here the spiral and the
  collinear reference — from frozen-potential eigenvalue sums. In this
  workflow it is protocol P-b (one-shot, frozen reference); protocol
  P-c re-runs a self-consistent calculation at every $\mathbf q$ and
  serves as the reference check.
- **Output contract.** The exchange must land in TB2J's standard
  format: an `exchange_Jdict` keyed by lattice translation and spin
  indices, positive $J$ meaning ferromagnetic, consumable by the
  magnon and thermodynamic modules. The conventions are pinned in
  `spiral_force_theorem_J.py` and the conventions appendix of the
  paper.
- **Scope.** No spin–orbit coupling: the generalized Bloch theorem
  assumes the spiral axis is compatible with the spin structure, and
  SOC is handled by the Green-function route (or a fully relativistic
  code), not here. Dzyaloshinskii–Moriya and anisotropic exchange are
  out of scope for the spiral protocol.

## 3. Notation

| Symbol | Meaning |
|---|---|
| $H_s,\,S_s$ | supercell spiral Hamiltonian and overlap, in the absolute gauge where the spiral lives in bond phases |
| $H_q(\mathbf k),\,S_q(\mathbf k)$ | folded (twisted) pencil for one primitive cell at $\mathbf k$ |
| $G_q(\mathbf k,E),\,G_s(E)$ | resolvents $[ES-H]^{-1}$ at energy $E$ |
| $W$ | twisted Bloch unitary that folds the supercell onto the primitive $\mathbf k$ mesh |
| $k_n = n/N$ | the $N$ primitive $\mathbf k$-points of the folded mesh |
| $f(z)$ | occupation function (Fermi function or smearing) |
| $M$ | transverse (local-force) response matrix of the cell spin directions |
| $j(R)$ | pair exchange kernel, $j(R)=-M(0,R)$; the LKAG object |
| $\sigma_s$ | spin eigenvalue: $+1$ (up) or $-1$ (down) |

## 4. Procedure: from context to final result

The whole feature is the following chain. Each step names the object
that is produced, the equation in this report that defines it, and the
code that will implement it.

**Step 0 — Load the reference.** Read the converged collinear result
from the `.tbupy.nc` file: real-space Hamiltonians for the two spin
channels, the overlap, the Fermi level, and the occupation function
`f`. Nothing in the spiral calculation is allowed to relax at this
stage; this is the frozen potential that defines the force theorem
(protocol P-b).

**Step 1 — Build the folded spiral pencil.** For the requested
$\mathbf q$, assemble $H_q(\mathbf k)$ and $S_q(\mathbf k)$ for every
$\mathbf k$ on the primitive mesh, using the twist angles
$\alpha_\mu = 2\pi\,\mathbf q\cdot\tau_\mu+\phi_\mu$ (script 1,
Eq. `hq`/`alpha`; the TBUpy-side implementation is
`assemble_multisublattice_hq_sq` on branch `spiral`). This is the
only place where the spiral is constructed, and it stays on a single
cell: no supercell, no commensurability requirement.

**Step 2 — Invert to the spiral Green's function.** At each $\mathbf k$,
invert the pencil to get the resolvent of Eq. (1), either directly as
a matrix inverse or, in production, through the spectral
decomposition of the pencil (Cholesky reduction) that produces the
poles and eigenvectors of Eq. (2).

**Step 3 — Sum the poles to get the spiral energy $E(\mathbf q)$.**
The MFT energy is the $S$-weighted trace of the resolvent integrated
with the occupation function, Eq. (5). Because the sum runs over the
folded primitive-cell mesh only, the cost of one $\mathbf q$ point is
one small matrix inversion per $\mathbf k$, independent of the
supercell size that a real-space supercell construction would need.
If site-resolved diagnostics are wanted (local moments of the spiral
state, or pair traces), the resolvent blocks are unfolded to the
supercell with the twist factors of Eq. (4).

**Step 4 — Take the force-theorem difference.** The quantity of
physical interest is $E(\mathbf q)-E(0)$, the contour integral of the
*difference* of the two resolvents, Eq. (6). The reference $E(0)$ is
computed once, on the collinear folded mesh, with the same
$\mathbf k$-mesh, weights, and Fermi level; the difference is therefore
insensitive to any constant offset in the band energy and isolates the
spiral response alone.

**Step 5 — Map the spiral energies onto exchange constants.** The
sequence $E(\mathbf q)$ over a $\mathbf q$-mesh or $\mathbf q$-path is
fitted to the Heisenberg energy mapping
$E(\mathbf q)-E(0)=J(\mathbf 0)-\operatorname{Re}J(\mathbf q)$ with
the pinned TB2J conventions (`spiral_force_theorem_J.py`; least
squares for arbitrary $\mathbf q$ sets, exact Fourier inversion on a
uniform grid). This produces the exchange dictionary
$\{J_{ij}(\mathbf R)\}$ in TB2J's standard sign and
double-counting conventions.

**Step 6 — Validate before trusting.** Three independent checks are
available, and all three are the content of this report:

1. *Spectral identity* — the folded resolvent is the exact twisted
   transform of the supercell resolvent, Eq. (4), verified matrix
   element by matrix element.
2. *Kernel identity* — the small-$\mathbf q$ curvature of the spiral
   energy is the contraction of the transverse response matrix,
   Eq. (7); its off-diagonal elements are exactly the pair exchange
   kernel $j(R) = -M(0,R)$, the same object LKAG computes.
3. *Cross-route identity* — the $q$-space Green-function module
   (`exchange_qspace`) computes the Fourier transform of the same
   kernel, Eq. (9), so the spiral $J(\mathbf q)$ can be compared
   against the existing Green-function result as a function of
   $\mathbf q$, shell by shell.

Two structural facts (Section 5.2) constrain how these steps are
implemented and how the results are interpreted: the spiral stiffness
is a flux response, and a fully filled band contributes nothing to it.
The companion script asserts every identity quoted above.

The computation chain, with the equation that implements each step:

```
  collinear reference H_0, f, e_F                 [step 0]
        |  site rotations U_a = Rz(theta_{a mu})
        v
  spiral supercell H_s        ~=  twisted unitary W
        |  W H_s W^dag =  _n H_q(k_n)
        v
  folded pencil (H_q(k), S_q(k))                  Eq. (1)   [step 1]
        |  resolvent
        v
  G_q(k,E) on the primitive k-mesh                Eq. (2)   [step 2]
        |  pole sum of Tr[S_q G_q]  ->  E(q)                 [step 3]
        |  unfolded blocks           ->  site quantities      (Eq. 4)
        |  contour difference        ->  E(q) - E(0)         (Eq. 6) [step 4]
        v
  Heisenberg fit of E(q) - E(0)   ->  J_ij(R)               [step 5]
        |
        v
  validation: spectral (Eq. 4), kernel (Eq. 7), q-space (Eq. 9) [step 6]
```

## 5. The results

### 5.1 The resolvent of the spiral pencil

**Result.** The spiral Green's function at each primitive $\mathbf k$
is the inverse of the folded pencil,

$$
G_q(\mathbf k,E) \;=\; \big[E\,S_q(\mathbf k) - H_q(\mathbf k)\big]^{-1} .
\tag{1}
$$

Expressed in the eigenbasis of that pencil — pairs
$(\varepsilon_n,c_n)$ with $c_n^\dagger S_q c_m=\delta_{nm}$, obtained
by Cholesky reduction — the resolvent is a sum of simple poles, and
its overlap-weighted trace is a sum of those poles with unit
residues:

$$
G_q(\mathbf k,E) \;=\; \sum_n \frac{c_n c_n^\dagger}{E-\varepsilon_n},
\qquad
\operatorname{Tr}\big[S_q(\mathbf k)\,G_q(\mathbf k,E)\big]
= \sum_n \frac{1}{E-\varepsilon_n} .
\tag{2}
$$

**Why the trace matters.** The force-theorem energy in Step 3 is built
from this trace alone, so production code can integrate it on any
contour or Matsubara mesh without ever forming eigenvectors. The
eigenvectors are needed only for site-resolved quantities, and those
are obtained from the same resolvent blocks by unfolding (Section
5.2).

*Verified:* Eqs. (1)–(2) hold to $10^{-10}$ on random two-sublattice
pencils.

### 5.2 Unfolding the supercell resolvent

**Result.** Site-resolved quantities of the spiral state — local
moments, pair traces — are computable from the *primitive-cell*
resolvents. The twisted Bloch unitary that performs the unfolding
carries the cell half-twist only; the sublattice twist angles
$\alpha_\mu$ live in $H_q(\mathbf k)$, not in the basis:

$$
W[(a\mu s),(n\mu's')] \;=\; \frac{1}{\sqrt N}\,
e^{-2\pi i k_n a}\,e^{+i\sigma_s\pi q a}\,
\delta_{\mu\mu'}\delta_{ss'} .
\tag{3}
$$

With this convention $W$ diagonalizes the supercell Hamiltonian
($W H_s W^\dagger=\bigoplus_n H_q(k_n)$) and therefore also its
resolvent, and every supercell resolvent block is the inverse twisted
transform of the folded resolvents:

$$
G_s[(a\mu s),(b\nu s')](E)
= \frac{1}{N}\sum_m e^{2\pi i k_m(a-b)}\,
\overline{\mathrm{tw}(s,m)}\;
G_q(k_m)_{\mu s,\nu s'}(E)\;\mathrm{tw}(s',m),
\qquad
\mathrm{tw}(s,m)=e^{+i\sigma_s\pi q m}.
\tag{4}
$$

**Reading Eq. (4).** Each folded block contributes to a supercell block
with the ordinary Bloch phase $e^{2\pi i k_m(a-b)}$ between cells $a$
and $b$, multiplied by the spin twist factors evaluated at the
folded-mesh index $m$ (up-spin $e^{+i\pi q m}$, down-spin its
conjugate). This is the exact generalized-Bloch analogue of the
ordinary Fourier relation $G(R)=\frac{1}{N}\sum_k e^{2\pi i k R}G(k)$.

*Verified:* $W$ is unitary to $10^{-12}$; Eq. (4) holds
matrix-element by matrix element ($<10^{-9}$) on random
two-sublattice rings, with and without overlap, at three twist values.
The independent numbers run of
`papers/TB2J_spinspiral_GBT/scripts/make_figures.py` on the same
models reports $6\times10^{-16}$.

### 5.3 The force theorem on the contour

**Result.** The MFT spiral energy is a pole sum of the folded
resolvent, and the force-theorem energy difference is the
corresponding difference of two resolvents — both evaluated on the
primitive cell. The band energy is

$$
E_{\rm band}(\mathbf q) \;=\; \frac{1}{2\pi i}\oint z\,f(z)\,
\operatorname{Tr}\big[S_q G_q(z)\big]\,dz
\;=\; \sum_n \varepsilon_n(\mathbf q)\, f(\varepsilon_n) ,
\tag{5}
$$

so the energy difference that enters the exchange fit is

$$
E(\mathbf q)-E(0) \;=\; \frac{1}{2\pi i}\oint z\,f(z)\,
\operatorname{Tr}\big[S_q G_q - S_0 G_0\big]\,dz .
\tag{6}
$$

The two routes through Eqs. (5) and (6) — direct eigenvalue sums per
folded $\mathbf k$, or the contour form integrated on a complex or
Matsubara mesh, exactly as `MAEGreen` already does for the collinear
problem — compute the same object, and the cost per $\mathbf q$ point
is one small inversion per $\mathbf k$ on the primitive mesh.

*Verified:* the folded pole sum reproduces the supercell eigenvalue sum
of the same state to $10^{-9}$ relative; the independent numbers run
reports $1.3\times10^{-15}$.

### 5.4 What the spiral curvature measures

The small-$\mathbf q$ curvature of $E(\mathbf q)$ can be read off the
Green's function in three equivalent ways, each stated as one exact
fact below.

#### 5.4.1 Transverse response matrix

**Result.** Rotate the local exchange fields transversally (changing
their lab-frame directions) and expand the energy to second order,
$E(\{\theta_a\})=E_0+\tfrac12\theta^{\sf T}M\theta$. Global SU(2)
invariance of the frozen Hamiltonian forces a zero mode, so the row
sums of $M$ vanish; the flat-spiral curvature is then the exact
contraction of $M$ with the spiral direction vector:

$$
C \equiv \frac{d^2}{dq^2}E(\theta_a = 2\pi q a)\Big|_0
= (2\pi)^2\,\mathbf a^{\sf T} M\,\mathbf a .
\tag{7}
$$

The **pair exchange kernel** is the off-diagonal element of this
response, $j(R)=-M(0,R)$: the local-force (LKAG) object, identical to
the dimer kernel verified in `spiral_force_theorem_J.py` §5. The
textbook sum rule

$$
C/N = (2\pi)^2\sum_{R\neq0} R^2 j(R)
\tag{8}
$$

holds in the infinite-chain limit. **On a finite ring it does not hold
exactly**, because the $R$ and $-R$ shells mix modulo $N$; Eq. (7) is
the exact finite-system statement, and an implementation should either
work in the thermodynamic limit or use Eq. (7) directly.

*Verified:* zero mode to $10^{-6}$, Eq. (7) to
$<5\times10^{-3}$ relative on three random metallic rings.

#### 5.4.2 Flux structure

**Result (i) — flux-free rotations are gauge.** In a spin-diagonal
model, rotating two cells by equal and opposite angles — zero net
angle change, zero flux — changes the band energy by nothing at all,
exactly. The spiral stiffness is a **flux** response at this
Hamiltonian level: it is carried by the twisted boundary condition of
the folded construction. An implementation that builds $E(\mathbf q)$
from local rotations without the seam and flux bookkeeping cannot see
the stiffness.

**Result (ii) — filled bands are inert.** A completely filled isolated
band contributes exactly zero spiral stiffness, because its energy is
the trace $\operatorname{Tr}H^{\uparrow\rm band}$ of the flux-twisted
matrix, which is flux-independent. The stiffness of an itinerant
magnet therefore lives in **partially filled** bands (the Stoner
mechanism), and a saturated single-band insulator is a degenerate
test case: the true stiffness and the naive exchange kernel both
vanish identically there.

*Verified:* gauge invariance to $10^{-12}$; saturated-insulator
stiffness $<10^{-6}$.

#### 5.4.3 The q-space kernel identity

**Result.** TB2J's q-space Green-function exchange module computes the
same object the spiral dispersion measures. The $\mathbf q$-shifted
product of collinear resolvents is the Fourier transform of the
real-space exchange kernel:

$$
\frac{1}{N_k}\sum_k \operatorname{Tr}\big[
\Delta\,G^{\uparrow}(k,E)\,\Delta\,G^{\downarrow}(k{+}\mathbf q,E)\big]
= \sum_{d} e^{+2\pi i \mathbf q\cdot d}\,A(d,E),
\qquad
A(d,E)=\operatorname{Tr}\big[\Delta\,G^{\uparrow}(d,E)\,
\Delta\,G^{\downarrow}(-d,E)\big] ,
\tag{9}
$$

with the sum over unique mod-$N$ displacements (the $N/2$ residue
appears once). This is an algebraic identity, not a model statement,
so any implementation can use `exchange_qspace` as an independent
cross-check of the spiral $J(\mathbf q)$. The sign convention follows
the code: the $\mathbf q$-product pairs with the
$e^{-2\pi i\mathbf q\cdot R}$ inversion of `exchange_qspace.q_to_r`
and `magnon3.Jq`; the onsite $d=0$ term present in Eq. (9) is dropped
when the pair kernel is built.

*Verified:* $<10^{-10}$ at $\mathbf q=1/6,\,2/6,\,1/2$ on a six-cell
ring.

## 6. Implementation consequences (for the spec)

- $E(\mathbf q)$ may be computed from eigenvalue sums per folded
  $\mathbf k$, or from the contour resolvent of Eqs. (5)–(6); these are
  the same object, and the resolvent path additionally yields
  site-resolved quantities through Eq. (4).
- The MFT-vs-LKAG cross-check is now grounded in two independent
  statements: the second-order kernel identity (Section 5.4.1 and
  script 2 §5) and the q-space Fourier identity (Eq. 9).
- Spiral codes must use the folded (twisted) construction: local
  rotations alone miss the flux (Section 5.4.2), and saturated
  references are degenerate test cases.
- Finite-ring diagnostics: $J(R)$ recovered from a small ring is
  reliable, but the stiffness sum rule Eq. (8) must be applied in the
  infinite-chain limit or replaced by Eq. (7).
