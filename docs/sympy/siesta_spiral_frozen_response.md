# Frozen total-H magnetic response: torque and full transverse curvature

Executable authority: [siesta_spiral_frozen_response.py](siesta_spiral_frozen_response.py).
This is the story-008 derivation gate, not a production response implementation.
The script uses exact SymPy equalities (no numerical fallback for symbolic
assertions) and independent SciPy generalized eigensolves and finite rotations.
Run from the TB2J repository in the `mydev` environment:

```sh
/home/hexu/projects/myenvs/mydev/bin/python docs/sympy/siesta_spiral_frozen_response.py
```

A successful run terminates with `ALL ASSERTIONS GREEN`; an assertion failure
exits nonzero. Section labels below match the executable checks. Revision 4
implements the **weighted shared-bond joint action** (the q=0 global
covariance repair): every spin-dependent block is allocated to its distinct
owner sites with weight $1/|P|$, and the joint action is the additive
allocation $B(\boldsymbol\theta)=B+\sum_p w_p(U_pBU_p^\dagger-B)$, replacing
the revision-3 ordered two-sided composition. The serialized vertex
certificate is `VERTEX_VERSION = "2.0"` with `partition_declaration`

```text
weighted site contributions, w=1/count of distinct endpoint owners, each conjugated by full U_site; scalar h0 and S untouched
```

(undeclared case unchanged: `undeclared; rotation gates must refuse`;
`VERTEX_CONVENTION` stays `oroszlany_total_h_spin_split`, which pins
extraction, not rotation). The declaration string is hash-pinned into the
certificate digest, so stale revision-3 (`v1.0`) vertices reject fail-closed.
The response module mirrors this script: `extract_magnetic_vertex` tags each
block with its owner set, `FrozenResponseProblem._phased_blocks` allocates
the weighted shares, `vertex_pencil`/`second_vertex_pencil` build
$V_p=w_p\varphi_p(B)$ and the same-site $W_{pp}$, and `rotated_pencil`
evaluates the weighted sum instead of sequential passes.

## Scope, frozen quantities and normalization

- Rotate the **total-H spin splitting**, not an independently reconstructed XC
  field: each lab-frame magnetic block is $B=\mathbf b\cdot\boldsymbol\sigma$,
  with $b_x=h_3$, $b_y=h_4$, $b_z=(h_1-h_2)/2$ and
  $H_{12}=h_3-i h_4$. The factor $1/2$ is in the SU(2) generator, not an
  additional factor multiplying this declared $B$.
- The reference density and potentials defining the saved Hamiltonian,
  magnetic amplitudes, scalar Hamiltonian, basis, overlap $S$, and electronic
  ensemble are frozen. The eigenstates **do** respond to the rotated
  Hamiltonian. No longitudinal relaxation, screening or self-consistent
  re-minimization is included.
- $Hc_n=\varepsilon_nSc_n$, $c_n^\dagger Sc_m=\delta_{nm}$; $S$ is Hermitian
  positive definite, $H$ and all perturbation derivatives are Hermitian.
  Use the assembled spinor $S(\mathbf k)$ from the pinned spiral pencil,
  including its spin-dependent $\mathbf k\pm\mathbf q/2$ sectors, never a
  replacement scalar lab overlap.
- $T=0$ and a gap separates the occupied and empty subspaces (in the main
  model the split sits at $\mu=0$; the diagnostic variants only need the two
  lowest eigenstates separated from the rest). Occupations are
  $f_n\in\{0,1\}$ and unchanged for sufficiently small rotations; fixed-$N$
  and grand-canonical derivatives coincide while $\mu$ stays in the gap.
  These are **not** metallic/finite-temperature formulas with discarded
  Fermi-surface terms.
- Angles are in radians. $g_a=\partial_aE$ has units energy/radian and
  $K_{ab}=\partial_a\partial_bE$ energy/radian squared. Every occupied
  **spinor state** is counted once, with no extra spin degeneracy factor.
  The model is one $\mathbf k$ point with unit weight; a periodic
  implementation sums these expressions with its normalized $\mathbf k$
  weights, giving energy per declared reference cell.

## A. The declared joint action: weighted site contributions

**Owner sets and weights.** For a spin-dependent block $(\mu,\nu,R)$ let
$P$ be its set of **distinct owner sites**: the sites owning the row
orbital and the column orbital. Onsite blocks have $|P|=1$; cross-site
blocks have $|P|=2$. Each owner $p\in P$ receives weight
$w_p=1/|P|$ — onsite blocks carry the full amplitude, shared blocks carry
**half at each end** — and the weighted share $B_p$ is $w_pB$ on the
$p$-owned blocks and zero elsewhere. The shares reconstruct the reference
splitting exactly, $\sum_p B_p=B$ (asserted).

**Declared convention (revision 4).** With per-site rotations
$U_p=U_p(\boldsymbol\theta)$ acting by full two-sided conjugation,

$$
H(\boldsymbol\theta)=H_{\rm scalar}+\sum_p U_pB_pU_p^\dagger
\;=\;H_{\rm scalar}+B+\sum_pw_p\bigl(U_pBU_p^\dagger-B\bigr),
\qquad S(\boldsymbol\theta)=S,
$$

with $H_{\rm scalar}$ and $S$ frozen. Per block this is

$$
AA:\ U_AB_{AA}U_A^\dagger,\qquad
BB:\ U_BB_{BB}U_B^\dagger,\qquad
AB,BA:\ \tfrac12\bigl(U_AWU_A^\dagger+U_BWU_B^\dagger\bigr).
$$

Every contribution is a conjugation by a **full** $U_p$ scaled by its
weight, so the action is a **sum**, not a composition. Pinned properties
(all asserted exactly in the script):

- **Hermiticity** for arbitrary independent per-site angles.
- **Site-order independence**: there is no composition order to declare.
  The script pins this with content: the declared sum differs from the
  revision-3 sequential ordered composition
  $U_BU_AWU_A^\dagger U_B^\dagger$ on shared blocks (asserted nonzero), and
  from the per-side block-diagonal rotation $\mathcal U B\mathcal
  U^\dagger$, $\mathcal U=\mathrm{diag}(U_A,U_B)$, whose shared-block
  sandwich $U_AWU_B^\dagger$ is **not even Hermitian** (asserted). The two
  algebraic forms of the declaration — share conjugations and the
  $\delta$-allocation $B+\sum_pw_p(U_pBU_p^\dagger-B)$ — agree exactly
  (asserted).
- **q=0 global covariance (the repair).** Under a common rotation the
  weighted contributions sum to **one** conjugation,
  $B(\theta,\theta)=UBU^\dagger$ once (asserted on a shared block and on
  the full joint path). At $\mathbf q=0$ the assembled overlap and the
  scalar Hamiltonian are spin-scalar, $[\,\mathcal U,S\,]=[\,\mathcal
  U,H_{\rm scalar}\,]=0$, so $H(\theta,\theta)=\mathcal
  U(H_{\rm scalar}+B)\mathcal U^\dagger$ with $S$ fixed: the all-site
  equal-angle action **is** a single global spinor rotation and the
  occupied band energy is exactly invariant. This is the normative
  assertion failed by revision 3, whose ordered double action drifted a
  q=0 SOC-free O$_2$ all-site $0.1$ rad $y$ rotation by $-1.682616$ meV;
  here the covariance is asserted symbolically and numerically
  (band energy exactly unchanged at several angles).
- **Common-shift covariance (Ward parent).** For any joint path, shifting
  all site angles of a common axis by the same amount is an exact
  conjugation,
  $H(\mathbf t+\vartheta\mathbf 1)=\mathcal
  U(\vartheta)H(\mathbf t)\mathcal U(\vartheta)^\dagger$
  (asserted; revision 3 asserted the *negation* of this identity). The
  section-G2 Ward identities are its derivatives.

**Vertices.** With $\varphi_p(M)=-i(t_\alpha M-Mt_\alpha)$,
$t_\alpha=\sigma_\alpha/2$, applied blockwise to the $p$-owned blocks:

$$
V_a=\varphi_p(B_p)=w_p\,\varphi_p(B),\qquad
W_{aa}=\varphi_{p,\alpha}\!\bigl(\varphi_{p,\alpha}(B_p)\bigr)
=-[t_\alpha,[t_\alpha,B_p]],
$$

(the mixed same-site entry for $\beta\ne\alpha$ is the symmetrized
$\tfrac12(\varphi_{p,\alpha}\varphi_{p,\beta}
+\varphi_{p,\beta}\varphi_{p,\alpha})(B_p)$, nonzero in general; shared
blocks carry $w=\tfrac12$ — **not** $w^2$, the map is linear in $B_p$), and

$$
W_{ab}=0\qquad\text{for }a\ne b\text{ on different sites:}
$$

the joint action is a sum of single-parameter conjugations, so the
cross-site mixed second vertex **vanishes identically** — with or without
shared magnetic amplitude. Consequently every off-site Hessian entry is
**purely electronic response**. All of these are derivatives of actual
trigonometric SU(2) matrices asserted in the script (all three axes, the
quadratic exponential chart, single-site paths, and the joint weighted
path through second order), not transcribed formulas. Signs are pinned:
positive rotation about $y$ sends $+z$ toward $+x$. The seagull matrices
are Hermitian and exactly symmetric in $(a,b)$. What survives of
chart dependence is **single-site and nonstationary**: two noncommuting
axes on one site still have chart-dependent second vertices (section G1).
At nonzero spiral $q$, derivatives must be applied to the same declared
weighted lab-block allocation and linear assembly map with its pinned
sector phases; the exact model here is the $q=0$ shadow where the
conjugations and the scalar Fourier phases commute.

## B–E. Resolvent, metric placement and signed first variation

Define the contravariant (dual-basis) pencil resolvent and its mixed form:

$$
\widetilde G(z)=(zS-H)^{-1}
=C(zI-\varepsilon)^{-1}C^\dagger,
\quad R(z)=\widetilde G(z)S=(zI-S^{-1}H)^{-1},
$$

so $\widetilde G=R S^{-1}$, with $S^{-1}$ on the **right**. Let
$\langle F\rangle_\Gamma=(2\pi i)^{-1}\oint_\Gamma F(z)\,dz$, with $\Gamma$
counterclockwise enclosing all occupied and no unoccupied poles. The
physical spectral trace is $\operatorname{tr}(S\widetilde G)$:

$$
N=\langle\operatorname{tr}(S\widetilde G)\rangle_\Gamma,
\quad E=\langle z\operatorname{tr}(S\widetilde G)\rangle_\Gamma,
\quad\Omega=\langle(z-\mu)\operatorname{tr}(S\widetilde G)\rangle_\Gamma.
$$

For **covariant Hamiltonian vertices** $V_a$, the response insertions are
$\widetilde G V_a$, **without an additional $S$**; equivalently use the
mixed resolvent and coefficient vertex $v_a=S^{-1}V_a$:
$\operatorname{tr}(Rv_a)=\operatorname{tr}(\widetilde GV_a)$,
$\operatorname{tr}(Rv_aRv_b)=\operatorname{tr}(\widetilde GV_a\widetilde GV_b)$.
Both the omitted-$S$ DOS and the extra-$S$ torque are executable negative
controls. A nonunitary congruence producing a spin-dependent metric is
checked exactly and numerically: the trace rules never assume spin-scalar
$S$.

Direct differentiation of $(zS-H)\widetilde G=I$ gives
$\partial_a\widetilde G=\widetilde GV_a\widetilde G$,
$\partial_z\widetilde G=-\widetilde GS\widetilde G$, hence
$\operatorname{tr}(S\widetilde GV_a\widetilde G)
=-\partial_z\operatorname{tr}(\widetilde GV_a)$, and contour integration by
parts (weight $z$ or $z-\mu$) gives

$$
g_a=\partial_aE=\partial_a\Omega
=\langle\operatorname{tr}(\widetilde G V_a)\rangle_\Gamma
=\sum_n f_n c_n^\dagger V_a c_n .
$$

The signed mechanical torque conjugate to a positive rotation is
**$\tau_a=-g_a$**, so $\delta E=-\sum_a\tau_a\delta\theta_a$. If an
implementation names $g_a$ its torque, its certificate must state the
gradient convention. No reference torque is dropped. The script
independently differentiates a nondegenerate two-level analytic eigenvalue
and its characteristic polynomial (factor two in the mixing term
asserted), and asserts the band/grand contour rearrangements exactly.

## E–F. Full second-order transverse curvature

$$
\partial_a\partial_b\widetilde G=
\widetilde G W_{ab}\widetilde G+
\widetilde G V_a\widetilde G V_b\widetilde G+
\widetilde G V_b\widetilde G V_a\widetilde G
\;\Longrightarrow\;
\boxed{K_{ab}=\langle\operatorname{tr}(\widetilde G W_{ab})+
\operatorname{tr}(\widetilde G V_a\widetilde G V_b)\rangle_\Gamma.}
$$

The first term is the **onsite/seagull** contribution, required even for
fixed-length moment rotations; under the weighted allocation it is
**diagonal in sites**, because $W_{ab}=0$ for $a\ne b$ on different sites.
Every off-site Hessian entry — in particular $K_{A_yB_y}$ — is therefore
the pure response term $\langle\operatorname{tr}(\widetilde GV_a\widetilde
GV_b)\rangle_\Gamma$. In the gapped eigenbasis,

$$
K_{ab}=\sum_n f_n(W_{ab})_{nn}
+\sum_{n,m}L_{nm}(V_a)_{nm}(V_b)_{mn},
\qquad
L_{nm}=\begin{cases}(f_n-f_m)/(\varepsilon_n-\varepsilon_m),&
\varepsilon_n\ne\varepsilon_m,\\0,&n=m.\end{cases}
$$

The result is real and exactly symmetric in $(a,b)$. The energy expansion is
$E=E_0+g_a\theta_a+\tfrac12K_{ab}\theta_a\theta_b+O(\theta^3)$; $K$, not
$K/2$, is the Hessian. The two-level check asserts
$\varepsilon_-''=W_{11}-2|V_{12}|^2/\mathrm{gap}$. Every divided-difference
entry, including repeated poles, is compared with exact SymPy residues, and
every entry of the four-coordinate $(A_x,A_y,B_x,B_y)$ Hessian is compared
with independent numerical generalized eigensolves and finite rotations.

## C. The rational two-site model

$$
L=\begin{pmatrix}1&1/4\\1/6&1\end{pmatrix}\otimes I_2,\quad
S=L^T L,\quad
Q=I_4-\frac{2uu^T}{30},\quad u=(1,2,3,4)^T,
$$

$$
D=\operatorname{diag}(-3,-1,2,4),\quad
H_0=L^TQD Q^T L,\quad C=L^{-1}Q .
$$

$L$ is site-mixed (asymmetric A/B orbital couplings) but **spin-scalar**,
matching the assembled lab overlap of the pinned $q=0$ shadow.
$Q^TQ=I$, $C^\dagger SC=I$ and $H_0C=SCD$ are exact assertions; the
occupied/empty gap is 3 energy units with the split at $\mu=0$. Each
$2\times2$ block is split into its scalar spin trace and traceless magnetic
part, and the weighted owner shares of section A are applied. The result is
a genuinely nonorthogonal, noncollinear, **nonstationary** frozen
reference.

## G1. Nonstationary counterexample: ordered versus exponential chart

For two noncommuting axes on **one** site the declared chart is
exponential $U_A=\exp[-i(\theta_x\sigma_x+\theta_z\sigma_z)/2]$, applied
to the site-A weighted share while site B keeps its share at the reference
value; contrast the ordered product $U_x(\theta_x)U_z(\theta_z)$. First
vertices coincide (asserted); the second mixed vertices differ by a Jacobi
identity,

$$
W^{\mathrm{ordered}}_{xz}-W^{\mathrm{exp}}_{xz}
=-\tfrac12V_y,
\qquad
K^{\mathrm{ordered}}_{xz}-K^{\mathrm{exp}}_{xz}
=-\tfrac12g_y=\tfrac12\tau_y .
$$

For the rational model $g_{Ay}=-5104/15525=-0.328760064412238$ and
$\tau_{Ay}=+0.328760064412238$; the failed identity's retained correction
is $2552/15525=+0.164380032206119$ energy/radian squared, checked by
finite rotations. This is genuine nonstationary physics of the frozen
reference and survives the weighted rule; it involves no cross-site
composition. A separate collinear stationary control asserts $g_y=0$ and a
vanishing correction under that stated assumption.

## G2. One-site/two-site hierarchy (STD-2) and the exact Ward identities

The one-site kernels $\kappa_p=K_{pp}$ come from the **single-site paths**;
the two-site cross kernel $K_{AB}=K_{A_yB_y}$ comes from the **joint
weighted action**. These are independent constructions, never a Taylor
identity — and under revision 4 they are linked by an exact Ward identity
instead of a conditional assumption.

Because the uniform direction $\mathbf u$ (equal angle, equal axis on all
sites) is an exact zero mode, $E(\boldsymbol\theta+s\mathbf u)=
E(\boldsymbol\theta)$ for all $s$, the directional gradient
$g_{\mathbf u}$ vanishes **identically**; differentiating gives

$$
\sum_bK_{ab}\,u_b=0\quad\text{for every }a,
$$

i.e. the Hessian annihilates $\mathbf u$ on **both** sides. At $\mathbf
q=0$ with spin-scalar $S$ the weighted rule therefore yields, exactly and
without any site-symmetry assumption,

$$
g_{Ax}+g_{Bx}=0,\qquad g_{Ay}+g_{By}=0,
$$

$$
K_{Ax Ax}+2K_{AxB x}+K_{BxB x}=0,\qquad
K_{Ay Ay}+2K_{AyBy}+K_{ByBy}=0,
$$

$$
\boxed{\;\kappa_A=-K_{AB}=\kappa_B\;}\qquad\text{(per-axis Ward identity),}
$$

and likewise $K_{Ax Ay}+K_{Ax By}+K_{Bx Ay}+K_{Bx By}=0$. On the rational
model (all values exact SymPy rationals):

$$
\kappa_A=\kappa_B=\frac{31540700264}{70299140625}=+0.448664094377043,
\qquad
K_{AB}=-\frac{31540700264}{70299140625}=-0.448664094377043,
$$

with $K_{AyAy}=\text{onsite }49991/31050=+1.61001610305958+
\text{response }-1.16135200868254$,
$K_{ByBy}=\text{onsite }36259/31050=+1.16776167471820+
\text{response }-0.719097580341154$, and
$K_{AyBy}=\text{onsite }0+\text{response }-0.448664094377043$: the
cross-site seagull is exactly zero (asserted), so the cross kernel is pure
electronic response. The $x$ sector satisfies the same Ward structure,
$K_{AxAx}=K_{BxBx}=-20906104/312440625=-0.0669122461267641$,
$K_{AxBx}=+0.0669122461267641$, and all mixed-axis Hessian entries vanish.

**Counter-model (assembled-S caveat).** Rebuilding the same construction
with a genuinely spin-textured frozen overlap (full-rank rational
whitening; texture asserted nonzero) breaks the gauge — $S$ no longer
commutes with a global spinor rotation — and with it every Ward identity:
$g_{Ay}+g_{By}=1127089/6426000=+0.175395113600996\ne0$,
$\kappa_A=+0.449967575998023$, $K_{AB}=-0.556399814999610$,
$\kappa_B=+0.360179473265550$; the per-site identifications and the sum
rule fail by computed nonzeros (never dropped), numerically re-verified by
independent finite-rotation extractions. The Ward identities are thus a
consequence of the **spin-scalar assembled metric at $\mathbf q=0$** plus
the weighted allocation — not of site symmetry.

**Cross kernel without shared amplitude.** Zeroing the shared AB/BA
magnetic splitting leaves the cross kernel nonzero
($-0.110065$ on the gapped variant): the mixed seagull is zero by
declaration whether or not shared amplitude exists, and the two-site
electronic response coupling survives on its own.

**Stationary collinear control.** A spin-diagonal, collinear orthogonal
reference has $g=0$, $\kappa_A=\kappa_B$ and $K_{AB}=-\kappa_A$; there the
Ward identities hold trivially under the stated stationarity/symmetry
assumption because no shared magnetic amplitude exists to allocate.

## H. Finite-rotation convergence and acceptance evidence

The numerical oracle applies the weighted site contributions with matrix
exponentials, keeps $H_{\rm scalar}$ and $S$ unchanged, and recomputes the
occupied sum using `scipy.linalg.eigvalsh`. All numerical rotation samples
retain $\varepsilon_2<0<\varepsilon_3$. All 16 Hessian entries and 4
gradients are checked against central finite rotations at step
$2\times10^{-4}$, and against independent generalized eigensolves. The
q=0 covariance is checked numerically: equal-angle same-axis joint
rotations leave the occupied band energy exactly $-4$ at several angles,
and the finite-rotation Ward residuals at step $10^{-3}$ are pure
truncation ($\lesssim10^{-7}$, asserted $\le2\times10^{-5}$).

For the nonzero-torque $A_y$ direction, compare
$D_g(h)=\frac{E(h)-E(-h)}{2h}$ and
$D_K(h)=\frac{E(h)-2E(0)+E(-h)}{h^2}$ with the exact gradient and Hessian.
The script asserts the central Taylor coefficients
$D_g=g+E'''h^2/6+E^{(5)}h^4/120+\cdots$ and
$D_K=K+E^{(4)}h^2/12+E^{(6)}h^4/360+\cdots$ symbolically, so the $O(h^2)$
and $O(h^4)$ convergence laws are symbolic results as well as measured
checks; Richardson combinations $[4D(h/2)-D(h)]/3$ cancel the $h^2$ term.
Since $\tau=-g$, torque and gradient error magnitudes are identical.

Observed with SymPy 1.14.0 in `mydev`: exit 0, `ALL ASSERTIONS GREEN`.

| $h$ (radian) | central $g$ error | central $K$ error | Richardson $g$ error | Richardson $K$ error |
|---:|---:|---:|---:|---:|
| 0.20 | $8.6966\,10^{-4}$ | $5.9423\,10^{-4}$ | $3.59300\,10^{-7}$ | $1.03708\,10^{-7}$ |
| 0.10 | $2.1769\,10^{-4}$ | $1.4864\,10^{-4}$ | $2.24780\,10^{-8}$ | $6.48279\,10^{-9}$ |
| 0.05 | $5.44382\,10^{-5}$ | $3.71639\,10^{-5}$ | $1.40530\,10^{-9}$ | $3.94478\,10^{-10}$ |

(measured ratios $\approx4$ and $\approx16$, within the asserted symbolic
convergence laws).

Key values of the main model (owner-weighted allocation):

- $g_{Ay}=-5104/15525=-0.328760064412238$,
  $\tau_{Ay}=+0.328760064412238$,
  $g_{By}=+5104/15525=+0.328760064412238$,
  $g_{Ax}=g_{Bx}=0$. The gradient sums vanish **exactly** — a uniform
  rotation of both sublattices is an exact energy gauge direction under
  the weighted shared-bond action.
- $K_{AyAy}=+0.448664094377043
  =\text{onsite }+1.61001610305958+\text{response }-1.16135200868254$.
  Dropping the onsite/seagull term would reverse the sign.
- $K_{AyBy}=-0.448664094377043=\text{onsite }0+\text{response
  }-0.448664094377043$: the cross-site mixed seagull is exactly zero and
  the cross kernel is pure response.
- $K_{ByBy}=+0.448664094377043$; Ward:
  $\kappa_A+K_{AB}=\kappa_B+K_{AB}=0$ exactly.

The report's formulas, signs, metric rules, convention statements and
Ward identities are backed by exact assertions; numerical spot checks
supplement them rather than substitute for the derivation.
