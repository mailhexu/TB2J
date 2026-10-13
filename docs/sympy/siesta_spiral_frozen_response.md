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
exits nonzero. Section labels below match the executable checks. Revision 3
implements the review findings STD-1 (story-005 incident partitions), STD-2
(independent one-site/two-site constructions with explicitly asserted
conditional equality) and the declared **joint-action convention**: the
story-005 two-sided same-U declaration applied per incident site, with the
passes composed in the declared site order (A first, then B).

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

## A. The declared joint action: ordered two-sided same-U passes

The declared partitions are the story-005 **incident-bond** partitions
(review STD-1): a block belongs to site $p$'s vertex when a side (row $\mu$
or column $\nu$) sits on $p$. For the two-site shadow,
$B_A=\{AA,AB,BA\}$ and $B_B=\{BB,BA,AB\}$: the shared AB/BA bond belongs to
**both** vertices.

**Declared convention (matching the story-005 two-sided same-U
certificate).** A single pass $p$ rotates every $p$-incident block
two-sided with the same $U_p$,

$$
W\mapsto U_p\,W\,U_p^\dagger\quad\text{for all blocks in }B_p,
$$

and the joint perturbation composes the passes in the declared site order,
A first then B:

$$
H(\boldsymbol\theta)=H_{\rm scalar}+
\mathcal P_B\!\left(\mathcal P_A\!\left(B\right)\right),
\qquad S(\boldsymbol\theta)=S,
$$

with $H_{\rm scalar}$ and $S$ frozen. Per block this gives

$$
AA:\ U_AW_{AA}U_A^\dagger,\qquad
BB:\ U_BW_{BB}U_B^\dagger,\qquad
AB,BA:\ U_BU_A\,W\,U_A^\dagger U_B^\dagger .
$$

Every block action is a **conjugation**, so all blocks stay Hermitian under
arbitrary per-site end rotations; the ordered composition is therefore the
convention consistent with the certificate. Contrast assertions pinned in
the script on a shared block $W=\sigma_x$ (A about $z$, B about $y$):

- single-site A action: $R_z\,\hat x\cdot\boldsymbol\sigma\,R_z^\dagger
  =\cos\theta_A\,\sigma_x+\sin\theta_A\,\sigma_y$ (vector rotation);
- ordered joint action: $(R_yR_z\hat x)\cdot\boldsymbol\sigma
  =\cos\theta_A\cos\theta_B\,\sigma_x+\sin\theta_A\,\sigma_y
  -\cos\theta_A\sin\theta_B\,\sigma_z$ — **both end frames compose**;
- the per-side sandwich $U_AWU_B^\dagger$ is **neither of these and is not
  even Hermitian** (asserted): the block-diagonal site rotation
  $\mathcal U=\mathrm{diag}(U_A,U_B)$ is a *different*, undeclared
  convention, and $\mathcal P_B\mathcal P_A(H_0)
  \ne\mathcal U(H_{\rm scalar}+B)\mathcal U^\dagger$ on shared bonds
  (asserted nonzero).

For coordinate $a=(p,\alpha)$ the blockwise two-sided pass derivative is
$\varphi_p(W)=-i(t_\alpha W-Wt_\alpha)$ (with $t_\alpha=\sigma_\alpha/2$)
applied to every $p$-incident block, and

$$
V_a=\varphi_p(B_p),\qquad
W_{aa}=\varphi_{p,\alpha}\!\bigl(\varphi_{p,\alpha}(B_p)\bigr)
=-[t_\alpha,[t_\alpha,B_p]],
$$

(the mixed same-site entry for $\beta\ne\alpha$ is the separate symmetrized
composition $\tfrac12(\varphi_{p,\alpha}\varphi_{p,\beta}
+\varphi_{p,\beta}\varphi_{p,\alpha})(B_p)$, which is nonzero in general;
the diagonal entry above is the pure double commutator),

$$
W_{a\in A,\,b\in B}
=\varphi_B\!\left(V_A\big|_{B\text{-incident}}\right)
=-[t_b,[t_a,W]]\quad\text{on the shared blocks}.
$$

All of these are derivatives of actual trigonometric SU(2) matrices asserted
in the script (all three axes, the quadratic exponential chart, single-site
paths, and the joint ordered path through second order), not transcribed
formulas. Signs are pinned: positive rotation about $y$ sends $+z$ toward
$+x$. The shared-bond mixed seagull is asserted nonzero; the seagull
matrices are Hermitian and exactly symmetric in $(a,b)$ (Clairaut: one
mixed object per axis pair). For **same-axis** end rotations the ordered
mixed vertex coincides with the symmetrized form
$-\tfrac12([t_a,[t_b,W]]+[t_b,[t_a,W]])$ (asserted zero difference); for
**mixed axes** it does not (asserted nonzero) — the declared composition
order is a genuine convention there, and the ordered vertex is the
authority. At nonzero spiral $q$, derivatives must be applied to the same
declared lab-block rotation and linear assembly map with its pinned sector
phases; the exact model here is the $q=0$ shadow where conjugation and the
scalar Fourier phases commute.

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
fixed-length moment rotations, and — under the incident partitions — it
includes the **shared-bond mixed seagull** $W_{ab}$ for $a\ne b$. In the
gapped eigenbasis,

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
part, and the incident partitions of section A are applied. The result is a
genuinely nonorthogonal, noncollinear, **nonstationary** frozen reference.

## G1. Nonstationary counterexample: ordered versus exponential chart

For two axes on one site the declared chart is exponential
$U_A=\exp[-i(\theta_x\sigma_x+\theta_z\sigma_z)/2]$; contrast the ordered
product $U_x(\theta_x)U_z(\theta_z)$ (both applied as two-sided passes; B
frozen). First vertices coincide (asserted); the second mixed vertices
differ by a Jacobi identity,

$$
W^{\mathrm{ordered}}_{xz}-W^{\mathrm{exp}}_{xz}
=-\tfrac12V_y,
\qquad
K^{\mathrm{ordered}}_{xz}-K^{\mathrm{exp}}_{xz}
=-\tfrac12g_y=\tfrac12\tau_y .
$$

For the rational model $g_{Ay}=-212344/357075=-0.594676188476$ and
$\tau_{Ay}=+0.594676188476$; the failed identity's retained correction is
$106172/357075=+0.297338094238$ energy/radian squared, checked by finite
rotations. A separate collinear stationary control asserts $g_y=0$ and a
vanishing correction under that stated assumption.

## G2. One-site/two-site hierarchy (STD-2) on the ordered path

The one-site kernels $\kappa_p=K_{pp}$ come from the **single-site paths**
(the other pass is the identity there, so they are convention-independent
objects); the two-site cross kernel $K_{AB}=K_{A_yB_y}$ comes from the
**joint ordered action** with its shared-bond composition. These are
independent constructions, never a Taylor identity.

Under the declared two-sided convention the common-angle shift is **not** a
conjugation of the joint path: shared blocks rotate by the composed
$U(t_p+\vartheta)U(t_q+\vartheta)$ while onsite blocks rotate by
$U(t_p+\vartheta)$, $U(t_q+\vartheta)$ (asserted nonzero difference). Hence
no symmetry forces $\kappa_A=-K_{AB}=\kappa_B$, and on the rational model
the identification **fails exactly**:

$$
\kappa_A=\frac{5254549055516}{37188245390625}=+0.141295965979,\qquad
K_{AB}=\frac{570363769048}{12396081796875}=+0.0460116170895,
$$

$$
\kappa_B=-\frac{430648886956}{4132027265625}=-0.104222182884,
$$

$$
\kappa_A+K_{AB}=+0.187307583069\ne0,\quad
\kappa_B+K_{AB}=-0.058210565794\ne0,
$$

$$
\kappa_A+2K_{AB}+\kappa_B=\frac{27433666784}{212504259375}
=+0.129102\ne0 .
$$

**Conditional statement.** The one-site/two-site identification
$K_{AB}=-\kappa_A=-\kappa_B$ would require the shared-bond composition to
act as a *single* rotation of the shared amplitude (commuting, single-sided
end action). The declared two-sided composition does not have that
property, so the two-site kernel is the authority; one-site curvatures must
never be substituted for it. The equal-angle sum rule
$\kappa_A+2K_{AB}+\kappa_B=0$ is likewise **not** a property of this
convention (the nonzero residual above is retained, never dropped), and the
gradient sum $g_{Ay}+g_{By}=-0.531832\ne0$ fails with it: a uniform
rotation of both sublattices is not an energy gauge direction under the
declared shared-bond action.

**Counter-model (assembled-S caveat).** Rebuilding the same construction
with a genuinely spin-textured frozen overlap (full-rank rational
whitening; texture asserted nonzero) breaks the residual structure
further: $g_{Ay}+g_{By}=-165358547/131090400=-1.26141\ne0$,
$\kappa_A=-0.231349$, $K_{AB}=-0.514353$, $\kappa_B=-0.106595$ — all three
identifications and the sum rule fail again, with different mechanisms
(metric texture on top of the shared-bond composition). Numerically
re-verified by independent finite-rotation extractions.

**Conditional assumption on the mixed seagull.** Zeroing the shared AB/BA
magnetic splitting makes the mixed seagull vanish **exactly** (asserted
symbolically: it is carried solely by the shared blocks), while the
two-site electronic response coupling survives (asserted numerically on the
gapped variant, whose occupation split is the two lowest eigenstates). A
vanishing cross kernel therefore requires more than the absence of shared
magnetic amplitude.

**Stationary collinear control.** A spin-diagonal, collinear orthogonal
reference has $g=0$, $\kappa_A=\kappa_B$ and $K_{AB}=-\kappa_A$; there the
identification is valid under the stated stationarity/symmetry assumption
because no shared magnetic amplitude exists to compose.

## H. Finite-rotation convergence and acceptance evidence

The numerical oracle applies the ordered two-sided passes with matrix
exponentials, keeps $H_{\rm scalar}$ and $S$ unchanged, and recomputes the
occupied sum using `scipy.linalg.eigvalsh`. All numerical rotation samples
retain $\varepsilon_2<0<\varepsilon_3$. All 16 Hessian entries and 4
gradients are checked against central finite rotations at step
$2\times10^{-4}$, and against independent generalized eigensolves.

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
| 0.20 | $3.02526\,10^{-3}$ | $1.82813\,10^{-3}$ | $2.50374\,10^{-6}$ | $1.26061\,10^{-6}$ |
| 0.10 | $7.54440\,10^{-4}$ | $4.56090\,10^{-4}$ | $1.56578\,10^{-7}$ | $7.95948\,10^{-8}$ |
| 0.05 | $1.88490\,10^{-4}$ | $1.13960\,10^{-4}$ | $9.78737\,10^{-9}$ | $4.98508\,10^{-9}$ |

Key values of the main model (incident partitions, ordered two-sided
shared-bond action):

- $g_{Ay}=-212344/357075=-0.594676188476$,
  $\tau_{Ay}=+0.594676188476$,
  $g_{By}=1496/23805=+0.0628439403487$,
  $g_{Ax}=g_{Bx}=0$. The gradient sum
  $g_{Ay}+g_{By}=-0.531832$ is a nonzero, retained consequence of the
  declared shared-bond composition.
- $K_{AyAy}=+0.141295965979
  =\text{onsite }+2.53481901561+\text{response }-2.39352304963$.
  Dropping the onsite/seagull term would reverse the sign.
- $K_{AyBy}=+0.0460116170895
  =\text{onsite }+1.84960582511+\text{response }-1.80359420802$: the
  shared-bond mixed seagull is required and nonzero.
- $K_{ByBy}=-0.104222182884$.

The report's formulas, signs, metric rules, convention statements and
conditional identities are backed by exact assertions; numerical spot
checks supplement them rather than substitute for the derivation.
