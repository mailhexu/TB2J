# EMC magnon angle, AFM local-frame response, and metric normalization

## Authority and execution

The single executable algebra authority is
[`emc_magnon_normalization.py`](emc_magnon_normalization.py). Its
`named_expressions()` exposes the named SymPy expressions, `derive()` returns
the lexically ordered Kosmic MathDoc `results` records, and `run_checks()`
returns numerical observations after executing the symbolic assertions,
finite-angle oracle, stability controls, and retained FM pins. Run from the
TB2J root with:

```text
python docs/sympy/emc_magnon_normalization.py
python docs/sympy/emc_magnon_normalization.py --kosmic-json
```

The JSON option executes the same checks and emits only the version-1
`schema_version` / Python / SymPy / named-`results` payload. The parent
executed both commands in `mydev`; all assertions passed. The Evidence
boundary section records the measured residuals and synthetic-model limits.

## Source convention evidence

The frame and component order below are taken from implementation, not
inferred from a generic Bogoliubov convention:

- `TB2J/magnon/magnon_math.py:30-60` constructs the local frame from
  `magmoms`, uses the projection fallback at collinearity, and defines
  `U = R[:, 0] + 1j * R[:, 1]`, `V = R[:, 2]`. With the default local
  quantization direction `+z`, that fallback gives
  $(\mathbf e_1,\mathbf e_2,\mathbf e_3)=(\mathbf x,\mathbf y,\mathbf z)$
  for a `+z` moment and $(\mathbf x,-\mathbf y,-\mathbf z)$ for a `-z`
  moment. Thus `U=(1,+i,0)` and `(1,-i,0)`, respectively.
- `TB2J/magnon/magnon3.py:61-88,199-240` sets
  $S_i=\lVert\mathrm{magmom}_i\rVert/2$, normalizes the reference moments,
  and assembles `Hq`. Its upper-left normal block contracts
  `U_i J(-q) U_j*`, its upper-right pairing block contracts
  `U_i J(-q) U_j`, the lower-left block is the adjoint, and the lower-right
  normal block uses the conjugate local frame. The full matrix is
  `[[A1-C, B], [B†, A2-C]]`; this pins the two Nambu halves and their
  creation/annihilation placement.
- `TB2J/magnon/magnon3.py:242-269,307-315` factors `H=K K†`, diagonalizes
  `K† g K`, and returns the positive half of its eigenvalues and, when
  requested, the corresponding Euclidean-unit eigenvectors. Its public
  result metadata labels energies in eV. Those returned raw vectors are not
  themselves the physical Nambu vector used below.
  The existing EMC consumer contract in
  `emc/modes.py:17-75` applies
  $\psi=\sqrt{\omega}\,K^{-\dagger}r$, checks the generalized-pencil
  residual and bosonic norm, and keeps singular modes outside that
  Cholesky inverse. This authority preserves and exercises that conversion.
- `TB2J/magnon/magnon3.py:131-181` defines
  $J(q)=\sum_R J(R)e^{-2\pi i q\cdot R}$; the oracle uses the corresponding
  inverse real-space phase $e^{+2\pi i q\cdot R}$ and Fourier extraction
  with $e^{-2\pi i q\cdot R}$.
- `TB2J/magnon/eigenstates.py:228-252,271-297` confirms the exported
  spin-rotation path reads the first `nspin` components and applies the
  frame vector and Bloch phase. It is a visualization route, not the
  Cholesky back-transformed full-Nambu response used in this derivation.

The stability and singular-mode contract is the approved
`electron-magnon-coupling` ADR-004/005/011 contract: test the Hermitian energy
Hessian separately from the bosonic metric and signed dynamical pencil;
reject an indefinite Hessian; select positive-energy, positive-norm modes;
and normalize a singular global rotation only as an angle. This certificate
does not change TB2J production code. In particular, the current
`magnon3.py` fallback shifts a non-Cholesky Hessian to continue its
diagonalization; that fallback is not treated here as a physical stability
waiver.

## FM pins retained

For a local `+z` ferromagnetic spin, the Holstein--Primakoff expansion gives
$n_+=n_x+i n_y=\sqrt{2/S}\,u\,b_q$ and its adjoint
$n_-=\sqrt{2/S}\,u^* b_q^\dagger$. Inserting these into
$D(n_x\sigma_x+n_y\sigma_y)$ yields the annihilation matrix coefficient
$D\sqrt{2/S}\,u\sigma_-$ and the creation coefficient
$D\sqrt{2/S}\,u^*\sigma_+$. The original independent-boson symbolic
derivatives and the small-angle $\sin\theta$ limit remain asserted.

The existing generic Cholesky identities also remain asserted:
$H=K K^\dagger$, $g=\operatorname{diag}(1,-1)$,
$A=K^\dagger gK$, $Ar=\omega r$, and
$\psi=\sqrt{\omega}\,K^{-\dagger}r$. They establish
$A^{-1}=K^{-1}gK^{-\dagger}$ and
$\psi^\dagger g\psi=\omega r^\dagger A^{-1}r=1$ for a Euclidean-unit
positive-energy eigenvector. The zero-substitution pin is not used to
regularize a singular $K$ or to assign a zero-mode boson.

## AFM spin-operator and Nambu derivation

Let $\eta_i=+1$ or $-1$ label an antiparallel collinear reference,
$\mathbf e_{1i}=\mathbf x$, $\mathbf e_{2i}=\eta_i\mathbf y$, and
$\mathbf e_{3i}=\eta_i\mathbf z$. These are the two collinear frame choices
from the source fallback above. For local HP bosons $a_i,a_i^\dagger$,
$$
\delta\mathbf n_i =
\frac{(\mathbf e_{1i}-i\mathbf e_{2i})a_i+
      (\mathbf e_{1i}+i\mathbf e_{2i})a_i^\dagger}{\sqrt{2S_i}}.
$$
Consequently, in the lab frame $n_+=n_x+i n_y$ is
$\sqrt{2/S_i}\,a_i$ on the `+z` site and
$\sqrt{2/S_i}\,a_i^\dagger$ on the `-z` site. The opposite local frame
therefore changes the lab transverse operator; it is not a global FM
substitution.

The source `Hq` blocks identify the ordered local Nambu column as
$$
\Xi_q=(a_{1q},\ldots,a_{nq},
       a^\dagger_{1,-q},\ldots,a^\dagger_{n,-q})^T,\qquad
g=\operatorname{diag}(I_n,-I_n).
$$
Indeed, the $U_iJ(-q)U_j^*$ upper-left contraction is the normal
$a^\dagger_q a_q$ block, while the $U_iJ(-q)U_j$ upper-right contraction
is the pair-creation $a^\dagger_q a^\dagger_{-q}$ block; its lower-left
partner is the adjoint. After the physical back-transform, write the
positive mode as $\psi_q=(u_q,v_q)^T$, where the first half multiplies
$a_q$ and the second half multiplies $a^\dagger_{-q}$ in the mode
annihilator contribution. This ordering is the reason for the following
conversion, rather than an assumed sign choice.

The independent local Cartesian operator quadratures are
$$
n_{x,i}(q)=\frac{a_{iq}+a^\dagger_{i,-q}}{\sqrt{2S_i}},\qquad
n^{\mathrm{loc}}_{y,i}(q)=
\frac{a_{iq}-a^\dagger_{i,-q}}{i\sqrt{2S_i}}.
$$
Substitution of the sourced Nambu components gives their positive-mode
annihilation coefficients and adjoint reverse-process coefficients:
$$
X_i(q)=\frac{u_{iq}+v_{iq}}{\sqrt{2S_i}},\qquad
Y^{\mathrm{loc}}_i(q)=\frac{u_{iq}-v_{iq}}{i\sqrt{2S_i}},
$$
$$
X_i^{\mathrm{cre}}(-q)=X_i(q)^*,\qquad
Y_i^{\mathrm{loc,cre}}(-q)=Y_i^{\mathrm{loc}}(q)^*.
$$
The lab-frame response is
$X_i^{\mathrm{lab}}=X_i$ and
$Y_i^{\mathrm{lab}}=\eta_iY_i^{\mathrm{loc}}$. Thus both transverse
quadratures, the hole contribution, the spin-length factor, the adjoint,
and the opposite-sublattice frame enter explicitly. For a coherent
amplitude $\beta$, the real-space patterns are
$$
\delta n_{x,i}(R)=2\operatorname{Re}
 [\beta X_i(q)e^{+2\pi i q\cdot R}],\quad
\delta n_{y,i}^{\mathrm{lab}}(R)=2\operatorname{Re}
 [\eta_i\beta Y_i^{\mathrm{loc}}(q)e^{+2\pi i q\cdot R}].
$$
The $\beta=1$ and $\beta=i$ cases are the independent cosine and sine
standing-wave quadratures; their $-q$ Fourier components are the complex
conjugates required by a real spin displacement. The script derives these
expressions from independent $a_q$ and $a^\dagger_{-q}$ variables and
exports the formulas as named SymPy results.

## Executable AFM oracle and gates

The runner constructs an explicit two-sublattice Nambu Hessian in the
source block order, with a complex pairing phase. It checks a stable
positive energy Hessian, both signed branches of $gH$, the positive-mode
Cholesky back-transform and bosonic norm, and a nonzero hole component.
Its distinct spin lengths are $(1,2)$, reference moments are
$(+\mathbf z,-\mathbf z)$, and the commensurate four-cell wavevector is
$q=2\pi/4$. For each $\beta=1,i$, the independent finite-angle producer
builds
$\alpha_{Rs}=\epsilon[u_s\beta e^{+iqR}+(v_s\beta)^*e^{-iqR}]$ from the
full Nambu field and its adjoint, maps it to a local spin with polar angle
$\theta_{Rs}=2\arcsin(|\alpha_{Rs}|/\sqrt{2S_s})$ and azimuth
$\arg\alpha_{Rs}$, then transforms that vector with the actual opposite
local frames and spin lengths. Candidate $X,Y$ derivatives are used only
as comparison targets after generating the finite moments. Central
differences compare both local transverse components and their $q/-q$
Fourier pair to the analytic result. The deliberately wrong map
$v_q\mapsto v_q^*$ must disagree with the independently generated oracle.

The toy Hessian is an explicit stable source-order bosonic pencil used to
exercise the AFM component mapping. Its arbitrary coefficients use the
TB2J eigenstate energy unit (eV), but it is not asserted to have been
generated from a material exchange file or from a particular TB2J `JR`
dataset. The finite-angle oracle tests the spin-rotation kinematics
and Nambu map, not a loader, vertex, bond-sign precondition, or material
response. No universal all-bonds-negative exchange rule is imposed.

Separate executable controls reject a Hermitian but indefinite energy
Hessian even though the signed pencil is distinct. The singular AFM
positive-semidefinite projector has null vector
$w=(\sqrt{S_A},\sqrt{S_B},-\sqrt{S_A},-\sqrt{S_B})$ in the ordered
Nambu components. It is a global-angle direction: its quadratic energy and
bosonic metric norm are zero, while the local transverse response is
normalized in angle space and compared with finite-angle rotations of the
opposite moments. For the reduced null-sector family
$H_\delta=\left(\begin{smallmatrix}1+\delta&1\\1&1+\delta\end{smallmatrix}\right)$,
$g=\operatorname{diag}(1,-1)$, the characteristic polynomial of $gH_\delta$
is $\omega^2-\delta(2+\delta)$, so its positive gap tends to zero as
$\delta\to0^+$. The singular calculation itself never inverts $K$,
normalizes a zero mode as a boson, or assigns an eV/boson value.

## Native-route convention (EMC Option-B local axis)

The EMC native vertex consumes per-site lab-frame quadratures in an
interleaved spinor basis with a local-axis field. This section certifies the
exact convention that route must implement; the downstream consumption is
EMC implementation scope, not this authority.

For a site with local frame $(\mathbf x,\eta\mathbf y,\eta\mathbf z)$,
positive local-axis half-splitting $d$, source-ordered Nambu halves
$(u_i, v_i)$, and global-channel half-splitting $D_z$, the certified
amplitudes and factors are:

| Quantity | Certified value | Named result |
|----------|-----------------|--------------|
| $\sigma_x$ amplitude | $u+v$ | `native_route_site_x_amplitude` |
| $\sigma_y$ amplitude | $\eta\,(u-v)$ | `native_route_site_y_amplitude` |
| Local-axis half-splitting | $\eta\,D_z$ | `native_route_local_axis_half_splitting` |
| Generator $\omega_x$ | $-(u-v)$ | `native_route_omega_x` |
| Generator $\omega_y$ | $\eta\,(u+v)$ | `native_route_omega_y` |
| Boson annihilation | $((u+v)-i\eta(u-v))/\sqrt{2S}$ | `native_route_boson_annihilation` |
| Reverse creation $\sigma_x$ | $\overline{u+v}$ | `native_route_reverse_creation_x` |
| Reverse creation $\sigma_y$ | $\overline{\eta\,(u-v)}$ | `native_route_reverse_creation_y` |

The bridge identity certified for $\eta=+1$ and $\eta=-1$ is

$$
d\,\bigl(x\,\sigma_x + y\,\sigma_y\bigr)
= -\frac{i}{2}\bigl[\omega_x\sigma_x+\omega_y\sigma_y,\; d\,\eta\,\sigma_z\bigr],
\qquad x=u+v,\; y=\eta(u-v),
$$

with $d\,\eta\,\sigma_z$ the lab Hamiltonian coefficient of a local-axis
positive field. The half-splitting relation $d=\eta D_z$ returns the lab
coefficient $D_z$ extracted from global spin channels. Symbolic negative
controls establish that conjugating the hole ($v\to\bar v$) changes both
amplitudes and that flipping $\eta$ flips only the $y$ amplitude, each with
generically nonzero difference.

## Evidence boundary

The retained FM formulas are pre-existing pins. The AFM operator mapping,
both local quadratures, opposite-frame lab rotation, nonzero-$v$ numerical
case, finite-angle reverse pairing, wrong-conjugation control, and
stable/unstable/singular controls are new executable assertions.
The native-route amplitudes, bridge identity, half-splitting relation, and
their negative controls are additional executable assertions.
Executed in `mydev` with SymPy 1.14.0 using
`python docs/sympy/emc_magnon_normalization.py`; all assertions passed.

| Check | Observed value |
|-------|----------------|
| Positive energy (synthetic pencil, eV) | `1.9270509831248428` |
| Positive-mode bosonic metric norm | `0.9999999999999998` |
| Nonzero hole amplitude norm | `0.1474760819589093` |
| Minimum energy-Hessian eigenvalue (eV) | `1.1909830056250532` |
| Cosine finite-angle maximum absolute error | `3.651479119071155e-11` |
| Sine finite-angle maximum absolute error | `3.651479119071155e-11` |
| q/-q reverse-pair maximum absolute error | `0.0` |
| Wrong-hole-conjugation mismatch | `0.057429891366605326` |
| Native bridge maximum absolute error | `0.0` |
| Native wrong-eta mismatch | `1.0108161033164496` |
| Native wrong-hole operator mismatch | `0.793385508117839` |
| Singular finite-angle maximum absolute error | `1.6666779067975313e-11` |
| Singular angle-response magnitude | `1.0` |

The signed dynamical eigenvalues were `(-2.4541634565979926,
-1.4270509831248424, 1.927050983124842, 2.9541634565979917)` eV.
The indefinite-Hessian control was rejected. The singular spectral gap
limit was zero; no singular boson normalization was assigned.
The independent finite-angle producer uses the full coherent HP field,
not the candidate Cartesian derivative, as its input.

This authority does not certify loader behavior, EMC vertex integration,
a material, a DFT response model, or a physical Goldstone gate.
