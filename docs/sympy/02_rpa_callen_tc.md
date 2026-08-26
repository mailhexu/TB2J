# 02 — Temperature-dependent magnons, RPA, Callen decoupling, and critical temperature

Companion script: [`02_rpa_callen_tc.py`](02_rpa_callen_tc.py) — **all assertions PASS**.

This is the from-scratch finite-temperature continuation of
[`01_heisenberg_hp_lswt.md`](01_heisenberg_hp_lswt.md). It derives and verifies the
Tyablikov/RPA and Callen Green-function decouplings, the Callen magnetization relation,
RPA $T_\mathrm C$, and the contrasting HP mean-field breakdown. The same calculation
applies to a collinear antiferromagnet after choosing local sublattice frames; the
multi-site matrix generalization and the TB2J bridge are in
[`03_anisotropy_multisite_conventions.md`](03_anisotropy_multisite_conventions.md).

Primary source: arXiv:2405.00477, source at `Refs/2405.00477/main.tex`; named equation
labels below are those in that source.

---

## Assumptions and notation

The starting model is a collinear Bravais ferromagnet with dimensionless spins and
$\hbar=k_\mathrm B=1$:

$$
H=-\frac12\sum_{i\ne j}J_{ij}\,\mathbf S_i\!\cdot\!\mathbf S_j,
\qquad J_{ij}=J_{ji},\quad J>0\ \text{ferromagnetic}.
$$

$$
J_{\mathbf q}=\sum_{\mathbf R}J_{0\mathbf R}e^{i\mathbf q\cdot\mathbf R},
\qquad J_0=J_{\mathbf q=0},
\qquad m\equiv\langle S^z\rangle .
$$

The retarded transverse Green function and Bose factor are

$$
G^{+-}_{ij}(t)=-i\theta(t)\langle[S_i^+(t),S_j^-]\rangle,
\qquad
n^{\rm B}(\omega,T)=\frac{1}{e^{\omega/T}-1}.
$$

The uniform $\Gamma$ Goldstone mode is excluded from finite discrete-q Bose sums. It is
one state of measure zero in the thermodynamic integral, while $n^{\rm B}(0)$ is
undefined on a finite mesh.

| Symbol | Meaning |
|---|---|
| $m$ | longitudinal magnetization $\langle S^z\rangle$ |
| $\phi$ | mean boson factor $N_q^{-1}\sum_{\mathbf q}n^{\rm B}(\omega_{\mathbf q})$ |
| $\alpha$ | Callen parameter, $m/(2S^2)$ |
| $C_{ij}$ | transverse correlation $\langle S_i^-S_j^+\rangle$ |
| RPA | Tyablikov decoupling of the Green-function EOM |
| CD | Callen decoupling; RPA plus transverse-correlation feedback |
| HP-MF | quartic Holstein–Primakoff interaction in Hartree–Fock decoupling |

---

## 1. Exact equation of motion

The spin commutator (`spin-comm-H`) is

$$
[S_i^+,H]=\sum_{k\ne i}J_{ik}\left(S_i^+S_k^z-S_i^zS_k^+\right).
$$

Fourier-transforming the retarded EOM (`eqnofmotion-Gfn`) gives

$$
(\omega+i\eta)G_{ij}^{+-}(\omega)
=2m\delta_{ij}+
\sum_{k\ne i}J_{ik}
\left\langle\!\left\langle
S_i^+S_k^z-S_i^zS_k^+;S_j^-\right\rangle\!\right\rangle_\omega .
$$

It is not closed: the exchange term contains a three-spin Green function. Finite-
 temperature magnon schemes differ exactly in how they decouple this object.

**Verified.** The script constructs exact $S=\frac12$ and $S=1$ two-site matrices and
checks the commutator. It independently constructs the Lehmann Green function for the
$S=\frac12$ dimer at finite $\beta$ and verifies the EOM for three frequencies and three
site pairs. Thus the approximations below start from an exact identity, not an assumed
pole form.

## 2. Tyablikov / RPA decoupling

RPA replaces a longitudinal operator on a different site by its thermal average
(`rpa-approx`):

$$
\left\langle\!\left\langle S_i^zS_j^+;S_k^-\right\rangle\!\right\rangle
\simeq mG_{jk}^{+-},\qquad i\ne j.
$$

The translationally invariant EOM is then

$$
\left[\omega-m\left(J_0-J_{\mathbf q}\right)\right]
G_{\mathbf q}^{+-}(\omega)=2m,
$$

so that

$$
\boxed{
G_{\mathbf q}^{+-}(\omega)=
\frac{2m}{\omega-\omega_{\mathbf q}^{\rm RPA}+i\eta},
\qquad
\omega_{\mathbf q}^{\rm RPA}=m\left(J_0-J_{\mathbf q}\right).}
$$

This is paper Eqs. `G_rpa` and `rpa-magnons3d`. At $T=0$, $m=S$, hence it is exactly
the LSWT dispersion in part 01. At nonzero $T$, RPA softens every mode through $m(T)$.

**Verified.** On a periodic four-site ring, the script solves the site-space
RPA EOM symbolically, Fourier transforms it, and proves its poles are
$m[J_0-2J\cos(q)]$ with residue $2m$. The $q=0$ pole remains exactly zero.

## 3. Callen decoupling

Callen starts from three algebraically equivalent representations of $S^z$ and packages
them (`Sz-cd`) as

$$
S_i^z=
\alpha\bigl[S(S+1)-(S_i^z)^2\bigr]
+\frac{1-\alpha}{2}S_i^+S_i^-
-\frac{1+\alpha}{2}S_i^-S_i^+ .
$$

For arbitrary $\alpha$, this is an exact spin identity. The decoupling adds the
transverse correlation neglected by RPA (`cd-approx`):

$$
\left\langle\!\left\langle S_i^zS_j^+;S_k^-\right\rangle\!\right\rangle
\simeq
mG_{jk}^{+-}
-\alpha\langle S_i^-S_j^+\rangle G_{ik}^{+-},
\qquad i\ne j,
$$

with Callen's interpolation choice

$$
\alpha=\frac{m}{2S^2}.
$$

The fluctuation-dissipation relation follows from the Green-function pole:

$$
\langle S_i^-S_j^+\rangle=
\frac{2m}{N_q}\sum_{\mathbf q}
e^{i\mathbf q\cdot(\mathbf R_j-\mathbf R_i)}
 n^{\rm B}(\omega_{\mathbf q}).
$$

Substitution into the EOM yields paper Eq. `cd-magnon3d`:

$$
\boxed{
\omega_{\mathbf q}^{\rm CD}=
 m(J_0-J_{\mathbf q})+
 \frac{m^2}{S^2N_q}\sum_{\mathbf q'}
 \bigl(J_{\mathbf q'}-J_{\mathbf q-\mathbf q'}\bigr)
 n^{\rm B}(\omega_{\mathbf q'}).}
$$

This correction is the HP interaction correction multiplied by $m^2/S^2$; it therefore
vanishes smoothly with the order parameter near $T_\mathrm C$.

**Verified.** The script checks the $\alpha$ identity by exact spin matrices for
$S=\frac12,1,\frac32,2$. On the two-site ring, it solves the CD-decoupled EOM exactly:
$G_\Gamma=2m/\omega$ and
$G_X=2m/[\omega-4J(m+\alpha C)]$. It then substitutes the spectral-theorem correlation
and proves that result is precisely the $N_q=2$ specialization of the boxed formula.

## 4. Magnetization closure

The RPA/CD pole locations alone do not determine $m$. Callen's auxiliary-Green-function
closure (`mag`) is

$$
\boxed{
m=
\frac{(S-\phi)(1+\phi)^{2S+1}+(S+1+\phi)\phi^{2S+1}}
{(1+\phi)^{2S+1}-\phi^{2S+1}},
\qquad
\phi=\frac1{N_q}\sum_{\mathbf q}n^{\rm B}(\omega_{\mathbf q}).}
$$

The two useful limits are

$$
m=S-\phi+\mathcal O(\phi^2)\quad(\phi\to0),
\qquad
m=\frac{S(S+1)}{3\phi}+\mathcal O(\phi^{-3})\quad(\phi\to\infty).
$$

For $S=\frac12$, the formula reduces exactly to

$$
m=\frac{1}{2+4\phi},
$$

which also follows directly from
$S^-S^+=\frac12-S^z$ and
$\langle S^-S^+\rangle=2m\phi$.

**Verified.** Both limits are expanded symbolically; the $S=\frac12$ relation is
checked by exact simplification and by the spin-$\frac12$ operator identity.

## 5. RPA critical temperature

Near a continuous transition, $m\to0$ and therefore

$$
n^{\rm B}(\omega)=\frac{T}{\omega}-\frac12+\frac{\omega}{12T}+
\mathcal O(\omega^3).
$$

Using $\omega_{\mathbf q}^{\rm RPA}=m(J_0-J_{\mathbf q})$ and the large-$\phi$ limit
of Callen's formula gives

$$
\phi=\frac{T}{m}\frac1{N_q}\sum_{\mathbf q}\frac1{J_0-J_{\mathbf q}}+\mathcal O(1),
$$

and hence

$$
\boxed{
k_\mathrm BT_\mathrm C^{\rm RPA}=
\frac{S(S+1)}{3}
\left[\frac1{N_q}\sum_{\mathbf q}\frac1{J_0-J_{\mathbf q}}\right]^{-1}.}
$$

This is `Tc3d-rpa`. In the classical limit, replace $S(S+1)$ by $S^2$. The same general
critical formula with the Weiss dispersion $\omega=mJ_0$ yields
$k_\mathrm BT_\mathrm C^{\rm MFA}=J_0S(S+1)/3$.

The sum exposes the Mermin–Wagner constraint: for an isotropic 1D or 2D model,
$J_0-J_{\mathbf q}\sim q^2$ makes the integral diverge, so RPA gives
$T_\mathrm C=0$. A physical anisotropy gap regularizes that infrared divergence.

## 6. HP mean-field comparison

Retaining the HP quartic terms and Hartree–Fock decoupling gives paper
`HP_magnon_energy`:

$$
\omega_{\mathbf q}^{\rm HP}=
 m(J_0-J_{\mathbf q})+
\frac1{N_q}\sum_{\mathbf q'}
\bigl(J_{\mathbf q'}-J_{\mathbf q-\mathbf q'}\bigr)n^{\rm B}_{\mathbf q'},
\qquad m=S-\phi.
$$

Unlike CD, this interaction correction does **not** acquire the $m^2/S^2$ factor. With
increasing $T$, it can drive one or more mode energies negative while $m>0$. Bose factors
then cease to exist; the paper identifies the accompanying $dm/dT\to-\infty$ point as
its HP critical-temperature estimate. It is not a continuous $m\to0$ transition.

## 7. Numerical simple-cubic benchmark ($J=1$)

For simple-cubic nearest-neighbour exchange,

$$
J_{\mathbf q}=2(\cos q_x+\cos q_y+\cos q_z),
\qquad J_0=6.
$$

A sequence of $8^3$ through $128^3$ $\Gamma$-excluded meshes, with endpoint Richardson
extrapolation, gives the Watson lattice-Green-function value

$$
3\left\langle\frac1{3-\cos q_x-\cos q_y-\cos q_z}\right\rangle
=1.51638532,
$$

within $5\times10^{-5}$ of $1.516386059$. Consequently,

$$
k_\mathrm BT_\mathrm C^{\rm RPA}=1.318926\,J\,S(S+1),
\qquad
k_\mathrm BT_{\mathrm C,\mathrm{cl}}^{\rm RPA}=1.318926\,JS^2.
$$

The self-consistent $24^3$ RPA implementation at $S=\frac12$ gives
$T_\mathrm C=1.027452J$, agreeing with its same-mesh closed form
$1.027352J$ to $9.74\times10^{-5}$ relative error. HP first reaches a negative mode at
finite $m/S\simeq0.28$ in the tested $S=\frac12,1,20$ cases. CD instead decreases
smoothly toward zero and approaches the classical value $1.49JS^2$, close to the paper's
classical Monte-Carlo reference $1.4429JS^2$; RPA is $8.6\%$ below it.

---

## Verified checks and implementation boundary

The executable report performs all of the following before printing `ALL CHECKS PASSED`:

1. Exact spin commutators and `spin-comm-H` on two-site $S=\frac12,1$ clusters.
2. Exact Lehmann EOM for the $S=\frac12$ dimer.
3. Symbolic four-site RPA pole and residue derivation.
4. Exact Callen $\alpha$ identity, two-site CD pole, and spectral-theorem reduction to
   `cd-magnon3d`.
5. Exact $S=\frac12$, low-$\phi$, and large-$\phi$ checks of `mag`; symbolic Bose expansion
   and the resulting `Tc3d-rpa` / MFA formula.
6. Watson-sum convergence, RPA self-consistency versus its closed form, CD/HP
   self-consistent magnetization curves, HP finite-$m$ termination, and classical-limit
   RPA/CD values.

TB2J currently provides the $T=0$ multi-sublattice BdG solver only. The exact mapping
from these collinear formulae (including anisotropy and documented appendix errata) to
`TB2J/magnon/magnon3.py` is established in part 03. Before implementation, unresolved
product decisions remain: which methods are exposed; whether CD is allowed for low spin;
how the q-mesh converges near a 2D gap; and how the paper's equivalent-spin FM decoupling
is extended to inequivalent-sublattice antiferromagnets and a Néel temperature.
