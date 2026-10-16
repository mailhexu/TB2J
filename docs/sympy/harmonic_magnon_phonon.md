# Harmonic mixed magnon–phonon BdG derivation (Story 011)

Run the assertion-checked derivation and solver sweep with:

```bash
python docs/sympy/harmonic_magnon_phonon.py
```

## Controlled quadratic Hamiltonian

Use unit-commutator magnon and phonon bosons, with bare energies $\alpha,\beta>0$ and real coupling $g$, all in eV:

$$H=\alpha b^\dagger b+\beta a^\dagger a+g(b+b^\dagger)(a+a^\dagger).$$

The coupling expands into the number-conserving and counter-rotating terms
$g(b^\dagger a+a^\dagger b+ba+b^\dagger a^\dagger)$.  In the Nambu
ordering $\Psi=(b,a,b^\dagger,a^\dagger)^T$,

$$H=\tfrac12\Psi^\dagger M\Psi+\mathrm{const},\qquad
M=\begin{pmatrix}A&B\\B^T&A\end{pmatrix},\quad
A=\begin{pmatrix}\alpha&g\\g&\beta\end{pmatrix},\quad
B=\begin{pmatrix}0&g\\g&0\end{pmatrix},\quad
G=\operatorname{diag}(1,1,-1,-1).$$

The executable SymPy witness asserts the characteristic polynomial

$$\det(M-EG)=E^4-(\alpha^2+\beta^2)E^2+\alpha^2\beta^2-4\alpha\beta g^2,$$
 
$$E_\pm^2=\frac{\alpha^2+\beta^2}{2}\pm
\sqrt{\left(\frac{\alpha^2-\beta^2}{2}\right)^2+4\alpha\beta g^2}.$$

This is the full quadratic result; dropping $ba+b^\dagger a^\dagger$ would
instead give a number-conserving avoided-crossing model and the wrong
energies.

## Stability

The dynamical frequencies are real and strictly positive exactly when
$M$ is positive definite.  Its determinant is

$$\det M=\alpha\beta(\alpha\beta-4g^2),$$

and the smaller $E_-^2$ decreases monotonically with $g^2$ and reaches zero
at $g^2=\alpha\beta/4$.  Thus

$$4g^2<\alpha\beta$$

is the stable side; equality is marginal and above threshold is unstable.
The implementation measures the unshifted Hessian spectrum, flags
$\lambda_{\min}(M)\leq\mathrm{tol}$, returns NaN energies at those points,
and records a zero shift.  It never rescues a crossing by changing the
Hamiltonian.

## Paraunitary normalization

Let $K$ be NumPy's lower Cholesky factor, $M=KK^\dagger$.  The Hermitian
Colpa matrix is $C=K^\dagger G K$; its eigenpairs are Euclidean-normalized
$(\epsilon_\nu,x_\nu)$.  For every positive branch,

$$t_\nu=\sqrt{\epsilon_\nu}(K^\dagger)^{-1}x_\nu,$$

and negative branches are similarly scaled by $\sqrt{|\epsilon_\nu|}$.
Ordering positive-norm columns before negative-norm partners gives the
paraunitary $T$ satisfying

$$T^\dagger G T=G,\qquad T^\dagger M T=
\operatorname{diag}(E_-,E_+,E_+,E_-).$$

The numerical implementation uses the shared
`TB2J.magnon.eigenstates.colpa_diagonalize_factor` primitive called by both
`Magnon.get_site_boson_modes` and the mixed solver.  The derivation checks
the generalized problem $Mt=EGt$ and the unit metric norm $t^\dagger Gt=1$
symbolically for the controlled positive branch, and checks the full
paraunitary/diagonalization identities numerically at rational stable
parameter points.  It is not an electron-like Hermitian eigensolver.

## Observable weight shares

For one positive mode with energy $E$, define

$$r=\frac{E^2-\alpha^2}{2g\beta},\quad
x_-=(1,r),\quad x_+=(\alpha,\beta r)/E,\quad
u=(x_++x_-)/\sqrt2,\quad v=(x_+-x_-)/\sqrt2.$$

These solve the generalized BdG equation.  A common scale fixed by
$t^\dagger Gt=1$ cancels from the species shares.  Define the magnetic and
lattice coordinate strengths as

$$W_m=|u_b|^2+|v_b|^2=1+(\alpha/E)^2,\qquad
W_p=|u_a|^2+|v_a|^2=r^2[1+(\beta/E)^2],$$

and the returned observable shares as

$$w_m=\frac{W_m}{W_m+W_p},\qquad w_p=1-w_m.$$

At resonance $\alpha=\beta$, substitution of both roots
$E_\pm^2=\alpha^2\pm2\alpha g$ gives $w_m=w_p=1/2$ for each mode.
As $g\to0$, the upper/magnon-like branch tends to $w_m=1$, while the
lower/phonon-like branch tends to $w_m=0$.  The script asserts these exact
limits and compares the implemented mode-column shares with these formulas
through the detuning sweep.

## Hessian-to-coupling conversion (applied exactly once)

The typed input is the physical mixed Hessian
$G_{i,\beta}=\partial^2E/\partial s_{i}\partial w_{\beta}$ with the local
transverse spin amplitude $s_i=S_{i1}+iS_{i2}$ (dimensionless spin units,
$\hbar=1$) and the displacement coordinate in the declared basis.  With
unit-commutator HP bosons and the mass-weighted normal coordinate of a
mode with energy $\omega$,

$$s_i=\sqrt{S_i/2}\,(b_i+b_i^\dagger),\qquad
w_\beta=e_{\beta\nu}(a_\nu+a_\nu^\dagger)/\sqrt{2\omega_\nu},$$

an energy term $G\,s\,w$ is a bosonic coupling

$$\Gamma_{i\nu}=G_{i,\beta}\,e_{\beta\nu}\sqrt{S_i/2}\,/\sqrt{2\omega_\nu},$$

with the per-mode factor $1/\sqrt{2\omega_\nu}$ applied to the
normal-mode amplitudes BEFORE the eigenvector projection (it is indexed
by the mode $\nu$, not by the coordinate $\beta$).  The phonon Nambu
rows are normal-mode bosons: their diagonal block is $\mathrm{diag}(
\omega_\nu)$ in that same mode basis.

with the spin-length and phonon zero-point factors applied exactly once,
in the solver and nowhere else.  For a Cartesian displacement basis
($w=\sqrt{m}\,u$) the derivative converts first as
$G_{\rm mw}=G_{\rm cart}/\sqrt{m}$.  The controlled model is expressed
through this route: with $S=2$ and $\omega=\beta$ the physical Hessian
$G=g\sqrt{2\beta}$ returns exactly the bosonic coupling $g$.

## Operator-defined observable strengths

The physical transverse-spin operator is $\zeta_i=S_{i1}+iS_{i2}=
\sqrt{2S_i}\,b_i$ to linear order (pure annihilation, so its residue is
the particle component alone) and the Hermitian mass-weighted
displacement is $Q=(a+a^\dagger)/\sqrt{2\beta}$.  With the hybrid
expansion $a=\sum_\lambda(t_a\gamma_\lambda+t_{\bar a}\gamma_
\lambda^\dagger)$ the conjugate gives
$\langle0|a^\dagger|\lambda\rangle=t_{\bar a}^*$, so BOTH phonon
components enter the displacement element:

$$\langle0|\zeta|\nu\rangle=\sqrt{2S}\,t_b,\qquad
\langle0|Q|\nu\rangle=(t_a+t_{\bar a}^*)/\sqrt{2\beta},$$

$$M_\nu=2S\,|t_b|^2,\qquad D_\nu=|t_a+t_{\bar a}^*|^2/(2\beta).$$

Bare limits pin both normalizations: the magnon-like mode has
$M=2S$, $D=0$; the phonon-like mode has $M=0$, $D=1/(2\beta)$ (the
zero-point amplitude of $Q$; $t_{\bar a}\to0$ there, so the value is unchanged).
At resonance $t_b=t_a$ on both branches,
so the strength shares are exactly $1/2$.  The solver returns these as
`magnetic_strength`/`displacement_strength` alongside the normalized
amplitude shares `magnetic_weight`/`displacement_weight`.

## General typed Hessian and solver

`MixedSpinDisplacementHessian` names (1) one local transverse spin
coordinate per magnetic site with its spin length $S_i$; (2) displacement
labels `(atom, axis)` and the displacement basis (`mass_weighted` or
`cartesian_angstrom` with `masses_amu`); (3) eV energy units; (4) the
reciprocal-fractional Fourier convention
`Gamma(k) = sum_R Gamma(R) exp(+2 pi i k.R)`; and (5) producer provenance
including observed channels.  Its bare magnon provider returns the full
Story-006 Nambu Hessian in eV — embedded exactly as supplied, including
its opposite-k lower blocks, so nonreciprocal references are preserved
(never reconstructed from the upper block); its bare phonon provider
returns the mass-weighted dynamical matrix in eV$^2$ in the same
displacement ordering.  The full assembled Nambu Hessian retains the
normal and anomalous mixed blocks, so counter-rotating terms are kept.

A scalar-collinear producer claiming a nonzero mixed Hessian is rejected
because it does not observe the required transverse/longitudinal mixed
torque channels.  The explicit
`scalar_collinear_forbidden_control(...)` input instead supplies the bare
sectors with an identically zero mixed block, reproducing bare modes and
zero transfer without fabricating a derivative.

`HarmonicMixedModel` exposes the consumer call
`model.hybrid_modes(points, mixed_hessian)`; the same calculation is
available as the numerical kernel `hybrid_modes(points, mixed_hessian)`.

The derivation script runs `hybrid_modes` on the controlled model for six
detunings and three stable couplings, checks energies, species shares and
operator strengths, then crosses the stability threshold and checks that
no shift was applied.  The maximum observed detuning-sweep discrepancies
are printed by the script; the script's assertions are normative.
