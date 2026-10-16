# Canonical site-boson modes from the TB2J Cholesky eigensystem

Run the assertion-checked derivation with `python docs/sympy/site_boson_modes.py`.

## Matrix and Cholesky conventions

For `N` magnetic sites the code constructs a Hermitian bosonic Hessian

$$H(k)=\begin{pmatrix} A(k)&B(k)\\B(k)^\dagger&A(-k)^*\end{pmatrix},\qquad
 g=\operatorname{diag}(I_N,-I_N).$$

The code calls `numpy.linalg.cholesky(H)`, which returns a **lower** factor
$K$ satisfying $H=KK^\dagger$. It then diagonalizes

$$M=K^\dagger gK,\qquad Mx_\nu=\epsilon_\nu x_\nu,$$

using Euclidean-normalized eigenvectors $x_\nu$. In particular, this is not
the alternative $KgK^\dagger$ convention. The exact similarity relation is

$$M=K^\dagger(gH)(K^\dagger)^{-1}.$$

Thus the physical generalized eigenvector of $gH$ is

$$\psi_\nu=(K^\dagger)^{-1}x_\nu,$$

not the transformed-coordinate eigenvector $x_\nu$ itself. For a positive
branch $\epsilon_\nu=\omega_\nu>0$, unit Euclidean norm of $x_\nu$ gives
$\psi_\nu^\dagger g\psi_\nu=1/\omega_\nu$ and
$\psi_\nu^\dagger H\psi_\nu=1$. The normalized physical column is

$$t_\nu=\sqrt{\omega_\nu}(K^\dagger)^{-1}x_\nu.$$

Using all positive and negative branches, scaling each column by
$\sqrt{|\epsilon_\nu|}$, and ordering positive-norm columns before negative-
norm columns gives the paraunitary matrix $T$:

$$T^\dagger gT=g,\qquad T^\dagger HT=\operatorname{diag}(\omega_1,\ldots,\omega_N,
\omega_1,\ldots,\omega_N).$$

Its row order is $(b(k),b^\dagger(-k))$ and its column order is
$(\gamma(k),\gamma^\dagger(-k))$. The exposed `u` and `v` are the two row
blocks of the positive-norm columns, transposed to `(mode, site)`. Under a
mode phase, the one-magnon spin matrix element gets that phase, but its
spectral residue (the matrix element times its adjoint) is invariant. Within
a degenerate subspace, only the summed projector/residue is invariant under a
unitary basis rotation. Story 009 vertices must transform covariantly in the
same mode subspace; this API does not fix a phase or degenerate-subspace gauge.

## Reciprocal partners and local-frame spin factors

The exchange transform used in `Magnon.Jq` is
$J(q)=\sum_R J(R)e^{-2\pi i q\cdot R}$. The bosonic particle-hole identity is

$$H(-k)=\tau_x H(k)^*\tau_x,$$

where $\tau_x$ exchanges particle and hole blocks. For the dynamical matrix
$D(k)=gH(k)$ this gives $D(-k)=-\tau_xD(k)^*\tau_x$. Thus positive-energy
branches at $-k$ pair with the negative-norm branch at $k$; positive energies
at $k$ and $-k$ need not be equal in nonreciprocal systems. Arbitrary
eigenvector phases (and degenerate-subspace rotations) remain unfixed.

`Hq` folds $\sqrt{S_iS_j}$ into normal and anomalous exchange blocks and
uses $S_l$ in the on-site longitudinal term. The returned `u,v` therefore
refer to unit-commutator Holstein--Primakoff bosons, not spin-length-scaled
coordinates. The local frame stores `U_i=e_{1i}+i e_{2i}` and `V_i=e_{3i}`
with $e_{3i}$ along the reference moment. A transverse spin amplitude
carries the separate factor $\sqrt{2S_i}$; it must not be counted again in
`u,v`.

## Small symbolic models

The executable certificate checks three model classes:

- **FM chain:** $B=0$ and $\omega(q)=4JS[1-\cos(q)]$ for the two directed
  nearest-neighbour bonds. The mode is a pure HP boson ($u=1,v=0$), including
  the $q\to0$ Goldstone limit.
- **Two-sublattice AFM:** the symmetric normal/anomalous form
  $A=2JS$, $B=2JS\cos(q)$ has $\omega=2JS|\sin(q)|$. The check retains the
  particle/hole amplitudes and their unit metric norm.
- **Engineered anomalous case:** a positive-stable $2\times2$ model with
  $A>|B|>0$ checks nonzero pair amplitudes and the identity
  $|u|^2-|v|^2=1$.

For an anomalous branch, $v^2=(A/\omega-1)/2$ can diverge as a Goldstone
frequency tends to zero while the bosonic metric norm remains one. Exactly
zero-frequency Cholesky normalization is singular, so the accessor
regularizes only the mode reconstruction with the explicitly reported
$10^{-6}$ eV shift.

A one-magnon energy is exported as an exact symmetry zero only when a
derived criterion holds at a reciprocal-lattice point: the global-rotation
Holstein--Primakoff displacement
$\beta_i=\sqrt{S_i/2}\,U_i^*\cdot(n\times V_i)$ (one candidate per Cartesian
axis, particle and hole blocks conjugated) must be a null vector of the
unshifted Hessian, with residual $\|H\beta\|/\|\beta\|$ reported in
`diagnostics.symmetry_residual`, and the certified candidate count must be
even (global rotations pair particles and holes). The count caps the zeroed
modes. References whose Hessian minimum lies below
$-n\,\varepsilon_{\mathrm{mach}}\,\|H\|$ (a scale-aware numeric-zero
tolerance, reported per point as `stability_tol`) are rejected as unstable;
the legacy $10^{-6}$ eV Cholesky retry shift is never used to mask a
genuinely negative mode.
