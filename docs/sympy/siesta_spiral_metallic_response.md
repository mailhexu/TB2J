# Fixed-N finite-temperature frozen response (story 013 / FR-016)

Executable authority: `siesta_spiral_metallic_response.py` (mydev). The
story-004 Fe4 primitive producer has a sampled indirect gap of
−0.24038078 eV at 32 electrons; the existing gapped T=0 contour does not
apply. SIESTA declares an electronic temperature of 300 K.

## Ensemble and exact identities

For a single global chemical potential and equal weights for all bands at
each k point (k weights $w_k$), write

$$
\Omega(\theta,\mu)=-\sigma\sum_{kn}w_k\log(1+e^{(\mu-\epsilon_{kn}(\theta))/\sigma}),\quad
N=\sum_{kn}w_k f_{kn},\quad \sigma=k_BT,
$$

$$
F_N(\theta)=\Omega(\theta,\mu(\theta))+\mu(\theta)N_0,\qquad
N(\theta,\mu(\theta))=N_0.
$$

The script asserts $\Omega_\mu=-N$, $f'_{\epsilon}=-f(1-f)/\sigma$,
and the exact degenerate limit of $(f_n-f_m)/(\epsilon_n-\epsilon_m)$.
Per level, $F_N=\sum w f\epsilon-TS$; the numerical independent oracle
asserts this entropy identity at every finite-difference sample.

For vertices $V^a=\partial_aH$, $W^{ab}=\partial_a\partial_bH$ in the
$S$-orthonormal eigenbasis, the fixed-$\mu$ curvature is

$$
\Omega_{ab}=\sum_{kn}w_k f_nW^{ab}_{nn}
+\sum_{knm}w_k\frac{f_n-f_m}{\epsilon_n-\epsilon_m}
V^a_{nm}V^b_{mn},
$$

with the diagonal and degenerate kernel set to $f'_n$. A two-level
symbolic pencil independently differentiates its exact eigenvalues and
asserts the interband denominator, onsite $W$, and diagonal $f'$ terms.

Set $A_a=\sum_{kn}w_k f'_nV^a_{nn}$ and
$D=\sum_{kn}w_k f'_n<0$; note $N_\mu=-D>0$. Implicit differentiation of
the *one* count constraint gives $\mu_b=A_b/D$ and the fixed-$N$ Hessian

$$
K_{ab}=\Omega_{ab}-\frac{A_aA_b}{D}.
$$

The Schur sign is asserted symbolically and by an independent numerical
finite difference of $F_N$ with $\mu$ re-solved at each displaced
Hamiltonian. At $\sigma=0.2$ eV on the two-level test, the analytic/FD
curvature maximum error is $2.08\times10^{-8}$ eV/rad² (step 0.002 rad);
using fixed-$\mu$ curvature instead errs by $3.45\times10^{-4}$
eV/rad². A common level shift has exactly zero fixed-$N$ curvature:
the fixed-$\mu$ diagonal term $D$ is canceled by the Schur term.

The production metallic path must therefore use Fermi-Dirac free-energy
derivatives, one global chemical potential, the occupation derivative on
coincident eigenvalues, and the fixed-$N$ correction. It must not mark
the absent contour/eigenbasis comparison as a numerical zero; its
finite-rotation free-energy check is a separate certificate.

## Verification

`python docs/sympy/siesta_spiral_metallic_response.py` in mydev:
`ALL ASSERTIONS GREEN`; `ruff check` passes. The scripted two-level
Hamiltonian, μ root, and finite-difference energy are independent of the
production SIESTA response implementation.
