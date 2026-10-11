# QE pseudo-atomic-wavefunction projector pairing

Verification script: `qe_atomic_pao_trace.py` (run in `mydev`). This pins the
algebra of the optional UPF pseudo-atomic-wavefunction channel; it does **not**
prove physical completeness of a finite atomic basis.

Let $\Phi$ be the columns of QE's `atomic_wfc`, $\beta$ the Kleinman–Bylander
projectors, $M=\Phi^\dagger\Phi$, and $B=\Phi^\dagger\beta$. The pseudo
Hamiltonian's site spin vertex contains the smooth spin-dependent XC
potential plus its nonlocal augmentation coefficient $D$ (the spin difference
of `deeq`). Its **covariant** atomic representation is

$$
\Delta_{\mathrm{cov}}=\Phi^\dagger\Delta V_{xc}\Phi
  +B D B^\dagger,\qquad B=\Phi^\dagger\beta.
$$

The exported coefficients $C=\Phi^\dagger\Psi$ and their spectral Green
matrix $G_{\mathrm{cov}}=\Phi^\dagger G\Phi$ are **primal**, unlike the KB
`becp`/separable-`deeq` pair. Consequently the site vertex contracts with the
**dual** atomic Green function,

$$
G_{\mathrm{dual}}=M^{-1}G_{\mathrm{cov}}M^{-1},\qquad
K_{ij}=\operatorname{Tr}[\Delta_{i,\mathrm{cov}}
 G_{ij,\mathrm{dual}}^\uparrow\Delta_{j,\mathrm{cov}}
 G_{ji,\mathrm{dual}}^\downarrow].
$$

`overlap_k=M(k)` records the *full* k-dependent atomic overlap, including
inter-site blocks. In a complete square invertible atomic basis the channel
trace equals the full-space trace exactly. A rectangular basis is an
approximation; the script asserts an example where it differs from the full
trace. No metric-independent KB shortcut or zero-valued `deeq` placeholders
may be used for atomic data. The site augmentation contribution $B D B^\dagger$
is included for US/PAW; it vanishes for NC because `deeq=dvan` is spin
independent. Labels follow QE's real-harmonic ordinal $m=1,\ldots,2l+1$,
not a signed spherical-harmonic magnetic quantum number.

## Site locality of a Bloch-summed atomic basis

`atomic_wfc(k)` is a Bloch sum over periodic images, not an isolated
orbital. Its local covariant matrix $D_a(k)$ has Fourier terms
$\sum_R e^{i k\cdot R} D_a(R)$; the **site-local** $R=0$ vertex must be
extracted from the weighted BZ average,

$$
D_a(R=0)=\frac{\sum_k w_k D_a(k)}{\sum_k w_k}.
$$

The script asserts a two-point exact example where averaging the $\Gamma$
and zone-edge matrices recovers $D(0)$ but the $\Gamma$ matrix alone does
not. In the actual bcc-Fe PAW atomic dump, the extended 4s channel has
Bloch-Gram diagonal $M_{4s,4s}(\Gamma)=10.706$ versus the full-BZ average
$1.06$: a first-$k$ vertex silently includes neighboring images and
substantially overestimates exchange. QE therefore averages the
same-site smooth-XC and $B D B^\dagger$ covariant blocks over up-spin
k-points, while exporting the **full** $M(k)$ per k for the Green dressing.

The script exercises three independent exact Gaussian-integer realizations
in both the complete ($2\times2$) and truncated ($3\times2$) cases; compare
numerical exchange separately on bccFe and FeO before claiming that the
finite pseudo-atomic basis is exchange-ready for a material.
