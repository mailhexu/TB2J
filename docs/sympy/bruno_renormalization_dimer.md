# Bruno-renormalization dimer authority (Phase 1)

Executable authority: `bruno_renormalization_dimer.py` (run with
`--kosmic-json`; every statement is an assertion). Scope: exactly-solvable
two-site, single-orbital dimers with real hopping 1/2 eV and onsite
splitting 4 eV; T = 0 (step occupations) and beta = 1/eV where stated.

## Verified statements

| # | Statement | Exact result |
|---|-----------|--------------|
| S1 | FM-aligned staggered reference (kernel authority anchor) | kernel = band = $-1/30$ |
| S2 | FM fully-polarized filling (up empty, down filled) | kernel = band = $-1/30$ |
| S3 | Mirror-staggered dimer (double pole at $-\sqrt{17}/2$) | kernel = band = $2\sqrt{17}/289$ |
| S4 | $\theta$-$\phi$ cross curvature $B$ of the grand potential | $B \equiv 0$ (FM and staggered; T=0 and $\beta=1$) |
| S5 | Implemented $J_{\rm norm}$ correction in-domain (required: 0) | $-17/160$ (M = spinat), $-1/80$ (M = self-consistent) |
| S6 | Uniform null mode of the exact longitudinal curvature $C$ | $C\,(1,1)^T = 0$; code `Xq` invertible in this class |

## Consequences

1. **The pinned kernel identity is filling-robust** in this class: the
   Lichtenstein kernel equals the band pair curvature at every tested
   filling/configuration, including the fully-polarized case. An earlier
   session note claiming a "factor 2" was a scratch-contour
   normalization error and is retracted.
2. **In-domain, the Bruno correction must vanish identically**: with
   $B \equiv 0$ the relaxed curvature equals the frozen one. The
   implemented formula $J + \tfrac12 (M-K_q^T)X_q^{-1}(M-K_q)$ emits
   nonzero rational corrections (S5). Neither the `spinat` input moment
   nor the self-consistent local moment makes $M-K_q$ vanish or place it
   in any null space: **the formula structure is defective**, not its
   inputs.
3. **Singular handling (FR-020)**: the exact longitudinal curvature $C$
   carries the uniform-mode null vector $(1,1)$ — a uniform shift of both
   site fields leaves the grand potential invariant. The code `Xq` is
   invertible in this dimer class; its observed singularity on real
   SrMnO$_3$ models (Story 003 `LinAlgError`) has a different, still
   uncharacterized cause and must be diagnosed on the real orbital/k-grid
   structure before FR-020 design freezes.

## Remaining Phase-1 work (Story 008)

(i) Derive the correct relaxation correction from the self-consistency
structure in the minimal multi-orbital dimer where $B \neq 0$ (single
orbital cannot exercise it); (ii) characterize the code `Xq` singularity
on real models; (iii) only then re-align `bruno_renormalize` and the
dual raw/Bruno output, with the S5 dimer as the failing-before/passing-after
regression (required pair value: correction exactly 0).

## Multi-orbital extension (mpmath, 40 digits, session-verified)

S4 generalizes: in a 2-site, 2-orbital staggered dimer (orbital splittings
4/2 eV, hoppings 1/2/0.3 eV, $\beta = 1$/eV, partially occupied), the
$\theta$-$\phi$ cross curvature $B$ remains identically zero, and $C$
keeps the uniform null mode. Direct $\Omega(\theta)$ finite differences
match the corrected perturbation machinery to 8 digits
($A_{11} = 0.0059172602$). Conclusion: **within any collinear
one-particle dimer class, transverse and longitudinal stiffnesses
decouple at second order** — orbital structure alone cannot activate the
relaxation channel. $B \neq 0$ requires potential relaxation
(self-consistency), which is exactly the content of Bruno's correction
in the literature. Two consequences for Phase 2:

1. **Degenerate-limit rule**: whenever the reference spectrum is exactly
   degenerate (mirror-staggered and FM-aligned dimers are), the response
   weights need $(f_n-f_m)/(\varepsilon_n-\varepsilon_m) \to
   -\beta f(1-f)$ at $\varepsilon_n=\varepsilon_m$; skipping exact
   degeneracies (as scratch scripts did) silently drops finite terms and
   misstates $A$ by $\mathcal{O}(1)$. The ported spectral oracle and the
   production divided differences must declare this limit for real
   metals with degenerate bands.
2. **Design principle for the corrected formula**: a correct Bruno
   renormalization must satisfy $M-K=0$ identically when the reference
   is self-consistent (then the correction vanishes in-domain, matching
   $B=0$), and activate only for non-self-consistent input moments. The
   implemented $M$ (`spinat` input) and $K$ (its own normalization) do
   not satisfy this identity in-domain — the redesign target is a
   consistent $(M,K,X)$ triple derived from the same Green functions
   with the self-consistency variational structure
   ($\Delta E_{\rm scf} = \Delta E_{\rm frozen} - \frac12 \Delta V^\dagger
   \chi^{-1} \Delta V$ shape), not the current bookkeeping.

## Corrected formula derived and validated (Phase 2, mpmath, session)

The relaxation channel acts at **noncollinear references**: for collinear
references the self-consistent longitudinal field responds only at
$O(\theta^2)$, so the correction vanishes identically. At a canted
reference ($\theta_1=\alpha$, $\theta_2=-\alpha$) with a Stoner
double-counting term:

$$
J_{\rm scf} \;=\; J_{\rm frozen} \;-\; \bigl(B\,C^{-1}B^{T}\bigr)_{12},
$$

with $B=\partial^2E_{\rm tot}/\partial\theta\,\partial\varphi$ (transverse–
longitudinal cross response) and $C=\partial^2E_{\rm tot}/\partial\varphi^2$
(screened longitudinal stiffness). Validated against direct SCF
minimization to 9 digits ($\alpha=0.2/0.4/0.6$; e.g. correction
$-0.0195995240$ direct vs $-0.0195995243$ perturbative at $\alpha=0.2$).
Band parts at $\alpha=0.4$: $B_{11}=+0.05649$, $B_{12}=+0.02376$
(antisymmetric pattern), $C_{\rm band}=-0.11927$.

The implemented $J+\tfrac12(M{-}K)^T X^{-1}(M{-}K)$ matches neither the
structure nor the value: $B\,C^{-1}B^T$ is a product of two distinct
responses; $M{-}K$ can vanish while $B$ contributes (self-consistent
collinear input at a canted reference) and vice versa. $X$ is, up to
normalization, the bare band susceptibility $C_{\rm band}$; $K$ and $M$
are not part of the exact correction. Implementation bridge (next):
express $B$ and the screened $C$ from the one-particle spectral machinery
in the spiral/Bloch frame plus the double-counting screening; replace
`bruno_renormalize` accordingly, with regressions (i) collinear reference
$\Rightarrow$ correction exactly 0 (S5 dimer), (ii) canted reference
$\Rightarrow$ correction $=-(BC^{-1}B^T)_{12}$.

## Phase 2 bridge specification (session)

Frame construction (validated, gauge-equivalence 1.8e-15): Bloch frame for
the canted pair is
$H_{\rm frame} = {\rm diag}(D\sigma_z/2) + t\,(E_{01}\otimes e^{-i\alpha\sigma_y/2})$,
local-z fields $p\,\sigma_z$ site-diagonal. Vertex dictionary:
transverse vertex $=$ the complex $\sigma_y$-generator on the hopping
(site-asymmetric: $\partial_{\theta_1} = -\tfrac{i}{2}\sigma_y M$,
$\partial_{\theta_2} = +\tfrac{i}{2}\sigma_y M$,
$\partial_{\theta_1}\partial_{\theta_2} = +\tfrac14 M$);
longitudinal vertex $=$ site-diagonal $\sigma_z$. The correction
$-(BC^{-1}B^T)_{12}$ is gauge-invariant; the matrices are not.

Implementation specification for `exchange_qspace`:
1. **New spectral object** $B(i,j;q)$: residue/divided-difference sum of
   ${\rm Tr}[V_\theta G\, V_\varphi G]$ (one transverse vertex, one
   channel-sign vertex), degenerate-limit rule applied.
2. **$C(q)$**: $C_{\rm band}$ is the $\sigma_z\sigma_z$ spectral response
   (existing `Xqe` up to a normalization pinned against
   $C_{\rm band} = -0.11927$ at the reference $\alpha=0.4$, $D=4$,
   $t=1/2$, $\beta=1$); screening needs an XC kernel TB2J lacks:
   default $C = C_{\rm band}$ documented, optional screening parameter.
3. **`bruno_renormalize`**: $J_q - (BC^{-1}B^T)$ per $q$, on the non-null
   subspace of $C$ (declared pseudoinverse; never `np.linalg.inv`).
4. **Regressions**: collinear $\Rightarrow J_{\rm norm}=J_q$ exactly;
   canted dimer $\Rightarrow -(BC^{-1}B^T)_{12}$ equals the validated
   values; $C_{\rm band} = c\,X_{qe}$ with pinned constant.
5. $K$ and $M$ (`spinat`) leave the Bruno path; $M$ diagnostic-only.
