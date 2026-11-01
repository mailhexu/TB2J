# Bruno-renormalization dimer authority

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
| S6 | Uniform null mode of the exact longitudinal curvature $C$ | exact **only at T = 0**; finite-T $C$ invertible |
| S7 | Variational (Schur) identity, symbolic | $A_{\rm scf} = A_{\rm frozen} - B\,C^{-1}\,B$ (single mode) |
| S8 | Canted-reference loop: direct SCF vs perturbative correction | agree to 30 digits; $\Delta J = +\tfrac12 (B C^{-1} B^T)_{12}$ |

## Consequences

1. **The pinned kernel identity is filling-robust** in this class: the
   Lichtenstein kernel equals the band pair curvature at every tested
   filling/configuration, including the fully-polarized case. An earlier
   session note claiming a "factor 2" was a scratch-contour
   normalization error and is retracted.
2. **In-domain, the Bruno correction must vanish identically**: with
   $B \equiv 0$ the relaxed curvature equals the frozen one. The
   legacy $J + \tfrac12 (M-K_q^T)X_q^{-1}(M-K_q)$ formula emits nonzero
   rational corrections (S5): its structure is defective, not its
   inputs. $K$ and $M$ are not part of the exact correction.
3. **Uniform null mode (CR-001 revision)**: at T = 0 the Fermi-derivative
   diagonal terms vanish and the exact $C$ carries the uniform null
   vector $(1,1)$ (a uniform longitudinal shift leaves the band energy
   invariant). At finite temperature under fixed $\mu$ the grand
   potential is *not* invariant under a uniform $\sigma_z$ shift — up and
   down levels move oppositely — and the diagonal
   $-\beta f(1-f)\,(h'_{\phi,nn})^2$ terms make $C$ generically
   invertible. The production pseudoinverse on the non-null subspace
   (FR-020) is therefore a **declared gauge projection at T = 0**, not an
   exact finite-T symmetry. The code `Xq` is invertible in this dimer
   class; its observed singularity on real SrMnO$_3$ models (Story 003
   `LinAlgError`) has a separate, still uncharacterized cause.
4. **Curvature-to-exchange conversion (CR-004)**: with the pinned mapping
   $J = -A_{12}/2$ (the kernel-authority convention behind the $-1/30$
   anchor), the exchange-space relaxation correction is
   $\Delta J = -\Delta A_{12}/2 = +\tfrac12 (B C^{-1} B^T)_{12}$. The
   production formula is $J_{\rm norm} = J_q + \tfrac12 (B C^+ B^T)$
   with $C^+$ the declared pseudoinverse.

## Functional and reference used by the canted loop (CR-002/CR-003)

- **Functional**: the fixed-$\mu$ grand potential
  $\Omega = -\beta^{-1}\sum_n \ln(1 + e^{-\beta\varepsilon_n})$ plus a
  Stoner double-counting term $-\tfrac{I}{2}\sum_i |m_i|^2$ in the
  rotation-invariant **local** moment magnitude
  $|m_i| = \cos\theta_i\,(a_i-b_i) + \sin\theta_i\, 2\,{\rm Re}(c_i)$
  (site density-matrix block $\begin{psmallmatrix} a & c \\ c^\ast &
  b\end{psmallmatrix}$). The CFR2 responses are $\Omega$-Hessians;
  energies of the $\sum f_n\varepsilon_n$ type validate a different
  functional. The p-field itself is the local-z field
  $p\,\sigma_z^{({\rm local})}$.
- **Reference**: $(\theta_1, \theta_2) = (0.4, 0.1)$ — **not**
  mirror-antisymmetric. With the $|m|$ Stoner term a mirror pair
  $(\alpha, -\alpha)$ pins the antisymmetric longitudinal mode ($B = 0$
  by symmetry): the earlier nonzero corrections at mirror pairs were
  artifacts of a non-invariant $m_z$-type Stoner term.
- **S8 result**: direct SCF relaxation ($\Delta J_{\rm direct} = 5.4877
  \times 10^{-3}$) equals the perturbative $+\tfrac12 B_1 B_2 / C$ to 30
  digits. All curvature stencils (frozen $A_{12}$, $B_1$, $B_2$, $C$)
  are evaluated **on the self-consistent branch**
  $p = p^\ast(\theta_{\rm ref})$; the Schur identity is a statement
  about that stationary branch. The branch is a maximum here
  ($C < 0$) — the identity holds for any smooth stationary branch.

## Code contract implemented in `TB2J/exchange_qspace.py`

1. $B_q \equiv 0$ until the finite-q transverse vertex exists (CR-005:
   the earlier transverse-longitudinal trace products are *not* a spin
   trace — for collinear block-diagonal $G$ the exact mixed response
   vanishes identically, while the coded products give a false nonzero
   $B$ for nonreciprocal dispersions). The true vertex is the phase
   derivative of the multi-site basis-phase authority (Story 009).
2. $C_q$ is kept **complex Hermitian** (CR-006): the response is
   extracted as $C = {\rm H}[-i\,M]$ with $M$ the raw contour integral
   and ${\rm H}[X] = (X+X^\dagger)/2$. For pole-sum contours
   ($M = i\times$ response) this is exact; for geometric contours the
   Hermitian symmetrization removes the symmetric principal-value
   partner. Elementwise `.imag` would destroy the Hermitian structure a
   multi-site $C(q)$ carries. The spectral stiffness pinned by
   `tests/test_qspace_raw_response.py` is reproduced.
3. `bruno_renormalize`: $J_{\rm norm} = J_q + \tfrac12 (B C^+ B^T)$ per
   $q$ (CR-004 sign and factor), $C^+$ = Moore-Penrose on the retained
   subspace via `eigh` of the adjoint symmetrization with a relative
   cutoff; never `np.linalg.inv` (FR-020).
4. Dual output (CR-007): `get_Jdict` emits the raw table
   (`exchange_Jdict`) *and* a separately labeled
   `exchange_Jdict_bruno` from $J_{\rm norm,R}$; both plus the flag and
   scope note (`longitudinal_correction_applied`, `bruno_meta`) reach
   `TB2J.pickle` through `SpinIO`. Raw values are retained for
   downstream formats.

## Multi-orbital extension (still valid)

In a 2-site, 2-orbital staggered dimer (orbital splittings 4/2 eV,
hoppings 1/2/0.3 eV, $\beta = 1$/eV, partially occupied), $B$ remains
identically zero: **within any collinear one-particle dimer class,
transverse and longitudinal stiffnesses decouple at second order** —
orbital structure alone cannot activate the relaxation channel. $B \neq 0$
requires potential relaxation (self-consistency) at a noncollinear
reference, which is the content of Bruno's correction in the literature.

**Degenerate-limit rule**: whenever the reference spectrum is exactly
degenerate (mirror-staggered and FM-aligned dimers are), the response
weights need $(f_n-f_m)/(\varepsilon_n-\varepsilon_m) \to -\beta f(1-f)$
at $\varepsilon_n=\varepsilon_m$; skipping exact degeneracies silently
drops finite terms and misstates $A$ by $\mathcal{O}(1)$. The ported
spectral oracle and the production divided differences declare this
limit for real metals with degenerate bands.

## Regressions

- Collinear reference $\Rightarrow B_q = 0$, $J_{\rm norm} = J_q$
  exactly, flag False, dual tables coincide
  (`test_bruno_correction_vanishes_for_collinear_reference`).
- $C_q$ matches the independent spectral divided-difference sum as a
  complex Hermitian matrix
  (`test_longitudinal_stiffness_matches_spectral_reference`).
- S8 (authority): direct SCF equals $+\tfrac12(B C^{-1}B^T)_{12}$ at a
  nonzero-B canted reference.
