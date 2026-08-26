# 03 — Single-ion anisotropy, anisotropic exchange, multi-site generalization, and TB2J conventions

Companion script: [`03_anisotropy_multisite_conventions.py`](03_anisotropy_multisite_conventions.py)
(all assertions in this report are executed there; 68 `PASS` lines).

Reference: Pavizhakumari, Skovhus, Olsen, *Beyond the random phase approximation for calculating
Curie temperatures in ferromagnets*, arXiv:2405.00477 (LaTeX source
`TB2J/Refs/2405.00477/main.tex`). Equation labels below refer to that source.

Run with:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
python 03_anisotropy_multisite_conventions.py
```

---

## 1. Assumptions and notation

- Heisenberg Hamiltonian (paper eq. `heisenbrghamiltonian`): $H=-\tfrac12\sum_{i\neq j}J_{ij}\,\mathbf S_i\!\cdot\!\mathbf S_j$
  with $J>0$ ferromagnetic; $\sum_{i\ne j}$ runs over ordered pairs, so each bond appears twice.
  Ground state polarized along $z$; $S^\pm = S^x\pm iS^y$; $\hbar=1$.
- Anisotropy (paper eq. `2dheisenberghamiltonian`):

$$
\Delta H \;=\; -\frac12\sum_{i\neq j}\lambda_{ij}S_i^zS_j^z \;-\; A\sum_i (S_i^z)^2 ,
$$

  with $\lambda$ the *anisotropic exchange* and $A>0$ easy-axis *single-ion anisotropy* (SIA).
- Fourier transforms: $J_{\mathbf q}=\sum_R J_{0R}\,e^{i\mathbf q\cdot\mathbf R}$,
  $\lambda_{\mathbf q}=\sum_{R\ne0}\lambda_{0R}\,e^{i\mathbf q\cdot\mathbf R}$ (paper eqs.
  `Jij-q-space`, `lambda-q-space`); $J_0\equiv J_{\mathbf q=0}$, $\lambda_0\equiv\lambda_{\mathbf q=0}$.
- $\langle S^z\rangle\equiv m$ (site-independent for equivalent sites), Bose factors
  $n^{\mathrm B}_{\mathbf q}=n^{\mathrm B}(\omega_{\mathbf q},T)$, and $\phi=\frac{1}{N_q}\sum_{\mathbf q}n^{\mathrm B}_{\mathbf q}$.
- Multi-site: sublattices $a,b\in\{1,\dots,N_a\}$, cells $i,j$;
  $J^{ab}_{\mathbf q}=\sum_i J^{ab}_{0i}e^{i\mathbf q\cdot\mathbf R_i}$ (eq. `2siteJab`),
  likewise $\lambda^{ab}_{\mathbf q}$ (eq. `2SiteAE`); $J^{aa}_{00}=\lambda^{aa}_{00}=0$; equivalent spins $S_a=S$.
- Verification tools: exact spin-matrix representations ($S=\tfrac12,1,\tfrac32,2$), a 6-state boson
  Fock representation, noncommutative sympy normal ordering with $[a,a^\dagger]=1$ and $[S^z,S^\pm]=\pm S^\pm$,
  exact diagonalization (ED) of small clusters, and a numpy reimplementation of the TB2J conventions.
  Random-coupling tests use the fixed seed `20240517` (deterministic).

---

## 2. Single-ion anisotropy in the HP scheme (eqs. `sia-reorder`, `hp_correction`)

### 2.1 The re-ordering ambiguity

$S_i^z=S-a_i^\dagger a_i$ turns the SIA into $-A(S-a^\dagger a)^2=-AS^2+2SA\,a^\dagger a-A(a^\dagger a)^2$.
The quartic term can be written in two ways that differ in the single-magnon sector (paper eq. `sia-reorder`):

$$
a_i^\dagger a_i\,a_i^\dagger a_i \;=\; a_i^\dagger a_i^\dagger a_i a_i \;+\; a_i^\dagger a_i .
$$

**Verified** (script §2): (i) noncommutative sympy proof by normal ordering with $a\,a^\dagger=a^\dagger a+1$;
(ii) symbolic Fock-state algebra $(a^\dagger a)^2|n\rangle=n^2|n\rangle$, $a^{\dagger 2}a^2|n\rangle=n(n-1)|n\rangle$,
so $n^2=n(n-1)+n$ for all $n$; (iii) explicit $6\times6$ boson matrices.

Only the right-hand side has its quartic piece $a^\dagger a^\dagger aa$ vanish on one-magnon states
($a^\dagger a^\dagger aa\,|1\text{ magnon}\rangle=0$), so HP fixes the ordering *naturally*: the quadratic
piece $a^\dagger a$ belongs in LSWT, giving the SIA contribution

$$
\Delta\omega^{\mathrm{HP}}_{\mathrm{SIA}} \;=\; A(2S-1) \qquad(\text{from } 2AS-A).
$$

### 2.2 Exactness of the one-magnon sector and the T=0 gap

Because $S^z|m{=}S\rangle=S$, $S^z|m{=}S-1\rangle=S-1$, $\langle m{=}S-1|S^-|m{=}S\rangle=\sqrt{2S}$ and
$(S^z)^2$ is diagonal, the quadratic HP Hamiltonian reproduces the one-magnon sector of the *exact*
spectrum for any $S$. **Verified** (script §3): exact ED of the 2-site cluster
$H=-J\,\mathbf S_1\!\cdot\!\mathbf S_2-\lambda S_1^zS_2^z-A[(S_1^z)^2+(S_2^z)^2]$ in the $M=2S-1$ sector gives

$$
\omega_{1,2}\;=\;A(2S-1)+S\lambda_0 \;+\; \{0,\;2SJ\},
$$

symbolically for generic $S$ and by full matrix ED for $S=\tfrac12,1,\tfrac32,2$. Hence the T=0 gap
(paper eq. `delta_cd`, HP branch, together with eq. `hp_correction` at $n^{\mathrm B}=0$):

$$
\Delta^{\mathrm{HP}} \;=\; A(2S-1)+S\lambda_0 \qquad\text{(exact, no decoupling involved).}
$$

The finite-temperature HP correction (paper eq. `hp_correction`):

$$
\Delta\omega^{\mathrm{HP}}_{\mathbf q}=A(2S-1)+\lambda_0\langle S^z\rangle-\frac{1}{N_q}\sum_{\mathbf q'}\bigl(\lambda_{\mathbf q-\mathbf q'}+4A\bigr)n^{\mathrm B}_{\mathbf q'} .
$$

---

## 3. RPA and the operator-ordering ambiguity (eqs. `standard_order`, `rpa_correct_ordering`, `rpa_correction`)

The SIA enters the equation of motion through

$$
-A\bigl\langle\!\bigl\langle [S_i^+,(S_i^z)^2];S_j^-\bigr\rangle\!\bigr\rangle
= A\bigl\langle\!\bigl\langle S_i^zS_i^+ + S_i^+S_i^z;S_j^-\bigr\rangle\!\bigr\rangle ,
\tag{standard\_order}
$$

and, using $[S^z,S^\pm]=\pm S^\pm$ (paper eq. `spin-comm-1`), through the *equivalent* forms

$$
S^zS^++S^+S^z \;=\; 2S^zS^+ - S^+ \;=\; 2S^+S^z + S^+ .
\tag{rpa\_correct\_ordering}
$$

**Verified** (script §1): matrix identities for $S=\tfrac12,1,\tfrac32$ *and* abstract noncommutative proofs
from $[S^z,S^+]=S^+$ alone. Replacing $S^z\to\langle S^z\rangle$ then produces three different coefficients:

| ordering used | SIA coefficient in $\Delta\omega$ |
|---|---|
| `standard_order` (symmetrized) | $2A\langle S^z\rangle$ |
| $2S^zS^+-S^+$ | $A(2\langle S^z\rangle-1)$ |
| $2S^+S^z+S^+$ | $A(2\langle S^z\rangle+1)$ |

so that (paper eq. `rpa_correction`)

$$
\Delta\omega^{\mathrm{RPA}}_{\mathbf q}\;=\;2A\langle S^z\rangle+\lambda_0\langle S^z\rangle ,
$$

q-independent. The anisotropic exchange needs no ordering choice
($[S^+_i,-\tfrac12\sum_j\lambda_{ij}S_i^zS_j^z]=\sum_j\lambda_{ij}S_i^+S_j^z$, verified in §5 below for the
multisite commutator).

**T=0 gaps** (paper eqs. `delta_rpa`, `delta_cd`; script §4):

$$
\Delta^{\mathrm{RPA}}=2AS+S\lambda_0,\qquad
\Delta^{\mathrm{CD}}=\Delta^{\mathrm{HP}}=A(2S-1)+S\lambda_0 .
$$

**$S=\tfrac12$ irrelevance**: $A(2S-1)\equiv0$ for HP/CD (the SIA is the constant $-A/4$), whereas RPA gives a
spurious gap $2AS=A$ — the `standard_order` RPA decoupling cannot reproduce it, and the
`rpa_correct_ordering` variant that does ($2\langle S^z\rangle\to1=2S-1$ at $T=0$) produces a magnetization
that does not vanish continuously at $T_c$; continuity fixes the ordering to `standard_order` (paper §II.G).

---

## 4. Callen decoupling of the SIA (eq. `callen_correction`)

Callen's decoupling (paper eq. `cd-approx`) with $\alpha=\langle S^z\rangle/2S^2$ applied on-site to the
`standard_order` operator gives (fluctuation $\psi\equiv\langle S^-S^+\rangle$, using
$\langle S^+S^-\rangle=\langle S^-S^+\rangle+2\langle S^z\rangle$, i.e. the natural partner fluctuation of
the reversed ordering):

$$
\bigl\langle\!\bigl\langle S^zS^+ + S^+S^z\bigr\rangle\!\bigr\rangle_{\mathrm{CD}}
\;\approx\;\Bigl[2\langle S^z\rangle-\frac{\langle S^z\rangle}{S^2}\bigl(\langle S^z\rangle+\bar\psi\bigr)\Bigr]G,
\qquad \bar\psi\equiv\frac{1}{N_q}\sum_{\mathbf q'}\psi^{aa}_{\mathbf q'}=2\langle S^z\rangle\,\phi ,
$$

hence (paper eq. `callen_correction`)

$$
\Delta\omega^{\mathrm{CD}}_{\mathbf q}
=A\Bigl(2\langle S^z\rangle-\frac{\langle S^z\rangle^2}{S^2}\Bigr)+\lambda_0\langle S^z\rangle
-\frac{\langle S^z\rangle^2}{S^2N_q}\sum_{\mathbf q'}\bigl(\lambda_{\mathbf q-\mathbf q'}+2A\bigr)n^{\mathrm B}_{\mathbf q'} .
$$

**Verified** (script §4): the CD A-coefficient above equals the paper's two A-terms after substituting
$\bar\psi=2\langle S^z\rangle\phi$ (symbolic), and reduces at $T=0$ ($\langle S^z\rangle\to S$, $\bar\psi\to0$)
to $A(2S-1)$, restoring both the HP gap and the $S=\tfrac12$ irrelevance — while retaining a (reduced,
$\propto\langle S^z\rangle^2$) finite-temperature SIA renormalization.

---

## 5. Multi-site generalization (paper Appendix `sec:multisite`)

### 5.1 Commutator

For $H=-\tfrac12\sum_{aibj}J^{ab}_{ij}\mathbf S_{ai}\!\cdot\!\mathbf S_{bj}-\tfrac12\sum_{aibj}\lambda^{ab}_{ij}S^z_{ai}S^z_{bj}
-A\sum_{ai}(S^z_{ai})^2$:

$$
[S^+_{ai},H]=\sum_{ck}J^{ac}_{ik}\bigl(S^+_{ai}S^z_{ck}-S^z_{ai}S^+_{ck}\bigr)
+\sum_{ck}\lambda^{ac}_{ik}S^+_{ai}S^z_{ck}
+A\bigl(S^z_{ai}S^+_{ai}+S^+_{ai}S^z_{ai}\bigr).
$$

> **Erratum 1 (found & verified, script §5):** the appendix prints the last term as
> $A(S^z_{ai}S^+_{ai}-S^+_{ai}S^z_{ai})=A[S^z,S^+]=A\,S^+$, which is wrong: the correct commutator of
> $-A(S^z)^2$ is $A(S^zS^++S^+S^z)$ (symmetrized, i.e. `standard_order`). The matrix check fails for the
> printed form and passes for the corrected one ($S=\tfrac12,1$).

### 5.2 RPA dynamical matrix (paper eq. `multisite-H_rpa`)

With the Tyablikov decoupling $\langle\!\langle S^z_{ck}S^+_{ai}\rangle\!\rangle\approx\langle S^z_c\rangle G_{ai,bj}$:

$$
\boxed{\;
H^{ab,\mathrm{RPA}}_{\mathbf q}=\delta_{ab}\sum_c\langle S^z_c\rangle\bigl(J^{ac}_0+\lambda^{ac}_0\bigr)
+\delta_{ab}\,2A\langle S^z_a\rangle-\langle S^z_a\rangle J^{ab}_{\mathbf q}\;}
$$

and $G^{+-}_{\mathbf q}(\omega)=2\langle S^z_b\rangle\sum_n U_{\mathbf q an}U^*_{\mathbf q bn}/(\omega-\omega_{\mathbf q n}+i\eta)$
(eq. `multi-site-Gfn`), $\omega_{\mathbf q n}=\mathrm{eig}H_{\mathbf q}^{\mathrm{RPA}}$.

**Verified** (script §6): Hermitian for all $\mathbf q$ whenever $J^{ab}(R),\lambda^{ab}(R)$ are real with the
pair symmetry $J^{ba}(-R)=J^{ab}(R)$ (random-coupling test); for $N_a=1$ reduces to
$\omega^{\mathrm{RPA}}_{\mathbf q}+\Delta\omega^{\mathrm{RPA}}_{\mathbf q}$ — symbolically on an $N_q=2$
mini-Brillouin-zone and numerically on random 16-point meshes.

### 5.3 CD matrix (paper eq. `multisite-H_cd`) — with corrections

$$
\boxed{\;
\begin{aligned}
H^{ab,\mathrm{CD}}_{\mathbf q}=&\;\delta_{ab}\,\langle S^z\rangle\sum_c\bigl(J^{ac}_0+\lambda^{ac}_0\bigr)
-\langle S^z\rangle J^{ab}_{\mathbf q}\\
&+\frac{\langle S^z\rangle}{2S^2N_q}\sum_{\mathbf q'}\Bigl\{\delta_{ab}\sum_c J^{ac}_{\mathbf q'}\psi^{ca}_{\mathbf q'}
-\bigl(\lambda^{ab}_{\mathbf q-\mathbf q'}+J^{ab}_{\mathbf q-\mathbf q'}\bigr)\psi^{ab}_{\mathbf q'}\Bigr\}\\
&+\delta_{ab}A\Bigl(2\langle S^z\rangle-\frac{\langle S^z\rangle}{S^2}\bigl(\langle S^z\rangle+\bar\psi_a\bigr)\Bigr),
\qquad \bar\psi_a\equiv\frac{1}{N_q}\sum_{\mathbf q'}\psi^{aa}_{\mathbf q'},
\end{aligned}}
$$

with the fluctuation-dissipation result (from the pole representation of $G^{+-}$ and
$\mathrm{Im}\,\frac{1}{\omega-\omega_n+i\eta}=-\pi\delta(\omega-\omega_n)$):

$$
\psi^{ab}_{\mathbf q}\equiv\langle S^-_{\mathbf q b}S^+_{\mathbf q a}\rangle
=2\langle S^z\rangle\sum_n U_{\mathbf q an}U^*_{\mathbf q bn}\,n^{\mathrm B}(\omega_{\mathbf q n}) ,
\qquad
\phi=\frac{1}{N_qN_a}\sum_{\mathbf q n}n^{\mathrm B}(\omega_{\mathbf q n}).
\qquad(\text{eq. }\textit{phi\_multiple})
$$

> **Errata 2–4 (found & verified, script §6a):** the printed `multisite-H_cd` cannot be correct as written:
>
> 1. **Sign:** the second term is printed $+\langle S^z\rangle J^{ab}_{\mathbf q}$; it must be *minus*, otherwise
>    the matrix cannot reduce to $\langle S^z\rangle(J_0-J_{\mathbf q})$ at $T=0$ (demonstrated symbolically).
> 2. **A-term:** the printed $\frac{\langle S^z\rangle}{S^2}\bigl(1+\bar\psi\bigr)$ is inconsistent with the
>    paper's own eq. `callen_correction`; it must read $\frac{\langle S^z\rangle}{S^2}\bigl(\langle S^z\rangle+\bar\psi\bigr)$.
>    The difference is $A\,m(m-1)/S^2\neq0$ for $m\neq1$ (and the printed form would give a T=0 gap
>    $A(2S-\tfrac1S)$ instead of $A(2S-1)$).
> 3. **Index:** in the last fluctuation term the printed $\psi^{ac}_{\mathbf q'}$ must carry the same pair
>    indices as the coupling, $\psi^{ab}_{\mathbf q'}$ (Hadamard/elementwise product); a c-contracted reading
>    breaks Hermiticity. The same index pattern appears in the HP matrix as $n^{ba}_{\mathbf q'}$, consistent
>    with $\psi^{ab}\leftrightarrow n^{ba}$.
>
> With these three fixes the matrix reduces *exactly* to $\omega^{\mathrm{CD}}_{\mathbf q}+\Delta\omega^{\mathrm{CD}}_{\mathbf q}$
> for $N_a=1$ — verified symbolically on the $N_q=2$ mini-BZ at $q=0,\pi$ and numerically (16-point meshes,
> random couplings). The HP multisite matrix (appendix, $H^{\mathrm{HP}}_{\mathbf q}$) is verified numerically to
> reduce to $\omega^{\mathrm{HP}}+\Delta\omega^{\mathrm{HP}}$ with $\langle S^z\rangle=S-\phi$.

### 5.4 Hermiticity

- $H^{\mathrm{RPA}}$ and (corrected) $H^{\mathrm{CD}}$ are Hermitian for all $\mathbf q$ when the couplings are real
  in $R$-space with $J^{ba}(-R)=J^{ab}(R)$, for **real $J_{\mathbf q}^{ab}$** (inversion-even bond sets — e.g. the
  honeycomb CrI$_3$ nearest-neighbor structure factor, which is real). Verified with random couplings and in a
  fully self-consistent finite-$T$ CD iteration on a 2-sublattice chain ($N_q=64$, Callen magnetization
  eq. `mag`, damped fixed point; converged, $\psi$ Hermitian, $\omega>0$, $0<\langle S^z\rangle<S$).
- For **complex $J^{ab}_{\mathbf q}$** (non-even bond sets) the finite-$\bar\psi$ CD matrix picks up an
  anti-Hermitian part of order $\mathrm{Im}\,J_{\mathbf q}\cdot\bar\psi$; at $\bar\psi=0$ ($T=0$) it is Hermitian
  regardless. Numerically demonstrated; remedy: hermitize $\tfrac12(H+H^\dagger)$ at finite $T$.
- $\phi$ consistency (eq. `phi_multiple`): $\sum_a n^{aa}_{\mathbf q}=\sum_n n^{\mathrm B}_n$ per $\mathbf q$
  (unitarity of $U$), and $\frac{1}{N_a}\sum_a\bar\psi_a=2\langle S^z\rangle\phi$ — both verified.

---

## 6. Bridge to TB2J `magnon3.py` conventions

TB2J (`TB2J/magnon/magnon3.py`, `Magnon.Jq/Hq/_diagonalize_magnon_hamiltonian`) never introduces $S^\pm$:
it works with **3×3 exchange tensors in spin-local frames** and a bosonic BdG matrix. The mapping below is
verified numerically in script §7 on a 2-sublattice model ($S=1.2$, inter-sublattice bonds $J^{12}(0)=1.7$,
$J^{12}(-1)=0.9$ plus anisotropic parts, intra-sublattice $\pm1$ bonds, SIA $A=0.21$, 21 k-points) with
**complex $J^{12}_{\mathbf q}$** (phase-convention stress test), and cross-validated against the *real*
`Magnon` class with the same stored tensors (max deviation $4\times10^{-15}$).

### 6.1 Storage and normalization conventions

| quantity | TB2J convention | paper equivalent |
|---|---|---|
| stored pair tensor | $\mathrm{JR}^{ab}_{\alpha\beta}(R)=\tfrac12 S_aS_b\bigl[J^{ab}(R)\delta_{\alpha\beta}+\lambda^{ab}(R)\hat z_\alpha\hat z_\beta\bigr]$ | $J^{ab},\lambda^{ab}$ as in eqs. `2siteJab`/`2SiteAE` |
| SIA tensor | added on-site: $\mathrm{JR}^{aa}_{zz}(0)\mathrel{+}=k1_a$, $k1_a=A_aS_a^2$ | $-A(S^z)^2$ |
| spin norms | `Snorm = |magmom|/2 = S`, division $1/(S_iS_j)$ in `Jq` | spins dimensionless, $m=2S\,\mu_B$ |
| Fourier phase | $\exp(-2\pi i\,\mathbf k\!\cdot\!\mathbf R)$, $\mathbf k$ fractional (cell vectors only) | $\exp(+i\mathbf q\!\cdot\!\mathbf R)$ |

Consequences (all asserted):

- $\tilde J^{ab}_{\mathrm{TB2J}}(\mathbf k)\equiv\sum_R \mathrm{JR}^{ab}(R)/(S_aS_b)\,e^{-2\pi i\mathbf k\mathbf R}
  =\tfrac12 J^{ab}_{\mathrm{paper}}(-\mathbf k)=\tfrac12\bigl[J^{ab}_{\mathrm{paper}}(\mathbf k)\bigr]^*$ for real-in-$R$ couplings.
- The half factor encodes TB2J's per-ordered-pair bookkeeping: each bond appears once as $(a,b,R)$ and once
  as $(b,a,-R)$, matching the paper's $\tfrac12\sum_{i\ne j}$.
- The paper's multi-site Fourier phase uses **cell vectors only** (sublattice offsets $\tau_b-\tau_a$ are not
  in the phase) — exactly like TB2J; including them would only gauge-transform the off-diagonal blocks.
- Anisotropy extraction from the normalized tensor:
  $2(\tilde J^{zz}-\tilde J^{xx})^{ab}_{\mathbf q}=\lambda^{ab}_{\mathbf q}+2A\,\delta_{ab}$,
  and $A=k1_a/S_a^2$.

### 6.2 BdG construction and identification with $H^{\mathrm{RPA},T=0}_{\mathbf q}$

For collinear-$z$ moments `get_rotation_arrays` returns $U=\hat x+i\hat y$, $V=\hat z$ per site, and

$$
A_1=U^\dagger\!\cdot\![-\tilde J(-\mathbf k)]_{\text{swapped}}\!\cdot U^{\!*},\quad
B=U^\dagger\!\cdot\![-\tilde J(-\mathbf k)]_{\text{swapped}}\!\cdot U,\quad
C=\mathrm{diag}\Bigl(\sum_l V^\dagger\,[2J_0]_{al}\,V\,S_l\Bigr),\quad
\mathcal H=\begin{pmatrix}A_1-C&B\\B^\dagger&A_2-C\end{pmatrix}.
$$

Multiplying by $\sqrt{S_aS_b}$ and evaluating the bilinear forms for the paper's diagonal tensor
($J^{xx}=J^{yy}=J_{\mathrm{iso}}$, $J^{zz}=J_{\mathrm{iso}}+\lambda$, SIA on-site):

$$
(A_1-C)^{ab}(\mathbf k)
\;=\;2S\,\delta_{ab}\sum_c \tilde J^{zz}_{0,ac}\;-\;S\,\bigl(\tilde J^{xx}+\tilde J^{yy}\bigr)^{ab}_{\mathbf k}
\;=\;S\,\delta_{ab}\sum_c\bigl(J^{ac}_0+\lambda^{ac}_0+2A_a\,\delta_{ac}\bigr)\;-\;S\,J^{ab}_{\mathbf q}\Big|_{\mathbf q=\mathbf k}
\;=\;H^{ab,\mathrm{RPA},T=0}_{\mathbf q}\Big|_{\mathbf q=\mathbf k},
$$

using $\tilde J^{zz}_0=\tfrac12(J_0+\lambda_0)+A\,\delta$ and $\tilde J^{xx}+\tilde J^{yy}=J^{\mathrm{paper}}$
(the $\tfrac12$ storage factor meets the leading 2 in $C=V^\dagger(2J_0)V$).

i.e. **the TB2J positive-mode block $A_1-C$ is exactly the paper's RPA dynamical matrix at $T=0$
($\langle S^z\rangle=S$), elementwise at the same k-label** (the site+tensor `swapaxes` in `Jmq` compensates
the Fourier sign difference between $e^{-2\pi i\mathbf k\mathbf R}$ and $e^{+i\mathbf q\mathbf R}$).
For the paper's zz-anisotropy model $B\equiv0$, so the bosonic eigenproblem reduces to the positive block;
TB2J's Cholesky routine $K^\dagger K=\mathcal H$, $\ \mathrm{eig}(K^\dagger gK)$, $g=\mathrm{diag}(\mathbb 1,-\mathbb 1)$
then returns exactly $\mathrm{eig}(H^{\mathrm{RPA},T=0}_{\mathbf q})$ — asserted over the whole k-mesh
(atol $10^{-10}$).

At $\Gamma$ the TB2J gap reproduces paper eq. `delta_rpa` (and eq. `optical`):

$$
\Delta^{\mathrm{TB2J}}=\Delta^{\mathrm{RPA}}=2AS+S\lambda_0,\qquad
\omega^{\mathrm{opt}}_{\Gamma}-\Delta=2S\,J^{12}_0 .
$$

**This is the key convention statement: TB2J implements the RPA / semi-classical SIA convention
($2AS$), *not* the HP/CD one ($A(2S-1)$).** Numerically: gap $=0.828=2AS+S\lambda_0$ vs the HP/CD value
$0.618=A(2S-1)+S\lambda_0$ for the test model. The difference $2AS-A(2S-1)=A$ is *independent of $S$*:
the two conventions agree only for $A=0$ (or, fractionally, in the large-$S$ limit where $A/2SJ\to0$).
For $S=\tfrac12$ the HP/CD gap from SIA vanishes identically while TB2J-style LSWT retains the spurious
gap $2AS=A$.

Other verified bridge points:

- `A2` is the Hermitian partner block (`A2 = conj(A1)` structure for real couplings); $B=0$ for zz-only
  anisotropy; $B\neq0$ generically (DMI, canted/non-collinear states) is where TB2J's full BdG machinery
  goes beyond the paper's collinear-$z$ formalism.
- Unequal spins: TB2J's $\sqrt{S_aS_b}$ ansatz generalizes the paper's equivalent-spin appendix formulas;
  the bridge above is exact for $S_a=S_b=S$.
- TB2J is a $T=0$ (LSWT) theory: thermal renormalization ($\langle S^z\rangle$, $\psi$, Bose sums) and
  $T_c$ extraction are outside `magnon3.py` and require the RPA/CD/HP self-consistency of parts 01/02.

---

## 7. Verified checks (script output, 68 PASS lines)

| # | paper eq | check |
|---|---|---|
| 1–12 | `spin-comm` | $[S^z,S^\pm]=\pm S^\pm$, $[S^+,S^-]=2S^z$ and the three `standard_order`/`rpa_correct_ordering` identities for $S=\tfrac12,1,\tfrac32$ (matrices) |
| 13–14 | `standard_order` | same identities from abstract noncommutative algebra |
| 15–17 | `sia-reorder` | $a^\dagger a\,a^\dagger a=a^{\dagger2}a^2+a^\dagger a$: normal-ordering proof, Fock algebra $n^2=n(n-1)+n$, $6\times6$ matrices |
| 18–22 | `hp_correction`, `delta_cd` | exact one-magnon ED (2-site cluster, generic $S$ symbolic + $S=\tfrac12,1,\tfrac32,2$ matrix ED) $=A(2S-1)+S\lambda_0\;(+2SJ)$ |
| 23–26 | `rpa_correction`, `callen_correction`, `delta_rpa`, `delta_cd` | T=0 gaps $2AS+S\lambda_0$ (RPA) and $A(2S-1)+S\lambda_0$ (HP/CD); CD A-coefficient ↔ paper A-terms with $\bar\psi=2\langle S^z\rangle\phi$ |
| 27–28 | §II.G | $S=\tfrac12$: HP/CD SIA gap vanishes; RPA spurious gap $=A$ |
| 29–31 | appendix commutator | corrected multisite $[S^+_{ai},H]$ (matrix, $S=\tfrac12,1$); printed $A(S^zS^+-S^+S^z)$ form fails (Erratum 1) |
| 32–35 | `multisite-H_rpa`, `multisite-H_cd` | $N_a=1$ reductions to $\omega^{\mathrm{RPA}}+\Delta\omega^{\mathrm{RPA}}$ and $\omega^{\mathrm{CD}}+\Delta\omega^{\mathrm{CD}}$ (mini-BZ symbolic, $q=0,\pi$) |
| 36–37 | `multisite-H_cd` | Errata 2–3: printed $+mJ_{\mathbf q}$ sign and $(1+\bar\psi)$ A-term inconsistent with single-site eqs |
| 38–39 | appendix $H^{\mathrm{HP}}$ | numeric $N_a=1$ reduction to $\omega^{\mathrm{HP}}+\Delta\omega^{\mathrm{HP}}$ (random couplings, all q) |
| 40–44 | `multisite-H_rpa`/`-H_cd` | Hermiticity (real couplings); $H^{\mathrm{CD}}(\bar\psi{=}0,m{=}S)$ = LSWT matrix with gap $A(2S-1)$ |
| 45–48 | `multisite-H_cd` | complex-$J_{\mathbf q}$ caveat: Hermitian at $\bar\psi=0$, $O(\mathrm{Im}J\cdot\bar\psi)$ anti-Hermitian part at finite $T$, hermitization remedy |
| 49–50 | `phi_multiple`, FDT | $\sum_an^{aa}=\sum_nn^{\mathrm B}$ (unitarity); $\frac1{N_a}\sum_a\bar\psi_a=2\langle S^z\rangle\phi$ |
| 51–58 | `multisite-H_cd`, `mag`, `phi_multiple` | fully self-consistent CD on a 2-sublattice chain: convergence, Hermiticity of $H$ and $\psi$, $\omega>0$, $0<\langle S^z\rangle<S$, $T\to0$ limit = LSWT matrix |
| 59–67 | `2siteJab`, `2SiteAE`, `multisite-H_rpa`, `delta_rpa`, `optical` | TB2J bridge: phase/½-normalization convention, $\lambda_{\mathbf q}+2A\delta=2(\tilde J^{zz}-\tilde J^{xx})$, $A_1-C\equiv H^{\mathrm{RPA},T=0}_{\mathbf q}$ elementwise, $B=0$, BdG eigenvalue equality (atol $10^{-10}$), $\Gamma$ gap $2AS+S\lambda_0$, optical splitting $2SJ^{12}_0$, RPA-like (not HP/CD-like) SIA convention |
| 68 | — | even-bond model: real $J_{\mathbf q}$, elementwise $A_1-C=H^{\mathrm{paper}}(\mathbf k)$ |

Out-of-band cross-validation (not part of the script, run against the installed package): the real
`TB2J.magnon.magnon3.Magnon` class, given the stored tensors of §6.1, reproduces
$\mathrm{eig}(H^{\mathrm{RPA},T=0}_{\mathbf q})$ to $4\times10^{-15}$, confirming that the minimal
reimplementation and the storage conventions above describe the production code.

---

## 8. Summary of findings for TB2J

1. `magnon3.py` at $T=0$ *is* the paper's multi-site RPA/semi-classical matrix, including its SIA convention
   $\Delta=2AS+S\lambda_0$; equivalently TB2J answers "semi-classical spin-wave theory (relevant for classical
   MC) or RPA" in paper eq. `delta_rpa`. Users comparing to HP/CD gaps must apply the $2AS\to A(2S-1)$
   correction themselves — relevant for 2D magnets and low-spin systems.
2. The paper's appendix contains four typographical defects (commutator A-term sign, $+mJ_{\mathbf q}$ sign,
   $(1+\bar\psi)$ vs $(\langle S^z\rangle+\bar\psi)$, index $\psi^{ac}$ vs $\psi^{ab}$); with the corrections
   derived here, RPA/CD/HP multi-site matrices all reduce exactly to their single-site counterparts and are
   Hermitian for real couplings with inversion-even bond sets.
3. The multi-site CD machinery (corrected) runs self-consistently and stably at finite $T$; for complex
   $J^{ab}_{\mathbf q}$ hermitize the dynamical matrix.
