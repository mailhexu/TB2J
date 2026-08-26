# 01 — Heisenberg model, Holstein–Primakoff transformation, and linear spin-wave theory

Companion script: [`01_heisenberg_hp_lswt.py`](01_heisenberg_hp_lswt.py) — **77 assertion checks, all PASS**

This note is the authoritative derivation reference for the $T=0$ input of the TB2J thermal-magnon
formalism: the isotropic Heisenberg Hamiltonian, its exact two-site verification anchor, the
Holstein–Primakoff (HP) boson representation, the quadratic (linear) spin-wave Hamiltonian of a Bravais
ferromagnet, and the Bogoliubov diagonalization of the collinear two-sublattice antiferromagnet that is
needed later for the Néel temperature (part 02).

Source for equation labels: **arXiv:2405.00477** (LaTeX source `TB2J/Refs/2405.00477/main.tex`,
Sec. 2 "Theory"). Numbers below refer to that preprint.

---

## Assumptions

1. Isotropic nearest-neighbour-general Heisenberg model on a Bravais lattice with pairwise exchange
   $J_{ij} = J_{ji}$, $J_{ii}=0$; $J>0$ **ferromagnetic** (paper convention). The collinear AFM is
   obtained by $J \to -J_{\rm af}<0$, i.e. $H = +J_{\rm af}\sum_{\langle i j\rangle}\mathbf S_i\cdot\mathbf S_j$.
2. Atomic units, $\hbar=1$; spins dimensionless (paper: "commutation relations in atomic units").
3. Ferromagnetic ground state: fully polarized along the $+z$ quantization axis (paper, Sec. 2).
4. HP bosons: one mode per site, $[a_i,a_j^\dagger]=\delta_{ij}$; magnon–magnon interactions (quartic
   and higher boson terms) are **dropped** in LSWT; their Hartree–Fock resummation at $T>0$ is the
   subject of part 02.
5. Periodic boundary conditions on small clusters (2-site dimer, 4-site ring, 8-site two-sublattice
   chain) used as exact verification anchors; the derivations themselves keep $J_{ij}$ symbolic.
6. Operator identities are verified *exactly* on matrix representations: spin matrices for
   $S=\tfrac12,1,\tfrac32$, and truncated boson Fock spaces with $n\le 2S+2$ (one unphysical boson
   number beyond $n=2S$ included to delimit the physical block).

## Notation

| Symbol | Meaning |
|---|---|
| $\mathbf S_i$, $S_i^{x,y,z}$ | spin operator on site $i$ (dimensionless, $\hbar=1$) |
| $S_i^\pm = S_i^x \pm i S_i^y$ | circular spin components (paper Eq. (2)) |
| $J_{ij}$, $J_0=\sum_{j\ne 0}J_{0j}$ | exchange constants; on-site sum |
| $J_{\mathbf q}=\sum_{\mathbf R}J_{0\mathbf R}\,e^{i\mathbf q\cdot\mathbf R}$ | Fourier transform, paper Eq. (11) |
| $a_i$, $a_i^\dagger$, $n_i=a_i^\dagger a_i$ | HP bosons, number operator |
| $a_{\mathbf q}=N^{-1/2}\sum_i a_i\,e^{-i\mathbf q\cdot\mathbf R_i}$ | momentum bosons, paper Eq. (10) |
| $N$ ($N_q$) | number of sites (of $\mathbf q$ points) |
| $a,b$ ($\alpha,\beta$) | sublattice bosons (Bogoliubov quasiparticles) for the AFM |
| $A_k,B_k$ | Bogoliubov coefficients: number term, anomalous (pairing) term |
| $\langle S^z\rangle$, $\phi$ | magnetization and Bose-sum $\phi=\frac{1}{N_q}\sum_{\mathbf q}n^{\rm B}_{\mathbf q}$ |
| $E_{\rm cl}$ | classical (fully polarized) energy |

---

## 1. Spin algebra (checks 1.1–1.5)

Exact matrices in the $|m\rangle$ basis ($m=S,S\!-\!1,\dots,-S$):

$$
S^z|m\rangle = m|m\rangle,\qquad
S^\pm|m\rangle = \sqrt{S(S+1)-m(m\pm1)}\;|m\pm 1\rangle .
$$

Verified for $S=\tfrac12,\,1,\,\tfrac32$ by explicit sympy matrices (checks **1.1–1.5**):

$$
[S^z,S^\pm]=\pm S^\pm,\qquad [S^+,S^-]=2S^z,\qquad
\mathbf S\cdot\mathbf S = S(S+1)\,\mathbb 1 .
$$

The last two coincide with paper Eqs. (4a) [spin-comm-1] and (4b) [spin-comm-2] for $i=j$
(the $\delta_{ij}$ is trivial on one site).

## 2. Heisenberg Hamiltonian and the two-site anchor (checks 2.1–2.4)

Paper Eq. (1) [heisenbrghamiltonian]:

$$
H=-\frac12\sum_{i\ne j}J_{ij}\,\mathbf S_i\cdot\mathbf S_j .
$$

With $S_i^\pm=S_i^x\pm iS_i^y$ (paper Eq. (2)) this becomes the circular-coordinate form, paper
Eq. (3) [eq:hamiltonian in circular coordinates]:

$$
H=-\frac12\sum_{i\ne j}J_{ij}\Big(\tfrac12\big[S_i^+S_j^-+S_i^-S_j^+\big]+S_i^zS_j^z\Big).
$$

**Two-site cluster** ($J_{12}=J_{21}=J$, so $H=-J\,\mathbf S_1\cdot\mathbf S_2$):

* Check **2.1**: the circular form equals the Cartesian form $-J(S_1^xS_2^x+S_1^yS_2^y+S_1^zS_2^z)$
  as $(2S+1)^2\times(2S+1)^2$ matrices for $S=\tfrac12,1$.
* Check **2.2**: paper Eq. (4c) [spin-comm-H], the commutator with the Hamiltonian used later in
  Green's-function EOMs, holds as a matrix identity:
  $$[S_i^\pm,H]=\pm\sum_{j\ne i}J_{ij}\big(S_i^\pm S_j^z-S_i^zS_j^\pm\big).$$
* Check **2.3** — exact diagonalization vs Clebsch–Gordan. Using
  $\mathbf S_1\cdot\mathbf S_2=\tfrac12\big[\mathbf S_{\rm tot}^2-2S(S+1)\big]$,
  $$E(S_{\rm tot})=-\frac{J}{2}\Big(S_{\rm tot}(S_{\rm tot}+1)-2S(S+1)\Big),\qquad
  S_{\rm tot}=0,\dots,2S .$$
  Exact eigenvalues (with multiplicities $2S_{\rm tot}+1$): $S=\tfrac12$: $\{-\tfrac J4^{\,(3)},\,+\tfrac{3J}{4}^{\,(1)}\}$;
  $S=1$: $\{-J^{(5)},+J^{(3)},+2J^{(1)}\}$ — reproduced by `eigenvals()`.
* Check **2.4** — one-magnon sector. The fully polarized state has
  $$E_0=-JS^2,$$ and the $S^z_{\rm tot}=2S-1$ sector has exact energies
  $$\omega\in\{0,\;2JS\}$$ relative to $E_0$ (symmetric = $q{=}0$ magnon at zero cost,
  antisymmetric = $q{=}\pi/a$ magnon). These are the anchor values for LSWT below.

## 3. Holstein–Primakoff transformation (checks 3.1–3.7)

Paper Eqs. (5a–c) [hp_spin±z], with bosons $[a_i,a_j^\dagger]=\delta_{ij}$:

$$
S_i^+=\sqrt{2S-n_i}\;a_i,\qquad
S_i^-=a_i^\dagger\sqrt{2S-n_i},\qquad
S_i^z=S-n_i .
$$

*Verification strategy* (exactly as required): represent $a,a^\dagger$ on a Fock space truncated at
$n\le 2S+2$ (dimension $2S+3$; the square root becomes imaginary beyond $n=2S$, which delimits the
physical block $n\le 2S$). On the physical block, for $S=\tfrac12$ and $S=1$:

* **3.2**: $[a,a^\dagger]=1$ on the $n\le 2S+1$ block.
* **3.3–3.4**: $[S^z,S^\pm]=\pm S^\pm$ and $[S^+,S^-]=2S^z$ **exactly** — the HP representation
  preserves the spin algebra on the physical subspace.
* **3.5**: $\mathbf S\cdot\mathbf S=S(S+1)\mathbb 1$.
* **3.6**: the HP matrix block coincides *entry by entry* with the exact spin-$S$ matrices of Sec. 1
  (the HP representation is faithful: $S^+|n\rangle=\sqrt{n(2S-n+1)}\,|n-1\rangle$ reproduces
  $S^+|m\rangle=\sqrt{S(S+1)-m(m+1)}\,|m+1\rangle$ with $m=S-n$).
* **3.7**: bosons on different sites commute.
* **3.1**: square-root expansion used for the quadratic Hamiltonian,
  $$\sqrt{2S-n}=\sqrt{2S}\Big(1-\frac{n}{4S}\Big)+\mathcal O(n^2)$$
  (valid as an operator series because it is a function of $n$ alone, placed left of $a_i$ exactly as
  in Eqs. (5a,b)).

## 4. Quadratic HP Hamiltonian and $\omega_{\mathbf q}=S(J_0-J_{\mathbf q})$ (checks 4.1–4.5)

### 4.1 Expansion of a bond (checks 4.1a–c)

Insert Eqs. (5a–c) into Eq. (3) and keep terms with at most two boson operators (verified
symbolically with noncommutative sympy symbols, dropping $\mathcal O(n^2)$ under the square root):

$$
S_i^zS_j^z=(S-n_i)(S-n_j)\;\xrightarrow{\text{quadratic}}\;S^2-S(n_i+n_j),
$$

$$
\tfrac12\big(S_i^+S_j^-+S_i^-S_j^+\big)\;\xrightarrow{\text{quadratic}}\;
S\big(a_i^\dagger a_j+a_j^\dagger a_i\big),
$$

where $a_i a_j^\dagger=a_j^\dagger a_i$ for $i\ne j$ (check 4.1c). Hence

$$
\boxed{\;H_2=-\frac{S^2}{2}\sum_{i\ne j}J_{ij}
+SJ_0\sum_i a_i^\dagger a_i
-S\sum_{i\ne j}J_{ij}\,a_i^\dagger a_j\;}
$$

with $E_{\rm cl}=-\tfrac{N}{2}J_0S^2$ the classical energy and $J_0=\sum_{j\ne0}J_{0j}$.
Check **4.2a** verifies $E_{\rm cl}$ on the 4-site ring (where the antipodal distance-2 neighbour is
counted once, $J_0=2J_1+J_2$ for couplings $J_1,J_2$).

### 4.2 Fourier diagonalization (checks 4.2b–c)

With paper Eq. (10) [hp-q-transform] and its unitary inverse
$a_i=N^{-1/2}\sum_{\mathbf q}e^{i\mathbf q\cdot\mathbf R_i}a_{\mathbf q}$, all cross terms
$a_{\mathbf q}^\dagger a_{\mathbf q'}$ ($\mathbf q\ne\mathbf q'$) cancel and

$$
\boxed{\;H_2=E_{\rm cl}+\sum_{\mathbf q}\underbrace{S\,(J_0-J_{\mathbf q})}_{\displaystyle\omega_{\mathbf q}}\;
a_{\mathbf q}^\dagger a_{\mathbf q}\;}
$$

with $J_{\mathbf q}$ from paper Eq. (11) [Jij-q-space]. This is verified *exactly and symbolically*
(check 4.2c) on the $N=4$ ring with symbolic $J_1,J_2$: every monomial of the transformed $H_2$ is
normal-ordered and the coefficient of $a^\dagger_{\mathbf q}a_{\mathbf q}$ equals
$S(J_0-J_{\mathbf q})$, with $J_{\mathbf q}=J_1(e^{iq}+e^{-iq})+J_2e^{2iq}$ on that ring.

### 4.3 Relation to the paper's finite-$T$ dispersion (checks 4.3a–b)

The paper's thermally renormalized magnon energy, Eq. (12) [HP_magnon_energy], is

$$
\omega_{\mathbf q}^{\rm HP}
=\langle S^z\rangle\,(J_0-J_{\mathbf q})
+\frac{1}{N_q}\sum_{\mathbf q'}\big(J_{\mathbf q'}-J_{\mathbf q-\mathbf q'}\big)\,n^{\rm B}_{\mathbf q'},
\qquad \langle S^z\rangle=S-\phi,\quad
\phi=\frac1{N_q}\sum_{\mathbf q}n^{\rm B}_{\mathbf q}
$$

(Eqs. (13) [HP_mag], (14) [phi]). Checks:

* **4.3a**: at $T=0$, $n^{\rm B}_{\mathbf q'}=0\Rightarrow\phi=0\Rightarrow\langle S^z\rangle=S$, and
  Eq. (12) reduces exactly to the LSWT result $\omega_{\mathbf q}=S(J_0-J_{\mathbf q})$
  (verified for every $\mathbf q$ of the 4-site ring with symbolic $J_1,J_2$).
* **4.3b**: for a *uniform* Bose occupation $n^{\rm B}_{\mathbf q'}=\phi$ the interaction term
  vanishes identically, $\frac1N\sum_{\mathbf q'}(J_{\mathbf q'}-J_{\mathbf q-\mathbf q'})=0$ —
  because both sums equal $J_{\mathbf R=0}=0$ ($J_{ii}=0$). Uniformly distributed magnons soften the
  dispersion only through the magnetization prefactor, $\omega_{\mathbf q}=(S-\phi)(J_0-J_{\mathbf q})$.

### 4.4 Bosonic dimer vs the exact one-magnon sector (checks 4.4a–b)

For the dimer, $H_2=E_{\rm cl}+A(n_1+n_2)+B(a_1^\dagger a_2+a_2^\dagger a_1)$ with $A=SJ$, $B=-SJ$.

* **4.4a**: the $2\times2$ single-particle matrix has eigenvalues $\{0,\,2JS\}$ — **identical to the
  exact one-magnon sector of check 2.4**. (The one-magnon sector of the Heisenberg ferromagnet is
  exactly harmonic; HP interactions first matter at two magnons.)
* **4.4b**: diagonalizing the full bosonic $H_2$ numerically on a truncated two-mode Fock space
  reproduces $E_{\rm cl}+m\cdot 2JS$ on the trunculation-safe subspace (total boson number
  $N_{\rm tot}\le D-1$) to $10^{-15}$.

### 4.5 Exact one-magnon diagonalization of the 4-site ring (checks 4.5a–b)

For $J_1=1$, $J_2=0.3$, $S=\tfrac12$ and $S=1$: the exact Heisenberg Hamiltonian on the 4-site ring,
projected onto the $S^z_{\rm tot}=NS-1$ sector, has eigenvalues (relative to $E_0=-\tfrac N2J_0S^2$,
check 4.5a)

$$
\{S(J_0-J_{\mathbf q})\}_{\mathbf q=0,\frac\pi2,\pi,\frac{3\pi}2}
=\{0,\;2S(J_1{+}J_2),\;4SJ_1,\;2S(J_1{+}J_2)\},
$$

in **exact agreement** with LSWT $\omega_{\mathbf q}=S(J_0-J_{\mathbf q})$ (check 4.5b) — for any $S$,
not only in the large-$S$ limit, as expected for the exactly-solvable one-magnon sector.

## 5. Two-sublattice antiferromagnet: Bogoliubov diagonalization (checks 5.0–5.8)

This provides the AFM (Néel) input used in part 02. Take the collinear Néel state
($A$ sublattice $\uparrow$, $B$ sublattice $\downarrow$) of $H=+J_{\rm af}\sum_{\langle i j\rangle}
\mathbf S_i\cdot\mathbf S_j$, $J_{\rm af}>0$, and quantize the $B$ spins in a rotated frame
$\tilde{\mathbf S}$: $\tilde S^z=-S^z$, $\tilde S^\pm=S^\mp$ (algebra preserved, check **5.0**).
HP bosons $a_i$ ($A$) and $b_j$ ($B$) then give, per bond, to quadratic order:

$$
S_i^zS_j^z=-(S-n_i)(S-n_j)\;\xrightarrow{\text{quadratic}}\;-S^2+S(n_i+n_j),
\qquad
\tfrac12\big(S_i^+S_j^-+S_i^-S_j^+\big)\xrightarrow{\text{quadratic}}S\big(a_ib_j+a_i^\dagger b_j^\dagger\big).
$$

With each site in $z$ bonds:

$$
H_2=E_{\rm cl}+A\sum_i\big(n^a_i+n^b_i\big)+J_{\rm af}S\sum_{\langle i\in A,j\in B\rangle}
\big(a_ib_j+a_i^\dagger b_j^\dagger\big),\qquad
E_{\rm cl}=-N_{\rm bonds}J_{\rm af}S^2,\quad A=zJ_{\rm af}S .
$$

### 5.1–5.2 Equations of motion and the BdG spectrum (checks 5.1–5.2)

Elementary boson commutators are verified on two-mode Fock matrices (check **5.1**); with them, the
symbolic commutators of the single-$k$ block $H_k=A(a^\dagger a+b^\dagger b)+B_k(a^\dagger b^\dagger+ab)$
are (check **5.2a**)

$$
[H_k,a]=-Aa-B_kb^\dagger,\qquad [H_k,b^\dagger]=Ab^\dagger+B_ka
\;\;\Longrightarrow\;\;
i\frac{d}{dt}\begin{pmatrix}a\\ b^\dagger\end{pmatrix}
=\begin{pmatrix}A&B_k\\-B_k&-A\end{pmatrix}\begin{pmatrix}a\\ b^\dagger\end{pmatrix}.
$$

The Bogoliubov–de Gennes matrix squares to $(A^2-B_k^2)\mathbb 1$ with zero trace (Cayley–Hamilton,
check **5.2b**), so the magnon energy is

$$
\boxed{\;\omega_k=\sqrt{A^2-B_k^2}\;}
$$

### 5.3 Bogoliubov angle (checks 5.3a–c)

The transformation $\alpha=u a-v b^\dagger$, $\beta=u b-v a^\dagger$, $u=\cosh\theta$, $v=\sinh\theta$
diagonalizes $H_k$ when $\tanh 2\theta=B_k/A$, i.e. $\tanh\theta=(A-\omega_k)/B_k$; the double-angle
identity $\tanh2\theta=2t/(1+t^2)$ is verified symbolically and with exact rationals
($A=5,B_k=3\Rightarrow\omega=4$, $\tanh\theta=\tfrac13$, $\tanh2\theta=\tfrac35$). The diagonal form
of one self-conjugate block is

$$
H_k=\omega_k\big(\alpha^\dagger\alpha+\beta^\dagger\beta\big)+(\omega_k-A).
$$

### 5.4 AFM dimer ground state (check 5.4)

For $z=1$: $E_{\rm cl}=-J_{\rm af}S^2$, $A=B_0=J_{\rm af}S$, so $\omega=0$ and
$E_{\rm GS}^{\rm LSWT}=E_{\rm cl}+(\omega-A)=-J_{\rm af}S(S+1)$ — **exactly** the singlet energy for
all $S$. (Excitations are *not* exact: see 5.8.)

### 5.5–5.6 The 1D chain: $\omega_k=2J_{\rm af}S|\sin ka|$ (checks 5.5–5.6)

For the nearest-neighbour chain (8-site ring, 4 two-site cells, $a=1$), Fourier transforming with
sublattice positions included ($A$ at $2n$, $B$ at $2n+1$; cell reciprocal period $\pi$) gives,
verified coefficient by coefficient with symbolic $J_{\rm af},S$ (checks **5.5a–b**):

$$
H_2=E_{\rm cl}+\sum_k\Big[A\big(a_k^\dagger a_k+b_k^\dagger b_k\big)
+B_k\big(a_k b_{-k}+a_k^\dagger b_{-k}^\dagger\big)\Big],
\qquad A=2J_{\rm af}S,\quad B_k=2J_{\rm af}S\cos ka .
$$

(The pairing coefficient carries the momentum label of the $b$ mode, $k_{-m}$; $\cos$ flips sign
between zone-folded representatives while $|\sin|$ does not — the dispersion is unaffected.)
Hence (checks **5.6**, one per $k$ of the cell BZ):

$$
\boxed{\;\omega_k=\sqrt{(2J_{\rm af}S)^2-(2J_{\rm af}S\cos ka)^2}
=2J_{\rm af}S\,|\sin ka|\;}
$$

— the textbook LSWT dispersion of the 1D Heisenberg antiferromagnet. Since $|\sin ka|$ is invariant
under $k\to k+\pi/a$, the two zone-folded branches agree and the formula holds over the full atomic
Brillouin zone; $\omega_{k=0}=0$ is the Goldstone mode.

### 5.7 Numerical Fock-space diagonalization of the $k$ blocks (checks 5.7a–c)

The truncated two-mode block $A(n_a+n_b)+B(ab+a^\dagger b^\dagger)$ is diagonalized numerically:

* **5.7a**: generic block $A=5,B=3$ ($\omega=4$): levels $(\omega-A)+m\omega$ reproduced to
  $6\times10^{-12}$ ($D=20$ per mode).
* **5.7b**: the three non-singular chain blocks ($J_{\rm af}=1,S=\tfrac12$: $\omega=|\sin k|\in
  \{\tfrac{\sqrt2}{2},1,\tfrac{\sqrt2}{2}\}$) match $2J_{\rm af}S|\sin ka|$ to $2\times10^{-12}$.
* **5.7c**: the $k=0$ (Goldstone, $\omega=0$) block is *marginal*: $t=(A-\omega)/B=1$, and
  $\lambda_{\min}$ approaches the formal offset $\omega-A=-A$ only as $\sim 1/D$; the monotone
  convergence with that scaling is asserted ($-0.830\to-0.956$ for $D=8\to32$).

### 5.8 Exact AFM dimer spectrum — documented caveat (check 5.8)

Exact $S=\tfrac12$ AFM dimer: singlet $-\tfrac34$, triplet $+\tfrac14$ (gap 1). LSWT reproduces the
ground-state energy (5.4) but predicts a spurious $\omega=0$ instead of the gap — the well-known
breakdown of harmonic spin-wave theory in zero dimensions; the reliable anchor for the AFM dispersion
is the chain result of 5.6, which becomes exact for $S\to\infty$ (and in 3D up to $1/S$ corrections).

---

## Verified checks (77 total, all PASS)
| IDs | Content |
|---|---|
| 1.1–1.5 | spin commutators, Casimir, Hermiticity, spectrum for $S=\tfrac12,1,\tfrac32$ |
| 2.1 | circular $\equiv$ Cartesian two-site Hamiltonian |
| 2.2 | $[S_i^\pm,H]$ = paper Eq. (4c) |
| 2.3 | exact dimer spectrum $=$ Clebsch–Gordan $E(S_{\rm tot})$ |
| 2.4 | $E_0=-JS^2$; exact one-magnon energies $\{0,2JS\}$ |
| 3.1 | $\sqrt{2S-n}$ series to $\mathcal O(n)$ |
| 3.2–3.6 | HP algebra exact on the $n\le2S$ block ($S=\tfrac12,1$); HP $=$ exact spin matrices |
| 3.7 | $[a_i,a_j^\dagger]=0$, $i\ne j$ |
| 4.1a–c | quadratic parts of $S_i^zS_j^z$ and flip-flop term; cross-site normal ordering |
| 4.2a–c | $E_{\rm cl}$; Fourier transform gives exactly $H_2=E_{\rm cl}+\sum_{\mathbf q}S(J_0-J_{\mathbf q})a^\dagger_{\mathbf q}a_{\mathbf q}$ |
| 4.3a–b | Eq. (12) $\to$ LSWT at $T=0$; uniform-$n^{\rm B}$ interaction term vanishes |
| 4.4a–b | bosonic dimer one-magnon $\{0,2JS\}$ and full $E_{\rm cl}+m\,2JS$ spectrum |
| 4.5a–b | exact one-magnon diagonalization of the 4-site ring $=$ $S(J_0-J_{\mathbf q})$ ($S=\tfrac12,1$) |
| 5.0 | rotated sublattice frame preserves the spin algebra |
| 5.1 | elementary two-mode boson commutators (matrix rep) |
| 5.2a–b | EOM/BdG matrix; $\omega_k=\sqrt{A^2-B_k^2}$ |
| 5.3a–c | Bogoliubov angle identities (symbolic + exact rational) |
| 5.4 | AFM dimer LSWT ground state $=-J_{\rm af}S(S+1)$ (exact) |
| 5.5a–b | $A_k=2J_{\rm af}S$, $B_k=2J_{\rm af}S\cos ka$; all cross terms vanish |
| 5.6 | $\omega_k=2J_{\rm af}S|\sin ka|$ for every $k$ of the cell BZ |
| 5.7a–c | numeric Fock diagonalization: generic block, chain blocks, Goldstone-block convergence |
| 5.8 | exact AFM dimer spectrum (LSWT caveat documented) |

## Mapping to paper equation labels (arXiv:2405.00477)

| Paper equation (label) | Statement | Verified in |
|---|---|---|
| (1) `heisenbrghamiltonian` | $H=-\tfrac12\sum_{i\ne j}J_{ij}\mathbf S_i\cdot\mathbf S_j$ | Secs. 2, 4 (checks 2.1, 4.5) |
| (2) | $S_i^\pm=S_i^x\pm iS_i^y$ | Sec. 1 (checks 1.1–1.4) |
| (3) `eq:hamiltonian in circular coordinates` | circular form of $H$ | check 2.1 |
| (4a) `spin-comm-1` | $[S_i^z,S_j^\pm]=\pm S_i^\pm\delta_{ij}$ | checks 1.1, 3.3 |
| (4b) `spin-comm-2` | $[S_i^+,S_j^-]=2S_i^z\delta_{ij}$ | checks 1.2, 3.4 |
| (4c) `spin-comm-H` | $[S_i^\pm,H]=\pm\sum_{j\ne i}J_{ij}(S_i^\pm S_j^z-S_i^zS_j^\pm)$ | check 2.2 |
| (5a–c) `hp_spin+`,`hp_spin-`,`hp_spinz` | HP transformation | Sec. 3 (checks 3.2–3.6) |
| (10) `hp-q-transform` | $a_{\mathbf q}=N^{-1/2}\sum_i a_i e^{-i\mathbf q\mathbf R_i}$ | Sec. 4.2 (check 4.2c) |
| (11) `Jij-q-space` | $J_{\mathbf q}=\sum_{\mathbf R}J_{0\mathbf R}e^{i\mathbf q\mathbf R}$ | checks 4.2c, 4.5b |
| (12) `HP_magnon_energy` | $\omega^{\rm HP}_{\mathbf q}=\langle S^z\rangle(J_0-J_{\mathbf q})+\frac1{N_q}\sum_{\mathbf q'}(J_{\mathbf q'}-J_{\mathbf q-\mathbf q'})n^{\rm B}_{\mathbf q'}$ | checks 4.3a–b ($T=0$ reduction $\to$ LSWT) |
| (13) `HP_mag`, (14) `phi` | $\langle S^z\rangle=S-\phi$, $\phi=\frac1{N_q}\sum n^{\rm B}$ | check 4.3a (finite-$T$ self-consistency: part 02) |

## Caveats and scope

* LSWT is exact for the **one-magnon sector** of any Heisenberg ferromagnet (checks 2.4/4.4a/4.5b);
  magnon–magnon interactions enter at two and more magnons and matter at finite $T$ (paper Sec. 2.B,
  part 02 of this series).
* HP matrix identities hold on the physical block $n\le 2S$ only; the truncated Fock representation
  beyond $2S$ is unphysical (imaginary square roots) and is used only to delimit the block.
* Fock-space diagonalizations of hopping-type blocks are exact on the total-number $\le D-1$
  subspace; squeezed (AFM pairing) blocks converge as $t^{2D}$ with $t=(A-\omega)/B$, except the
  marginal Goldstone block ($t=1$, $\sim1/D$) — see checks 4.4b, 5.7a–c.
* The AFM result is derived for a collinear two-sublattice order on a bipartite lattice; multi-site
  spirals, anisotropy, and the TB2J `magnon3.py` conventions are treated in part 03.
