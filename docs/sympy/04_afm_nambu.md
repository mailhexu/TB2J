# 04 — Local-frame Nambu formulation for a collinear bipartite antiferromagnet

Companion executable derivation: [`04_afm_nambu.py`](04_afm_nambu.py). It uses
SymPy assertions and a finite numerical two-sublattice reduction. Run:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
python 04_afm_nambu.py
```

The script must print `ALL ... CHECKS PASSED`. This is the AFM theory gate from
[ADR 0004](../adr/0004-afm-derivation-and-thermal-results.md): an AFM is not a
scalar FM thermal equation with its exchange sign reversed.

## 1. Conventions and local frames

TB2J's pair-once convention is

$$
H=-\sum_{\langle ij\rangle}J_{ij}\,\mathbf S_i\cdot\mathbf S_j.
$$

Thus an AFM bond has $J_{AB}=-K$, $K>0$. All energies ($J$, $K$, $\omega$, and
$k_B T$) are eV. Wave vectors are fractional reciprocal coordinates, so a
translation $R$ has phase $\exp(2\pi i\,q\cdot R)$.

For a collinear bipartite state, keep A unchanged and rotate B by $\pi$ around
local $x$:

$$
\mathbf S_B^{\rm global}=(\widetilde S_B^x,-\widetilde S_B^y,-\widetilde S_B^z).
$$

Both ordered moments therefore point along local $+z$, and the script asserts

$$
\mathbf S_A\cdot\mathbf S_B=
\frac{\widetilde S_A^+\widetilde S_B^++\widetilde S_A^-\widetilde S_B^-}{2}
-\widetilde S_A^z\widetilde S_B^z.
\tag{1}
$$

The rotated transverse coupling creates/removes pairs; it is not FM magnon
hopping.

## 2. HP Hamiltonian and bosonic BdG structure

In the aligned frames, $\widetilde S^+=\sqrt{2S}a$,
$\widetilde S^-=\sqrt{2S}a^\dagger$, and
$\widetilde S^z=S-a^\dagger a$. One AFM bond is

$$
H_{AB}=-KS^2+KS(n_a+n_b+ab+a^\dagger b^\dagger)-K n_a n_b.
\tag{2}
$$

LSWT retains the first two terms. For equivalent sublattices,

$$
H_2=\sum_q\left[A_q(a_q^\dagger a_q+b_q^\dagger b_q)
+B_q(a_qb_{-q}+a_q^\dagger b_{-q}^\dagger)\right]+E_{\rm cl}.
\tag{3}
$$

A sublattice gauge may make $B_q$ real. Otherwise use the complex-conjugate
pair in the Nambu matrix. For
$\Psi_q=(a_q,b_q,a_{-q}^\dagger,b_{-q}^\dagger)^T$,

$$
H_{\rm BdG}(q)=
\begin{pmatrix}A_q&0&0&B_q\\0&A_q&B_q&0\\0&B_q&A_q&0\\B_q&0&0&A_q\end{pmatrix},
\qquad D_q=\Sigma_3H_{\rm BdG}(q),\quad
\Sigma_3=\operatorname{diag}(1,1,-1,-1).
\tag{4}
$$

The physical bosonic problem is the non-Hermitian dynamical matrix $D_q$, not
ordinary diagonalization of $H_{\rm BdG}$. The symbolic assertions prove

$$
D_q^2=(A_q^2-|B_q|^2)I,\qquad
\omega_q=\sqrt{A_q^2-|B_q|^2}>0\quad(A_q>|B_q|).
\tag{5}
$$

A future solver must retain stable positive-frequency, positive-$\Sigma_3$-norm
modes and reject $A_q^2<|B_q|^2$.

## 3. Normal/anomalous contractions and staggered order

With $u_q^2=(A_q/\omega_q+1)/2$, $v_q^2=(A_q/\omega_q-1)/2$, and
$u_qv_q=B_q/(2\omega_q)$, the required local-frame contractions are

$$
\langle a_q^\dagger a_q\rangle=
\frac12\left[\frac{A_q}{\omega_q}(2n_q^B+1)-1\right],\qquad
\langle a_qb_{-q}\rangle=-\frac{B_q}{2\omega_q}(2n_q^B+1).
\tag{6}
$$

The anomalous contraction is required AFM state, not an optional correction.
At zero temperature,

$$
m_A^{\rm loc}=m_B^{\rm loc}=S-\frac1{N_q}\sum_q v_q^2.
\tag{7}
$$

The global B moment has the opposite sign. Thus the physical staggered order is
$(m_A^{\rm global}-m_B^{\rm global})/2=m^{\rm loc}$, rather than uniform
global magnetization. An exact Goldstone zero in a finite isotropic system makes
the broken-symmetry boson vacuum singular; the solver needs a thermodynamic
mesh/integral treatment or an explicitly documented regulator.

## 4. RPA transition linearization

For isotropic Tyablikov/RPA, the Nambu form remains but

$$
A_q=mK_0,\qquad B_q=mK_q,\qquad
\omega_q=m\epsilon_q,\qquad
\epsilon_q=\sqrt{K_0^2-|K_q|^2}.
\tag{8}
$$

The critical kernel follows from the local normal contraction (6), not from
the Bose occupation alone. Because $A_q/\omega_q=K_0/\epsilon_q$ is
$m$-independent, the contraction
$n_q=\frac12\left[\frac{K_0}{\epsilon_q}(2n_q^B+1)-1\right]$ with
$2n_q^B+1=\coth(m\epsilon_q/2k_BT)$ obeys, as $m\to0^+$,

$$
n_q=\frac{k_BT}{m}\,\frac{K_0}{\epsilon_q^2}-\frac12+O(m).
\tag{9}
$$

The divergent part of the site occupation is therefore weighted by
$K_0/\epsilon_q^2$: the amplification of the occupation by the Nambu
Bogoliubov factor $A_q/\omega_q$ is what makes the AFM kernel second order
in $1/\epsilon_q$. With Callen's on-site relation
$m=S(S+1)/(3\Phi)+O(\Phi^{-2})$ and
$\Phi=(k_BT/m)F+O(m^0)$, cancellation of $m$ gives

$$
k_BT_N=\frac{S(S+1)}{3F},\qquad
F=\frac1{N_q}\sum_q{}'\,\frac{K_0}{\epsilon_q^2}.
\tag{10}
$$

The prime excludes exact Goldstone modes ($\epsilon_q=0$, e.g. $\Gamma$ on
any finite mesh of the isotropic model): that exclusion is the finite-mesh
regulator of the weighted kernel. A scalar $\epsilon_q^{-1}$ weight must not
be substituted: it belongs to the ferromagnetic normal spectrum, its 2D mean
$\int d^2q\,/\epsilon_q\sim\int q\,dq/q$ converges, and it would yield a
spurious finite $T_N$ violating Mermin–Wagner.

With the weighted kernel, isotropic 1D gives $T_N=0$ through a power-law
divergence $\int_\delta dq\,K_0/\epsilon_q^2\sim1/(8\pi^2K\delta)$ at the
Goldstone point (asserted with the exact prefactor), and the 2D mesh sum
$F_N\sim\ln N$ grows without bound, so $T_N\to0$ as the mesh refines. In 3D
$F$ is finite and $T_N>0$; the regression tests pin a converging 3D value
against the analytic mesh kernel.

### Production RPA scope and scaling

The production AFM RPA closure supplies the Callen order relation with the
local normal contraction (6), including its $T=0$ $v_q^2$ depletion, so its
finite-temperature bands use the same $K_0/\epsilon_q^2$ linearization as
(10). The classical regime follows the FM prescription exactly: solve the
quantum equations at $S_{\rm eff}=KS$ and $T_{\rm eff}=K^2T$, then report
order and energies divided by $K$.

Equation (8) is a restricted equivalent two-site reduction, not a generic
collinear AFM formula. TB2J therefore accepts only one magnetic site per
sublattice and requires both normal BdG blocks to be the q-independent scalar
$K_0I$ (verified over the lattice Fourier support). Same-sublattice transverse
exchange, multisite anomalous covariances, or inequivalent normal weights
would require q- and eigenvector-resolved site covariance weights, so they are
rejected rather than incorrectly using a Gamma-point $K_0$ for every q. A
singular positive-semidefinite BdG block is treated as a marginal Goldstone
mode; an indefinite block or a complex metric spectrum is rejected as
dynamically unstable.

## 5. Finite two-sublattice reduction and implementation requirements

The executable reduces a nearest-neighbour isotropic chain with 16 magnetic
cells, $K=0.040$ eV, and $S=3/2$. Its mesh is
$q=m/(2N_{\rm cell})$; the doubled magnetic cell identifies $q$ and $q+1/2$:

$$
A=2KS,\qquad B_q=2KS\cos(2\pi q),\qquad
\omega_q=2KS|\sin(2\pi q)|.
\tag{11}
$$

It verifies the full four-dimensional Nambu eigenspectrum at every mesh point
and verifies $\omega_{q=0}=0$. Therefore a TB2J AFM thermal solver must:

1. construct exchange in aligned local sublattice frames;
2. retain both normal and anomalous correlators;
3. diagonalize $\Sigma_3H_{\rm BdG}$ and select stable positive-norm branches;
4. evolve sublattice order and report global staggered order;
5. enforce Goldstone/Mermin--Wagner zero-transition constraints; and
6. preserve eV and fractional-$q$ conventions.

Anisotropic exchange and SIA modify the Nambu coefficients, but do not remove
these requirements.

## 6. Quartic AFM closures required by HP, CD, and RPA+CD

For an AFM bond with local-frame normal and anomalous contractions
\(n=\langle a^\dagger a\rangle=\langle b^\dagger b\rangle\) and
\(\kappa=\langle ab\rangle=\langle a^\dagger b^\dagger\rangle\), HFB
decoupling through the same \(1/S\) order as the transverse HP
square-root yields
\[
V_4^{\rm HFB}=-K(n+\kappa)(n_a+n_b+ab+a^\dagger b^\dagger)
+K(n+\kappa)^2.
\]
Consequently the normal and anomalous bond kernels both carry
\(K[S-n-\kappa]\). A production HP closure must retain this paired
renormalization.

The local-frame Callen closure requires independent normal and anomalous
contractions: \(d=zK[m-\alpha P]\), \(e=zK[m-\alpha C]\), with
\(\alpha=m/(2S^2)\). The AFM RPA+CD SIA contribution is local:
\(\Delta_{\rm RPA}=2Am\) and
\(\Delta_{\rm CD}=A[2m-m(m+\psi)/S^2]\). These equations define the
required Nambu kernels; their numerical implementation still requires
an exact two-sublattice regression test.
