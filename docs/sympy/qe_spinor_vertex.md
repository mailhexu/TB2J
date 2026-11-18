# QE KB spinor magnetic vertex: exact convention checks

Run `qe_spinor_vertex.py` with the `mydev` environment. Its exact
assertions passed on 2026-10-10. They pin source conventions,
nine-response rotation/finite difference and the scalar collinear
limit; they do **not** replace native QE `fcoef`, `add_vuspsi`
or three-reference exchange validation.

## Native spin packing and sign

`PW/src/vloc_psi_acc.f90:331-340` applies the real-space local potential
as

$$
H_{\mathrm{loc}}=v_1 I+v_2\sigma_x+v_3\sigma_y+v_4\sigma_z,
$$

with offdiagonal upper/lower elements $v_2\mp i v_3$.
`PW/src/newd_acc.f90:529-547` assembles the non-SO one-center blocks
in the order **(uu, ud, du, dd)**. `PW/src/add_vuspsi.f90:208-213`
independently confirms that this is the order *applied* to a spinor:

$$
D_{\mathrm{nc}}=
\begin{pmatrix}
  d_{\mathrm{van}}+d_1+d_4 & d_2-i d_3\\
  d_2+i d_3 & d_{\mathrm{van}}+d_1-d_4
\end{pmatrix}.
$$

The script checks both matrices exactly and their Hermiticity.

## SOC must not be the exchange rotation vertex

The total native `deeq_nc` includes fixed ionic/SO operator contributions.
For a hypothetical *same-source* zero-field assembly at the same density,
UPFs, basis and projector gauge, the magnetic difference is

$$
D_{\mathrm{mag}}=D_{\mathrm{total}}[\mathbf B]
     -D_{\mathrm{total}}[\mathbf B=0].
$$

An exact unitary spin-orbital CNOT is used as a **negative control**:
conjugating $\sigma_x^{\mathrm{orb}}\otimes\sigma_x^{\mathrm{spin}}$
can yield $\sigma_x^{\mathrm{orb}}\otimes I^{\mathrm{spin}}$.
Therefore it is incorrect to discard the spin trace of an arbitrary
field-dependent block *before* checking the native field action. Taking
the traceless Pauli component of `deeq_nc_total` is worse: it retains
lattice-fixed SOC even for $\mathbf B=0$. For SO-bearing QE
`fcoef`, the existing `magnetic_tangent_vertices` commutator
is **not** the native source derivative: bare-spin covariance
fails on real Fe US/PAW. NC1.1 uses the explicit source response.
The CNOT is a counterexample, not a model of QE's
Clebsch--Gordan coefficients.

The rotation gate uses $U_x=(I-i\sigma_x)/\sqrt{2}$ and
$U_y=(I-i\sigma_y)/\sqrt{2}$. Exact SymPy assertions establish
$U_x\sigma_zU_x^\dagger=-\sigma_y$ and
$U_x\sigma_yU_x^\dagger=\sigma_z$, so the native source field rotated
about $x$ is $(B_x,-B_z,B_y)$, **not** $(B_x,B_z,-B_y)$.
Likewise $U_y$ maps $(B_x,B_y,B_z)$ to $(B_z,B_y,-B_x)$.
The signs were checked against a failing real Fe scalar-relativistic
dump's second rotation leg; after correction, a real 4³ full-BZ
US spinor dump passed native rotation, action and reader checks.

The PAW one-center correction is a **second magnetic source**.
`PW/src/add_paw_to_deeq_acc.f90:31-49` adds `ddd_paw(ijh,na,1:4)`
to the same four components as `newq(v)`; `ddd_paw` is produced by
`PAW_potential(rho%bec,...)` at the frozen density
(`PW/src/non_scf.f90:153-155`). Zeroing only smooth components
`v_2,v_3,v_4` while keeping all four `ddd_paw` components cancels
the physical PAW magnetic Hxc term from $D_{\rm mag}$. The zero
source must also zero `ddd_paw(:,na,2:4)`; rotated-source comparisons
must rotate those components with the same SU(2) vector law.
The script exactly asserts
$D_{\rm mag}=(B_z+D^{1}_{z,\mathrm{PAW}})\sigma_z$ and its
collinear splitting $2(B_z+D^{1}_{z,\mathrm{PAW}})$ while the scalar
PAW charge part and fixed SOC cancel. Before this correction, a real
12³ Fe PAW SOC-off z transverse leg gave 15.9644623 meV versus
same-UPF collinear 14.7600550 meV even though its Fermi energies
differed by only $1.17\times10^{-6}$ Ry and first bands by
$1.6\times10^{-6}$ Ry. That 1.2044073 meV gap was a failed G3,
not a density difference to scale away. After correcting the
source separation, the 12³ PAW z-leg yields
14.7589663679 meV against 14.7600549888 meV collinear
(difference −0.0010886209 meV); the NC1.1 rerun reproduces
the PAW scalar leg to reported precision. The NC1.1 US leg
gives 17.6997098 versus 17.6957657 meV collinear
(difference +0.0039441 meV). Both satisfy the 0.01-meV
same-pseudopotential SOC-off G3 bound.

## Why a source tangent is required for SOC-on projector families

At a fixed density and projector gauge, write the native field-dependent
mapping as a linear operator $\mathcal L[\mathbf B]$. For a reference
direction $\hat{\mathbf n}$ and a Cartesian displacement $\hat{\mathbf t}$,
rotating **only** the magnetic source with generator
$\hat{\mathbf n}\times\hat{\mathbf t}$ gives the exact infinitesimal
vertex $\mathcal L[(\hat{\mathbf n}\times\hat{\mathbf t})
\times\mathbf B]$; fixed ionic/SOC terms have zero derivative. The
commutator of the mapped magnetic splitting instead assumes
$\mathcal L[R\mathbf B]=U\mathcal L[\mathbf B]U^\dagger$ for every rotation.
That equality does not follow from linearity.

The script verifies an abstract counterexample
$\mathcal L[B_z\hat z]=\lambda\sigma_z$ and
$\mathcal L[B_x\hat x]=2\lambda\sigma_x$. An exact derivative of
$\mathcal L[\sin\theta\,\hat x+\cos\theta\,\hat z]$ at $\theta=0$
is $2\lambda\sigma_x$, whereas the existing splitting-commutator
tangent is $\lambda\sigma_x$; a symmetric finite difference at
$\lambda=7/10$ checks the symbolic result numerically. The
Pauli-covariant limit, $\mathcal L[\hat x]=\lambda\sigma_x$,
reduces to the existing tangent. This counterexample is not a QE
`fcoef` derivation.

The actual QE fixed-density Fe fully relativistic US and PAW
`TB2J_COV_LEG 1` tests measured relative bare-spin
covariance residuals 0.33148 and 0.27662. NC1.0
therefore rejects SO-bearing dumps. NC1.1 instead exports
nine native magnetic-source blocks with fixed ionic/SOC terms
unchanged: on both real Fe US and PAW, diagonal reconstruction
is within ~`6e-15` relative, quarter-turn/tilted native
recombination within ~`1.5e-14`, action parity within
~`2.6e-14`, and central-difference relative error
`1.6667e-7` at `1e-3` rad. The trace and three-independent-leg
rank-nine merger are unchanged; an independent-reference
repeated-row failure remains a separate stop condition.

## Exact source-response reconstruction

Let $B_j(\mathbf r)$ be the original frozen smooth magnetic field in
source channel $j$ (and include the corresponding frozen PAW magnetic
$D^1_j$). Define the native nonlocal response
$D_{c\leftarrow j}$ as the same-gauge zero-source-subtracted
assembly when the **spatial profile of source $j$** is placed
in output Pauli channel $c$. There are nine $(c,j)$ blocks:
the diagonal three alone cannot construct a $z\to x$ tangent when
the reference has $B_z\ne0$ but $B_x=0$.
For $\hat{\mathbf n}$ and $\hat{\mathbf t}_a$, set

$$
K^{(a)}_{cj}=\sum_d\epsilon_{cdj}
       (\hat{\mathbf n}\times\hat{\mathbf t}_a)_d,
\qquad
D_{\rm mag}(\theta)=
 \sum_{c,j} [\exp(\theta K^{(a)})]_{cj}D_{c\leftarrow j}.
$$

Frozen-source linearity then gives the physical TB2J tangent:

$$
V_a^{(t_a)} =
 \left.\frac{\partial H_{{\rm mag},a}(\theta)}
 {\partial\theta}\right|_{\theta=0}
=\sum_{c,j}K^{(a)}_{cj}
 \left[
  (M_a^{+}A_{a,j}M_a^{+})\otimes\sigma_c
  +D_{a,c\leftarrow j}
 \right],
\quad
\Delta_a=2H_{{\rm mag},a}.
$$

$D$ is already in the dual beta channel and is **not**
Gram-dressed; only the smooth field $A_{a,j}$ is. Apply Ry-to-eV
conversion once to $V$, with no extra factor of two:
the existing commutator tangent is $\tfrac12\,\partial_\theta\Delta$
when the source is Pauli-covariant. The SymPy script asserts the
nine-block rotation identity at an exact rational tilted direction
$(3/5,0,4/5)$ and checks a symmetric finite difference numerically
(step $10^{-4}$, maximum error below $10^{-8}$). For a scalar $z$
source, $D_{c\leftarrow z}=B_z\sigma_c$ recovers the existing
commutator transverse $V_x=B_z\sigma_x$ exactly. This is a
representation identity, not native QE G2 parity. In the
implementation, the conditioned Hermitian beta Gram uses
`rcond=1e-6` for the smooth projection alone: fully relativistic
Fe Gram condition ~`1.1e10` would otherwise amplify the
smooth term to ~`1.5e8 Ry`. The scalar Gram (~58) is
untruncated. Neither already-dual Green coefficients nor
native nonlocal source blocks are conditioned.

## Full-complex smooth field and scalar collinear anchor

A real field need not give real offdiagonal beta-channel elements. The
script uses $\beta=(1,i)$ with a real field of strength two to assert

$$
\beta^\dagger (2)\beta=
\begin{pmatrix}2&2i\\-2i&2\end{pmatrix},
$$

which is Hermitian and loses valid data if only its real part is exported.
For a two-channel complex-Hermitian beta Gram $M$, smooth projected
$A_z=\langle\beta|B_z|\beta\rangle$ and augmentation $D_z$, the
orbital-first spinor splitting is

$$
\Delta_z=2(M^{-1}A_zM^{-1}+D_z)\otimes\sigma_z.
$$

The exact SymPy assertions give
$\Delta_{uu}=M^{-1}(2A_z)M^{-1}+2D_z$,
$\Delta_{dd}=-\Delta_{uu}$, and zero offdiagonal spin blocks.
Since QE's collinear `dbeta_xc=2A_z` and
`deeq_up-deeq_down=2D_z`, **the uu block**, not `uu-dd`, is the
validated v1.2 collinear exchange vertex. The existing tangent law
on the z reference, with $\hat z\times\hat x=\hat y$, returns
$(M^{-1}A_zM^{-1}+D_z)\otimes\sigma_x$. Reversing both the
splitting and inferred reference direction leaves this transverse
vertex unchanged; the assertion pins the sign-even convention used
by `magnetic_tangent_vertices`.

The verification is algebraic; exchange readiness remains
an empirical range/family property. NC1.1 real US x/y/z
first-neighbor exchange passes the repeated-row gate
(`8.367697e-9 eV` versus `5e-8 eV`), while its six-angstrom
far shells fail (`2.586256e-5 eV`). PAW SO-bearing native
source checks pass; the original `conv_thr=1e-8 Ry` SCF
family fails the first shell (`4.117504e-6 eV`),
but rerunning all three PAW SCFs to `1e-12 Ry` and NSCF
exports passes first-neighbor rank-nine merging
(`2.086799e-8 eV`, `Jiso=14.7775026 meV`).
Its six-angstrom far shell still fails (`1.558e-5 eV`
at the first inconsistent pair). No extended US/PAW
tensor follows from these exact identities.

## NC smooth-only source, retained subspace, and the antisymmetric counterexample

Story 002 adds three exact checks (`check_nc_smooth_only_source_and_tangent`,
`check_gram_retained_subspace_reconstruction`,
`check_normalized_antisymmetric_counterexample`).

**NC smooth-only vertex (ADR-002).** With Gram $M$, projected smooth
field $B_z$ and *measured-zero* native responses
$D_{c\leftarrow j}=0$, the operator
$\Delta=2E\,M^{-1}B_zM^{-1}\otimes\sigma_z$ and the explicit
source-resolved tangent $V_x=E\,M^{-1}B_zM^{-1}\otimes\sigma_x$ satisfy
the commutator law exactly; the longitudinal vertex vanishes; a nonzero
$D_{c\leftarrow j}$ breaks the commutator equality (the SO-bearing
case). Conjugating the source by $U_y$ ($\sigma_z\to\sigma_x$)
conjugates the operator and rotates the tangent legs exactly — the
covariance the reader asserts numerically on `has_so=F` species.

**Retained-subspace reconstruction.** With the rcond-truncated
transform $M_\tau^+$ the dual identity degrades to the retained
projector, $M_\tau^+M=P$: a field supported on the retained subspace
reconstructs exactly, while any component coupling into a discarded
mode is annihilated instead of dressed. A conditioned pseudoinverse
reconstructs only the principal retained subspace.

**Normalized antisymmetric counterexample.** For
$M=\begin{pmatrix}1&s\\ s&1\end{pmatrix}$ (eigenvalues $1+s$ retained,
$1-s$ discarded), $u_-=(1,-1)^T/\sqrt2$, constant field $A=M$, and
primal coefficients $x=u_-/\sqrt{1-s}$, the **physical** state
$|\psi\rangle=\beta x$ has norm one:
$x^\dagger Mx=1$, although $\|x\|^2=1/(1-s)$ diverges as $s\to1$.
The dressed contraction $\langle Mx|\,M^+AM^+\,|Mx\rangle$ equals
$1$ with the full inverse and $0$ when the antisymmetric mode
is discarded, for every $0<s<1$ satisfying that truncation
condition; numerical check at $s=0.99999999$ and
$rcond=10^{-6}$ agrees. The prior universal $O(1-s)$ error
bound silently assumed bounded, unnormalized primal coefficients
and is disproved; it is not re-pinned. $(j,m_j)$ spinor projectors
are not a square transform of QE's native channels.

These are algebraic identities only. A clean Gram eigengap is a
numerical-conditioning diagnostic: retained modes as small as
$\sim5\times10^{-6}$ (measured PseudoDojo FR NC Fe, first k) amplify
the $M^+AM^+$ dressing without bound, and discarded modes already
carry $O(1)$ normalized occupied-projector weight. Production
therefore refuses SO-bearing NC **and** NC with truncated Gram
modes after QE clears `has_so` on an SOC-off FR UPF; a
window-resolved spectral-weight and exchange-threshold-sensitivity
certificate would be needed to lift that refusal.
