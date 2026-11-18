# Spin-Spiral Generalized Bloch Theorem — Sympy Derivation Report

Script: `spin_spiral_generalized_bloch.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-24).
Serves the spin-spiral MFT spec (`Projects/TB2J/specs/spin-spiral-mft`).

## Setup and conventions

- Spinor layout is interleaved, index `2*(cell*L + mu) + s`, matching
  `TB2J/pauli.py` and TBUpy `2*orb+spin`.
- Flat spiral about $+\hat z$ with sublattice positions $\tau_\mu$ (in
  lattice units) and sublattice phases $\phi_\mu$:

$$
\theta_{a\mu} = 2\pi\,\mathbf q\cdot(a + \tau_\mu) + \phi_\mu,
\qquad
R_z(\theta) = e^{-i\theta\sigma_z/2}.
$$

- Reference Hamiltonian: collinear, spin-diagonal — hopping
  $T_{\mu\nu}(\mathbf R)$ spin independent, on-site exchange field
  $B_\mu\sigma_z$.
- Spiral Hamiltonian $H_s = U H_0 U^\dagger$ with
  $U = \bigoplus_{a\mu} R_z(\theta_{a\mu})$. Because the rotation axis
  is $z$:

$$
U^\dagger\,(B_\mu\sigma_z)\,U = B_\mu\sigma_z
\quad\text{(on-site field invariant)},
$$

$$
\langle a\mu\sigma|H_s|b\nu\sigma'\rangle
= T_{\mu\nu}\,\delta_{\sigma\sigma'}\,
e^{-i\sigma(\theta_{b\nu}-\theta_{a\mu})/2}.
$$

  All spiral physics sits in the **bond phases**; a spin rotation about
  $z$ cannot rotate a $z$-axis field. (For non-$z$ spiral axes, rotate
  the whole problem first; SOC breaks this freedom and is out of scope
  for v1 — see the spec.)

## Generalized Bloch theorem (central result)

The ring (supercell) spectrum equals the union over primitive
k-points $k_n = n/N$ of the folded (twisted) Hamiltonian

$$
H_q(k)_{\mu s,\nu s'}
= \sum_{\mathbf R} e^{2\pi i k R}\,T_{\mu\nu}(\mathbf R)\,
e^{+i s\alpha_\mu/2}\,e^{-i s'\alpha_\nu(\mathbf R)/2},
$$

$$
\alpha_\mu = 2\pi q\,\tau_\mu + \phi_\mu,
\qquad
\alpha_\nu(\mathbf R) = 2\pi q\,(\mathbf R + \tau_\nu) + \phi_\nu .
$$

The spin-conserving blocks are therefore

$$
H^{\uparrow\uparrow}_{\mu\nu}(q,k)
= \sum_R e^{2\pi i (k - q/2)R}\,T_{\mu\nu}(R)\,
e^{-i[\alpha_\nu(R)-\alpha_\mu]/2},
$$

and the $\downarrow\downarrow$ block with $q \to -q$
($k \to k + q/2$). Verified:

1. symbolically (characteristic polynomials identical as polynomials
   in symbolic $q$, and for $N=2,L=2$ additionally in symbolic
   $\tau_B,\phi_B$),
2. numerically for random multi-sublattice Hermitian models
   ($N=5$, $L=2$, 3 seeds $\times$ 3 values of $q$, machine precision).

## TBUpy convention check

For one orbital per cell ($\tau = \phi = 0$) the folded Hamiltonian is
exactly

$$
H_q(k) = \begin{pmatrix} H_0(k - q/2) & 0 \\ 0 & H_0(k + q/2) \end{pmatrix},
$$

i.e. the `PHASE_CONVENTION` of
`tbupy.generalized_bloch.SpinSpiralConfig` ("up=H0(k-q/2),
down=H0(k+q/2); phase=2*pi*dot(q_frac,R)") is the $L=1$ special case of
the theorem. The existing TBUpy provider is therefore **single
sublattice**; multi-sublattice spirals need the $\alpha_\mu$ factors
above plus the exchange field written in the twisted gauge
$B_\mu(\cos\alpha_\mu\,\sigma_z + \sin\alpha_\mu\,\sigma_x)$ when the
sublattice phases $\phi_\mu \neq 0$ are used to build canting (e.g.
$\Delta\phi = \pi$ AFM references).

## Hermiticity, time reversal, force theorem, nonorthogonality

- **Hermiticity** of $H_s$ and $H_q(k)$ follows from the physical class
  pairing $t(\nu,\mu,-\mathbf R) = t(\mu,\nu,\mathbf R)^*$ (asserted).
- **$q \to -q$**: spectra of $H_s(q)$ and $H_s(-q)$ coincide for
  TR-invariant (real-hopping) references — the two chiralities are
  related by the antiunitary $\Theta$ — so the force-theorem $E(q)$ is
  even in $q$, as a Heisenberg map requires. For complex-hopping
  (TR-broken) references the two chiralities are genuinely inequivalent
  (checked numerically).
- **Force theorem**: at a common reference Fermi level with a smooth
  occupation function, the occupied band energy on the folded
  $\{H_q(k_n)\}$ mesh equals the supercell value (asserted to
  $2\times10^{-14}$).
- **Nonorthogonal basis**: with the rotated overlap
  $\tilde S_{ab} = U_a^\dagger S_{ab} U_b$ (the FPLO convention,
  Koepernik–Eschrig PRB 59, 1743), the generalized pencil
  $(H_q(k_n), S_q(k_n))$ has exactly the supercell pencil spectrum
  (asserted to $10^{-9}$).

## Implementation notes pinned for the spec

- Bond classes must be keyed by **signed** $\mathbf R$; the ring
  Hamiltonian sums every class at its residue $R \bmod N$. A pair
  dictionary has no $R=0$ shell for $i\neq j$ pairs.
- Hermiticity comes from class pairing, never from an ad-hoc closure
  pass (a closure pass double-counts classes when $2R \equiv 0$).
- The reuse seams are `TB2J.mathutils.rotate_spin.rotate_spinor_matrix`
  (same $U^\dagger M U$ passive form), `magnon_math.get_rotation_arrays`
  (per-site local frames), `spinham.supercell` (`exp(2 pi i q . sc_vec)`
  phase helper), and `MAEGreen` (fixed-efermi band-energy machinery).
