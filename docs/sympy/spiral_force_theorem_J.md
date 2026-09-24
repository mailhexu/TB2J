# Spiral Force-Theorem Energy Mapping to Heisenberg Exchange — Sympy Report

Script: `spiral_force_theorem_J.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-24).
Serves the spin-spiral MFT spec (`Projects/TB2J/specs/spin-spiral-mft`).

All conventions are TB2J conventions: `SpinIO` exchange
$J_{ij}(\mathbf R) > 0$ = ferromagnetic, Heisenberg
$E = -\sum_{ij} J_{ij}\,\mathbf e_i\cdot\mathbf e_j$ (each ordered pair
counted once), magnon Fourier convention
$J(\mathbf q) = \sum_{\mathbf R\neq 0} J(\mathbf R)e^{-2\pi i\mathbf q\cdot\mathbf R}$
(sign and $|\mathbf q|\le$ half-BZ as in `exchange_qspace.q_to_r` and
`magnon3.Jq`), spiral spin angle
$\theta_{\mathbf R} = 2\pi\mathbf q\cdot\mathbf R$ right-handed about
$+\hat z$.

## 1. Classical flat-spiral energy

$$
E(\mathbf q) - E(0) \;=\; J(\mathbf 0) - \mathrm{Re}\,J(\mathbf q)
\;\ge\; 0 \quad\text{for FM-positive } J,
$$

where **$J(\mathbf 0)$ means $J(\mathbf q{=}0)=\sum_{\mathbf R\neq0}J(\mathbf R)$**,
not an $R=0$ shell — a pair dictionary has no onsite term. Symbolic
shells $\{\pm1,\pm2\}$ verified; $\mathrm{Im}\,J(\mathbf q)=0$ from the
$\pm\mathbf R$ pairing.

## 2. Exact energy mapping (inversion)

On the grid $q_n = n/N$:

$$
J(\mathbf R) = \frac{1}{N}\sum_{n} \big[E(0) - E(q_n)\big]\,e^{+2\pi i q_n R},
$$

with shell multiplicities: the $\pm\mathbf R$ pair enters once, and the
self-paired $R = N/2$ shell (even $N$) enters once. Verified
symbolically against a known $J$ set.

## 3. LSWT bridge (magnon consistency)

With the same $J$ and spin $S$:

$$
\omega(\mathbf q) = 2S\big[J(\mathbf 0) - J(\mathbf q)\big],
\qquad
E(\mathbf q) - E(0) = S^2\big[J(\mathbf 0) - J(\mathbf q)\big]
= \frac{S}{2}\,\omega(\mathbf q),
$$

Goldstone $\omega(\mathbf 0) = 0$, and the small-$q$ stiffness
$\frac{d^2E}{dq^2}\big|_0 = (2\pi)^2\sum_R J(R)R^2 > 0$. This pins the
factor conventions against the "moment $M$" conventions of e.g.
Daglum PRB **113**, 214401 (2026), where $M = 2S$ in $\mu_B$ units and
their $J$ differs by a factor 2 from the TB2J double-counted `Jdict`.

## 4. Multi-sublattice helical energy and bipartite AFM

$$
\frac{E}{N} = -\sum_{\mu\nu\mathbf R} S_\mu S_\nu J_{\mu\nu}(\mathbf R)\,
\cos\!\big(2\pi\mathbf q\cdot\mathbf R + \alpha_\nu - \alpha_\mu\big),
\qquad
\alpha_\mu = 2\pi\mathbf q\cdot\tau_\mu + \phi_\mu .
$$

NN bipartite chain (cell $= 2$ sites, $\tau_B - \tau_A = 1/2$):

$$
e(q) = -4S^2J\cos(\pi q)\cos(\Delta\phi),
$$

minimized at $|\Delta\phi| = \pi$ (AFM), periodic under
$q \to q + 2/\delta$ (magnetic-cell folding). Bipartite LSWT with the
same $J$:

$$
\omega(k) = 2S\sqrt{J_{AA}J_{BB} - |J_{AB}(k)|^2} = 4SJ|\sin(\pi k)|,
$$

with zero modes at the chemical $\Gamma$ and at the zone corner $R$ —
the chemical-$\Gamma$ / magnetic-$\Gamma$ folding of the
incommensurate-magnon notes.

## 5. Force-theorem band energy = second-order exchange kernel

Two-site model, local fields $\pm h$ rotated by $\pm\theta/2$,
$H(\theta) = H_0 + \theta V_1 + \theta^2 V_2 + O(\theta^3)$:

$$
V_1 = \frac{h}{2}\big(\sigma_x^A + \sigma_x^B\big),
\qquad
V_2 = -\frac{h}{8}\sigma_z^A + \frac{h}{8}\sigma_z^B .
$$

Fixed-occupation band-sum curvature (the one-shot MFT $E(q)$ per pair,
protocol P-b) equals the second-order resolvent kernel — the $O(\theta^2)$
form of the Liechtenstein (LKAG) contour trace:

$$
\frac{d^2E_{\text{band}}}{d\theta^2}\bigg|_0
= 2\,\mathrm{Tr}[V_2\rho_0]
+ 2\sum_{n\neq m} f_n\,\frac{|\langle n|V_1|m\rangle|^2}{\varepsilon_n-\varepsilon_m},
$$

verified numerically to $10^{-7}$ relative. This is the bridge that
makes spiral-MFT $J$ and TB2J's Green-function LKAG $J$ the **same
object at second order** (full proof: Szilva et al., Rev. Mod. Phys.
**95**, 035004 (2023), Sec. V; state-sum form: Skovhus et al., PRB 2025).

Pinned caveats (all demonstrated in-script):

- The force theorem is $O(\theta^3)$-accurate: fixed-potential
  eigenvalue-sum differences match self-consistent differences to
  $<1\%$ at $\theta = 0.01$.
- The self-consistent (relaxed-$\rho$, per-$q$ SCF, protocol P-c)
  curvature differs from the P-b curvature at finite smearing — the
  frozen-potential error that literature benchmarks (Daglum 2026)
  measure at up to hundreds of percent in high-moment systems. P-c
  stays available as the reference protocol.
- Plain (non-degenerate) second-order perturbation theory requires a
  non-degenerate reference spectrum; the symmetric AFM dimer is exactly
  degenerate and needs degenerate perturbation theory (the script uses
  unequal fields).
