# Signed Local beta/delta Gradient on Nonorthogonal Eigenpairs

Companion script: `local_gradient_nonorthogonal.py` (assertion-checked;
run in the `mydev` environment). Status: **all assertions pass**
(2026-09-25).

Story 001 (spiral-first-order-response), NFR-001/NFR-005, FR-002/FR-003.
Companion to `planar_su2_frame.md` (same conventions, one source).

## Physics invariant (binding for all production gradients)

A LOCAL spin rotation perturbs the magnetic on-site field (plus declared
co-rotating local operators), **not** the whole basis. In the fixed
planar local frame the flat field is $B_f\sigma_z$ and

$$
V_1^{\beta}=B_f\sigma_x,\qquad V_1^{\delta}=B_f\sigma_y,\qquad
\partial S=0 .
$$

In the lab frame at angle $\Theta$ the same vertices read
$V_1^{\beta}=B_f(-\sin\Theta\,\sigma_z+\cos\Theta\,\sigma_x)$,
$V_1^{\delta}=B_f\sigma_y$, with the covariance
$V_1^{\rm lab}(\Theta)=U(\Theta)\,V_1^{\rm local}\,U(\Theta)^{\dagger}$
(all asserted symbolically). The second-order vertex is
$V_2=-B_f\,\hat n\cdot\boldsymbol\sigma$ per unit rotation squared, and
the splitting of $B_f\,\hat n\cdot\boldsymbol\sigma$ is $2B_f=B_{\rm local}$
(the Pauli-field amplitude is half the stored up/down splitting).

Rotating the whole H **and** S by site unitaries is pure gauge and has
exactly zero band-energy derivative — see check [D]: the `dH` artefact
alone is material ($-7.8\times10^{-3}$ eV/rad) and is cancelled to
$-2.0\times10^{-16}$ by the $-\varepsilon\,\partial S$ term. A nonzero
$\partial S$ therefore marks a *representation change*, never a local
physical gradient, unless proven otherwise.

## Central identity

For a generalized eigenpair $Hc=\varepsilon Sc$ with $c^{\dagger}Sc=1$:

$$
\mathrm d\varepsilon=c^{\dagger}(\mathrm dH-\varepsilon\,\mathrm dS)c,
\qquad
g_i^a=\sum_{\mathbf k}w_{\mathbf k}\sum_n f_{n\mathbf k}\,
c_{n\mathbf k}^{\dagger}\bigl(V_{1,i}^a-\varepsilon_{n\mathbf k}\,
\mathrm dS_i^a\bigr)c_{n\mathbf k}
$$

in eV/rad, atom-summed over the orbitals owned by atom $i$ (orbital
contributions sum before atom reporting). With $\partial S=0$ this is
$\operatorname{Tr}(\rho V_1)$.

## Checks and observed values

| Check | Content | Observed |
|-------|---------|----------|
| [A] symbolic vertices | $V_1$/$V_2$, local flat forms, $U$-covariance, splitting | exact |
| [B] pencil identity | exact rational pencil: $\mathrm d\varepsilon=c^{\dagger}(\mathrm dH-\varepsilon\mathrm dS)c$; dropping $-\varepsilon\mathrm dS$ provably wrong | exact |
| [C] numeric gradients | analytic (local) == analytic (lab) $\le10^{-9}$ == frozen FD (folded **and** lab) $\le10^{-7}$; multi-orbital atoms; off-equilibrium $\max|g^{\beta}|=2.7\times10^{-3}$ (complex) / $6.6\times10^{-3}$ (real); $\sum_i g_i^{\beta}=0$ to $8\times10^{-17}$ | pass |
| [C] delta channel | vanishes by planar reflection symmetry on the time-reversal-symmetric (real-amplitude) model, analytic AND FD: $2.4\times10^{-16}$; **nonzero** ($2.0\times10^{-2}$) with complex (flux) hoppings — physics, not gauge | pass |
| [C] uniform-FM control | $q=0$: all gradients $\le2.4\times10^{-17}$ | pass |
| [D] pure gauge | one-form $-2.0\times10^{-16}$ == 0 while dH-only artefact $-7.8\times10^{-3}$; band-energy drift along the gauge direction $1.8\times10^{-11}$ | pass |

## Two physics notes pinned by the numeric controls

1. **Planar symmetry needs time reversal.** The delta channel vanishes
   by the reflection $M=i\sigma_y$ composed with $R_y(\pi)$ only when
   the hopping amplitudes are real. Complex (flux) hoppings give a
   genuine $\langle\sigma_y\rangle\ne0$ and a nonzero $g^{\delta}$: the
   production delta channel must be *calculated*, not assumed zero, and
   compared against a real-amplitude control.
2. **A collinear spiral reference is not stationary per cell.** With
   $q\ne0$ and $\varphi=\tau=0$ the fields still wind ($\Theta_a=2\pi
   qa$): $g^{\beta}\ne0$ with equal-and-opposite atom pairs. Only the
   $q=0$ uniform-FM reference has all gradients zero.

## Implementation consequences

1. Production local gradients: local-frame vertices $B_f\sigma_x$ /
   $B_f\sigma_y$ per atom's orbitals, `dS = None`, frozen reference
   occupations, atom-summed over `orbital_to_atom`.
2. Never differentiate a whole-basis dressing to infer a physical
   local torque: the complete $-\varepsilon\partial S$ term makes the
   basis-rotation one-form vanish, whereas a $dH$-only calculation can
   produce a spurious nonzero apparent gradient (negative control [D]).
3. The frozen FD oracle is field-only in BOTH frames; the lab ring and
   the folded pencil must use the same frozen occupations (spectrum
   equality makes the ascending lists identical).
