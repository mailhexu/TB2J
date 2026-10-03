# Constrained Envelope and Finite-Width Free-Energy Identities

Companion script: `constrained_free_energy.py` (assertion-checked; run
in the `mydev` environment). Status: **all assertions pass**
(2026-09-25).

Story 001 (spiral-first-order-response), NFR-005, FR-010 conventions.
Independent of TB2J/TBUpy; these identities decide the sign of the
multiplier term in a constrained total-energy slope and why a raw
eigenvalue-sum difference is not a free-energy slope at finite smearing.

## Constrained envelope (sign pinned)

For a constrained branch $C(x,\theta)=0$ and Lagrangian
$\mathcal L=E+\lambda C$ (THIS sign convention is pinned):

$$
\frac{\mathrm dE_{\rm con}}{\mathrm d\theta}
=\Bigl(\partial_\theta E+\lambda\,\partial_\theta C\Bigr)
\Big|_{x=x_*(\theta)} .
$$

The multiplier term $\lambda\,\partial_\theta C$ is the constraint work;
it is genuinely nonzero at generic $\theta$ (asserted). With the
opposite convention $\mathcal L'=E-\lambda' C$ the multiplier flips,
$\lambda'=-\lambda$: misstating the convention misstates the constraint
work. The script verifies the identity, the flip, and a numerical
Newton branch whose re-solved central difference matches
$\partial_\theta E+\lambda\,\partial_\theta C$ to $2\times10^{-12}$.

## Finite width (what the raw sum misses)

Mermin free energy at fixed electron number $N$:

$$
F=\sum_n f_n\varepsilon_n
+T\sum_n\bigl[f_n\ln f_n+(1-f_n)\ln(1-f_n)\bigr]
=\sum_n f_n\varepsilon_n-TS ,
$$

stationary in the occupations at fixed $N$
($\varepsilon_n-\mu=T\ln(f_n/(1-f_n))$, positive entropy
$S=-\sum[f\ln f+(1-f)\ln(1-f)]$). Along the branch of stationary
states (envelope theorem):

$$
\frac{\mathrm dF}{\mathrm dq}=\sum_n f_n\frac{\partial\varepsilon_n}{\partial q}
\qquad\text{(frozen VARIATIONAL occupations)},
$$

while the raw eigenvalue sum with re-Fermi-filled occupations obeys

$$
\frac{\mathrm dE_{\rm raw}}{\mathrm dq}
=\frac{\mathrm dF}{\mathrm dq}
+\sum_n f_n'\,(\varepsilon_n-\mu)\;\ne\;\frac{\mathrm dF}{\mathrm dq}
\quad(T>0).
$$

Re-Fermi filling alone does not turn a raw-sum slope into the
free-energy slope; the entropy piece — equivalently, evaluating
$\sum f\,\partial_q\varepsilon$ at the variational occupations — is
required. As $T\to0$ the difference closes.

## Checks and observed values

| Check | Content | Observed |
|-------|---------|----------|
| [A] symbolic envelope | branch identity, $\lambda'=-\lambda$ under $\mathcal L'=E-\lambda'C$, nonzero constraint work | exact |
| [B] numeric branch | FD of re-solved $E_{\rm con}$ == $\partial_\theta E+\lambda\partial_\theta C$ | $2\times10^{-12}$ |
| [C] symbolic Mermin | $\mathrm dF/\mathrm dq=\sum f\,\partial_q\varepsilon$ on the branch ($\partial_fF=0$); raw excess $=\sum_n f_n'(\varepsilon_n-\mu)$ | exact |
| [D] numeric (4 levels) | FD of re-solved $F$ == frozen-variational slope | $<10^{-12}$ |
| [D] negative control | raw re-Fermi FD differs by $+2.165\times10^{-1}$, matching $\sum f'(\varepsilon-\mu)$ to $10^{-9}$ | pass |
| [D] zero-width limit | $|{\rm raw}-F|$ slope difference: $2.4\times10^{-1}$ (T=0.2) $\to$ $0.0$ (T=1e-3) | pass |

## Implementation consequences

1. A matched-q SCF pitch slope (FR-010) may use a reported free energy
   only if the functional includes the entropy term exactly as above
   and the occupations are the fixed-N variational ones; a slope of the
   raw eigenvalue sum (with or without re-Fermi filling) is a different
   number at finite width, off by $\sum f'(\varepsilon-\mu)$.
2. Constrained-slope results must state the Lagrangian sign convention;
   the multiplier term enters with $+\lambda\,\partial_\theta C$.
3. The frozen-occupation folded slope of `pitch_slope_generalized.md`
   coincides with the free-energy slope at the variational occupations
   — the reason a frozen-occupation oracle is valid at finite width.
