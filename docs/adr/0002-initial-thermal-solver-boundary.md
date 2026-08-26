# ADR 0002: Initial thermal-solver physical and numerical boundary

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owners:** TB2J maintainers

## Context

The selected RPA, Callen, HP, and RPA+CD methods in arXiv:2405.00477 are controlled for
the collinear exchange-plus-longitudinal-anisotropy Hamiltonian. TB2J's $T=0$ magnon
solver is more general: it accepts full tensors, DMI, noncollinear reference frames, and
multi-sublattice BdG spectra. Extending the thermal decouplings to every existing
$T=0$ state would silently exceed the derivation.

Critical-temperature sums are infrared-sensitive. In particular, a finite fixed q mesh
can report a nonphysical finite transition for a nearly gapless 2D magnet.

## Decision

The initial thermal solver will:

1. support uniform FM and a **collinear bipartite AFM** with two equivalent antiparallel
   sublattices; the latter is the initial $T_\mathrm N$ contract;
2. accept only $J_\mathrm{iso}$, longitudinal exchange anisotropy $\lambda$, and SIA $A$;
   reject DMI, transverse/off-diagonal bilinear tensors, and nonstationary/canted inputs;
3. default to $S_i=|\mu_i|/(2\mu_\mathrm B)$ and allow explicit per-site $S_i$ override;
4. require a q-mesh convergence protocol, preserving mesh history and tolerance; and
5. return structured physical statuses: `zero_transition`, `unstable_reference`, and
   `hp_breakdown`, separate from invalid-input and numerical-convergence errors; and
6. require explicit `thermal_dimensionality` (1, 2, or 3) and `order_mode`
   (`ferromagnetic` or `bipartite_afm`) rather than inferring either property.

## Consequences

- A result for an unsupported tensor cannot be obtained by dropping terms. The user must
  reformulate the model or await a separately derived general interacting-BdG method.
- The solver can state $T_\mathrm C$/$T_\mathrm N=0$ for a stable Mermin–Wagner-limited model
  without treating that result as a numerical failure.
- General multi-sublattice, inequivalent-spin, and noncollinear finite-temperature
  thermodynamics remain deliberately out of scope and need an independent ADR and
  derivation before support.
