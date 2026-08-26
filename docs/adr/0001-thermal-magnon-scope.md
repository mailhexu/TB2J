# ADR 0001: Thermal-magnon product scope

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owners:** TB2J maintainers

## Context

TB2J has a multi-sublattice $T=0$ magnon BdG solver but no finite-temperature magnon
renormalization or critical-temperature solver. arXiv:2405.00477 provides RPA, Callen,
HP mean-field, and RPA+CD formulations for the Heisenberg model. Its ferromagnetic
formulas do not by themselves establish a Néel-temperature implementation.

The authoritative derivations and TB2J convention bridge are in [docs/sympy](../sympy/README.md).

## Decision

The thermal-magnon feature will:

1. expose **RPA, Callen, HP mean-field, and RPA+CD** as distinct methods;
2. calculate temperature-dependent bands and both **Curie** and **Néel** temperatures;
3. accept isotropic exchange $J_\mathrm{iso}$, longitudinal exchange anisotropy $\lambda$,
   and single-ion anisotropy $A$;
4. expose explicit **quantum** and **classical** spin regimes; and
5. provide a reusable library API plus reproducible `TB2J_magnon2` CLI/TOML configuration.

## Consequences

- Results must identify the chosen method and spin regime. They must not silently
  substitute a different decoupling or SIA convention.
- Néel-temperature support requires an explicitly validated local-frame, multi-sublattice
  formulation; it cannot reuse the ferromagnetic scalar formula by changing the sign of $J$.
- The next decisions must define the AFM input boundary, treatment of DMI/transverse tensor
  terms, q-mesh convergence protocol, and failure semantics for unstable/no-transition models.
