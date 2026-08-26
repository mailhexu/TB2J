# ADR 0003: Thermal-method result semantics and corrected Callen reference

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owners:** TB2J maintainers

## Context

RPA and Callen decoupling have continuous ordered-moment loss and can report
$T_\mathrm C$/$T_\mathrm N$. HP mean field becomes invalid at a positive-order-parameter
spectrum instability. The paper’s printed multi-site Callen appendix also fails the
single-site and $T=0$ limits; `docs/sympy/03_anisotropy_multisite_conventions.py`
verifies the necessary corrections.

## Decision

1. HP returns `temperature_hp_breakdown` and the associated finite magnetization. It does
   **not** label that number $T_\mathrm C$ or $T_\mathrm N$.
2. The implementation uses the corrected multi-site Callen equations documented and
   asserted in `docs/sympy/03_*`; the original arXiv source remains in `Refs/`.
3. The feature must deliver all selected methods for both Curie and the initial collinear
   bipartite Néel domain. CD/HP AFM support is therefore gated on a new local-frame
   interacting-Nambu/BdG derivation and benchmark suite; it may not reuse the FM scalar
   convolution.
4. Temperature-dependent bands are evaluated only at explicit requested temperatures and
   a user-selected band path.
5. A failed mesh tolerance returns the estimate and full history as `converged: false`;
   strict mode reports it as an error.

## Consequences

- Results are comparable only when their method status is included.
- The initial feature cannot start implementation until the AFM CD/HP derivation is
  promoted from a documented research task to verified reference mathematics.
- Consumers can distinguish physical HP breakdown from a numerical failure, and inspect
  mesh sensitivity instead of receiving a silently arbitrary transition temperature.
