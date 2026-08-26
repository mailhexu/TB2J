# ADR 0005: Low-spin validity and real effective-spin policy

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owners:** TB2J maintainers

## Context

The selected finite-temperature methods have known regime limitations. The source paper
warns that Callen is less reliable for low spin. Swendsen’s AFM Callen treatment reports
an intermediate-temperature $S=\frac12$ fcc instability. For a true quantum spin
$S=\frac12$, $-A(S^z)^2$ is a constant: plain RPA’s standard SIA ordering instead creates
the unphysical gap $2AS$.

DFT local moments are generally not quantized, yet users need a continuum of effective
spin lengths when mapping first-principles exchange to a spin model.

## Decision

1. CD/HP outputs are computed when self-consistency is stable, but include an explicit
   method-validity status and reason for low-spin/known-instability regimes. No method is
   silently replaced by RPA.
2. For exact quantum $S=\frac12$ with nonzero SIA, reject plain RPA. Require RPA+CD, CD,
   or HP so SIA remains physically constant at this spin.
3. Quantum mode permits real effective $S_i$, defaulting to $|\mu_i|/(2\mu_\mathrm B)$.
   Results distinguish `physical_quantum_spin` (integer/half-integer) from
   `effective_quantum_spin` (real continuation). Per-site overrides remain available.

## Consequences

- The exact $S=\frac12$ SIA rule is applied only when $S$ is numerically equal to one half;
  it must not be generalized to nearby effective real spins.
- Effective-spin results are still reproducible and useful, but they cannot claim the
  exact finite-dimensional spin representation used by the SymPy matrix checks.
- The result schema must preserve method-validity and spin-interpretation metadata.
