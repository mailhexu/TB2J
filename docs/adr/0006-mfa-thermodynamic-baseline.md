# ADR 0006: MFA thermodynamic baseline boundaries and CLI surface

- **Status:** Accepted
- **Date:** 2026-08-27
- **Decision owners:** TB2J maintainers

## Context

Story 021 added a Weiss mean-field approximation (MFA) so method comparisons against
Pavizhakumari, Skovhus, and Olsen (arXiv:2405.00477) can distinguish the uncorrelated
single-site baseline from HP, RPA, and Callen decoupling. Review found the method
boundary was recorded only in code comments and the registered CLI exposed no thermal
entry point: `add_thermal_args`/`thermal_parameters_from_args` existed but nothing
consumed them (the ADR-0004 gap).

MFA is qualitatively different from the other thermal methods: it neglects correlations
entirely, so it has no temperature-dependent magnon spectrum at all — only the
single-site Brillouin order parameter `m(T)` and the analytic linearized transition.

## Decision

1. **MFA is an FM-only, bandless Weiss thermodynamic baseline outside the four
   magnon-spectrum methods** (RPA, Callen, HP, RPA+CD). `calculate()` answers requested
   temperatures with per-temperature band blocks that carry only `m(T)`
   (`order_parameters`, per-site `magnetization`, empty `kpoints`/`energies_eV`,
   `zero_transition` block status once the ordered solution is gone at or above `T_C`);
   no band k-points are consumed. The result metadata keeps labeling the method
   `thermodynamic_baseline` rather than `magnon_spectrum`.
2. **ADR-0003's both-order-modes coverage applies to the magnon-spectrum methods only.**
   MFA stays ferromagnetic: `thermal_method='mfa'` with `bipartite_afm` remains a
   validation error; the correlated bipartite-AFM answer is the Nambu RPA method.
3. **Low-dimensional finite MFA transitions carry `method_validity: limited`.** MFA
   returns a finite transition for isotropic 1D/2D ferromagnets in direct violation of
   Mermin–Wagner; the flagged validity record is the contract, not a suppressed result.
4. **`m(T)` is solved by bracketed root finding on the Brillouin residual**
   `g(m) = m − S·B_S(β J_W S m)` over `[ε·S, S]`, with the exact linearized
   existence condition `k_B T_C = J_W S(S+1)/3` as the disordered gate.
   Strict concavity, `B_S(0)=0`, and `B_S(x)<1` make `g < 0` just above zero
   and `g(S) > 0` below `T_C`, so the bracket always encloses the unique
   ordered root — at low temperature (roots near `S`, above any fixed probe
   ladder) and arbitrarily close to `T_C` (roots far below any fixed probe
   floor). Brent's method converges superlinearly there, where the plain
   fixed-point iteration `m ← S·B_S(β J_W S m)` decays only as `1/√n` and a
   capped iteration count would silently return a non-self-consistent `m`.
   The bracket width respects `thermal_max_iterations` via the root finder's
   iteration budget; failure raises instead of returning a stale iterate.
5. **The registered CLI (`TB2J_magnon.py` console command → `TB2J.magnon.magnon_cli`)
   consumes the thermal argument surface.** Any `--thermal-*` option triggers
   `ThermalMagnonSolver` on the same TB2J-results model the band/DOS paths build and
   serializes the versioned `tb2j.magnon.thermal` JSON alongside existing outputs
   (`--thermal-output`, default `<export-prefix>.thermal.json`); a thermal-only
   invocation without `--bands`/`--dos` is valid. The unregistered legacy
   `TB2J/scripts/TB2J_magnon.py` (old plot pipeline, does not delegate to
   `magnon_cli`) stays unwired by ADR-0004's do-not-promote rule.

## Consequences

- Thermal band blocks are no longer guaranteed to carry magnon energies: consumers
  must read `order_parameters` for MFA results; JSON round-trips rebuild empty
  `kpoints`/`energies` shapes, while NetCDF preserves the mode count.
- MFA block `status="zero_transition"` means "no ordered solution at this
  temperature" (same semantics as the spectrum methods' vanishing-order blocks), not
  a Mermin–Wagner model verdict, which stays in the transition record's validity.
- Serialization of explicit-temperature thermal bands from the CLI for
  magnon-spectrum methods follows the same dispatch; k-path selection for those
  temperature-resolved bands reuses the existing band-path options where provided.
