# Split-SOC Insertion Algebra — Generalized-S Resolvent Derivation Report

Story 001 of the split-SOC KS-band spec. Promotes the research-note algebra
check (`split_soc_without_lcao_research.md`, "Algebra check performed for this
report") to a repo script.

Script: `split_soc_insertion.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-27). Exact symbolic (sympy, rational
matrices) + numeric (random complex, 1e-14 / machine precision).

## Contract objects

```
G_λ(z)   = [zS − H0 − λW]^{-1}     generalized propagator (non-orthogonal S)
G0       = [zS − H0]^{-1}          strength-0 reference (λ-free)
W        = W_SO                    enters the propagator only
V_a, V_b = magnetic-site vertices  untouched by λ, never fused with W
```

## Resolvent derivatives (asserted)

Exact symbolic, both `S = I` and positive-definite non-diagonal `S`:

```
dG/dλ|_0    = G0 W G0
d²G/dλ²|_0  = 2 G0 W G0 W G0
```

Numeric (random complex, `n = 4`): the derivative is re-derived from the
independent defining equations `A0 G′ = W G0` and `A0 G″ = 2 W G′`
(`A0 = zS − H0`; solved without reusing the closed form) — dev ≤ 1.1e-16 —
plus the exact cubic resolvent identity

```
(A0 − hW)(G0 + hG0WG0 + h²G0WG0WG0) = I − h³ (W G0)³     (dev ≤ 1.5e-16)
```

## Generalized-S factorization (asserted)

```
zS − H = S^{1/2} (z1 − S^{-1/2} H S^{-1/2}) S^{1/2}
```

exact symbolic for diagonal S; numeric 7.0e-14 (scale 6.0) for a general SPD
non-diagonal S. The orthonormalized propagator equals the generalized one —
the `S^{-1}`-dressing conventions of the spinor kernel apply to the
propagator, and once states are S-normalized the band-space identity is the
ordinary resolvent identity.

## Two-vertex insertion topologies (asserted)

```
d/dλ Tr[V_a G_λ V_b G_λ]|_0
   = Tr[V_a G0 W G0 V_b G0] + Tr[V_a G0 V_b G0 W G0]
```

- Richardson finite-difference corroboration (dev 3.8e-9, scale 1.0);
- each topology alone is not the derivative (gap 8.1e-2): **both orderings
  required**;
- SOC appears only sandwiched between propagators — never fused into a
  vertex (`V_a`, `V_b` are λ-independent by construction);
- site decomposition `W = W1 + W2` (W_SO covers **all** atoms, ligands
  included): the derivative is additive — `G0(W1+W2)G0 = Σ_c G0 Wc G0`
  (dev 0.0), likewise for the trace;
- strength-0 principle: `G0 = [zS − H0]^{-1}` is λ-free; all identities are
  evaluated at the λ = 0 reference.

## Scope note

This pins the **algebraic topology** of the first-order insertion and the
second-order term. It does not fix TB2J prefactors/signs of the final tensor
decomposition (those are pinned by `spinor_projector_green.md` and the
backend oracles), nor the physical completeness of any site's vertex or of
the KS window (Feshbach O(λ²) leakage — note gates 1–2).

## Consequences for implementation

1. First-order insertion diagnostic (PRD FR-013/FR-031 validation ladder):
   build `G0` from the strength-0 eigenvalues, apply the two topologies.
2. Second-order term `2 G0 W G0 W G0` is available for convergence audits of
   the second-variation (full-window) mode.
3. Non-orthogonal backends: use the generalized form (or equivalently the
   `S^{-1/2}`-orthonormalized form — asserted identical); do not mix metrics.
4. Site-resolved `W_SO` sums linearly; ligand sites contribute without
   magnetic vertices (vertices only on magnetic sites — contract).
