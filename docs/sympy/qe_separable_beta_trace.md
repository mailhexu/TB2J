# QE separable beta-operator trace — SymPy verification report

Script: `qe_separable_beta_trace.py` (run with the `mydev` environment).
Spec: `memnotes/Projects/TB2J/specs/qe-projector-exporter/` (Stories 1–2, ADR-002).

## Claim verified

QE's nonlocal Hamiltonian is the separable operator

$$
V_{\rm NL}=\beta D\beta^\dagger,\qquad D\equiv \texttt{deeq},\qquad P\equiv\texttt{becp}=\beta^\dagger\psi .
$$

The two-spin exchange trace therefore reduces, **exactly**, to a channel-space
contraction over `deeq` and the undressed beta-projected Green matrix
$G_\beta=\beta^\dagger G\beta$:

$$
K_{\rm phys}=\operatorname{Tr}\!\left[(\beta D_\uparrow\beta^\dagger)\,G_\uparrow\,(\beta D_\downarrow\beta^\dagger)\,G_\downarrow\right]
=\operatorname{Tr}\!\left[D_\uparrow G_{\beta,\uparrow}D_\downarrow G_{\beta,\downarrow}\right].
$$

## Method

Three independent complex-rational realizations (seed 20261008): random
$\beta\in\mathbb{Q}(i)^{3\times2}$ (nonsingular Gram), Hermitian $D_{\uparrow,\downarrow}\in\mathbb{Q}(i)^{2\times2}$,
Hermitian $G_{\uparrow,\downarrow}\in\mathbb{Q}(i)^{3\times3}$. All identities are asserted with
`sympy.simplify(...) == 0` — exact arithmetic, no floating point.

## Assertions per trial

1. **Direct separable identity** — `Tr[D_up Gb_up D_dn Gb_dn] == Tr[V_up G_up V_dn G_dn]`. Passes.
2. **Jointly transformed equivalent** — with $M=\beta^\dagger\beta$,
   $\Delta_{\rm cov}=MDM$ and $G_{\rm dual}=M^{-1}G_\beta M^{-1}$ reproduce the same trace. Passes
   (algebraically valid; not used by TB2J — needless Gram inversions).
3. **Falsified alternative** — dressing *only* the Green matrix by $M^{-1}$
   while leaving $D$ unchanged fails the identity in every trial. This is the
   algebraic reason `qq_at` / $M$ must never be applied as a channel metric to
   the undressed `becp` Green.

## Empirical corroboration (real QE dumps)

Fork `mailhexu/q-e` branch `TB2J` (`PW/src/becp_dump.f90`), dump v1.1 golden
runs (2026-10-09): first-k beta Gram vs `qq_at`, per-atom Frobenius
mismatches — O US 791.715, H US 4223.086, Cu PAW 121.609, O US (in Cu PAW
run) 788.963. The Gram hypothesis is also numerically excluded.

## Consequences (TB2J contract)

- `ProjectorGreenData.overlap_k=None`: `becp` coefficients are already dual
  relative to the implicit basis dual to beta.
- `hij = deeq↑ − deeq↓` (Ry→eV) is the matching covariant operator
  (`hij_definition="qe_deeq_spin_difference"`).
- `qq_at` and the dumped Gram remain diagnostics only; `qq_at` belongs to the
  wavefunction overlap operator $S_\psi=I+\beta q_{\rm at}\beta^\dagger$.
