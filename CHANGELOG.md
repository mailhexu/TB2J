# Changelog

## Unreleased

### Wannier90 Wigner-Seitz weights (ws-weights epic)

- `WannierManager` now records which Wigner-Seitz interpolation scheme was
  auto-detected (scheme 1 = global `ndegen`, scheme 2 = per-orbital `_wsvec.dat`)
  in its output description, along with a migration note.
- **Breaking (correctness)**: because HamiltonIO's `WannierHam.gen_ham` now
  correctly divides by `ndegen(R)` (and applies `_wsvec.dat` when present),
  Wannier90-derived exchange constants **will differ from previous TB2J
  versions**. The new values are correct. Re-run any Wannier90-based TB2J
  calculation to update results.
- No `WannierManager` API change; wsvec is auto-detected from the Wannier90
  output directory. To force scheme 1 for A/B comparison, temporarily move
  `{prefix}_wsvec.dat` aside.
