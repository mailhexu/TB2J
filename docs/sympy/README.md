# TB2J SymPy Derivations — Authoritative Reference

This directory is the reference for all magnon and spin-thermodynamics mathematics in
TB2J (per workspace `AGENTS.md`). Every derivation is a self-contained sympy script with
assertion checks (`PASS` output) plus a companion Markdown report. Run with the `mydev`
environment:

```bash
source /home/hexu/projects/myenvs/mydev/bin/activate
python 01_heisenberg_hp_lswt.py
```

| File | Contents | Source |
|------|----------|--------|
| `01_heisenberg_hp_lswt.py/.md` | Spin algebra, Heisenberg model, Holstein–Primakoff transformation, linear spin-wave theory (FM + 2-sublattice AFM), Bogoliubov diagonalization | from scratch; paper Eq. HP |
| `02_rpa_callen_tc.py/.md` | Green's-function equation of motion, RPA (Tyablikov) decoupling, Callen decoupling, Callen magnetization formula, Tc expressions, self-consistent finite-T dispersions | from scratch; arXiv:2405.00477 Sec. II |
| `03_anisotropy_multisite_conventions.py/.md` | Single-ion anisotropy & anisotropic exchange in HP/RPA/CD, operator ordering, multi-site dynamical matrices, bridge to TB2J `magnon3.py` conventions | from scratch; arXiv:2405.00477 Sec. II.G + Appendix |

Primary method paper: arXiv:2405.00477 (source in `Refs/2405.00477/main.tex`).

## Conventions (must match implementation)

- Hamiltonian `H = -Σ_{i≠j} J_ij S_i·S_j` (J>0 ferromagnetic), energies in eV internally.
- Fourier: `J_q = Σ_R J_0R e^{-2πi q·R}`; magnon3 normalizes J by `1/(S_i S_j)` before the transform.
- Spins dimensionless; moments in μB with `m = 2S μB`.
- q-point meshes Γ-centered; Bose factors `n^B(ω,T) = 1/(e^{ω/kBT}-1)`.

When adding a new derivation or changing magnon conventions, add/update files here
first, then make the implementation match.
