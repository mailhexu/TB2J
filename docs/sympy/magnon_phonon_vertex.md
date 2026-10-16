# Explicit magnon–phonon vertex F and g

Run the assertion-checked derivation with `python docs/sympy/magnon_phonon_vertex.py`.
Implemented by `TB2J/magnon_phonon/vertex.py` (Story 009).

## Conventions

- Spin Hamiltonian `H = -sum_b J_b S_i . S_j` over a **directed** bond list
  `(i, j, R)` (both directions present); classical full-moment `J` in eV,
  `dJ/dtau` in eV/Angstrom, `J > 0` ferromagnetic (TB2J convention).
- Fourier `A(q) = sum_R A(R) exp(-2 pi i q . R)` (the `Magnon.Jq` convention).
- Collinear reference: all `V_i` parallel.  Local frames `U_i = e1_i + i e2_i`
  from the Story-006 solver; `sqrt(2 S_i)` lives in the transverse spin, never
  in `u, v`; `u, v` multiply unit-commutator HP bosons.
- Site bosons: `a_{iL} = N^{-1/2} sum_p e^{+2 pi i p L} a_i(p)`.
- Phonon displacement of atom `a` in cell `L` for mode `(q, nu)`:
  `tau_{a alpha}(L) = ZPA_{a nu} e^{mw}_{a alpha, nu}(q) e^{2 pi i q L}
  (b_{q nu} + b^dag_{-q nu}) / sqrt(N_cell)` with
  `ZPA = sqrt(hbar / (2 M_a omega))` and `e^{mw}` the mass-weighted unit-L2
  eigenvector including the phonopy basis-position phase.  The zero-point
  factor is applied **exactly once**, when the classical `dJ/dtau` rows are
  contracted with the pattern.
- Vertex convention (manuscript eq:mp-vertex):
  `H_mp = sum_{k q} g a^dag_{k+q} a_k (b_q + b^dag_{-q}) / sqrt(N_cell)`,
  `g = - sum_b O^pat_b(q) F_b`, `O^pat_{b,nu}(q) = sum_{a alpha}
  (dJ_b/dtau_{a alpha}) ZPA_{a nu} e^{mw}_{a alpha, nu}(q)`.

## Derivation

1. **HP expansion** (symbolic): `S_i = S_i V_i + sqrt(S_i/2)(a_i U_i* + a_i^dag U_i)`.
   The bilinear of bond `(i, j, R)` is, at linear spin wave,

   - transverse: `t_ij a_i^dag a_j + t_ij* a_i a_j^dag` with
     `t_ij = sqrt(S_i S_j)/2 (U_i . U_j*)` (asserted `= 2 sqrt(SiSj)/2 e^{i theta}`
     for a relative transverse-frame rotation `theta`);
   - anomalous site structure: `sqrt(S_iS_j)/2 (U_i* . U_j*) a_i a_j + h.c.`
     — **vanishes identically for any two collinear frames** because
     `U_i . U_j = 0` for real orthonormal transverse frames related by a
     rotation about the common `V` (asserted); the production vertex therefore
     enforces collinearity and rejects noncollinear references;
   - longitudinal: `- S_j a_i^dag a_i - S_i e^{-2 pi i q R} a_j^dag a_j`:
     each number operator carries the **partner** spin, and the `a_j^dag a_j`
     of bond `(i@L, j@L+R)` sits in cell `L+R`, producing the
     `e^{-2 pi i q R}` factor relative to the bond's home-cell modulation;
   - no linear-in-boson terms for a collinear stationary reference (asserted),
     hence **no bilinear magnon–phonon `a^dag b` mixing block** can be built
     from the scalar collinear coupling — the symmetry-forbidden control is
     asserted symbolically and re-tested numerically;
   - an on-site bond `(i, i, 0)` cancels exactly (transverse `2 S a a^dag`
     minus longitudinal `2 S a^dag a` = commutator constant), so its vertex is
     identically zero (asserted).

2. **Bloch routing** (numerical identity, N = 6 ring): with the modulation
   tied to the bond home cell,

   - `sum_L e^{2 pi i q L} a_i^dag(L) a_j(L+R) -> e^{+2 pi i k R} a_i^dag(k+q) a_j(k)`
   - `sum_L e^{2 pi i q L} a_i(L) a_j^dag(L+R) -> e^{-2 pi i (k+q) R} a_j^dag(k+q) a_i(k)`
   - `sum_L e^{2 pi i q L} a_i^dag(L) a_i(L) -> a_i^dag(k+q) a_i(k)`
   - `sum_L e^{2 pi i q L} a_j^dag(L+R) a_j(L+R) -> e^{-2 pi i q R} a_j^dag(k+q) a_j(k)`

3. **Bogoliubov projection** (symbolic): `a_a(p) = sum_l [u_{l a}(p) gamma_{p l}
   + w_{l a}(p) gamma^dag_{-p l}]` with `w(p)` the Story-006 hole block at `p`.
   The number-conserving projection of `a_c^dag(p1) a_a(p2)` onto
   `gamma^dag_{k+q, mu} gamma_{k, nu}` is

   `P_{mu nu}[c, a] = u*_{mu c}(k+q) u_{nu a}(k) + w*_{nu c}(-k) w_{mu a}(-k-q)`,

   the second (anomalous) term coming from normal-ordering the `w w` piece.
   A `a_c a_a` site structure contributes only `u w` cross terms proportional
   to `U_i* . U_j*`, which vanish for collinear frames.

4. **Closed form** (implemented verbatim by `vertex.py`):

```
F_b[mu, nu](k, q) =
    e^{+2 pi i k.R}     t_ij  [u*_{mu i}(k+q) u_{nu j}(k) + w*_{nu i}(-k) w_{mu j}(-k-q)]
  + e^{-2 pi i (k+q).R} t_ij* [u*_{mu j}(k+q) u_{nu i}(k) + w*_{nu j}(-k) w_{mu i}(-k-q)]
  - S_j [u*_{mu i}(k+q) u_{nu i}(k) + w*_{nu i}(-k) w_{mu i}(-k-q)]
  - S_i e^{-2 pi i q.R} [u*_{mu j}(k+q) u_{nu j}(k) + w*_{nu j}(-k) w_{mu j}(-k-q)]

g_{mu nu nu'}(k, q) = - sum_b O^pat_{b, nu'}(q) F_b[mu, nu](k, q)
```

5. **Hermiticity** (symbolic and numeric): with `O(-q) = O(q)*` (real classical
   model), `F_b[mu, nu](k, q) = [F_b[nu, mu](k+q, -q)]*` holds **per directed
   bond**, hence `g_{mu nu nu'}(k, q) = [g_{nu mu nu'}(k+q, -q)]*` — the
   manuscript eq:hermiticity partner form.

6. **Finite-model reference** (numeric): a 1D two-atom-cell chain with
   exponential bond springs (a genuinely nonlinear classical `J(tau)` model,
   including an inert on-site bond) is displaced along the cos/sin components
   of a frozen pattern; the real-space modulated bilinear is assembled
   mechanically, transformed with a general two-momentum Bloch sum, and
   Bogoliubov-projected through an independent Colpa diagonalization.  The
   closed form matches to `<= 1e-9` at sampled `(k, q)`, the extracted linear
   coefficient is amplitude-independent (weak-displacement limit), and the
   contracted residue spectrum `eig(g g^dag)` is invariant under
   `u -> u Theta^dag`, `w -> w Theta_{-p}^T` rephasing (`g -> Theta g Theta^dag`).

## Scope notes

- Only the one-phonon, number-conserving magnon vertex `a^dag a (b + b^dag)` is
  produced.  Pair-creation channels (`a a`, `a^dag a^dag` site structures) and
  the c-number bond-energy (magnetostriction/tadpole) piece are excluded; they
  belong to the scattering story's kernel bookkeeping, not to this vertex.
- `w(p)` at the four momenta `k+q, k, -k, -k-q` is required; the Story-006
  accessor is called with all four folded points.
- The `home_cell` derivative-row convention is exact only at `q = 0` (all
  pattern phases unity).  At general `q` the rows must already be
  pattern-projected (finite-q campaigns) — `vertex.py` enforces this.
- S002 finite-q reports enter through
  `TB2J.magnon_phonon.loaders.load_s002_report(report_path, campaign, mode,
  expected_bonds, mode_evidence_path)`. The adapter hash-checks the companion
  mode-evidence JSON, requires the report/source q and mode label to agree,
  preserves its complex Cartesian polarization plus O/FD components, and
  refuses missing row components or an incomplete directed-bond set. Use the
  returned `ModeProjectedDerivatives` directly as the model's derivative input;
  call `model.vertex(k, q, phonon_modes=..., mode_index=...)`.
- A finite-q mode-projected record is joined only when the supplied
  `PhononModeInput.provenance` includes the same `mode_evidence_sha256` and
  `mode` label, and its phonon branch has the report's eigenvector index and
  matching unit-Cartesian polarization. A missing or mismatched identity is an
  unsupported input, not a guessed projection.
- `load_matched_rows` accepts only complete Cartesian-axis coverage for each
  displaced atom represented in the home-cell evidence. Missing axes are
  rejected rather than zero-filled.
