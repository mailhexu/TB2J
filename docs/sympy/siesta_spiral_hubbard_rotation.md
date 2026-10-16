# +U Hubbard rotation contract: co-rotating occupation, frozen double counting

Run: `mydev python docs/sympy/siesta_spiral_hubbard_rotation.py`
Story 015 (FR-010 / ADR-007), derivation-first per ADR-003.  Sections
[A]-[D] are exact SymPy (no production code imported); section [E] is the
numeric verification on the certified story-014 ground truth.

## Conventions (pinned, mirrored from the certified module)

- Spinor spinor layout: file order D1..D8 with the diag3k fix; channels
  (uu, dd, ud, du) in SIESTA's `Dij(1..4)` order.  The spin index is
  OUTER in all symbolic matrices: element `(2*orbital + spin)`.
- Local spin rotation: `U_s = exp(-i theta sigma.n_hat / 2)`, right-handed
  active axis; the symbolic axis is parametrized exactly as
  `n_hat = (sin(phi) cos(lambda), sin(phi) sin(lambda), cos(phi))`.
- FLL assembly (m_pot_dftu.F90:1057-1141), certified slot order:
  `direct = vee(m, m'', m', m''')`, `fock = vee(m, m''', m'', m')`;
  `V_uu = vee(n_uu + n_dd) - fock(n_uu)`, `V_dd = vee(n_uu + n_dd) - fock(n_dd)`,
  `V_ud = -fock(n_ud)`, `V_du = -fock(n_du)`.
- Double counting (m_pot_dftu.F90:974-1035): `V_dc[uu/dd] = -delta (U (n_tot
  - 1/2) - J (t_uu/t_dd - 1/2))`, `V_dc[ud/du] = +delta J t_ud/du` —
  scalar x identity per spin channel, built from the CONVERGED traces.
- Units: everything internal is Ry (U = 0.2205 Ry, J = 0.05145 Ry on the
  fe4 probe); provenance carries eV alongside.

## What is pinned

### [A] Occupation co-rotation

`N(theta) = (U_s (x) I_orb) N (U_s^dag (x) I_orb)` with, exactly:

- `U_s` unitary; the rotation axis invariant;
- `U_s sigma_a U_s^dag = sum_b R[b, a] sigma_b` (transposed index
  placement; checked against the direct axis-z computation
  `U sigma_x U^dag = cos theta sigma_x + sin theta sigma_y`);
- full spinor trace `Tr N(theta) = Tr N` and `t_uu + t_dd` invariant;
- single channel traces NOT invariant (the algebraic reason why dc
  freezing is a contract rather than a consequence);
- the moment vector `m_a = Tr(sigma_a N)` = `(t_ud + t_du, i(t_ud - t_du),
  t_uu - t_dd)` rotates with the active Rodrigues map: `m(theta) = R m(0)`.

### [B] FLL Hubbard operator co-rotates (direct AND fock)

With `vee(m1, m2, m3, m4) = vee(m2, m1, m4, m3)` (certified slot symmetry)
and NO spin indices on the kernel:

$$
W[N(\theta)] = (U_s \otimes I_{orb})\, W[N]\, (U_s^{\dagger} \otimes I_{orb})
$$

proven separately for the direct contraction, the fock contraction, and
the pinned assembly.  The algebra: both contractions are LINEAR in N and
spin-free, so the spin rotation passes through unchanged.

### [C] Double counting frozen under the MFT variation

- `V_dc` is exactly scalar x orbital identity, with all FOUR spinor
  channels on the orbital diagonal (uu, dd, ud, du as in the certified
  assembler): every orbital-off-diagonal entry zero, the orbital-
  diagonal ud/du entries carrying `+delta J t_ud` / `+delta J t_du`;
- `d V_dc / d theta = 0` exactly — the contract freezes V_dc at the
  reference traces; no rebuild happens;
- the isotropic piece `-U (n_tot - 1/2)` is trace-protected (a rebuild
  would reproduce it, because `n_tot` is invariant per [A]);
- the J pieces of ALL channels (uu/dd AND ud/du) are NOT protected:
  `t_uu` and `t_ud` are single channel traces, not invariant per [A];
  a rebuilt channel drifts by exactly `+J (t_x' - t_x)` (for uu/dd the
  U term cancels against the invariant `n_tot`);
- co-rotating `V_dc` would give a DIFFERENT operator (rejected reading,
  pinned by the nonidentity of the spin off-diagonal entry);
- `W` genuinely varies (`dW/dtheta != 0`): freezing W entirely or
  reducing it to a static scalar splitting is refused (ADR-007).

### [D] Assembled rotated operator, separately labeled

$$
B_{tot}(\theta) = U_s B_{xc} U_s^{\dagger} + V_U^s(\theta),\qquad
V_U^s = W - \tfrac{1}{2}\mathrm{Tr}_{spin}(W)\otimes I_{orb}
$$

The spin-trace subtraction is the orbital identity times the spin-trace
scalar: entry `(p, q)` of `Tr_spin(W)` is
`sum_sigma W[sigma, p, sigma, q]`, subtracted (times 1/2) from BOTH
spin-diagonal entries of EVERY orbital block `(p, q)` — not only the
orbital-diagonal ones.

- `V_U^s` co-rotates (its spin trace is SU(2) invariant) and carries no
  B_xc symbols (clean separation);
- `B_xc` enters as frozen extracted data: the density channel of XC is
  frozen under MFT (ADR-005), so the XC vertex varies ONLY through the
  explicit conjugation — no self-consistent XC regeneration term exists;
- first-order variation `dB_tot/dtheta|_0 = (i/2)[B_tot, sigma.n_hat]`,
  pinned exactly and pinned NON-zero.

### [E] Numeric verification (certified ground truth)

Data: `tests/data/fe4_occ.dftu_occ` (instrumented producer dump) and
`tests/data/fe4_dftu_story014.npz` (U = 0.2205 Ry, J = 0.05145 Ry,
F0 = U, F2 = 14 J / 1.625, F4 = 0.625 F2).

Ground-truth texture (reproduced by the script):

| quantity | value |
|---|---|
| n_d per site | 6.3860, 6.3856, 6.3860, 6.3856 |
| m_z per site | +2.5448, -0.0000, -2.5448, -0.0000 |
| abs(t_updn) per site | 0.0000, 1.2727, 0.0000, 1.2727 |
| worst site-block non-hermiticity | 1.5e-13 |

Certified vee tensor checks (d shell, real harmonics):

| check | value |
|---|---|
| slot symmetry max abs(v - v.perm(1,0,3,2)) | 1.2e-32 |
| hermiticity max abs(v - conj(v.perm(2,3,0,1))) | 3.5e-18 |
| F0-only reduction max abs(v - F0 delta delta) | 7.5e-15 |
| rotation scalar: exact real-harmonic orbital rotation (Wigner D
  conjugated with the module's own T; D orthogonal within 1e-12, Rz(pi)
  sign pattern pinned) | rel 2.1e-14 (tol 1e-11) |

Finite local rotations (6 random axes/angles in [0.15, 2.9] rad + pi/2
about x and y, applied per site):

| gate | worst value | tolerance |
|---|---|---|
| `W[n']` vs `U W[n] U^dag` (reassembled vs conjugated) | 5.073e-16 relative | 1e-12 (approved) |
| `n_tot'` vs `n_tot` | 2.782e-16 relative | 1e-12 |
| moment `m'` vs `R m` | 6.108e-16 relative | 1e-12 |
| contract dc vs reference dc | bit-exact (frozen) | exact |
| rebuilt-from-rotated dc drift (counterfactual) | 1.261101e-01 Ry | reported, nonzero |
| dc drift decomposition residual | 8.3e-17 Ry | 1e-12 |

The counterfactual drift (0.123 Ry max) shows dc freezing is a real
contract, not a no-op: rebuilding V_dc from the co-rotated occupations
would move it by orders more than any tolerance, while the approved
contract keeps it bit-exactly at the converged reference.

## Limits

- The symbolic covariance is orbital-size independent (proven on a
  two-orbital shell); the numeric section exercises the full 5x5 d shell.
- The vee rotation-scalar check rotates the ORBITAL harmonics (exact
  Wigner-D); the spin rotation acts only on spin indices by construction.
- The dc "frozen" statement is an MFT definition (ADR-005/ADR-007), not a
  consequence of the rotation algebra: [A] explicitly pins that the
  channel traces would change, [C] pins that the contract does not rebuild.
- Matched frozen-rotation checks against the AC-005-validated response
  core (story-010 machinery) are the AC-008 gate of story-016, not this
  story.
