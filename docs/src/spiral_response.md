# Additive frozen-spiral response (`tb2j_spiral_response`)

Story-007 response API: given a persisted `tbupy_spiral_state` **v2**
bundle (`*.spiral.nc`), rebuild the planar spiral reference through the
real `tbupy.planar_spiral.PlanarSpiralProvider` and evaluate the additive
frozen-band response triple **without any SpinIO object**:

| report key | quantity | units | source module |
|---|---|---|---|
| `frozen_band_gradient` | per-atom local-angle gradient `g_beta`, `g_delta` (+ frozen-band torques) | eV/rad | `TB2J.spiral_first_order` |
| `frozen_band_pitch_slope` | `+dE/dq_a` at fixed variational occupations | eV / primitive cell / fractional q component | `TB2J.spiral_pitch` |
| nonstationary curvature | per-atom `(beta, delta)` curvature (`local_curvature`), full lab-angle Hessian blocks on the result object | eV/rad² | `TB2J.spiral_nonstationary` |
| `self_consistent_pitch_slope` | certified `+dF/dq_a` from a TBUpy matched-q SCF record — **only when such a record is supplied and passes the provenance gates; never fabricated** | eV / primitive cell / fractional q component | `tbupy.spiral_pitch` (matched-q driver) |

## CLI

```bash
python -m TB2J.scripts.tb2j_spiral_response \
    --spiral-state reference.spiral.nc \
    --output spiral_response.json
```

Optional flags: `--matched-q-record matched_q.json`, `--density-tol 1e-8`,
`--field-tol 1e-8` and `--translation-cutoff 6` (1D) or
`--translation-cutoff 3 3 1` (2D product mesh; all sampled k axes must form
a complete uniform Fourier grid).

The legacy `tb2j_spiral` CLI (magnetic-force-theorem exchange) is
untouched; this tool is additive and writes a single JSON report.

## What the layer gates (fail closed)

All gates abort with a non-zero exit code and write no output:

* **schema**: strict `tbupy_spiral_state` version 2 (legacy v1 is
  supported only by the unchanged stationary exchange path);
* **provenance**: required `gauge, q_frac, electron_count, occupation_rule,
  width, hubbard, constraint, field_symmetry, field_role` and
  `field_rotation_policy` records. The frozen fixed occupations reproduce
  the electron count, the q vector matches the sidecar, and Hubbard U/J,
  functional type and double-counting policy are explicit;
* **density**: Frobenius discrepancy between the stored same-frame
  density and the density rebuilt from the provider eigenpairs at the
  persisted occupations within `--density-tol`;
* **field symmetry**: planar co-rotating declaration (out-of-plane axis
  `y`, co-rotating rotation policy) plus vanishing out-of-plane moment
  `|<sigma_y>|` per orbital within `--field-tol`;
* **matched-q record** (if supplied): a full 24-leg components-mode
  certificate with converged and energy-certified plus/minus h and h/2
  legs, a ledger reproducing every free energy, and slopes reproduced
  from the base legs; exact k mesh/weights, q center, electron count,
  width, Hubbard U/J, functional/DC type and co-rotating field policy must
  match the sidecar. A bare `supported: true` flag is not accepted.
  `supported: true` flag is not accepted. Unsupported or incomplete
  records fail closed with a reason.

A constrained v2 reference may hold a separately recorded laboratory
constraint operator fixed while the intrinsic `B_local` reference field
remains co-rotating. Its frozen gradient and pitch slope remain defined;
the report labels global Ward and pair-model checks inapplicable because
that held field breaks their rotation symmetry. A request to reinterpret
the *entire* stored `B_local` as a laboratory-fixed field is refused:
the primitive planar pencil could not reconstruct that different
nonperiodic Hamiltonian from this sidecar.

## Sign and unit conventions

* Spin basis is **interleaved** everywhere: `(orb0 up, orb0 dn, ...)`.
* Twisted `planar_y` gauge: on-site field `Bf sz` with `Bf = B_local/2`,
  lab angles `alpha_i = 2 pi q . tau_i + phi_i`.
* **Gradient** (`eV/rad`, `+dE/d(angle)`): the perturbation rotates the
  magnetic on-site field only — hopping and overlap are untouched
  (rotating the whole basis is pure gauge).  In the fixed planar local
  basis the folded vertices are the plain Pauli images
  `beta -> Bf sx` (in-plane) and `delta -> Bf sy` (out-of-plane); both
  are `alpha`-independent because `sy` commutes with the `planar_y`
  twist `U(alpha) = exp(-i alpha sy / 2)`.  `torque_beta`/`torque_delta`
  are the negatives (frozen-band components, explicitly *not* the
  physical total-energy torque).
* **Pitch slope** (`+dE/dq_a` at fixed frozen occupations, no re-Fermi):
  includes the moving-basis `dS/dq` of a nonorthogonal overlap through
  the provider `q_derivative`; the q-independent rotating-frame `V_U`
  and constraint potential enter the pencil but not `dH/dq`.
* **Curvature** (`+d²E/dangle²`): per-atom `local_curvature` over
  `(beta, delta)`. For a three-component translation extent, the full
  cell-major Hessian blocks have dimension `prod(extents)*norb`;
  they stay on the result object while the JSON records block shapes.
  A complete uniform product k mesh is required for the primitive
  Fourier transform; an explicit commensurate lab supercell is an
  independent oracle, not the production evaluator.
* The JSON reports measured fixed-occupation central-FD residuals for
  both local gradients and all three pitch components, gradient-corrected
  Ward residuals when the field/constraint protocol is rotation symmetric,
  and the pair-once curvature prediction/mismatch as a diagnostic. The
  latter is a Heisenberg identity only under an explicit pairwise-isotropic
  assumption. A held laboratory constraint marks those Ward/pair checks
  inapplicable, without suppressing the frozen-band gradients.
* The `metadata` block echoes full provenance, checked gate tolerances,
  numerical FD step/residuals and sign/unit conventions.

## Python API

```python
from TB2J.spiral_response import (
    load_response_bundle,        # v2 sidecar -> gated ResponseBundle
    planar_field_vertices,       # per-atom folded (v1_beta, v1_delta)
    compute_frozen_response,     # full triple (+ optional matched-q record)
    response_report,             # JSON-ready dict
)

result = compute_frozen_response("reference.spiral.nc")
report = response_report(result)
```

`ResponseBundle` exposes the rebuilt provider (wrapped with the
q-independent `V_U`/constraint terms via
`PlanarSpiralProvider.with_periodic_operators`), the reference
eigenpairs, the persisted fixed occupations, and
`response_provider(q_frac=...)` for frozen-q finite-difference probes at
an overridden spiral wavevector.

## Relation to the matched-q driver

`tbupy.spiral_pitch.run_matched_q_scf_pitch_slope` certifies a
self-consistent pitch slope from SCF legs at `q_center +- h` steps.  Its
serialized `MatchedQPitchResult` can be attached here:

```bash
python -m TB2J.scripts.tb2j_spiral_response \
    --spiral-state reference.spiral.nc \
    --matched-q-record matched_q_result.json
```

The frozen-band pitch slope (this tool) and the matched-q free-energy
slope are deliberately distinct outputs: the former is the frozen
variational derivative, the latter the SCF-certified envelope derivative.
They are emitted side by side only when both exist.
