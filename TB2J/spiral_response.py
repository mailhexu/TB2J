"""Standalone additive frozen-spiral response API (story 007).

Loads a persisted ``tbupy_spiral_state`` **v2** sidecar, rebuilds the
planar reference through the real
:class:`tbupy.planar_spiral.PlanarSpiralProvider`, and evaluates the
additive response triple on the frozen reference:

* ``frozen_band_gradient`` -- per-atom local-angle frozen-band gradient
  (beta/delta channels, eV/rad
  :mod:`TB2J.spiral_first_order`),
* ``frozen_band_pitch_slope`` -- frozen-occupation pitch slope dE/dq
  (eV per primitive cell per fractional q component

  :mod:`TB2J.spiral_pitch`, including the moving-basis dS/dq),
* nonstationary per-atom curvature (eV/rad^2

  :mod:`TB2J.spiral_nonstationary`),

plus an optional certified ``self_consistent_pitch_slope`` passed
through from a TBUpy matched-q SCF record (never fabricated).

Everything derives from the sidecar and the provider rebuild -- no
SpinIO object is involved anywhere.  Provenance, density-mismatch and
field-symmetry gates fail closed.

Sign/unit conventions (all reported in the JSON metadata):

* gradient: ``g_beta``/``g_delta`` are +dE/d(angle) in eV/rad of the
  local field direction.  In the fixed planar local basis the local
  field is ``Bf sz`` (``Bf = B_local/2``), so the folded vertices are
  the plain Pauli images ``beta -> Bf sx`` (in-plane rotation) and
  ``delta -> Bf sy`` (out-of-plane tilt): both are alpha-independent
  because ``sy`` commutes with the ``planar_y`` twist
  ``U(alpha) = exp(-i alpha sy/2)``.  The perturbation rotates the
  magnetic on-site field only (basis and overlap fixed, dS = 0).
* pitch slope: +dE/dq_a in eV per primitive cell per fractional q
  component at FIXED variational occupations (no re-Fermi)
  the
  moving-basis dS/dq of a nonorthogonal overlap is included through
  the provider ``q_derivative``.
* curvature: positive d^2E/d(angle)^2 in eV/rad^2.

CLI: ``python -m TB2J.scripts.tb2j_spiral_response``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SpiralResponseError",
    "SpiralResponseProvenanceError",
    "SpiralResponseDensityMismatchError",
    "SpiralResponseFieldSymmetryError",
    "SpiralResponseMatchedQError",
    "ResponseBundle",
    "SpiralResponseResult",
    "reference_eigenpairs",
    "spectral_density",
    "response_provider",
    "planar_field_vertices",
    "load_response_bundle",
    "compute_frozen_response",
    "response_report",
]

SCHEMA_NAME = "tb2j_spiral_response"
SCHEMA_VERSION = 1

REQUIRED_PROVENANCE_KEYS = (
    "schema_version",
    "gauge",
    "q_frac",
    "electron_count",
    "occupation_rule",
    "constraint",
    "hubbard",
    "field_symmetry",
    "field_role",
    "field_rotation_policy",
)

FIELD_ROLES = ("external", "constraint_proxy", "intrinsic_exchange")
CO_ROTATING_POLICIES = ("co_rotating_local", "co_rotating")

DEFAULT_DENSITY_TOL = 1e-8
DEFAULT_FIELD_TOL = 1e-8
KWEIGHT_TOL = 1e-6
ELECTRON_COUNT_TOL = 1e-6
HERMITICITY_TOL = 1e-10
PER_ATOM_CONFIG_TOL = 1e-10

PITCH_UNITS = "eV / primitive cell / fractional q component"
GRADIENT_UNITS = "eV/rad"
CURVATURE_UNITS = "eV/rad^2"


class SpiralResponseError(ValueError):
    """Base class of the fail-closed response-layer gates."""


class SpiralResponseProvenanceError(SpiralResponseError):
    """Missing/inconsistent v2 provenance (schema, gauge, counts)."""


class SpiralResponseDensityMismatchError(SpiralResponseError):
    """Stored density disagrees with the rebuilt spectral density."""


class SpiralResponseFieldSymmetryError(SpiralResponseError):
    """Bundle is not a planar co-rotating field reference."""


class SpiralResponseMatchedQError(SpiralResponseError):
    """Matched-q record unsupported or provenance-inconsistent."""


# ---------------------------------------------------------------------------
# spectral helpers (single source for the frozen protocol)
# ---------------------------------------------------------------------------


def reference_eigenpairs(provider, kpts, extra=None):
    """Generalized eigenpairs of the rebuilt pencil at every ``k``.

    ``provider.gen_ham(k)`` supplies ``(Hq, Sq)`` (interleaved spinor)

    optional ``extra`` Hermitian operator is added to ``Hq`` (local-angle
    FD probes).  Returns ``(eigenvalues (nk, nb), eigenvectors
    (nk, n, nb))`` in ascending order with ``c^dag S c = I``.
    """
    kpts = np.asarray(kpts, dtype=float)
    all_evals, all_evecs = [], []
    for ik in range(len(kpts)):
        hq, sq = provider.gen_ham(kpts[ik])
        if extra is not None:
            hq = hq + extra
        eps, c = eigh(hq, sq)
        all_evals.append(eps)
        all_evecs.append(c)
    return np.array(all_evals), np.array(all_evecs)


def spectral_density(evecs, occupations, kweights):
    """``rho = sum_k w_k sum_n f_nk c_nk c_nk^dag`` (electrons)."""
    evecs = np.asarray(evecs)
    occ = np.asarray(occupations, dtype=float)
    w = np.asarray(kweights, dtype=float)
    n = evecs.shape[1]
    rho = np.zeros((n, n), dtype=complex)
    for ik in range(len(w)):
        c = evecs[ik]
        rho += w[ik] * (c * occ[ik][None, :]) @ c.conj().T
    return rho


def response_provider(base, v_u=None, constraint_potential=None):
    """Wrap a planar provider with the q-independent rotating-frame terms
    of the v2 bundle (V_U / constraint potential, cell-periodic R=0
    operators per the bundle contract) via the provider's first-class
    ``with_periodic_operators`` seam: ``gen_ham`` adds them, ``q_derivative``
    delegates unchanged (dV_U/dq = dV_con/dq = 0 exactly), and nested
    wraps are rejected by the provider itself."""
    with_periodic = getattr(base, "with_periodic_operators", None)
    if with_periodic is None:
        raise SpiralResponseError(
            "PlanarSpiralProvider lacks with_periodic_operators; cannot "
            "attach the bundle V_U/constraint potential"
        )
    return with_periodic(V_U=v_u, V_con=constraint_potential)


# ---------------------------------------------------------------------------
# folded local-angle vertices (bundle-contract planar forms)
# ---------------------------------------------------------------------------


def planar_field_vertices(bundle):
    """Per-atom folded local-angle vertices ``(v1_beta, v1_delta)``.

    In the fixed planar local basis the on-site field is ``Bf sz`` with
    ``Bf = B_local[mu]/2``, so the field-only rotation vertices are the
    plain Pauli images on the orbitals ``mu`` of atom ``i``:

    * beta (in-plane rotation):      ``Bf sx``

    * delta (out-of-plane tilt):     ``Bf sy``.

    Both are alpha-independent in the planar_y gauge (``sy`` commutes
    with the twist).  Returns two ``(natom, 2 norb, 2 norb)`` Hermitian
    arrays acting on the interleaved spinor basis
    the representation
    (hopping/overlap) is NOT moved, so dS = 0 for these channels.
    """
    state = bundle.state
    atom_of_orbital = bundle.orbital_to_atom
    b_local = np.asarray(state.B_local, dtype=float)
    norb = len(b_local)
    n = 2 * norb
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])

    v1_beta = np.zeros((bundle.natom, n, n), dtype=complex)
    v1_delta = np.zeros((bundle.natom, n, n), dtype=complex)
    for mu in range(norb):
        iatom = int(atom_of_orbital[mu])
        bf = 0.5 * b_local[mu]
        sl = slice(2 * mu, 2 * mu + 2)
        v1_beta[iatom][sl, sl] += bf * sx
        v1_delta[iatom][sl, sl] += bf * sy
    return v1_beta, v1_delta


# ---------------------------------------------------------------------------
# v2 bundle loading + gates
# ---------------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class ResponseBundle:
    """Persisted v2 sidecar + real planar provider + frozen reference."""

    state: object
    provider: object  # real PlanarSpiralProvider
    response: object  # provider wrapped with V_U/constraint terms
    kpts: np.ndarray
    kweights: np.ndarray
    occupations: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    overlap: np.ndarray
    q_frac: np.ndarray
    orbital_to_atom: np.ndarray
    natom: int
    taus_atom: np.ndarray
    phis_atom: np.ndarray
    alpha_atom: np.ndarray
    efermi: float
    width: float
    nel: float
    provenance: dict
    field_symmetry: object
    density_discrepancy: float
    path: str
    density_tol: float
    field_tol: float

    def response_provider(self, q_frac=None):
        """Wrapped provider; ``q_frac`` rebuilds the base provider at an
        overridden spiral wavevector (frozen-q FD probes keep the stored
        occupations and the q-independent V_U/constraint terms)."""
        if q_frac is None:
            return self.response
        base = _build_planar_provider(
            self.state, self.orbital_to_atom, config_q_frac=np.asarray(q_frac, float)
        )
        return response_provider(
            base,
            v_u=self.state.V_U,
            constraint_potential=self.state.constraint_potential,
        )


def _require(condition, exc, message):
    if not condition:
        raise exc(message)


def _reduce_per_atom(values, atom_of_orbital, natom, name, tol=PER_ATOM_CONFIG_TOL):
    """Per-orbital v1-shaped array -> per-atom table (explicit check)."""
    values = np.asarray(values, dtype=float)
    out = np.zeros((natom,) + values.shape[1:], dtype=float)
    for iatom in range(natom):
        rows = values[atom_of_orbital == iatom]
        _require(
            len(rows) > 0,
            SpiralResponseProvenanceError,
            f"atom {iatom} has no orbitals in orbital_to_atom",
        )
        _require(
            np.allclose(rows, rows[0], atol=tol, rtol=0.0),
            SpiralResponseProvenanceError,
            f"per-orbital {name} values disagree inside atom {iatom}; the v2 "
            "planar config is per-atom and must be reduced explicitly, not "
            "double-expanded",
        )
        out[iatom] = rows[0]
    return out


def _build_planar_provider(state, orbital_to_atom, config_q_frac=None):
    """Construct the real PlanarSpiralProvider from v2 stored fields.

    The sidecar keeps the per-orbital v1 ``taus``/``phis`` shape
    the
    planar config is per-atom, so the arrays are reduced explicitly
    (validated equal inside each atom) -- never double-expanded.
    """
    from tbupy.planar_spiral import PlanarSpiralConfig, PlanarSpiralProvider

    meta = json.loads(state.metadata_json)
    q_frac = (
        np.asarray(config_q_frac, dtype=float)
        if config_q_frac is not None
        else np.asarray(state.q_frac, dtype=float)
    )
    atom_of_orbital = np.asarray(orbital_to_atom, dtype=int)
    natom = int(atom_of_orbital.max()) + 1
    taus_atom = _reduce_per_atom(state.taus, atom_of_orbital, natom, "taus")
    phis_atom = _reduce_per_atom(state.phis, atom_of_orbital, natom, "phis")

    hr_up = np.asarray(state.HR_up)
    hr_dn = np.asarray(state.HR_dn)
    _require(
        np.allclose(hr_up, hr_dn, atol=1e-10),
        SpiralResponseProvenanceError,
        "v2 bundle has spin-dependent collinear channels (HR_up != HR_dn); "
        "the planar spin-scalar provider requires spin-independent classes",
    )
    sr = np.asarray(state.SR)
    norb = hr_up.shape[1]
    offdiag = sr.copy()
    onsite = np.flatnonzero(np.all(np.asarray(state.Rlist) == 0, axis=1))
    if len(onsite) == 1:
        offdiag[onsite[0]] -= np.eye(norb)
    is_orthogonal = bool(len(onsite) == 1 and np.max(np.abs(offdiag)) < 1e-12)

    config = PlanarSpiralConfig(q_frac, taus_atom, phis_atom, atom_of_orbital)
    return PlanarSpiralProvider(
        hr_up,
        sr,
        np.asarray(state.Rlist, dtype=np.int64),
        config,
        np.asarray(state.B_local, dtype=float),
        nel=float(json.loads(state.metadata_json)["electron_count"]),
        is_orthogonal=is_orthogonal,
        source_flags=meta.get("guards"),
    )


def load_response_bundle(
    path, *, density_tol=DEFAULT_DENSITY_TOL, field_tol=DEFAULT_FIELD_TOL
):
    """Load a persisted v2 ``*.spiral.nc`` bundle and gate it.

    Gates (fail closed):

    * schema: ``tbupy_spiral_state`` with explicit ``schema_version == 2``
      (the tbupy loader is strict
      re-checked here)

    * provenance: required metadata keys ``schema_version, gauge,
      q_frac, electron_count, occupation_rule, constraint,
      field_symmetry, field_role, field_rotation_policy``

      ``gauge == 'planar_y'``; stored ``q_frac``
      consistent with metadata
      electron count consistent with
      ``sum_k w_k f_nk`` of the persisted fixed occupations

    * density: Frobenius discrepancy between the stored density and the
      density rebuilt from the provider eigenpairs at the stored frozen
      occupations within ``density_tol``

    * field symmetry: planar co-rotating field declaration (out-of-plane
      axis ``y``) plus vanishing out-of-plane moment ``|<sigma_y>|`` of
      every stored local block within ``field_tol``.
    """
    from tbupy.spiral_state import load_spiral_state

    path = Path(path)
    state = load_spiral_state(path)
    _require(
        int(getattr(state, "schema_version", 1)) == 2,
        SpiralResponseProvenanceError,
        f"{path}: schema_version 2 required, got "
        f"{getattr(state, 'schema_version', 1)}",
    )
    meta = json.loads(state.metadata_json)
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in meta]
    _require(
        not missing,
        SpiralResponseProvenanceError,
        f"{path}: missing v2 provenance keys {missing}",
    )
    _require(
        int(meta["schema_version"]) == 2,
        SpiralResponseProvenanceError,
        f"{path}: metadata schema_version must be 2",
    )
    _require(
        meta["gauge"] == "planar_y",
        SpiralResponseFieldSymmetryError,
        f"{path}: gauge {meta['gauge']!r} is not the planar_y response gauge",
    )
    field_symmetry = meta["field_symmetry"]
    if isinstance(field_symmetry, dict):
        axis = field_symmetry.get("local_field_axis")
        _require(
            axis in (None, "y"),
            SpiralResponseFieldSymmetryError,
            f"{path}: field_symmetry.local_field_axis {axis!r} is not 'y'",
        )
        policy = field_symmetry.get("field_axis_policy")
        _require(
            policy in (None, "co_rotating_local"),
            SpiralResponseFieldSymmetryError,
            f"{path}: field_symmetry.field_axis_policy {policy!r} is not "
            "'co_rotating_local'",
        )
    _require(
        meta["field_role"] in FIELD_ROLES,
        SpiralResponseProvenanceError,
        f"{path}: field_role {meta['field_role']!r} is not one of " f"{FIELD_ROLES}",
    )
    rotation_policy = meta["field_rotation_policy"]
    rotation_tag = (
        rotation_policy.get("kind")
        if isinstance(rotation_policy, dict)
        else rotation_policy
    )
    _require(
        rotation_tag in CO_ROTATING_POLICIES,
        SpiralResponseFieldSymmetryError,
        f"{path}: field_rotation_policy {rotation_policy!r} is not a "
        f"co-rotating policy {CO_ROTATING_POLICIES}; the additive response "
        "is defined for the co-rotating local-field protocol only",
    )

    q_frac = np.asarray(state.q_frac, dtype=float)
    _require(
        np.allclose(np.asarray(meta["q_frac"], dtype=float), q_frac, atol=1e-12),
        SpiralResponseProvenanceError,
        f"{path}: metadata q_frac disagrees with the stored q_frac",
    )
    nel = float(meta["electron_count"])
    kpts = np.asarray(state.kpts, dtype=float)
    kweights = np.asarray(state.kweights, dtype=float)
    occupations = np.asarray(state.occupations, dtype=float)
    _require(
        kpts.ndim == 2 and kpts.shape[1] == 3,
        SpiralResponseProvenanceError,
        f"{path}: kpts must have shape (nk, 3), got {kpts.shape}",
    )
    _require(
        abs(float(kweights.sum()) - 1.0) <= KWEIGHT_TOL,
        SpiralResponseProvenanceError,
        f"{path}: kweights must sum to 1 (got {float(kweights.sum())!r})",
    )
    _require(
        occupations.shape == (len(kpts), 2 * int(np.asarray(state.HR_up).shape[1])),
        SpiralResponseProvenanceError,
        f"{path}: occupations shape {occupations.shape} does not match the "
        f"interleaved spinor bands ({len(kpts)}, 2*norb)",
    )
    _require(
        np.all(np.isfinite(occupations))
        and occupations.min() >= 0.0
        and occupations.max() <= 1.0,
        SpiralResponseProvenanceError,
        f"{path}: occupations must be finite fractions in [0, 1]",
    )
    frozen_count = float(np.sum(kweights[:, None] * occupations))
    _require(
        abs(frozen_count - nel) <= ELECTRON_COUNT_TOL,
        SpiralResponseProvenanceError,
        f"{path}: electron_count {nel} inconsistent with sum_k w_k f_nk = "
        f"{frozen_count}",
    )
    _require(
        "width" in meta and float(meta["width"]) > 0,
        SpiralResponseProvenanceError,
        f"{path}: positive frozen-occupation smearing 'width' required",
    )
    width = float(meta["width"])

    orbital_to_atom = np.asarray(state.orbital_to_atom, dtype=int)
    natom = int(orbital_to_atom.max()) + 1
    _require(
        sorted(set(orbital_to_atom.tolist())) == list(range(natom)),
        SpiralResponseProvenanceError,
        f"{path}: orbital_to_atom must label atoms 0..{natom - 1} " "without gaps",
    )

    base = _build_planar_provider(state, orbital_to_atom)
    for name, term in (
        ("V_U", state.V_U),
        ("constraint_potential", state.constraint_potential),
    ):
        if term is None:
            continue
        t = np.asarray(term, dtype=complex)
        _require(
            np.linalg.norm(t - t.conj().T) <= HERMITICITY_TOL,
            SpiralResponseError,
            f"{path}: {name} is not Hermitian "
            f"(||T - T^dag||_F > {HERMITICITY_TOL:g})",
        )
    response = response_provider(
        base, v_u=state.V_U, constraint_potential=state.constraint_potential
    )

    eigenvalues, eigenvectors = reference_eigenpairs(response, kpts)
    overlap = np.array([response.gen_ham(k)[1] for k in kpts])

    rho_spec = spectral_density(eigenvectors, occupations, kweights)
    discrepancy = float(np.linalg.norm(rho_spec - np.asarray(state.rho)))
    _require(
        discrepancy <= density_tol,
        SpiralResponseDensityMismatchError,
        f"{path}: stored density mismatch ||rho_spec - rho||_F = "
        f"{discrepancy:.3e} > density_tol {density_tol:.3e}",
    )

    transverse = _max_transverse_moment(state.rho)
    _require(
        transverse <= field_tol,
        SpiralResponseFieldSymmetryError,
        f"{path}: field symmetry violated: max orbital |<sigma_y>| = "
        f"{transverse:.3e} > field_tol {field_tol:.3e} (the planar_y "
        "reference must carry no out-of-plane moment)",
    )

    taus_atom = _reduce_per_atom(state.taus, orbital_to_atom, natom, "taus")
    phis_atom = _reduce_per_atom(state.phis, orbital_to_atom, natom, "phis")
    alpha_atom = 2.0 * np.pi * taus_atom @ q_frac + phis_atom

    provenance = {k: meta[k] for k in REQUIRED_PROVENANCE_KEYS}
    # the frozen-band gradient contract pins schema_version as a string tag
    provenance["schema_version"] = str(meta["schema_version"])
    provenance["width"] = width

    return ResponseBundle(
        state=state,
        provider=base,
        response=response,
        kpts=kpts,
        kweights=kweights,
        occupations=occupations,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        overlap=overlap,
        q_frac=q_frac,
        orbital_to_atom=orbital_to_atom,
        natom=natom,
        taus_atom=taus_atom,
        phis_atom=phis_atom,
        alpha_atom=alpha_atom,
        efermi=float(state.efermi),
        width=width,
        nel=nel,
        provenance=provenance,
        field_symmetry=field_symmetry,
        density_discrepancy=discrepancy,
        path=str(path),
        density_tol=float(density_tol),
        field_tol=float(field_tol),
    )


def _max_transverse_moment(rho):
    """Max per-orbital ``|<sigma_y>|`` of an interleaved density matrix."""
    rho = np.asarray(rho)
    norb = rho.shape[0] // 2
    worst = 0.0
    for mu in range(norb):
        sl = slice(2 * mu, 2 * mu + 2)
        block = rho[sl, sl]
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        worst = max(worst, abs(float(np.trace(block @ sy).real)))
    return worst


# ---------------------------------------------------------------------------
# matched-q record gates (certified slope passthrough)
# ---------------------------------------------------------------------------


def _verify_matched_q_ledger(record, bundle, slope):
    """Reject incomplete or internally inconsistent physical-slope records."""

    def require(condition):
        if not condition:
            raise ValueError("inconsistent certified leg")

    try:
        protocol = record["protocol"]
        mesh = np.asarray(protocol["kpts"], dtype=float)
        weights = np.asarray(protocol["kweights"], dtype=float)
        steps = np.asarray(record["steps"], dtype=float)
        legs = record["legs"]
        require(record["slope_mode"] == "components")
        require(record["tolerance_stability_checked"] is True)
        require(np.isfinite(float(record["tolerance_stability_residual"])))
        require(
            mesh.shape == bundle.kpts.shape
            and np.allclose(mesh, bundle.kpts, atol=1e-12)
        )
        require(
            weights.shape == bundle.kweights.shape
            and np.allclose(weights, bundle.kweights, atol=1e-12)
        )
        require(steps.shape == (2,) and np.all(np.isfinite(steps)))
        require(steps[0] > 0 and steps[1] == steps[0] / 2)
        require(len(legs) == 24)
        checked = {}
        for leg in legs:
            axis = np.asarray(leg["axis_vector"], dtype=float)
            require(axis.shape == (3,) and np.all(np.isfinite(axis)))
            index = int(np.argmax(np.abs(axis)))
            require(np.array_equal(axis, np.eye(3)[index]))
            q = np.asarray(leg["q_frac"], dtype=float)
            step = float(leg["step"])
            require(any(np.isclose(abs(step), h, rtol=0, atol=1e-14) for h in steps))
            require(
                q.shape == (3,)
                and np.allclose(q, bundle.q_frac + step * axis, atol=1e-12)
            )
            require(leg["pass_label"] in ("base", "tightened"))
            require(leg["converged"] is True and leg["energy_certified"] is True)
            require(np.isfinite(float(leg["spectral_density_deviation"])))
            require(float(leg["spectral_density_deviation"]) <= 1e-6)
            require(not any(abs(float(x)) > 1e-8 for x in leg["constraint_residuals"]))
            energy = float(leg["free_energy"])
            components = leg["energy_components"]
            require(
                np.isfinite(energy)
                and np.isclose(energy, float(components["free_energy"]), atol=1e-10)
            )
            rebuilt = (
                float(components["band_energy"])
                - float(components["constraint_potential_expectation"])
                + float(components["hubbard_energy"])
                + float(components["multiplier_residual_energy"])
                - float(components["entropy_term"])
            )
            require(np.isfinite(rebuilt) and np.isclose(energy, rebuilt, atol=1e-9))
            key = (leg["pass_label"], index, step)
            require(key not in checked)
            checked[key] = energy
        for axis in range(3):
            h = float(steps[1])
            value = (checked[("base", axis, h)] - checked[("base", axis, -h)]) / (2 * h)
            require(np.isclose(value, slope[axis], rtol=0.0, atol=1e-8))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise SpiralResponseMatchedQError(
            "matched-q record lacks a consistent certified per-leg energy ledger"
        ) from exc


def parse_matched_q_record(record, bundle):
    """Validate a TBUpy matched-q pitch record against the bundle.

    Accepted record = the serialized ``MatchedQPitchResult.as_dict()``:
    keys ``supported``, ``self_consistent_pitch_slope``, ``q_center``/
    ``q_direction``, ``protocol`` (with ``nel``, ``width``,
    ``field_policy``), ``unsupported_reason``.  Fails closed on
    unsupported records (reason passed through verbatim), absent
    certified slopes, q/electron/width provenance mismatch, or a
    non-co-rotating field policy.
    """
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text())
    for key in ("supported", "self_consistent_pitch_slope", "protocol"):
        _require(
            key in record,
            SpiralResponseMatchedQError,
            f"matched-q record missing key {key!r}",
        )
    _require(
        bool(record["supported"]),
        SpiralResponseMatchedQError,
        "matched-q record unsupported: "
        f"{record.get('unsupported_reason') or 'no reason given'}",
    )
    slope = record["self_consistent_pitch_slope"]
    _require(
        slope is not None,
        SpiralResponseMatchedQError,
        "matched-q record carries no certified self_consistent_pitch_slope",
    )
    slope = np.asarray(slope, dtype=float)
    _require(
        slope.shape == (3,),
        SpiralResponseMatchedQError,
        f"matched-q slope must have shape (3,), got {slope.shape}",
    )
    q_ref = record.get("q_center")
    _require(
        q_ref is not None
        and np.allclose(np.asarray(q_ref, dtype=float), bundle.q_frac, atol=1e-8),
        SpiralResponseMatchedQError,
        f"matched-q evaluation point {q_ref!r} disagrees with the bundle "
        f"q_frac {bundle.q_frac.tolist()}",
    )
    protocol = record["protocol"]
    _require(
        protocol.get("field_policy") == "co_rotating",
        SpiralResponseFieldSymmetryError,
        f"matched-q field_policy {protocol.get('field_policy')!r} is not "
        "'co_rotating'",
    )
    _require(
        abs(float(protocol["nel"]) - bundle.nel) <= ELECTRON_COUNT_TOL,
        SpiralResponseProvenanceError,
        f"matched-q protocol nel {protocol.get('nel')!r} disagrees with the "
        f"bundle electron count {bundle.nel}",
    )
    _require(
        "width" in protocol and abs(float(protocol["width"]) - bundle.width) <= 1e-8,
        SpiralResponseProvenanceError,
        f"matched-q protocol width {protocol.get('width')!r} disagrees with "
        f"the bundle width {bundle.width}",
    )
    functional = bundle.provenance["hubbard"]
    _require(
        (protocol.get("hubbard_dict") or {}) == (functional["hubbard_dict"] or {})
        and protocol.get("hubbard_type") == functional["hubbard_type"]
        and protocol.get("dc_type") == functional["dc_type"],
        SpiralResponseMatchedQError,
        "matched-q Hubbard/DC functional disagrees with the frozen reference",
    )
    _verify_matched_q_ledger(record, bundle, slope)
    return {
        "slope": slope.tolist(),
        "units": PITCH_UNITS,
        "source": "tbupy matched-q record (certified free-energy slope)",
        "q_center": [float(x) for x in q_ref],
        "vector_scope_statement": record.get("vector_scope_statement"),
        "protocol": protocol,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _frozen_fd_report(bundle, gradient, pitch, step=1e-5):
    """Independent exact-angle and q central differences at fixed occupations."""
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    n = 2 * len(bundle.orbital_to_atom)

    def energy(provider, extra=None):
        eigenvalues, _ = reference_eigenpairs(provider, bundle.kpts, extra=extra)
        return float(
            np.sum(bundle.kweights[:, None] * bundle.occupations * eigenvalues)
        )

    local = {"beta": np.zeros(bundle.natom), "delta": np.zeros(bundle.natom)}
    for atom in range(bundle.natom):
        for channel, direction in (("beta", sigma_x), ("delta", sigma_y)):

            def rotated(angle):
                perturbation = np.zeros((n, n), dtype=complex)
                for orbital, owner in enumerate(bundle.orbital_to_atom):
                    if owner == atom:
                        block = slice(2 * orbital, 2 * orbital + 2)
                        field = 0.5 * bundle.state.B_local[orbital]
                        perturbation[block, block] = field * (
                            (np.cos(angle) - 1.0) * sigma_z + np.sin(angle) * direction
                        )
                return energy(bundle.response, perturbation)

            local[channel][atom] = (rotated(step) - rotated(-step)) / (2 * step)
    pitch_fd = np.zeros(3)
    for component in range(3):
        displacement = np.zeros(3)
        displacement[component] = step
        plus = energy(bundle.response_provider(q_frac=bundle.q_frac + displacement))
        minus = energy(bundle.response_provider(q_frac=bundle.q_frac - displacement))
        pitch_fd[component] = (plus - minus) / (2 * step)
    local_residual = max(
        np.max(np.abs(local["beta"] - gradient.g_beta)),
        np.max(np.abs(local["delta"] - gradient.g_delta)),
    )
    pitch_residual = np.max(np.abs(pitch_fd - pitch.vector))
    return {
        "step": step,
        "occupation_protocol": "fixed reference f_nk and frozen periodic potentials",
        "local_gradient_fd": {name: values.tolist() for name, values in local.items()},
        "pitch_fd": pitch_fd.tolist(),
        "local_gradient_max_abs_eV_per_rad": float(local_residual),
        "pitch_max_abs_eV_per_fractional_q": float(pitch_residual),
    }


def _curvature_diagnostics(bundle, gradient, curvature):
    """Attach symmetry-conditional Ward and pair-model comparisons."""
    from TB2J.spiral_nonstationary import (
        j_fit_report,
        pair_once_beta,
        ward_report,
    )

    orbital_gradient = np.asarray(gradient.g_beta_orbital).reshape(-1, 2).sum(axis=1)
    site_gradient = np.tile(orbital_gradient, curvature.ncell)
    translations = getattr(curvature, "translations", None)
    if translations is None:
        translations = np.column_stack(
            (np.arange(curvature.ncell), np.zeros((curvature.ncell, 2)))
        )
    translations = np.asarray(translations, dtype=float)
    theta = np.array(
        [
            2 * np.pi * np.dot(bundle.q_frac, R + bundle.state.taus[orbital])
            + bundle.state.phis[orbital]
            for R in translations
            for orbital in range(len(bundle.orbital_to_atom))
        ]
    )
    constraint = bundle.provenance["constraint"]
    held_fixed = any(
        bool(site.get("hold_fixed")) or site.get("rotation") == "lab_fixed"
        for site in constraint.get("sites", [])
    )
    symmetry = "lab_field" if held_fixed else "co_rotating"
    ward = ward_report(curvature, site_gradient, thetas=theta, symmetry=symmetry)
    ward = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in ward.items()
    }
    if held_fixed:
        pair = {
            "applicable": False,
            "reason": "fixed laboratory constraint breaks rotation symmetry",
        }
    else:
        predicted = pair_once_beta(curvature, theta)
        mismatch = site_gradient - predicted
        fit = j_fit_report(site_gradient, theta)
        pair = {
            "applicable": bool(bundle.provenance.get("pairwise_isotropic", False)),
            "pairwise_isotropic_assumption": "required; raw mismatch is a diagnostic, not a Heisenberg identity",
            "measured_g_beta": site_gradient.tolist(),
            "predicted_g_beta": predicted.tolist(),
            "max_abs_mismatch": float(np.max(np.abs(mismatch))),
            "fit_rank": fit["rank"],
            "n_pairs": fit["n_pairs"],
            "unique_from_one_snapshot": fit["unique"],
        }
    return ward, pair


@dataclasses.dataclass(eq=False)
class SpiralResponseResult:
    """Assembled additive response of one persisted v2 bundle."""

    bundle: ResponseBundle
    frozen_band_gradient: object
    frozen_band_pitch_slope: object
    nonstationary_curvature: object
    matched_q: dict | None = None
    finite_difference: dict | None = None
    ward: dict | None = None
    pair_model: dict | None = None


def compute_frozen_response(
    bundle,
    *,
    matched_q_record=None,
    density_tol=DEFAULT_DENSITY_TOL,
    field_tol=DEFAULT_FIELD_TOL,
    translation_cutoff=1,
):
    """Full additive response of a persisted v2 bundle.

    ``bundle`` is a sidecar path or a loaded :class:`ResponseBundle`.
    ``matched_q_record`` optionally supplies a TBUpy matched-q pitch
    record (path or dict) whose certified slope is passed through as
    ``self_consistent_pitch_slope`` -- it is never fabricated here.
    """
    if not isinstance(bundle, ResponseBundle):
        bundle = load_response_bundle(
            bundle, density_tol=density_tol, field_tol=field_tol
        )

    from TB2J.spiral_first_order import frozen_band_gradient
    from TB2J.spiral_nonstationary import nonstationary_curvature
    from TB2J.spiral_pitch import frozen_pitch_slope

    v1_beta, v1_delta = planar_field_vertices(bundle)
    gradient = frozen_band_gradient(
        bundle.eigenvalues,
        bundle.eigenvectors,
        bundle.kweights,
        bundle.occupations,
        bundle.overlap,
        bundle.orbital_to_atom,
        v1_beta,
        v1_delta,
        rho_scf=np.asarray(bundle.state.rho),
        provenance=bundle.provenance,
        density_tol=density_tol,
    )
    pitch = frozen_pitch_slope(
        bundle.response,
        kpts=bundle.kpts,
        occupations=bundle.occupations,
        kweights=bundle.kweights,
    )
    curvature = nonstationary_curvature(
        bundle.response,
        kpts=bundle.kpts,
        kweights=bundle.kweights,
        occupations=bundle.occupations,
        q_frac=bundle.q_frac,
        taus=np.asarray(bundle.state.taus, dtype=float),
        phis=np.asarray(bundle.state.phis, dtype=float),
        B_local=np.asarray(bundle.state.B_local, dtype=float),
        translation_cutoff=translation_cutoff,
        orbital_to_atom=bundle.orbital_to_atom,
    )
    finite_difference = _frozen_fd_report(bundle, gradient, pitch)
    pitch.metadata["fd_residual"] = finite_difference[
        "pitch_max_abs_eV_per_fractional_q"
    ]
    ward, pair_model = _curvature_diagnostics(bundle, gradient, curvature)
    matched = (
        parse_matched_q_record(matched_q_record, bundle)
        if matched_q_record is not None
        else None
    )
    return SpiralResponseResult(
        bundle=bundle,
        frozen_band_gradient=gradient,
        frozen_band_pitch_slope=pitch,
        nonstationary_curvature=curvature,
        matched_q=matched,
        finite_difference=finite_difference,
        ward=ward,
        pair_model=pair_model,
    )


def _unwrap(as_dict, key):
    """Flatten a peer ``as_dict`` that nests everything under ``key``."""
    return as_dict[key] if list(as_dict) == [key] else as_dict


def response_report(result):
    """JSON-ready report dict with full sign/unit/provenance metadata."""
    bundle = result.bundle
    gradient = _unwrap(result.frozen_band_gradient.as_dict(), "frozen_band_gradient")
    pitch = _unwrap(result.frozen_band_pitch_slope.as_dict(), "frozen_band_pitch_slope")
    curvature = _unwrap(
        result.nonstationary_curvature.as_dict(), "nonstationary_curvature"
    )
    # full (translation_cutoff*norb)^2 lab-angle blocks stay on the result
    # object; the JSON carries the summary (units/protocol/v2_diagonal/
    # local_curvature) plus the block shapes
    curvature_blocks = curvature.pop("blocks", None)
    if curvature_blocks is not None:
        curvature["blocks_omitted"] = True
        curvature["block_shapes"] = {
            "".join(key): list(np.asarray(val).shape)
            for key, val in curvature_blocks.items()
        }
    curvature["ward"] = result.ward
    curvature["pair_model"] = result.pair_model
    state = bundle.state
    g_norm = float(
        np.linalg.norm(np.asarray(gradient["g_beta"], dtype=float))
        + np.linalg.norm(np.asarray(gradient["g_delta"], dtype=float))
    )
    report = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "frozen_band_gradient": gradient,
        "frozen_band_pitch_slope": pitch,
        "nonstationary_curvature": curvature,
        "metadata": {
            "bundle": bundle.path,
            "schema": {"name": "tbupy_spiral_state", "version": 2},
            "provenance": bundle.provenance,
            "field_symmetry": bundle.field_symmetry,
            "finite_difference": result.finite_difference,
            "gates": {
                "density_discrepancy": bundle.density_discrepancy,
                "density_tol": bundle.density_tol,
                "max_transverse_moment": _max_transverse_moment(state.rho),
                "field_tol": bundle.field_tol,
                "policy": "fail_closed",
            },
            "model": {
                "norb": int(np.asarray(state.HR_up).shape[1]),
                "natom": int(bundle.natom),
                "nspin_interleaved": True,
                "q_frac": bundle.q_frac.tolist(),
                "B_local": np.asarray(state.B_local, dtype=float).tolist(),
                "taus_per_atom": bundle.taus_atom.tolist(),
                "phis_per_atom": bundle.phis_atom.tolist(),
                "alpha_atom": bundle.alpha_atom.tolist(),
                "atom_of_orbital": bundle.orbital_to_atom.tolist(),
                "nonorthogonal_overlap": not bundle.provider.metadata.is_orthogonal,
                "nonstationary_reference": bool(g_norm > 1e-10),
                "frozen_gradient_norm_eV_per_rad": g_norm,
            },
            "protocol": {
                "nk": int(len(bundle.kpts)),
                "nbands": int(bundle.occupations.shape[1]),
                "kweight_sum": float(bundle.kweights.sum()),
                "efermi": bundle.efermi,
                "width": bundle.width,
                "electron_count": bundle.nel,
                "occupation_rule": bundle.provenance["occupation_rule"],
                "occupations": "fixed frozen variational (no re-Fermi)",
                "potential_terms": "V_U and constraint_potential added "
                "cell-periodically (R=0), q-independent (frozen)",
            },
            "conventions": {
                "spin_basis": "interleaved (orb0 up, orb0 dn, ...)",
                "gauge": "planar_y twisted (alpha_i = 2 pi q.tau_i + phi_i)",
                "gradient": {
                    "units": GRADIENT_UNITS,
                    "sign": "+dE/d(local positive field rotation)",
                    "beta": "in-plane field rotation; folded vertex "
                    "Bf sx (plain Pauli image; alpha-independent because "
                    "the co-rotating field vertex is twist-invariant); "
                    "basis/overlap fixed",
                    "delta": "out-of-plane field tilt; folded vertex "
                    "Bf sy (sy commutes with the planar_y twist); "
                    "basis/overlap fixed",
                },
                "pitch_slope": {
                    "units": PITCH_UNITS,
                    "sign": "+dE/dq_a at fixed frozen occupations",
                    "dS_dq": "moving-basis overlap derivative included via "
                    "provider q_derivative",
                },
                "curvature": {
                    "units": CURVATURE_UNITS,
                    "sign": "+d2E/dangle^2",
                },
                "self_consistent_pitch_slope": {
                    "units": PITCH_UNITS,
                    "rule": "emitted only from a supplied, supported, "
                    "provenance-matched TBUpy matched-q record; never "
                    "fabricated",
                },
            },
            "matched_q": result.matched_q,
        },
    }
    if result.matched_q is not None:
        report["self_consistent_pitch_slope"] = {
            "slope": result.matched_q["slope"],
            "units": PITCH_UNITS,
            "source": result.matched_q["source"],
        }
    return report
