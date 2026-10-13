"""Independent finite-rotation validation of a frozen response (Story010).

Consumer seam: ``validate_finite_rotations(problem, response, ...)`` returns
an enriched, detached ResponseSet. No derivative/analytic observable helper is
called: energies are sums of occupied generalized eigenvalues of independently
rotated pencils. The declared incident-block ordered same-U rotation seam is
reused; the full assembled overlap and spin-scalar blocks remain frozen.

The default approved practical ladder is 8, 4, 2, 1 milliradians. A stable
regime requires THREE consecutive rungs with every component's analytic error
and both adjacent changes inside the 1e-3 torque/curvature budgets. Estimated
floating-point subtraction floors must also be below those budgets. This is
not a pitch fit, a stationarity assertion, or a metallic-ensemble check.
"""

import json
from dataclasses import replace

import numpy as np
from scipy.linalg import eigh

from .siesta_spiral_response import (
    CURVATURE_BUDGET_EV_PER_RAD2,
    ENSEMBLE_GAP_TOL_EV,
    RESPONSE_IDENTITY_TOL_EV,
    TORQUE_BUDGET_EV_PER_RAD,
    ResponseSet,
    SpiralResponseError,
)

DEFAULT_ANGLE_LADDER_RAD = (8e-3, 4e-3, 2e-3, 1e-3)
_FRAME_TOL = 1e-10
_GAP_MIN_EV = 1e-8


def _coord_key(coord):
    return json.dumps(list(coord), separators=(",", ":"))


def _pair_key(a, b):
    return json.dumps([list(a), list(b)], separators=(",", ":"))


def _unit_vector(value, name):
    vector = np.asarray(value, dtype=float)
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or abs(np.linalg.norm(vector) - 1.0) > _FRAME_TOL
    ):
        raise SpiralResponseError(f"local frame {name} must be a finite unit 3-vector")
    return vector


def _frames(problem, response, axes_by_site, reference_directions_by_site):
    sites = problem.site_order
    if (
        tuple(response.site_order) != sites
        or set(axes_by_site) != set(sites)
        or set(reference_directions_by_site) != set(sites)
    ):
        raise SpiralResponseError(
            "local frames must cover exactly the declared ordered sites"
        )
    frames, coords = {}, []
    for site in sites:
        axes = dict(axes_by_site[site])
        if len(axes) != 2:
            raise SpiralResponseError(
                f"local frame for {site!r} requires two transverse axes"
            )
        normal = _unit_vector(reference_directions_by_site[site], f"normal of {site}")
        vectors = [
            _unit_vector(v, f"axis {label} of {site}") for label, v in axes.items()
        ]
        frame = np.column_stack((*vectors, normal))
        if (
            not np.allclose(frame.T @ frame, np.eye(3), atol=_FRAME_TOL, rtol=0)
            or abs(np.linalg.det(frame) - 1.0) > _FRAME_TOL
        ):
            raise SpiralResponseError(
                f"local frame for {site!r} must be orthonormal, transverse to the "
                "declared reference direction, and right-handed"
            )
        if set(response.axes.get(site, {})) != set(axes):
            raise SpiralResponseError("local frame axes differ from ResponseSet axes")
        for label, axis in axes.items():
            coord = (site, label)
            if not np.allclose(
                axis, problem.axis_of(coord), atol=_FRAME_TOL, rtol=0
            ) or not np.allclose(
                axis, response.axes[site][label], atol=_FRAME_TOL, rtol=0
            ):
                raise SpiralResponseError(
                    "local frame axes differ from the frozen problem/response"
                )
            coords.append(coord)
        frames[site] = {
            "axes": {a: list(v) for a, v in axes.items()},
            "reference_direction": normal.tolist(),
            "determinant": float(np.linalg.det(frame)),
        }
    if tuple(coords) != tuple(response.coords):
        raise SpiralResponseError(
            "local frames must cover every response coordinate in declared order"
        )
    return tuple(coords), frames


def _mesh(kpoints, k_weights):
    pts = np.asarray(kpoints, dtype=float)
    weights = np.asarray(k_weights, dtype=float)
    if (
        pts.ndim not in (1, 2)
        or len(pts) == 0
        or not np.isfinite(pts).all()
        or weights.shape != (len(pts),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise SpiralResponseError("invalid finite-rotation mesh/weights")
    return pts, weights / weights.sum()


def _max_error(values, other):
    return max(abs(values[k] - other[k]) for k in values)


def _stable_windows(rungs, field, budget):
    windows = []
    for end in range(2, len(rungs)):
        window = rungs[end - 2 : end + 1]
        if all(
            r[field + "_max_abs_mismatch"] <= budget
            and r[field + "_roundoff_floor"] <= budget
            for r in window
        ) and all(r[field + "_max_delta_vs_previous"] <= budget for r in window[1:]):
            windows.append(
                {
                    "rung_indices": list(range(end - 2, end + 1)),
                    "angles_rad": [r["angle_rad"] for r in window],
                }
            )
    return windows


def validate_finite_rotations(
    problem,
    response,
    *,
    axes_by_site,
    reference_directions_by_site,
    ensemble,
    kpoints,
    k_weights,
    angle_ladder=DEFAULT_ANGLE_LADDER_RAD,
):
    """Validate all local gradients and intra/inter-site curvature components.

    ``axes_by_site`` maps each site to TWO ordered label/unit-vector pairs (or
    an insertion-ordered mapping). ``reference_directions_by_site`` supplies
    explicit unit reference moments; axes must be perpendicular to each moment
    and together with it form a right-handed frame. Nothing is inferred from
    global x/y, or from possibly parallel real-material orientations.

    The mesh, ensemble, vertex identity and axes must match the raw ResponseSet.
    Invalid frames, collapsed magnetic operators, and gap closure raise
    SpiralResponseError. Numerical disagreement or a finite-difference floor
    returns an evidence-bearing failed verdict, never a fabricated convergence.
    Only ``finite_rotation_cross_check`` changes; raw torques, curvature,
    nonstationarity state and original cross-check remain available verbatim.
    Serialize the result with ResponseSet.to_dict() (including its digest).
    """
    ensemble.validate()
    # Snapshot consumer data, preserving all raw fields and preventing aliasing.
    raw = ResponseSet.from_dict(response.to_dict())
    coords, frames = _frames(problem, raw, axes_by_site, reference_directions_by_site)
    angles = np.asarray(tuple(angle_ladder), dtype=float)
    if (
        angles.ndim != 1
        or len(angles) < 3
        or not np.isfinite(angles).all()
        or np.any(angles <= 0)
        or np.any(np.diff(angles) >= 0)
    ):
        raise SpiralResponseError(
            "angle ladder requires at least three strictly decreasing positive rungs"
        )
    pts, weights = _mesh(kpoints, k_weights)
    mesh = {"kpoints": pts.tolist(), "k_weights": weights.tolist()}
    recorded_pts, recorded_weights = _mesh(
        raw.integration["kpoints"], raw.integration["k_weights"]
    )
    if (
        pts.shape != recorded_pts.shape
        or not np.array_equal(pts, recorded_pts)
        or not np.allclose(weights, recorded_weights, atol=1e-14, rtol=0)
    ):
        raise SpiralResponseError(
            "finite-rotation mesh differs from the analytic ResponseSet mesh"
        )
    for name, value in ensemble.to_record().items():
        if raw.ensemble.get(name) != value:
            raise SpiralResponseError(
                f"finite-rotation ensemble {name} differs from ResponseSet"
            )
    certificate = problem.vertex.certificate
    for name in ("source_bundle_sha256", "vertex_sha256"):
        if raw.provenance.get(name) != certificate.get(name):
            raise SpiralResponseError(
                "ResponseSet is not bound to this frozen problem's vertex"
            )
    if (
        not np.isfinite(raw.contour_vs_band_max_abs)
        or raw.contour_vs_band_max_abs > RESPONSE_IDENTITY_TOL_EV
    ):
        raise SpiralResponseError(
            "ResponseSet has not passed the contour/eigenbasis identity gate"
        )
    gradients = {_coord_key(c): raw.gradient[c] for c in coords}
    curvatures = {_pair_key(a, b): raw.curvature[a, b] for a in coords for b in coords}
    if not np.isfinite(list(gradients.values()) + list(curvatures.values())).all():
        raise SpiralResponseError("analytic response contains nonfinite components")
    baselines = [problem.pencil(k) for k in pts]
    nocc = ensemble.n_occupied
    dim = baselines[0][0].shape[0]
    if not 0 < nocc < dim:
        raise SpiralResponseError(
            "gapped ensemble must leave occupied and unoccupied bands"
        )
    norb = dim // 2

    def energy(parameters):
        thetas = {}
        for (site, label), angle in parameters.items():
            thetas.setdefault(site, {})[label] = angle
        total, scale, homo, lumo = 0.0, 0.0, -np.inf, np.inf
        for k, weight, (h0, s0) in zip(pts, weights, baselines):
            h, s = problem.rotated_pencil(k, thetas)
            if not np.array_equal(s, s0):
                raise SpiralResponseError(
                    "finite rotation changed the frozen full overlap"
                )
            scalar = h[:norb, :norb] + h[norb:, norb:]
            scalar0 = h0[:norb, :norb] + h0[norb:, norb:]
            if not np.allclose(scalar, scalar0, atol=1e-10, rtol=0):
                raise SpiralResponseError(
                    "finite rotation changed the frozen scalar Hamiltonian"
                )
            if not np.allclose(h, h.conj().T, atol=1e-8, rtol=0):
                raise SpiralResponseError(
                    "finite rotation produced a non-Hermitian Hamiltonian"
                )
            try:
                values = eigh(h, s, eigvals_only=True)
            except np.linalg.LinAlgError as exc:
                raise SpiralResponseError(
                    "finite rotation has an invalid generalized pencil"
                ) from exc
            if not np.isfinite(values).all():
                raise SpiralResponseError(
                    "finite rotation produced nonfinite generalized energies"
                )
            homo, lumo = max(homo, values[nocc - 1]), min(lumo, values[nocc])
            total += float(weight * values[:nocc].sum())
            scale += float(weight * np.abs(values[:nocc]).sum())
        gap = float(lumo - homo)
        if gap <= _GAP_MIN_EV:
            raise SpiralResponseError(
                "finite-rotation indirect gap closed; fixed occupations uncertifiable"
            )
        # Conservative floating-point energy/subtraction estimate, explicitly
        # recorded as an estimate, not a certified eigensolver error bound.
        roundoff = 32 * np.finfo(float).eps * max(1.0, scale, abs(total))
        return {
            "energy_ev": total,
            "roundoff_estimate_ev": roundoff,
            "gap_ev": gap,
            "homo_ev": float(homo),
            "lumo_ev": float(lumo),
        }

    reference = energy({})
    if abs(reference["gap_ev"] - ensemble.gap_ev) > ENSEMBLE_GAP_TOL_EV:
        raise SpiralResponseError(
            "declared ensemble gap differs from the full reference mesh"
        )
    if not reference["homo_ev"] < ensemble.fermi_energy_ev < reference["lumo_ev"]:
        raise SpiralResponseError(
            "ensemble Fermi level is outside the reference indirect gap"
        )
    # Story006 physical legitimacy, at a finite probe independent of tiny FD
    # rungs: a collapsed/commuting operator cannot certify a local response.
    rotation_change = {}
    for coord in coords:
        change = max(
            float(
                np.linalg.norm(
                    problem.rotated_pencil(k, {coord[0]: {coord[1]: 0.35}})[0] - h0
                )
            )
            for k, (h0, _s0) in zip(pts, baselines)
        )
        rotation_change[_coord_key(coord)] = change
        if change <= 1e-12:
            raise SpiralResponseError(
                f"magnetic collapse/no physical rotation for coordinate {coord!r}"
            )

    rungs = []
    for step in angles:
        samples = {(): reference}

        def sample(parameters):
            key = tuple((c, float(parameters[c])) for c in coords if c in parameters)
            if key not in samples:
                samples[key] = energy(parameters)
            return samples[key]["energy_ev"]

        gradient, curvature = {}, {}
        e0 = reference["energy_ev"]
        for coord in coords:
            ep, em = sample({coord: step}), sample({coord: -step})
            gradient[_coord_key(coord)] = float((ep - em) / (2 * step))
            curvature[_pair_key(coord, coord)] = float(
                ((ep - e0) + (em - e0)) / step**2
            )
        for i, a in enumerate(coords):
            for b in coords[i + 1 :]:
                pp = sample({a: step, b: step})
                pm = sample({a: step, b: -step})
                mp = sample({a: -step, b: step})
                mm = sample({a: -step, b: -step})
                value = float(((pp - pm) - (mp - mm)) / (4 * step**2))
                curvature[_pair_key(a, b)] = curvature[_pair_key(b, a)] = value
        max_roundoff = max(s["roundoff_estimate_ev"] for s in samples.values())
        previous = rungs[-1] if rungs else None
        rung = {
            "angle_rad": float(step),
            "gradient": gradient,
            "torque": {k: -v for k, v in gradient.items()},
            "curvature": curvature,
            "gradient_max_abs_mismatch": _max_error(gradient, gradients),
            "curvature_max_abs_mismatch": _max_error(curvature, curvatures),
            "gradient_abs_mismatch": {
                k: abs(v - gradients[k]) for k, v in gradient.items()
            },
            "curvature_abs_mismatch": {
                k: abs(v - curvatures[k]) for k, v in curvature.items()
            },
            "gradient_max_delta_vs_previous": (
                _max_error(gradient, previous["gradient"]) if previous else None
            ),
            "curvature_max_delta_vs_previous": (
                _max_error(curvature, previous["curvature"]) if previous else None
            ),
            "gradient_roundoff_floor": float(max_roundoff / step),
            "curvature_roundoff_floor": float(4 * max_roundoff / step**2),
            "curvature_roundoff_floor_ev_per_rad2": float(4 * max_roundoff / step**2),
            "minimum_rotated_gap_ev": min(s["gap_ev"] for s in samples.values()),
            "samples": [
                {"angles": {_coord_key(c): v for c, v in key}, **s}
                for key, s in samples.items()
            ],
        }
        rungs.append(rung)
    budgets = {
        "gradient": TORQUE_BUDGET_EV_PER_RAD,
        "curvature": CURVATURE_BUDGET_EV_PER_RAD2,
    }
    windows = {
        field: _stable_windows(rungs, field, budget)
        for field, budget in budgets.items()
    }
    stable = {field: bool(windows[field]) for field in budgets}
    # Require a COMMON regime for first/second order, not disjoint lucky rungs.
    common = [w for w in windows["gradient"] if w in windows["curvature"]]
    reasons = [
        field + "_no_stable_error_regime" for field in budgets if not stable[field]
    ]
    if not common and any(
        r[field + "_roundoff_floor"] > budget
        for r in rungs
        for field, budget in budgets.items()
    ):
        reasons.append("finite_difference_floor")
    if all(stable.values()) and not common:
        reasons.append("no_common_stable_error_regime")
    summary = {
        "version": "1.0",
        "method": "independent generalized occupied-energy central differences",
        "scope": "all declared per-site gradients and intra/inter-site curvature pairs",
        "angle_ladder_rad": angles.tolist(),
        "frames": frames,
        "mesh": mesh,
        "ensemble": ensemble.to_record(),
        "reference_gap_ev": reference["gap_ev"],
        "rotation_order": list(problem.site_order),
        "physical_rotation_probe_rad": 0.35,
        "physical_rotation_change_norm": rotation_change,
        "raw_response_sha256": response.to_dict()["content_sha256"],
        "source_bundle_sha256": certificate["source_bundle_sha256"],
        "vertex_sha256": certificate["vertex_sha256"],
        "budgets": {
            "torque_ev_per_rad": TORQUE_BUDGET_EV_PER_RAD,
            "curvature_ev_per_rad2": CURVATURE_BUDGET_EV_PER_RAD2,
        },
        "stable_regime_rule": "three consecutive rungs: all analytic errors, adjacent deltas and estimated FD floors within budget",
        "roundoff_estimate": "32*machine_epsilon*max(1,sum weighted absolute occupied energies); central-stencil amplification; estimate, not rigorous eigensolver bound",
        "rungs": rungs,
        "stable_windows": windows,
        "common_stable_windows": common,
        "stable_error_regime": stable,
        "passed": bool(common),
        "failure_reasons": reasons,
        "reference_gradient": gradients,
        "reference_torque": {_coord_key(c): raw.torque[c] for c in coords},
        "torque_state": raw.torque_state,
        "torque_interpretation": "signed torque=-gradient retained; nonstationarity is separate from curvature agreement",
        "previous_cross_check": raw.finite_rotation_cross_check,
    }
    return replace(raw, finite_rotation_cross_check=summary)
