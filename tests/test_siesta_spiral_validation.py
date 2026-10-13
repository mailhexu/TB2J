"""Story010: AC-003/005 finite rotations at the approved frozen-pencil seam.

Synthetic orientations are the two onsite total-H Pauli vectors, not imposed
SCF moments. O2 retains every producer k point and its original weight.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from TB2J.interfaces.siesta_spiral import BUNDLE_VERSION, SpiralBundle
from TB2J.interfaces.siesta_spiral_io import KB_EV, load_fixture
from TB2J.interfaces.siesta_spiral_response import (
    CURVATURE_BUDGET_EV_PER_RAD2,
    ENSEMBLE_GAPPED_T0,
    RESPONSE_IDENTITY_TOL_EV,
    TORQUE_BUDGET_EV_PER_RAD,
    EnergyContour,
    Ensemble,
    FrozenResponseProblem,
    ResponseSet,
    SpiralResponseError,
    evaluate_response_set,
)
from TB2J.interfaces.siesta_spiral_validation import validate_finite_rotations
from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex

ANGLES = (8e-3, 4e-3, 2e-3, 1e-3)
FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_o2.npz"


def _frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    x = np.cross((0.0, 1.0, 0.0), normal)
    x /= np.linalg.norm(x)
    return {"x": tuple(x), "y": tuple(np.cross(normal, x))}


def _synthetic(collapsed=False, closing_gap=False):
    sparse = {
        (0, 0, 0): (2.0, 0.5, 0.3, 0.1),
        (1, 1, 0): (1.5, -0.5, -0.2, 0.05),
        (0, 1, 0): (0.05, -0.02, 0.15, -0.08),
        (1, 0, 0): (0.05, -0.02, 0.15, -0.08),
    }
    if collapsed:
        sparse = {(0, 0, 0): (-2.0, -2.0, 0.0, 0.0), (1, 1, 0): (2.0, 2.0, 0.0, 0.0)}
    if closing_gap:
        # AF onsite +/-sigma_z and spin-scalar hopping t=1. At a pi rotation
        # of either site the aligned spectrum is {-2, 0, 0, 2}: exact closure.
        sparse = {
            (0, 0, 0): (1.0, -1.0, 0.0, 0.0),
            (1, 1, 0): (-1.0, 1.0, 0.0, 0.0),
            (0, 1, 0): (1.0, 1.0, 0.0, 0.0),
            (1, 0, 0): (1.0, 1.0, 0.0, 0.0),
        }
    bundle = SpiralBundle(
        version=BUNDLE_VERSION,
        norb=2,
        ncell=1,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse=sparse,
        overlap={
            (0, 0, 0): 1.0,
            (1, 1, 0): 1.0,
            (0, 1, 0): 0.0 if closing_gap else 0.05,
            (1, 0, 0): 0.0 if closing_gap else 0.05,
        },
    )
    frames = {"A": _frame((0.3, 0.1, 0.75)), "B": _frame((-0.2, 0.05, 1.0))}
    if closing_gap:
        frames = {"A": _frame((0.0, 0.0, 1.0)), "B": _frame((0.0, 0.0, -1.0))}
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    problem = FrozenResponseProblem(
        bundle, vertex, axes_by_site={s: tuple(f.items()) for s, f in frames.items()}
    )
    values = eigh(*problem.pencil(0.0), eigvals_only=True)
    ensemble = Ensemble(
        ENSEMBLE_GAPPED_T0,
        2,
        2.0,
        float((values[1] + values[2]) / 2),
        0.0,
        float(values[2] - values[1]),
    )
    response = _response(problem, ensemble, [0.0], [1.0], float(values[0]), 256, 0.4)
    normals = {s: tuple(np.cross(*f.values())) for s, f in frames.items()}
    return (
        problem,
        response,
        dict(
            axes_by_site=frames,
            ensemble=ensemble,
            reference_directions_by_site=normals,
            kpoints=[0.0],
            k_weights=[1.0],
            angle_ladder=ANGLES,
        ),
    )


def _response(problem, ensemble, kpoints, weights, occupied_min, nodes, margin):
    coords = tuple((s, a) for s in problem.site_order for a in ("x", "y"))
    contour = EnergyContour.declared(
        z0=ensemble.fermi_energy_ev,
        occupied_min=occupied_min,
        gap_ev=ensemble.gap_ev,
        margin_ev=margin,
        n_nodes=nodes,
    )
    return evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=kpoints,
        k_weights=weights,
        contour=contour,
        finite_steps=(),
    )


def _assert_stable(result, raw):
    summary = result.finite_rotation_cross_check
    assert summary["passed"] is True
    assert summary["stable_error_regime"]["gradient"] is True
    assert summary["stable_error_regime"]["curvature"] is True
    assert summary["stable_windows"]["gradient"]
    assert summary["stable_windows"]["curvature"]
    for rung in summary["rungs"]:
        assert rung["gradient_max_abs_mismatch"] <= TORQUE_BUDGET_EV_PER_RAD
        assert rung["curvature_max_abs_mismatch"] <= CURVATURE_BUDGET_EV_PER_RAD2
        assert set(rung["gradient"]) == set(raw.to_dict()["gradient"])
        assert set(rung["curvature"]) == set(raw.to_dict()["curvature"])
        for pair, finite in rung["curvature"].items():
            assert finite == pytest.approx(raw.to_dict()["curvature"][pair], abs=1e-3)
    assert result.gradient == raw.gradient
    assert result.torque == raw.torque
    assert result.curvature == raw.curvature
    assert result.curvature_split == raw.curvature_split
    assert result.torque_state == raw.torque_state
    assert summary["torque_state"] == raw.torque_state
    assert summary["reference_torque"] == raw.to_dict()["torque"]
    assert summary["previous_cross_check"] == raw.finite_rotation_cross_check
    raw_payload, result_payload = raw.to_dict(), result.to_dict()
    assert summary["raw_response_sha256"] == raw_payload["content_sha256"]
    for name, value in raw_payload.items():
        if name not in ("finite_rotation_cross_check", "content_sha256"):
            assert result_payload[name] == value
    assert ResponseSet.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_inequivalent_frames_all_gradients_and_intra_inter_site_curvature():
    problem, raw, kwargs = _synthetic()
    before = raw.to_dict()
    frames = kwargs["axes_by_site"]
    assert (
        np.dot(np.cross(*frames["A"].values()), np.cross(*frames["B"].values())) < 0.95
    )
    assert raw.torque_state["nonzero_reference_torque"] is True
    assert max(abs(v) for v in raw.torque.values()) > 1e-5
    result = validate_finite_rotations(problem, raw, **kwargs)
    _assert_stable(result, raw)
    assert raw.to_dict() == before
    assert result.finite_rotation_cross_check["angle_ladder_rad"] == list(ANGLES)
    for site in ("A", "B"):
        assert result.finite_rotation_cross_check["frames"][site]["axes"] == {
            label: list(axis) for label, axis in frames[site].items()
        }
    # Independent rotations retain intersite couplings, rather than validating
    # only an accidentally zero or site-diagonal curvature.
    cross = [v for (a, b), v in raw.curvature.items() if a[0] != b[0]]
    assert max(abs(v) for v in cross) > 1e-5


def test_uniform_q0_o2_rotation_is_one_global_spin_rotation():
    path = (
        Path(__file__).parent
        / "siesta_spiral_fixtures"
        / "siesta_collinear_o2"
        / "o2_fm_q0.npz"
    )
    bundle, reference = load_fixture(path)
    assert np.allclose(bundle.q, 0.0)
    n = bundle.norb
    vertex = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={"O1": range(n // 2), "O2": range(n // 2, n)},
    )
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"O1": (("y", (0, 1, 0)),), "O2": (("y", (0, 1, 0)),)},
    )
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=12,
        electron_count=12.0,
        fermi_energy_ev=reference["ef"],
        smearing_width_ev=0.0,
        gap_ev=2.49684889,
    )
    angle = 0.1
    both = {"O1": {"y": angle}, "O2": {"y": angle}}
    u = np.kron(
        np.array(
            [
                [np.cos(angle / 2), -np.sin(angle / 2)],
                [np.sin(angle / 2), np.cos(angle / 2)],
            ]
        ),
        np.eye(n),
    )
    for k in reference["kp"]:
        h, s = problem.pencil(k)
        rotated, unchanged_s = problem.rotated_pencil(k, both)
        np.testing.assert_allclose(rotated, u @ h @ u.conj().T, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(unchanged_s, s)
    energy = problem.occupied_energy(
        reference["kp"], reference["kp_weight"], {}, ensemble
    )
    uniform = problem.occupied_energy(
        reference["kp"], reference["kp_weight"], both, ensemble
    )
    local = problem.occupied_energy(
        reference["kp"], reference["kp_weight"], {"O1": {"y": angle}}, ensemble
    )
    assert abs(uniform - energy) < 1e-8
    assert abs(local - energy) > 1e-4


def test_full_four_point_o2_producer_mesh():
    bundle, reference = load_fixture(FIXTURE)
    assert len(reference["kp"]) == 4
    per_site = bundle.norb // 2
    vertex = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={
            "O1": list(range(per_site)),
            "O2": list(range(per_site, bundle.norb)),
        },
    )
    # Native Mulliken atomic populations, o2_all_states.out lines 540-541.
    # Both sites are almost +x; global x/y would include a longitudinal axis.
    moment = np.array((0.999975, 0.001691, 0.0))
    normal = tuple(moment / np.linalg.norm(moment))
    normals = {s: normal for s in ("O1", "O2")}
    frames = {s: _frame(normal) for s in ("O1", "O2")}
    problem = FrozenResponseProblem(
        bundle, vertex, axes_by_site={s: tuple(f.items()) for s, f in frames.items()}
    )
    occupied = int(round(reference["electron_count"]))
    assert occupied == 12
    homo = float(np.max(reference["eig"][:, occupied - 1]))
    lumo = float(np.min(reference["eig"][:, occupied]))
    ensemble = Ensemble(
        ENSEMBLE_GAPPED_T0,
        occupied,
        reference["electron_count"],
        reference["ef"],
        KB_EV * reference["temperature"],
        lumo - homo,
    )
    kpoints = reference["kp"].tolist()
    weights = reference["kp_weight"].tolist()
    raw = _response(
        problem,
        ensemble,
        kpoints,
        weights,
        float(np.min(reference["eig"][:, :occupied])),
        1024,
        0.8,
    )
    assert raw.contour_vs_band_max_abs <= RESPONSE_IDENTITY_TOL_EV
    result = validate_finite_rotations(
        problem,
        raw,
        axes_by_site=frames,
        ensemble=ensemble,
        reference_directions_by_site=normals,
        kpoints=kpoints,
        k_weights=weights,
        angle_ladder=ANGLES,
    )
    _assert_stable(result, raw)
    summary = result.finite_rotation_cross_check
    assert summary["mesh"]["kpoints"] == kpoints
    np.testing.assert_allclose(summary["mesh"]["k_weights"], weights, atol=1e-14)
    assert summary["reference_gap_ev"] > 1.0
    assert len(result.curvature) == 16
    assert result.ensemble["occupations_per_k"] == [[1] * 12 + [0] * 24] * 4


def test_stable_but_wrong_analytic_gradient_does_not_pass():
    problem, raw, kwargs = _synthetic()
    gradient = dict(raw.gradient)
    gradient[raw.coords[0]] += 0.05
    torque = {c: -g for c, g in gradient.items()}
    wrong = replace(raw, gradient=gradient, torque=torque)
    result = validate_finite_rotations(problem, wrong, **kwargs)
    summary = result.finite_rotation_cross_check
    assert summary["passed"] is False
    assert summary["stable_error_regime"]["gradient"] is False
    assert summary["stable_error_regime"]["curvature"] is True
    assert summary["rungs"][-1]["gradient_max_abs_mismatch"] > 0.04


def test_finite_difference_floor_is_not_convergence():
    problem, raw, kwargs = _synthetic()
    result = validate_finite_rotations(
        problem,
        raw,
        **{**kwargs, "angle_ladder": (4e-9, 2e-9, 1e-9)},
    )
    summary = result.finite_rotation_cross_check
    assert summary["passed"] is False
    assert summary["stable_error_regime"]["curvature"] is False
    assert all(
        r["curvature_roundoff_floor_ev_per_rad2"] > 1e-3 for r in summary["rungs"]
    )
    assert "finite_difference_floor" in summary["failure_reasons"]


def test_collapsed_magnetic_operator_cannot_certify_physical_rotations():
    problem, raw, kwargs = _synthetic(collapsed=True)
    with pytest.raises(SpiralResponseError, match="collapse|no physical"):
        validate_finite_rotations(problem, raw, **kwargs)


def test_gap_closure_on_a_displaced_stencil_is_refused():
    problem, raw, kwargs = _synthetic(closing_gap=True)
    assert kwargs["ensemble"].gap_ev == pytest.approx(2 * np.sqrt(2))
    with pytest.raises(SpiralResponseError, match="gap closed"):
        validate_finite_rotations(
            problem,
            raw,
            **{**kwargs, "angle_ladder": (np.pi, np.pi / 2, np.pi / 4)},
        )


def test_left_handed_frame_and_longitudinal_axes_are_refused():
    problem, raw, kwargs = _synthetic()
    normal = kwargs["reference_directions_by_site"]["A"]
    with pytest.raises(SpiralResponseError, match="right-handed"):
        validate_finite_rotations(
            problem,
            raw,
            **{
                **kwargs,
                "reference_directions_by_site": {
                    **kwargs["reference_directions_by_site"],
                    "A": tuple(-np.array(normal)),
                },
            },
        )
    frames = {
        **kwargs["axes_by_site"],
        "A": {"x": normal, "y": kwargs["axes_by_site"]["A"]["y"]},
    }
    with pytest.raises(SpiralResponseError, match="transverse"):
        validate_finite_rotations(problem, raw, **{**kwargs, "axes_by_site": frames})


def test_stable_but_wrong_intersite_curvature_does_not_pass():
    problem, raw, kwargs = _synthetic()
    a, b = ("A", "x"), ("B", "y")
    curvature = {
        **raw.curvature,
        (a, b): raw.curvature[a, b] + 0.05,
        (b, a): raw.curvature[b, a] + 0.05,
    }
    result = validate_finite_rotations(
        problem, replace(raw, curvature=curvature), **kwargs
    )
    summary = result.finite_rotation_cross_check
    assert summary["passed"] is False
    assert summary["stable_error_regime"]["gradient"] is True
    assert summary["stable_error_regime"]["curvature"] is False
    assert summary["rungs"][-1]["curvature_max_abs_mismatch"] > 0.04


@pytest.mark.parametrize(
    "bad",
    [
        {"x": (2.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0)},
        {"x": (1.0, 0.0, 0.0), "y": (1.0, 0.0, 0.0)},
        {"x": (float("nan"), 0.0, 0.0), "y": (0.0, 1.0, 0.0)},
        {"x": (1.0, 0.0, 0.0)},
    ],
)
def test_bad_or_mismatched_local_frames_are_refused(bad):
    problem, raw, kwargs = _synthetic()
    frames = {**kwargs["axes_by_site"], "A": bad}
    with pytest.raises(SpiralResponseError, match="frame|axes"):
        validate_finite_rotations(problem, raw, **{**kwargs, "axes_by_site": frames})


def test_wrong_declared_gap_and_different_mesh_are_refused():
    problem, raw, kwargs = _synthetic()
    wrong_gap = replace(kwargs["ensemble"], gap_ev=kwargs["ensemble"].gap_ev + 0.1)
    with pytest.raises(SpiralResponseError, match="ensemble|gap"):
        validate_finite_rotations(problem, raw, **{**kwargs, "ensemble": wrong_gap})
    with pytest.raises(SpiralResponseError, match="mesh"):
        validate_finite_rotations(problem, raw, **{**kwargs, "kpoints": [0.1]})


@pytest.mark.parametrize(
    "angles",
    [
        (0.004,),
        (0.004, 0.002),
        (0.002, 0.004, 0.001),
        (0.004, 0.004, 0.001),
        (0.004, 0.002, 0.0),
    ],
)
def test_angle_sequence_requires_three_distinct_decreasing_positive_rungs(angles):
    problem, raw, kwargs = _synthetic()
    with pytest.raises(SpiralResponseError, match="angle ladder"):
        validate_finite_rotations(problem, raw, **{**kwargs, "angle_ladder": angles})
