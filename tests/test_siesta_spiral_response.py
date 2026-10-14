"""Story-009 frozen response core tests: primitive-cell torque and curvature.

Engine-level tests run the weighted site-owned declaration on the rational
story-008 [C2] model (exact fractions, re-derived for the additive
allocation: per block ``B(theta) = sum_p w_p U_p B U_p^dag`` with
``w_p = 1/count`` of the distinct endpoint owners; cross-site second
vertices exactly zero). Production-path tests run a spinor-form synthetic
bundle end to end in BOTH vertex operator modes (ADR-004 default
``onsite_delta`` and the recorded ``weighted_total_h`` diagnostic variant)
and an O2 fixture smoke on a reduced k subset.
"""

from __future__ import annotations

import json
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
    RESPONSE_VERSION,
    TORQUE_BUDGET_EV_PER_RAD,
    EnergyContour,
    Ensemble,
    FrozenResponseProblem,
    ResponseSet,
    SpiralResponseError,
    _payload_digest,
    contour_curvature,
    contour_gradient,
    evaluate_response_set,
    first_vertex_blocks,
    occupied_band_energy,
    pencil_curvature,
    pencil_gradient,
    second_vertex_blocks,
)
from TB2J.interfaces.siesta_spiral_vertex import (
    OPERATOR_MODE_ONSITE_DELTA,
    OPERATOR_MODE_WEIGHTED_TOTAL_H,
    PARTITION_DECLARATION_ONSITE,
    MagneticVertex,
    _content_digest,
    extract_magnetic_vertex,
)

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_o2.npz"

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0)}
COORDS = (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))

# Pinned weighted-site-owned quantities (story-008 [C2] rational model,
# exact fractions; coordinate order Ax, Ay, Bx, By).  Shared bonds enter
# every vertex at w = 1/2 per endpoint; cross-site pairs have an exactly
# zero seagull; the global-uniform sum rule
# kappa_Ay + 2 K_AyBy + kappa_By = 0 holds exactly because a rotation
# common to all sites is one global conjugation of a spin-scalar S pencil.
PINNED_GRADIENT = {
    ("A", "x"): 0,
    ("A", "y"): -5104 / 15525,
    ("B", "x"): 0,
    ("B", "y"): 5104 / 15525,
}
PINNED_CURVATURE = {
    (("A", "x"), ("A", "x")): -20906104 / 312440625,
    (("A", "x"), ("A", "y")): 0,
    (("A", "x"), ("B", "x")): 20906104 / 312440625,
    (("A", "x"), ("B", "y")): 0,
    (("A", "y"), ("A", "y")): 31540700264 / 70299140625,
    (("A", "y"), ("B", "x")): 0,
    (("A", "y"), ("B", "y")): -31540700264 / 70299140625,
    (("B", "x"), ("B", "x")): -20906104 / 312440625,
    (("B", "x"), ("B", "y")): 0,
    (("B", "y"), ("B", "y")): 31540700264 / 70299140625,
}
PINNED_ONSITE_AYAY = 49991 / 31050
PINNED_RESPONSE_AYAY = -163284096347 / 140598281250
# cross-site pair: seagull exactly zero, response kernel carries everything
PINNED_ONSITE_AYBY = 0
PINNED_RESPONSE_AYBY = -31540700264 / 70299140625


# ---------------------------------------------------------------------------
# pinned story-008 derivation model, at the general-spin-block level
# ---------------------------------------------------------------------------
def _derivation_model():
    """H0, S and incident spin blocks of the story-008 [C2] model."""
    orbital_l = np.array([[1.0, 1.0 / 4.0], [1.0 / 6.0, 1.0]])
    lower = np.kron(orbital_l, np.eye(2))
    householder = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.eye(4) - 2.0 * np.outer(householder, householder) / 30.0
    diagonal = np.diag([-3.0, -1.0, 2.0, 4.0])
    h0 = lower.T @ q @ diagonal @ q.T @ lower
    overlap = lower.T @ lower
    blocks, s_blocks, partition = {}, {}, {}
    for i in range(2):
        for j in range(2):
            blocks[(i, j)] = h0[2 * i : 2 * i + 2, 2 * j : 2 * j + 2].copy()
            s_blocks[(i, j)] = overlap[2 * i : 2 * i + 2, 2 * j : 2 * j + 2].copy()
    partition["A"] = {(0, 0), (0, 1), (1, 0)}
    partition["B"] = {(1, 1), (0, 1), (1, 0)}
    return blocks, s_blocks, partition


def _site_weights(partition):
    """w = 1/count of distinct endpoint owners, computed once per model."""
    count = {}
    for keys in partition.values():
        for key in keys:
            count[key] = count.get(key, 0) + 1
    return {key: 1.0 / count[key] for key in count}


def _weighted(blocks, weights, keys):
    """Declare the once-weighted blocks B_p = w_p B of one site."""
    return {key: weights[key] * blocks[key] for key in keys}


def _place(pieces, dim=4):
    out = np.zeros((dim, dim), dtype=complex)
    for (i, j), blk in pieces.items():
        out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += blk
    return out


def _model_observables(blocks, s_blocks, partition):
    h = _place(blocks)
    s = _place(s_blocks)
    weights = _site_weights(partition)
    vertices = {
        coord: _place(
            first_vertex_blocks(
                _weighted(blocks, weights, partition[coord[0]]),
                partition[coord[0]],
                AXES[coord[1]],
            )
        )
        for coord in COORDS
    }
    curvatures = {}
    for i, ca in enumerate(COORDS):
        for cb in COORDS[i:]:
            if ca[0] == cb[0]:
                w_mat = _place(
                    second_vertex_blocks(
                        _weighted(blocks, weights, partition[ca[0]]),
                        partition[ca[0]],
                        AXES[ca[1]],
                        AXES[cb[1]],
                    )
                )
            else:
                # additive site contributions never compose: exact zero
                w_mat = np.zeros((4, 4), dtype=complex)
            curvatures[(ca, cb)] = pencil_curvature(
                h, s, w_mat, vertices[ca], vertices[cb], 2
            )
    return h, s, vertices, curvatures


def _rotated_model_energy(blocks, s_blocks, partition, thetas):
    """Weighted site-owned contributions to the ORIGINAL blocks.

    ``thetas`` maps site -> list of (axis vector, angle) generators summed
    into one exponential chart per site.  Every block receives
    ``sum_p w_p U_p B U_p^dag`` against the original block, so the result
    is independent of the site order and a rotation common to all sites is
    exactly one global conjugation of the magnetic operator.
    """
    from scipy.linalg import expm

    owners = {}
    for site, keys in partition.items():
        for key in keys:
            owners.setdefault(key, []).append(site)
    weights = _site_weights(partition)
    unitaries = {
        site: expm(-1j * sum(angle * _pauli(axis) for axis, angle in entries) / 2.0)
        for site, entries in thetas.items()
        if entries
    }
    rotated = {}
    for key, blk in blocks.items():
        total = np.zeros_like(blk)
        for site in owners[key]:
            u = unitaries.get(site)
            total = total + (
                weights[key] * blk
                if u is None
                else weights[key] * (u @ blk @ u.conj().T)
            )
        rotated[key] = total
    values = eigh(_place(rotated), _place(s_blocks), eigvals_only=True)
    assert values[1] < 0.0 < values[2], values
    return float(values[:2].sum())


def _pauli(axis):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.diag([1.0, -1.0]).astype(complex)
    return axis[0] * sx + axis[1] * sy + axis[2] * sz


# ---------------------------------------------------------------------------
# engine level: pinned story-008 numbers
# ---------------------------------------------------------------------------
def test_engine_reproduces_pinned_fractions():
    blocks, s_blocks, partition = _derivation_model()
    h, s, vertices, curvatures = _model_observables(blocks, s_blocks, partition)

    for coord in COORDS:
        gradient = pencil_gradient(h, s, vertices[coord], 2)
        assert gradient == pytest.approx(PINNED_GRADIENT[coord], abs=1e-12)
        # torque convention tau = -gradient (weighted: tau_Ay = +0.328760...)
        assert -gradient == pytest.approx(-PINNED_GRADIENT[coord], abs=1e-12)
    assert -PINNED_GRADIENT[("A", "y")] == pytest.approx(0.3287600644122, abs=1e-11)
    # a rotation common to all sites carries no torque: g_Ay + g_By = 0
    assert PINNED_GRADIENT[("A", "y")] + PINNED_GRADIENT[("B", "y")] == 0

    assert set(curvatures) == set(PINNED_CURVATURE)
    for pair, (total, onsite, response) in curvatures.items():
        assert total == pytest.approx(PINNED_CURVATURE[pair], abs=1e-9), pair
        assert onsite + response == pytest.approx(total, abs=1e-9)
        # transverse curvature blocks are symmetric (Clairaut, story-008 [C])
        mirror = (pair[1], pair[0])
        if mirror in curvatures:
            assert curvatures[mirror][0] == pytest.approx(total, abs=1e-12)
    ayay = curvatures[(("A", "y"), ("A", "y"))]
    assert ayay[1] == pytest.approx(PINNED_ONSITE_AYAY, abs=1e-9)
    assert ayay[2] == pytest.approx(PINNED_RESPONSE_AYAY, abs=1e-9)
    ayby = curvatures[(("A", "y"), ("B", "y"))]
    # cross-site pair: additive contributions never compose, so the seagull
    # is exactly zero and the response kernel carries the whole curvature
    assert ayby[1] == pytest.approx(PINNED_ONSITE_AYBY, abs=1e-12)
    assert ayby[2] == pytest.approx(PINNED_RESPONSE_AYBY, abs=1e-9)

    # global-uniform sum rule: a common rotation is one global conjugation
    # of a spin-scalar-S pencil, so the energy is exactly invariant
    kappa_a = curvatures[(("A", "y"), ("A", "y"))][0]
    kappa_b = curvatures[(("B", "y"), ("B", "y"))][0]
    cross = curvatures[(("A", "y"), ("B", "y"))][0]
    assert kappa_a + 2.0 * cross + kappa_b == pytest.approx(0.0, abs=1e-12)


def test_engine_contour_identity_within_approved_budget():
    blocks, s_blocks, partition = _derivation_model()
    h, s, vertices, curvatures = _model_observables(blocks, s_blocks, partition)
    weights = _site_weights(partition)
    contour = EnergyContour.declared(
        z0=0.5, occupied_min=-3.0, gap_ev=3.0, margin_ev=0.4, n_nodes=256
    )

    for coord in COORDS:
        band = pencil_gradient(h, s, vertices[coord], 2)
        circ = contour_gradient(h, s, vertices[coord], contour)
        assert abs(band - circ) <= RESPONSE_IDENTITY_TOL_EV, coord
    for pair, (total, onsite, response) in curvatures.items():
        a, b = pair
        if a[0] == b[0]:
            w_mat = _place(
                second_vertex_blocks(
                    _weighted(blocks, weights, partition[a[0]]),
                    partition[a[0]],
                    AXES[a[1]],
                    AXES[b[1]],
                )
            )
        else:
            w_mat = np.zeros((4, 4), dtype=complex)
        c_total, c_onsite, c_response = contour_curvature(
            h,
            s,
            w_mat,
            vertices[a],
            vertices[b],
            contour,
        )
        assert abs(total - c_total) <= RESPONSE_IDENTITY_TOL_EV, pair
        assert abs(onsite - c_onsite) <= RESPONSE_IDENTITY_TOL_EV, pair
        assert abs(response - c_response) <= RESPONSE_IDENTITY_TOL_EV, pair


def test_engine_finite_rotation_convergence_orders():
    blocks, s_blocks, partition = _derivation_model()
    _, _, _, curvatures = _model_observables(blocks, s_blocks, partition)
    exact_tau = -PINNED_GRADIENT[("A", "y")]
    exact_k = PINNED_CURVATURE[(("A", "y"), ("A", "y"))]
    exact_cross = PINNED_CURVATURE[(("A", "y"), ("B", "y"))]

    energy0 = _rotated_model_energy(blocks, s_blocks, partition, {})

    def torque(step):
        plus = _rotated_model_energy(
            blocks, s_blocks, partition, {"A": [(AXES["y"], step)]}
        )
        minus = _rotated_model_energy(
            blocks, s_blocks, partition, {"A": [(AXES["y"], -step)]}
        )
        return -(plus - minus) / (2.0 * step)

    def curvature(step):
        plus = _rotated_model_energy(
            blocks, s_blocks, partition, {"A": [(AXES["y"], step)]}
        )
        minus = _rotated_model_energy(
            blocks, s_blocks, partition, {"A": [(AXES["y"], -step)]}
        )
        return (plus - 2.0 * energy0 + minus) / step**2

    def cross(step):
        def energy(ta, tb):
            return _rotated_model_energy(
                blocks,
                s_blocks,
                partition,
                {"A": [(AXES["y"], ta)], "B": [(AXES["y"], tb)]},
            )

        return (
            energy(step, step)
            - energy(step, -step)
            - energy(-step, step)
            + energy(-step, -step)
        ) / (4.0 * step**2)

    # a rotation common to all sites is exactly one global conjugation of
    # the spin-scalar-S pencil: the occupied energy is invariant for ANY
    # angle, not only to second order
    for angle in (0.1, 0.7):
        uniform = _rotated_model_energy(
            blocks,
            s_blocks,
            partition,
            {"A": [(AXES["y"], angle)], "B": [(AXES["y"], angle)]},
        )
        assert uniform == pytest.approx(energy0, abs=1e-10)

    steps = (8e-3, 4e-3, 2e-3)
    tau_errors = [abs(torque(h_) - exact_tau) for h_ in steps]
    k_errors = [abs(curvature(h_) - exact_k) for h_ in steps]
    cross_errors = [abs(cross(h_) - exact_cross) for h_ in steps]
    # central differences converge at the pinned O(h^2) rate (story-008 [H])
    for errors in (tau_errors, k_errors, cross_errors):
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        assert np.all((ratios > 3.0) & (ratios < 5.5)), ratios
    # user-approved budgets hold between the last two ladder rungs
    assert abs(tau_errors[-1] - tau_errors[-2]) <= TORQUE_BUDGET_EV_PER_RAD
    assert abs(k_errors[-1] - k_errors[-2]) <= CURVATURE_BUDGET_EV_PER_RAD2
    assert abs(cross_errors[-1] - cross_errors[-2]) <= CURVATURE_BUDGET_EV_PER_RAD2


def test_engine_same_site_cross_axis_seagull_is_symmetric():
    blocks, s_blocks, partition = _derivation_model()
    h, s, vertices, curvatures = _model_observables(blocks, s_blocks, partition)
    xy = curvatures[(("A", "x"), ("A", "y"))]
    yx = pencil_curvature(
        h,
        s,
        _place(
            second_vertex_blocks(
                _weighted(blocks, _site_weights(partition), partition["A"]),
                partition["A"],
                AXES["y"],
                AXES["x"],
            )
        ),
        vertices[("A", "y")],
        vertices[("A", "x")],
        2,
    )
    assert xy[0] == pytest.approx(0.0, abs=1e-12)
    assert yx[0] == pytest.approx(xy[0], abs=1e-12)
    assert xy[1] == pytest.approx(yx[1], abs=1e-12)


# ---------------------------------------------------------------------------
# production path: spinor-form synthetic bundle
# ---------------------------------------------------------------------------
def _spinor_bundle():
    return SpiralBundle(
        version=BUNDLE_VERSION,
        norb=2,
        ncell=1,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse={
            (0, 0, 0): (2.0, 0.5, 0.3, 0.1),
            (1, 1, 0): (1.5, -0.5, -0.2, 0.05),
            (0, 1, 0): (0.05, -0.02, 0.15, -0.08),
            (1, 0, 0): (0.05, -0.02, 0.15, -0.08),
        },
        overlap={(0, 0, 0): 1.0, (1, 1, 0): 1.0, (0, 1, 0): 0.05, (1, 0, 0): 0.05},
    )


def _bundle_problem(operator=OPERATOR_MODE_ONSITE_DELTA):
    bundle = _spinor_bundle()
    vertex = extract_magnetic_vertex(
        bundle, orbitals_by_site={"A": [0], "B": [1]}, operator=operator
    )
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)},
    )
    h, s = problem.pencil(0.0)
    eig = eigh(h, s, eigvals_only=True)
    return bundle, vertex, problem, eig


def test_spinor_bundle_response_set_end_to_end():
    bundle, vertex, problem, eig = _bundle_problem()
    gap = float(eig[2] - eig[1])
    assert gap > 0.5
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=float(0.5 * (eig[1] + eig[2])),
        smearing_width_ev=0.0,
        gap_ev=gap,
    )
    coords = (("A", "y"), ("B", "y"))
    contour = EnergyContour.declared(
        z0=float(0.5 * (eig[1] + eig[2])),
        occupied_min=float(eig[0]),
        gap_ev=gap,
        margin_ev=0.4,
        n_nodes=256,
    )
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=[0.0],
        k_weights=[1.0],
        contour=contour,
        finite_steps=(4e-3, 2e-3),
        contour_node_rungs=(64, 128, 256),
    )

    assert response.version == RESPONSE_VERSION
    # the record labels the rotating operator that produced every number
    assert response.operator_mode == OPERATOR_MODE_ONSITE_DELTA
    assert response.provenance["operator_mode"] == OPERATOR_MODE_ONSITE_DELTA
    assert (
        response.frozen_declarations["magnetic_amplitudes"]["certificate"][
            "operator_mode"
        ]
        == OPERATOR_MODE_ONSITE_DELTA
    )
    # torque state carried verbatim: tau == -gradient, nonzero entries kept
    for coord in coords:
        assert response.torque[coord] == pytest.approx(
            -response.gradient[coord], abs=1e-14
        )
    assert response.torque_state["nonzero_reference_torque"] is True
    assert any(abs(response.torque[c]) > 1e-6 for c in coords)
    # analytic identity between band and contour evaluation
    assert response.contour_vs_band_max_abs <= RESPONSE_IDENTITY_TOL_EV
    # curvature symmetry over ordered pairs
    a, b = coords
    assert response.curvature[(a, b)] == pytest.approx(
        response.curvature[(b, a)], abs=1e-12
    )
    split = response.curvature_split[(a, b)]
    assert split["onsite"] + split["response"] == pytest.approx(
        response.curvature[(a, b)], abs=1e-9
    )
    # ladder budgets and convergence flags
    assert response.ladder["budgets"] == {
        "torque_ev_per_rad": TORQUE_BUDGET_EV_PER_RAD,
        "curvature_ev_per_rad2": CURVATURE_BUDGET_EV_PER_RAD2,
    }
    assert response.ladder["converged"]["torque"] is True
    assert response.ladder["converged"]["curvature"] is True
    assert len(response.ladder["finite_rotation"]) == 2
    assert len(response.ladder["contour_nodes"]) == 3
    # frozen declarations and provenance
    assert (
        response.frozen_declarations["magnetic_amplitudes"]["vertex_sha256"]
        == vertex.certificate["vertex_sha256"]
    )
    assert "scalar_potential" in response.frozen_declarations
    assert (
        response.provenance["source_bundle_sha256"]
        == vertex.certificate["source_bundle_sha256"]
    )
    assert response.ensemble["n_occupied"] == 2
    assert response.ensemble["kind"] == ENSEMBLE_GAPPED_T0

    # round trip through json, with tamper detection
    clone = ResponseSet.from_dict(json.loads(json.dumps(response.to_dict())))
    assert clone == response
    tampered = response.to_dict()
    first_pair = next(iter(tampered["curvature"]))
    tampered["curvature"][first_pair] = tampered["curvature"][first_pair] + 1.0
    with pytest.raises(SpiralResponseError, match="digest"):
        ResponseSet.from_dict(tampered)
    wrong_version = response.to_dict()
    wrong_version["version"] = "0.9"
    with pytest.raises(SpiralResponseError, match="version"):
        ResponseSet.from_dict(wrong_version)
    # a serialized record of the retired ordered two-sided convention is
    # rejected by the convention gate, not silently reinterpreted
    stale_convention = response.to_dict()
    stale_convention["convention"] = "frozen_total_h_ordered_two_sided"
    with pytest.raises(SpiralResponseError, match="convention"):
        ResponseSet.from_dict(stale_convention)
    # a record of a DIFFERENT rotating operator is refused when the consumer
    # declares which mode it expects; a relabeled payload is refused by the
    # digest, and a digest-consistent foreign mode by the mode gate
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(
            response.to_dict(), expect_operator_mode=OPERATOR_MODE_WEIGHTED_TOTAL_H
        )


def test_response_set_records_and_gates_operator_mode():
    """The weighted variant is selectable and every serialization is labeled."""
    bundle, vertex, problem, eig = _bundle_problem(
        operator=OPERATOR_MODE_WEIGHTED_TOTAL_H
    )
    assert vertex.certificate["operator_mode"] == OPERATOR_MODE_WEIGHTED_TOTAL_H
    assert problem.operator_mode == OPERATOR_MODE_WEIGHTED_TOTAL_H
    gap = float(eig[2] - eig[1])
    ensemble = Ensemble(
        ENSEMBLE_GAPPED_T0, 2, 2.0, float((eig[1] + eig[2]) / 2), 0.0, gap
    )
    contour = EnergyContour.declared(
        z0=ensemble.fermi_energy_ev,
        occupied_min=float(eig[0]),
        gap_ev=gap,
        margin_ev=0.4,
        n_nodes=256,
    )
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=(("A", "y"), ("B", "y")),
        kpoints=[0.0],
        k_weights=[1.0],
        contour=contour,
        finite_steps=(4e-3, 2e-3),
    )
    assert response.operator_mode == OPERATOR_MODE_WEIGHTED_TOTAL_H
    payload = json.loads(json.dumps(response.to_dict()))
    clone = ResponseSet.from_dict(payload)
    assert clone == response
    # the weighted record is accepted only under its own mode expectation
    assert (
        ResponseSet.from_dict(
            payload, expect_operator_mode=OPERATOR_MODE_WEIGHTED_TOTAL_H
        ).operator_mode
        == OPERATOR_MODE_WEIGHTED_TOTAL_H
    )
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(payload, expect_operator_mode=OPERATOR_MODE_ONSITE_DELTA)
    # digest-consistent but unknown mode: refused, never reinterpreted
    unknown = dict(payload)
    unknown["operator_mode"] = "total_h_ordered"
    unknown["content_sha256"] = _payload_digest(
        {k: v for k, v in unknown.items() if k != "content_sha256"}
    )
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(unknown)
    # a missing operator_mode fails closed even with a consistent digest
    missing = {k: v for k, v in payload.items() if k != "operator_mode"}
    missing["content_sha256"] = _payload_digest(missing)
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(missing)
    # SPEC-004/STD-002: internally contradictory mode labels are refused
    # even when the payload digest is re-signed consistently
    contradicting = json.loads(json.dumps(payload))
    contradicting["provenance"]["operator_mode"] = OPERATOR_MODE_ONSITE_DELTA
    contradicting["content_sha256"] = _payload_digest(
        {k: v for k, v in contradicting.items() if k != "content_sha256"}
    )
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(contradicting)
    embedded = json.loads(json.dumps(payload))
    embedded["frozen_declarations"]["magnetic_amplitudes"]["certificate"][
        "operator_mode"
    ] = OPERATOR_MODE_ONSITE_DELTA
    embedded["content_sha256"] = _payload_digest(
        {k: v for k, v in embedded.items() if k != "content_sha256"}
    )
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        ResponseSet.from_dict(embedded)


def test_spinor_bundle_band_matches_finite_rotation():
    _, _, problem, eig = _bundle_problem()
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=float(0.5 * (eig[1] + eig[2])),
        smearing_width_ev=0.0,
        gap_ev=float(eig[2] - eig[1]),
    )
    coords = (("A", "y"), ("B", "y"))
    thetas = [{}, {coords[0][0]: {"y": 2e-3}}, {coords[0][0]: {"y": -2e-3}}]
    energies = [
        problem.occupied_energy([0.0], [1.0], theta, ensemble) for theta in thetas
    ]
    finite_tau = -(energies[1] - energies[2]) / (4e-3)
    band = problem.band_observables([0.0], [1.0], ensemble, coords)
    analytic_tau = -band["gradient"][coords[0]]
    assert finite_tau == pytest.approx(analytic_tau, abs=1e-5)
    finite_k = (energies[1] - 2.0 * energies[0] + energies[2]) / (2e-3) ** 2
    assert finite_k == pytest.approx(
        band["curvature"][(coords[0], coords[0])], abs=1e-4
    )


# ---------------------------------------------------------------------------
# production path: operator-mode finite rotations
# ---------------------------------------------------------------------------
def test_uniform_common_rotation_is_one_global_conjugation():
    """q=0 synthetic bundle: weighted variant conjugates globally.

    The weighted site contributions of every shared and onsite block sum to
    the full ``U B U^dag``, so the pencil transforms by exactly one global
    spinor conjugation (S is spin-scalar here), while a single-site
    rotation remains a genuine physical change.
    """
    _, _, problem, _eig = _bundle_problem(operator=OPERATOR_MODE_WEIGHTED_TOTAL_H)
    assert problem.operator_mode == OPERATOR_MODE_WEIGHTED_TOTAL_H
    angle = 0.13
    c, sn = np.cos(angle / 2.0), np.sin(angle / 2.0)
    u = np.kron(np.array([[c, -sn], [sn, c]]), np.eye(problem.bundle.norb))
    h, s = problem.pencil(0.0)
    rotated, s_rot = problem.rotated_pencil(0.0, {"A": {"y": angle}, "B": {"y": angle}})
    np.testing.assert_allclose(rotated, u @ h @ u.conj().T, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(s_rot, s)
    # order of the per-site angles is irrelevant: additive contributions
    rotated_again, s_again = problem.rotated_pencil(
        0.0, {"B": {"y": angle}, "A": {"y": angle}}
    )
    np.testing.assert_array_equal(rotated_again, rotated)
    np.testing.assert_array_equal(s_again, s_rot)
    # a single-site rotation is NOT a global conjugation: it must move the
    # pencil (local response stays nontrivial)
    local, _s_local = problem.rotated_pencil(0.0, {"A": {"y": angle}})
    assert not np.allclose(local, h, atol=1e-8)


def test_onsite_delta_common_rotation_freezes_intersite_blocks():
    """ADR-004 default: only R=0 same-site blocks rotate; intersite frozen.

    A rotation common to both sites conjugates the two onsite blocks but
    leaves the intersite magnetic blocks bit-identical, so the common
    rotation is no longer a global conjugation -- while the zero-angle
    rotation still returns exactly the reference pencil.
    """
    bundle, vertex, problem, _eig = _bundle_problem()
    assert vertex.certificate["operator_mode"] == OPERATOR_MODE_ONSITE_DELTA
    assert vertex.partition["A"] == frozenset({(0, 0, 0)})
    assert vertex.partition["B"] == frozenset({(1, 1, 0)})
    assert problem.operator_mode == OPERATOR_MODE_ONSITE_DELTA
    angle = 0.13
    c, sn = np.cos(angle / 2.0), np.sin(angle / 2.0)
    u = np.kron(np.array([[c, -sn], [sn, c]]), np.eye(bundle.norb))
    h, s = problem.pencil(0.0)
    rotated, s_rot = problem.rotated_pencil(0.0, {"A": {"y": angle}, "B": {"y": angle}})
    # the intersite magnetic blocks never move, bitwise
    intersite = np.ix_([0, bundle.norb], [1, 1 + bundle.norb])
    np.testing.assert_array_equal(rotated[intersite], h[intersite])
    reverse = np.ix_([1, 1 + bundle.norb], [0, bundle.norb])
    np.testing.assert_array_equal(rotated[reverse], h[reverse])
    # ... hence a common rotation is NOT one global conjugation anymore
    assert not np.allclose(rotated, u @ h @ u.conj().T, atol=1e-8)
    # the onsite blocks themselves did rotate (local response nontrivial)
    onsite_a = np.ix_([0, bundle.norb], [0, bundle.norb])
    assert not np.allclose(rotated[onsite_a], h[onsite_a], atol=1e-8)
    np.testing.assert_array_equal(s_rot, s)
    # zero-angle rotation returns exactly the reference pencil
    frozen, s_frozen = problem.rotated_pencil(0.0, {})
    np.testing.assert_array_equal(frozen, h)
    np.testing.assert_array_equal(s_frozen, s)
    # single-site rotation still moves the pencil
    local, _s_local = problem.rotated_pencil(0.0, {"A": {"y": angle}})
    assert not np.allclose(local, h, atol=1e-8)


def test_onsite_delta_rotation_leaves_nonzero_r_images_bit_frozen():
    """SPEC-002: rotating amplitudes are only the R=0 keys; images frozen.

    All same-orbital magnetism lives in the R=+-1 images and the R=0 blocks
    are spin-scalar, so the onsite_delta rotating amplitudes carry no
    magnetism at all: the first-order vertices vanish identically and every
    image-bearing position is unchanged (bitwise on the image-only
    intersite positions). Under the former R-summed assembled-block
    decomposition the images rode along with the (mu, mu) block and rotated
    with the site.
    """
    bundle = _spinor_bundle()
    bundle.sparse[(0, 0, 0)] = (2.0, 2.0, 0.0, 0.0)
    bundle.sparse[(1, 1, 0)] = (1.5, 1.5, 0.0, 0.0)
    bundle.ncell = 3
    for orb in range(bundle.norb):
        for r in (-1, 1):
            bundle.sparse[orb, orb, r] = (2.3, 1.7, 0.12, -0.07)
            bundle.overlap[orb, orb, r] = 0.03
    for r, partner_r in ((1, -1), (-1, 1)):
        bundle.sparse[(0, 1, r)] = (0.05, -0.02, 0.1, -0.04)
        bundle.sparse[(1, 0, partner_r)] = (0.05, -0.02, 0.1, -0.04)
        bundle.overlap[(0, 1, r)] = 0.02
        bundle.overlap[(1, 0, partner_r)] = 0.02
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    assert vertex.partition["A"] == frozenset({(0, 0, 0)})
    assert vertex.partition["B"] == frozenset({(1, 1, 0)})
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)},
    )
    h, s = problem.pencil(0.0)
    rotated, s_rot = problem.rotated_pencil(0.0, {"A": {"y": 0.4}, "B": {"y": -0.25}})
    n = bundle.norb
    # the intersite positions carry only frozen keys (R=0 blocks and their
    # R=+-1 images): they never move, bitwise
    np.testing.assert_array_equal(
        rotated[np.ix_([0, n], [1, 1 + n])], h[np.ix_([0, n], [1, 1 + n])]
    )
    np.testing.assert_array_equal(
        rotated[np.ix_([1, 1 + n], [0, n])], h[np.ix_([1, 1 + n], [0, n])]
    )
    # the same-orbital R=+-1 images ride in the frozen complement: the
    # rotating amplitudes (the spin-scalar R=0 keys) have an identically
    # zero first vertex, and the pencil is unchanged to roundoff
    for coord in (("A", "y"), ("B", "y")):
        v = problem.vertex_pencil(0.0, coord)
        np.testing.assert_array_equal(v, np.zeros_like(v))
    np.testing.assert_allclose(rotated, h, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(s_rot, s)


def test_onsite_delta_rotates_full_r0_same_site_orbital_block():
    """SPEC-001/002: mu != nu same-site R=0 blocks rotate onsite.

    Site A owns orbitals 0 and 1, so its rotating amplitude is the FULL
    R=0 site block -- (0,0,0), (1,1,0) and the off-diagonal
    (0,1,0)/(1,0,0) -- matching ExchangeCL2.get_Delta, while the intersite
    blocks to site B stay bit-frozen.
    """
    bundle = SpiralBundle(
        version=BUNDLE_VERSION,
        norb=3,
        ncell=1,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse={
            (0, 0, 0): (2.0, 0.5, 0.3, 0.1),
            (1, 1, 0): (1.8, -0.4, 0.12, 0.05),
            (2, 2, 0): (-3.0, 0.2, 0.2, -0.06),
            (0, 1, 0): (0.2, -0.1, 0.15, -0.08),
            (1, 0, 0): (0.2, -0.1, 0.15, -0.08),
            (0, 2, 0): (0.05, -0.02, 0.1, -0.03),
            (2, 0, 0): (0.05, -0.02, 0.1, -0.03),
            (1, 2, 0): (0.05, -0.02, -0.06, 0.04),
            (2, 1, 0): (0.05, -0.02, -0.06, 0.04),
        },
        overlap={
            (0, 0, 0): 1.0,
            (1, 1, 0): 1.0,
            (2, 2, 0): 1.0,
            (0, 1, 0): 0.05,
            (1, 0, 0): 0.05,
            (0, 2, 0): 0.02,
            (2, 0, 0): 0.02,
            (1, 2, 0): 0.02,
            (2, 1, 0): 0.02,
        },
    )
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0, 1], "B": [2]})
    assert vertex.partition["A"] == frozenset(
        {(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)}
    )
    assert vertex.partition["B"] == frozenset({(2, 2, 0)})
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)},
    )
    h, s = problem.pencil(0.0)
    rotated, s_rot = problem.rotated_pencil(0.0, {"A": {"y": 0.35}})
    n = bundle.norb
    off = np.ix_([0, n], [1, 1 + n])
    # the off-diagonal same-site R=0 block rotated with site A
    assert not np.allclose(rotated[off], h[off], atol=1e-8)
    onsite_a = np.ix_([0, n, 1, 1 + n], [0, n, 1, 1 + n])
    assert not np.allclose(rotated[onsite_a], h[onsite_a], atol=1e-8)
    # the intersite magnetic blocks to site B are bit-frozen
    for a, b in ((0, 2), (1, 2), (2, 0), (2, 1)):
        idx = np.ix_([a, a + n], [b, b + n])
        np.testing.assert_array_equal(rotated[idx], h[idx])
    np.testing.assert_array_equal(s_rot, s)
    # the analytic first vertex of site A carries the off-diagonal block
    v = problem.vertex_pencil(0.0, ("A", "y"))
    assert not np.allclose(v[off], np.zeros((2, 2)), atol=1e-8)


# ---------------------------------------------------------------------------
# fail-closed gates
# ---------------------------------------------------------------------------
def test_fail_closed_on_undeclared_partition():
    bundle = _spinor_bundle()
    vertex = extract_magnetic_vertex(bundle)
    with pytest.raises(SpiralResponseError, match="undeclared"):
        FrozenResponseProblem(bundle, vertex, axes_by_site={"A": (("y", AXES["y"]),)})


def test_fail_closed_on_bad_axes_ensembles_and_contours():
    bundle = _spinor_bundle()
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    with pytest.raises(SpiralResponseError, match="zero"):
        FrozenResponseProblem(
            bundle,
            vertex,
            axes_by_site={"A": (("y", (0.0, 0.0, 0.0)),), "B": (("y", AXES["y"]),)},
        )
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)},
    )
    h, s = problem.pencil(0.0)
    eig = eigh(h, s, eigvals_only=True)
    good_ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=0.0,
        smearing_width_ev=0.0,
        gap_ev=float(eig[2] - eig[1]),
    )
    with pytest.raises(SpiralResponseError, match="unsupported ensemble"):
        Ensemble(
            kind="metallic_step",
            n_occupied=2,
            electron_count=2.0,
            fermi_energy_ev=0.0,
            smearing_width_ev=0.0,
            gap_ev=1.0,
        )
    with pytest.raises(SpiralResponseError, match="n_occupied"):
        Ensemble(
            kind=ENSEMBLE_GAPPED_T0,
            n_occupied=0,
            electron_count=2.0,
            fermi_energy_ev=0.0,
            smearing_width_ev=0.0,
            gap_ev=1.0,
        )
    with pytest.raises(SpiralResponseError, match="margin"):
        EnergyContour.declared(
            z0=0.0,
            occupied_min=-3.0,
            gap_ev=1.0,
            margin_ev=0.9,
            n_nodes=64,
        )
    with pytest.raises(SpiralResponseError, match="n_nodes"):
        EnergyContour.declared(
            z0=0.0, occupied_min=-3.0, gap_ev=3.0, margin_ev=0.4, n_nodes=8
        )
    with pytest.raises(SpiralResponseError, match="unknown site"):
        evaluate_response_set(
            problem,
            ensemble=good_ensemble,
            coords=(("C", "y"),),
            kpoints=[0.0],
            k_weights=[1.0],
            contour=EnergyContour.declared(
                z0=float(0.5 * (eig[1] + eig[2])),
                occupied_min=float(eig[0]),
                gap_ev=float(eig[2] - eig[1]),
                margin_ev=0.4,
                n_nodes=64,
            ),
        )
    with pytest.raises(SpiralResponseError, match="weight"):
        evaluate_response_set(
            problem,
            ensemble=good_ensemble,
            coords=(("A", "y"),),
            kpoints=[0.0],
            k_weights=[0.0],
            contour=EnergyContour.declared(
                z0=float(0.5 * (eig[1] + eig[2])),
                occupied_min=float(eig[0]),
                gap_ev=float(eig[2] - eig[1]),
                margin_ev=0.4,
                n_nodes=64,
            ),
        )


def test_fail_closed_on_vertex_bundle_mismatch():
    bundle = _spinor_bundle()
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    shifted = _spinor_bundle()
    shifted.sparse[(0, 0, 0)] = (2.1, 0.5, 0.3, 0.1)
    with pytest.raises(SpiralResponseError, match="bound"):
        FrozenResponseProblem(
            shifted,
            vertex,
            axes_by_site={"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)},
        )


def test_fail_closed_on_unknown_or_mislabeled_operator_mode():
    """Operator labeling fails closed at the response layer, too.

    An unknown ``operator_mode`` is refused outright; a weighted partition
    relabeled ``onsite_delta`` (digest re-signed, declaration text fixed to
    pass the vertex gates) is refused because it would rotate shared
    intersite blocks under a label that promises not to.
    """
    bundle = _spinor_bundle()
    axes = {"A": (("y", AXES["y"]),), "B": (("y", AXES["y"]),)}
    onsite = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={"A": [0], "B": [1]},
        operator=OPERATOR_MODE_ONSITE_DELTA,
    )
    unknown = MagneticVertex(
        onsite.blocks,
        onsite.partition,
        dict(onsite.certificate, operator_mode="total_h_ordered"),
    )
    with pytest.raises(SpiralResponseError, match="operator_mode"):
        FrozenResponseProblem(bundle, unknown, axes_by_site=axes)

    weighted = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={"A": [0], "B": [1]},
        operator=OPERATOR_MODE_WEIGHTED_TOTAL_H,
    )
    forged_cert = dict(
        weighted.certificate,
        operator_mode=OPERATOR_MODE_ONSITE_DELTA,
        partition_declaration=PARTITION_DECLARATION_ONSITE,
    )
    forged_cert["vertex_sha256"] = _content_digest(
        weighted.blocks,
        weighted.partition,
        {k: v for k, v in forged_cert.items() if k != "vertex_sha256"},
    )
    forged = MagneticVertex(weighted.blocks, weighted.partition, forged_cert)
    with pytest.raises(SpiralResponseError, match="onsite_delta.*shared|shared block"):
        FrozenResponseProblem(bundle, forged, axes_by_site=axes)


def test_gap_gate_refuses_closed_gap():
    h = np.zeros((4, 4), dtype=complex)
    s = np.eye(4)
    with pytest.raises(SpiralResponseError, match="gap"):
        occupied_band_energy(h, s, 2)


# ---------------------------------------------------------------------------
# O2 fixture smoke on a reduced k subset
# ---------------------------------------------------------------------------
def test_o2_smoke_reduced_k_ladder_rung():
    bundle, reference = load_fixture(FIXTURE)
    norb = bundle.norb
    assert norb % 2 == 0
    per_site = range(norb // 2)
    vertex = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={
            "O1": list(per_site),
            "O2": [o + norb // 2 for o in per_site],
        },
    )
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={
            "O1": tuple(AXES.items()),
            "O2": tuple(AXES.items()),
        },
    )
    kp = np.asarray(reference["kp"], dtype=float)
    subset = kp[:2]  # reduced k subset per story-009 approval
    weights = [0.5, 0.5]
    n_occupied = int(round(reference["electron_count"]))
    assert n_occupied == 12  # spinor bands have unit electron multiplicity
    homos = [reference["eig"][i][n_occupied - 1] for i in range(len(subset))]
    lumos = [reference["eig"][i][n_occupied] for i in range(len(subset))]
    homo_max, lumo_min = max(homos), min(lumos)
    gap = lumo_min - homo_max
    assert gap > 1.0
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=n_occupied,
        electron_count=float(reference["electron_count"]),
        fermi_energy_ev=float(reference["ef"]),
        smearing_width_ev=KB_EV * float(reference["temperature"]),
        gap_ev=float(gap),
    )
    contour = EnergyContour.declared(
        z0=float(0.5 * (homo_max + lumo_min)),
        occupied_min=float(
            min(min(reference["eig"][i][:n_occupied]) for i in range(2))
        ),
        gap_ev=float(gap),
        margin_ev=0.8,
        n_nodes=1024,
    )
    coords = tuple((site, axis) for site in ("O1", "O2") for axis in ("x", "y"))
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=[tuple(k) for k in subset],
        k_weights=weights,
        contour=contour,
        finite_steps=(4e-3, 2e-3),
        contour_node_rungs=(256, 512, 1024),
        contour_rungs=tuple(
            EnergyContour.declared(
                z0=contour.z0,
                occupied_min=contour.occupied_min,
                gap_ev=gap,
                margin_ev=height,
                n_nodes=1024,
            )
            for height in (0.7, 0.8)
        ),
        # The k-mesh ladder is exercised on the synthetic bundle (its meshes
        # agree by construction).  On this reduced two-point O2 subset the
        # one-vs-two-point toy meshes are NOT k-converged under the
        # onsite_delta operator (recorded torque delta 1.1e-3 eV/rad between
        # the toy meshes), so the smoke keeps the node and contour-height
        # ladders whose production budgets gate the integration.
    )

    values = list(response.gradient.values()) + list(response.curvature.values())
    assert all(np.isfinite(values))
    assert response.operator_mode == OPERATOR_MODE_ONSITE_DELTA
    assert response.contour_vs_band_max_abs <= RESPONSE_IDENTITY_TOL_EV
    assert len(response.curvature) == 16
    for a in coords:
        for b in coords:
            assert response.curvature[a, b] == pytest.approx(
                response.curvature[b, a], abs=1e-8
            )
    assert (("O1", "x"), ("O2", "x")) in response.curvature
    assert (("O1", "x"), ("O2", "y")) in response.curvature
    assert response.finite_rotation_cross_check["torque_max_abs_mismatch"] < 1e-6
    assert response.finite_rotation_cross_check["curvature_max_abs_mismatch"] < 1e-5
    assert response.ladder["converged"]["torque"] is True
    assert response.ladder["converged"]["curvature"] is True
    assert response.ensemble["n_occupied"] == 12
    assert response.ensemble["gap_ev"] == pytest.approx(gap, abs=1e-9)
    assert response.ensemble["occupations_per_k"] == [[1] * 12 + [0] * 24] * 2
    assert response.ensemble["occupation_smearing_width_ev"] == 0.0
    # The per-key phased seam keeps the onsite rotating amplitudes in the
    # importer's Bloch phases: derivative and rotated pencils stay Hermitian.
    for k in subset:
        for coord in coords:
            v = problem.vertex_pencil(k, coord)
            np.testing.assert_allclose(v, v.conj().T, atol=1e-8)
        rotated, s_rot = problem.rotated_pencil(k, {"O1": {"y": 0.01}})
        _h, s_ref = problem.pencil(k)
        np.testing.assert_allclose(rotated, rotated.conj().T, atol=1e-8)
        np.testing.assert_array_equal(s_rot, s_ref)
    clone = ResponseSet.from_dict(json.loads(json.dumps(response.to_dict())))
    assert clone == response


@pytest.mark.parametrize(
    "operator", (OPERATOR_MODE_ONSITE_DELTA, OPERATOR_MODE_WEIGHTED_TOTAL_H)
)
def test_known_helix_full_transverse_response_and_frozen_metric(operator):
    """Two-site q=0.7 helix with spin-textured phased overlap."""
    bundle = _spinor_bundle()
    bundle.q = (0.7,)
    bundle.ncell = 3
    for orb in range(bundle.norb):
        for r in (-1, 1):
            bundle.sparse[orb, orb, r] = (0.04, -0.02, 0.0, 0.0)
            bundle.overlap[orb, orb, r] = 0.03
    vertex = extract_magnetic_vertex(
        bundle, orbitals_by_site={"A": [0], "B": [1]}, operator=operator
    )
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": tuple(AXES.items()), "B": tuple(AXES.items())},
    )
    k = 0.3
    h, s = problem.pencil(k)
    eig = eigh(h, s, eigvals_only=True)
    gap = float(eig[2] - eig[1])
    ef = float((eig[1] + eig[2]) / 2.0)
    ensemble = Ensemble(ENSEMBLE_GAPPED_T0, 2, 2.0, ef, 0.0, gap)
    contour = EnergyContour.declared(
        z0=ef,
        occupied_min=float(eig[0]),
        gap_ev=gap,
        margin_ev=min(0.4, gap * 0.3),
        n_nodes=256,
    )
    result = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=COORDS,
        kpoints=[k],
        k_weights=[1.0],
        contour=contour,
        finite_steps=(4e-3, 2e-3),
        contour_node_rungs=(64, 128, 256),
    )
    assert len(result.curvature) == 16
    assert result.contour_vs_band_max_abs < 1e-8
    assert result.operator_mode == operator
    h_zero, s_zero = problem.rotated_pencil(k, {})
    np.testing.assert_allclose(h_zero, h, atol=1e-14)
    np.testing.assert_array_equal(s_zero, s)
    # Phases make S genuinely spin-dependent; it is never replaced by lab S.
    assert abs(s[0, 0] - s[2, 2]) > 1e-4
    rotated, frozen_s = problem.rotated_pencil(
        k, {"A": {"x": 0.13, "y": -0.04}, "B": {"y": -0.17}}
    )
    np.testing.assert_array_equal(frozen_s, s)
    np.testing.assert_allclose(rotated, rotated.conj().T, atol=1e-14)
    for mu in range(bundle.norb):
        for nu in range(bundle.norb):
            idx = np.ix_([mu, mu + bundle.norb], [nu, nu + bundle.norb])
            assert np.trace(rotated[idx]) == pytest.approx(np.trace(h[idx]), abs=1e-14)
    for a in COORDS:
        for b in COORDS:
            assert result.curvature[a, b] == pytest.approx(
                result.curvature[b, a], abs=1e-12
            )
            finite = result.ladder["finite_rotation"][-1]["curvature"][
                json.dumps([list(a), list(b)], separators=(",", ":"))
            ]
            assert finite == pytest.approx(result.curvature[a, b], abs=2e-6)


def test_gapped_equal_occupation_degeneracy_has_no_zero_division():
    h = np.diag([-3.0, -3.0, 2.0, 4.0])
    s = np.eye(4)
    v = np.zeros((4, 4))
    v[0, 2] = v[2, 0] = 1.0
    w = np.diag([1.0, 2.0, 0.0, 0.0])
    with np.errstate(divide="raise", invalid="raise"):
        total, onsite, response = pencil_curvature(h, s, w, v, v, 2)
    assert onsite == pytest.approx(3.0)
    assert response == pytest.approx(-0.4)
    assert total == pytest.approx(2.6)


def test_declared_contour_must_enclose_exactly_the_occupied_states():
    _, _, problem, eig = _bundle_problem()
    gap = float(eig[2] - eig[1])
    ensemble = Ensemble(
        ENSEMBLE_GAPPED_T0, 2, 2.0, float((eig[1] + eig[2]) / 2), 0.0, gap
    )
    bad = EnergyContour.declared(
        z0=float(eig[-1] + 2.0),
        occupied_min=float(eig[0]),
        gap_ev=gap,
        margin_ev=0.4,
        n_nodes=64,
    )
    with pytest.raises(SpiralResponseError, match="exactly the occupied"):
        evaluate_response_set(
            problem,
            ensemble=ensemble,
            coords=(("A", "y"),),
            kpoints=[0.0],
            k_weights=[1.0],
            contour=bad,
        )
    with pytest.raises(SpiralResponseError, match="unit multiplicity"):
        Ensemble(ENSEMBLE_GAPPED_T0, 2, 4.0, 0.0, 0.0, gap)


def test_contour_public_boundary_refuses_nonhermitian_pencil():
    contour = EnergyContour.declared(
        z0=0.5, occupied_min=-3.0, gap_ev=3.0, margin_ev=0.4, n_nodes=64
    )
    h = np.diag([-3.0, -1.0, 2.0, 4.0])
    h[0, 1] = 0.1
    s, v = np.eye(4), np.eye(4)
    with pytest.raises(SpiralResponseError, match="not Hermitian"):
        contour_gradient(h, s, v, contour)
    with pytest.raises(SpiralResponseError, match="not Hermitian"):
        contour_curvature(h, s, v, v, v, contour)


def _response_run_kwargs():
    _, _, problem, eig = _bundle_problem()
    gap = float(eig[2] - eig[1])
    ensemble = Ensemble(
        ENSEMBLE_GAPPED_T0, 2, 2.0, float((eig[1] + eig[2]) / 2), 0.0, gap
    )
    contour = EnergyContour.declared(
        z0=ensemble.fermi_energy_ev,
        occupied_min=float(eig[0]),
        gap_ev=gap,
        margin_ev=0.4,
        n_nodes=256,
    )
    return problem, dict(
        ensemble=ensemble,
        coords=(("A", "y"), ("B", "y")),
        kpoints=[0.0],
        k_weights=[1.0],
        contour=contour,
        finite_steps=(4e-3, 2e-3),
        contour_node_rungs=(64, 128, 256),
    )


def test_undersampled_final_contour_fails_identity_gate():
    problem, kwargs = _response_run_kwargs()
    kwargs["contour"] = kwargs["contour"].at_nodes(32)
    with pytest.raises(SpiralResponseError, match="contour.*identity"):
        evaluate_response_set(problem, **kwargs)


def test_ensemble_gap_must_match_sampled_spectrum():
    from dataclasses import replace

    problem, kwargs = _response_run_kwargs()
    kwargs["ensemble"] = replace(
        kwargs["ensemble"], gap_ev=kwargs["ensemble"].gap_ev + 0.1
    )
    with pytest.raises(SpiralResponseError, match="declared.*gap"):
        evaluate_response_set(problem, **kwargs)


def test_mesh_and_contour_height_ladders_and_finite_mismatch_roundtrip():
    problem, kwargs = _response_run_kwargs()
    c = kwargs["contour"]
    kwargs["contour_rungs"] = [
        EnergyContour.declared(
            z0=c.z0,
            occupied_min=c.occupied_min,
            gap_ev=c.gap_ev,
            margin_ev=height,
            n_nodes=256,
        )
        for height in (0.35, 0.4)
    ]
    kwargs["k_mesh_rungs"] = [
        {
            "label": "one",
            "kpoints": [0.0],
            "k_weights": [1.0],
            "ensemble": kwargs["ensemble"],
        },
        {
            "label": "two",
            "kpoints": [0.0, 0.0],
            "k_weights": [0.5, 0.5],
            "ensemble": kwargs["ensemble"],
        },
    ]
    result = evaluate_response_set(problem, **kwargs)
    assert len(result.ladder["k_mesh"]) == len(result.ladder["contour_parameters"]) == 2
    for ladder in ("k_mesh", "contour_parameters"):
        rung = result.ladder[ladder][-1]
        assert rung["torque_max_delta_vs_previous"] <= TORQUE_BUDGET_EV_PER_RAD
        assert rung["curvature_max_delta_vs_previous"] <= CURVATURE_BUDGET_EV_PER_RAD2
        assert rung["within_approved_budgets"] is True
    mismatch = result.finite_rotation_cross_check
    assert 0.0 < mismatch["torque_max_abs_mismatch"] < 1e-5
    assert 0.0 < mismatch["curvature_max_abs_mismatch"] < 1e-5
    clone = ResponseSet.from_dict(json.loads(json.dumps(result.to_dict())))
    assert clone.finite_rotation_cross_check == mismatch
    assert clone == result


def test_problem_snapshot_remains_bound_without_hot_path_rehashing(monkeypatch):
    import TB2J.interfaces.siesta_spiral_response as core

    bundle, _vertex, problem, _eig = _bundle_problem()
    h0, s0 = problem.pencil(0.0)
    # Source inputs and exported snapshots are not mutable backdoors into
    # the problem's once-bound data.
    bundle.q = (0.2,)
    exported = problem.bundle
    exported.q = (0.3,)
    calls = []
    real_digest = core._source_bundle_digest

    def digest(bundle):
        calls.append(1)
        return real_digest(bundle)

    monkeypatch.setattr(core, "_source_bundle_digest", digest)
    problem.vertex_pencil(0.0, ("A", "y"))
    h1, s1 = problem.rotated_pencil(0.0, {})
    np.testing.assert_allclose(h1, h0, atol=1e-14)
    np.testing.assert_array_equal(s1, s0)
    assert calls == []
