"""Anomalous bosonic modes and validity diagnostics (ticket 03).

The anomalous benchmark is the honeycomb FM of ticket 02 plus a small
symmetric anisotropic NN exchange (Jani) delta*diag(1,-1,0): the certified
pairing contraction b(J) = J_xx - J_yy = 2*delta makes the model genuinely
anomalous (B != 0) while the spectrum stays stable. Conventions are pinned by
``docs/sympy/magnon_topology_bosonic_modes.py``.
"""

import json
import sys

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import BulkRequest, calculate_bulk, prepare_topology_model
from TB2J.magnon.topology.bosonic import solve_bosonic_modes

J1 = 0.01
J2 = 0.002
K_SIA = 0.001
DELTA = 0.0003


def honeycomb_spinio(D=0.0005, delta=0.0):
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    atoms = Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, -1 / 3, 0.0]],
        cell=basis,
        pbc=True,
    )
    magmoms = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    exc, dmi, jani = {}, {}, {}

    def rev(R):
        return tuple(-np.array(R))

    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        exc[(R, 0, 1)] = J1
        exc[(rev(R), 1, 0)] = J1
        if delta:
            t = delta * np.diag([1.0, -1.0, 0.0])
            jani[(R, 0, 1)] = t
            jani[(rev(R), 1, 0)] = t.T
    for R in [(1, 0, 0), (-1, 1, 0), (0, -1, 0)]:
        exc[(R, 0, 0)] = J2
        exc[(rev(R), 0, 0)] = J2
        exc[(R, 1, 1)] = J2
        exc[(rev(R), 1, 1)] = J2
        dmi[(R, 0, 0)] = np.array([0.0, 0.0, D])
        dmi[(rev(R), 0, 0)] = np.array([0.0, 0.0, -D])
        dmi[(R, 1, 1)] = np.array([0.0, 0.0, -D])
        dmi[(rev(R), 1, 1)] = np.array([0.0, 0.0, D])
    sia = {0: np.diag([0.0, 0.0, K_SIA]), 1: np.diag([0.0, 0.0, K_SIA])}
    spinio = SpinIO(
        atoms=atoms,
        spinat=magmoms,
        charges=np.zeros(2),
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        dmi_ddict=dmi if D else None,
        Jani_dict=jani if delta else None,
        sia_tensor=sia,
    )
    spinio._build_Rlist()
    return spinio


@pytest.fixture(scope="module")
def anomalous_model():
    return prepare_topology_model(honeycomb_spinio(delta=DELTA), source_path="bench")


@pytest.fixture(scope="module")
def baseline_model():
    return prepare_topology_model(honeycomb_spinio(delta=0.0), source_path="bench")


def test_anomalous_hole_block_is_the_conjugate_negative_momentum_block(
    anomalous_model,
):
    from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian

    kpoint = np.array([[0.11, 0.23, 0.0]])
    hamiltonian = local_frame_bosonic_hamiltonian(
        anomalous_model.JR,
        anomalous_model.Rlist,
        anomalous_model.magmoms,
        kpoint,
    )[0]
    reverse = local_frame_bosonic_hamiltonian(
        anomalous_model.JR,
        anomalous_model.Rlist,
        anomalous_model.magmoms,
        -kpoint,
    )[0]
    particle = hamiltonian[:2, :2]
    hole = hamiltonian[2:, 2:]
    np.testing.assert_allclose(particle, particle.conj().T, atol=1e-12)
    assert np.max(np.abs(particle - particle.conj())) > 1e-6
    np.testing.assert_allclose(hole, reverse[:2, :2].conj(), atol=1e-12)

    real_bond_model = prepare_topology_model(
        honeycomb_spinio(D=0.0, delta=DELTA), source_path="real-bond"
    )
    real_bond_h = local_frame_bosonic_hamiltonian(
        real_bond_model.JR,
        real_bond_model.Rlist,
        real_bond_model.magmoms,
        kpoint,
    )[0]
    np.testing.assert_allclose(real_bond_h[2:, 2:], real_bond_h[:2, :2], atol=1e-12)


def test_anomalous_pairing_present_and_stable(anomalous_model):
    magnon = anomalous_model.magnon()
    H = magnon.Hq(np.array([[0.11, 0.23, 0.0]]))[0]
    assert np.max(np.abs(H[:2, 2:])) > 1e-6  # B != 0: genuinely anomalous
    m = solve_bosonic_modes(H)
    assert m.classification == "positive_definite"
    assert m.residuals["max_metric_norm_error"] < 1e-8
    assert m.residuals["max_generalized_residual"] < 1e-10


def test_anomalous_bulk_matches_baseline_chern(anomalous_model, baseline_model):
    # small delta opens no gap closing: same pinned band Chern as ticket 02
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(12, 12), bands=[0]))
    ref = calculate_bulk(baseline_model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid
    assert res.chern == ref.chern == 1
    assert res.diagnostics["solver"]["max_pairing_residual"] < 1e-8


def test_anomalous_cli_end_to_end(tmp_path):
    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(delta=DELTA)
    root = tmp_path / "res_anomalous"
    exc.write_pickle(str(root))
    out = tmp_path / "topo.json"
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "12",
        "12",
        "--topology-bands",
        "0",
        "--topology-out",
        str(out),
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 0
    data = json.loads(out.read_text())
    assert data["valid"] is True and data["chern"] == 1


def test_semidefinite_goldstone_reports_hessian_nullspace_without_singular_solve():
    H = np.ones((2, 2), dtype=complex)

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.null_space.shape == (2, 1)
    np.testing.assert_allclose(H @ modes.null_space, 0.0, atol=1e-12)
    assert modes.residuals["null_space_residual"] < 1e-12


def test_zero_mode_group_touching_zero_refused():
    # drop the SIA: Goldstone touches band 0 on the mesh
    exc = honeycomb_spinio(delta=0.0)
    exc.sia_tensor = {0: np.zeros((3, 3)), 1: np.zeros((3, 3))}
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert any("touches stable zero modes" in f for f in res.diagnostics["failures"])


def test_separated_band_group_proceeds_with_zero_modes():
    # band 1 stays separated from the Goldstone: validated invariant proceeds
    exc = honeycomb_spinio(delta=0.0)
    exc.sia_tensor = {0: np.zeros((3, 3)), 1: np.zeros((3, 3))}
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[1]))
    assert res.valid, res.diagnostics["failures"]
    assert res.chern == -1
    assert res.diagnostics["solver"]["n_mesh_points_with_zero_modes"] > 0


def test_canted_torque_detected_by_assembly():
    # canted moments: nonzero transverse torque; the Hessian matrix at Gamma
    # is positive definite, so only the assembly diagnostic can refuse.
    exc = honeycomb_spinio(delta=0.0)
    exc.spinat = np.array([[1.0, 1.0, np.sqrt(2.0)], [1.0, 1.0, np.sqrt(2.0)]])
    model = prepare_topology_model(exc, source_path="bench")
    magnon = model.magnon()
    H_gamma = magnon.Hq(np.array([[0.0, 0.0, 0.0]]))[0]
    # the matrix alone looks solvable at Gamma
    np.linalg.cholesky(H_gamma)
    from TB2J.magnon.topology.assembly import classical_transverse_gradient

    stat = classical_transverse_gradient(model.JR, model.Rlist, model.magmoms)
    assert not stat.stationary
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert any("nonstationary" in f for f in res.diagnostics["failures"])


def test_unequal_spin_lengths_preserve_noncollinear_stationarity():
    from TB2J.magnon.topology.assembly import classical_transverse_gradient
    from TB2J.magnon.topology.strip import _local_fields

    directions = np.array(
        [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2, 0.0], [-0.5, -np.sqrt(3.0) / 2, 0.0]]
    )
    moments = directions * np.array([2.0, 4.0, 6.0])[:, None]
    exchange = np.zeros((1, 3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                exchange[0, i, j] = -np.eye(3)
    rlist = np.zeros((1, 3), dtype=int)

    stat = classical_transverse_gradient(exchange, rlist, moments)
    assert stat.stationary
    np.testing.assert_allclose(stat.max_transverse, 0.0, atol=1e-14)
    fields = _local_fields(exchange, rlist, moments)
    np.testing.assert_allclose(
        fields - np.einsum("ij,ij->i", fields, directions)[:, None] * directions,
        0.0,
        atol=1e-14,
    )


def test_negative_hessian_real_frequency_refused_with_character(tmp_path):
    # easy-plane overcritical SIA: torque-free saddle with real dynamical
    # frequencies but a negative Hessian direction.
    basis = 10.0 * np.eye(3)
    atoms = Atoms(
        "Cr2", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], cell=basis, pbc=True
    )
    exc = {}
    for R in [(1, 0, 0)]:
        rev = tuple(-np.array(R))
        exc[(R, 0, 1)] = 0.01
        exc[(rev, 1, 0)] = 0.01
    sia = {0: np.diag([0.02, 0.02, 0.001]), 1: np.diag([0.02, 0.02, 0.001])}
    io = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        charges=[0.0, 0.0],
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        sia_tensor=sia,
    )
    io._build_Rlist()
    model = prepare_topology_model(io, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.diagnostics["solver"]["unstable_frequency_character"] == ["real"]

    # the legacy path still silently shifts and returns energies (baseline)
    magnon = model.magnon()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = magnon._magnon_energies(np.array([[0.0, 0.0, 0.0]]))
    assert len(caught) > 0
    assert np.all(np.isfinite(np.real(legacy)))


def test_complex_frequency_instability_reports_character():
    exc = honeycomb_spinio(delta=0.0)
    exc.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    chars = res.diagnostics["solver"]["unstable_frequency_character"]
    assert "complex" in chars


def test_near_zero_resolved_mode_stays_positive(anomalous_model):
    # a small but resolved positive energy stays positive_definite; the
    # explicit margins are carried in the diagnostics rather than hidden.
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert res.valid
    assert res.diagnostics["solver"]["min_physical_energy"] > 0
    assert res.diagnostics["solver"]["classifications"] == {"positive_definite": 36}


def test_json_carries_validity_classes(tmp_path, anomalous_model):
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(6, 6), bands=[0]))
    data = json.loads(json.dumps(res.to_json_dict()))
    solver = data["diagnostics"]["solver"]
    assert solver["classifications"] == {"positive_definite": 36}
    assert data["conventions"]["certificate"] == (
        "docs/sympy/magnon_topology_lswt_ferromagnet.py"
    )
