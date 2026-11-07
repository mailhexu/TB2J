"""Isolated band groups and sampling convergence (ticket 04).

Group invariance under unitary mixing within the group, the
individual-band-touching refusal, external-gap closure behavior, multi-group
requests and the inspectable refinement policy are exercised here on the
certificate-pinned honeycomb benchmark. The group-mixing identity is
certified symbolically in
``docs/sympy/magnon_topology_lswt_ferromagnet.py`` (section 5).
"""

import json
import sys

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import BulkRequest, calculate_bulk, prepare_topology_model
from TB2J.magnon.topology.bulk import refine_bulk

# test parameters in eV; mu = 2 muB so executable energies match textbook S=1
J1 = 0.01
J2 = 0.002
K_SIA = 0.001


def honeycomb_spinio(D):
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    atoms = Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, -1 / 3, 0.0]],
        cell=basis,
        pbc=True,
    )
    magmoms = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    exc, dmi = {}, {}

    def rev(R):
        return tuple(-np.array(R))

    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        exc[(R, 0, 1)] = J1
        exc[(rev(R), 1, 0)] = J1
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
        dmi_ddict=dmi if D != 0 else None,
        sia_tensor=sia,
    )
    spinio._build_Rlist()
    return spinio


@pytest.fixture(scope="module")
def model():
    return prepare_topology_model(honeycomb_spinio(0.0005), source_path="bench")


def test_unitary_mixing_within_group_preserves_invariant(model):
    # apply a k-dependent random unitary to the selected band columns: the
    # plaquette link product telescopes and the invariant must not move.
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid and res.chern == 1

    rng = np.random.default_rng(42)
    nu, nv = 12, 12
    mgn = model.magnon()
    k = np.stack(np.meshgrid(np.arange(nu) / nu, np.arange(nv) / nv, indexing="ij"), -1)
    kp = np.concatenate([k.reshape(-1, 2), np.zeros((nu * nv, 1))], axis=1)
    H = mgn.Hq(kp)
    from TB2J.magnon.topology.bosonic import solve_bosonic_modes

    psi = np.empty((nu * nv, 4, 1), dtype=complex)
    for idx in range(nu * nv):
        m = solve_bosonic_modes(H[idx])
        col = m.vectors[:, [0]]
        theta = rng.uniform(0, 2 * np.pi)
        u_mix = np.exp(1j * theta)  # 1x1 unitary mixing within the 1-band group
        psi[idx] = u_mix * col
    prod = np.ones((nu, nv), dtype=complex)
    for i in range(nu):
        for j in range(nv):
            p0, p1 = i * nv + j, ((i + 1) % nu) * nv + j
            p2, p3 = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            for p, q in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
                d = np.linalg.det(psi[p].conj().T @ psi[q])
                prod[i, j] *= d / abs(d)
    chern_mixed = int(np.rint((np.angle(prod) / (2 * np.pi)).sum()))
    assert chern_mixed == res.chern == 1


def test_group_request_with_two_groups(model):
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), groups=[[0], [1]]))
    assert res.valid
    assert res.groups is not None and len(res.groups) == 2
    assert res.groups[0]["chern"] == 1
    assert res.groups[1]["chern"] == -1
    assert res.groups[0]["bands"] == [0] and res.groups[1]["bands"] == [1]


def test_group_json_round_trip(model, tmp_path):
    res = calculate_bulk(model, BulkRequest(mesh=(8, 8), groups=[[0], [1]]))
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(res.to_json_dict()))
    data = json.loads(path.read_text())
    assert data["valid"] is True
    assert [g["chern"] for g in data["groups"]] == [1, -1]
    assert all(g["plaquette_flux"] is not None for g in data["groups"])
    assert data["conventions"]["group_curvature"].startswith("vertex value")


def test_single_band_touch_refused_while_group_valid():
    # at D=0 the two bands touch at the valleys: the individual band is
    # refused through the external gap, the full group stays trivially valid.
    model_gapless = prepare_topology_model(honeycomb_spinio(D=0.0), source_path="bench")
    single = calculate_bulk(model_gapless, BulkRequest(mesh=(12, 12), bands=[0]))
    assert not single.valid
    assert any("not isolated" in f for f in single.diagnostics["failures"])
    full = calculate_bulk(model_gapless, BulkRequest(mesh=(12, 12)))
    assert full.valid  # total Chern of all positive bands is exactly 0
    assert full.chern == 0


def test_refinement_report_evidence(model):
    report = refine_bulk(model, BulkRequest(mesh=(8, 8), bands=[0]))
    assert report["coarse"]["mesh"] == [8, 8]
    assert report["fine"]["mesh"] == [16, 16]
    assert report["chern_agrees"] is True
    assert report["coarse"]["chern_per_group"] == [1]
    cg, fg = (
        report["coarse"]["min_external_gap_per_group"],
        report["fine"]["min_external_gap_per_group"],
    )
    # the measured minimum converges DOWNWARD toward the continuum minimum:
    # a finer mesh samples closer to the valley
    assert 0 < fg[0] <= cg[0]
    assert "not proof" in report["note"]


def test_cli_groups_and_refine(tmp_path):
    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_groups"
    exc.write_pickle(str(root))
    out = tmp_path / "topo_groups.json"
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "12",
        "12",
        "--topology-groups",
        "0;1",
        "--topology-refine",
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
    assert [g["chern"] for g in data["groups"]] == [1, -1]
    assert data["refinement"]["chern_agrees"] is True


def test_cli_invalid_group_request_nonzero(tmp_path):
    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0)
    root = tmp_path / "res_gapless_groups"
    exc.write_pickle(str(root))
    out = tmp_path / "topo_gapless_groups.json"
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "12",
        "12",
        "--topology-groups",
        "0;1",
        "--topology-out",
        str(out),
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 1
    data = json.loads(out.read_text())
    assert data["valid"] is False
    assert all(g["chern"] is None for g in data["groups"])
