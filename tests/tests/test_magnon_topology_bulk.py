"""Bulk magnon topology: certificate-pinned benchmark and validity policy.

Conventions are pinned by the executed certificate
``docs/sympy/magnon_topology_lswt_ferromagnet.py`` (executable cell-phase
convention): plaquette order ``(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)`` on
the ``(a1,a2)`` torus, band Chern numbers ``C_lower = +sign(D_z)`` and
``C_upper = -sign(D_z)`` for the gapped honeycomb ferromagnet with NNN
Kane-Mele DMI and easy-axis SIA, Gamma energies ``omega- = K`` and
``omega+ = 6 J1 + K`` at ``S = 1``.

The source is a ``SpinIO``, whose exchange carries TB2J's unit-spin
convention (both pair directions); the magnon kernel converts it by
``2 / (S_i S_j)`` before the physical-spin kernel, so at ``S = 1`` the
executable energies are twice the textbook values of these couplings.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import BulkRequest, calculate_bulk, prepare_topology_model

# test parameters in eV; mu = 2 muB so executable energies match textbook S=1
J1 = 0.01
J2 = 0.002
K_SIA = 0.001


def honeycomb_spinio(D, k_sia=K_SIA):
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
    sia = {0: np.diag([0.0, 0.0, k_sia]), 1: np.diag([0.0, 0.0, k_sia])}
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
def model_pos():
    return prepare_topology_model(honeycomb_spinio(0.0005), source_path="bench")


def test_chern_lower_band_matches_certificate_pin(model_pos):
    res = calculate_bulk(model_pos, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid, res.diagnostics["failures"]
    assert res.chern == 1  # +sign(D_z) for the lower band


def test_chern_upper_band_matches_certificate_pin(model_pos):
    res = calculate_bulk(model_pos, BulkRequest(mesh=(12, 12), bands=[1]))
    assert res.valid
    assert res.chern == -1  # -sign(D_z) for the upper band


def test_chern_anti_aligns_with_dmi_sign():
    model_neg = prepare_topology_model(honeycomb_spinio(-0.0005), source_path="bench")
    res = calculate_bulk(model_neg, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid
    assert res.chern == -1


def test_gamma_energies_match_certificate(model_pos):
    res = calculate_bulk(model_pos, BulkRequest(mesh=(6, 6), bands=[0]))
    # the SpinIO unit-spin exchange is converted by 2/(S_i S_j): at S = 1
    # the executable energies are twice the textbook S=1 values.
    np.testing.assert_allclose(
        res.energies[0, 0], [2 * K_SIA, 2 * (6 * J1 + K_SIA)], atol=1e-10
    )


def test_mesh_refinement_preserves_chern(model_pos):
    c8 = calculate_bulk(model_pos, BulkRequest(mesh=(8, 8), bands=[0])).chern
    c16 = calculate_bulk(model_pos, BulkRequest(mesh=(16, 16), bands=[0])).chern
    assert c8 == c16 == 1


def test_gap_closing_refuses_validated_invariant():
    model_gapless = prepare_topology_model(honeycomb_spinio(0.0), source_path="bench")
    res = calculate_bulk(model_gapless, BulkRequest(mesh=(12, 12), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.plaquette_flux is None
    assert any("isolated" in f for f in res.diagnostics["failures"])


def test_full_band_group_has_zero_chern(model_pos):
    res = calculate_bulk(model_pos, BulkRequest(mesh=(12, 12)))
    assert res.valid
    assert res.chern == 0


def test_json_round_trip(tmp_path, model_pos):
    res = calculate_bulk(model_pos, BulkRequest(mesh=(8, 8), bands=[0]))
    path = tmp_path / "bulk.json"
    path.write_text(json.dumps(res.to_json_dict()))
    data = json.loads(path.read_text())
    assert data["kind"] == "bulk"
    assert data["schema_version"] == "tb2j-topology.bulk/1"
    assert data["chern"] == 1
    assert data["valid"] is True
    assert data["model_digest"] == model_pos.digest
    np.testing.assert_allclose(data["plaquette_flux"], res.plaquette_flux, atol=0)


def test_rejects_bare_magnon(model_pos):
    magnon = model_pos.magnon()
    with pytest.raises(TypeError, match="topology-ready"):
        calculate_bulk(magnon, BulkRequest(mesh=(6, 6), bands=[0]))


def test_rejects_nonstationary_reference():
    # Moments canted away from the easy axis: the easy-axis SIA field has a
    # transverse component at the reference, so the state is not stationary.
    exc = honeycomb_spinio(0.0005)
    exc.spinat = np.array([[1.0, 1.0, np.sqrt(2.0)], [1.0, 1.0, np.sqrt(2.0)]])
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert any("nonstationary" in f for f in res.diagnostics["failures"])


def test_nonstationary_reference_refuses_every_group_without_flux():
    # Band 1 alone has a resolved external gap here, so only the global
    # stationarity failure can invalidate it: its group record must refuse
    # instead of reporting valid=True with an all-zero uncomputed flux.
    exc = honeycomb_spinio(0.0005)
    exc.spinat = np.array([[1.0, 1.0, np.sqrt(2.0)], [1.0, 1.0, np.sqrt(2.0)]])
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), groups=[[1], [0]]))
    assert not res.valid
    assert res.chern is None
    assert res.plaquette_flux is None
    assert res.vertex_curvature is None
    assert res.groups[0]["bands"] == [1]
    assert res.groups[0]["min_external_gap"] > 0.0
    for group in res.groups:
        assert group["valid"] is False
        assert group["chern"] is None
        assert group["plaquette_flux"] is None
        assert group["vertex_curvature"] is None
    record = res.to_json_dict()
    assert all(g["plaquette_flux"] is None for g in record["groups"])
    assert any("nonstationary" in f for f in res.diagnostics["failures"])


def test_degenerate_zero_spectrum_refuses_groups_without_vectors():
    # A lone spin with only a site-local exchange gives a vanishing
    # Hamiltonian: every mesh point is an all-zero spectrum, so the solver
    # emits no positive-mode vectors or index map. This is a structural
    # fixture, not a solver-criterion artifact, and the public API must
    # refuse instead of fabricating an invariant or indexing None.
    atoms = Atoms(
        "Cr",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.0, 1.0, 10.0]),
        pbc=True,
    )
    exc = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=np.zeros(1),
        index_spin=[0],
        colinear=False,
        exchange_Jdict={((0, 0, 0), 0, 0): 0.001},
    )
    exc._build_Rlist()
    model = prepare_topology_model(exc, source_path="bench")

    res = calculate_bulk(model, BulkRequest(mesh=(4, 4), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.plaquette_flux is None
    assert res.groups[0]["valid"] is False
    assert res.groups[0]["plaquette_flux"] is None
    assert res.diagnostics["missing_mode_points"] == list(range(16))
    assert any("validated mode vectors" in f for f in res.diagnostics["failures"])

    # The default all-positive request must also refuse rather than iterate
    # a missing positive-index map.
    default = calculate_bulk(model, BulkRequest(mesh=(4, 4)))
    assert not default.valid
    assert default.groups == []
    assert any("default all-positive" in f for f in default.diagnostics["failures"])


def test_zero_mode_mesh_keeps_isolated_upper_band_valid():
    # Without easy-axis SIA the Goldstone mode survives at Gamma, while the
    # DMI opens the valley gaps: the non-touching upper band stays usable.
    model = prepare_topology_model(
        honeycomb_spinio(0.0005, k_sia=0.0), source_path="bench"
    )
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[1]))
    assert res.valid, res.diagnostics["failures"]
    assert res.chern == -1
    assert res.plaquette_flux is not None
    assert res.groups[0]["valid"] is True
    assert res.diagnostics["zero_mode_points"] == [0]


def test_unstable_spectrum_refused_without_shift():
    # FM couplings with antiparallel moments: transverse gradient vanishes
    # (collinear torque-free) but the Hessian is not positive definite.
    exc = honeycomb_spinio(0.0005)
    exc.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.energies.shape[-1] == 0
    assert any("unstable" in f for f in res.diagnostics["failures"])


def test_band_index_out_of_range(model_pos):
    with pytest.raises(ValueError, match="out of range"):
        calculate_bulk(model_pos, BulkRequest(mesh=(6, 6), bands=[2]))


def test_identity_embedding_map(model_pos):
    from TB2J.magnon.topology import identity_embedding

    emb = identity_embedding()
    k = emb.to_input_k(np.array([[0.25, 0.5]]))
    np.testing.assert_allclose(k[0], [0.25, 0.5, 0.0], atol=1e-15)


def test_oblique_embedding_uses_reciprocal_dual_and_preserves_chern(model_pos):
    from TB2J.magnon.topology.embedding import LatticeEmbedding

    embedding = LatticeEmbedding(
        basis=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        normal_k=0.25,
    )
    np.testing.assert_array_equal(
        embedding.reciprocal_dual,
        [[1, 0, 0], [-1, 1, 0], [0, 0, 1]],
    )
    np.testing.assert_allclose(
        embedding.to_input_k([0.1, 0.2]),
        [[-0.1, 0.2, 0.25]],
        atol=1e-15,
    )

    identity = calculate_bulk(model_pos, BulkRequest(mesh=(6, 6), bands=[0]))
    oblique = calculate_bulk(
        model_pos,
        BulkRequest(mesh=(6, 6), bands=[0], embedding=embedding),
    )
    assert identity.valid and oblique.valid
    assert identity.chern == oblique.chern
    np.testing.assert_allclose(
        np.sort(identity.energies.ravel()),
        np.sort(oblique.energies.ravel()),
        atol=1e-12,
    )


def test_fixed_normal_momentum_retains_normal_translation_couplings():
    from TB2J.magnon.topology.embedding import LatticeEmbedding

    exc = honeycomb_spinio(0.0005)
    for i in range(2):
        exc.exchange_Jdict[((0, 0, 1), i, i)] = 0.003
        exc.exchange_Jdict[((0, 0, -1), i, i)] = 0.003
    exc._build_Rlist()
    model = prepare_topology_model(exc, source_path="normal-coupling-test")
    at_gamma = calculate_bulk(
        model,
        BulkRequest(
            mesh=(6, 6),
            bands=[0],
            embedding=LatticeEmbedding(np.eye(3, dtype=int), normal_k=0.0),
        ),
    )
    at_quarter = calculate_bulk(
        model,
        BulkRequest(
            mesh=(6, 6),
            bands=[0],
            embedding=LatticeEmbedding(np.eye(3, dtype=int), normal_k=0.25),
        ),
    )
    assert at_gamma.valid and at_quarter.valid
    assert not np.allclose(at_gamma.energies, at_quarter.energies)
    record = at_quarter.to_json_dict()
    assert record["embedding"]["normal_k"] == 0.25
    assert [0, 0, 1] in record["model_snapshot"]["Rlist"]
    assert record["model_snapshot"]["JR_eV"]


def test_cli_topology_mode(tmp_path):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res"
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
        "--topology-basis",
        "1",
        "1",
        "0",
        "0",
        "1",
        "0",
        "0",
        "0",
        "1",
        "--topology-normal-k",
        "0.25",
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
    assert data["valid"] is True
    assert data["chern"] == 1
    assert data["embedding"]["basis"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert data["embedding"]["reciprocal_dual"] == [
        [1, 0, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ]
    assert data["embedding"]["normal_k"] == 0.25
    assert data["model_snapshot"]["reference_magmoms_muB"]


def test_cli_topology_invalid_exit(tmp_path):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0)
    root = tmp_path / "res_gapless"
    exc.write_pickle(str(root))
    out = tmp_path / "topo_gapless.json"
    argv = sys.argv
    # 12x12 mesh contains the K/K' valleys at 1/3 fillings
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
    assert exit_info.value.code == 1
    data = json.loads(out.read_text())
    assert data["valid"] is False
    assert data["chern"] is None
    out.write_text("preserve the previous diagnostic")
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
        with pytest.raises(SystemExit) as second_exit:
            main()
    finally:
        sys.argv = argv
    assert second_exit.value.code == 1
    assert out.read_text() == "preserve the previous diagnostic"


def test_curvature_field_concentrates_at_valleys(model_pos):
    # Independent check of the local curvature observable: the gapped Dirac
    # cones at K=(1/3,2/3) and K'=(2/3,1/3) must dominate the curvature map,
    # and the vertex sum must integrate to the Chern number.
    res = calculate_bulk(model_pos, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid
    vertex = res.vertex_curvature
    assert abs(vertex.sum() - res.chern) < 1e-9
    valley_cells = {(4, 8), (8, 4)}  # 12*(1/3)=4, 12*(2/3)=8 (i,j) cells
    others = [
        (i, j, abs(vertex[i, j]))
        for i in range(12)
        for j in range(12)
        if (i, j) not in valley_cells
    ]
    max_other = max(v for _, _, v in others)
    for cell in valley_cells:
        assert abs(vertex[cell]) > max_other


def test_cli_refuses_unsupported_channels(tmp_path, capsys):
    # A pickle carrying nonzero biquadratic exchange must be refused by the
    # strict topology selection inside the real CLI (nonzero exit).
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    exc.biquadratic_Jdict = {(0, 0, 0, 0, 0): 0.001}
    root = tmp_path / "res_biquad"
    exc.write_pickle(str(root))
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "6",
        "6",
        "--topology-out",
        str(tmp_path / "topo.json"),
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 2
    assert "biquadratic" in capsys.readouterr().out


def test_nonprimitive_embedding_refused():
    from TB2J.magnon.topology import LatticeEmbedding

    with pytest.raises(ValueError, match="nonprimitive"):
        LatticeEmbedding(basis=2 * np.eye(3, dtype=int))


def test_vanishing_width_link_refuses_invariant_without_chern():
    atoms = Atoms(
        "Cr2", positions=[[0, 0, 0], [1.5, 1.5, 0]],
        cell=3 * np.eye(3), pbc=True,
    )
    exc = SpinIO(
        atoms=atoms, spinat=np.array([[0, 0, 2.0], [0, 0, 2.0]]),
        charges=np.zeros(2), index_spin=[0, 1], colinear=False,
        exchange_Jdict={
            ((0, 1, 0), 0, 1): 0.01,
            ((0, -1, 0), 1, 0): 0.01,
        },
        sia_tensor={
            0: np.diag([0.0, 0.0, 0.001]),
            1: np.diag([0.0, 0.0, 0.001]),
        },
    )
    exc._build_Rlist()
    res = calculate_bulk(prepare_topology_model(exc), BulkRequest(mesh=(2, 2), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.plaquette_flux is None
    assert res.groups[0]["min_det_overlap"] < 1e-8
    assert any("link determinant" in reason for reason in res.diagnostics["failures"])


def test_cli_rejects_simultaneous_bands_and_groups(tmp_path, capsys):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_conflict"
    exc.write_pickle(str(root))
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-bands",
        "0",
        "--topology-groups",
        "0;1",
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_cli_refuses_unsupported_ordering_vector(tmp_path, capsys):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_ordering"
    exc.write_pickle(str(root))
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "6",
        "6",
        "--ordering-vector",
        "0.5",
        "0",
        "0",
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 2
    assert "--ordering-vector" in capsys.readouterr().out


def test_cli_refuses_rotation_axis_without_ordering_vector(tmp_path, capsys):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_axis"
    exc.write_pickle(str(root))
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "6",
        "6",
        "--rotation-axis",
        "0",
        "0",
        "1",
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 2
    assert "--rotation-axis" in capsys.readouterr().out


def test_cli_refuses_unsupported_local_axes(tmp_path, capsys, monkeypatch):
    import sys

    from TB2J.magnon.magnon_cli import main

    root = tmp_path / "results"
    honeycomb_spinio(0.0005).write_pickle(str(root))
    monkeypatch.setattr(
        sys, "argv",
        ["TB2J_magnon", "-p", str(root), "--topology", "--uz-file", "axes"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--uz-file" in capsys.readouterr().out


def test_cli_applies_spin_conf_reference_moments(tmp_path):
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_spinconf"
    exc.write_pickle(str(root))
    out = tmp_path / "topo_spinconf.json"
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "6",
        "6",
        "--spin-conf",
        "0",
        "0",
        "4",
        "0",
        "0",
        "4",
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
    np.testing.assert_allclose(
        data["energies_eV"][0][0], [K_SIA, 6 * J1 + K_SIA], atol=1e-10
    )


def test_cli_standalone_bulk_input_error_is_refused(tmp_path, capsys):
    # An invalid bulk request must be refused with the paired route's exit
    # code and message instead of raising out of the CLI.
    import sys

    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(0.0005)
    root = tmp_path / "res_badband"
    exc.write_pickle(str(root))
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "6",
        "6",
        "--topology-bands",
        "5",
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 2
    assert "out of range" in capsys.readouterr().out
