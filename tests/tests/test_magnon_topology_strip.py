"""Open-strip construction, physical localization and consumer regressions."""

import json
import sys

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import (
    LatticeEmbedding,
    StripRequest,
    calculate_strip,
    expand_magnetic_cell,
    prepare_topology_model,
    strip_site_map,
)
from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian
from TB2J.magnon.topology.bosonic import solve_bosonic_modes
from TB2J.magnon.topology.strip import (
    _assemble_strip_bonds,
    _projected_fraction_spectrum,
)

J1 = 0.01
J2 = 0.002
K_SIA = 0.001


def honeycomb_spinio(
    dmi_z=0.0005,
    normal_coupling=0.0,
    sia_z=(K_SIA, K_SIA),
):
    cell = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    atoms = Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, -1 / 3, 0.0]],
        cell=cell,
        pbc=True,
    )
    moments = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    exchange = {}
    dmi = {}

    def reverse(R):
        return tuple(-np.asarray(R))

    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        exchange[(R, 0, 1)] = J1
        exchange[(reverse(R), 1, 0)] = J1
    for R in [(1, 0, 0), (-1, 1, 0), (0, -1, 0)]:
        for spin in (0, 1):
            exchange[(R, spin, spin)] = J2
            exchange[(reverse(R), spin, spin)] = J2
        dmi[(R, 0, 0)] = np.array([0.0, 0.0, dmi_z])
        dmi[(reverse(R), 0, 0)] = np.array([0.0, 0.0, -dmi_z])
        dmi[(R, 1, 1)] = np.array([0.0, 0.0, -dmi_z])
        dmi[(reverse(R), 1, 1)] = np.array([0.0, 0.0, dmi_z])
    if normal_coupling:
        for spin in (0, 1):
            exchange[((0, 0, 1), spin, spin)] = normal_coupling
            exchange[((0, 0, -1), spin, spin)] = normal_coupling
    exc = SpinIO(
        atoms=atoms,
        spinat=moments,
        charges=np.zeros(2),
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exchange,
        dmi_ddict=dmi if dmi_z else None,
        sia_tensor={
            0: np.diag([0.0, 0.0, sia_z[0]]),
            1: np.diag([0.0, 0.0, sia_z[1]]),
        },
    )
    exc._build_Rlist()
    return exc


def chain_spinio(dmi_y=0.0, exchange_j=0.01):
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.diag([1.0, 5.0, 5.0]), pbc=True)
    exchange = {
        ((1, 0, 0), 0, 0): exchange_j,
        ((-1, 0, 0), 0, 0): exchange_j,
    }
    dmi = None
    if dmi_y:
        dmi = {
            ((1, 0, 0), 0, 0): np.array([0.0, dmi_y, 0.0]),
            ((-1, 0, 0), 0, 0): np.array([0.0, -dmi_y, 0.0]),
        }
    exc = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=np.zeros(1),
        index_spin=[0],
        colinear=False,
        exchange_Jdict=exchange,
        dmi_ddict=dmi,
        sia_tensor={0: np.diag([0.0, 0.0, 0.001])},
    )
    exc._build_Rlist()
    return exc


def test_open_strip_recomputes_boundary_field_and_requires_stationary_reference():
    embedding = LatticeEmbedding(basis=[[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    stationary_model = prepare_topology_model(chain_spinio())
    stationary = calculate_strip(
        stationary_model, StripRequest(width=4, kpoints=7, embedding=embedding)
    )
    assert stationary.valid, stationary.diagnostics["failures"]
    fields = stationary.local_field[:, 2]
    left = stationary.edge_regions["left"]["site_indices"][0]
    interior = stationary.edge_regions["interior_site_indices"][0]
    assert not np.isclose(fields[left], fields[interior])
    assert stationary.diagnostics["stationarity"]["stationary"]
    assert stationary.diagnostics["retained_directed_bond_records"] > 0
    assert stationary.diagnostics["cut_directed_bond_records"] > 0

    torqued_model = prepare_topology_model(chain_spinio(dmi_y=0.001))
    torqued = calculate_strip(
        torqued_model, StripRequest(width=4, kpoints=7, embedding=embedding)
    )
    assert not torqued.valid
    assert torqued.energies is None
    assert not torqued.diagnostics["stationarity"]["stationary"]
    assert any(
        "explicitly corrected" in reason for reason in torqued.diagnostics["failures"]
    )

    # A supplied reference is retained exactly; no implicit edge relaxation occurs.
    sites = strip_site_map(torqued_model, width=4, embedding=embedding)
    explicit = np.asarray(
        [torqued_model.magmoms[site["source_spin"]] for site in sites]
    )
    refused = calculate_strip(
        torqued_model,
        StripRequest(
            width=4, kpoints=7, embedding=embedding, reference_magmoms=explicit
        ),
    )
    assert refused.diagnostics["reference_source"] == "explicit_strip_reference"
    assert not refused.valid


def test_periodic_restoration_matches_folded_bulk_with_oblique_and_normal_bonds():
    model = prepare_topology_model(
        honeycomb_spinio(dmi_z=0.0005, normal_coupling=0.003)
    )
    embedding = LatticeEmbedding(basis=[[1, 1, 0], [0, 1, 0], [0, 0, 1]], normal_k=0.25)
    open_result = calculate_strip(
        model,
        StripRequest(width=3, kpoints=5, embedding=embedding),
    )
    assert open_result.valid, open_result.diagnostics["failures"]
    assert [0, 0, 1] in open_result.diagnostics["retained_translations_input_cell"]
    width = 3
    JR_strip, R_strip, sites, _retained, _cut = _assemble_strip_bonds(
        model,
        width,
        embedding,
        (0.0, 0.0),
        periodic_width=True,
    )
    moments_strip = np.asarray([model.magmoms[site["source_spin"]] for site in sites])
    edge_k = 0.173
    strip_k = embedding.to_input_k([[edge_k, 0.0, embedding.normal_k]])
    strip_modes = solve_bosonic_modes(
        local_frame_bosonic_hamiltonian(JR_strip, R_strip, moments_strip, strip_k)[0]
    )
    folded = []
    for width_index in range(width):
        bulk_k = embedding.to_input_k(
            [[edge_k, width_index / width, embedding.normal_k]]
        )
        bulk_modes = solve_bosonic_modes(
            local_frame_bosonic_hamiltonian(
                model.JR, model.Rlist, model.magmoms, bulk_k
            )[0]
        )
        folded.extend(bulk_modes.energies.tolist())
    assert open_result.to_json_dict()["embedding"]["normal_k_fractional"] == 0.25
    np.testing.assert_allclose(
        np.sort(strip_modes.energies), np.sort(folded), rtol=1e-10, atol=1e-11
    )


def test_periodic_strip_matches_bulk_with_unequal_spin_moments():
    from TB2J.magnon.topology import BulkRequest, calculate_bulk

    exc = honeycomb_spinio(dmi_z=0.0, sia_z=(K_SIA, 1.5 * K_SIA))
    exc.spinat = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 3.0]])
    model = prepare_topology_model(exc)
    embedding = LatticeEmbedding(np.eye(3, dtype=int))
    bulk = calculate_bulk(
        model,
        BulkRequest(mesh=(4, 4), bands=[0], embedding=embedding),
    )
    assert bulk.valid, bulk.diagnostics["failures"]

    width = 2
    JR_strip, R_strip, sites, *_ = _assemble_strip_bonds(
        model, width, embedding, (0.0, 0.0), periodic_width=True
    )
    moments = np.asarray([model.magmoms[site["source_spin"]] for site in sites])
    strip = solve_bosonic_modes(
        local_frame_bosonic_hamiltonian(
            JR_strip, R_strip, moments, np.array([[0.0, 0.0, 0.0]])
        )[0]
    ).energies
    bulk_kpoints = bulk.kpoints.reshape(-1, 3)
    bulk_energies = bulk.energies.reshape(-1, model.nspin)
    folded = []
    for width_index in range(width):
        target = embedding.to_input_k([[0.0, width_index / width, 0.0]])[0]
        matches = np.flatnonzero(
            np.all(np.isclose(bulk_kpoints, target, rtol=0.0, atol=1e-12), axis=1)
        )
        assert len(matches) == 1
        folded.extend(bulk_energies[matches[0]].tolist())
    np.testing.assert_allclose(np.sort(strip), np.sort(folded), rtol=1e-10, atol=1e-11)


def test_degenerate_projected_physical_edge_observable_is_unitary_invariant():
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    spin_lengths = np.array([0.5, 1.0, 1.5])
    left_region = np.array([True, False, False])
    projected = _projected_fraction_spectrum(
        vectors, spin_lengths, left_region.astype(float)
    )
    unitary, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    mixed = _projected_fraction_spectrum(
        vectors @ unitary, spin_lengths, left_region.astype(float)
    )
    np.testing.assert_allclose(projected, mixed, rtol=1e-12, atol=1e-12)


def test_strip_json_has_physical_normalized_localization_and_provenance():
    model = prepare_topology_model(honeycomb_spinio(dmi_z=0.0005))
    result = calculate_strip(model, StripRequest(width=4, kpoints=9))
    assert result.valid, result.diagnostics["failures"]
    assert result.energies.shape == (9, len(result.site_map))
    assert set(site["region"] for site in result.site_map) >= {
        "left_edge",
        "right_edge",
        "interior",
    }
    total = (
        result.left_edge_fraction
        + result.right_edge_fraction
        + result.interior_fraction
    )
    resolved = np.isfinite(total)
    np.testing.assert_allclose(total[resolved], 1.0, atol=1e-12)
    assert result.diagnostics["interior_reference_mismatch"]["matches_bulk_reference"]

    record = result.to_json_dict()
    encoded = json.dumps(record, allow_nan=False)
    round_trip = json.loads(encoded)
    assert round_trip["kind"] == "strip"
    assert round_trip["schema_version"] == "tb2j-topology.strip/1"
    assert round_trip["parent_model_digest"] == model.digest
    assert round_trip["reference"]["site_magmoms_muB"]
    assert round_trip["geometry"]["edge_regions"]["left"]["site_indices"]
    assert round_trip["geometry"]["edge_regions"]["right"]["site_indices"]


def test_actual_cli_runs_nontrivial_and_trivial_strip_controls(tmp_path, monkeypatch):
    from TB2J.magnon.magnon_cli import main

    outputs = []
    for label, dmi, width, termination, sia_z in (
        ("nontrivial", 0.0005, 4, (0.0, 0.0), (K_SIA, K_SIA)),
        ("trivial", 0.0, 5, (0.25, 0.25), (K_SIA, 1.5 * K_SIA)),
    ):
        source = tmp_path / f"{label}_results"
        honeycomb_spinio(dmi_z=dmi, sia_z=sia_z).write_pickle(str(source))
        output = tmp_path / f"{label}_strip.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon",
                "-p",
                str(source),
                "--topology",
                "--topology-strip-width",
                str(width),
                "--topology-strip-kpoints",
                "9",
                "--topology-strip-termination",
                str(termination[0]),
                str(termination[1]),
                "--topology-out",
                str(output),
            ],
        )
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 0
        record = json.loads(output.read_text())
        assert record["kind"] == "strip"
        assert record["valid"] is True
        assert record["request"]["width_cells"] == width
        assert record["parent_model_digest"]
        assert record["energies_eV"]
        outputs.append(record)
    assert (
        outputs[0]["model_snapshot"]["JR_eV"] != outputs[1]["model_snapshot"]["JR_eV"]
    )
    assert outputs[0]["geometry"]["site_count"] != outputs[1]["geometry"]["site_count"]


def test_invalid_strip_cli_does_not_overwrite_existing_output(tmp_path, monkeypatch):
    from TB2J.magnon.magnon_cli import main

    source = tmp_path / "torqued_results"
    chain_spinio(dmi_y=0.001).write_pickle(str(source))
    output = tmp_path / "kept.json"
    output.write_text("preserve this result")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "TB2J_magnon",
            "-p",
            str(source),
            "--topology",
            "--topology-strip-width",
            "4",
            "--topology-strip-kpoints",
            "5",
            "--topology-basis",
            "0",
            "1",
            "0",
            "1",
            "0",
            "0",
            "0",
            "0",
            "-1",
            "--topology-out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert output.read_text() == "preserve this result"


def test_stable_anomalous_bosonic_terms_use_the_shared_strip_solver(
    tmp_path, monkeypatch
):
    exc = honeycomb_spinio(dmi_z=0.0005)
    anisotropic = {}
    pair_tensor = 0.0003 * np.diag([1.0, -1.0, 0.0])
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        reverse = tuple(-np.asarray(R))
        anisotropic[(R, 0, 1)] = pair_tensor
        anisotropic[(reverse, 1, 0)] = pair_tensor.T
    exc.Jani_dict = anisotropic
    exc._build_Rlist()
    model = prepare_topology_model(exc)
    result = calculate_strip(model, StripRequest(width=4, kpoints=7))
    assert result.valid, result.diagnostics["failures"]

    JR, Rlist, sites, _retained, _cut = _assemble_strip_bonds(
        model, 4, LatticeEmbedding(np.eye(3, dtype=int)), (0.0, 0.0)
    )
    moments = np.asarray([model.magmoms[site["source_spin"]] for site in sites])
    H = local_frame_bosonic_hamiltonian(
        JR, Rlist, moments, np.array([[0.173, 0.0, 0.0]])
    )[0]
    nsite = len(sites)
    assert np.max(np.abs(H[:nsite, nsite:])) > 1e-8
    source = tmp_path / "anomalous_results"
    exc.write_pickle(str(source))
    output = tmp_path / "anomalous_strip.json"
    from TB2J.magnon.magnon_cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "TB2J_magnon",
            "-p",
            str(source),
            "--topology",
            "--topology-strip-width",
            "4",
            "--topology-strip-kpoints",
            "7",
            "--topology-out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert json.loads(output.read_text())["valid"] is True


def test_explicit_magnetic_cell_strip_restores_folded_spectrum():
    primitive = prepare_topology_model(chain_spinio(exchange_j=-0.01))
    magnetic = expand_magnetic_cell(primitive, Q=(0.5, 0.0, 0.0), axis=(0.0, 1.0, 0.0))
    embedding = LatticeEmbedding(basis=[[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    open_strip = calculate_strip(
        magnetic,
        StripRequest(width=3, kpoints=5, embedding=embedding),
    )
    assert open_strip.valid, open_strip.diagnostics["failures"]
    assert open_strip.model.digest == magnetic.digest
    width = 3
    JR_strip, R_strip, sites, _retained, _cut = _assemble_strip_bonds(
        magnetic,
        width,
        embedding,
        (0.0, 0.0),
        periodic_width=True,
    )
    strip_moments = np.asarray(
        [magnetic.magmoms[site["source_spin"]] for site in sites]
    )
    edge_k = 0.19
    strip_k = embedding.to_input_k([[edge_k, 0.0, 0.0]])
    strip_energies = solve_bosonic_modes(
        local_frame_bosonic_hamiltonian(JR_strip, R_strip, strip_moments, strip_k)[0]
    ).energies
    folded = []
    for width_index in range(width):
        bulk_k = embedding.to_input_k([[edge_k, width_index / width, 0.0]])
        modes = solve_bosonic_modes(
            local_frame_bosonic_hamiltonian(
                magnetic.JR, magnetic.Rlist, magnetic.magmoms, bulk_k
            )[0]
        )
        folded.extend(modes.energies.tolist())
    np.testing.assert_allclose(
        np.sort(strip_energies), np.sort(folded), rtol=1e-10, atol=1e-11
    )
    assert magnetic.magnetic_cell_transform["folded_mode_count"] == 2
    assert (
        open_strip.to_json_dict()["provenance"]["magnetic_cell_transform"][
            "folded_mode_count"
        ]
        == 2
    )
