"""Installed Siesta spiral CLI contract, through subprocess and SpinIO seams."""

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from ase.units import Bohr

from TB2J.io_exchange import SpinIO


def _run_cli(*arguments):
    executable = shutil.which("tb2j-siesta-spiral")
    assert executable is not None, "Install TB2J to provide tb2j-siesta-spiral"
    return subprocess.run(
        [executable, *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_installed_siesta_spiral_help():
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "manifest" in result.stdout.lower()
    assert "--output" in result.stdout


FIXTURE_ROOT = Path(__file__).parent / "siesta_spiral_fixtures"
REAL_BUNDLE = FIXTURE_ROOT / "siesta_collinear_o2" / "o2_fm_q0.npz"
MANDATORY_GATES = ["AC-003", "AC-005", "AC-007"]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path, manifest_directory):
    return {
        "path": str(path.relative_to(manifest_directory))
        if path.is_relative_to(manifest_directory)
        else str(path),
        "sha256": _sha256(path),
    }


def _write_manifest(directory, manifest):
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _manifest(
    directory,
    bundle,
    structure,
    *,
    orbitals,
    normals,
    axes,
    ensemble,
    contour,
    producer,
):
    return {
        "version": 1,
        "inputs": {
            "bundle": _descriptor(bundle, directory),
            "structure": _descriptor(structure, directory),
        },
        "producer": producer,
        "operator": {"mode": "weighted_total_h", "orbitals_by_site": orbitals},
        "axes_by_site": {site: deepcopy(axes) for site in orbitals},
        "ensemble": ensemble,
        "response": {
            "sampling": "recorded",
            "finite_steps": [4e-3, 2e-3],
            "contour": contour,
            "contour_node_rungs": [contour["n_nodes"] // 2, contour["n_nodes"]],
        },
        "mapping": {"reference_directions_by_site": normals},
        "spin": {
            "spinat": list(normals.values()),
            "charges": [0.0, 0.0],
            "index_spin": [0, 1],
        },
        "required_gates": list(MANDATORY_GATES),
        "requires_plusu": False,
    }


@pytest.fixture
def analytic_manifest(tmp_path):
    """A complete pencil, not a precomputed response or mocked export.

    H = -sigma_z on both sites plus spin-independent hopping t=1/5,
    S=I, N=2. For relative angle theta its occupied energy is
    -sqrt(1+t^2+2t*cos(theta/2))-sqrt(1+t^2-2t*cos(theta/2)).
    Thus J_pair_once=-t^2/(2*(1-t^2))=-1/48 and J_stored=-1/96 eV.
    """
    bundle = tmp_path / "analytic.npz"
    np.savez(
        bundle,
        fixture_version="1.0",
        bundle_version="1.1",
        norb=2,
        sc_off=np.array([[0, 0, 0]]),
        q_cart=np.zeros(3),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        mu=np.array([0, 0, 1, 1]),
        nu=np.array([0, 1, 0, 1]),
        isc=np.zeros(4, dtype=int),
        h=np.array([[-1, 1, 0, 0], [0.2, 0.2, 0, 0], [0.2, 0.2, 0, 0], [-1, 1, 0, 0]]),
        s=np.array([1.0, 0.0, 0.0, 1.0]),
        cell=np.eye(3) * 8.0,
        orbital_positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        source_label="analytic-collinear-two-site-pencil",
        ef_hsx=0.0,
        kp=np.zeros((1, 3)),
        kp_weight=np.ones(1),
        eig=np.array([[-1.2, -0.8, 0.8, 1.2]]),
        ef=0.0,
        electron_count=2.0,
        temperature=100.0,
    )
    structure = tmp_path / "structure.extxyz"
    write(
        structure,
        Atoms(
            "H2",
            positions=np.array([[0, 0, 0], [2, 0, 0]]) * Bohr,
            cell=np.eye(3) * 8 * Bohr,
            pbc=True,
        ),
    )
    return _manifest(
        tmp_path,
        bundle,
        structure,
        orbitals={"A": [0], "B": [1]},
        normals={"A": [0, 0, 1], "B": [0, 0, 1]},
        axes={"x": [1, 0, 0], "y": [0, 1, 0]},
        ensemble={
            "kind": "gapped_fixed_occupations",
            "n_occupied": 2,
            "electron_count": 2.0,
            "fermi_energy_ev": 0.0,
            "smearing_width_ev": 8.617333262e-3,
            "gap_ev": 1.6,
        },
        contour={
            "z0": 0.0,
            "occupied_min": -1.2,
            "gap_ev": 1.6,
            "margin_ev": 0.3,
            "n_nodes": 256,
        },
        producer={
            "kind": "analytic",
            "software_revisions": {"analytic_model": "two-site pencil v1"},
            "pseudopotentials": [],
            "basis_definitions": ["bundle"],
            "numerical_settings": {
                "onsite_spin_splitting_ev": 2.0,
                "spin_independent_hopping_ev": 0.2,
                "overlap": "identity",
                "electron_count": 2.0,
                "recorded_kpoints": [[0.0, 0.0, 0.0]],
            },
            "input_files": ["bundle"],
        },
    )


@pytest.fixture
def real_manifest(tmp_path):
    if not REAL_BUNDLE.is_file():
        pytest.skip("curated SIESTA q=0 O2 NPZ fixture is not installed")
    # This is the actual NC q=0 producer fixture from the mapping/anchor
    # tests, not the metallic Fe3 spiral. Use its RECORDED four-point mesh,
    # not the anchor's distinct five-point comparison mesh.
    with np.load(REAL_BUNDLE, allow_pickle=False) as data:
        n_orbitals = int(data["norb"])
        n_occupied = int(data["electron_count"])
        eig = data["eig"]
        homo = float(eig[:, n_occupied - 1].max())
        lumo = float(eig[:, n_occupied].min())
        gap = lumo - homo
        assert n_orbitals == 18
        assert n_occupied == 12
        assert gap == pytest.approx(2.49684889, abs=1e-8)
        np.testing.assert_array_equal(data["q_cart"], np.zeros(3))
        ensemble = {
            "kind": "gapped_fixed_occupations",
            "n_occupied": n_occupied,
            "electron_count": float(data["electron_count"]),
            "fermi_energy_ev": float(data["ef"]),
            "smearing_width_ev": 8.617333262e-5 * float(data["temperature"]),
            "gap_ev": gap,
        }
        contour = {
            "z0": (homo + lumo) / 2,
            "occupied_min": float(eig.min()),
            "gap_ev": gap,
            "margin_ev": 0.8,
            "n_nodes": 1024,
        }
    structure = tmp_path / "o2.extxyz"
    write(
        structure,
        Atoms(
            "O2", positions=[[2.895, 3.5, 2], [4.105, 3.5, 2]], cell=[7, 7, 4], pbc=True
        ),
    )
    manifest = _manifest(
        tmp_path,
        REAL_BUNDLE,
        structure,
        orbitals={"O1": list(range(9)), "O2": list(range(9, 18))},
        normals={"O1": [1, 0, 0], "O2": [1, 0, 0]},
        axes={"y": [0, 1, 0], "z": [0, 0, 1]},
        ensemble=ensemble,
        contour=contour,
        producer={
            "kind": "siesta",
            "software_revisions": {
                "siesta": "5.4.2-24-g71c860291",
                "siesta_executable_sha256": (
                    "98be67909dd7f0b17d73828630bc1b7d37119372fa64d182901bed788151bdd1"
                ),
            },
            "pseudopotentials": ["oxygen_psf"],
            "basis_definitions": ["producer_fdf"],
            "numerical_settings": {
                "xc": "PBE",
                "basis": "SZP",
                "mesh_cutoff_ry": 150,
                "kmesh": [1, 1, 4],
                "electronic_temperature_k": 100,
            },
            "input_files": ["producer_fdf", "producer_record"],
        },
    )
    for role, path in {
        "oxygen_psf": REAL_BUNDLE.parent / "nc" / "O.psf",
        "producer_fdf": REAL_BUNDLE.parent / "nc" / "o2_fm.fdf",
        "producer_record": REAL_BUNDLE.parent / "PROVENANCE.md",
    }.items():
        manifest["inputs"][role] = _descriptor(path, tmp_path)
    return manifest


def _assert_digest(value):
    assert isinstance(value, str) and len(value) == 64
    int(value, 16)


def _canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _assert_export(output, manifest_path, manifest):
    loaded = SpinIO.load_pickle(path=str(output))
    provenance = json.loads((output / "provenance.json").read_text())
    run = json.loads((output / "run.json").read_text())
    assert provenance["export_kind"] == "certified_chain"
    assert "spinio_kwargs" not in provenance
    assert provenance["units"]["energy"] == "eV"
    assert provenance["units"]["curvature"] == "eV/rad^2"
    assert provenance["reference_metadata"]["site_order"] == list(
        manifest["operator"]["orbitals_by_site"]
    )
    assert (
        provenance["vertex_provenance"]["operator_mode"] == manifest["operator"]["mode"]
    )
    for name in ("source_bundle_sha256", "vertex_sha256"):
        _assert_digest(provenance["vertex_provenance"][name])
    assert provenance["mapping"]["compatibility"]["label"] == "heisenberg_compatible"
    assert provenance["mapping"]["identifiability"]
    assert provenance["mapping"]["residuals"]["chi2_floor_eV_per_rad2"] == 1e-6
    assert provenance["mapping"]["residuals"]["torque_chi2_floor_eV_per_rad"] == 1e-6
    assert provenance["raw_response"]
    assert provenance["convergence_evidence"]["finite_rotation_cross_check"]["passed"]
    assert provenance["convergence_evidence"]["integration"]
    assert provenance["convergence_evidence"]["ladder"]
    assert provenance["ensemble"]["response"]["kind"] == manifest["ensemble"]["kind"]
    assert provenance["labels"]["uncertified_exchange_absent"] is False
    api = provenance["provenance"]
    assert api["created_utc"]
    assert api["code_revision"]
    _assert_digest(api["code_sha256"])
    _assert_digest(api["content_sha256"])
    assert api["content_sha256"] == _canonical_digest(
        {key: value for key, value in api.items() if key != "content_sha256"}
    )
    assert api["input_digests"]["response_set"] == _canonical_digest(
        provenance["raw_response"]
    )
    assert api["input_digests"]["mapping_output"] == _canonical_digest(
        provenance["mapping"]
    )
    gates = {
        report.get("gate", report.get("acceptance_criterion")): report
        for report in api["gate_verdicts"]
    }
    assert set(MANDATORY_GATES) <= gates.keys()
    for gate in MANDATORY_GATES:
        assert gates[gate].get("passed") is True or gates[gate].get("status") == "pass"
    cli = provenance["cli"]
    assert cli["producer"] == manifest["producer"]
    assert cli["manifest_sha256"] == _sha256(manifest_path)
    assert cli["input_digests"] == {
        role: descriptor["sha256"] for role, descriptor in manifest["inputs"].items()
    }
    assert cli["code_revision"]
    for key in ("settings_sha256", "code_sha256", "reproducibility_sha256"):
        _assert_digest(cli[key])
    assert run["created_utc"]
    assert run["output_sha256"] == {
        name: _sha256(output / name) for name in ("TB2J.pickle", "provenance.json")
    }
    assert loaded.exchange_Jdict
    assert set(loaded.exchange_Jdict) == {
        ((0, 0, 0), 0, 1),
        ((0, 0, 0), 1, 0),
    }
    for pair in provenance["mapping"]["parameters"].values():
        assert loaded.get_J(0, 1, (0, 0, 0)) == pytest.approx(pair["j_stored_eV"])
    assert loaded.colinear is False
    assert loaded.dmi_ddict is None
    assert loaded.Jani_dict is None
    assert loaded.dJdx is None
    np.testing.assert_allclose(loaded.spinat, manifest["spin"]["spinat"])
    np.testing.assert_allclose(loaded.charges, manifest["spin"]["charges"])
    assert list(loaded.index_spin) == manifest["spin"]["index_spin"]
    return loaded, provenance, run


def test_analytic_pencil_exports_known_ordered_pair_exchange(
    tmp_path, analytic_manifest
):
    manifest_path = _write_manifest(tmp_path, analytic_manifest)
    output = tmp_path / "result"
    result = _run_cli(manifest_path, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded, _, _ = _assert_export(output, manifest_path, analytic_manifest)
    assert loaded.get_J(0, 1, (0, 0, 0)) == pytest.approx(-1 / 96, abs=1e-8)
    assert loaded.get_J(1, 0, (0, 0, 0)) == pytest.approx(-1 / 96, abs=1e-8)


def test_analytic_hsx_displacement_exports_periodic_image_sum_gauge(
    tmp_path, analytic_manifest
):
    """Real HSX serialization and a subprocess certify a nonzero dV/dR.

    These endpoints are analytic Hamiltonians, not claimed SIESTA SCFs.
    The reference remains the two-site NPZ pencil; only atom B moves, by
    +/-0.01 Bohr, and its onsite spin splitting changes linearly with x.
    Transverse slopes also exercise HSX's H4=-Im(H12) packing.
    """
    sisl = pytest.importorskip("sisl")
    step_bohr = 0.01
    expected_derivative = np.array([-0.3, 0.3, 0.07, -0.11])
    endpoint_roles = ("displacement_plus", "displacement_minus")
    for role, sign in zip(endpoint_roles, (1, -1)):
        displacement_ang = sign * step_bohr * Bohr
        positions_ang = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * Bohr
        positions_ang[1, 0] += displacement_ang
        geometry = sisl.Geometry(
            positions_ang,
            atoms=sisl.Atom("H", R=1.0),
            lattice=sisl.Lattice(np.eye(3) * 8 * Bohr, nsc=[1, 1, 1]),
        )
        hamiltonian = sisl.Hamiltonian(
            geometry, spin="non-collinear", orthogonal=False, dtype=np.float64
        )
        for orbital in (0, 1):
            hamiltonian[orbital, orbital, hamiltonian.M11] = -1.0
            hamiltonian[orbital, orbital, hamiltonian.M22] = 1.0
            hamiltonian.S[orbital, orbital] = 1.0
        hamiltonian[1, 1, hamiltonian.M11] += expected_derivative[0] * displacement_ang
        hamiltonian[1, 1, hamiltonian.M22] += expected_derivative[1] * displacement_ang
        hamiltonian[1, 1, hamiltonian.M12r] = expected_derivative[2] * displacement_ang
        # sisl's public component is Im(H12), opposite to the file H4.
        hamiltonian[1, 1, hamiltonian.M12i] = -expected_derivative[3] * displacement_ang
        for first, second in ((0, 1), (1, 0)):
            hamiltonian[first, second, hamiltonian.M11] = 0.2
            hamiltonian[first, second, hamiltonian.M22] = 0.2
            hamiltonian.S[first, second] = 0.0
        endpoint = tmp_path / f"{role}.HSX"
        # The supported HSX v2 writer stores an actual Ef=0 header and
        # four NC components plus explicit overlap, matching the loader.
        hamiltonian.write(str(endpoint), version=2, dtype=np.float64)
        assert sisl.get_sile(str(endpoint)).read_fermi_level() == pytest.approx(0.0)
        analytic_manifest["inputs"][role] = _descriptor(endpoint, tmp_path)

    analytic_manifest["operator"]["mode"] = "onsite_delta"
    analytic_manifest["required_gates"].append("AC-009")
    analytic_manifest["displacement"] = {
        "pattern": {"atom_index": 2, "axis": "x", "step_bohr": step_bohr},
        "gauge": "periodic-image-sum",
        "reference_fermi_ev": 0.0,
    }
    producer = analytic_manifest["producer"]
    producer["software_revisions"]["sisl"] = sisl.__version__
    producer["input_files"].extend(endpoint_roles)
    producer["numerical_settings"]["displacement_model"] = {
        "kind": "analytic_linear_onsite",
        "atom_index": 2,
        "packed_dH_ev_per_ang": expected_derivative.tolist(),
        "hsx_version": 2,
        "hsx_fermi_energy_ev": 0.0,
    }
    manifest_path = _write_manifest(tmp_path, analytic_manifest)
    output = tmp_path / "displacement-result"
    result = _run_cli(manifest_path, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded, provenance, _ = _assert_export(output, manifest_path, analytic_manifest)
    assert loaded.get_J(0, 1, (0, 0, 0)) == pytest.approx(-1 / 96, abs=1e-8)
    assert loaded.dJdx is None  # dV/dR is not an exchange derivative.
    derivative = provenance["displacement"]
    assert derivative["operator_mode"] == "onsite_delta"
    assert derivative["partition"] == "fixed"
    assert derivative["partition_declared"] is True
    assert derivative["gauge"] == "periodic-image-sum"
    assert derivative["units"] == {"dV": "eV/Ang"}
    assert provenance["labels"]["gauge"] == "periodic-image-sum"
    assert derivative["step_bohr"] == step_bohr
    assert derivative["pattern"]["atom_index"] == 2
    assert derivative["pattern"]["axis"] == "x"
    assert derivative["branch"] == "central"
    assert provenance["provenance"]["input_digests"][
        "displacement"
    ] == _canonical_digest(derivative)
    expected_blocks = {
        json.dumps([0, 0, [0, 0, 0]]): np.zeros(4),
        json.dumps([1, 1, [0, 0, 0]]): expected_derivative,
    }
    assert set(derivative["dV"]) == set(expected_blocks)
    for key, expected in expected_blocks.items():
        np.testing.assert_allclose(derivative["dV"][key], expected, atol=1e-8, rtol=0)
    gates = provenance["provenance"]["gate_verdicts"]
    ac009 = next(
        report for report in gates if report.get("acceptance_criterion") == "AC-009"
    )
    assert ac009["status"] == "pass"
    assert ac009["max_deviation_ev_per_ang"] <= ac009["tolerance_ev_per_ang"]
    assert ac009["operator_mode"] == "onsite_delta"
    assert ac009["pattern"] == {
        "atom_index": 2,
        "axis": "x",
        "step_bohr": step_bohr,
        "branch": "central",
        "direction": [1.0, 0.0, 0.0],
    }
    assert ac009["gauge"] == {
        name: "periodic-image-sum"
        for name in (
            "requested",
            "seam_record",
            "vertex_derivative_record",
            "displaced_reference_route",
        )
    }
    assert provenance["labels"]["ac009_gauge"] == ac009["gauge"]
    assert ac009["geometry"]["reference_digests"] == [
        f"sha256:{analytic_manifest['inputs'][role]['sha256']}"
        for role in endpoint_roles
    ]
    assert ac009["geometry"]["displaced_axis_offset_bohr"] == pytest.approx(
        step_bohr, abs=1e-6
    )
    assert len(ac009["agreement"]) == 2
    for orbital, expected in ((0, np.zeros(4)), (1, expected_derivative)):
        agreement = ac009["agreement"][str((orbital, orbital, (0, 0, 0)))]
        np.testing.assert_allclose(agreement["derived"], expected, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            agreement["reference_fd"], expected, atol=1e-8, rtol=0
        )


def test_real_curated_npz_end_to_end_and_reproducible_rerun(tmp_path, real_manifest):
    manifest_path = _write_manifest(tmp_path, real_manifest)
    exports = []
    for name in ("first", "second"):
        output = tmp_path / name
        result = _run_cli(manifest_path, "--output", output)
        assert result.returncode == 0, result.stdout + result.stderr
        exports.append(_assert_export(output, manifest_path, real_manifest))
    first, first_provenance, _ = exports[0]
    second, second_provenance, _ = exports[1]
    assert first.exchange_Jdict == pytest.approx(second.exchange_Jdict, abs=1e-12)
    assert all(np.isfinite(value) for value in first.exchange_Jdict.values())
    assert abs(first.get_J(0, 1, (0, 0, 0))) > 1e-6
    assert first_provenance["raw_response"] == second_provenance["raw_response"]
    assert first_provenance["mapping"] == second_provenance["mapping"]
    assert first_provenance["cli"] == second_provenance["cli"]


def _assert_refused(tmp_path, manifest, expected):
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "refused-output"
    result = _run_cli(manifest_path, "--output", output)
    assert result.returncode != 0
    diagnostic = (result.stdout + result.stderr).lower()
    assert any(word.lower() in diagnostic for word in expected), diagnostic
    assert not output.exists()
    assert not list(tmp_path.rglob("TB2J.pickle"))
    assert not list(tmp_path.rglob("provenance.json"))
    assert not list(tmp_path.rglob("run.json"))
    return diagnostic


def test_missing_operator_refuses_without_output(tmp_path, analytic_manifest):
    del analytic_manifest["operator"]
    _assert_refused(tmp_path, analytic_manifest, ["operator"])


def test_missing_producer_provenance_refuses_without_output(
    tmp_path, analytic_manifest
):
    del analytic_manifest["producer"]
    _assert_refused(tmp_path, analytic_manifest, ["producer"])


def test_missing_declared_producer_artifact_refuses(tmp_path, analytic_manifest):
    analytic_manifest["producer"]["basis_definitions"] = ["missing_basis"]
    _assert_refused(tmp_path, analytic_manifest, ["missing_basis", "basis", "artifact"])


@pytest.mark.parametrize("gate", ["AC-008", "AC-009", "external-consistency"])
def test_failing_consumed_gate_cannot_be_ignored(tmp_path, analytic_manifest, gate):
    report = tmp_path / "gate.json"
    report.write_text(
        json.dumps(
            {
                "gate": gate,
                "status": "fail",
                "passed": False,
                "failure_reasons": ["deliberate consumed-evidence rejection"],
                "declared_ensemble": analytic_manifest["ensemble"],
            }
        )
    )
    analytic_manifest["inputs"]["external_gate"] = _descriptor(report, tmp_path)
    analytic_manifest["gate_reports"] = ["external_gate"]
    _assert_refused(tmp_path, analytic_manifest, [gate])


def test_consumed_gate_with_incompatible_ensemble_refuses(tmp_path, analytic_manifest):
    incompatible = deepcopy(analytic_manifest["ensemble"])
    incompatible["kind"] = "metallic_fermi_dirac_fixed_n"
    report = tmp_path / "gate.json"
    report.write_text(
        json.dumps(
            {
                "gate": "external-consistency",
                "status": "pass",
                "passed": True,
                "declared_ensemble": incompatible,
            }
        )
    )
    analytic_manifest["inputs"]["external_gate"] = _descriptor(report, tmp_path)
    analytic_manifest["gate_reports"] = ["external_gate"]
    _assert_refused(tmp_path, analytic_manifest, ["ensemble"])


def test_bad_input_byte_digest_refuses_without_output(tmp_path, analytic_manifest):
    bundle = tmp_path / analytic_manifest["inputs"]["bundle"]["path"]
    with bundle.open("ab") as stream:
        stream.write(b"\nchanged bytes after manifest digest")
    _assert_refused(tmp_path, analytic_manifest, ["sha256", "digest", "hash"])


def test_localized_displacement_gauge_refuses_missing_dp_endpoints(
    tmp_path, analytic_manifest
):
    analytic_manifest["inputs"]["displacement_plus"] = deepcopy(
        analytic_manifest["inputs"]["bundle"]
    )
    analytic_manifest["inputs"]["displacement_minus"] = deepcopy(
        analytic_manifest["inputs"]["bundle"]
    )
    analytic_manifest["required_gates"].append("AC-009")
    analytic_manifest["displacement"] = {
        "pattern": {"atom_index": 1, "axis": "x", "step_bohr": 0.01},
        "gauge": "localized",
        "reference_fermi_ev": 0.0,
    }
    analytic_manifest["operator"]["mode"] = "onsite_delta"
    _assert_refused(tmp_path, analytic_manifest, ["localized", "dp", "gauge"])


def test_npz_displacement_endpoints_refuse_without_real_producer_interface(
    tmp_path, analytic_manifest
):
    analytic_manifest["operator"]["mode"] = "onsite_delta"
    source = tmp_path / analytic_manifest["inputs"]["bundle"]["path"]
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for role, sign in (("displacement_plus", 1), ("displacement_minus", -1)):
        displaced = dict(arrays)
        positions = arrays["orbital_positions"].copy()
        positions[1, 0] += sign * 0.01
        displaced["orbital_positions"] = positions
        endpoint = tmp_path / f"{role}.npz"
        np.savez(endpoint, **displaced)
        analytic_manifest["inputs"][role] = _descriptor(endpoint, tmp_path)
    analytic_manifest["required_gates"].append("AC-009")
    analytic_manifest["displacement"] = {
        "pattern": {"atom_index": 1, "axis": "x", "step_bohr": 0.01},
        "gauge": "periodic-image-sum",
        "reference_fermi_ev": 0.0,
    }
    diagnostic = _assert_refused(tmp_path, analytic_manifest, ["npz"])
    assert any(word in diagnostic for word in ("hsx", "sisl", "producer interface"))


@pytest.mark.parametrize("sampling", ["interpolated", "uniform", "dense"])
def test_unsupported_sampling_never_silently_changes_mesh(
    tmp_path, analytic_manifest, sampling
):
    analytic_manifest["response"]["sampling"] = sampling
    _assert_refused(tmp_path, analytic_manifest, ["sampling", "recorded"])


def test_plus_u_npz_path_refuses_without_certified_ingest(tmp_path, analytic_manifest):
    analytic_manifest["requires_plusu"] = True
    analytic_manifest["required_gates"].append("AC-008")
    _assert_refused(tmp_path, analytic_manifest, ["ac-008", "+u", "hubbard"])


@pytest.mark.parametrize("missing_gate", MANDATORY_GATES)
def test_required_execution_gate_cannot_be_omitted(
    tmp_path, analytic_manifest, missing_gate
):
    analytic_manifest["required_gates"].remove(missing_gate)
    _assert_refused(tmp_path, analytic_manifest, [missing_gate, "required"])


def test_existing_output_directory_is_preserved(tmp_path, analytic_manifest):
    manifest_path = _write_manifest(tmp_path, analytic_manifest)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "TB2J.pickle"
    sentinel.write_bytes(b"user-owned existing output; do not overwrite")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    result = _run_cli(manifest_path, "--output", output)
    assert result.returncode != 0
    assert "exist" in (result.stdout + result.stderr).lower()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_duplicate_finite_steps_refuse_ac005(tmp_path, analytic_manifest):
    """[0.004, 0.004] is one rotation point, not a two-step ladder."""
    analytic_manifest["response"]["finite_steps"] = [0.004, 0.004]
    diagnostic = _assert_refused(
        tmp_path, analytic_manifest, ["ac-005", "distinct", "duplicate"]
    )
    assert "0.004" in diagnostic


def test_near_duplicate_finite_steps_refuse_ac005(tmp_path, analytic_manifest):
    """Near-duplicates within 1e-12 rad cannot fake a distinct second step."""
    analytic_manifest["response"]["finite_steps"] = [0.004, 0.0040000000001]
    diagnostic = _assert_refused(
        tmp_path, analytic_manifest, ["ac-005", "distinct", "duplicate"]
    )
    assert "0.004" in diagnostic


def _as_metallic(manifest, smearing_width_ev):
    """Turn the analytic pencil manifest into a metallic fixed-N one."""
    manifest["ensemble"]["kind"] = "metallic_fermi_dirac_fixed_n"
    manifest["ensemble"]["smearing_width_ev"] = smearing_width_ev
    manifest["response"].pop("contour", None)
    manifest["response"].pop("contour_node_rungs", None)
    return manifest


PRODUCER_WIDTH_EV = 8.617333262e-5 * 100.0


def test_metallic_matching_producer_temperature_exports(tmp_path, analytic_manifest):
    """A metallic width equal to the recorded producer temperature passes.

    The analytic pencil is rescaled so the metallic fixed-N free energy is
    actually resolvable at that width (bands near mu, count still 2 within
    the AC-003 budget), proving the width gate accepts the aligned case
    rather than merely failing late elsewhere.
    """
    with np.load(tmp_path / "analytic.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    # Gapped pencil eigenvalues are [-1.2, -0.8, 0.8, 1.2] (gap 1.6 eV).
    # Rescale so the root gap is 24 smearing widths: occupations still
    # round to 2 within 1e-4, yet f*(1-f) is far above the derivative floor.
    scale = (24 * PRODUCER_WIDTH_EV) / 1.6
    arrays["h"] = scale * arrays["h"]
    arrays["eig"] = scale * arrays["eig"]
    bundle = tmp_path / "analytic_metallic.npz"
    np.savez(bundle, **arrays)
    analytic_manifest["inputs"]["bundle"] = _descriptor(bundle, tmp_path)
    manifest = _as_metallic(analytic_manifest, PRODUCER_WIDTH_EV)
    manifest["ensemble"]["gap_ev"] = 24 * PRODUCER_WIDTH_EV
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "out"
    result = _run_cli(manifest_path, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr


def test_metallic_smearing_width_mismatch_refuses_before_response(
    tmp_path, analytic_manifest
):
    """The AC-003 count oracle is width-independent (it uses the producer
    temperature), so the count stays valid at both widths -- only the
    width mismatch must refuse, before any response/export."""
    assert PRODUCER_WIDTH_EV != 2 * PRODUCER_WIDTH_EV
    manifest = _as_metallic(analytic_manifest, 2 * PRODUCER_WIDTH_EV)
    _assert_refused(
        tmp_path, manifest, ["metallic smearing consistency", "width", "temperature"]
    )
