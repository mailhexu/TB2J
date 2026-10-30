"""Behavior of the private full-grid study entry point."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "srmno3_validation_runner",
    Path(__file__).with_name("run_srmno3_qspace_validation.py"),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_incomplete_settings_are_refused_without_vault(tmp_path):
    with pytest.raises(runner.PreflightRefusal, match="missing required settings"):
        runner.run_case({"run_id": "incomplete"}, tmp_path)


@pytest.fixture
def settings(tmp_path):
    authorities = {}
    for name in runner.REQUIRED_MATH_AUTHORITIES:
        path = (
            Path("/home/hexu/projects/memnotes/memnotes/Projects/TB2J/mathdocs")
            / f"{name}.py"
        )
        if not path.is_file():
            pytest.skip("TB2J Project MathDocs are not available on this machine")
        authorities[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified_at": "test",
        }
    return {
        "run_id": "chain",
        "model": {
            "kind": "chain",
            "npoints": 5,
            "onsite_up_eV": -2.0,
            "onsite_dn_eV": 2.0,
            "hopping_eV": -0.5,
            "basis": ["Fe1|d"],
            "magnetic_site": 0,
        },
        "physical": {
            "efermi_eV": 0.0,
            "response_temperature_K": 0.0,
            "density_smearing_eV": 0.01,
            "emin_eV": -15.0,
            "emax_eV": 0.0,
        },
        "grid": {"kmesh": [5, 1, 1], "qmesh": [5, 1, 1], "nz_refinements": [129]},
        "integration": {"method": "semicircle", "nproc": 1},
        "conventions": dict(runner.REQUIRED_CONVENTIONS),
        "acceptance": {
            "atol_eV": 1e-8,
            "rtol": 0.0,
            "near_zero_floor_eV": 1e-8,
            "real_gauge_tol_eV": 1e-8,
        },
        "math_authorities": authorities,
        "source_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(runner._TB2J_FILE).resolve().parent.parent),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
    }


def test_chain_output_matches_spectral_reference(settings, tmp_path):
    summary = runner.run_case(settings, tmp_path / "output")
    out = tmp_path / "output" / "chain"
    with np.load(out / "raw.npz") as raw:
        k = np.arange(-2, 3) / 5
        up = -2 - np.cos(2 * np.pi * k)
        expected = [
            -4 * np.mean(1 / (up - (2 - np.cos(2 * np.pi * (k + q))))) for q in k
        ]
        np.testing.assert_allclose(
            raw["Jq_direct"][-1, :, 0, 0], expected, atol=2e-7, rtol=0
        )
        residual = raw["residual"][-1]
        assert np.iscomplexobj(residual)
        assert summary["max_abs_residual_eV"] == pytest.approx(np.max(np.abs(residual)))
        assert np.any(np.all(raw["R"] == 0, axis=1))
    assert summary["status"] == "pass"
    assert (
        json.loads((out / "manifest.json").read_text())["status"] == summary["status"]
    )
    env = json.loads((out / "manifest.json").read_text())["environment"]
    parser = Path(env["hamiltonio_package_root"]) / "wannier" / "w90_parser.py"
    assert (
        env["hamiltonio_python_files"]["wannier/w90_parser.py"]
        == hashlib.sha256(parser.read_bytes()).hexdigest()
    )
    assert json.loads((out / "summary.json").read_text())["worst"] == summary["worst"]
    with pytest.raises(runner.PreflightRefusal, match="evidence already exists"):
        runner.run_case(settings, tmp_path / "output")


def test_mismatched_grid_and_math_hash_are_refused(settings, tmp_path):
    settings["grid"]["qmesh"] = [3, 1, 1]
    with pytest.raises(runner.PreflightRefusal, match="qmesh"):
        runner.run_case(settings, tmp_path / "bad-grid")
    settings["grid"]["qmesh"] = [5, 1, 1]
    settings["math_authorities"][runner.REQUIRED_MATH_AUTHORITIES[0]]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(runner.PreflightRefusal, match="hash mismatch"):
        runner.run_case(settings, tmp_path / "bad-hash")
    assert not (tmp_path / "bad-hash" / "chain" / "raw.npz").exists()
    settings["physical"]["response_temperature_K"] = 300.0
    with pytest.raises(runner.PreflightRefusal, match="must be zero"):
        runner.run_case(settings, tmp_path / "bad-temperature")


def test_mismatched_site_mapping_is_refused(settings, tmp_path):
    settings["model"]["magnetic_site"] = 1
    with pytest.raises(runner.PreflightRefusal, match="explicit mapping"):
        runner.run_case(settings, tmp_path / "bad-mapping")


def test_wannier_loader_refuses_unpinned_or_reordered_inputs(tmp_path):
    source = Path(
        "/home/hexu/projects/TB2J_dev/TB2J/tests/data/data-repo/inputs/2_SrMnO3_wannier/data"
    )
    if not (source / "abinit.in").is_file():
        pytest.skip("SrMnO3 Wannier fixture is unavailable")
    for name in runner.CORE_WANNIER_FILES:
        shutil.copyfile(source / name, tmp_path / name)
    model = {
        "kind": "wannier",
        "input_root": str(tmp_path),
        "posfile": "abinit.in",
        "prefix_up": "abinito_w90_up",
        "prefix_dn": "abinito_w90_down",
        "magnetic_elements": ["Mn"],
        "input_hashes": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in runner.CORE_WANNIER_FILES
        },
    }
    settings = {"model": model, "run_id": "input-check"}
    (tmp_path / "abinito_w90_up_tb.dat").write_text("unverified competing input")
    with pytest.raises(runner.PreflightRefusal, match="not hashed"):
        runner._check_inputs(settings)
    (tmp_path / "abinito_w90_up_tb.dat").unlink()
    path = tmp_path / "abinito_w90_down_centres.xyz"
    lines = path.read_text().splitlines()
    lines[2], lines[7] = lines[7], lines[2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(runner.PreflightRefusal, match="incompatible ordered orbital"):
        runner._load_models_wannier(model, "mapping-check")
