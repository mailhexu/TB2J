"""Actual public WFSX runs, measured calls, and failure-path ownership."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sisl")

from test_siesta_wfsx_exchange import exchange_params  # noqa: E402
from test_siesta_wfsx_green import source_model  # noqa: E402


@pytest.mark.parametrize("metadata_kind", ["mapping", "json"])
def test_python_entry_writes_measured_run_without_ordinary_inputs(
    tmp_path, monkeypatch, metadata_kind
):
    import scipy.linalg
    import sisl
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    from TB2J.interfaces import gen_exchange_siesta
    from TB2J.io_exchange import SpinIO

    inputs = tmp_path / "input"
    inputs.mkdir()
    model, metadata, _, _ = source_model(inputs, "collinear_nspin2_reduced")
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = cache / "user.txt"
    marker.write_text("keep")

    def forbidden(*args, **kwargs):
        pytest.fail("WFSX public entry reached ordinary parsing/eigensolve")

    monkeypatch.setattr(SislParser, "__init__", forbidden)
    for module in (np.linalg, scipy.linalg):
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            monkeypatch.setattr(module, name, forbidden)
    original_sile = sisl.get_sile

    def wfsx_only(path, *args, **kwargs):
        assert Path(path) == model.eigenstate_source.wfsx_path
        return original_sile(path, *args, **kwargs)

    monkeypatch.setattr(sisl, "get_sile", wfsx_only)
    output = tmp_path / "results"
    gen_exchange_siesta(
        None,
        input_mode="wfsx",
        wfsx_file=model.eigenstate_source.wfsx_path,
        wfsx_metadata=metadata
        if metadata_kind == "mapping"
        else inputs / "metadata.json",
        cache_path=cache,
        **exchange_params(output),
    )
    summary = json.loads((output / "wfsx_run.json").read_text())
    assert summary["electronic_eigensolver"]["calls"] == 0
    assert summary["electronic_eigensolver"]["seconds"] == 0
    assert "NumPy" in summary["electronic_eigensolver"]["method"]
    assert summary["peak_rss"]["unit"] == "bytes"
    assert summary["peak_rss"]["value"] > 0
    assert "RUSAGE_SELF" in summary["peak_rss"]["method"]
    assert summary["kpoints"]["supplied"] == 3
    assert summary["kpoints"]["requested"] == 5
    assert summary["kpoints"]["representatives_per_channel"] == [3, 3]
    assert summary["states"]["supplied"] == 108
    assert summary["states"]["requested"] == 180
    assert summary["cache"]["spectral_disk_bytes"] == 63072
    assert summary["source_precision"] == [["complex64", "float64"]]
    assert summary["energy_reference"]["shifted_fermi_ev"] == 0
    assert summary["metadata"]["source_provenance"] == metadata["source_provenance"]
    for field in ("import", "preparation", "integration", "output", "total"):
        assert summary["timing_seconds"][field] > 0
    assert summary["timing_seconds"]["total"] >= sum(
        summary["timing_seconds"][key]
        for key in ("import", "preparation", "integration", "output")
    )
    assert summary["inputs"][0]["sha256"] == before["states.WFSX"]
    result = SpinIO.load_pickle(path=output)
    tensors = result.get_full_Jtensor_for_Rlist(order="ij33")
    assert np.isfinite(tensors).all()
    assert any(np.any(np.asarray(r)) for r in result.Rlist)
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()
    } == before
    assert list(cache.iterdir()) == [marker]
    assert marker.read_text() == "keep"


def test_eigensolver_monitor_counts_actual_solver_and_restores_after_failure():
    import scipy.linalg
    from HamiltonIO import lcao_hamiltonian
    from HamiltonIO.siesta import SislParser
    from test_siesta_wfsx_green import fixture_case

    from TB2J.interfaces.siesta_interface import _measure_wfsx_eigensolvers

    case, manifest, _ = fixture_case("collinear_nspin2_full")
    model = SislParser(str(case / manifest["files"]["fdf"])).get_model()[0]
    original = scipy.linalg.eigh
    alias = lcao_hamiltonian.eigh
    with pytest.raises(RuntimeError):
        with _measure_wfsx_eigensolvers() as stats:
            h, overlap, values, vectors = model.HSE_k([0, 0, 0])
            np.testing.assert_allclose(
                h @ vectors, (overlap @ vectors) * values, atol=1e-10
            )
            assert stats["calls"] == 1
            assert stats["seconds"] > 0
            raise RuntimeError("controlled consumer failure")
    assert scipy.linalg.eigh is original
    assert lcao_hamiltonian.eigh is alias


@pytest.mark.parametrize(
    "options",
    [
        {"read_H_soc": True},
        {"orth": True},
        {"use_gpu": True},
        {"ibz": True},
        {"nproc": 2},
    ],
)
def test_public_entry_rejects_operator_modes_before_parsers(
    tmp_path, monkeypatch, options
):
    import scipy.linalg
    from HamiltonIO.siesta.sisl_wrapper import SislParser
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible public mode reached parser/factors")

    monkeypatch.setattr(SislParser, "__init__", forbidden)
    monkeypatch.setattr(scipy.linalg, "lu_factor", forbidden)
    with pytest.raises(UnsupportedOperatorMode):
        gen_exchange_siesta(
            None,
            input_mode="wfsx",
            wfsx_file=model.eigenstate_source.wfsx_path,
            wfsx_metadata=metadata,
            **{**exchange_params(tmp_path / "out"), **options},
        )


def test_public_failed_import_removes_owned_factors_and_preserves_inputs(tmp_path):
    from HamiltonIO.siesta.wfsx import IncompleteSpectrum

    from TB2J.interfaces import gen_exchange_siesta

    inputs = tmp_path / "input"
    inputs.mkdir()
    model, metadata, _, _ = source_model(inputs, "collinear_nspin2_full")
    path = model.eigenstate_source.wfsx_path
    path.write_bytes(path.read_bytes()[:-20])
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()
    }
    parent = tmp_path / "cache"
    parent.mkdir()
    marker = parent / "user.txt"
    marker.write_text("keep")
    with pytest.raises(IncompleteSpectrum):
        gen_exchange_siesta(
            input_mode="wfsx",
            wfsx_file=path,
            wfsx_metadata=metadata,
            cache_path=parent,
            **exchange_params(tmp_path / "output"),
        )
    assert list(parent.iterdir()) == [marker]
    assert marker.read_text() == "keep"
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()
    } == before


@pytest.mark.parametrize("invalid", ["malformed", "missing", "field"])
def test_public_metadata_failures_keep_named_categories(tmp_path, invalid):
    from HamiltonIO.siesta.wfsx import InconsistentMetadata, MissingMetadata

    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")
    bad = tmp_path / "bad.json"
    category = MissingMetadata
    if invalid == "malformed":
        bad.write_text("{")
        category = InconsistentMetadata
    elif invalid == "field":
        metadata.pop("fermi_source")
        bad.write_text(json.dumps(metadata))
    with pytest.raises(category):
        gen_exchange_siesta(
            input_mode="wfsx",
            wfsx_file=model.eigenstate_source.wfsx_path,
            wfsx_metadata=bad,
            **exchange_params(tmp_path / "output"),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"read_H_soc": True},
        {"skip": True},
        {"vectorize_energy": True},
        {"e_batch_size": 4},
    ],
)
def test_operator_refusal_precedes_metadata_or_wavefunction_access(tmp_path, options):
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.interfaces import gen_exchange_siesta

    with pytest.raises(UnsupportedOperatorMode):
        gen_exchange_siesta(
            input_mode="wfsx",
            wfsx_file=tmp_path / "absent.WFSX",
            wfsx_metadata=tmp_path / "absent.json",
            **options,
        )


@pytest.mark.parametrize(
    "option",
    [
        ["--split_soc"],
        ["--orth"],
        ["--use_gpu"],
        ["--ibz"],
        ["--mae_angles", "0", "0", "0"],
        ["--vectorize_energy"],
        ["--e_batch_size", "4"],
        ["--np", "2"],
    ],
)
def test_cli_operator_options_propagate_stable_category_before_inputs(
    monkeypatch, option
):
    import sys

    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.scripts.siesta2J import run_siesta2J

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "siesta2J.py",
            "--input-mode",
            "wfsx",
            "--elements",
            "O",
            "--wfsx-file",
            "absent.WFSX",
            "--wfsx-metadata",
            "absent.json",
            *option,
        ],
    )
    with pytest.raises(UnsupportedOperatorMode):
        run_siesta2J()


def test_missing_wavefunction_uses_reader_completeness_category(tmp_path):
    from HamiltonIO.siesta.wfsx import IncompleteSpectrum

    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")
    model.eigenstate_source.wfsx_path.unlink()
    with pytest.raises(IncompleteSpectrum):
        gen_exchange_siesta(
            input_mode="wfsx",
            wfsx_file=model.eigenstate_source.wfsx_path,
            wfsx_metadata=metadata,
            **exchange_params(tmp_path / "output"),
        )


def test_published_solver_figures_measure_an_actual_injected_solver_call(
    tmp_path, monkeypatch
):
    import scipy.linalg

    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_reduced")
    original_factor = scipy.linalg.lu_factor
    injected = False

    def factor_with_measurement_probe(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            np.testing.assert_array_equal(
                scipy.linalg.eigh(np.diag([1.0, 3.0]), eigvals_only=True), [1, 3]
            )
        return original_factor(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", factor_with_measurement_probe)
    output = tmp_path / "output"
    summary = gen_exchange_siesta(
        input_mode="wfsx",
        wfsx_file=model.eigenstate_source.wfsx_path,
        wfsx_metadata=metadata,
        **exchange_params(output),
    )
    published = json.loads((output / "wfsx_run.json").read_text())
    assert summary["electronic_eigensolver"]["calls"] == 1
    assert published["electronic_eigensolver"]["calls"] == 1
    assert published["electronic_eigensolver"]["seconds"] > 0


def test_ordinary_import_without_unix_resource_and_explicit_wfsx_refusal():
    import subprocess
    import sys

    code = """import sys
sys.modules['resource'] = None
from TB2J.interfaces import gen_exchange_siesta
from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode
try:
    gen_exchange_siesta(input_mode='wfsx', wfsx_file='absent.WFSX', wfsx_metadata='absent.json')
except UnsupportedOperatorMode:
    print('ordinary import available; measurement prerequisite refused')
else:
    raise AssertionError('unavailable RSS measurement was silently accepted')
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_numpy_mesh_and_width_publish_json_native_summary(tmp_path):
    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_reduced")
    params = exchange_params(tmp_path / "output")
    params.update(
        kmesh=np.array([1, 1, 5], dtype=np.int64),
        smearing=np.float32(params["smearing"]),
    )
    summary = gen_exchange_siesta(
        input_mode="wfsx",
        wfsx_file=model.eigenstate_source.wfsx_path,
        wfsx_metadata=metadata,
        **params,
    )
    published = json.loads((tmp_path / "output" / "wfsx_run.json").read_text())
    assert (
        summary["kpoints"]["rlist_mesh"]
        == published["kpoints"]["rlist_mesh"]
        == [1, 1, 5]
    )
    assert (
        summary["integration_width_ev"]
        == published["integration_width_ev"]
        == float(params["smearing"])
    )


@pytest.mark.parametrize(
    "fixture, filenames, basis_size",
    [
        ("collinear_nspin2_reduced", ["rho_up.pickle", "rho_dn.pickle"], 18),
        ("noncollinear_nspin4_full", ["rhoR.pickle"], 36),
    ],
)
def test_cli_density_export_uses_full_rlist_and_preserves_density_symmetry(
    tmp_path, monkeypatch, fixture, filenames, basis_size
):
    import pickle
    import sys

    from TB2J.scripts.siesta2J import run_siesta2J

    model, _, _, _ = source_model(tmp_path, fixture)
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "siesta2J.py",
            "--input-mode",
            "wfsx",
            "--wfsx-file",
            str(model.eigenstate_source.wfsx_path),
            "--wfsx-metadata",
            str(tmp_path / "metadata.json"),
            "--elements",
            "O",
            "--kmesh",
            "1",
            "1",
            "5",
            "--nz",
            "8",
            "--rcut",
            "4.3",
            "--output_path",
            str(output),
            "--write_dm",
        ],
    )
    run_siesta2J()
    for filename in filenames:
        with (output / filename).open("rb") as stream:
            result = pickle.load(stream)
        points = {tuple(r): index for index, r in enumerate(result["Rlist"])}
        assert set(points) == {(0, 0, index) for index in range(-2, 3)}
        density = result["rhoR"]
        assert density.shape == (5, basis_size, basis_size)
        assert np.isfinite(density).all()
        for r, index in points.items():
            np.testing.assert_allclose(
                density[index],
                density[points[tuple(-x for x in r)]].conj().T,
                atol=1e-10,
                rtol=1e-10,
            )


@pytest.mark.parametrize(
    "fixture, reduce, provenance, validation",
    [
        ("collinear_nspin2_full", False, "not-used", False),
        (
            "collinear_nspin2_full",
            True,
            "exported-partners-require-operator-validation",
            True,
        ),
        ("collinear_nspin2_reduced", False, "declaration-only", False),
    ],
)
def test_summary_distinguishes_declared_from_exported_partner_evidence(
    tmp_path, fixture, reduce, provenance, validation
):
    from TB2J.interfaces import gen_exchange_siesta

    model, metadata, _, _ = source_model(tmp_path, fixture)
    summary = gen_exchange_siesta(
        input_mode="wfsx",
        wfsx_file=model.eigenstate_source.wfsx_path,
        wfsx_metadata=metadata,
        reduce_scalar=reduce,
        **exchange_params(tmp_path / "output"),
    )
    assert summary["metadata"]["scalar_symmetry"] == metadata["scalar_symmetry"]
    assert [item["provenance"] for item in summary["symmetry"]] == [
        provenance,
        provenance,
    ]
    assert [item["operator_validation_required"] for item in summary["symmetry"]] == [
        validation,
        validation,
    ]
