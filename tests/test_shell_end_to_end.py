"""Consumer end-to-end parity for shell authoring and native preset reloads."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO


def _run(*arguments):
    from TB2J.scripts.TB2J_shell import main

    try:
        return main([str(argument) for argument in arguments])
    except SystemExit as error:
        return error.code


def _json_payload(value):
    return json.loads(json.dumps(asdict(value), default=str))


def _toml_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(float(value))


def _case(tmp_path, scenario):
    if scenario == "sg":
        atoms = Atoms(
            "Fe", cell=2.0 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True
        )
        magnetic_atoms = [1]
        moments = [2.0]
        spinat = np.array([[0.0, 0.0, 2.0]])
        charges = np.array([8.0])
        colinear = True
        cutoff = 2.01
        pair = ((1, 0, 0), 0, 0)
        symmetry = "sg"
    elif scenario == "afm-scalar":
        atoms = Atoms(
            "Fe2",
            cell=np.diag([4.0, 4.0, 2.0]),
            scaled_positions=[(0, 0, 0), (0, 0, 0.5)],
            pbc=True,
        )
        magnetic_atoms = [1, 2]
        moments = [2.0, -2.0]
        spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
        charges = np.array([8.0, 8.0])
        colinear = True
        cutoff = 1.01
        pair = ((0, 0, 0), 0, 1)
        symmetry = "msg"
    elif scenario == "axial":
        atoms = Atoms(
            ["Fe", "O", "Co"],
            cell=4.0 * np.eye(3),
            scaled_positions=[(0, 0, 0), (0.2, 0.2, 0.2), (0.35, 0.11, 0.07)],
            pbc=True,
        )
        magnetic_atoms = [1, 3]
        moments = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        spinat = np.asarray(moments)
        charges = np.array([8.0, 6.0, 9.0])
        colinear = False
        cutoff = 1.5
        pair = ((0, 0, 0), 0, 1)
        symmetry = "msg"
    else:
        raise AssertionError(f"unknown test scenario: {scenario}")

    structure = tmp_path / "structure.vasp"
    atoms.write(structure, vasp5=True)
    moments_path = None
    if moments is not None:
        moments_path = tmp_path / "moments.toml"
        moments_path.write_text(f"moments = {_toml_value(moments)}\n")

    index_spin = [-1] * len(atoms)
    for spin_index, atom_number in enumerate(magnetic_atoms):
        index_spin[atom_number - 1] = spin_index
    base = SpinIO(
        atoms,
        spinat=spinat,
        charges=charges,
        index_spin=index_spin,
        colinear=colinear,
        exchange_Jdict={pair: 0.00125},
    )
    base.damping = None
    base.gyro_ratio = None
    base_directory = tmp_path / "base"
    shells.write_exchange(base, base_directory, "pickle")
    return {
        "atoms": atoms,
        "base_path": base_directory / "TB2J.pickle",
        "charges": charges,
        "cutoff": cutoff,
        "index_spin": index_spin,
        "magnetic_atoms": magnetic_atoms,
        "moments": moments,
        "moments_path": moments_path,
        "scenario": scenario,
        "spinat": spinat,
        "structure": structure,
        "symmetry": symmetry,
    }


def _template_args(case, mode, output):
    arguments = [
        "template",
        case["structure"],
        "--magnetic-atoms",
        *(str(number) for number in case["magnetic_atoms"]),
        "--cutoff",
        str(case["cutoff"]),
        "--symmetry",
        case["symmetry"],
        "--mode",
        mode,
        "-o",
        output,
    ]
    if case["moments_path"] is not None:
        arguments.extend(("--reference-spins", case["moments_path"]))
    return arguments


def _assert_primed_half_translation(document):
    assert any(
        np.array_equal(operation.rotation, np.eye(3, dtype=int))
        and np.allclose(operation.translation, (0.0, 0.0, 0.5), atol=1e-12)
        and operation.time_reversal
        for operation in document.operations
    )


def _assert_native_channels(actual, expected, atol):
    assert list(actual.index_spin) == list(expected.index_spin)
    for name in ("exchange_Jdict", "dmi_ddict", "Jani_dict"):
        actual_values = getattr(actual, name) or {}
        expected_values = getattr(expected, name) or {}
        assert set(actual_values) == set(expected_values), name
        for key in expected_values:
            np.testing.assert_allclose(
                actual_values[key], expected_values[key], atol=atol, rtol=0
            )
    np.testing.assert_allclose(actual.spinat, expected.spinat, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.charges, expected.charges, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("scenario", "mode"),
    [
        pytest.param("sg", "isotropic", id="sg-isotropic"),
        pytest.param("sg", "tensor", id="sg-tensor"),
        pytest.param("afm-scalar", "isotropic", id="msg-scalar-primed-isotropic"),
        pytest.param("afm-scalar", "tensor", id="msg-scalar-primed-tensor"),
        pytest.param("axial", "tensor", id="msg-noncollinear-axial-tensor"),
    ],
)
def test_cli_api_symmetry_value_grid_reaches_native_preset(
    tmp_path, capsys, scenario, mode
):
    case = _case(tmp_path, scenario)
    cli_shells = tmp_path / "cli.shells.toml"
    api_shells = tmp_path / "api.shells.toml"

    assert _run(*_template_args(case, mode, cli_shells)) == 0
    api_document = shells.template(
        case["atoms"],
        case["magnetic_atoms"],
        case["cutoff"],
        symmetry=case["symmetry"],
        reference_spins=case["moments"],
        mode=mode,
    )
    shells.write_shells(api_document, api_shells)
    assert cli_shells.read_bytes() == api_shells.read_bytes()
    assert shells.read_shells(cli_shells) == api_document

    if scenario == "afm-scalar":
        _assert_primed_half_translation(api_document)
        assert api_document.moments == (2.0, -2.0)
    elif scenario == "axial":
        moments = np.asarray(api_document.moments, dtype=float)
        assert moments.shape == (3, 3)
        assert np.linalg.norm(np.cross(moments[0], moments[2])) > 0
        assert np.array_equal(moments[1], (0.0, 0.0, 0.0))

    assert _run("validate", cli_shells) == 0
    shells.validate_shells(api_document)
    assert _run("validate", cli_shells, "--for-export") != 0
    validation_error = capsys.readouterr().err.lower()
    assert "shell" in validation_error and "unfilled" in validation_error

    blocked = tmp_path / "blocked-native"
    assert (
        _run(
            "convert",
            cli_shells,
            "--from",
            "shells",
            "--to",
            "tb2j",
            "-o",
            blocked,
            "--base",
            case["base_path"],
            "--allow-loss",
            "charges",
        )
        != 0
    )
    assert not blocked.exists()

    wire = cli_shells.read_text()
    replacement = (
        "1.25" if mode == "isotropic" else "[[1.25, 0, 0], [0, 1.25, 0], [0, 0, 1.25]]"
    )
    assert wire.count('J = "unfilled"') == len(api_document.shells)
    cli_shells.write_text(wire.replace('J = "unfilled"', f"J = {replacement}"))
    document = shells.read_shells(cli_shells)
    assert _run("validate", cli_shells, "--for-export") == 0
    shells.validate_shells(document, for_export=True)

    base_read = shells.read_exchange(case["base_path"], "pickle")
    expanded = shells.expand(document, base=base_read.spinio)
    api_output = tmp_path / "api-native"
    api_write_report = shells.write_exchange(
        expanded.spinio, api_output, "tb2j", allow_loss=("charges",)
    )

    cli_output = tmp_path / "cli-native"
    report_path = tmp_path / "conversion.json"
    assert (
        _run(
            "convert",
            cli_shells,
            "--from",
            "shells",
            "--to",
            "tb2j",
            "-o",
            cli_output,
            "--base",
            case["base_path"],
            "--allow-loss",
            "charges",
            "--report",
            report_path,
        )
        == 0
    )
    cli_reports = json.loads(report_path.read_text())
    expected_reports = {
        "validate": _json_payload(shells.validate_shells(document, for_export=True)),
        "base": _json_payload(base_read.report),
        "expand": _json_payload(expanded.report),
        "write": _json_payload(api_write_report),
    }
    assert cli_reports == expected_reports
    assert any(
        loss[0:2] == ["charges", "xml"] for loss in cli_reports["write"]["losses"]
    )

    expected_files = {"TB2J.pickle", "exchange.out", "exchange.xml", "structure.vasp"}
    assert {path.name for path in api_output.iterdir()} == expected_files
    assert {path.name for path in cli_output.iterdir()} == expected_files

    reference_spins = (
        base_read.spinio.spinat[:, 2]
        if base_read.spinio.colinear
        else base_read.spinio.spinat
    )
    context = shells.NativeContext(
        atoms=case["atoms"],
        index_spin=case["index_spin"],
        reference_spins=reference_spins,
        charges=case["charges"],
    )
    tolerances = {"pickle": 0.0, "text": 5e-8, "xml": 5e-9}
    filenames = {
        "pickle": "TB2J.pickle",
        "text": "exchange.out",
        "xml": "exchange.xml",
    }
    for format_name, filename in filenames.items():
        api_read = shells.read_exchange(
            api_output / filename, format_name, context=context
        )
        cli_read = shells.read_exchange(
            cli_output / filename, format_name, context=context
        )
        assert _json_payload(api_read.report) == _json_payload(cli_read.report)
        _assert_native_channels(
            api_read.spinio, expanded.spinio, tolerances[format_name]
        )
        _assert_native_channels(
            cli_read.spinio, expanded.spinio, tolerances[format_name]
        )
        if format_name == "text":
            assert api_read.report.precision_profile == "exchange.out-default"
        elif format_name == "xml":
            assert api_read.report.precision_profile == "xml-%.5e"


def _without_comment_lines(source, destination):
    destination.write_text(
        "\n".join(
            line
            for line in source.read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        + "\n"
    )


def test_cli_comment_removal_preserves_generated_and_converted_shell_semantics(
    tmp_path,
):
    case = _case(tmp_path, "sg")
    generated = tmp_path / "generated.toml"
    generated_without_comments = tmp_path / "generated-no-comments.toml"
    assert _run(*_template_args(case, "isotropic", generated)) == 0
    _without_comment_lines(generated, generated_without_comments)
    assert shells.read_shells(generated_without_comments) == shells.read_shells(
        generated
    )

    converted = tmp_path / "converted.toml"
    converted_without_comments = tmp_path / "converted-no-comments.toml"
    assert (
        _run(
            "convert",
            case["base_path"],
            "--from",
            "pickle",
            "--to",
            "shells",
            "-o",
            converted,
        )
        == 0
    )
    _without_comment_lines(converted, converted_without_comments)
    assert shells.read_shells(converted_without_comments) == shells.read_shells(
        converted
    )


def test_cli_sparse_masks_and_explicit_orbit_completion_match_public_api(tmp_path):
    atoms = Atoms("Fe", cell=2.0 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict={((1, 0, 0), 0, 0): 0.1},
        Jani_dict={((0, 1, 0), 0, 0): np.diag([1.0, -2.0, 1.0])},
    )
    source.damping = None
    source.gyro_ratio = None
    base_directory = tmp_path / "sparse-source"
    shells.write_exchange(source, base_directory, "pickle")
    base_path = base_directory / "TB2J.pickle"
    loaded = shells.read_exchange(base_path, "pickle")
    compressed = shells.compress(loaded.spinio, mode="tensor")
    api_shells = tmp_path / "api-sparse.toml"
    shells.write_shells(compressed.document, api_shells)

    cli_shells = tmp_path / "cli-sparse.toml"
    compress_report_path = tmp_path / "compress.json"
    assert (
        _run(
            "convert",
            base_path,
            "--from",
            "pickle",
            "--to",
            "shells",
            "--mode",
            "tensor",
            "-o",
            cli_shells,
            "--report",
            compress_report_path,
        )
        == 0
    )
    assert cli_shells.read_bytes() == api_shells.read_bytes()
    assert json.loads(compress_report_path.read_text()) == {
        "read": _json_payload(loaded.report),
        "compress": _json_payload(compressed.report),
    }
    document = shells.read_shells(cli_shells)
    entry = document.shells[0]
    assert entry.support_members == (4, 5)
    assert entry.Jiso_members == (5,)
    assert entry.DMI_members == ()
    assert entry.Jani_members == (4,)

    partial = shells.expand(document)
    api_partial_output = tmp_path / "api-partial"
    api_partial_write = shells.write_exchange(
        partial.spinio, api_partial_output, "pickle"
    )
    cli_partial_output = tmp_path / "cli-partial"
    partial_report_path = tmp_path / "partial.json"
    assert (
        _run(
            "convert",
            cli_shells,
            "--from",
            "shells",
            "--to",
            "pickle",
            "-o",
            cli_partial_output,
            "--report",
            partial_report_path,
        )
        == 0
    )
    partial_reports = json.loads(partial_report_path.read_text())
    assert partial_reports == {
        "validate": _json_payload(shells.validate_shells(document, for_export=True)),
        "expand": _json_payload(partial.report),
        "write": _json_payload(api_partial_write),
    }
    partial_loaded = shells.read_exchange(
        cli_partial_output / "TB2J.pickle", "pickle"
    ).spinio
    assert set(partial_loaded.exchange_Jdict) == {((1, 0, 0), 0, 0)}
    assert set(partial_loaded.Jani_dict) == {((0, 1, 0), 0, 0)}
    assert not partial_loaded.dmi_ddict

    completed = shells.expand(document, complete_orbits=True)
    api_complete_output = tmp_path / "api-complete"
    api_complete_write = shells.write_exchange(
        completed.spinio, api_complete_output, "pickle"
    )
    cli_complete_output = tmp_path / "cli-complete"
    complete_report_path = tmp_path / "complete.json"
    assert (
        _run(
            "convert",
            cli_shells,
            "--from",
            "shells",
            "--to",
            "pickle",
            "-o",
            cli_complete_output,
            "--complete-orbits",
            "--report",
            complete_report_path,
        )
        == 0
    )
    complete_reports = json.loads(complete_report_path.read_text())
    assert complete_reports == {
        "validate": _json_payload(shells.validate_shells(document, for_export=True)),
        "expand": _json_payload(completed.report),
        "write": _json_payload(api_complete_write),
    }
    original_keys = set(source.exchange_Jdict) | set(source.Jani_dict)
    full_keys = set(completed.spinio.exchange_Jdict) | set(completed.spinio.Jani_dict)
    assert set(completed.report.created_pair_keys) == full_keys - original_keys
    assert set(completed.report.created_channel_keys) == {
        (channel, key)
        for channel, field in (("Jiso", "exchange_Jdict"), ("Jani", "Jani_dict"))
        for key in set(getattr(completed.spinio, field)) - set(getattr(source, field))
    }
    assert set(completed.report.absent_channel_keys) == {
        ("DMI", key) for key in full_keys
    }

    complete_loaded = shells.read_exchange(
        cli_complete_output / "TB2J.pickle", "pickle"
    ).spinio
    for name in ("exchange_Jdict", "Jani_dict"):
        assert set(getattr(complete_loaded, name)) == set(
            getattr(completed.spinio, name)
        )
    assert len(complete_loaded.exchange_Jdict) == 6
    assert len(complete_loaded.Jani_dict) == 6
    assert not complete_loaded.dmi_ddict
