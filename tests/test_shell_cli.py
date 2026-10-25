"""Consumer-visible shell CLI behavior; scientific algorithms stay in the API."""

import json
from dataclasses import asdict, replace

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


@pytest.fixture
def inputs(tmp_path):
    atoms = Atoms("Fe", cell=3 * np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
    structure = tmp_path / "structure.vasp"
    atoms.write(structure, vasp5=True)
    moments = tmp_path / "moments.toml"
    moments.write_text("moments = [2.0]\n")
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2]],
        charges=[8.0],
        index_spin=[0],
        colinear=True,
        exchange_Jdict={((1, 0, 0), 0, 0): 0.00125},
    )
    source.damping = source.gyro_ratio = None
    shells.write_exchange(source, tmp_path / "base", "pickle")
    return atoms, structure, moments, source, tmp_path / "base" / "TB2J.pickle"


def _template_args(inputs, output):
    return (
        "template",
        inputs[1],
        "--magnetic-atoms",
        "1",
        "--cutoff",
        "3.1",
        "--reference-spins",
        inputs[2],
        "-o",
        output,
    )


def test_template_manual_edit_validation_and_native_preset(tmp_path, inputs, capsys):
    path = tmp_path / "exchange.shells.toml"
    assert _run(*_template_args(inputs, path)) == 0
    assert _run("validate", path) == 0
    assert _run("validate", path, "--for-export") != 0
    assert "unfilled" in capsys.readouterr().err
    output = tmp_path / "native"
    assert _run("convert", path, "--to", "tb2j", "-o", output, "--base", inputs[4]) != 0
    assert not output.exists()
    path.write_text(path.read_text().replace('"unfilled"', "1.25"))
    assert _run("validate", path, "--for-export", "--check-producer-ops") == 0
    report = tmp_path / "report.json"
    assert (
        _run(
            "convert",
            path,
            "--to",
            "tb2j",
            "-o",
            output,
            "--base",
            inputs[4],
            "--allow-loss",
            "charges",
            "--report",
            report,
        )
        == 0
    )
    context = shells.NativeContext(
        atoms=inputs[0], index_spin=[0], reference_spins=[2.0], charges=[8.0]
    )
    for fmt, name in (
        ("pickle", "TB2J.pickle"),
        ("text", "exchange.out"),
        ("xml", "exchange.xml"),
    ):
        loaded = shells.read_exchange(output / name, fmt, context=context).spinio
        assert len(loaded.exchange_Jdict) == 6
        assert set(loaded.exchange_Jdict.values()) == {0.00125}
    assert (output / "structure.vasp").exists()
    written_report = json.loads(report.read_text())["write"]
    assert written_report["losses"][0][0:2] == ["charges", "xml"]


@pytest.mark.parametrize("mode", ["isotropic", "tensor"])
def test_template_and_native_compression_have_api_byte_parity(tmp_path, inputs, mode):
    cli = tmp_path / "cli.toml"
    api = tmp_path / "api.toml"
    assert _run(*_template_args(inputs, cli), "--mode", mode) == 0
    document = shells.template(inputs[0], [1], 3.1, reference_spins=[2.0], mode=mode)
    shells.write_shells(document, api)
    assert cli.read_bytes() == api.read_bytes()
    report = tmp_path / "conversion.json"
    assert (
        _run(
            "convert",
            inputs[4],
            "--from",
            "pickle",
            "--to",
            "shells",
            "-o",
            cli,
            "--mode",
            mode,
            "--report",
            report,
        )
        == 0
    )
    imported = shells.read_exchange(inputs[4], "pickle").spinio
    compressed = shells.compress(imported, mode=mode)
    shells.write_shells(compressed.document, api)
    assert cli.read_bytes() == api.read_bytes()
    expected_report = json.loads(json.dumps(asdict(compressed.report), default=str))
    assert json.loads(report.read_text())["compress"] == expected_report


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--symmetry", "sg"),
        ("--mode", "isotropic"),
        ("--magnetic-atoms", "1"),
        ("--reference-spins", "unused.toml"),
        ("--symprec", "1e-5"),
        ("--magnetic-tol", "1e-5"),
        ("--energy-unit", "meV"),
        ("--zero-tol", "1e-10"),
        ("--zero-tol", "0"),
        ("--value-atol", "1e-8"),
        ("--value-rtol", "1e-6"),
    ],
)
def test_shell_header_flags_reject_even_explicit_default_values(
    tmp_path, inputs, flag, value
):
    source = tmp_path / "template.toml"
    shells.write_shells(shells.template(inputs[0], [1], 3.1), source)
    output = tmp_path / "out"
    assert (
        _run(
            "convert",
            source,
            "--from",
            "shells",
            "--to",
            "pickle",
            "-o",
            output,
            flag,
            value,
        )
        != 0
    )
    assert not output.exists()


def test_snap_zero_is_native_compression_only(tmp_path, inputs):
    source = tmp_path / "filled.toml"
    document = shells.template(inputs[0], [1], 3.1, reference_spins=[2.0])
    document = replace(
        document,
        shells=tuple(replace(entry, Jiso=0.00125) for entry in document.shells),
    )
    shells.write_shells(document, source)
    output = tmp_path / "out"
    assert _run("convert", source, "--to", "pickle", "-o", output, "--snap-zero") != 0
    assert not output.exists()
    native = inputs[3]
    native.exchange_Jdict = {((1, 0, 0), 0, 0): 1e-12}
    shells.write_exchange(native, tmp_path / "tiny", "pickle")
    snapped = tmp_path / "snapped.toml"
    assert (
        _run(
            "convert",
            tmp_path / "tiny" / "TB2J.pickle",
            "--from",
            "pickle",
            "--to",
            "shells",
            "-o",
            snapped,
            "--snap-zero",
        )
        == 0
    )
    assert shells.read_shells(snapped).shells[0].Jiso == 0.0


def test_auto_detection_never_loads_pickle_or_unknown_bytes(tmp_path, inputs):
    output = tmp_path / "out.toml"
    assert _run("convert", inputs[4], "--to", "shells", "-o", output) != 0
    assert not output.exists()
    unknown = tmp_path / "unknown"
    unknown.write_bytes(b"not a supported exchange file\x00")
    assert _run("convert", unknown, "--to", "shells", "-o", output) != 0
    assert not output.exists()
    assert (
        _run("convert", inputs[4], "--from", "pickle", "--to", "shells", "-o", output)
        == 0
    )
    assert shells.read_shells(output).shells[0].Jiso == pytest.approx(0.00125)


def test_comment_suppression_is_semantic_only_on_template_and_conversion(
    tmp_path, inputs
):
    ordinary = tmp_path / "ordinary.toml"
    quiet = tmp_path / "quiet.toml"
    assert _run(*_template_args(inputs, ordinary)) == 0
    assert _run(*_template_args(inputs, quiet), "--no-equivalent-comments") == 0
    assert ordinary.read_bytes() != quiet.read_bytes()
    assert shells.read_shells(ordinary) == shells.read_shells(quiet)
    for path, flags in ((ordinary, ()), (quiet, ("--no-equivalent-comments",))):
        assert (
            _run(
                "convert",
                inputs[4],
                "--from",
                "pickle",
                "--to",
                "shells",
                "-o",
                path,
                *flags,
            )
            == 0
        )
    assert ordinary.read_bytes() != quiet.read_bytes()
    assert shells.read_shells(ordinary) == shells.read_shells(quiet)
    output = tmp_path / "native"
    assert (
        _run(
            "convert",
            ordinary,
            "--to",
            "pickle",
            "-o",
            output,
            "--no-equivalent-comments",
        )
        != 0
    )
    assert not output.exists()


HAND = '''exchange = "isotropic"
[structure]
coordinates = "fractional"
lattice = """
4 0 0
0.3 5 0
0.2 0.4 6
"""
atoms = """
1 Fe7 -0.87 0.21 0.17 Y 1
2 Co4 1.39 0.36 0.23 Y 0
"""
[reference_spins]
moments = """
Co4 -1
Fe7 2
"""
[symmetry]
mode = "spacegroup"
[[shell]]
representative = "Fe7 Co4 1 0 -1"
J = 2
[metadata]
sample = "hand-edited"
'''


@pytest.mark.parametrize("mode", ["isotropic", "tensor"])
def test_snapshot_free_unnamed_hand_file_retains_raw_native_identity(tmp_path, mode):
    source = tmp_path / "hand.toml"
    text = (
        HAND
        if mode == "isotropic"
        else HAND.replace('exchange = "isotropic"', 'exchange = "tensor"').replace(
            "J = 2", "J = [[2,0,0],[0,2,0],[0,0,2]]"
        )
    )
    source.write_text(text)
    canonical = tmp_path / "canonical.toml"
    assert _run("convert", source, "--to", "shells", "-o", canonical) != 0
    assert not canonical.exists()
    assert (
        _run(
            "convert",
            source,
            "--from",
            "shells",
            "--to",
            "shells",
            "-o",
            canonical,
            "--no-equivalent-comments",
        )
        == 0
    )
    document = shells.read_shells(canonical)
    assert document.shells[0].name is None
    assert document.shells[0].n_equivalent_pairs == 1
    expected_distance = np.linalg.norm(
        np.array([3.26, 0.15, -0.94]) @ np.asarray(document.cell_angstrom)
    )
    assert document.shells[0].distance == pytest.approx(expected_distance)
    native = tmp_path / "native"
    assert (
        _run("convert", source, "--from", "shells", "--to", "pickle", "-o", native) == 0
    )
    loaded = shells.read_exchange(native / "TB2J.pickle", "pickle").spinio
    assert loaded.index_spin == [1, 0]
    assert loaded.exchange_Jdict == {((1, 0, -1), 1, 0): 0.002}
    np.testing.assert_allclose(
        loaded.atoms.get_scaled_positions(wrap=False), document.positions_frac
    )
    assert loaded._shell_user_metadata == {"sample": "hand-edited"}
    back = tmp_path / "back.toml"
    assert (
        _run(
            "convert",
            native / "TB2J.pickle",
            "--from",
            "pickle",
            "--to",
            "shells",
            "-o",
            back,
            "--mode",
            mode,
        )
        == 0
    )
    assert shells.read_shells(back).metadata == document.metadata
    invalid = tmp_path / "invalid"
    assert (
        _run(
            "convert",
            source,
            "--from",
            "shells",
            "--to",
            "pickle",
            "-o",
            invalid,
            "--symmetry",
            "sg",
        )
        != 0
    )
    assert not invalid.exists()


def test_text_context_and_named_loss_cli_paths(tmp_path, inputs, capsys):
    source = inputs[3]
    source.damping = [0.1]
    source.gyro_ratio = [1.2]
    shells.write_exchange(source, tmp_path / "metadata", "pickle")
    output = tmp_path / "text"
    native = tmp_path / "metadata" / "TB2J.pickle"
    assert (
        _run("convert", native, "--from", "pickle", "--to", "text", "-o", output) != 0
    )
    assert not output.exists()
    assert (
        _run(
            "convert",
            native,
            "--from",
            "pickle",
            "--to",
            "text",
            "-o",
            output,
            "--allow-loss",
            "damping",
            "gyromag",
        )
        == 0
    )
    assert "damping" in capsys.readouterr().out
    converted = tmp_path / "from-text.toml"
    assert (
        _run("convert", output / "exchange.out", "--to", "shells", "-o", converted) != 0
    )
    assert not converted.exists()
    assert (
        _run(
            "convert",
            output / "exchange.out",
            "--to",
            "shells",
            "-o",
            converted,
            "--magnetic-atoms",
            "1",
            "--reference-spins",
            inputs[2],
        )
        == 0
    )
    assert shells.read_shells(converted).moments == (2.0,)


def test_modeled_spins_are_explicit_and_never_enable_msg(tmp_path, inputs):
    path = tmp_path / "model.toml"
    args = (
        "template",
        inputs[1],
        "--magnetic-atoms",
        "1",
        "--cutoff",
        "3.1",
        "--reference-spins-model",
        "unit-along-z",
        "-o",
        path,
    )
    assert _run(*args, "--symmetry", "msg") != 0
    assert not path.exists()
    assert _run(*args) == 0
    document = shells.read_shells(path)
    assert document.moment_source == "model:unit-along-z"
    assert document.moments == (1.0,)


@pytest.mark.parametrize("moments", ["[2.0]", "[[0.0, 0.0, 2.0]]"])
def test_msg_external_scalar_and_axial_context_have_api_parity(
    tmp_path, inputs, moments
):
    inputs[2].write_text(f"moments = {moments}\n")
    output = tmp_path / "msg.toml"
    assert (
        _run(*_template_args(inputs, output), "--symmetry", "msg", "--mode", "tensor")
        == 0
    )
    import tomli

    reference = tomli.loads(inputs[2].read_text())["moments"]
    api = shells.template(
        inputs[0], [1], 3.1, symmetry="msg", reference_spins=reference, mode="tensor"
    )
    expected = tmp_path / "api.toml"
    shells.write_shells(api, expected)
    assert output.read_bytes() == expected.read_bytes()


def test_xml_auto_detection_requires_and_accepts_identity_context(tmp_path, inputs):
    xml = tmp_path / "xml"
    shells.write_exchange(inputs[3], xml, "xml", allow_loss=("charges",))
    output = tmp_path / "from-xml.toml"
    assert _run("convert", xml / "exchange.xml", "--to", "shells", "-o", output) != 0
    assert not output.exists()
    assert (
        _run(
            "convert",
            xml / "exchange.xml",
            "--to",
            "shells",
            "-o",
            output,
            "--structure",
            inputs[1],
        )
        == 0
    )
    assert shells.read_shells(output).shells[0].Jiso == pytest.approx(0.00125)


def test_missing_hand_reference_row_rejects_before_native_output(tmp_path):
    source = tmp_path / "incomplete.toml"
    source.write_text(HAND.replace("Co4 -1\n", ""))
    output = tmp_path / "native"
    assert (
        _run("convert", source, "--from", "shells", "--to", "pickle", "-o", output) != 0
    )
    assert not output.exists()


@pytest.mark.parametrize("collision", ["shell_output", "native_artifact", "input"])
def test_report_collision_cannot_replace_exchange_data(tmp_path, collision):
    source = tmp_path / "hand.toml"
    source.write_text(HAND)
    original = source.read_bytes()
    target = "pickle" if collision == "native_artifact" else "shells"
    output = tmp_path / ("native" if target == "pickle" else "out.toml")
    report = {
        "shell_output": output,
        "native_artifact": output / "TB2J.pickle",
        "input": source,
    }[collision]
    assert (
        _run(
            "convert",
            source,
            "--from",
            "shells",
            "--to",
            target,
            "-o",
            output,
            "--report",
            report,
        )
        != 0
    )
    assert source.read_bytes() == original
    assert not output.exists()


def test_hardlinked_report_cannot_replace_input(tmp_path):
    source = tmp_path / "hand.toml"
    source.write_text(HAND)
    report = tmp_path / "report.json"
    report.hardlink_to(source)
    original = source.read_bytes()
    output = tmp_path / "out.toml"
    assert (
        _run(
            "convert",
            source,
            "--from",
            "shells",
            "--to",
            "shells",
            "-o",
            output,
            "--report",
            report,
        )
        != 0
    )
    assert source.read_bytes() == report.read_bytes() == original
    assert not output.exists()


@pytest.mark.parametrize("extra_banners", [0, 2])
def test_native_description_is_not_a_shell_signature(tmp_path, inputs, extra_banners):
    source = inputs[3]
    source.description = "schema_version = 2\n"
    native = tmp_path / "text"
    shells.write_exchange(source, native, "text")
    path = native / "exchange.out"
    path.write_text(("===\n\n" * extra_banners) + path.read_text())
    output = tmp_path / "converted.toml"
    assert (
        _run(
            "convert",
            native / "exchange.out",
            "--to",
            "shells",
            "-o",
            output,
            "--magnetic-atoms",
            "1",
            "--reference-spins",
            inputs[2],
        )
        == 0
    )
    assert shells.read_shells(output).shells[0].Jiso == pytest.approx(0.00125)


def test_quoted_root_schema_key_is_an_unambiguous_shell_signature(tmp_path):
    source = tmp_path / "quoted.toml"
    source.write_text('"schema_version" = 2\n' + HAND)
    output = tmp_path / "native"
    assert _run("convert", source, "--to", "pickle", "-o", output) == 0
    actual = shells.read_exchange(output / "TB2J.pickle", "pickle").spinio
    assert actual.exchange_Jdict == {((1, 0, -1), 1, 0): 0.002}
