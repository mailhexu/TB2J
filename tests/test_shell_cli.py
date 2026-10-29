"""Consumer-visible shell CLI behavior; scientific algorithms stay in the API."""

from dataclasses import replace

import numpy as np
import pytest
import tomli
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO

CORE_RESULTS = ("TB2J.pickle", "exchange.out", "structure.vasp")
SOLVERS = ("Multibinit", "TomASD", "Vampire", "ESPInS")
NO_SPIN_OMISSION = "magnetic reference moments are missing or zero"
DAMPING_OMISSION = "damping or gyromagnetic ratio is missing"


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
    source.write_pickle(tmp_path / "base")
    return atoms, structure, moments, source, tmp_path / "base" / "TB2J.pickle"


def _generate_args(inputs, output, *extra):
    return (
        "generate",
        inputs[1],
        "--magnetic-atoms",
        "1",
        "--cutoff",
        "3.1",
        "-o",
        output,
        *extra,
    )


def _fill(path, value=0.00125):
    document = shells.read_shells(path)
    filled = replace(
        document,
        shells=tuple(replace(shell, Jiso=value) for shell in document.shells),
    )
    shells.write_shells(filled, path)
    return filled


def test_structure_generate_matches_public_api_bytes(tmp_path, inputs):
    cli = tmp_path / "cli.toml"
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-species",
            "Fe",
            "--cutoff",
            "3.1",
            "--reference-spins",
            inputs[2],
            "-o",
            cli,
        )
        == 0
    )
    api = tmp_path / "api.toml"
    shells.write_shells(
        shells.template(inputs[0], [1], 3.1, reference_spins=[2.0], mode="isotropic"),
        api,
    )
    assert cli.read_bytes() == api.read_bytes()


def test_species_and_atom_selectors_produce_identical_documents(tmp_path, inputs):
    species = tmp_path / "species.toml"
    atoms = tmp_path / "atoms.toml"
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-species",
            "Fe",
            "--cutoff",
            "3.1",
            "-o",
            species,
        )
        == 0
    )
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-atoms",
            "1",
            "--cutoff",
            "3.1",
            "-o",
            atoms,
        )
        == 0
    )
    assert species.read_bytes() == atoms.read_bytes()
    document = shells.read_shells(species)
    assert document.magnetic_atoms == (1,)
    assert len(document.shells) == 1


def test_generate_selection_and_route_errors_are_actionable(tmp_path, inputs, capsys):
    output = tmp_path / "out.toml"
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-species",
            "Fe",
            "--magnetic-atoms",
            "1",
            "--cutoff",
            "3.1",
            "-o",
            output,
        )
        == 2
    )
    assert not output.exists()
    assert _run("generate", inputs[1], "--cutoff", "3.1", "-o", output) == 2
    captured = capsys.readouterr()
    assert "exactly one of magnetic_species or magnetic_atoms" in captured.err
    assert _run(*_generate_args(inputs, output, "--project")) == 2
    assert "project applies only to imported exchange values" in capsys.readouterr().err
    assert (
        _run(
            "generate",
            tmp_path / "missing.vasp",
            "--magnetic-atoms",
            "1",
            "--cutoff",
            "3.1",
            "-o",
            output,
        )
        == 2
    )
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-atoms",
            "2",
            "--cutoff",
            "3.1",
            "-o",
            output,
        )
        == 2
    )
    assert not output.exists()


def test_pickle_generate_writes_companion_and_imports_values(tmp_path, inputs, capsys):
    output = tmp_path / "native.toml"
    assert _run("generate", inputs[4], "-o", output) == 0
    companion = tmp_path / "native.context.npz"
    assert companion.is_file()
    captured = capsys.readouterr()
    assert "native.context.npz" in captured.out
    assert "move it together" in captured.out
    document = shells.read_shells(output)
    assert document.source_format == "pickle"
    assert document.context_sha256
    assert document.energy_unit == "meV"
    assert document.shells[0].Jiso == pytest.approx(0.00125)
    assert document.moments == (2.0,)


def test_pickle_generate_rejects_structure_controls(tmp_path, inputs, capsys):
    output = tmp_path / "out.toml"
    for flag, value in (
        ("--magnetic-species", "Fe"),
        ("--magnetic-atoms", "1"),
        ("--cutoff", "3.1"),
        ("--reference-spins", inputs[2]),
    ):
        assert _run("generate", inputs[4], flag, value, "-o", output) == 2
        assert not output.exists()
    assert "pickle input does not accept" in capsys.readouterr().err


@pytest.mark.parametrize("moments", ["[2.0]", "[[0.0, 0.0, 2.0]]"])
def test_reference_moments_file_enables_msg_with_api_parity(
    tmp_path, inputs, moments, capsys
):
    inputs[2].write_text(f"moments = {moments}\n")
    output = tmp_path / "msg.toml"
    assert (
        _run(
            *_generate_args(
                inputs,
                output,
                "--symmetry",
                "msg",
                "--mode",
                "tensor",
                "--reference-spins",
                inputs[2],
            )
        )
        == 0
    )
    document = shells.read_shells(output)
    assert document.moment_source == "explicit"
    expected_moments = (2.0,) if moments == "[2.0]" else ((0.0, 0.0, 2.0),)
    assert document.moments == expected_moments
    reference = tomli.loads(inputs[2].read_text())["moments"]
    api = tmp_path / "api.toml"
    shells.write_shells(
        shells.template(
            inputs[0],
            [1],
            3.1,
            symmetry="msg",
            reference_spins=reference,
            mode="tensor",
        ),
        api,
    )
    assert output.read_bytes() == api.read_bytes()
    missing = tmp_path / "missing.toml"
    assert _run(*_generate_args(inputs, missing, "--symmetry", "msg")) != 0
    assert not missing.exists()
    assert "moments" in capsys.readouterr().err


def test_symmetry_tolerances_are_forwarded(tmp_path, inputs):
    output = tmp_path / "tolerances.toml"
    assert (
        _run(
            *_generate_args(
                inputs, output, "--symprec", "1e-4", "--magnetic-tol", "1e-3"
            )
        )
        == 0
    )
    document = shells.read_shells(output)
    assert document.symprec == pytest.approx(1e-4)
    assert document.magnetic_tol == pytest.approx(1e-3)


def test_no_equivalent_comments_suppresses_only_pair_lists(tmp_path, inputs):
    ordinary = tmp_path / "ordinary.toml"
    quiet = tmp_path / "quiet.toml"
    assert _run(*_generate_args(inputs, ordinary)) == 0
    assert _run(*_generate_args(inputs, quiet, "--no-equivalent-comments")) == 0
    assert ordinary.read_bytes() != quiet.read_bytes()
    assert shells.read_shells(ordinary) == shells.read_shells(quiet)
    text = quiet.read_text()
    assert "Equivalent ordered pairs" not in text
    assert "representative: first label" in text


def test_unfilled_export_fails_before_publication(tmp_path, inputs, capsys):
    source = tmp_path / "template.toml"
    assert _run(*_generate_args(inputs, source)) == 0
    results = tmp_path / "TB2J_results"
    assert _run("export", source, "-o", results) != 0
    assert not results.exists()
    assert "unfilled" in capsys.readouterr().err


def test_structure_origin_export_reports_every_omission(tmp_path, inputs, capsys):
    source = tmp_path / "template.toml"
    assert _run(*_generate_args(inputs, source)) == 0
    _fill(source)
    results = tmp_path / "TB2J_results"
    assert _run("export", source, "-o", results) == 0
    captured = capsys.readouterr()
    for target in SOLVERS:
        assert f"omitted {target}: {NO_SPIN_OMISSION}" in captured.err
    assert "Wrote standard TB2J results to" in captured.out
    for name in CORE_RESULTS:
        assert (results / name).is_file()
    for target in SOLVERS:
        assert not (results / target).exists()


def test_structure_origin_export_with_moments_reloads_native(tmp_path, inputs, capsys):
    source = tmp_path / "template.toml"
    assert _run(*_generate_args(inputs, source, "--reference-spins", inputs[2])) == 0
    _fill(source)
    results = tmp_path / "TB2J_results"
    assert _run("export", source, "-o", results) == 0
    errors = capsys.readouterr().err
    for target in ("Multibinit", "TomASD"):
        assert f"omitted {target}: {DAMPING_OMISSION}" in errors
    loaded = SpinIO.load_pickle(results)
    assert loaded.index_spin == [0]
    assert len(loaded.exchange_Jdict) == 6
    np.testing.assert_allclose(sorted(loaded.exchange_Jdict.values()), [0.00125] * 6)


def test_pickle_origin_export_survives_relocation_and_edit(tmp_path, inputs, capsys):
    origin = tmp_path / "origin.toml"
    assert _run("generate", inputs[4], "-o", origin) == 0
    moved = tmp_path / "moved"
    moved.mkdir()
    source = moved / "exchange.shells.toml"
    origin.rename(source)
    (tmp_path / "origin.context.npz").rename(moved / "exchange.shells.context.npz")
    document = shells.read_shells(source)
    shells.write_shells(
        replace(
            document,
            shells=tuple(replace(shell, Jiso=0.002) for shell in document.shells),
        ),
        source,
    )
    results = tmp_path / "results"
    assert _run("export", source, "-o", results) == 0
    loaded = SpinIO.load_pickle(results)
    assert loaded.index_spin == [0]
    assert loaded.exchange_Jdict == {((1, 0, 0), 0, 0): pytest.approx(0.002)}
    np.testing.assert_allclose(loaded.charges, [8.0])
    orphan = tmp_path / "orphan.toml"
    source.rename(orphan)
    lonely = tmp_path / "lonely-results"
    assert _run("export", orphan, "-o", lonely) != 0
    assert "companion" in capsys.readouterr().err
    assert not lonely.exists()


def test_export_updates_existing_results_preserving_unrelated(tmp_path, inputs):
    source = tmp_path / "template.toml"
    assert _run(*_generate_args(inputs, source)) == 0
    _fill(source)
    results = tmp_path / "TB2J_results"
    assert _run("export", source, "-o", results) == 0
    notes = results / "notes.txt"
    notes.write_text("user data\n")
    assert _run("export", source, "-o", results) == 0
    assert notes.read_text() == "user data\n"
    for name in CORE_RESULTS:
        assert (results / name).is_file()


def test_export_refuses_to_replace_hardlinked_input(tmp_path, inputs):
    source = tmp_path / "template.toml"
    assert _run(*_generate_args(inputs, source)) == 0
    _fill(source)
    results = tmp_path / "TB2J_results"
    results.mkdir()
    alias = results / "TB2J.pickle"
    alias.hardlink_to(source)
    original = source.read_bytes()
    assert _run("export", source, "-o", results) != 0
    assert source.read_bytes() == alias.read_bytes() == original


WITHDRAWN_WITH_VALUE = (
    ("--from", "pickle"),
    ("--to", "pickle"),
    ("--base", "base.pickle"),
    ("--original-pickle", "base.pickle"),
    ("--value-atol", "1e-8"),
    ("--value-rtol", "1e-6"),
    ("--zero-tol", "1e-10"),
    ("--energy-unit", "meV"),
    ("--reference-spins-model", "unit-along-z"),
    ("--report", "report.json"),
    ("--allow-loss", "charges"),
)
WITHDRAWN_BARE = (
    "--for-export",
    "--check-producer-ops",
    "--complete-orbits",
    "--snap-zero",
)


def test_withdrawn_commands_and_flags_are_gone(tmp_path, inputs):
    source = tmp_path / "doc.toml"
    assert _run(*_generate_args(inputs, source)) == 0
    for command in ("template", "convert", "validate"):
        assert _run(command, inputs[1]) == 2
    unused = tmp_path / "unused.toml"
    for flag in WITHDRAWN_WITH_VALUE:
        assert _run(*_generate_args(inputs, unused, *flag)) == 2
        assert _run("export", source, *flag) == 2
    for flag in WITHDRAWN_BARE:
        assert _run(*_generate_args(inputs, unused, flag)) == 2
        assert _run("export", source, flag) == 2
    assert _run(*_generate_args(inputs, unused, "--symprec", "abc")) == 2
    assert not unused.exists()


def test_default_paths(tmp_path, inputs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        _run(
            "generate",
            inputs[1],
            "--magnetic-atoms",
            "1",
            "--cutoff",
            "3.1",
            "--reference-spins",
            inputs[2],
        )
        == 0
    )
    assert (tmp_path / "exchange.shells.toml").is_file()
    _fill(tmp_path / "exchange.shells.toml")
    assert _run("export", "exchange.shells.toml") == 0
    assert (tmp_path / "TB2J_results" / "TB2J.pickle").is_file()
