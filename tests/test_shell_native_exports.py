"""Guarded native exports are exercised through real files and native readers."""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO

CELL = np.array([[4.1, 0.2, 0.1], [0.4, 5.2, 0.3], [0.2, 0.5, 6.3]])
FRACTIONAL_POSITIONS = np.array(
    [[0.13, 0.21, 0.17], [1.27, 0.11, 0.41], [-0.39, 0.36, 0.23]]
)
POSITIONS = FRACTIONAL_POSITIONS @ CELL
INDEX_SPIN = [1, -2, 0]
PAIR = ((1, -1, 2), 1, 0)


def _assert_no_output(directory):
    assert not directory.exists() or not any(directory.iterdir())


def _model(*, context=True, annotations=False):
    atoms = Atoms(
        ["Fe", "O", "Co"],
        cell=CELL,
        positions=POSITIONS,
        pbc=True,
    )
    source = SpinIO(
        atoms,
        spinat=np.array([[0.0, 0.0, 2.25], [0.0, 0.0, 0.0], [0.0, 0.0, -1.75]])
        if context
        else None,
        charges=np.array([8.1, 6.2, 9.3]) if context else None,
        index_spin=INDEX_SPIN.copy(),
        colinear=True,
        exchange_Jdict={PAIR: 0.00123456},
        dmi_ddict={PAIR: np.array([1.234e-5, -2.345e-5, 3.456e-5])},
        Jani_dict={
            PAIR: np.array(
                [
                    [2.1e-4, 1.2e-5, 0.0],
                    [1.2e-5, -1.1e-4, 2.3e-5],
                    [0.0, 2.3e-5, -1.0e-4],
                ]
            )
        },
    )
    # Constructor defaults are intentionally excluded from ordinary loss cases.
    source.damping = None
    source.gyro_ratio = None
    if annotations:
        source._shell_user_metadata = {
            "owner": "regression",
            "labels": ["alpha", "beta"],
        }
        source._shell_site_labels = ["Fe7", "O4", "Co2"]
    return source


def _context(source):
    return shells.NativeContext(
        atoms=source.atoms.copy(),
        index_spin=list(source.index_spin),
        reference_spins=np.array(source.spinat, copy=True),
        charges=np.array(source.charges, copy=True),
    )


def _assert_full_channels(actual, source, *, scalar=0.0, matrix=0.0):
    assert actual.index_spin == INDEX_SPIN
    assert set(actual.exchange_Jdict) == {PAIR}
    assert set(actual.dmi_ddict) == {PAIR}
    assert set(actual.Jani_dict) == {PAIR}
    assert actual.exchange_Jdict[PAIR] == pytest.approx(
        source.exchange_Jdict[PAIR], abs=scalar, rel=0
    )
    np.testing.assert_allclose(
        actual.dmi_ddict[PAIR], source.dmi_ddict[PAIR], atol=scalar, rtol=0
    )
    np.testing.assert_allclose(
        actual.Jani_dict[PAIR], source.Jani_dict[PAIR], atol=matrix, rtol=0
    )
    coordinate_quantum = 3e-6 if scalar == 5e-9 else 5e-9
    np.testing.assert_allclose(
        actual.atoms.cell.array,
        source.atoms.cell.array,
        atol=coordinate_quantum,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.atoms.positions, source.atoms.positions, atol=coordinate_quantum, rtol=0
    )
    assert actual.atoms.get_chemical_symbols() == source.atoms.get_chemical_symbols()


def _native_file(directory, fmt):
    return {
        "pickle": directory / "TB2J.pickle",
        "text": directory / "exchange.out",
        "xml": directory / "exchange.xml",
    }[fmt]


@pytest.mark.parametrize("fmt", ["pickle", "text", "xml"])
def test_native_full_channel_export_roundtrips_raw_frame_and_support(tmp_path, fmt):
    source = _model()
    original_positions = source.atoms.positions.copy()
    original_channels = (
        dict(source.exchange_Jdict),
        {key: value.copy() for key, value in source.dmi_ddict.items()},
        {key: value.copy() for key, value in source.Jani_dict.items()},
    )
    output = tmp_path / fmt
    report = shells.write_exchange(
        source,
        output,
        fmt,
        context=_context(source),
        allow_loss=("charges",) if fmt == "xml" else (),
    )
    path = _native_file(output, fmt)
    assert path.is_file()
    if fmt == "pickle":
        loaded = SpinIO.load_pickle(path=output)
        scalar_quantum = 0.0
        matrix_quantum = 0.0
    else:
        loaded = shells.read_exchange(path, fmt, context=_context(source)).spinio
        scalar_quantum = 5e-8 if fmt == "text" else 5e-9
        matrix_quantum = 5e-7 if fmt == "text" else 5e-9
    _assert_full_channels(loaded, source, scalar=scalar_quantum, matrix=matrix_quantum)
    restored_bond = (
        loaded.atoms.positions[2]
        + np.asarray(PAIR[0]) @ loaded.atoms.cell.array
        - loaded.atoms.positions[0]
    )
    # Native R translates the second endpoint; raw site images are part of that key.
    raw_bond = (
        source.atoms.positions[2]
        + np.asarray(PAIR[0]) @ source.atoms.cell.array
        - source.atoms.positions[0]
    )
    bond_quantum = 3e-6 if fmt == "xml" else 5e-9
    np.testing.assert_allclose(restored_bond, raw_bond, atol=bond_quantum, rtol=0)
    np.testing.assert_array_equal(source.atoms.positions, original_positions)
    assert source.exchange_Jdict == original_channels[0]
    np.testing.assert_array_equal(source.dmi_ddict[PAIR], original_channels[1][PAIR])
    np.testing.assert_array_equal(source.Jani_dict[PAIR], original_channels[2][PAIR])
    assert report.precision_profile
    assert report.quanta


def test_text_requires_real_charges_then_reports_damping_and_gyromag_loss(tmp_path):
    source = _model()
    source.charges = None
    output = tmp_path / "text"
    base_context = shells.NativeContext(
        atoms=source.atoms.copy(),
        index_spin=INDEX_SPIN,
        reference_spins=source.spinat.copy(),
    )
    with pytest.raises(ValueError, match="charges"):
        shells.write_exchange(source, output, "text", context=base_context)
    _assert_no_output(output)

    charges_context = shells.NativeContext(
        atoms=source.atoms.copy(),
        index_spin=INDEX_SPIN,
        reference_spins=source.spinat.copy(),
        charges=np.array([8.1, 6.2, 9.3]),
    )
    source.damping = [0.11, 0.22, 0.33]
    source.gyro_ratio = [1.1, 2.2, 3.3]
    with pytest.raises(ValueError) as error:
        shells.write_exchange(source, output, "text", context=charges_context)
    assert "damping" in str(error.value) and "gyromag" in str(error.value)
    _assert_no_output(output)
    report = shells.write_exchange(
        source,
        output,
        "text",
        context=charges_context,
        allow_loss=("damping", "gyromag"),
    )
    assert _native_file(output, "text").is_file()
    names = {row[0] for row in report.losses}
    assert {"damping", "gyromag"} <= names


def test_normalized_sia_tensor_is_named_loss_for_text_and_xml(tmp_path):
    source = _model()
    source = SpinIO(
        source.atoms,
        spinat=source.spinat.copy(),
        charges=source.charges.copy(),
        index_spin=source.index_spin.copy(),
        colinear=True,
        exchange_Jdict=dict(source.exchange_Jdict),
        k1=[0.002, 0.0, 0.0],
        k1dir=[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
    )
    source.damping = source.gyro_ratio = None
    for fmt in ("text", "xml"):
        output = tmp_path / fmt
        xml_charge_loss = ("charges",) if fmt == "xml" else ()
        with pytest.raises(ValueError, match="sia_tensor"):
            shells.write_exchange(
                source,
                output,
                fmt,
                context=_context(source),
                allow_loss=xml_charge_loss,
            )
        _assert_no_output(output)
        report = shells.write_exchange(
            source,
            output,
            fmt,
            context=_context(source),
            allow_loss=("sia_tensor",) + xml_charge_loss,
        )
        assert any(row[0] == "sia_tensor" for row in report.losses)
        assert _native_file(output, fmt).is_file()


def test_xml_unequal_diagonal_encoding_is_reported_and_reloads(tmp_path):
    source = _model()
    source.exchange_Jdict[PAIR] = np.array([0.0011, 0.0022, 0.0033])
    output = tmp_path / "xml"
    report = shells.write_exchange(
        source,
        output,
        "xml",
        context=_context(source),
        allow_loss=("charges",),
    )
    loaded = shells.read_exchange(
        _native_file(output, "xml"), "xml", context=_context(source)
    ).spinio
    total = loaded.exchange_Jdict[PAIR] * np.eye(3) + loaded.Jani_dict[PAIR]
    expected = np.diag(source.exchange_Jdict[PAIR]) + source.Jani_dict[PAIR]
    np.testing.assert_allclose(total, expected, atol=5e-9, rtol=0)
    assert report.encoding_changes


def test_tb2j_preset_preflights_all_members_and_writes_inherited_names(tmp_path):
    source = _model()
    source.damping = [0.1, 0.2, 0.3]
    failed = tmp_path / "blocked"
    with pytest.raises(ValueError, match="damping"):
        shells.write_exchange(
            source,
            failed,
            "tb2j",
            context=_context(source),
            allow_loss=("charges",),
        )
    _assert_no_output(failed)

    source.damping = None
    source.gyro_ratio = None
    output = tmp_path / "preset"
    shells.write_exchange(
        source, output, "tb2j", context=_context(source), allow_loss=("charges",)
    )
    assert {"TB2J.pickle", "exchange.out", "exchange.xml", "structure.vasp"} <= {
        path.name for path in output.iterdir()
    }
    structure = read(output / "structure.vasp")
    np.testing.assert_allclose(structure.cell.array, CELL, atol=5e-9, rtol=0)
    np.testing.assert_allclose(structure.positions, POSITIONS, atol=5e-9, rtol=0)
    assert structure.get_chemical_symbols() == ["Fe", "O", "Co"]


def test_pickle_absent_context_is_tagged_and_not_dynamics_ready(tmp_path):
    source = _model(context=False)
    report = shells.write_exchange(source, tmp_path, "pickle")
    loaded = SpinIO.load_pickle(path=tmp_path)
    assert loaded.spinat is None and loaded.charges is None
    assert any("dynamics" in str(message).lower() for message in report.messages)


def test_scalar_reference_context_becomes_signed_native_vectors(tmp_path):
    source = _model(context=False)
    scalar_moments = np.array([-2.25, 0.0, 1.75])
    context = shells.NativeContext(
        atoms=source.atoms.copy(),
        index_spin=INDEX_SPIN,
        reference_spins=scalar_moments,
        charges=np.array([8.1, 6.2, 9.3]),
    )
    shells.write_exchange(source, tmp_path, "pickle", context=context)
    loaded = SpinIO.load_pickle(path=tmp_path)
    assert loaded.colinear is True
    np.testing.assert_array_equal(
        loaded.spinat, [[0, 0, -2.25], [0, 0, 0], [0, 0, 1.75]]
    )


def test_pickle_restores_annotations_and_other_targets_report_omission(tmp_path):
    source = _model(annotations=True)
    shells.write_exchange(source, tmp_path / "pickle", "pickle")
    loaded = shells.read_exchange(tmp_path / "pickle" / "TB2J.pickle", "pickle").spinio
    assert loaded._shell_user_metadata == source._shell_user_metadata
    assert loaded._shell_site_labels == source._shell_site_labels
    report = shells.write_exchange(
        source, tmp_path / "text", "text", context=_context(source)
    )

    actions = " ".join(map(str, report.metadata_actions)).lower()
    assert "metadata" in actions and "label" in actions


def test_channel_only_xml_and_named_text_drops_keep_retained_channel_union(tmp_path):
    source = _model()
    dmi_key = ((-1, 0, 2), 0, 1)
    jani_key = ((2, 1, -1), 1, 0)
    source.dmi_ddict[dmi_key] = np.array([0.0001, 0.0002, -0.0003])
    source.Jani_dict[jani_key] = np.diag([0.0002, -0.0001, -0.0001])
    xml = tmp_path / "xml"
    shells.write_exchange(source, xml, "xml", allow_loss=("charges",))
    loaded = shells.read_exchange(
        xml / "exchange.xml", "xml", context=_context(source)
    ).spinio
    assert set(loaded.exchange_Jdict) == {PAIR}
    assert set(loaded.dmi_ddict) == {PAIR, dmi_key}
    assert set(loaded.Jani_dict) == {PAIR, jani_key}
    text = tmp_path / "text"
    with pytest.raises(ValueError):
        shells.write_exchange(source, text, "text")
    _assert_no_output(text)
    report = shells.write_exchange(source, text, "text", allow_loss=("dmi", "jani"))
    restored = shells.read_exchange(
        text / "exchange.out", "text", context=_context(source)
    ).spinio
    assert (
        set(restored.exchange_Jdict)
        == set(restored.dmi_ddict)
        == set(restored.Jani_dict)
        == {PAIR}
    )
    assert {row[0] for row in report.losses} == {"dmi", "jani"}
    np.testing.assert_allclose(
        restored.dmi_ddict[PAIR], source.dmi_ddict[PAIR], atol=5e-8
    )
    np.testing.assert_allclose(
        restored.Jani_dict[PAIR], source.Jani_dict[PAIR], atol=5e-7
    )
    assert set(source.dmi_ddict) == {PAIR, dmi_key}
    assert set(source.Jani_dict) == {PAIR, jani_key}


@pytest.mark.parametrize("decimals", [2, 6])
def test_unknown_numpy_matrix_printer_regime_blocks_whole_preset(tmp_path, decimals):
    source = _model()
    with np.printoptions(
        formatter={"float_kind": lambda value: f"{value:.{decimals}f}"}
    ):
        with pytest.raises(ValueError, match="regime"):
            shells.write_exchange(
                source, tmp_path / "preset", "tb2j", allow_loss=("charges",)
            )
    _assert_no_output(tmp_path / "preset")


@pytest.mark.parametrize("field", ["charges", "spinat", "exchange_Jdict"])
def test_nonfinite_native_context_fails_before_output(tmp_path, field):
    source = _model()
    if field == "exchange_Jdict":
        source.exchange_Jdict[PAIR] = float("nan")
    else:
        getattr(source, field).flat[0] = float("nan")
    with pytest.raises(ValueError):
        shells.write_exchange(source, tmp_path / "invalid", "pickle")
    _assert_no_output(tmp_path / "invalid")


@pytest.mark.parametrize("channel", ["dmi", "jani"])
def test_channel_only_text_drop_has_no_fabricated_isotropic_pairs(tmp_path, channel):
    source = _model()
    source.exchange_Jdict = {}
    if channel == "dmi":
        source.Jani_dict = None
    else:
        source.dmi_ddict = None
    output = tmp_path / "text"
    with pytest.raises(ValueError):
        shells.write_exchange(source, output, "text")
    _assert_no_output(output)
    report = shells.write_exchange(source, output, "text", allow_loss=(channel,))
    loaded = shells.read_exchange(
        output / "exchange.out", "text", context=_context(source)
    ).spinio
    assert not loaded.exchange_Jdict and not loaded.dmi_ddict and not loaded.Jani_dict
    assert report.losses[0][:3] == (channel, "text", (PAIR,))
    assert not source.exchange_Jdict
    assert set(getattr(source, {"dmi": "dmi_ddict", "jani": "Jani_dict"}[channel])) == {
        PAIR
    }


def test_collinear_text_rejects_transverse_moments_before_preset_io(tmp_path):
    source = _model()
    source.spinat[0, 0] = 0.5
    output = tmp_path / "preset"
    with pytest.raises(ValueError):
        shells.write_exchange(source, output, "tb2j", allow_loss=("charges",))
    _assert_no_output(output)


def test_structure_constraint_failure_preflights_entire_native_preset(tmp_path):
    from ase.constraints import FixedLine

    source = _model()
    source.atoms.set_constraint(FixedLine(0, [1, 1, 1]))
    output = tmp_path / "preset"
    with pytest.raises(RuntimeError):
        shells.write_exchange(source, output, "tb2j", allow_loss=("charges",))
    _assert_no_output(output)


@pytest.mark.parametrize("fmt", ["pickle", "xml"])
def test_structure_context_does_not_replace_native_isotope_masses(tmp_path, fmt):
    source = _model()
    masses = np.array([60.0, 18.0, 61.0])
    source.atoms.set_masses(masses)
    context = shells.NativeContext(
        atoms=Atoms(
            source.atoms.get_chemical_symbols(),
            cell=source.atoms.cell,
            positions=source.atoms.positions,
            pbc=True,
        ),
        index_spin=source.index_spin,
    )
    output = tmp_path / fmt
    shells.write_exchange(
        source,
        output,
        fmt,
        context=context,
        allow_loss=("charges",) if fmt == "xml" else (),
    )
    loaded = shells.read_exchange(
        _native_file(output, fmt), fmt, context=context
    ).spinio
    np.testing.assert_allclose(loaded.atoms.get_masses(), masses, atol=5e-5, rtol=0)
    np.testing.assert_array_equal(source.atoms.get_masses(), masses)


@pytest.mark.parametrize("attribute", ["NJT_Jdict", "NJT_ddict"])
def test_text_normalizes_retained_njt_vectors_without_source_mutation(
    tmp_path, attribute
):
    source = _model()
    values = [0.0001, -0.0002, 0.0003]
    setattr(source, attribute, {PAIR: values.copy()})
    shells.write_exchange(source, tmp_path, "text")
    field = "Jani_NJt:" if attribute == "NJT_Jdict" else "DMI_NJt:"
    line = next(
        line
        for line in (tmp_path / "exchange.out").read_text().splitlines()
        if field in line
    )
    printed = np.fromstring(line.split(":", 1)[1].strip().strip("()"), sep=" ") * 1e-3
    np.testing.assert_allclose(printed, values, atol=5e-8, rtol=0)
    np.testing.assert_array_equal(getattr(source, attribute)[PAIR], values)


@pytest.mark.parametrize("stage", ["decomposition", "merge"])
def test_finite_diagonal_overflow_rejects_pickle_before_output(tmp_path, stage):
    source = _model()
    if stage == "decomposition":
        source.exchange_Jdict[PAIR] = np.array([1e308, 0.0, 0.0])
    else:
        source.exchange_Jdict[PAIR] = np.array([7e307, 0.0, 0.0])
        source.Jani_dict[PAIR] = np.diag([1.5e308, -7.5e307, -7.5e307])
    output = tmp_path / "pickle"
    with pytest.raises(ValueError):
        shells.write_exchange(source, output, "pickle")
    _assert_no_output(output)


@pytest.mark.parametrize("attribute", ["dJdx", "dJdx2"])
def test_partial_derivative_loss_names_populated_keys_removed(tmp_path, attribute):
    source = _model()
    other = ((-1, 1, -2), 0, 1)
    source.exchange_Jdict[other] = 0.0005
    setattr(source, attribute, {PAIR: 0.0001})
    output = tmp_path / attribute
    with pytest.raises(ValueError):
        shells.write_exchange(source, output, "text")
    _assert_no_output(output)
    report = shells.write_exchange(source, output, "text", allow_loss=("djdx",))
    assert report.losses[0][:3] == ("djdx", "text", (PAIR,))
    loaded = shells.read_exchange(
        output / "exchange.out", "text", context=_context(source)
    ).spinio
    assert set(loaded.exchange_Jdict) == {PAIR, other}
    assert getattr(source, attribute) == {PAIR: 0.0001}
