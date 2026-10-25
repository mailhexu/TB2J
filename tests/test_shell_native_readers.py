"""Native reader behavior on real file boundaries, without writer mocks."""

import os
import pickle
import re

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict


def test_explicit_pickle_preserves_channel_only_support_and_context_absence(tmp_path):
    atoms = Atoms("Fe", cell=3 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        dmi_ddict={key: np.array([0.001, 0.002, 0.003])},
    )
    source.damping = None
    source.gyro_ratio = None
    source.write_pickle(path=tmp_path, fname="native.pickle")
    result = shells.read_exchange(tmp_path / "native.pickle", "pickle")
    assert result.spinio.exchange_Jdict is None
    assert result.spinio.Jani_dict is None
    assert set(result.spinio.dmi_ddict) == {key}
    np.testing.assert_array_equal(result.spinio.dmi_ddict[key], [0.001, 0.002, 0.003])
    assert result.spinio.spinat is None and result.spinio.charges is None
    assert result.spinio.damping is None and result.spinio.gyro_ratio is None
    assert result.report.precision_profile == "exact"


def test_unknown_format_never_unpickles_input(tmp_path):
    marker = tmp_path / "unpickle-side-effect"

    class Payload:
        def __reduce__(self):
            return os.mkdir, (str(marker),)

    path = tmp_path / "unknown"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(ValueError, match="format"):
        shells.read_exchange(path, "auto")
    assert not marker.exists()


def test_xml_requires_explicit_species_identity(tmp_path):
    path = tmp_path / "exchange.xml"
    path.write_text(
        '<System_definition><unit_cell units="bohrradius">'
        "5.66918e+00 0.00000e+00 0.00000e+00 "
        "0.00000e+00 5.66918e+00 0.00000e+00 "
        "0.00000e+00 0.00000e+00 5.66918e+00</unit_cell>"
        '<atom mass="5.58450e+01" damping_factor="0.10000" '
        'gyroratio="1.00000e+00" index_spin="1">'
        '<position units="bohrradius">0.00000e+00 0.00000e+00 0.00000e+00</position>'
        "<spinat>0.00000e+00 0.00000e+00 1.00000e+00</spinat></atom>"
        '<spin_exchange_list units="eV"><nterms>1</nterms><spin_exchange_term>'
        "<ijR>1 1 1 0 0</ijR><data>1.00000e-03 2.00000e-03 3.00000e-03</data>"
        "</spin_exchange_term></spin_exchange_list></System_definition>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="species|structure"):
        shells.read_exchange(path, "xml")


def _write_native_text(spinio, path):
    # The existing writer requires derived distances; import-side physical
    # context does not imply the future guarded export/preflight contract.
    spinio._build_Rlist()
    magnetic_atoms = [
        spinio.index_spin.index(spin) for spin in range(len(spinio.ind_atoms))
    ]
    spinio.distance_dict = gen_distance_dict(magnetic_atoms, spinio.atoms, spinio.Rlist)
    spinio.write_txt(path=path)


def _current_text(
    tmp_path, *, anisotropic=True, colinear=False, moment=-2.0, njt=False
):
    atoms = Atoms("Fe", cell=3 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=np.array([[0.0, 0.0, moment]]),
        charges=np.array([8.0]),
        index_spin=[0],
        colinear=colinear,
        exchange_Jdict={key: 0.00123456},
        dmi_ddict={key: np.array([0.00001234, -0.00002345, 0.00003456])}
        if anisotropic
        else None,
        Jani_dict={key: np.diag([0.0002, -0.0001, -0.0001])} if anisotropic else None,
        NJT_ddict={key: np.array([0.0001, 0.0002, 0.0003])} if njt else None,
    )
    _write_native_text(source, tmp_path)
    return tmp_path / "exchange.out", source, key


def test_current_text_channels_are_authoritative_at_declared_precision(tmp_path):
    path, source, key = _current_text(tmp_path)
    result = shells.read_exchange(path, "text")
    assert set(result.spinio.exchange_Jdict) == {key}
    assert result.spinio.exchange_Jdict[key] == pytest.approx(0.0012346, abs=1e-16)
    np.testing.assert_allclose(
        result.spinio.dmi_ddict[key], source.dmi_ddict[key], atol=5e-8, rtol=0
    )
    np.testing.assert_allclose(
        result.spinio.Jani_dict[key], source.Jani_dict[key], atol=5e-7, rtol=0
    )
    np.testing.assert_array_equal(result.spinio.spinat, [[0.0, 0.0, -2.0]])
    np.testing.assert_array_equal(result.spinio.charges, [8.0])
    assert result.report.precision_profile == "exchange.out-default"


def test_text_combined_mismatch_fails_without_replacing_channels(tmp_path):
    path, source, key = _current_text(tmp_path)
    head, matrix = path.read_text().split("Combined J tensor", 1)
    matrix = re.sub(r"\[\[\s*[-+.\deE]+", "[[9.000", matrix, count=1)
    path.write_text(head + "Combined J tensor" + matrix)
    with pytest.raises(ValueError, match="(?i)combined"):
        shells.read_exchange(path, "text")


def test_vintage_text_without_combined_blocks_keeps_authoritative_values(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False)
    text = path.read_text()
    path.write_text(text.split("Combined J tensor", 1)[0])
    result = shells.read_exchange(path, "text")
    assert result.spinio.exchange_Jdict[key] == pytest.approx(0.0012346, abs=1e-16)
    assert result.spinio.dmi_ddict is None and result.spinio.Jani_dict is None


def test_positive_collinear_text_requires_signed_reference_context(tmp_path):
    path, source, key = _current_text(
        tmp_path, anisotropic=False, colinear=True, moment=2.0
    )
    with pytest.raises(ValueError, match="(?i)reference|direction|moment"):
        shells.read_exchange(path, "text")
    result = shells.read_exchange(
        path, "text", context=shells.NativeContext(reference_spins=[-2.0])
    )
    np.testing.assert_array_equal(result.spinio.spinat, [[0.0, 0.0, -2.0]])
    assert result.spinio.colinear is True
    assert result.spinio.index_spin == [0]


def test_text_unsupported_njt_requires_named_loss_and_remains_absent(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False, njt=True)
    with pytest.raises(ValueError, match="njt"):
        shells.read_exchange(path, "text")
    result = shells.read_exchange(path, "text", allow_loss=("njt",))
    assert result.spinio.dmi_ddict is None and result.spinio.Jani_dict is None
    assert tuple(row[0] for row in result.report.losses) == ("njt",)


def _current_xml(tmp_path, *, diagonal=False, dmi_only=False):
    from TB2J.io_exchange.io_multibinit import write_xml

    atoms = Atoms("Fe", cell=3 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict=None
        if dmi_only
        else {key: np.array([0.001, 0.002, 0.003]) if diagonal else 0.002},
        dmi_ddict={key: np.array([0.001, 0.002, 0.003])} if dmi_only else None,
    )
    path = tmp_path / "exchange.xml"
    write_xml(source, path)
    return path, key


def test_xml_unequal_exchange_diagonal_is_not_scalar_truncated(tmp_path):
    path, key = _current_xml(tmp_path, diagonal=True)
    result = shells.read_exchange(
        path, "xml", context=shells.NativeContext(species=["Fe"])
    )
    total = result.spinio.exchange_Jdict[key] * np.eye(3) + result.spinio.Jani_dict[key]
    np.testing.assert_allclose(
        total, np.diag([0.001, 0.002, 0.003]), atol=1e-15, rtol=0
    )
    assert result.spinio.dmi_ddict is None
    assert result.spinio.index_spin == [0]
    assert result.spinio.atoms.get_chemical_symbols() == ["Fe"]
    assert result.report.precision_profile == "xml-%.5e"


def test_xml_reuses_nterms_integrity_assertion(tmp_path):
    path, key = _current_xml(tmp_path)
    path.write_text(
        path.read_text().replace("<nterms>1</nterms>", "<nterms>2</nterms>", 1)
    )
    with pytest.raises(AssertionError, match="nterms"):
        shells.read_exchange(path, "xml", context=shells.NativeContext(species=["Fe"]))


def test_xml_channel_only_payload_does_not_invent_isotropic_entries(tmp_path):
    path, key = _current_xml(tmp_path, dmi_only=True)
    result = shells.read_exchange(
        path, "xml", context=shells.NativeContext(species=["Fe"])
    )
    assert result.spinio.exchange_Jdict is None and result.spinio.Jani_dict is None
    assert set(result.spinio.dmi_ddict) == {key}
    np.testing.assert_allclose(
        result.spinio.dmi_ddict[key], [0.001, 0.002, 0.003], atol=1e-15
    )


def test_text_full_atom_labels_respect_explicit_permuted_magnetic_map(tmp_path):
    atoms = Atoms(
        ["Fe", "O", "Co"],
        cell=5 * np.eye(3),
        pbc=True,
        scaled_positions=[(0.1, 0.2, 0.3), (0.2, 0.1, 0.4), (0.4, 0.3, 0.1)],
    )
    key = ((1, 0, -1), 1, 0)
    source = SpinIO(
        atoms,
        spinat=np.array([[0, 0, 2], [0, 0, -0.05], [0, 1, 0]]),
        charges=np.array([8, 6, 9]),
        index_spin=[1, -1, 0],
        colinear=False,
        exchange_Jdict={key: 0.0032},
    )
    _write_native_text(source, tmp_path)
    path = tmp_path / "exchange.out"
    with pytest.raises(ValueError, match="(?i)membership|magnetic|index_spin"):
        shells.read_exchange(path, "text")
    result = shells.read_exchange(
        path, "text", context=shells.NativeContext(index_spin=[1, -1, 0])
    )
    assert result.spinio.index_spin == [1, -1, 0]
    assert result.spinio.atoms.get_chemical_symbols() == ["Fe", "O", "Co"]
    assert result.spinio.exchange_Jdict == {key: 0.0032}
    np.testing.assert_array_equal(result.spinio.spinat, source.spinat)


def test_negative_collinear_text_preserves_explicit_signed_moments(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False, colinear=True)
    result = shells.read_exchange(path, "text")
    np.testing.assert_array_equal(result.spinio.spinat, [[0.0, 0.0, -2.0]])
    assert result.spinio.colinear is True
    assert result.spinio.exchange_Jdict[key] == pytest.approx(0.0012346, abs=1e-16)


def test_pickle_scalar_context_produces_consistent_native_collinear_output(tmp_path):
    atoms = Atoms("Fe", cell=3 * np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict={key: 0.002},
    )
    source.write_pickle(path=tmp_path, fname="absent-context.pickle")
    result = shells.read_exchange(
        tmp_path / "absent-context.pickle",
        "pickle",
        context=shells.NativeContext(reference_spins=[-2.0], charges=[8.0]),
    )
    output = tmp_path / "text"
    _write_native_text(result.spinio, output)
    reread = shells.read_exchange(output / "exchange.out", "text")
    np.testing.assert_array_equal(reread.spinio.spinat, [[0.0, 0.0, -2.0]])
    np.testing.assert_array_equal(reread.spinio.charges, [8.0])
    assert reread.spinio.exchange_Jdict == {key: 0.002}


def test_scientific_text_preserves_explicit_large_and_tiny_exchange_tokens(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False)
    prefix, combined = path.read_text().split("Combined J tensor", maxsplit=1)
    matrix = (
        "[[ 1.235e+103 -2.346e+103  1.111e-097]\n"
        " [ 2.001e+104  5.679e+103 -9.877e-098]\n"
        " [ 3.211e+103  4.321e+103  7.654e+103]]"
    )
    path.write_text(
        prefix
        + "[Testing!]J_ani:\n"
        + matrix
        + "\nCombined J tensor"
        + combined.splitlines()[0]
        + "\n"
        + matrix
        + "\n"
    )
    result = shells.read_exchange(path, "text")
    expected = np.array(
        [
            [1.235e100, -2.346e100, 1.111e-100],
            [2.001e101, 5.679e100, -9.877e-101],
            [3.211e100, 4.321e100, 7.654e100],
        ]
    )
    np.testing.assert_array_equal(result.spinio.Jani_dict[key], expected)


def test_unknown_scalar_precision_is_rejected_not_silently_assumed(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False)
    path.write_text(
        re.sub(r"(J_iso:\s*[-+]?\d+\.\d{3})\d", r"\1", path.read_text(), count=1)
    )
    with pytest.raises(ValueError, match="(?i)precision|printer regime"):
        shells.read_exchange(path, "text")


@pytest.mark.parametrize("reference", [[[2.0, 0.0, 0.0]], [[0.0, 0.0, -2.0]]])
def test_unsigned_text_magnitudes_accept_explicit_axial_directions(tmp_path, reference):
    path, source, key = _current_text(
        tmp_path, anisotropic=False, colinear=True, moment=2.0
    )
    result = shells.read_exchange(
        path, "text", context=shells.NativeContext(reference_spins=reference)
    )
    np.testing.assert_array_equal(result.spinio.spinat, reference)
    assert result.spinio.colinear is False


def test_vector_text_honors_matching_explicit_scalar_reference_mode(tmp_path):
    path, source, key = _current_text(tmp_path, anisotropic=False)
    result = shells.read_exchange(
        path, "text", context=shells.NativeContext(reference_spins=[-2.0])
    )
    np.testing.assert_array_equal(result.spinio.spinat, [[0, 0, -2]])
    assert result.spinio.colinear is True


def test_xml_exact_nonmagnetic_sentinel_survives_native_offset(tmp_path):
    from TB2J.io_exchange.io_multibinit import write_xml

    atoms = Atoms(
        "FeO",
        cell=4 * np.eye(3),
        pbc=True,
        scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)],
    )
    source = SpinIO(
        atoms,
        spinat=np.array([[0, 0, 2], [0, 0, 0]]),
        charges=None,
        index_spin=[0, -2],
        colinear=False,
        exchange_Jdict={((1, 0, 0), 0, 0): 0.002},
    )
    path = tmp_path / "exchange.xml"
    write_xml(source, path)
    result = shells.read_exchange(
        path, "xml", context=shells.NativeContext(species=["Fe", "O"])
    )
    assert result.spinio.index_spin == [0, -2]
    assert result.spinio.exchange_Jdict == source.exchange_Jdict


def test_combined_check_excludes_isotropic_uncertainty_off_diagonal(tmp_path):
    atoms = Atoms("Fe", cell=4 * np.eye(3), pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=np.zeros((1, 3)),
        charges=np.array([8.0]),
        index_spin=[0],
        colinear=False,
        exchange_Jdict={key: 0.001},
        dmi_ddict={key: np.array([0.0, 0.0, 6e-7])},
    )
    _write_native_text(source, tmp_path)
    path = tmp_path / "exchange.out"
    head, combined = path.read_text().split("Combined J tensor", 1)
    combined = combined.replace("0.001", "0.000", 1)
    path.write_text(head + "Combined J tensor" + combined)
    with pytest.raises(ValueError, match="(?i)combined"):
        shells.read_exchange(path, "text")
