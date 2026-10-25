"""Public shell expansion preserves context and never hides ancillary loss."""

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO


def _source(*, ancillary=True, colinear=True):
    atoms = Atoms(
        ["Fe", "O", "Co"],
        cell=[[4.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.2, 0.4, 6.0]],
        scaled_positions=[(0.13, 0.21, 0.17), (0.27, 0.11, 0.41), (0.39, 0.36, 0.23)],
        pbc=True,
    )
    atoms.set_masses([56.0, 17.0, 60.0])
    key = ((1, 0, -1), 1, 0)
    source = SpinIO(
        atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        charges=np.array([8.0, 6.0, 9.0]) if ancillary else None,
        index_spin=[1, -2, 0],
        colinear=colinear,
        exchange_Jdict={key: 0.002},
        damping=[0.1, 0.2, 0.3] if ancillary else None,
        gyro_ratio=[2.0, 3.0, 4.0] if ancillary else None,
        description="Native context must survive shell edits.",
    )
    if not ancillary:
        source.damping = None
        source.gyro_ratio = None
    return source, key


def _edit(document, value):
    assert len(document.shells) == 1
    return replace(document, shells=(replace(document.shells[0], Jiso=value),))


def test_native_precision_and_ancillary_requirements_survive_shell_save_reload(
    tmp_path,
):
    source, key = _source()
    source.write_pickle(path=tmp_path)
    native = shells.read_exchange(tmp_path / "TB2J.pickle", "pickle")
    compressed = shells.compress(native.spinio, mode="isotropic")
    path = tmp_path / "shells.toml"
    shells.write_shells(compressed.document, path)
    document = shells.read_shells(path)
    assert document.source_format == "pickle" and document.precision_profile == "exact"
    assert set(document.required_base_fields) == {"charges", "damping", "gyromag"}
    with pytest.raises(ValueError) as error:
        shells.expand(document)
    assert all(name in str(error.value) for name in document.required_base_fields)


def test_public_base_update_changes_only_exchange_and_rebuilds_exact_derived_map():
    source, key = _source()
    source.biquadratic_Jdict = {key: 0.0004}
    source.NJT_ddict = {key: np.array([0.0001, 0.0002, 0.0003])}
    source.Rlist = [(99, 0, 0)]
    source.distance_dict = {key: (np.zeros(3), -1.0)}
    document = shells.compress(source, mode="isotropic").document
    result = shells.expand(_edit(document, 0.003), base=source)
    output = result.spinio
    assert output.exchange_Jdict == {key: 0.003}
    assert output.dmi_ddict is None and output.Jani_dict is None
    assert output.index_spin == [1, -2, 0]
    np.testing.assert_array_equal(output.spinat, source.spinat)
    np.testing.assert_array_equal(output.charges, source.charges)
    np.testing.assert_array_equal(output.atoms.get_masses(), source.atoms.get_masses())
    np.testing.assert_array_equal(output.damping, source.damping)
    np.testing.assert_array_equal(output.gyro_ratio, source.gyro_ratio)
    assert output.description == source.description
    assert output.biquadratic_Jdict == source.biquadratic_Jdict
    np.testing.assert_array_equal(output.NJT_ddict[key], source.NJT_ddict[key])
    vector = (
        source.atoms.positions[2]
        + np.array(key[0]) @ source.atoms.cell.array
        - source.atoms.positions[0]
    )
    np.testing.assert_allclose(output.distance_dict[key][0], vector, atol=1e-14, rtol=0)
    assert output.distance_dict[key][1] == pytest.approx(np.linalg.norm(vector))
    assert output.Rlist == [(0, 0, 0), key[0]]
    assert (
        source.exchange_Jdict == {key: 0.002} and source.distance_dict[key][1] == -1.0
    )
    assert ("charges", "keep") in {
        (row[0], row[1]) for row in result.report.metadata_actions
    }


@pytest.mark.parametrize(
    "field", ["cell", "positions", "species", "index_spin", "moments"]
)
def test_incompatible_base_fails_naming_the_context_field(field):
    source, key = _source()
    document = shells.compress(source, mode="isotropic").document
    if field == "cell":
        source.atoms.cell[0, 0] += 0.1
    elif field == "positions":
        source.atoms.positions[0, 0] += 0.1
    elif field == "species":
        source.atoms[0].symbol = "Ni"
    elif field == "index_spin":
        source.index_spin = [0, -2, 1]
    else:
        source.spinat[0, 2] += 0.1
    with pytest.raises(ValueError, match=field):
        shells.expand(document, base=source)


def test_edited_values_require_named_drop_of_each_stale_derived_group():
    source, key = _source()
    source.Jiso_orb = {key: np.array([[0.002]])}
    source.dJdx = {key: np.array([0.1, 0.2, 0.3])}
    source.dJdx_orb = {key: np.array([[0.1]])}
    document = shells.compress(source, mode="isotropic").document
    equal = shells.expand(document, base=source)
    np.testing.assert_array_equal(equal.spinio.Jiso_orb[key], source.Jiso_orb[key])
    np.testing.assert_array_equal(equal.spinio.dJdx[key], source.dJdx[key])
    changed = _edit(document, 0.003)
    with pytest.raises(ValueError, match="orbital.*djdx|djdx.*orbital"):
        shells.expand(changed, base=source)
    with pytest.raises(ValueError, match="djdx"):
        shells.expand(changed, base=source, allow_loss=("orbital",))
    result = shells.expand(changed, base=source, allow_loss=("orbital", "djdx"))
    assert result.spinio.Jiso_orb is None and result.spinio.dJdx is None
    assert result.spinio.dJdx_orb is None
    assert {row[0] for row in result.report.losses} == {"orbital", "djdx"}
    assert result.spinio.exchange_Jdict == {key: 0.003}
    assert source.Jiso_orb is not None and source.dJdx is not None


def test_charge_loss_is_explicit_and_does_not_fabricate_writer_context():
    source, key = _source(ancillary=False)
    source.charges = np.array([8.0, 6.0, 9.0])
    document = shells.compress(source, mode="isotropic").document
    assert document.required_base_fields == ("charges",)
    with pytest.raises(ValueError, match="charges"):
        shells.expand(document)
    result = shells.expand(document, allow_loss=("charges",))
    assert result.spinio.charges is None
    assert result.spinio.damping is None and result.spinio.gyro_ratio is None
    assert result.spinio.exchange_Jdict == {key: 0.002}
    assert {row[0] for row in result.report.losses} == {"charges"}
    assert any(
        "charge" in message and "target" in message
        for message in result.report.messages
    )


def test_compatible_base_must_supply_each_manifest_requirement():
    source, key = _source(ancillary=False)
    document = replace(
        shells.compress(source, mode="isotropic").document,
        required_base_fields=("charges", "orbital"),
    )
    with pytest.raises(ValueError) as error:
        shells.expand(document, base=source)
    assert "charges" in str(error.value) and "orbital" in str(error.value)
    result = shells.expand(document, base=source, allow_loss=("charges", "orbital"))
    assert {row[0] for row in result.report.losses} == {"charges", "orbital"}
    assert result.spinio.charges is None and result.spinio.Jiso_orb is None


def test_legacy_sia_normalizes_once_without_mutating_base():
    source, key = _source()
    source.k1 = np.array([0.001, 0.0])
    source.k1dir = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source.has_uniaxial_anistropy = True
    document = shells.compress(source, mode="isotropic").document
    assert "sia" in document.required_base_fields
    result = shells.expand(document, base=source)
    np.testing.assert_array_equal(
        result.spinio.sia_tensor[0], np.diag([0.001, 0.0, 0.0])
    )
    assert result.spinio.k1 is None and result.spinio.k1dir is None
    assert source.sia_tensor is None and source.k1[0] == 0.001


@pytest.mark.parametrize("scalar", [False, True])
def test_explicit_reference_shape_controls_native_collinearity(scalar):
    source, key = _source(ancillary=False, colinear=not scalar)
    reference = source.spinat[:, 2] if scalar else source.spinat
    document = shells.compress(
        source, mode="isotropic", reference_spins=reference
    ).document
    result = shells.expand(document, base=source)
    assert result.spinio.colinear is scalar
    np.testing.assert_array_equal(result.spinio.spinat, source.spinat)
    if scalar:
        np.testing.assert_array_equal(result.spinio.magmoms, [2.0, 0.0, -1.0])


def test_public_completion_preserves_sparse_support_unless_explicitly_requested():
    atoms = Atoms("Fe", cell=3 * np.eye(3), pbc=True)
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict={key: 0.002},
    )
    source.damping = None
    source.gyro_ratio = None
    document = shells.compress(source, mode="isotropic").document
    partial = shells.expand(document)
    complete = shells.expand(document, complete_orbits=True)
    assert partial.spinio.exchange_Jdict == {key: 0.002}
    assert set(complete.spinio.exchange_Jdict) == {
        ((a, b, c), 0, 0)
        for a, b, c in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    }
    assert set(complete.report.created_pair_keys) == set(
        complete.spinio.exchange_Jdict
    ) - {key}
    assert complete.spinio.dmi_ddict is None and complete.spinio.Jani_dict is None
    assert any("not calculated" in message for message in complete.report.messages)


def test_public_expansion_rejects_unfilled_values_and_never_snaps_authored_values():
    atoms = Atoms("Fe", cell=3 * np.eye(3), pbc=True)
    unfilled = shells.template(atoms, magnetic_atoms=[1], cutoff=3.1, mode="isotropic")
    with pytest.raises(ValueError, match="(?i)unfilled|Jiso"):
        shells.expand(unfilled)
    source, key = _source(ancillary=False)
    source.exchange_Jdict[key] = 1e-11
    document = shells.compress(source, mode="isotropic").document
    result = shells.expand(document)
    assert result.spinio.exchange_Jdict[key] == 1e-11
    assert result.report.snapped_values == ()


def test_numpy_index_map_keeps_native_key_identity_through_base_update():
    source, key = _source(ancillary=False)
    source.index_spin = np.array(source.index_spin, dtype=np.int64)
    document = shells.compress(source, mode="isotropic").document
    result = shells.expand(document, base=source)
    np.testing.assert_array_equal(result.spinio.index_spin, [1, -2, 0])
    assert result.spinio.exchange_Jdict == {key: 0.002}
    assert result.spinio.ind_atoms == {1: 0, 0: 2}


def test_encoding_normalization_does_not_invalidate_equal_physical_derived_data():
    source, key = _source()
    source.exchange_Jdict = None
    source.Jani_dict = {key: np.eye(3) * 0.002}
    source.Jiso_orb = {key: np.array([[0.002]])}
    document = shells.compress(source, mode="tensor").document
    result = shells.expand(document, base=source)
    total = result.spinio.exchange_Jdict[key] * np.eye(3) + result.spinio.Jani_dict[key]
    np.testing.assert_allclose(total, source.Jani_dict[key], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(result.spinio.Jiso_orb[key], source.Jiso_orb[key])
    assert result.report.losses == ()


def test_scalar_reference_never_relabels_a_transverse_base_as_collinear():
    source, key = _source(ancillary=False, colinear=False)
    source.spinat[0, 0] = 1e-7
    document = shells.compress(
        source, mode="isotropic", reference_spins=source.spinat[:, 2], magnetic_tol=1e-5
    ).document
    with pytest.raises(ValueError, match="moments"):
        shells.expand(document, base=source)


@pytest.mark.parametrize("symmetry", ["sg", "msg"])
@pytest.mark.parametrize("mode", ["isotropic", "tensor"])
def test_raw_site_images_keep_native_keys_and_bond_vectors(tmp_path, symmetry, mode):
    source, key = _source(ancillary=False)
    offsets = np.array([[-1, 0, 2], [0, 1, -1], [2, -1, 0]])
    source.atoms.positions += offsets @ source.atoms.cell.array
    if mode == "tensor":
        source.dmi_ddict = {key: np.array([0.0003, -0.0002, 0.0001])}
        source.Jani_dict = {key: np.diag([0.0001, -0.0001, 0.0])}
    expected_vector = (
        source.atoms.positions[2]
        + np.array(key[0]) @ source.atoms.cell.array
        - source.atoms.positions[0]
    )
    document = shells.compress(source, symmetry=symmetry, mode=mode).document
    path = tmp_path / "raw.shells.toml"
    shells.write_shells(document, path)
    document = shells.read_shells(path)
    np.testing.assert_allclose(
        document.positions_frac,
        source.atoms.get_scaled_positions(wrap=False),
        atol=1e-14,
        rtol=0,
    )
    for base in (None, source):
        output = shells.expand(document, base=base).spinio
        assert output.index_spin == [1, -2, 0]
        assert set(output.exchange_Jdict) == {key}
        assert output.exchange_Jdict[key] == pytest.approx(0.002, abs=1e-15)
        np.testing.assert_allclose(
            output.distance_dict[key][0], expected_vector, atol=1e-14, rtol=0
        )
        assert output.distance_dict[key][1] == pytest.approx(
            np.linalg.norm(expected_vector)
        )
        if mode == "tensor":
            np.testing.assert_allclose(
                output.dmi_ddict[key], source.dmi_ddict[key], atol=1e-15
            )
            np.testing.assert_allclose(
                output.Jani_dict[key], source.Jani_dict[key], atol=1e-15
            )
    # Periodic equivalence cannot authorize replay of unchanged native R.
    source.atoms.positions[0] += source.atoms.cell.array[0]
    with pytest.raises(ValueError, match="positions"):
        shells.expand(document, base=source)


@pytest.mark.parametrize("symmetry", ["sg", "msg"])
def test_template_uses_raw_images_for_short_physical_bonds(symmetry):
    source, key = _source(ancillary=False)
    source.atoms.positions += (
        np.array([[-1, 0, 2], [0, 1, -1], [2, -1, 0]]) @ source.atoms.cell.array
    )
    document = shells.template(
        source.atoms, [1, 3], 1.5, symmetry=symmetry, reference_spins=source.spinat
    )
    np.testing.assert_allclose(
        document.positions_frac,
        source.atoms.get_scaled_positions(wrap=False),
        atol=1e-14,
    )
    # This intersite neighbor requires translations well outside +/- one cell.
    pair = next(
        entry.representative
        for entry in document.shells
        if entry.representative[:2] == (1, 3)
    )
    assert max(abs(value) for value in pair[2:]) > 1
    cell = source.atoms.cell.array
    vector = (
        source.atoms.positions[2]
        + np.array(pair[2:]) @ cell
        - source.atoms.positions[0]
    )
    assert np.linalg.norm(vector) <= 1.5
