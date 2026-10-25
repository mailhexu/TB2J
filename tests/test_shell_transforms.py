"""Consumer regressions for shell compression and internal replay."""

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO


def test_isotropic_sparse_support_roundtrip_preserves_absent_channels():
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict={((1, 0, 0), 0, 0): 0.1},
    )
    result = shells.compress(source, mode="tensor")
    assert [
        (
            entry.id,
            entry.n_equivalent_pairs,
            entry.support_members,
            entry.Jiso_members,
            entry.DMI_members,
            entry.Jani_members,
        )
        for entry in result.document.shells
    ] == [("1:1:-1,0,0", 6, (5,), (5,), (), ())]
    replay = shells.expand(result.document, base=source)
    assert replay.spinio.exchange_Jdict == source.exchange_Jdict
    assert not replay.spinio.dmi_ddict
    assert not replay.spinio.Jani_dict
    assert replay.report.absent_channel_keys == (
        ("DMI", ((1, 0, 0), 0, 0)),
        ("Jani", ((1, 0, 0), 0, 0)),
    )
    assert replay.spinio.spinat is None and replay.spinio.charges is None
    again = shells.compress(replay.spinio, mode="tensor")
    assert again.document == result.document


def _cubic_source(**channels):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    return SpinIO(
        atoms, spinat=None, charges=None, index_spin=[0], colinear=False, **channels
    )


def test_equal_distance_inequivalent_bonds_remain_distinct_shells():
    atoms = Atoms(
        ["Fe", "Co", "Ni"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[(0, 0, 0), (0.25, 0, 0), (0, 0.25, 0)],
        pbc=True,
    )
    assert atoms.get_distance(0, 1, mic=True) == pytest.approx(1.0)
    assert atoms.get_distance(0, 2, mic=True) == pytest.approx(1.0)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0, 1, 2],
        colinear=False,
        exchange_Jdict={
            ((0, 0, 0), 0, 1): 0.1,
            ((0, 0, 0), 0, 2): 0.2,
        },
    )

    result = shells.compress(source, mode="isotropic")

    assert len(result.document.shells) == 2
    assert {entry.Jiso for entry in result.document.shells} == {0.1, 0.2}
    assert len({entry.id for entry in result.document.shells}) == 2


def test_tensor_transport_matches_independent_cubic_orbit_oracle():
    orbit = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    observed = {}
    for vector in orbit:
        axis = int(np.flatnonzero(vector)[0])
        expected = np.diag([-0.5, -0.5, -0.5])
        expected[axis, axis] = 1.0
        observed[(vector, 0, 0)] = expected
    source = _cubic_source(Jani_dict=observed)

    result = shells.compress(source)

    entry = result.document.shells[0]
    assert entry.id == "1:1:-1,0,0"
    assert entry.n_equivalent_pairs == 6
    assert entry.support_members == tuple(range(6))
    np.testing.assert_allclose(entry.tensor, np.diag([1.0, -0.5, -0.5]), atol=1e-12)
    replay = shells._expand_core(result.document)
    assert set(replay.spinio.Jani_dict) == set(observed)
    for key, value in observed.items():
        np.testing.assert_allclose(replay.spinio.Jani_dict[key], value, atol=1e-12)
    again = shells.compress(replay.spinio)
    assert again.document.shells[0].support_members == entry.support_members
    assert again.document.shells[0].Jani_members == entry.Jani_members
    np.testing.assert_allclose(
        again.document.shells[0].tensor, entry.tensor, atol=1e-12
    )


def test_inconsistent_orbit_fails_or_projects_with_member_residuals():
    values = {}
    for vector, value in (
        ((1, 0, 0), 1.0),
        ((-1, 0, 0), 1.0),
        ((0, 1, 0), 3.0),
        ((0, -1, 0), 3.0),
        ((0, 0, 1), 2.0),
        ((0, 0, -1), 2.0),
    ):
        values[(vector, 0, 0)] = value
    source = _cubic_source(exchange_Jdict=values)
    from TB2J.tests.test_symmetrize_J import _brute_force_orbit_means

    atom_values = {(0, 0, R): value for (R, i, j), value in values.items()}
    oracle_values, partition = _brute_force_orbit_means(source.atoms, atom_values)
    assert len(partition) == 1
    oracle_mean = next(iter(oracle_values.values()))

    with pytest.raises(ValueError, match=r"1:1:-1,0,0.*Jiso.*deviation"):
        shells.compress(source, mode="isotropic")

    projected = shells.compress(source, mode="isotropic", on_inconsistent="project")
    assert projected.document.shells[0].Jiso == pytest.approx(oracle_mean)
    assert len(projected.report.residuals) >= len(values)
    assert all(
        row[0] == "1:1:-1,0,0" and row[1] == "Jiso"
        for row in projected.report.residuals
    )


def test_sparse_channels_keep_independent_masks_and_reject_incomplete_values():
    source = _cubic_source(
        exchange_Jdict={((1, 0, 0), 0, 0): 0.1},
        Jani_dict={((0, 1, 0), 0, 0): np.diag([1.0, -2.0, 1.0])},
    )

    result = shells.compress(source)

    entry = result.document.shells[0]
    orbit = sorted(
        {
            (1, 1, x, y, z)
            for x, y, z in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]
        }
    )
    assert entry.support_members == tuple(
        sorted(
            (
                orbit.index((1, 1, 1, 0, 0)),
                orbit.index((1, 1, 0, 1, 0)),
            )
        )
    )
    assert entry.Jiso_members == (orbit.index((1, 1, 1, 0, 0)),)
    assert entry.Jani_members == (orbit.index((1, 1, 0, 1, 0)),)
    assert entry.DMI_members == ()
    replay = shells._expand_core(result.document).spinio
    assert set(replay.exchange_Jdict) == set(source.exchange_Jdict)
    np.testing.assert_allclose(
        replay.exchange_Jdict[((1, 0, 0), 0, 0)], 0.1, atol=1e-12
    )
    assert set(replay.Jani_dict) == set(source.Jani_dict)
    np.testing.assert_allclose(
        replay.Jani_dict[((0, 1, 0), 0, 0)],
        source.Jani_dict[((0, 1, 0), 0, 0)],
        atol=1e-12,
    )
    assert not replay.dmi_ddict

    incomplete = _cubic_source(dmi_ddict={((1, 0, 0), 0, 0): [0.1, 0.2]})
    with pytest.raises(ValueError, match="underdetermined"):
        shells.compress(incomplete)


def test_bond_fixpoint_allows_site_stabilizer_anisotropy_and_rejects_bad_paths():
    source = _cubic_source(Jani_dict={((1, 0, 0), 0, 0): np.diag([2.0, -1.0, -1.0])})
    document = shells.compress(source).document

    replay = shells._expand_core(document).spinio
    assert set(replay.Jani_dict) == {((1, 0, 0), 0, 0)}
    np.testing.assert_allclose(
        replay.Jani_dict[((1, 0, 0), 0, 0)], np.diag([2.0, -1.0, -1.0])
    )

    object.__setattr__(
        document.shells[0], "tensor", tuple(map(tuple, np.diag([2.0, -2.0, 0.0])))
    )
    with pytest.raises(ValueError, match=r"shell .*image|shell .*fixpoint"):
        shells._expand_core(document)


def test_completion_reports_created_keys_and_reciprocity_is_present_only():
    source = _cubic_source(exchange_Jdict={((1, 0, 0), 0, 0): 0.1})
    document = shells.compress(source, mode="isotropic").document

    partial = shells._expand_core(document).spinio
    assert set(partial.exchange_Jdict) == set(source.exchange_Jdict)
    completed = shells._expand_core(document, complete_orbits=True)
    expected = {
        ((x, y, z), 0, 0)
        for x, y, z in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    }
    assert set(completed.spinio.exchange_Jdict) == expected
    assert set(completed.report.created_pair_keys) == expected - set(
        source.exchange_Jdict
    )
    assert set(completed.report.created_channel_keys) == {
        ("Jiso", key) for key in expected - set(source.exchange_Jdict)
    }
    assert not completed.spinio.dmi_ddict and not completed.spinio.Jani_dict

    inconsistent_reverse = _cubic_source(
        exchange_Jdict={
            ((1, 0, 0), 0, 0): 0.1,
            ((-1, 0, 0), 0, 0): 0.2,
        }
    )
    with pytest.raises(ValueError, match="recipro"):
        shells.compress(inconsistent_reverse, mode="isotropic")
    projected = shells.compress(
        inconsistent_reverse, mode="isotropic", on_inconsistent="project"
    )
    assert projected.document.shells[0].Jiso == pytest.approx(0.15)


def test_present_reciprocity_projects_values_across_distinct_shells():
    atoms = Atoms(
        ["Fe", "Co"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict={
            ((0, 0, 0), 0, 1): 0.1,
            ((0, 0, 0), 1, 0): 0.2,
        },
    )

    with pytest.raises(ValueError, match="recipro"):
        shells.compress(source, mode="isotropic")

    projected = shells.compress(source, mode="isotropic", on_inconsistent="project")
    assert {entry.id for entry in projected.document.shells} == {
        "1:2:0,0,0",
        "2:1:0,0,0",
    }
    assert all(entry.Jiso == pytest.approx(0.15) for entry in projected.document.shells)


def test_snap_zero_is_reported_and_exact_index_spin_and_context_replay():
    near_zero = _cubic_source(exchange_Jdict={((0, 0, 0), 0, 0): 1e-11})
    unchanged = shells.compress(near_zero, mode="isotropic")
    snapped = shells.compress(near_zero, mode="isotropic", snap_zero=True)
    assert unchanged.document.shells[0].Jiso == 1e-11
    assert snapped.document.shells[0].Jiso == 0.0
    snap = snapped.report.snapped_values[0]
    assert snap == ("Jiso", ((0, 0, 0), 0, 0), 1e-11, 0.0)

    atoms = Atoms(
        ["Fe", "O", "Co"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[(0, 0, 0), (0.2, 0.2, 0.2), (0.35, 0.11, 0.07)],
        pbc=True,
    )
    moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    source = SpinIO(
        atoms,
        spinat=moments,
        charges=None,
        index_spin=[1, -1, 0],
        colinear=False,
        exchange_Jdict={((0, 0, 0), 1, 0): 0.4},
    )
    result = shells.compress(source, mode="isotropic")
    assert result.document.index_spin == (1, -1, 0)
    assert result.document.moments == tuple(map(tuple, moments))
    replay = shells._expand_core(result.document).spinio
    assert replay.index_spin == [1, -1, 0]
    assert replay.atoms.get_chemical_symbols() == ["Fe", "O", "Co"]
    np.testing.assert_array_equal(replay.spinat, moments)
    assert replay.damping is None and replay.gyro_ratio is None
    assert replay.charges is None


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("operations", "unprimed identity"),
        ("canonical_id", "canonical minimum"),
        ("representative", "support_members"),
        ("presence_mask", "empty mask would discard"),
    ],
)
def test_core_expansion_revalidates_mutated_documents_before_spinio(
    monkeypatch, mutation, diagnostic
):
    document = shells.compress(
        _cubic_source(Jani_dict={((1, 0, 0), 0, 0): np.diag([2.0, -1.0, -1.0])})
    ).document
    entry = document.shells[0]
    if mutation == "operations":
        object.__setattr__(
            document,
            "operations",
            (
                shells.ShellOperation(
                    rotation=((0, -1, 0), (1, 0, 0), (0, 0, 1)),
                    translation=(0.0, 0.0, 0.0),
                ),
            ),
        )
    elif mutation == "canonical_id":
        object.__setattr__(entry, "id", "1:1:0,0,0")
    elif mutation == "representative":
        object.__setattr__(entry, "representative", (1, 1, 0, 1, 0))
    else:
        object.__setattr__(entry, "Jani_members", ())

    from TB2J.io_exchange import io_exchange

    def forbidden_constructor(**kwargs):
        raise AssertionError("SpinIO construction preceded document revalidation")

    monkeypatch.setattr(io_exchange, "SpinIO", forbidden_constructor)
    with pytest.raises(ValueError, match=diagnostic):
        shells._expand_core(document)


def test_noncanonical_jani_is_preserved_by_reported_decomposition():
    atoms = Atoms(
        ["Fe", "Co"],
        cell=np.eye(3) * 4.0,
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    key = ((0, 0, 0), 0, 1)
    raw = np.array(
        [[0.001553, 0.004815, 0.0], [0.004815, 0.001553, 0.0], [0.0, 0.0, 0.001553]]
    )
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0, 1],
        colinear=False,
        Jani_dict={key: raw},
    )

    result = shells.compress(source)

    entry = result.document.shells[0]
    assert entry.Jiso_members == (0,)
    assert entry.DMI_members == ()
    assert entry.Jani_members == (0,)
    assert result.report.encoding_changes
    replay = shells._expand_core(result.document).spinio
    normalized = replay.exchange_Jdict[key] * np.eye(3) + replay.Jani_dict[key]
    np.testing.assert_allclose(normalized, raw, atol=1e-15)


@pytest.mark.parametrize(
    "moments,colinear,expected_primed,native",
    [
        pytest.param([1.0, -1.0], True, True, False, id="scalar-collinear"),
        pytest.param(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], False, False, True, id="axial-vector"
        ),
    ],
)
def test_msg_snapshot_uses_scalar_sign_and_vector_axial_moments(
    moments, colinear, expected_primed, native
):
    atoms = Atoms(
        "Fe2",
        cell=np.eye(3) * 4.0,
        scaled_positions=[(0.25, 0, 0), (0.75, 0, 0)],
        pbc=True,
    )
    source = SpinIO(
        atoms,
        spinat=np.asarray(moments) if native else None,
        charges=None,
        index_spin=[0, 1],
        colinear=colinear,
        exchange_Jdict={},
    )
    result = shells.compress(
        source,
        symmetry="msg",
        reference_spins=None if native else moments,
    )

    inversion = [
        operation
        for operation in result.document.operations
        if np.array_equal(operation.rotation, -np.eye(3, dtype=int))
        and np.allclose(operation.translation, (0.0, 0.0, 0.0), atol=1e-12)
    ]
    assert inversion and inversion[0].time_reversal is expected_primed
    replay = shells._expand_core(result.document).spinio
    assert replay.colinear is colinear
    if colinear:
        np.testing.assert_array_equal(replay.spinat[:, 2], [1.0, -1.0])
    else:
        np.testing.assert_array_equal(replay.spinat, np.asarray(moments))


def test_noncanonical_jani_is_normalized_before_sparse_channel_comparison():
    source = _cubic_source(
        exchange_Jdict={((1, 0, 0), 0, 0): 0.1},
        Jani_dict={((0, 1, 0), 0, 0): np.eye(3) * 0.2},
    )
    with pytest.raises(ValueError, match=r"Jiso.*deviation"):
        shells.compress(source)
    projected = shells.compress(source, on_inconsistent="project")
    replay = shells._expand_core(projected.document).spinio
    assert set(replay.exchange_Jdict) == {((1, 0, 0), 0, 0), ((0, 1, 0), 0, 0)}
    for value in replay.exchange_Jdict.values():
        assert value == pytest.approx(0.15)
    assert set(replay.Jani_dict) == {((0, 1, 0), 0, 0)}
    np.testing.assert_allclose(replay.Jani_dict[((0, 1, 0), 0, 0)], 0, atol=1e-16)
    assert projected.report.encoding_changes


def test_hexagonal_tensor_transport_uses_cartesian_not_fractional_rotation():
    atoms = Atoms(
        "Fe",
        cell=[[2, 0, 0], [1, np.sqrt(3), 0], [0, 0, 5]],
        scaled_positions=[(0, 0, 0)],
        pbc=True,
    )
    scale = 1 / 1024
    # Exact coefficients verified with the registered 60-degree rotation
    # in the managed SymPy arithmetic proof, independently of pair_images.
    axis = np.diag([2.0, -1.0, -1.0]) * scale
    rotated = (
        np.array(
            [
                [-0.25, 3 * np.sqrt(3) / 4, 0],
                [3 * np.sqrt(3) / 4, 1.25, 0],
                [0, 0, -1.0],
            ]
        )
        * scale
    )
    reflected = rotated.copy()
    reflected[0, 1] *= -1
    reflected[1, 0] *= -1
    values = {
        ((1, 0, 0), 0, 0): axis,
        ((-1, 0, 0), 0, 0): axis,
        ((0, 1, 0), 0, 0): rotated,
        ((0, -1, 0), 0, 0): rotated,
        ((-1, 1, 0), 0, 0): reflected,
        ((1, -1, 0), 0, 0): reflected,
    }
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        index_spin=[0],
        colinear=False,
        Jani_dict=values,
    )
    document = shells.compress(source).document
    assert document.shells[0].n_equivalent_pairs == 6
    assert document.shells[0].Jiso_members == ()
    assert document.shells[0].DMI_members == ()
    replay = shells._expand_core(document).spinio
    assert set(replay.Jani_dict) == set(values)
    assert replay.exchange_Jdict is None and replay.dmi_ddict is None
    for key, expected in values.items():
        np.testing.assert_allclose(replay.Jani_dict[key], expected, atol=1e-15, rtol=0)


def test_sparse_inversion_bond_requires_endpoint_transpose():
    document = shells.ShellDocument(
        cell_angstrom=tuple(map(tuple, 3.0 * np.eye(3))),
        species=("Fe",),
        positions_frac=((0.0, 0.0, 0.0),),
        magnetic_atoms=(1,),
        index_spin=(0,),
        operations=tuple(
            shells.ShellOperation(
                tuple(map(tuple, sign * np.eye(3, dtype=int))), (0.0, 0.0, 0.0)
            )
            for sign in (1, -1)
        ),
        shells=(
            shells.ShellEntry(
                id="1:1:-1,0,0",
                tensor=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
                representative=(1, 1, 1, 0, 0),
                n_equivalent_pairs=2,
                support_members=(1,),
                Jiso_members=(),
                DMI_members=(1,),
                Jani_members=(),
            ),
        ),
        exchange_mode="tensor",
    )
    with pytest.raises(ValueError, match="endpoint.*transpose"):
        shells._expand_core(document)


@pytest.mark.parametrize("project", [False, True])
def test_identity_roundoff_preserves_p1_dmi(project):
    atoms = Atoms(
        ["Fe", "Co", "Ni"],
        cell=49.0 * np.eye(3),
        pbc=True,
        scaled_positions=[(0.11, 0.23, 0.37), (0.31, 0.16, 0.28), (0.43, 0.19, 0.61)],
    )
    key = ((0, 0, 0), 0, 1)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        colinear=False,
        index_spin=[0, 1, 2],
        dmi_ddict={key: np.array([0.001, 0.0, 0.0])},
    )
    result = shells.compress(source, on_inconsistent="project" if project else "error")
    assert len(result.document.operations) == 1
    replay = shells._expand_core(result.document)
    assert set(replay.spinio.dmi_ddict) == {key}
    np.testing.assert_allclose(
        replay.spinio.dmi_ddict[key], [0.001, 0.0, 0.0], atol=1e-15, rtol=1e-12
    )
    assert not replay.spinio.exchange_Jdict and not replay.spinio.Jani_dict


def test_compress_sparse_endpoint_constraint_projects_without_reverse_creation():
    atoms = Atoms(
        "Fe",
        cell=[[3, 0, 0], [0.6, 3.7, 0], [0.4, 0.8, 4.1]],
        scaled_positions=[(0, 0, 0)],
        pbc=True,
    )
    key = ((1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=None,
        charges=None,
        colinear=False,
        index_spin=[0],
        dmi_ddict={key: np.array([0.001, 0.0, 0.0])},
    )
    with pytest.raises(ValueError, match="shell .*DMI.*deviation"):
        shells.compress(source)
    result = shells.compress(source, on_inconsistent="project")
    assert len(result.document.operations) == 2
    assert result.document.shells[0].DMI_members == (1,)
    replay = shells._expand_core(result.document)
    assert set(replay.spinio.dmi_ddict) == {key}
    np.testing.assert_array_equal(replay.spinio.dmi_ddict[key], [0.0, 0.0, 0.0])
    assert result.report.created_pair_keys == ()
