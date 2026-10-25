"""Sparse native exchanges survive real pickle persistence without invented keys."""

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO


def _model(**channels):
    return SpinIO(
        atoms=Atoms("FeOFe", cell=np.eye(3) * 4, pbc=True),
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, -3.0]]),
        charges=np.array([6.0, -2.0, 6.0]),
        index_spin=[1, -1, 0],
        **channels,
    )


@pytest.mark.parametrize(
    "channel,value",
    [
        ("dmi_ddict", np.array([0.125, -0.25, 0.5])),
        ("Jani_dict", np.diag([0.125, -0.25, 0.125])),
    ],
)
def test_channel_only_pickle_retains_support_and_declared_spins(
    tmp_path, channel, value
):
    key = ((2, -1, 0), 0, 1)
    model = _model(exchange_Jdict={}, **{channel: {key: value}})
    model.write_pickle(path=tmp_path)
    loaded = SpinIO.load_pickle(path=tmp_path)
    assert loaded.exchange_Jdict == {}
    assert set(getattr(loaded, channel)) == {key}
    np.testing.assert_array_equal(getattr(loaded, channel)[key], value)
    assert set(loaded.Rlist) == {(0, 0, 0), (2, -1, 0)}
    assert loaded.ispin_list == [0, 1]
    assert loaded.nspin == 2
    assert loaded.index_spin == [1, -1, 0]
    assert loaded.iatom(0) == 2 and loaded.iatom(1) == 0


@pytest.mark.parametrize("exchange", [None, {}])
def test_empty_pairs_keep_isolated_declared_spins(tmp_path, exchange):
    model = _model(exchange_Jdict=exchange)
    model.write_pickle(path=tmp_path)
    loaded = SpinIO.load_pickle(path=tmp_path)
    assert loaded.exchange_Jdict == exchange
    assert loaded.dmi_ddict is None and loaded.Jani_dict is None
    assert loaded.ispin_list == [0, 1]
    assert loaded.nspin == 2
    assert set(loaded.Rlist) == {(0, 0, 0)}


@pytest.mark.parametrize("channel", ["exchange_Jdict", "dmi_ddict", "Jani_dict"])
def test_invalid_endpoint_reports_channel_key_and_index(tmp_path, channel):
    key = ((3, 0, 0), 0, 5)
    value = {"exchange_Jdict": 0.1, "dmi_ddict": np.ones(3), "Jani_dict": np.eye(3)}[
        channel
    ]
    model = _model(**{channel: {key: value}})
    model.write_pickle(path=tmp_path)
    with pytest.raises(ValueError) as error:
        SpinIO.load_pickle(path=tmp_path)
    message = str(error.value)
    assert channel in message and repr(key) in message and "5" in message


def test_full_channel_roundtrip_keeps_extra_translations_and_exact_values(tmp_path):
    channels = {
        "exchange_Jdict": {((0, 0, 0), 0, 1): 0.125},
        "dmi_ddict": {((1, 0, 0), 1, 0): np.array([0.25, -0.5, 0.75])},
        "Jani_dict": {
            ((0, -2, 0), 0, 1): np.array(
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, -5.0]]
            )
        },
    }
    _model(**channels).write_pickle(path=tmp_path / "first")
    first = SpinIO.load_pickle(path=tmp_path / "first")
    first.write_pickle(path=tmp_path / "second")
    second = SpinIO.load_pickle(path=tmp_path / "second")
    assert first.Rlist == second.Rlist
    assert set(second.Rlist) == {(0, 0, 0), (1, 0, 0), (0, -2, 0)}
    for channel, original in channels.items():
        for loaded in (first, second):
            actual = getattr(loaded, channel)
            assert actual.keys() == original.keys()
            for key, value in original.items():
                np.testing.assert_array_equal(actual[key], value)


def test_nonmagnetic_empty_model_has_no_spin_indices(tmp_path):
    model = SpinIO(
        atoms=Atoms("O", cell=np.eye(3), pbc=True),
        spinat=None,
        charges=None,
        index_spin=[-1],
        exchange_Jdict={},
    )
    model.write_pickle(path=tmp_path)
    loaded = SpinIO.load_pickle(path=tmp_path)
    assert loaded.nspin == 0
    assert loaded.ispin_list == []
    assert loaded.exchange_Jdict == {}


@pytest.mark.parametrize(
    "channel,value",
    [
        ("dmi_ddict", np.array([0.125, -0.25, 0.5])),
        ("Jani_dict", np.diag([0.125, -0.25, 0.125])),
    ],
)
def test_sparse_loaded_model_keeps_derived_lists_through_supercell(
    tmp_path, channel, value
):
    from TB2J.io_exchange.edit import make_supercell

    model = SpinIO(
        atoms=Atoms("Fe", cell=np.eye(3) * 3, pbc=True),
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=[6.0],
        index_spin=[0],
        exchange_Jdict=None,
        **{channel: {((1, 0, 0), 0, 0): value}},
    )
    model.write_pickle(path=tmp_path / "primitive")
    loaded = SpinIO.load_pickle(path=tmp_path / "primitive")
    supercell = make_supercell(loaded, [2, 1, 1])
    assert supercell.nspin == 2
    assert supercell.ispin_list == [0, 1]
    assert set(supercell.Rlist) == {(0, 0, 0), (1, 0, 0)}
    assert supercell.exchange_Jdict is None
    expected = {((0, 0, 0), 0, 1), ((1, 0, 0), 1, 0)}
    actual = getattr(supercell, channel)
    assert actual.keys() == expected
    for pair in expected:
        np.testing.assert_array_equal(actual[pair], value)
    supercell.write_pickle(path=tmp_path / "supercell")
    reloaded = SpinIO.load_pickle(path=tmp_path / "supercell")
    assert reloaded.nspin == supercell.nspin
    assert reloaded.Rlist == supercell.Rlist
    assert getattr(reloaded, channel).keys() == expected
