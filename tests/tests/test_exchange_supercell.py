import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange.edit import make_supercell, save
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict
from TB2J.scripts.TB2J_edit import main as tb2j_edit_main


def _simple_spinio():
    atoms = Atoms(
        "FeO",
        cell=np.eye(3),
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    spinio = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        charges=np.array([6.0, -2.0]),
        index_spin=[0, -1],
        distance_dict=gen_distance_dict([0], atoms, [(0, 0, 0), (1, 0, 0)]),
        exchange_Jdict={((0, 0, 0), 0, 0): 1.0, ((1, 0, 0), 0, 0): 2.0},
        dmi_ddict={((1, 0, 0), 0, 0): np.array([0.1, 0.2, 0.3])},
        Jani_dict={((0, 0, 0), 0, 0): np.eye(3)},
        sia_tensor={0: np.diag([0.1, 0.2, 0.3])},
    )
    spinio.k1 = [0.4]
    spinio.k1dir = [np.array([0.0, 0.0, 1.0])]
    spinio.has_uniaxial_anistropy = True
    return spinio


def test_make_supercell_keeps_atom_and_spin_indices_separate():
    spinio = _simple_spinio()
    original_index_spin = list(spinio.index_spin)
    original_exchange = dict(spinio.exchange_Jdict)

    sc_spinio = make_supercell(spinio, [2, 1, 1])

    assert len(sc_spinio.atoms) == 4
    assert list(sc_spinio.index_spin) == [0, -1, 1, -1]
    assert np.allclose(sc_spinio.charges, [6.0, -2.0, 6.0, -2.0])
    assert np.allclose(sc_spinio.spinat[:, 2], [2.0, 0.0, 2.0, 0.0])
    assert list(spinio.index_spin) == original_index_spin
    assert spinio.exchange_Jdict == original_exchange

    assert set(sc_spinio.sia_tensor) == {0, 1}
    assert np.allclose(sc_spinio.sia_tensor[0], spinio.sia_tensor[0])
    assert np.allclose(sc_spinio.sia_tensor[1], spinio.sia_tensor[0])
    assert sc_spinio.k1 == [0.4, 0.4]
    assert np.allclose(sc_spinio.k1dir, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def test_make_supercell_maps_pair_dictionaries_and_distances():
    sc_spinio = make_supercell(_simple_spinio(), [2, 1, 1])

    assert sc_spinio.exchange_Jdict == {
        ((0, 0, 0), 0, 0): 1.0,
        ((0, 0, 0), 0, 1): 2.0,
        ((0, 0, 0), 1, 1): 1.0,
        ((1, 0, 0), 1, 0): 2.0,
    }
    assert set(sc_spinio.dmi_ddict) == {((0, 0, 0), 0, 1), ((1, 0, 0), 1, 0)}
    for value in sc_spinio.dmi_ddict.values():
        assert np.allclose(value, [0.1, 0.2, 0.3])
    assert set(sc_spinio.Jani_dict) == {((0, 0, 0), 0, 0), ((0, 0, 0), 1, 1)}
    for value in sc_spinio.Jani_dict.values():
        assert np.allclose(value, np.eye(3))
    assert set(sc_spinio.distance_dict) == (
        set(sc_spinio.exchange_Jdict)
        | set(sc_spinio.dmi_ddict)
        | set(sc_spinio.Jani_dict)
    )
    for vec, distance in sc_spinio.distance_dict.values():
        assert vec.shape == (3,)
        assert distance == pytest.approx(np.linalg.norm(vec))


def test_make_supercell_repeats_distinct_multi_spin_anisotropy():
    atoms = Atoms(
        "Fe2O",
        cell=np.eye(3),
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    spinio = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -3.0], [0.0, 0.0, 0.0]]),
        charges=np.array([6.0, 7.0, -2.0]),
        index_spin=[0, 1, -1],
        sia_tensor={0: np.eye(3), 1: np.eye(3) * 2.0},
    )
    spinio.k1 = [0.1, 0.2]
    spinio.k1dir = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    spinio.has_uniaxial_anistropy = True

    sc_spinio = make_supercell(spinio, [2, 1, 1])

    assert list(sc_spinio.index_spin) == [0, 1, -1, 2, 3, -1]
    assert set(sc_spinio.sia_tensor) == {0, 1, 2, 3}
    assert np.allclose(sc_spinio.sia_tensor[0], np.eye(3))
    assert np.allclose(sc_spinio.sia_tensor[1], np.eye(3) * 2.0)
    assert np.allclose(sc_spinio.sia_tensor[2], np.eye(3))
    assert np.allclose(sc_spinio.sia_tensor[3], np.eye(3) * 2.0)
    assert sc_spinio.k1 == [0.1, 0.2, 0.1, 0.2]
    assert np.allclose(
        sc_spinio.k1dir,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )


def test_srmno3_style_validation_maps_only_magnetic_mn():
    atoms = Atoms(
        "SrMnO3",
        cell=np.eye(3) * 3.8,
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ],
        pbc=True,
    )
    spinio = SpinIO(
        atoms=atoms,
        spinat=np.array([[0, 0, x] for x in [0, 3, 0, 0, 0]], dtype=float),
        charges=np.array([2, 4, 5, 5, 5], dtype=float),
        index_spin=[-1, 0, -1, -1, -1],
        exchange_Jdict={((0, 0, 0), 0, 0): -1.0, ((0, 0, 1), 0, 0): -0.5},
        sia_tensor={0: np.eye(3) * 0.01},
    )

    sc_spinio = make_supercell(spinio, [2, 1, 1])

    assert len(sc_spinio.atoms) == 10
    assert list(sc_spinio.index_spin) == [-1, 0, -1, -1, -1, -1, 1, -1, -1, -1]
    assert set(sc_spinio.sia_tensor) == {0, 1}
    assert all(
        idx < 0 for idx in np.array(sc_spinio.index_spin)[[0, 2, 3, 4, 5, 7, 8, 9]]
    )


def test_tb2j_edit_supercell_cli_writes_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    save(_simple_spinio(), input_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "TB2J_edit",
            "supercell",
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "--matrix",
            "2",
            "1",
            "1",
        ],
    )
    tb2j_edit_main()

    assert (output_dir / "TB2J.pickle").is_file()
    assert (output_dir / "exchange.out").is_file()


def test_tb2j_edit_supercell_cli_rejects_bad_matrix(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    save(_simple_spinio(), input_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "TB2J_edit",
            "supercell",
            "-i",
            str(input_dir),
            "-o",
            str(tmp_path / "output"),
            "--matrix",
            "2",
            "1",
        ],
    )

    with pytest.raises(SystemExit):
        tb2j_edit_main()
