import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange.edit import make_supercell, open_boundary, save
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict
from TB2J.magnon.magnon3 import Magnon


def _chain_spinio():
    atoms = Atoms("Fe", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
    # both bond directions, like real TB2J pickles
    terms = {
        ((1, 0, 0), 0, 0): 2.0,
        ((2, 0, 0), 0, 0): 0.5,
        ((-1, 0, 0), 0, 0): 2.0,
        ((-2, 0, 0), 0, 0): 0.5,
    }
    return SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 3.0]]),
        charges=np.array([6.0]),
        index_spin=[0],
        distance_dict=gen_distance_dict([0], atoms, sorted({R for (R, _, _) in terms})),
        exchange_Jdict=terms,
        dmi_ddict=None,
        Jani_dict=None,
        sia_tensor=None,
    )


def _magnon(result_dir):
    magnon = Magnon.from_TB2J_results(path=str(result_dir), Jiso=True)
    magnon.set_reference(
        Q=(0, 0, 0),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([1.0, 0.0, 0.0]),
        magmoms=np.asarray(magnon.magmom, dtype=float),
    )
    return magnon


def _bands(spinio, kpoints, tmp_path, name):
    out = tmp_path / name
    save(spinio, out)
    magnon = _magnon(out)
    _, bands, _ = magnon.get_magnon_bands(kpoints=np.array(kpoints))
    return bands


@pytest.mark.parametrize("q", [0.0, 0.25])
def test_replicated_bands_contain_primitive_bands(tmp_path, q):
    # the failing oracle of the open-boundary record: with full Bloch phases
    # the 6x1x1 supercell band set at k_sc = 6q mod 1 contains the primitive
    # band at q (both sides carry the same global scale convention)
    prim_bands = _bands(_chain_spinio(), [[q, 0, 0]], tmp_path, "prim")
    sc_bands = _bands(
        make_supercell(_chain_spinio(), [6, 1, 1]),
        [[(6 * q) % 1.0, 0, 0]],
        tmp_path,
        "sc6",
    )
    assert np.any(np.isclose(sc_bands[0], prim_bands[0][0], atol=1e-8))


def test_replicated_chain_has_goldstone_mode(tmp_path):
    sc_bands = _bands(
        make_supercell(_chain_spinio(), [6, 1, 1]),
        [[0.0, 0, 0], [0.3, 0, 0], [0.5, 0, 0]],
        tmp_path,
        "sc6",
    )
    assert sc_bands[0, 0] == pytest.approx(0.0, abs=1e-8)
    # and the bands disperse (they were flat with the k.R-only transform)
    assert np.abs(sc_bands[2] - sc_bands[0]).max() > 0.1


def test_two_site_chain_jq_matches_exact_transform(tmp_path):
    atoms = Atoms("FeFe", cell=np.diag([1.0, 1, 1]), scaled_positions=[[0, 0, 0], [0.5, 0, 0]], pbc=True)
    terms = {
        ((0, 0, 0), 0, 1): 1.0,
        ((-1, 0, 0), 0, 1): 1.0,
        ((0, 0, 0), 1, 0): 1.0,
        ((1, 0, 0), 1, 0): 1.0,
    }
    spinio = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]),
        charges=np.array([6.0, 6.0]),
        index_spin=[0, 1],
        distance_dict=gen_distance_dict([0, 1], atoms, sorted({R for (R, _, _) in terms})),
        exchange_Jdict=terms,
        dmi_ddict=None,
        Jani_dict=None,
        sia_tensor=None,
    )
    out = tmp_path / "two_site"
    save(spinio, out)
    magnon = _magnon(out)
    # J01(k) with full phases = e^{2 pi i k/2} + e^{-2 pi i k/2} = 2 cos(pi k)
    # (the loader stores each bond once per direction, and Jq's Snorm^-2
    # rescale divides both sides equally; compare the shape against the
    # exact transform, not the old k.R-only form 1 + e^{-2 pi i k})
    ks = np.array([[0.1, 0, 0], [0.25, 0, 0], [0.4, 0, 0]])
    Jq = magnon.Jq(ks)
    ratios = []
    for iq, k in enumerate(ks):
        magnon3_01 = Jq[iq][0, 1, 0, 0]
        exact = 2 * np.cos(np.pi * k[0])
        ratios.append(magnon3_01 / exact)
    assert np.allclose(ratios, ratios[0], atol=1e-10)


def test_cut_ribbon_bands_finite(tmp_path):
    sc = make_supercell(_chain_spinio(), [6, 1, 1])
    open_boundary(sc, ["a"])
    bands = _bands(sc, [[0.0, 0, 0], [0.5, 0, 0]], tmp_path, "ribbon")
    assert np.all(np.isfinite(bands))
    # open boundary: no dispersion along a (bands at k_a = 0 and 0.5 equal)
    assert np.allclose(bands[0], bands[1], atol=1e-10)
