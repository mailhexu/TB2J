"""GPAW spinor CLI + bcc Fe SOC fixture tests (story 006)."""

import numpy as np
import pytest

gpaw = pytest.importorskip("gpaw")


def _fe_bcc(soc):
    """bcc Fe fixture (Ni PAW+PW SCs misbehave in this GPAW version)."""
    from ase import Atoms
    from gpaw import GPAW, PW, FermiDirac

    a = 2.42
    atoms = Atoms(
        "Fe",
        positions=[[0, 0, 0]],
        cell=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        pbc=True,
    )
    calc = GPAW(
        mode=PW(400),
        xc="LDA",
        kpts=(3, 3, 3),
        symmetry="off",
        magmoms=[[0, 0, 2.5]],
        soc=soc,
        txt=None,
        occupations=FermiDirac(0.05),
        convergence={"energy": 1e-3},
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc


@pytest.fixture(scope="module")
def ni_soc_nc(tmp_path_factory):
    from TB2J.interfaces.gpaw_spinor_projector import (
        save_gpaw_spinor_projector_netcdf,
    )

    tmp = tmp_path_factory.mktemp("ni_soc")
    calc = _fe_bcc(soc=True)
    path = tmp / "ni_soc.nc"
    save_gpaw_spinor_projector_netcdf(calc, path)
    return path


def test_cli_spinor_netcdf_exchange(ni_soc_nc, tmp_path):
    from TB2J.interfaces.gpaw_projector import gen_exchange_projector_netcdf

    out, jdict = gen_exchange_projector_netcdf(
        str(ni_soc_nc), output_path=str(tmp_path / "TB2J_results"), Rcut=5.0, nz=20
    )
    assert out.exists()
    text = out.read_text()
    assert "Combined J tensor" in text
    assert "DMI" in text
    # bcc Ni with inversion symmetry: DMI must vanish within tolerance
    js = [v for k, v in jdict.items() if np.linalg.norm(np.asarray(k[0])) > 0]
    assert js, "no offsite exchange entries"
    first_shell = [v for v in js if v > 0]
    assert first_shell, "FM nearest-neighbour exchange expected for bcc Fe"
    content = (tmp_path / "TB2J_results").exists()
    assert content


def test_dmi_symmetry_consistency(ni_soc_nc, tmp_path):
    """DMI vectors must vanish for inversion-symmetric bcc Fe pairs."""
    from TB2J.interfaces.gpaw_spinor_projector import (
        compute_spinor_projector_exchange,
    )
    from TB2J.projector_green import ProjectorGreenData

    data = ProjectorGreenData.load_netcdf(ni_soc_nc)
    Rpts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=int,
    )
    exchange = compute_spinor_projector_exchange(data, Rpts=Rpts, nz=16)
    for key, entry in exchange.items():
        R = np.asarray(key[0])
        if np.linalg.norm(R) < 0.5:
            continue
        dmi = np.asarray(entry["dmi"])
        assert np.linalg.norm(dmi) < 1e-6, f"nonzero DMI for {key}: {dmi}"
