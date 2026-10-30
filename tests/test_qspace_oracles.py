"""Independent zero-temperature spectral and finite-grid exchange checks."""

import numpy as np
from ase import Atoms
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian

from TB2J.exchange_qspace import ExchangeCLQspace


def test_dispersive_insulator_against_spectral_and_real_space_oracles():
    # One orthogonal orbital per site, eV; up occupied, down empty at mu=0.
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[1, 10, 10])
    r = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])

    def model(onsite):
        return LCAOHamiltonian(
            HR=np.array([[[-0.5]], [[onsite]], [[-0.5]]]),
            SR=np.array([[[0]], [[1]], [[0]]]),
            Rlist=r, nbasis=1, atoms=atoms, nspin=1, orth=True,
        )

    calc = ExchangeCLQspace(
        tbmodels=(model(-2), model(2)), atoms=atoms, magnetic_elements=["Fe"],
        basis=["Fe1|d"], kmesh=[5, 1, 1], efermi=0, nz=1025,
        nproc=1, Rcut=3, method="semicircle",
    )
    # MathDoc: tb2j/qspace-unrenormalized-kernel
    # Independent occupied-pole spectral sum, including the onsite response.
    k = np.arange(-2, 3) / 5
    up = -2 - np.cos(2 * np.pi * k)
    expected = np.array([
        -4 * np.mean(1 / (up - (2 - np.cos(2 * np.pi * (k + q)))))
        for q in k
    ])
    raw = calc.calculate_raw()[:, 0, 0]
    np.testing.assert_allclose(calc.qpts[:, 0], k, atol=1e-15, rtol=0)
    # Observed maximum errors: 6.1e-9, 3.5e-10, 2.1e-11 eV at 257, 513, 1025 points.
    np.testing.assert_allclose(raw, expected, atol=1e-10, rtol=0)
    assert np.ptp(expected) > 0.1

    # MathDoc: tb2j/qspace-finite-grid-transform
    # Inverse transform of the independent spectral values, retaining R=0.
    expected_r = np.array([
        np.mean(expected * np.cos(2 * np.pi * k * rv[0])) for rv in calc.Rlist
    ])
    np.testing.assert_allclose(calc.q_to_r()[:, 0, 0], expected_r, atol=1e-10, rtol=0)
    assert expected_r[np.flatnonzero(np.all(np.asarray(calc.Rlist) == 0, axis=1))[0]] > 1
