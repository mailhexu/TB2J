"""Absolute J/K/X references through the real reciprocal-space consumer."""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from scipy.special import expit
from test_cfr2_exchange import DOWN, ORIGIN, UP

from TB2J.exchange_qspace import ExchangeCLQspace


def qspace_calculator(method="cfr2", nz=80, nk=1):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = tuple(
        LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([np.eye(2)]),
            Rlist=np.array([ORIGIN]),
            nbasis=2,
            atoms=atoms,
            nspin=1,
            orth=True,
        )
        for h in (UP, DOWN)
    )
    return ExchangeCLQspace(
        tbmodels=models,
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|d", "Fe2|d"],
        kmesh=[nk, 1, 1],
        efermi=0,
        nz=nz,
        nproc=1,
        Rcut=2,
        method=method,
    )


def spectral_jkx(temperature=600):
    """Complete spectral pair sum, not samples of the implementation contour.

    MathDoc: tb2j/cfr2-integration-contract verifies the divided difference.
    This intentionally preserves the existing K1+2*K2 kernel, not a physical
    redefinition of K or a validation of Bruno renormalization.
    """
    eu, vu = np.linalg.eigh(UP)
    ed, vd = np.linalg.eigh(DOWN)
    fu = expit(-eu / (kB * temperature)) if temperature else (eu < 0).astype(float)
    fd = expit(-ed / (kB * temperature)) if temperature else (ed < 0).astype(float)
    divided = (fu[:, None] - fd[None, :]) / (eu[:, None] - ed[None, :])
    delta = np.diag(UP - DOWN)
    j, k, x = (np.empty((1, 2, 2)) for _ in range(3))
    for i in range(2):
        for l in range(2):
            residue = np.outer(vu[i] * vu[l].conj(), vd[l] * vd[i].conj())
            reverse = np.outer(vu[l] * vu[i].conj(), vd[i] * vd[l].conj())
            ud = -1j * np.pi * np.sum(residue * divided)
            du = -1j * np.pi * np.sum(reverse * divided)
            j[0, i, l] = ((ud + du) * delta[i] * delta[l] / (8 * np.pi)).imag
            k[0, i, l] = (-(ud + 2 * du) * delta[l] / (2 * np.pi)).imag
            x[0, i, l] = (2 * ud / np.pi).imag
    return j, k, x


@pytest.mark.parametrize("nk", [1, 3])
def test_actual_qspace_cfr2_against_all_spectral_observables(nk):
    calc = qspace_calculator(nk=nk)
    calc.calculate_all()
    for actual, expected in zip((calc.Jq, calc.Kq, calc.Xq), spectral_jkx()):
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape), atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("rule", ["simpson", "trapezoidal"])
def test_geometric_qspace_controls_against_zero_temperature_spectrum(rule):
    calc = qspace_calculator(method="semicircle", nz=4097)
    calc._prepare()
    calc.get_all_A()
    calc.integrate(method=rule)
    # The retained geometric contour is zero-temperature, not Fermi weighted.
    # Smooth gapped two-level products: at4097 nodes O(h^2) error is <2e-6.
    for actual, expected in zip((calc.Jq, calc.Kq, calc.Xq), spectral_jkx(0)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
