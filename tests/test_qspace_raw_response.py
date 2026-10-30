"""Raw q-space response through the public library entry point."""

import numpy as np
import pytest
from ase import Atoms
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian

from TB2J.exchange_qspace import ExchangeCLQspace


def calculator(method, matrices=None, nz=None):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = []
    matrices = matrices or (
        [[-0.3, 0.12], [0.12, -0.25]],
        [[0.3, 0.12], [0.12, 0.25]],
    )
    for h in matrices:
        models.append(
            LCAOHamiltonian(
                HR=np.array([h]),
                SR=np.array([np.eye(2)]),
                Rlist=np.zeros((1, 3), int),
                nbasis=2,
                atoms=atoms,
                nspin=1,
                orth=True,
            )
        )
    return ExchangeCLQspace(
        tbmodels=tuple(models),
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|d", "Fe2|d"],
        kmesh=[1, 1, 1],
        efermi=0,
        nz=nz or (4097 if method == "semicircle" else 8),
        nproc=1,
        Rcut=2,
        method=method,
    )


def test_raw_exchange_matches_zero_temperature_spectral_reference():
    raw = calculator("semicircle").calculate_raw()
    # Independent occupied-band spectral sum at zero temperature, in eV.
    np.testing.assert_allclose(raw[0, 0, 1], -0.016201620162016202, atol=2e-10, rtol=0)
    np.testing.assert_allclose(raw[0, 0, 0], 0.1662016201620161, atol=2e-10, rtol=0)


def test_raw_exchange_does_not_require_well_conditioned_bruno_matrix():
    h = [[0.0, 0.5], [0.5, 0.0]]
    calc = calculator("semicircle", matrices=(h, h), nz=64)
    raw = calc.calculate_raw()
    np.testing.assert_allclose(raw, 0.0, atol=1e-14, rtol=0)
    assert np.linalg.cond(calc.Xq[0]) > 1e12


def test_raw_exchange_rejects_a_pole_on_the_contour():
    h = [[-0.5, 0.5], [0.5, -0.5]]
    calc = calculator("semicircle", matrices=(h, h), nz=64)
    with pytest.raises(ValueError, match="non-finite"):
        calc.calculate_raw()


@pytest.mark.parametrize("method", ["cfr", "cfr2"])
def test_raw_exchange_rejects_pole_contours_without_compatible_integration(method):
    with pytest.raises(ValueError, match="requires a geometric contour"):
        calculator(method).calculate_raw()
