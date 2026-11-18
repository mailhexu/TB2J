import numpy as np

from TB2J.dmft_model import TBModelDMFT


class _StaticModel:
    nbasis = 2
    is_orthogonal = True
    R2kfactor = 1.0
    norb = 2
    atoms = None
    nel = 0.0
    efermi = 0.0
    colinear = True


class _Parser:
    orbital_map = {}

    def __init__(self):
        self.mesh = np.array([1j], dtype=complex)
        self.sigma = np.zeros((2, 1, 2, 2), dtype=complex)
        self.sigma[0, 0, 0, 0] = 3.0
        self.sigma[1, 0, 0, 0] = 4.0
        self.sigma_static = np.zeros((2, 2, 2), dtype=complex)
        self.sigma_static[0, 0, 0] = 1.0
        self.sigma_static[1, 0, 0] = 2.0

    def get_chemical_potential(self):
        return 0.0

    def read_self_energy(self):
        return self.sigma.copy(), self.mesh.copy()

    def get_static_sigma(self):
        return self.sigma_static.copy(), None


def test_dynamic_siginp_static_header_is_added_to_residual_for_dyson():
    model = TBModelDMFT(_StaticModel(), _Parser())

    sigma = model.get_sigma(1j, ispin=None)

    assert sigma[0, 0, 0] == 4.0
    assert sigma[1, 0, 0] == 6.0
