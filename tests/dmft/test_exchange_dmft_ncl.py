import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from TB2J.exchange_dmft import ExchangeDMFTNCL


class MockBasis:
    def __init__(self, iatom, sym):
        self.iatom = iatom
        self.sym = sym


class TestExchangeDMFTNCL(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.mesh = (None, [1j, 3j])  # Simple Matsubara mesh
        self.mock_model.Rlist = np.array([[0, 0, 0]])
        self.mock_model.SR = np.array([np.eye(10)])  # Spinor model (5 orbs * 2)

        # Mock atoms object
        self.mock_atoms = MagicMock()
        self.mock_atoms.get_chemical_symbols.return_value = ["Fe"]
        self.mock_atoms.get_tags.return_value = [0]
        self.mock_atoms.get_positions.return_value = [np.array([0, 0, 0])]
        self.mock_atoms.get_cell.return_value = np.eye(3)
        self.mock_atoms.__len__ = MagicMock(return_value=1)

        # Patch TBGreen to avoid actual heavy computation
        self.patcher = patch("TB2J.exchange_dmft.TBGreen")
        self.MockTBGreen = self.patcher.start()
        self.MockTBGreen.return_value.get_density_matrix.return_value = np.zeros((1, 1))
        self.MockTBGreen.return_value.H0 = np.zeros((1, 1))
        self.MockTBGreen.return_value.norb = 10
        self.MockTBGreen.return_value.nbasis = 5
        self.MockTBGreen.return_value.efermi = 0.0
        self.MockTBGreen.return_value.adjusted_emin = -10

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        # 5 orbitals * 2 spins = 10 basis functions
        # For NCL, each "basis" element in TB2J usually represents a spatial orbital,
        # and TB2J handles the spin doubling.
        # However, the error message indicates it sees 5 "spin-orbitals".
        # Let's provide 10 basis functions to simulate 5 orbitals * 2 spins.
        basis = []
        for i in range(10):
            basis.append(MockBasis(0, f"orb{i}"))

        exchange = ExchangeDMFTNCL(
            tbmodels=self.mock_model,
            atoms=self.mock_atoms,
            basis=basis,
            efermi=0.0,
            emin=-10,
            kmesh=[1, 1, 1],
            magnetic_elements=["Fe"],
            method="matsubara",
        )
        self.assertFalse(exchange._is_collinear)
        self.assertEqual(exchange.backend_name, "DMFT")
        self.assertEqual(exchange.contour, [1j, 3j])

    def test_temperature_calculation(self):
        basis = []
        for i in range(10):
            basis.append(MockBasis(0, f"orb{i}"))

        exchange = ExchangeDMFTNCL(
            tbmodels=self.mock_model,
            atoms=self.mock_atoms,
            basis=basis,
            efermi=0.0,
            emin=-10,
            kmesh=[1, 1, 1],
            magnetic_elements=["Fe"],
            method="matsubara",
        )
        expected_beta = np.pi
        self.assertAlmostEqual(exchange.beta, expected_beta)
        self.assertAlmostEqual(exchange.temperature, 1.0 / expected_beta)


if __name__ == "__main__":
    unittest.main()
