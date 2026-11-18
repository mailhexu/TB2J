import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from TB2J.exchange_dmft import ExchangeCLDMFT


class MockBasis:
    def __init__(self, iatom, sym):
        self.iatom = iatom
        self.sym = sym


class TestExchangeCLDMFT(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.mesh = (None, [1j, 3j])  # Simple Matsubara mesh
        self.mock_model.Rlist = np.array([[0, 0, 0]])
        self.mock_model.SR = np.array([np.eye(5)])

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
        self.MockTBGreen.return_value.norb = 5
        self.MockTBGreen.return_value.nbasis = 5
        self.MockTBGreen.return_value.efermi = 0.0
        self.MockTBGreen.return_value.adjusted_emin = -10

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        exchange = ExchangeCLDMFT(
            tbmodels=self.mock_model,
            atoms=self.mock_atoms,
            basis=[
                MockBasis(0, "d1"),
                MockBasis(0, "d2"),
                MockBasis(0, "d3"),
                MockBasis(0, "d4"),
                MockBasis(0, "d5"),
            ],
            efermi=0.0,
            emin=-10,
            kmesh=[1, 1, 1],
            magnetic_elements=["Fe"],
            method="matsubara",
        )
        self.assertTrue(exchange._is_collinear)
        self.assertEqual(exchange.backend_name, "DMFT")
        self.assertEqual(exchange.contour, [1j, 3j])

    def test_temperature_calculation(self):
        # mesh points: 1j, 3j.
        # iw0 = 1j.
        # beta = pi/1 = pi
        # T = 1/pi
        exchange = ExchangeCLDMFT(
            tbmodels=self.mock_model,
            atoms=self.mock_atoms,
            basis=[
                MockBasis(0, "d1"),
                MockBasis(0, "d2"),
                MockBasis(0, "d3"),
                MockBasis(0, "d4"),
                MockBasis(0, "d5"),
            ],
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
