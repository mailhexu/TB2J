import unittest
import numpy as np
from TB2J.myTB2 import MyTB
from TB2J.pauli import *  #pauli_block
from TB2J.green_SOC import TBGreen
from TB2J.utils import auto_assign_basis_name
from TB2J.exchange import ExchangeNCL, gen_exchange


class AssignTest(unittest.TestCase):
    def test_assign(self):
        pass


class PauliTest(unittest.TestCase):
    def test_pauli(self):
        M = np.array([[1 - 3j, 9 + 2j], [3 - 9j, -4]])
        #print(np.sum(M * s1T) * s1)
        M0, Mx, My, Mz = pauli_decomp(M)
        #print(M0, Mx, My, Mz)
        #print(sum([M0 * s0, Mx * s1, My * s2, Mz * s3]))

        MI = (pauli_block_I(M, 1))
        Mx = (pauli_block_x(M, 1))
        My = (pauli_block_y(M, 1))
        Mz = (pauli_block_z(M, 1))

        n = 8
        hn = n // 2
        M = np.random.random([n, n]) + 0.0j
        MI = (pauli_block_I(M, hn))
        Mx = (pauli_block_x(M, hn))
        My = (pauli_block_y(M, hn))
        Mz = (pauli_block_z(M, hn))


class TBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = '/media/hexu/Backup/materials/SOC/RuCl3_FM_SOC/'
        #m = MyTB.read_from_wannier_dir(
        #    path=self.path, prefix='wannier90', poscar='POSCAR', nls=True)
        #m.save('RuCl3.nc')
        cls.tbmodel = MyTB.load_MyTB('RuCl3.nc')
        cls.positions = cls.tbmodel.xred
        cls.atoms = cls.tbmodel.atoms

    def test_read_wannier(self):
        m = MyTB.read_from_wannier_dir(
            path=self.path, prefix='wannier90', poscar='POSCAR', nls=True)
        m.save('RuCl3.nc')
        pass

    def test_saveload_wannier(self):
        # test onsite energy
        Eonsite = self.tbmodel.onsite_energies
        self.assertEqual(first=Eonsite[0], second=-5.028565)
        self.assertAlmostEqual(Eonsite[-1], -6.9492, delta=0.0001)

        # test R=0
        ER0 = self.tbmodel.ham_R0
        self.assertAlmostEqual(ER0[0, 0], -5.028565, delta=0.0001)
        self.assertAlmostEqual(ER0[-1, -1], -6.9492, delta=0.0001)

        # test positive R
        ER100 = self.tbmodel.get_hamR(R=(1, 0, 0))
        self.assertAlmostEqual(ER100[0, 0], -0.000740, delta=0.0001)
        self.assertAlmostEqual(ER100[1, 0], -0.003922, delta=0.0001)

        # test negative R
        ERm100 = self.tbmodel.get_hamR(R=(-1, 0, 0))
        self.assertAlmostEqual(ERm100[0, 0], -0.000740, delta=0.0001)
        self.assertAlmostEqual(ERm100[1, 0], -0.002838, delta=0.0001)

    def test_assign_basis(self):
        basis_dict, shifted_pos = auto_assign_basis_name(
            positions=self.tbmodel.xred, atoms=self.atoms)
        # Note that it starts from 1.
        self.assertEqual(basis_dict['Ru1|orb_1'], 1)
        self.assertEqual(basis_dict['Ru1|orb_6'], 29)

    def test_Green(self):
        G = TBGreen(
            tbmodel=self.tbmodel,
            kmesh=[4, 4, 4],
            efermi=3.0,
        )
        G00 = G.get_Gk(ik=0, energy=0.0)
        self.assertAlmostEqual(G00[0, 0], 1.30602e-1, delta=1e-4)
        GR0 = G.get_GR(Rpts=[(0, 0, 0)], energy=0.0)
        self.assertAlmostEqual(GR0[(0, 0, 0)][0, 0], 0.130001329, delta=1e-4)


class ExchangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = '/media/hexu/Backup/materials/SOC/RuCl3_FM_SOC/'
        cls.exchange = gen_exchange(
            kmesh=[6, 6, 3],
            Rmesh=[1,1,1],
            nz1=50,
            nz3=50,
            nz2=300,
            path=path,
            efermi=-3.25,
            magnetic_elements=['Ru'])

    def test_orb_dict(self):
        print(self.exchange.orb_dict)

    def test_runall(self):
        self.exchange.calculate_all()
        print(self.exchange.Jtensor)
        print(self.exchange.DMI)

    def test_write_txt(self):
        self.exchange.write_output()


class IOTest(unittest.TestCase):
    def test_xml(self):
        pass

    def test_txt(self):
        pass


unittest.main()
