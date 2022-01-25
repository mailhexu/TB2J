"""
FDTB: Finite difference tight binding.
"""
import bz2
import pickle
import numpy as np
from TB2J.myTB import MyTB, merge_tbmodels_spin
from ase.io import read
from supercellmap import SupercellMaker


def FD12(a0, a1, a2, amp):
    """
    Finite difference of 1st and 2nd order derivative.
    """
    d1 = (a2 - a0) / (2 * amp)
    d2 = (a2 + a0 - 2 * a1) / amp**2
    return d1, d2


def FD1(a0, a1, amp):
    return (a1 - a0) / amp


class dHdx():
    """
    Finite difference to TB.
    """

    def __init__(self, Rlist, nbasis=0, dHR={}, dHR2={}, wsdeg=None):
        self.Rlist = Rlist
        self.Rdict = {tuple(R): i for i, R in enumerate(Rlist)}
        self.HR = None
        self.dHR = {}
        self.d2HR = {}
        self.nbasis = nbasis
        self.R2kfactor = 2j * np.pi
        self.amp = 0.1
        self.dHR = dHR
        self.dHR2 = dHR2
        if dHR and wsdeg is None:
            self.wsdeg = np.ones(len(self.Rlist))
        else:
            self.wsdeg = wsdeg

    @staticmethod
    def fit(tb0, tb1, tb2, amp):
        cls = dHdx(tb1.Rlist)
        cls.nbasis = tb2.nbasis
        for iR, R in enumerate(cls.Rlist):
            h0 = tb0.get_hamR(R)
            h1 = tb1.get_hamR(R)
            h2 = tb2.get_hamR(R)
            d1, d2 = FD12(h0, h1, h2, amp)
            cls.dHR[R] = d1
            cls.d2HR[R] = d2

        cls.wsdeg = np.ones(len(cls.Rlist))
        return cls

    def get_wsdeg(self, R):
        return self.wsdeg[self.Rdict[R]]

    def print_wsdeg(self):
        for iR, R in enumerate(self.Rlist):
            print(f"R={R}, wsdge(R)={self.wsdeg[iR]}")

    def get_dHR(self, R):
        """
        The first order derivative to HR
        """
        return self.dHR[tuple(R)]

    def get_d2HR(self, R):
        """
        The second order derivative to HR
        """
        return self.d2HR[tuple(R)]

    @property
    def dHR_0(self):
        return self.get_dHR(R=(0, 0, 0))

    def get_dHk(self, k):
        """
        the first order derivative of Hk
        """
        dHk = np.zeros((self.nbasis, self.nbasis), dtype="complex")
        for iR, R in enumerate(self.Rlist):
            mat = self.get_dHR(R)
            phase = np.exp(self.R2kfactor * np.dot(k, R))
            dHk += mat * phase / self.wsdeg[iR]
        dHk = (dHk + dHk.conj().T) / 2
        return dHk

    def get_dHk2(self, k):
        """
        the second order derivative of Hk
        """
        dHk2 = np.zeros((self.nbasis, self.nbasis), dtype="complex")
        for R in self.Rlist:
            mat = self.get_d2HR(R)
            phase = np.exp(self.R2kfactor * np.dot(k, R))
            dHk2 += mat * phase
        dHk2 = (dHk2 + dHk2.conj().T) / 2
        return dHk2

    def make_supercell(self, scmaker: SupercellMaker):
        sc_dHR = scmaker.sc_RHdict_notrans(self.dHR, self.nbasis)
        # Note that the electron phono coupling does not
        # translate. Therefore is only with sc_vec=0
        sc_R = np.array(list(sc_dHR.keys()), dtype=int)
        sc_nbasis = scmaker.ncell * self.nbasis
        sc_wsdeg = np.ones(len(sc_dHR), dtype=float)
        sc_dHdx = dHdx(sc_R,
                       nbasis=sc_nbasis,
                       dHR=sc_dHR,
                       dHR2=None,
                       wsdeg=sc_wsdeg)
        return sc_dHdx

    def duplicate_spin(self):
        nb = self.nbasis * 2
        newdHR = {}
        for R, val in self.dHR.items():
            d = np.zeros((nb, nb), dtype=complex)
            d[::2, ::2] = val
            d[1::2, 1::2] = val
            newdHR[R] = d
        newdHdx = dHdx(Rlist=self.Rlist,
                       nbasis=nb,
                       dHR=newdHR,
                       wsdeg=self.wsdeg)
        return newdHdx

    def save_to_pickle(self, fname):
        with bz2.BZ2File(fname, "wb") as myfile:
            pickle.dump(self, myfile)

    @staticmethod
    def load_from_pickle(fname):
        data = bz2.BZ2File(fname, "rb")
        return pickle.load(data)


def diffTB(posfile, path0, path1, path2, amp, output="dHdx.pickle"):
    atoms = read(posfile)
    tb0_up = MyTB.read_from_wannier_dir(path0,
                                        prefix="wannier90.up",
                                        atoms=atoms)
    tb1_up = MyTB.read_from_wannier_dir(path1,
                                        prefix="wannier90.up",
                                        atoms=atoms)
    tb2_up = MyTB.read_from_wannier_dir(path2,
                                        prefix="wannier90.up",
                                        atoms=atoms)

    tb0_dn = MyTB.read_from_wannier_dir(path0,
                                        prefix="wannier90.dn",
                                        atoms=atoms)
    tb1_dn = MyTB.read_from_wannier_dir(path1,
                                        prefix="wannier90.dn",
                                        atoms=atoms)
    tb2_dn = MyTB.read_from_wannier_dir(path2,
                                        prefix="wannier90.dn",
                                        atoms=atoms)

    tb0 = merge_tbmodels_spin(tb0_up, tb0_dn)
    tb1 = merge_tbmodels_spin(tb1_up, tb1_dn)
    tb2 = merge_tbmodels_spin(tb2_up, tb2_dn)

    difftb = dHdx.fit(tb0, tb1, tb2, amp)
    difftb.save_to_pickle(output)


def test():
    #path0 = "./U3_SrMnO3_111_rot-0.10"
    #path1 = "./U3_SrMnO3_111_slater0.00"
    #path2 = "./U3_SrMnO3_111_rot0.10"

    path0 = "./U3_SrMnO3_111_slater-0.02"
    path1 = "./U3_SrMnO3_111_slater0.00"
    path2 = "./U3_SrMnO3_111_slater0.02"
    diffTB("POSCAR.vasp", path0, path1, path2, amp=0.02)


def test_shiftz():
    path0 = "./U3_SrMnO3_111_shiftz-0.01"
    path1 = "./U3_SrMnO3_111_slater0.00"
    path2 = "./U3_SrMnO3_111_shiftz0.01"
    diffTB("POSCAR.vasp",
           path0,
           path1,
           path2,
           amp=0.01,
           output="dHdx_shiftz.pickle")


def test_read():
    m = dHdx.load_from_pickle("dHdx.pickle")
    print(m.Rlist)
    print(m.Rdict)


if __name__ == "__main__":
    # test()
    # test_read()
    test_shiftz()

# test()
