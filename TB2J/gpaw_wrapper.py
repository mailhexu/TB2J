try:
    from gpaw.lcao.tightbinding import TightBinding
    from gpaw import restart, GPAW
    from gpaw.lcao.tools import (
        get_lcao_hamiltonian,
        get_bf_centers,
    )

    _has_gpaw = True
except:
    _has_gpaw = False

# from banddownfolder.scdm.downfolder import BandDownfolder
import numpy as np
from scipy.linalg import eigh
from ase.dft.kpoints import monkhorst_pack
import pickle


class GPAWWrapper:
    """
    https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/lcao/tightbinding.py
    """

    def __init__(self, atoms=None, calc=None, gpw_fname=None):
        if not _has_gpaw:
            raise ImportError("GPAW was not imported. Please install gpaw.")
        if atoms is None or calc is None:
            atoms, calc = restart(gpw_fname, fixdensity=False, txt="nscf.txt")
        self.atoms = atoms
        self.calc = calc
        # self.wfs=self.calc.wfs
        # self.h=self.calc.hamiltonian
        # self.kpt_u = self.wfs.kpt_u
        self.cell = self.atoms.get_cell()
        # self.wann_centers=get_bf_centers(self.atoms)
        self.positions = self.cell.scaled_positions(get_bf_centers(self.atoms))
        # if calc.get_spin_polarized():
        #    self.positions=np.vstack([self.positions, self.positions])
        self.efermi = self.calc.get_fermi_level()
        self.R2kfactor = -2.0j * np.pi
        # self.nbasis=len(self.positions)
        self.norb = len(self.positions)
        self.nbasis = self.norb * 2
        self._name = "GPAW"

    # def gen_ham(self, k):
    #    H_MM=self.wfs.eigensolver.calculate_hamiltonian_matrix(self.h, self.wfs, k)
    #    tri2full(H_MM*Ha)
    #    return H_MM

    # def Sk(self, k):
    #    S_MM=self.wfs.S_qMM[kpt.q]
    #    tri2full(S_MM)
    #    return S_MM

    # def solve(self, k):
    #    return eigh(self.hamk(k), self.S(k))

    def get_kpts(self):
        return self.calc.get_ibz_k_points()

    def HS_and_eigen(self, kpts=None, convention=2):
        if kpts is not None:
            self.calc.set(kpts=kpts)
            self.atoms.calc = self.calc
            self.atoms.get_total_energy()

        self.calc.set(symmetry="off", fixdensity=True)
        self.atoms.get_total_energy()
        # self.kpt_u = self.wfs.kpt_u
        H, S = get_lcao_hamiltonian(self.calc)
        np.save("kpts.npy", kpts)
        np.save("H.npy", H)
        np.save("S.npy", S)

        nspin, nkpt, norb, _ = H.shape
        nbasis = nspin * norb
        evals = np.zeros((nkpt, nbasis), dtype=float)
        evecs = np.zeros((nkpt, nbasis, nbasis), dtype=complex)
        if self.calc.get_spin_polarized():
            H2 = np.zeros((nkpt, nbasis, nbasis), dtype=complex)
            # spin up
            H2[:, :norb, :norb] = H[0]
            # spin down
            H2[:, norb:, norb:] = H[1]
            S2 = np.zeros((nkpt, nbasis, nbasis), dtype=complex)
            S2[:, :norb, :norb] = S
            S2[:, norb:, norb:] = S

            for ikpt, k in enumerate(self.calc.get_ibz_k_points()):
                evals0, evecs0 = eigh(H[0, ikpt, :, :], S[ikpt, :, :])
                evals1, evecs1 = eigh(H[1, ikpt, :, :], S[ikpt, :, :])
                evals[ikpt, :norb] = evals0
                evals[ikpt, norb:] = evals1
                evecs[ikpt, :norb, :norb] = evecs0
                evecs[ikpt, norb:, norb:] = evecs1
        else:
            H2 = H[0]
            for ikpt, k in self.kpt_u:
                evals[ikpt], evecs[ikpt] = eigh(H[0, ikpt], S[ikpt])
        np.save("kpts.npy", kpts)
        np.save("H2.npy", H2)
        np.save("S2.npy", S2)
        np.save("evals.npy", evals)
        np.save("evecs.npy", evecs)
        return H2, S2, evals, evecs

    def solve_all(self, kpts, convention=2):
        H, evals, evecs = self.H_and_eigen(kpts=kpts, convention=convention)
        return evals, evecs


class GPAWTBWrapper:
    def __init__(self, calc=None, atoms=None, gpw_fname=None, pickle_fname=None):
        if not _has_gpaw:
            raise ImportError("GPAW was not imported. Please install gpaw.")
        if pickle_fname is None:
            if gpw_fname is not None:
                calc = GPAW(
                    gpw_fname,
                    fixdensity=True,
                    symmetry="off",
                    txt="TB2J_wrapper.log",
                    basis="dzp",
                    mode="lcao",
                )
                atoms = calc.atoms
                atoms.calc = calc
                atoms.get_potential_energy()
            tb = TightBinding(atoms, calc)
            self.H_NMM, self.S_NMM = tb.h_and_s()
            self.Rlist = tb.R_cN.T
        else:
            with open(gpw_fname, "rb") as myfile:
                self.H_NMM, self.S_NMM, self.Rlist = pickle.load(myfile)
        self.nR, self.nbasis, _ = self.H_NMM.shape
        self.positions = np.zeros((self.nbasis, 3))

    def save_pickle(self, fname):
        with open(fname, "wb") as myfile:
            pickle.dump([self.H_NMM, self.S_NMM, self.Rlist], myfile)

    def solve(self, k):
        self.Hk = np.zeros((self.nbasis, self.nbasis), dtype="complex")
        self.Sk = np.zeros((self.nbasis, self.nbasis), dtype="complex")
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(-2j * np.pi * np.dot(k, R))
            self.Hk += self.H_NMM[iR] * phase
            self.Sk += self.S_NMM[iR] * phase
        self.Sk = np.real((self.Sk + self.Sk.T) / 2)
        return eigh(self.Hk, self.Sk)
        try:
            return eigh(self.Hk, self.Sk)
        except:
            return np.zeros(self.nbasis), np.zeros((self.nbasis, self.nbasis))

    def solve_all(self, kpts):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs, dtype=complex, order="C")


def test_Ham():
    calc = GPAW(
        "/home/hexu/projects/gpaw/bccFe/atoms_nscf.gpw",
        fixdensity=True,
        symmetry="off",
        txt="nscf.txt",
    )
    atoms = calc.atoms
    atoms.calc = calc
    calc.set(kpts=(3, 3, 3))
    atoms.get_potential_energy()
    tb = TightBinding(atoms, calc)
    kpath = monkhorst_pack([3, 3, 3])
    evals, evecs = tb.band_structure(kpath, blochstates=True)


def test():
    g = GPAWWrapper(fname="STO_nscf.gpw")
    g.H_and_eigen(kpts=monkhorst_pack(2, 2, 2))
    # g.save_pickle('model.pickle')
    # df=GPAWDownfolder('model.pickle')
    # df.downfold(nwann=1, anchor_kpt=[0,0,0] )
    # df.plot_band_fitting(npoints=100)


# test_Ham()


def test():
    g = GPAWWrapper(gpw_fname="/home/hexu/projects/gpaw/bccFe/atoms_nscf.gpw")
    H, E, V = g.H_and_eigen(kpts=monkhorst_pack([2, 2, 2]))
    print(H.shape)
    print(E.shape)
    print(E)
    print(V.shape)
