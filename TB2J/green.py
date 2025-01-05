import copy
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from shutil import rmtree

import numpy as np
from HamiltonIO.model.occupations import GaussOccupations
from HamiltonIO.model.occupations import myfermi as fermi
from pathos.multiprocessing import ProcessPool

from TB2J.kpoints import ir_kpts, monkhorst_pack

# from TB2J.mathutils.fermi import fermi

MAX_EXP_ARGUMENT = np.log(sys.float_info.max)


def eigen_to_G2(H, S, efermi, energy):
    """calculate green's function from eigenvalue/eigenvector for energy(e-ef): G(e-ef).
    :param H: Hamiltonian matrix in eigenbasis
    :param S: Overlap matrix in eigenbasis
    :param efermi: fermi energy
    :param energy: energy level e - efermi
    """
    # G = ((E+Ef) S - H)^-1
    return np.linalg.inv((energy + efermi) * S - H)


def eigen_to_G(evals, evecs, efermi, energy):
    """calculate green's function from eigenvalue/eigenvector for energy(e-ef): G(e-ef).
    :param evals:  eigen values
    :param evecs:  eigen vectors
    :param efermi: fermi energy
    :param energy: energy
    :returns: Green's function G,
    :rtype:  Matrix with same shape of the Hamiltonian (and eigenvector)
    """
    # return (
    #    np.einsum("ij, j-> ij", evecs, 1.0 / (-evals + (energy + efermi)))
    #    @ evecs.conj().T
    # )
    return np.einsum(
        "ib, b, jb-> ij",
        evecs,
        1.0 / (-evals + (energy + efermi)),
        evecs.conj(),
        optimize=True,
    )


def find_energy_ingap(evals, rbound, gap=2.0):
    """
    find a energy inside a gap below rbound (right bound),
    return the energy gap top - 0.5.
    """
    m0 = np.sort(evals.flatten())
    m = m0[m0 < rbound]
    ind = np.where(np.diff(m) > gap)[0]
    if len(ind) == 0:
        return m0[0] - 0.5
    else:
        return m[ind[-1] + 1] - 0.5


class TBGreen:
    def __init__(
        self,
        tbmodel,
        kmesh=None,  # [ikpt, 3]
        ibz=False,  # if True, will interpolate the Green's function at Ir-kpoints
        efermi=None,  # efermi
        gamma=False,
        kpts=None,
        kweights=None,
        k_sym=False,
        use_cache=False,
        cache_path=None,
        nproc=1,
    ):
        """
        :param tbmodel: A tight binding model
        :param kmesh: size of monkhorst pack. e.g [6,6,6]
        :param efermi: fermi energy.
        """
        self.tbmodel = tbmodel
        self.is_orthogonal = tbmodel.is_orthogonal
        self.R2kfactor = tbmodel.R2kfactor
        self.k2Rfactor = -tbmodel.R2kfactor
        self.efermi = efermi
        self._use_cache = use_cache
        self.cache_path = cache_path
        if use_cache:
            self._prepare_cache()
        self.prepare_kpts(
            kmesh=kmesh,
            ibz=ibz,
            gamma=gamma,
            kpts=kpts,
            kweights=kweights,
            tbmodel=tbmodel,
        )

        self.norb = tbmodel.norb
        self.nbasis = tbmodel.nbasis
        self.k_sym = k_sym
        self.nproc = nproc
        self._prepare_eigen()

    def prepare_kpts(
        self, kmesh=None, gamma=True, ibz=False, kpts=None, kweights=None, tbmodel=None
    ):
        """
        Prepare the k-points for the calculation.

        Parameters:
        - kmesh (tuple): The k-mesh used to generate k-points.
        - gamma (bool): Whether to include the gamma point in the k-points.
        - ibz (bool): Whether to use the irreducible Brillouin zone.
        - kpts (list): List of user-defined k-points.
        - kweights (list): List of weights for each k-point.

        Returns:
        None
        """
        if kpts is not None:
            self.kpts = kpts
            self.nkpts = len(self.kpts)
            self.kweights = kweights
        elif kmesh is not None:
            if ibz:
                self.kpts, self.kweights = ir_kpts(
                    atoms=tbmodel.atoms,
                    mp_grid=kmesh,
                    ir=True,
                    is_time_reversal=False,
                )
                self.nkpts = len(self.kpts)
                print(f"Using IBZ of kmesh of {kmesh}")
                print(f"Number of kpts: {self.nkpts}")
                for kpt, weight in zip(self.kpts, self.kweights):
                    # format the kpt and weight, use 5 digits
                    print(f"{kpt[0]:8.5f} {kpt[1]:8.5f} {kpt[2]:8.5f} {weight:8.5f}")
            else:
                self.kpts = monkhorst_pack(kmesh, gamma_center=gamma)
                self.nkpts = len(self.kpts)
                self.kweights = np.array([1.0 / self.nkpts] * self.nkpts)
        else:
            self.kpts = tbmodel.get_kpts()
            self.nkpts = len(self.kpts)
            self.kweights = np.array([1.0 / self.nkpts] * self.nkpts)

    def _reduce_eigens(self, evals, evecs, emin, emax):
        ts = np.logical_and(evals >= emin, evals < emax)
        ts = np.any(ts, axis=0)
        ts = np.where(ts)[0]
        if len(ts) == 0:
            raise ValueError(
                f"Cannot find any band in the energy range specified by emin {emin} and emax {emax}, which are relative to the Fermi energy. Please check that the Fermi energy, the emin and emax are correct. If you're using Wannier90 output, check the Wannier functions give the right band structure."
            )
        istart, iend = ts[0], ts[-1] + 1
        return evals[:, istart:iend], evecs[:, :, istart:iend]

    def find_energy_ingap(self, rbound, gap=2.0):
        return find_energy_ingap(self.evals, rbound, gap)

    def _prepare_cache(self):
        if self.cache_path is None:
            if "TMPDIR" in os.environ:
                rpath = os.environ["TMPDIR"]
            else:
                rpath = "/dev/shm/TB2J_cache"
        else:
            rpath = self.cache_path
        if not os.path.exists(rpath):
            os.makedirs(rpath)
        self.cache_path = tempfile.mkdtemp(prefix="TB2J", dir=rpath)
        print(f"Writting wavefunctions and Hamiltonian in cache {self.cache_path}")

    def clean_cache(self):
        if (self.cache_path is not None) and os.path.exists(self.cache_path):
            rmtree(self.cache_path)

    def _prepare_eigen(self, solve=True, saveH=False):
        """
        calculate eigen values and vectors for all kpts and save.
        Note that the convention 2 is used here, where the
        phase factor is e^(ik.R), not e^(ik.(R+rj-ri))
        """
        nkpts = len(self.kpts)
        self.evals = np.zeros((nkpts, self.nbasis), dtype=float)
        self.nkpts = nkpts
        self.H0 = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        self.evecs = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        self.H = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        if not self.is_orthogonal:
            self.S = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        else:
            self.S = None
        if self.nproc == 1:
            results = map(self.tbmodel.HSE_k, self.kpts)
        else:
            executor = ProcessPool(nodes=self.nproc)
            results = executor.map(self.tbmodel.HSE_k, self.kpts, [2] * len(self.kpts))
            executor.close()
            executor.join()
            executor.clear()

        for ik, result in enumerate(results):
            if self.is_orthogonal:
                self.H[ik], _, self.evals[ik], self.evecs[ik] = result
            else:
                self.H[ik], self.S[ik], self.evals[ik], self.evecs[ik] = result
            self.H0 += self.H[ik] / self.nkpts

        if not saveH:
            self.H = None

        # get efermi
        if self.efermi is None:
            print("Calculating Fermi energy from eigenvalues")
            print(f"Number of electrons: {self.tbmodel.nel} ")

            # occ = Occupations(
            #    nel=self.tbmodel.nel, width=0.1, wk=self.kweights, nspin=2
            # )
            # self.efermi = occ.efermi(copy.deepcopy(self.evals))
            # print(f"Fermi energy found: {self.efermi}")

            occ = GaussOccupations(
                nel=self.tbmodel.nel, width=0.1, wk=self.kweights, nspin=2
            )
            self.efermi = occ.efermi(copy.deepcopy(self.evals))
            print(f"Fermi energy found: {self.efermi}")

        # self.evals, self.evecs = self._reduce_eigens(
        #    self.evals, self.evecs, emin=self.efermi - 15.0, emax=self.efermi + 10.1
        # )
        if self._use_cache:
            evecs = self.evecs
            self.evecs_shape = self.evecs.shape
            self.evecs = np.memmap(
                os.path.join(self.cache_path, "evecs.dat"),
                mode="w+",
                shape=self.evecs.shape,
                dtype=complex,
            )
            self.evecs[:, :, :] = evecs[:, :, :]
            del self.evecs

            if self.is_orthogonal:
                self.S = None
            else:
                S = self.S
                self.S = np.memmap(
                    os.path.join(self.cache_path, "S.dat"),
                    mode="w+",
                    shape=(nkpts, self.nbasis, self.nbasis),
                    dtype=complex,
                )
                self.S[:] = S[:]
            if not self.is_orthogonal:
                del self.S

    def get_evecs(self, ik):
        if self._use_cache:
            return np.memmap(
                os.path.join(self.cache_path, "evecs.dat"),
                mode="r",
                shape=self.evecs_shape,
                dtype=complex,
            )[ik]
        else:
            return self.evecs[ik]

    def get_evalue(self, ik):
        return self.evals[ik]

    def get_Hk(self, ik):
        if self._use_cache:
            return np.memmap(
                os.path.join(self.cache_path, "H.dat"),
                mode="r",
                shape=(self.nkpts, self.nbasis, self.nbasis),
                dtype=complex,
            )[ik]
        else:
            return self.H[ik]

    def get_Sk(self, ik):
        if self.is_orthogonal:
            return None
        elif self._use_cache:
            return np.memmap(
                os.path.join(self.cache_path, "S.dat"),
                mode="r",
                shape=(self.nkpts, self.nbasis, self.nbasis),
                dtype=complex,
            )[ik]
        else:
            return self.S[ik]

    def get_density_matrix(self):
        rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        if self.is_orthogonal:
            for ik, _ in enumerate(self.kpts):
                evecs_k = self.get_evecs(ik)
                # chekc if any of the evecs element is nan
                rho += (
                    (evecs_k * fermi(self.evals[ik], self.efermi, nspin=2))
                    @ evecs_k.T.conj()
                    * self.kweights[ik]
                )
        else:
            for ik, _ in enumerate(self.kpts):
                rho += (
                    (self.get_evecs(ik) * fermi(self.evals[ik], self.efermi, nspin=2))
                    @ self.get_evecs(ik).T.conj()
                    @ self.get_Sk(ik)
                    * self.kweights[ik]
                )
        # check if rho has nan values
        return rho

    def get_rho_R(self, Rlist):
        nR = len(Rlist)
        rho_R = np.zeros((nR, self.nbasis, self.nbasis), dtype=complex)
        for ik, kpt in enumerate(self.kpts):
            evec = self.get_evecs(ik)
            rhok = np.einsum(
                "ib,b, bj-> ij",
                evec,
                fermi(self.evals[ik], self.efermi, nspin=2),
                evec.conj().T,
            )
            for iR, R in enumerate(Rlist):
                rho_R[iR] += rhok * np.exp(self.k2Rfactor * kpt @ R) * self.kweights[ik]
        return rho_R

    def write_rho_R(self, Rlist, fname="rhoR.pickle"):
        rho_R = self.get_rho_R(Rlist)
        with open(fname, "wb") as myfile:
            pickle.dump({"Rlist": Rlist, "rhoR": rho_R}, myfile)

    def get_density(self):
        return np.real(np.diag(self.get_density_matrix()))

    def get_Gk(self, ik, energy, evals=None, evecs=None):
        """Green's function G(k) for one energy
        G(\epsilon)= (\epsilon I- H)^{-1}
        :param ik: indices for kpoint
        :returns: Gk
        :rtype:  a matrix of indices (nbasis, nbasis)
        """
        if evals is None:
            evals = self.get_evalue(ik)
        if evecs is None:
            evecs = self.get_evecs(ik)
        Gk = eigen_to_G(
            evals=evals,
            evecs=evecs,
            efermi=self.efermi,
            energy=energy,
        )

        # A slower version. For test.
        # Gk = np.linalg.inv((energy+self.efermi)*self.S[ik,:,:] - self.H[ik,:,:])
        return Gk

    def get_Gk_all(self, energy):
        """Green's function G(k) for one energy for all kpoints"""
        Gk_all = np.zeros((self.nkpts, self.nbasis, self.nbasis), dtype=complex)
        for ik, _ in enumerate(self.kpts):
            Gk_all[ik] = self.get_Gk(ik, energy)
        return Gk_all

    def get_GR(self, Rpts, energy, get_rho=False, Gk_all=None):
        """calculate real space Green's function for one energy, all R points.
        G(R, epsilon) = G(k, epsilon) exp(-2\pi i R.dot. k)
        :param Rpts: R points
        :param energy:
        :returns:  real space green's function for one energy for a list of R.
        :rtype:  dictionary, the keys are tuple of R, values are matrices of nbasis*nbasis
        """
        Rpts = [tuple(R) for R in Rpts]
        GR = defaultdict(lambda: 0.0j)
        rhoR = defaultdict(lambda: 0.0j)
        for ik, kpt in enumerate(self.kpts):
            if Gk_all is None:
                Gk = self.get_Gk(ik, energy)
            else:
                Gk = Gk_all[ik]
            if get_rho:
                if self.is_orthogonal:
                    rhok = Gk
                else:
                    rhok = self.get_Sk(ik) @ Gk
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                tmp = Gk * (phase * self.kweights[ik])
                GR[R] += tmp
                # change this if need full rho
                if get_rho and R == (0, 0, 0):
                    rhoR[R] += rhok * (phase * self.kweights[ik])
        if get_rho:
            return GR, rhoR
        else:
            return GR

    def get_GR_and_dGRdx1(self, Rpts, energy, dHdx):
        """
        calculate G(R) and dG(R)/dx.
        dG(R)/dx = \sum_k G(k) (dH(R)/dx) G(k).
        """
        Rpts = [tuple(R) for R in Rpts]
        GR = defaultdict(lambda: 0.0 + 0.0j)
        dGRdx = defaultdict(lambda: 0.0 + 0j)
        for ik, kpt in enumerate(self.kpts):
            Gk = self.get_Gk(ik, energy)
            Gkw = Gk * self.kweights[ik]
            # Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                GR[R] += Gkw * (phase * self.kweights[ik])
                dHRdx = dHdx.get_hamR(R)
                dGRdx[R] += Gkw @ dHRdx @ Gk
                # dGRdx[R] += Gk.dot(dHRdx).dot(Gkp)
        return GR, dGRdx

    def get_GR_and_dGRdx(self, Rpts, energy, dHdx):
        """
        calculate G(R) and dG(R)/dx.
        dG(k)/dx =  G(k) (dH(k)/dx) G(k).
        dG(R)/dx = \sum_k dG(k)/dx * e^{-ikR}
        """
        Rpts = [tuple(R) for R in Rpts]
        GR = defaultdict(lambda: 0.0 + 0.0j)
        dGRdx = defaultdict(lambda: 0.0 + 0j)
        for ik, kpt in enumerate(self.kpts):
            Gk = self.get_Gk(ik, energy)
            # Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
            Gkp = Gk * self.kweights[ik]
            dHk = dHdx.gen_ham(tuple(kpt))
            dG = Gk @ dHk @ Gkp
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                GR[R] += Gkp * (phase * self.kweights[ik])
                dGRdx[R] += dG * (phase * self.kweights[ik])
        return GR, dGRdx

    def get_GR_and_dGRdx_and_dGRdx2(self, Rpts, energy, dHdx, dHdx2):
        """
        calculate G(R) and dG(R)/dx.
        dG(k)/dx =  G(k) (dH(k)/dx) G(k).
        dG(R)/dx = \sum_k dG(k)/dx * e^{-ikR}
        """
        Rpts = [tuple(R) for R in Rpts]
        GR = defaultdict(lambda: 0.0 + 0.0j)
        dGRdx = defaultdict(lambda: 0.0 + 0j)
        dGRdx2 = defaultdict(lambda: 0.0 + 0j)
        for ik, kpt in enumerate(self.kpts):
            Gk = self.get_Gk(ik, energy)
            # Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
            Gkp = Gk * self.kweights[ik]
            dHk = dHdx.gen_ham(tuple(kpt))
            dHk2 = dHdx2.gen_ham(tuple(kpt))
            dG = Gk @ dHk @ Gkp
            dG2 = Gk @ dHk2 @ Gkp
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                GR[R] += Gkp * (phase * self.kweights[ik])
                dGRdx[R] += dG * (phase * self.kweights[ik])
                dGRdx2[R] += dG2 * (phase * self.kweights[ik])
        return GR, dGRdx, dGRdx2
