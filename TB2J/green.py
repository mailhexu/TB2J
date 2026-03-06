import copy
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
from HamiltonIO.model.occupations import GaussOccupations
from HamiltonIO.model.occupations import myfermi as fermi
from pathos.multiprocessing import ProcessPool

from TB2J.kpoints import ir_kpts, monkhorst_pack
from TB2J.sharedmem import (
    attach_shm,
    detach_shm,
    free_shm,
    read_shm,
    to_shm,
)

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
    # print("Finding energy in gap...")
    # print(f"Right bound: {rbound}, min gap size: {gap}")
    m0 = np.sort(evals.flatten())
    # print(f"Min eigenvalue: {m0[0]}, Max eigenvalue: {m0[-1]}")
    m = m0[m0 <= rbound]
    # append the next state above rbound to m
    if len(m0) > len(m):
        m = np.append(m, m0[len(m)])
    # print(f"Number of states below right bound: {len(m)}")
    # print(f"Max eigenvalue below right bound: {m[-1]}")
    ind = np.where(np.diff(m) > gap)[0]
    # print(f"Number of gaps found: {len(ind)}")
    # print("ind[-1]: ", ind[-1] if len(ind) > 0 else "N/A")
    emin = 0.0
    if len(ind) == 0:
        emin = m0[0] - 0.5
    else:
        emin = m[ind[-1] + 1] - 0.5
    # print("emin:", emin)
    return emin


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
        nproc=1,
        initial_emin=-25,
        smearing_width=0.01,
    ):
        """
        :param tbmodel: A tight binding model
        :param kmesh: size of monkhorst pack. e.g [6,6,6]
        :param efermi: fermi energy.
        :param gamma: whether to include gamma point in monkhorst pack
        :param kpts: user defined kpoints
        :param kweights: weights for user defined kpoints
        :param k_sym: whether the kpoints are symmetrized
        :param nproc: number of processes to use
        :param emin: minimum energy relative to fermi level to consider
        """
        self.initial_emin = initial_emin
        self.tbmodel = tbmodel
        self.is_orthogonal = tbmodel.is_orthogonal
        self.R2kfactor = tbmodel.R2kfactor
        self.k2Rfactor = -tbmodel.R2kfactor
        self.efermi = efermi
        self._use_cache = True
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
        self.fermi_width = float(smearing_width)
        print(
            f"starting to prepare eigenvalues and eigenvectors for {self.nkpts} k-points..."
        )
        t0 = time.time()
        self._prepare_eigen()
        print(
            f"Finished preparing eigenvalues and eigenvectors. Time taken: {time.time() - t0:.2f} seconds"
        )

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
        # istart, iend = ts[0], ts[-1] + 1
        iend = ts[-1] + 1
        # return evals[:, istart:iend], evecs[:, :, istart:iend]
        return evals[:, :iend], evecs[:, :, :iend]

    def find_energy_ingap(self, rbound, gap=2.0):
        return find_energy_ingap(self.evals, rbound, gap)

    def _prepare_cache(self):
        # Shared memory handles (populated in _prepare_eigen)
        self._shm_evecs = None
        self._shm_S = None

    def clean_cache(self):
        for attr in (
            "_shm_evecs",
            "_shm_S",
            "_par_shm_evals",
            "_par_shm_evecs",
            "_par_shm_S",
        ):
            shm = getattr(self, attr, None)
            if shm is not None:
                try:
                    free_shm(shm)
                except Exception:
                    pass
                setattr(self, attr, None)

    def enter_parallel(self):
        """Move evals/evecs (and S) into shared memory so worker processes
        can attach without copying the data through dill serialisation.
        The large arrays are removed from the object; workers reconstruct
        zero-copy views via the stored ShmDescriptor."""
        if getattr(self, "_in_parallel", False):
            return  # already in parallel mode
        self._in_parallel = True

        # evals
        self._par_shm_evals, self._par_desc_evals = to_shm(self.evals, "evals")
        del self.evals

        # evecs are already in use_cache shm — no need to duplicate
        self._par_shm_evecs = None

        # S is already in use_cache shm (if non-orthogonal) — no need to duplicate
        self._par_shm_S = None

    def exit_parallel(self):
        """Restore evals/evecs/S to normal numpy arrays and release shared memory."""
        if not getattr(self, "_in_parallel", False):
            return
        self._in_parallel = False

        arr, shm = attach_shm(self._par_desc_evals)
        self.evals = arr.copy()
        detach_shm(shm)
        free_shm(self._par_shm_evals)
        self._par_shm_evals = None

        if self._par_shm_evecs is not None:
            arr, shm = attach_shm(self._par_desc_evecs)
            self.evecs = arr.copy()
            detach_shm(shm)
            free_shm(self._par_shm_evecs)
            self._par_shm_evecs = None

        if self._par_shm_S is not None:
            arr, shm = attach_shm(self._par_desc_S)
            self.S = arr.copy()
            detach_shm(shm)
            free_shm(self._par_shm_S)
            self._par_shm_S = None

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

        self.adjusted_emin = (
            find_energy_ingap(
                self.evals, rbound=self.efermi + self.initial_emin, gap=2.0
            )
            - self.efermi
        )
        # print(f"Adjusted emin relative to Fermi level: {self.adjusted_emin}")
        self.evals, self.evecs = self._reduce_eigens(
            self.evals,
            self.evecs,
            emin=self.efermi + self.adjusted_emin,
            emax=self.efermi + 5.1,
            # emin=self.efermi -10,
            # emax=self.efermi + 10,
        )
        self._shm_evecs, self._desc_evecs = to_shm(self.evecs, "evecs")
        del self.evecs

        if self.is_orthogonal:
            self.S = None
        else:
            self._shm_S, self._desc_S = to_shm(self.S, "S")
            del self.S

    def get_evecs(self, ik):
        return read_shm(self._desc_evecs, ik)

    def get_evalue(self, ik):
        if getattr(self, "_in_parallel", False):
            return read_shm(self._par_desc_evals, ik)
        return self.evals[ik]

    def get_Hk(self, ik):
        return self.H[ik] if self.H is not None else None

    def get_Sk(self, ik):
        if self.is_orthogonal:
            return None
        return read_shm(self._desc_S, ik)

    def get_density_matrix(self):
        rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        if self.is_orthogonal:
            for ik, _ in enumerate(self.kpts):
                evecs_k = self.get_evecs(ik)
                evals_k = self.get_evalue(ik)
                # chekc if any of the evecs element is nan
                rho += (
                    (
                        evecs_k
                        * fermi(evals_k, self.efermi, width=self.fermi_width, nspin=2)
                    )
                    @ evecs_k.T.conj()
                    * self.kweights[ik]
                )
        else:
            for ik, _ in enumerate(self.kpts):
                evecs_k = self.get_evecs(ik)
                evals_k = self.get_evalue(ik)
                rho += (
                    (
                        evecs_k
                        * fermi(evals_k, self.efermi, width=self.fermi_width, nspin=2)
                    )
                    @ evecs_k.T.conj()
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
                fermi(
                    self.get_evalue(ik), self.efermi, width=self.fermi_width, nspin=2
                ),
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
        r"""Green's function G(k) for one energy
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

    def compute_GR(self, Rpts, kpts, Gks):
        Rvecs = np.array(Rpts)
        phase = np.exp(self.k2Rfactor * np.einsum("ni,mi->nm", Rvecs, kpts))
        phase *= self.kweights[None]
        GR = np.einsum("kij,rk->rij", Gks, phase, optimize="optimal")
        return GR

    def get_GR(self, Rpts, energy, Gk_all=None):
        r"""calculate real space Green's function for one energy, all R points.
        G(R, epsilon) = G(k, epsilon) exp(-2\pi i R.dot. k)
        :param Rpts: R points
        :param energy: energy value
        :param Gk_all: optional pre-computed Gk for all k-points
        :returns: real space green's function for one energy for a list of R.
        :rtype: numpy array of shape (len(Rpts), nbasis, nbasis)
        """
        if Gk_all is not None:
            return self.compute_GR(Rpts, self.kpts, Gk_all)

        Rvecs = np.array(Rpts)
        nR = len(Rvecs)
        GR = np.zeros((nR, self.nbasis, self.nbasis), dtype=complex)

        for ik, kpt in enumerate(self.kpts):
            Gk = self.get_Gk(ik, energy)
            weight = self.kweights[ik]
            phase_k = np.exp(self.k2Rfactor * np.dot(Rvecs, kpt)) * weight
            GR += Gk[None, :, :] * phase_k[:, None, None]

        return GR

    def get_GR_and_dGRdx1(self, Rpts, energy, dHdx):
        r"""
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
        r"""
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
        r"""
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
