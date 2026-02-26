import copy
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from shutil import rmtree

import numpy as np
from HamiltonIO.epw.epwparser import EpmatOneMode
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
    # ...ib, ...b, ...jb -> ...ij matches both (nb, nb) and (nk, nb, nb) cases
    return np.einsum(
        "...ib, ...b, ...jb-> ...ij",
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
        use_cache=False,
        cache_path=None,
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
        :param use_cache: whether to use cache to store wavefunctions
        :param cache_path: path to store cache
        :param nproc: number of processes to use
        :param emin: minimum energy relative to fermi level to consider
        """
        self.initial_emin = initial_emin
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
        self.fermi_width = float(smearing_width)

        # Initialize Rmap for spin-phonon coupling
        self._Rmap = None
        self._Rmap_rev = None

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
                    (
                        evecs_k
                        * fermi(
                            self.evals[ik], self.efermi, width=self.fermi_width, nspin=2
                        )
                    )
                    @ evecs_k.T.conj()
                    * self.kweights[ik]
                )
        else:
            for ik, _ in enumerate(self.kpts):
                rho += (
                    (
                        self.get_evecs(ik)
                        * fermi(
                            self.evals[ik], self.efermi, width=self.fermi_width, nspin=2
                        )
                    )
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
                fermi(self.evals[ik], self.efermi, width=self.fermi_width, nspin=2),
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
        if self._use_cache:
            # If using cache, self.evecs is a memmap.
            # We can still use it, or we might want to be careful.
            # But eigen_to_G should handle it.
            return eigen_to_G(self.evals, self.evecs, self.efermi, energy)
        else:
            return eigen_to_G(self.evals, self.evecs, self.efermi, energy)

    def compute_GR(self, Rpts, kpts, Gks):
        Rvecs = np.array(Rpts)
        phase = np.exp(self.k2Rfactor * np.einsum("ni,mi->nm", Rvecs, kpts))
        phase *= self.kweights[None]
        GR = np.einsum("kij,rk->rij", Gks, phase, optimize="optimal")
        return GR

    def get_GR(self, Rpts, energy, Gk_all=None):
        """calculate real space Green's function for one energy, all R points.
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

    def get_GR_and_dGRdx_from_epw(
        self, Rpts, Rjlist, energy, epc, Ru, cutoff=4.0, J_only=False
    ):
        """
        calculate G(R) and dG(R)/dx using k-space multiplication.
        dG(k)/dx =  G(k) (dH(k)/dx) G(k).
        dG(R)/dx = \\sum_k dG(k)/dx * e^{-ikR}
        """
        Rpts = [tuple(R) for R in Rpts]

        # 1. Compute G(k) for all k-points
        Gk_all = self.get_Gk_all(energy)  # (nk, nb, nb)

        # 2. Compute G(R) for requested points
        GR_array = self.compute_GR(Rpts, self.kpts, Gk_all)
        GR = {R: G for R, G in zip(Rpts, GR_array)}

        # 3. Compute rho(R=0)
        # Note: compute_GR includes kweights, so GR[(0,0,0)] is effectively rho if integrated?
        # But rhoR definition in original code:
        # rhoR[R] += rhok * (phase * self.kweights[ik])
        # And rhok includes skewness if not orthogonal.
        # For simplicity, we stick to original logic for rhoR
        rhoR = defaultdict(lambda: 0.0j)
        if (0, 0, 0) in GR:
            # This is just an approximation if non-orthogonal, but for now let's rely on compute_GR
            # logic matching the G contribution.
            # The original code calculates rhoR specifically for (0,0,0).
            pass

        # Re-implement rhoR specific logic to match original behavior exactly for density
        for ik, Gk in enumerate(Gk_all):
            if self.is_orthogonal:
                rhok = Gk
            else:
                rhok = self.get_Sk(ik) @ Gk
            # phase is 1 for R=0
            rhoR[(0, 0, 0)] += rhok * self.kweights[ik]

        dGRijdx = defaultdict(lambda: 0.0 + 0j)
        dGRjidx = defaultdict(lambda: 0.0 + 0j)
        dGRij_array = None
        dGRji_array = None

        if not J_only:
            # Use real-space dGR calculation (matches reference TB2J_spinphon)
            # This avoids the 3x overcounting issue in k-space Lambda approach
            dGRijdx, dGRjidx = self.get_dGR(GR, Rpts, Rjlist, epc, Ru, cutoff=cutoff)

            # Convert to arrays for compatibility with vectorized code
            dGRij_array = np.array([dGRijdx[R] for R in Rpts])
            dGRji_array = np.array([dGRjidx[R] for R in Rpts])

        return GR, dGRijdx, dGRjidx, rhoR, GR_array, dGRij_array, dGRji_array

    def get_dGR(self, GR, Rpts, Rjlist, epc: EpmatOneMode, Ru, cutoff=4.0, diag=False):
        Rpts = [tuple(R) for R in Rpts]
        Rset = set(Rpts)

        if self._Rmap is None:
            self._Rmap = []
            self._Rmap_rev = []
            counter = 0
            for Rj in Rjlist:
                for Rq in epc.Rqdict:
                    for Rk in epc.Rkdict:
                        if np.linalg.norm(Rk) < cutoff:
                            Rm = tuple(np.array(Ru) - np.array(Rq))
                            Rn = tuple(np.array(Rm) + np.array(Rk))
                            Rnj = tuple(np.array(Rj) - np.array(Rn))
                            if Rm in Rset and Rnj in Rset:
                                counter += 1
                                self._Rmap.append((Rq, Rk, Rm, Rnj, Rj))

            counter = 0
            for Rj in Rjlist:
                for Rq in epc.Rqdict:
                    for Rk in epc.Rkdict:
                        if np.linalg.norm(Rk) < cutoff:
                            Rn = tuple(np.array(Ru) - np.array(Rq))
                            Rm = tuple(np.array(Rn) + np.array(Rk))
                            Rjn = tuple(np.array(Rn) - np.array(Rj))
                            Rmi = tuple(-np.array(Rm))
                            if Rmi in Rset and Rjn in Rset:
                                counter += 1
                                self._Rmap_rev.append((Rq, Rk, Rjn, Rmi, Rj))

        dGRdxij = defaultdict(lambda: 0.0 + 0j)
        dGRdxji = defaultdict(lambda: 0.0 + 0j)
        for Rq, Rk, Rm, Rnj, Rj in self._Rmap:
            if diag:
                dV = np.diag(np.diag(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False)))
            else:
                dV = epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T
            dG = GR[Rm] @ dV @ GR[Rnj]
            dGRdxij[Rj] += dG

        for Rq, Rk, Rjn, Rmi, Rj in self._Rmap_rev:
            if diag:
                dV = np.diag(np.diag(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False)))
            else:
                dV = epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T
            dG = GR[Rjn] @ dV @ GR[Rmi]
            dGRdxji[Rj] += dG

        return dGRdxij, dGRdxji

    def get_dGR_vectorized(self, GR, Rpts, Rjlist, epc: EpmatOneMode, Ru, diag=False):
        """
        Vectorized version of get_dGR using einsum for performance.
        No cutoff is applied - processes all Rk vectors.
        """
        Rpts = [tuple(R) for R in Rpts]
        Rset = set(Rpts)

        if self._Rmap is None:
            self._build_Rmaps(Rpts, Rset, Rjlist, epc, Ru)

        nbasis = list(GR.values())[0].shape[0]
        dGRdxij = defaultdict(lambda: np.zeros((nbasis, nbasis), dtype=complex))
        dGRdxji = defaultdict(lambda: np.zeros((nbasis, nbasis), dtype=complex))

        if len(self._Rmap) > 0:
            Rj_groups = defaultdict(list)
            for entry in self._Rmap:
                Rq, Rk, Rm, Rnj, Rj = entry
                Rj_groups[Rj].append((Rq, Rk, Rm, Rnj))

            for Rj, entries in Rj_groups.items():
                n_entries = len(entries)
                GRm_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)
                GRnj_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)
                dV_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)

                for idx, (Rq, Rk, Rm, Rnj) in enumerate(entries):
                    GRm_array[idx] = GR[Rm]
                    GRnj_array[idx] = GR[Rnj]
                    if diag:
                        dV = np.diag(
                            np.diag(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False))
                        )
                    else:
                        dV = epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T
                    dV_array[idx] = dV

                dGRdxij[Rj] = np.einsum(
                    "nij,njk,nkl->il", GRm_array, dV_array, GRnj_array
                )

        if len(self._Rmap_rev) > 0:
            Rj_groups = defaultdict(list)
            for entry in self._Rmap_rev:
                Rq, Rk, Rjn, Rmi, Rj = entry
                Rj_groups[Rj].append((Rq, Rk, Rjn, Rmi))

            for Rj, entries in Rj_groups.items():
                n_entries = len(entries)
                GRjn_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)
                GRmi_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)
                dV_array = np.zeros((n_entries, nbasis, nbasis), dtype=complex)

                for idx, (Rq, Rk, Rjn, Rmi) in enumerate(entries):
                    GRjn_array[idx] = GR[Rjn]
                    GRmi_array[idx] = GR[Rmi]
                    if diag:
                        dV = np.diag(
                            np.diag(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False))
                        )
                    else:
                        dV = epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T
                    dV_array[idx] = dV

                dGRdxji[Rj] = np.einsum(
                    "nij,njk,nkl->il", GRjn_array, dV_array, GRmi_array
                )

        return dGRdxij, dGRdxji

    def _build_Rmaps(self, Rpts, Rset, Rjlist, epc, Ru):
        """Helper to build R-space mapping arrays for vectorization."""
        self._Rmap = []
        self._Rmap_rev = []

        counter = 0
        for Rj in Rjlist:
            for Rq in epc.Rqdict:
                for Rk in epc.Rkdict:
                    Rm = tuple(np.array(Ru) - np.array(Rq))
                    Rn = tuple(np.array(Rm) + np.array(Rk))
                    Rnj = tuple(np.array(Rj) - np.array(Rn))
                    if Rm in Rset and Rnj in Rset:
                        counter += 1
                        self._Rmap.append((Rq, Rk, Rm, Rnj, Rj))

        # print(f"ij path entries: {counter}")

        counter = 0
        for Rj in Rjlist:
            for Rq in epc.Rqdict:
                for Rk in epc.Rkdict:
                    Rn = tuple(np.array(Ru) - np.array(Rq))
                    Rm = tuple(np.array(Rn) + np.array(Rk))
                    Rjn = tuple(np.array(Rn) - np.array(Rj))
                    Rmi = tuple(-np.array(Rm))
                    if Rmi in Rset and Rjn in Rset:
                        counter += 1
                        self._Rmap_rev.append((Rq, Rk, Rjn, Rmi, Rj))

        # print(f"ji path entries: {counter}")

    def get_GR_and_dGRdx_from_epw_vectorized(self, Rpts, Rjlist, energy, epc, Ru):
        """
        Vectorized version of get_GR_and_dGRdx_from_epw without cutoff.
        """
        Rpts = [tuple(R) for R in Rpts]
        GR = defaultdict(lambda: 0.0 + 0.0j)
        rhoR = defaultdict(lambda: 0.0j)

        for ik, kpt in enumerate(self.kpts):
            Gk = self.get_Gk(ik, energy)
            if self.is_orthogonal:
                rhok = Gk
            else:
                rhok = self.get_Sk(ik) @ Gk
            Gkp = Gk * self.kweights[ik]
            for R in Rpts:
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                GR[R] += Gkp * phase
                if R == (0, 0, 0):
                    rhoR[R] += rhok * (phase * self.kweights[ik])

        dGRijdx, dGRjidx = self.get_dGR_vectorized(
            GR, Rpts, Rjlist, epc, Ru, diag=False
        )

        return GR, dGRijdx, dGRjidx, rhoR
