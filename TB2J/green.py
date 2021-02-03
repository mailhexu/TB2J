import numpy as np
import scipy.linalg as sl
from collections import defaultdict
from ase.dft.kpoints import monkhorst_pack
from shutil import rmtree
import os


def eigen_to_G(evals, evecs, efermi, energy):
    """ calculate green's function from eigenvalue/eigenvector for energy(e-ef): G(e-ef).
    :param evals:  eigen values
    :param evecs:  eigen vectors
    :param efermi: fermi energy
    :param energy: energy
    :returns: Green's function G,
    :rtype:  Matrix with same shape of the Hamiltonian (and eigenvector)
    """
    return evecs.dot(np.diag(1.0 / (-evals + (energy + efermi)))).dot(
        evecs.conj().T)

def find_energy_ingap(evals, rbound, gap=1.0):
    """
    find a energy inside a gap below rbound (right bound), 
    return the energy gap top - 0.5.
    """
    m=np.sort(evals.flatten())
    m=m[m<rbound]
    ind=np.where(np.diff(m)>gap)[0]
    if len(ind)==0:
        return m[0]-0.5
    else:
        return m[ind[-1]+1]-0.5

class TBGreen():
    def __init__(
        self,
        tbmodel,
        kmesh,  # [ikpt, 3]
        efermi,  # efermi
        k_sym=False,
        use_cache=False,
        cache_path='TB2J_results/cache'):
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
        if kmesh is not None:
            self.kpts = monkhorst_pack(size=kmesh)
        else:
            self.kpts = tbmodel.get_kpts()
        self.nkpts = len(self.kpts)
        self.kweights = [1.0 / self.nkpts] * self.nkpts
        self.norb = tbmodel.norb
        self.nbasis = tbmodel.nbasis
        self.k_sym = k_sym
        self._prepare_eigen()

    def _reduce_eigens(self, evals, evecs, emin, emax):
        ts = np.logical_and(evals >= emin, evals < emax)
        ts = np.any(ts, axis=0)
        ts = np.where(ts)[0]
        istart, iend = ts[0], ts[-1] + 1
        print(f"reducing nbands from {evals.shape[1]} to {len(ts)}")
        return evals[:, istart:iend], evecs[:, :, istart:iend]

    def find_energy_ingap(self, rbound, gap=1.0):
        return find_energy_ingap(self.evals, rbound, gap)

    def _prepare_cache(self):
        path = self.cache_path
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            rmtree(path)
            os.makedirs(path)

    def clean_cache(self):
        if os.path.exists(self.cache_path):
            rmtree(self.cache_path)

    def _prepare_eigen(self):
        """
        calculate eigen values and vectors for all kpts and save.
        Note that the convention 2 is used here, where the 
        phase factor is e^(ik.R), not e^(ik.(R+rj-ri))
        """
        nkpts = len(self.kpts)
        self.evals = np.zeros((nkpts, self.nbasis), dtype=float)
        self.nkpts = nkpts
        self.H0 = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        if not self._use_cache:
            #print("Preparing eigen in memory.")
            self.evecs = np.zeros((nkpts, self.nbasis, self.nbasis),
                                  dtype=complex)
            H = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
            if not self.is_orthogonal:
                self.S = np.zeros((nkpts, self.nbasis, self.nbasis),
                                  dtype=complex)
            else:
                self.S = None
            for ik, k in enumerate(self.kpts):
                if self.is_orthogonal:
                    H[ik], _, self.evals[ik], self.evecs[
                        ik] = self.tbmodel.HSE_k(k, convention=2)
                else:
                    H[ik], self.S[ik], self.evals[ik], self.evecs[
                        ik] = self.tbmodel.HSE_k(k, convention=2)
                self.H0 += H[ik] / self.nkpts
            self.evals, self.evecs = self._reduce_eigens(
                self.evals,
                self.evecs,
                emin=self.efermi - 10.0,
                emax=self.efermi + 10.1)
        else:  # Use cache
            #print("Preparing eigen in cache.")
            self.evecs = np.memmap(os.path.join(self.cache_path, 'evecs.dat'),
                                   mode='w+',
                                   shape=(nkpts, self.nbasis, self.nbasis),
                                   dtype=complex)
            H = np.memmap(os.path.join(self.cache_path, 'H.dat'),
                          mode='w+',
                          shape=(nkpts, self.nbasis, self.nbasis),
                          dtype=complex)
            if self.is_orthogonal:
                self.S = None
            else:
                self.S = np.memmap(os.path.join(self.cache_path, 'S.dat'),
                                   mode='w+',
                                   shape=(nkpts, self.nbasis, self.nbasis),
                                   dtype=complex)

            for ik, k in enumerate(self.kpts):
                Hk, Sk, evalue, evec = self.tbmodel.HSE_k(k)
                #self.H[ik] = Hk
                self.H0 += Hk / self.nkpts
                if not self.is_orthogonal:
                    self.S[ik] = Sk
                self.evals[ik] = evalue
                self.evecs[ik] = evec
            del self.evecs
            #del self.H
            if not self.is_orthogonal:
                del self.S

    def get_evecs(self, ik):
        if self._use_cache:
            return np.memmap(os.path.join(self.cache_path, 'evecs.dat'),
                             mode='r',
                             shape=(self.nkpts, self.nbasis, self.nbasis),
                             dtype=complex)[ik]
        else:
            return self.evecs[ik]

    def get_evalue(self, ik):
        return self.evals[ik]

    def get_Hk(self, ik):
        if self._use_cache:
            return np.memmap(os.path.join(self.cache_path, 'H.dat'),
                             mode='r',
                             shape=(self.nkpts, self.nbasis, self.nbasis),
                             dtype=complex)[ik]
        else:
            return self.evecs[ik]

    def get_Sk(self, ik):
        if self.is_orthogonal:
            return None
        elif self._use_cache:
            return np.memmap(os.path.join(self.cache_path, 'S.dat'),
                             mode='r',
                             shape=(self.nkpts, self.nbasis, self.nbasis),
                             dtype=complex)[ik]
        else:
            return self.S[ik]

    def get_Gk(self, ik, energy):
        """ Green's function G(k) for one energy
        G(\epsilon)= (\epsilon I- H)^{-1}
        :param ik: indices for kpoint
        :returns: Gk
        :rtype:  a matrix of indices (nbasis, nbasis)
        """
        Gk = eigen_to_G(evals=self.get_evalue(ik),
                        evecs=self.get_evecs(ik),
                        efermi=self.efermi,
                        energy=energy)

        # A slower version. For test.
        #Gk = np.linalg.inv((energy+self.efermi)*self.S[ik,:,:] - self.H[ik,:,:])
        return Gk

    def get_GR(self, Rpts, energy, get_rho=False):
        """ calculate real space Green's function for one energy, all R points.
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
            Gk = self.get_Gk(ik, energy)
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
            #Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                GR[R] += Gkw * (phase * self.kweights[ik])

                dHRdx = dHdx.get_hamR(R)
                dGRdx[R] += Gkw @ dHRdx @ Gk
                #dGRdx[R] += Gk.dot(dHRdx).dot(Gkp)
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
            #Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
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
            #Gmk = self.get_Gk(self.i_minus_k(kpt), energy)
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
