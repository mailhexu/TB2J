import numpy as np
import scipy.linalg as sl
from collections import defaultdict
from ase.dft.kpoints import monkhorst_pack
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

class TBGreen():
    def __init__(
            self,
            tbmodel,
            kmesh,  # [ikpt, 3]
            efermi,  # efermi
            k_sym=False):
        """
        :param tbmodel: A tight binding model
        :param kmesh: size of monkhorst pack. e.g [6,6,6]
        :param efermi: fermi energy.
        """
        self.tbmodel = tbmodel
        self.R2kfactor=tbmodel.R2kfactor
        self.k2Rfactor=-tbmodel.R2kfactor
        self.efermi = efermi
        if kmesh is not None:
            self.kpts = monkhorst_pack(size=kmesh)
        else:
            self.kpts = tbmodel.get_kpts()
        self.nkpts = len(self.kpts)
        self.kweights = [1.0 / self.nkpts] * self.nkpts
        self.norb = tbmodel.norb
        self.nbasis = tbmodel.nbasis
        self.k_sym=k_sym
        self._prepare_eigen()

    def _prepare_eigen(self):
        """
        calculate eigen values and vectors for all kpts and save.
        Note that the convention 2 is used here, where the 
        phase factor is e^(ik.R), not e^(ik.(R+rj-ri))
        """
        # self.evals = np.zeros((len(self.kpts), self.nbasis), dtype=complex)
        # self.evecs = np.zeros((len(self.kpts), self.nbasis, self.nbasis),
        #                       dtype=complex)
        # self.H0 = 0.0
        # for ik, k in enumerate(self.kpts):
        #     Hk = self.tbmodel.gen_ham(tuple(k))
        #     self.evals[ik, :], self.evecs[ik, :, :] =self.tbmodel.solve(tuple(k))
        #     self.H0 += Hk / len(self.kpts)
        H,S,self.evals, self.evecs=self.tbmodel.HS_and_eigen(self.kpts)
        self.H0=np.sum(H, axis=0)/len(self.kpts)
        if S is not None:
            self.is_orthogonal=False
            self.S=S
            self.S0=np.sum(S, axis=0)/len(self.kpts)
        else:
            self.is_orthogonal=True

    def get_Gk(self, ik, energy):
        """ Green's function G(k) for one energy
        G(\epsilon)= (\epsilon I- H)^{-1}
        :param ik: indices for kpoint
        :returns: Gk
        :rtype:  a matrix of indices (nbasis, nbasis)
        """
        Gk = eigen_to_G(
            evals=self.evals[ik, :],
            evecs=self.evecs[ik, :, :],
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
                    rhok=self.S[ik]@Gk
            for iR, R in enumerate(Rpts):
                phase = np.exp(self.k2Rfactor * np.dot(R, kpt))
                tmp=Gk * (phase * self.kweights[ik])
                GR[R] += tmp
                if get_rho:
                    rhoR[R]+=rhok*(phase * self.kweights[ik])
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
                GR[R] += Gkp * (phase*self.kweights[ik])
                dGRdx[R] += dG * (phase*self.kweights[ik])
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
                GR[R] += Gkp * (phase *self.kweights[ik])
                dGRdx[R] += dG * (phase*self.kweights[ik])
                dGRdx2[R] += dG2 * (phase*self.kweights[ik])
        return GR, dGRdx, dGRdx2


