"""
This module calculate Oiju from electron-phonon coupling matrix and Electron Wannier function.
"""

import numpy as np
import progressbar
from .exchangeCL2 import ExchangeCL2
from collections import defaultdict


class OijuWannEPC(ExchangeCL2):
    def __init__(
            self,
            tbmodels,
            atoms,
            efermi,
            basis=None,
            magnetic_elements=[],
            kmesh=[4, 4, 4],
            emin=-15,  # integration lower bound, relative to fermi energy
            emax=0.05,  # integration upper bound. Should be 0 (fermi energy). But DFT codes define Fermi energy in various ways.
            height=0.5,  # the delta in the (i delta) in green's function to prevent divergence
            nz1=150,  # grid from emin to emin+(i delta)
            nz2=300,  # grid from emin +(i delta) to emax+(i delta)
            nz3=150,  # grid from emax + (i delta) to emax
            exclude_orbs=[],  #
            Rmesh=[0, 0, 0],  # Rmesh.
            description='',
            qmesh=[2, 2, 2],
            EPCmats=None,
            WannUmats=None):
        super(self).__init__(
            tbmodels,
            atoms,
            efermi,
            basis=None,
            magnetic_elements=[],
            kmesh=[4, 4, 4],
            emin=-15,  # integration lower bound, relative to fermi energy
            emax=0.05,  # integration upper bound. Should be 0 (fermi energy).
            # But DFT codes define Fermi energy in various ways.
            height=0.5,  # the delta in the (i delta) in green's function.
            nz1=150,  # grid from emin to emin+(i delta)
            nz2=300,  # grid from emin +(i delta) to emax+(i delta)
            nz3=150,  # grid from emax + (i delta) to emax
            exclude_orbs=[],  #
            Rmesh=[0, 0, 0],  # Rmesh.
            description='',
        )

        # prepare EPC in electron wannier space
        self.EPCmat_up, self.EPCmat_dn = EPCmats
        self.Umat_up, self.Umat_dn = WannUmats
        self.prepare_epc_wann()

        # prepare dDelta
        self.calc_dDelta()

    def prepare_epc_wann(self):
        """
        prepare EPC in electron wannier function representation
        """
        self.EPCmat_wann_up = self.EPCmat_up.to_wann(self.Umat_up)
        self.EPCmat_wann_dn = self.EPCmat_dn.to_wann(self.Umat_dn)

    def calc_dDelta(self):
        """
        calculate $\delta \Delta$.
        <m 0| dV/d \tau(qV) | n 0>. Since only 0->0, q=0)

        \delta\Delta (q, v) = \sum_k w(k) e^{ik R=0} U\dagger (k+q) g(q,v) U(k)
                     =\sum_k w(k) U\dagger g(0, v) U(k)

        For q/=0. \delta\Delta=0.
        the result will be a matrix[nphon, nwann, nwann]
        """
        self.dDelta = np.zeros((self.nphon, self.nwann, self.nwann))
        #iq, iv, iR
        iR0 = iRlist[(0, 0, 0)]
        iq0 = iqlist[(0, 0, 0)]

        for iv in range(self.nphon):
            for ik, k in enumerate(self.klist):
                self.dDelta[iv] += self.kweight[ik] * (
                    self.EPCmat_wann_up[iq0, iv, ik, :, :] -
                    self.EPCmat_wann_dn[iq0, iv, ik, :, :])

    def deltaG_kvq(self, Gk, Umat, EPCmat):
        pass

    def get_Oijvq(self, GR_up, GR_dn, dGR_vq_up, dGR_vq_dn, iatom, jatom, Rj):
        """
        calculate Oiju(i, j, Rj, v, q)
        """
        iorbs = self.iorb(iatom)
        jorbs = self.iorb(jatom)

        self.get_Delta(iatom) @ dGR_vq_up[np.ix_(
            iorbs, jorbs)] @ self.get_Delta(jatom) @ GR_dn

    def _build_Rjdict(self, Rjlist):
        for iRj, Rj in enumerate(Rjlist):
            self.Rudict[tuple(Rj)] = iRj

    def _build_Rudict(self, Rulist):
        for iRu, Ru in enumerate(Rulist):
            self.Rudict[tuple(Ru)] = iRu

    def iRu(self, Ru):
        return self.Rudict[tuple(Ru)]

    def iRj(self, Rj):
        return self.Rjdict[tuple(Rj)]

    def Oij_vq(self, Rj_list, energy):
        GR_up = defaultdict(lambda: 0.0)
        GR_dn = defaultdict(lambda: 0.0)
        dGR_vq_up = defaultdict(lambda: 0.0)
        dGR_vq_dn = defaultdict(lambda: 0.0)
        self.Oijvq = defaultdict(lambda: 0.0)

        for iq, q in enumerate(self.qmesh):
            for v in self.vlist:
                # calculate dG(i,j, Rj, v, q)
                for ik, k in enumerate(self.klist):
                    # Gk
                    Gk_up = self.Gup.get_Gk(ik, energy)
                    Gk_dn = self.Gdn.get_Gk(ik, energy)

                    # GR
                    for iR, Rj in enumerate(Rj_list):
                        Rj = np.array(Rj)
                        phase = np.exp(-2j * np.pi * np.dot(Rj, k))
                        GR_up[self.iRj(
                            Rj)] += Gk_up * (phase * self.kweights[ik])
                        GR_dn[self.iRj(
                            Rj)] += Gk_dn * (phase * self.kweights[ik])

                    # deltaG(k, v, q, energy)
                    dGk_vq_up = self.deltaG_kvq(Gk_up, self.EPCmat_wann_up)
                    dGk_vq_dn = self.deltaG_kvq(Gk_dn, self.EPCmat_wann_dn)

                    for iR, Rj in enumerate(Rj_list):
                        phase = np.exp(-2j * np.pi * np.dot(Rj, k))
                        # nwann*nwann matrix
                        dGR_vq_up[
                            Rj] += dGk_vq_up * (phase * self.kweights[ik])
                        dGR_vq_dn[
                            Rj] += dGk_vq_dn * (phase * self.kweights[ik])

                    for iatom in self.iatom_list:
                        for jatom in self.iatom_list:
                            self.Oijvq[(iatom, jatom, Rj, v,
                                        q)] += self.get_Oijvq(
                                            GR_up,
                                            GR_dn,
                                            dGR_vq_up,
                                            dGR_vq_dn,
                                            iatom,
                                            jatom,
                                            Rj,
                                        )
                            for iRu, Ru in enumerate(self.Rulist):
                                phase_Ru = np.exp(-2j * np.pi * np.dot(q, Ru))
                                # u=v for atomic displacement.
                                self.Oiju[(iatom, jatom, Rj, v,
                                           Ru)] += phase_Ru * self.Oijvq[(
                                               iatom, jatom, Rj, v, q)]

    def calculate_all(self):
        print("Green's function Calculation started.")
        widgets = [
            ' [',
            progressbar.Timer(),
            '] ',
            progressbar.Bar(),
            ' (',
            progressbar.ETA(),
            ') ',
        ]
        bar = progressbar.ProgressBar(
            maxval=len(self.elist) - 1, widgets=widgets)
        bar.start()
        for ie in range(len(self.elist[:-1])):
            bar.update(ie)
            em, en = self.elist[ie], self.elist[ie + 1]
            e = (em + en) / 2
            de = en - em
            # Green's function for G(i, j, Rj)
            GR_up = self.Gup.get_GR(self.Rlist, energy=e)
            GR_dn = self.Gdn.get_GR(self.Rlist, energy=e)
            self.get_rho_e(GR_up, GR_dn, de)
            self.get_all_A(GR_up, GR_dn, de)
        self.get_rho_atom()
        self.A_to_Jtensor()
        bar.finish()

    def run(self):
        pass
