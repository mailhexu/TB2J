from collections import defaultdict

import numpy as np

from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.green import TBGreen


class ExchangeDMFTMixin:
    """
    Mixin for DMFT exchange calculations.
    Provides common methods for Matsubara integration and temperature handling.
    """

    def set_tbmodels(self, tbmodels):
        """
        tbmodels should be a TBModelDMFT instance.
        """
        self.tbmodel = tbmodels
        self.backend_name = "DMFT"

        # Initialize TBGreen
        self.G = TBGreen(
            tbmodel=self.tbmodel,
            kmesh=self.kmesh,
            ibz=self.ibz,
            gamma=True,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
            initial_emin=int(self.emin),
        )

        if self.efermi is None:
            self.efermi = self.G.efermi
        else:
            self.efermi = float(self.efermi)
        self.norb = self.G.norb
        self.nbasis = self.G.nbasis

        # DMFT specific: Get temperature from mesh
        _, self.dmft_mesh = self.tbmodel.mesh
        iw0 = self.dmft_mesh[0]
        inv_beta = (2 * 0 + 1) * np.pi / iw0.imag if iw0.imag > 0 else 1.0
        self.temperature = 1.0 / inv_beta
        self.beta = inv_beta

        # Initialize standard Exchange attributes
        self.rho = self.G.get_density_matrix()
        self.A_ijR_list = defaultdict(lambda: [])
        self.A_ijR = {}
        self.A_ijR_orb = dict()
        self.HR0 = self.G.H0
        self.Pdict = {}
        if self.write_density_matrix:
            self.G.write_rho_R()

    def _prepare_elist(self, method="matsubara"):
        if method.lower() != "matsubara":
            raise ValueError("DMFT calculations only support Matsubara mesh.")
        self.contour = self.dmft_mesh

    def integrate(self, AijRs, AijRs_orb=None, method="matsubara"):
        T = self.temperature
        for iR in self.R_ijatom_dict:
            R_vec = self.short_Rlist[iR]
            for iatom, jatom in self.R_ijatom_dict[iR]:
                f = AijRs[(R_vec, iatom, jatom)]
                integral = np.sum(f) * T
                self.A_ijR[(R_vec, iatom, jatom)] = integral
                if self.orb_decomposition:
                    self.A_ijR_orb[(R_vec, iatom, jatom)] = (
                        np.sum(AijRs_orb[(R_vec, iatom, jatom)], axis=0) * T
                    )

    def finalize(self):
        self.G.clean_cache()

    def run(self, path="TB2J_results"):
        self.calculate_all()
        self.write_output(path=path)
        self.finalize()


class ExchangeDMFTNCL(ExchangeDMFTMixin, ExchangeNCL):
    """
    Non-collinear Exchange calculation using DMFT self-energies.
    """

    def set_tbmodels(self, tbmodels):
        ExchangeDMFTMixin.set_tbmodels(self, tbmodels)
        self._is_collinear = False


class ExchangeCLDMFT(ExchangeDMFTMixin, ExchangeCL):
    """
    Collinear Exchange calculation using DMFT self-energies.
    """

    def set_tbmodels(self, tbmodels):
        ExchangeDMFTMixin.set_tbmodels(self, tbmodels)
        self._is_collinear = True
