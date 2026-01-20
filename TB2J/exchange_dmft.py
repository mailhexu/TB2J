from collections import defaultdict
from itertools import product

import numpy as np

from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.external import p_map
from TB2J.green import TBGreen
from TB2J.pauli import pauli_block_all, pauli_block_sigma_norm


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
        self.dmft_mesh = self.tbmodel.mesh
        iw0 = self.dmft_mesh[0]
        inv_beta = (2 * 0 + 1) * np.pi / iw0.imag if iw0.imag > 0 else 1.0
        self.temperature = 1.0 / inv_beta
        self.beta = inv_beta

        # Initialize standard Exchange attributes
        self.rho = 0.0
        self.A_ijR_list = defaultdict(lambda: [])
        self.A_ijR = {}
        self.A_ijR_orb = dict()
        self.HR0 = self.G.H0
        self.Pdict = {}
        if self.write_density_matrix:
            self.G.write_rho_R(self.short_Rlist)

    def _prepare_elist(self, method="matsubara"):
        if method.lower() != "matsubara":
            raise ValueError("DMFT calculations only support Matsubara mesh.")
        from TB2J.contour import MatsubaraContour

        self.contour = MatsubaraContour(self.dmft_mesh, self.beta)

    def integrate(self, AijRs, AijRs_orb=None, rho_list=None, method="matsubara"):
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
        if rho_list is not None:
            self.rho = np.sum(rho_list, axis=0) * T

    def get_rho_atom(self):
        """
        calculate charge and spin for each atom from integrated rho.
        """
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            if getattr(self, "_is_collinear", False):
                # rho shape is (2, nbasis, nbasis)
                tup = np.real(np.trace(self.rho[0][np.ix_(iorb, iorb)]))
                tdn = np.real(np.trace(self.rho[1][np.ix_(iorb, iorb)]))
                self.charges[iatom] = tup + tdn
                self.spinat[iatom, 2] = tup - tdn
            else:
                # rho shape is (nbasis, nbasis)
                tmp = self.rho[np.ix_(iorb, iorb)]
                from TB2J.pauli import pauli_block_all

                # *2 because there is a 1/2 in the pauli_block_all function
                rho_pauli = np.array(
                    [np.trace(x) * 2 for x in pauli_block_all(tmp)]
                ).real
                self.charges[iatom] = rho_pauli[0]
                self.spinat[iatom, :] = rho_pauli[1:]
        return self.charges, self.spinat

    def calculate_all(self):
        """
        The top level for DMFT exchange.
        """
        print("Green's function Calculation started.")
        AijRs = {}
        AijRs_orb = {}
        rho_list = []

        npole = len(self.contour.path)
        if self.nproc > 1:
            results = p_map(
                self.get_quantities_per_e, self.contour.path, num_cpus=self.nproc
            )
        else:
            from tqdm import tqdm

            results = map(
                self.get_quantities_per_e, tqdm(self.contour.path, total=npole)
            )

        for i, result in enumerate(results):
            rho_list.append(result["rho_e"])
            for iR in self.R_ijatom_dict:
                R_vec = self.short_Rlist[iR]
                for iatom, jatom in self.R_ijatom_dict[iR]:
                    if (R_vec, iatom, jatom) in AijRs:
                        AijRs[(R_vec, iatom, jatom)].append(
                            result["AijR"][(R_vec, iatom, jatom)]
                        )
                        if self.orb_decomposition:
                            AijRs_orb[(R_vec, iatom, jatom)].append(
                                result["AijR_orb"][(R_vec, iatom, jatom)]
                            )
                    else:
                        AijRs[(R_vec, iatom, jatom)] = [
                            result["AijR"][(R_vec, iatom, jatom)]
                        ]
                        if self.orb_decomposition:
                            AijRs_orb[(R_vec, iatom, jatom)] = [
                                result["AijR_orb"][(R_vec, iatom, jatom)]
                            ]

        self.integrate(AijRs, AijRs_orb, rho_list=rho_list)
        self.get_rho_atom()
        self.A_to_Jtensor()
        # self.A_to_Jtensor_orb() # TODO: check if this works for DMFT

    def get_P_iatom_e(self, iatom, e):
        """
        Calculate frequency-dependent magnetic splitting P(e).
        P(e) = pauli_block_sigma_norm(H_local + Sigma_local(e))
        """
        # H_local (static)
        orbs = self.iorb(iatom)
        H0_local = self.HR0[np.ix_(orbs, orbs)]

        # Sigma_local(e). Pass ispin=None to get both spins if available.
        try:
            sigma = self.tbmodel.get_sigma(e, ispin=None)
        except TypeError:
            sigma = self.tbmodel.get_sigma(e)

        # sigma shape is (n_spin, n_orb, n_orb) or (n_spinor, n_spinor)

        if sigma.ndim == 3:  # Spin-polarized or NCL
            n_atom_orb = len(orbs)
            if getattr(self, "_is_collinear", False):
                # Collinear: return (H_up + Sigma_up - (H_dn + Sigma_dn)) / 2
                # Note: sigma[0] is up, sigma[1] is dn
                if H0_local.shape[0] == 2 * n_atom_orb:
                    # Static model is already spinor/spin-polarized
                    MI, Mx, My, Mz = pauli_block_all(H0_local)
                else:
                    # Static model is non-polarized (single H)
                    Mz = np.zeros((n_atom_orb, n_atom_orb), dtype=complex)
                sigma_up = sigma[0, orbs][:, orbs]
                sigma_dn = sigma[1, orbs][:, orbs]
                return Mz + (sigma_up - sigma_dn) / 2.0
            else:
                # NCL: reconstruct spinor sigma from up/down if needed
                if sigma.shape[0] == 2:
                    # Construct diagonal spinor sigma
                    sigma_full = np.zeros(
                        (2 * n_atom_orb, 2 * n_atom_orb), dtype=complex
                    )
                    sigma_full[:n_atom_orb, :n_atom_orb] = sigma[0, orbs][:, orbs]
                    sigma_full[n_atom_orb:, n_atom_orb:] = sigma[1, orbs][:, orbs]
                    # If H0_local is already spinor
                    if H0_local.shape[0] == 2 * n_atom_orb:
                        return pauli_block_sigma_norm(H0_local + sigma_full)
                    else:
                        # Static H0 is single, we need to kron it with identity
                        H_full = np.kron(np.eye(2), H0_local)
                        return pauli_block_sigma_norm(H_full + sigma_full)
                else:
                    # Already spinor or single spin
                    return pauli_block_sigma_norm(H0_local + sigma[0, orbs][:, orbs])
        else:
            # Not spin polarized
            return pauli_block_sigma_norm(H0_local + sigma[orbs][:, orbs])

    def get_all_A_collinear(self, Gup, Gdn):
        """
        Collinear version of get_all_A for DMFT.
        """
        Jorb_list = dict()
        JJ_list = dict()
        for iR in self.R_ijatom_dict:
            for iatom, jatom in self.R_ijatom_dict[iR]:
                iRm = self.R_negative_index[iR]
                Gij_up = Gup[iR][np.ix_(self.iorb(iatom), self.iorb(jatom))]
                Gji_dn = Gdn[iRm][np.ix_(self.iorb(jatom), self.iorb(iatom))]
                # Delta = 2 * P
                Deltai = 2.0 * self.Pdict[iatom]
                Deltaj = 2.0 * self.Pdict[jatom]
                t = np.einsum(
                    "ij, ji-> ij", np.matmul(Deltai, Gij_up), np.matmul(Deltaj, Gji_dn)
                )
                tmp = np.sum(t)
                R_vec = self.short_Rlist[iR]
                # Pre-factor 1/4 from formula
                Jorb_list[(R_vec, iatom, jatom)] = t / 4.0
                JJ_list[(R_vec, iatom, jatom)] = tmp / 4.0
        return JJ_list, Jorb_list

    def get_all_A_vectorized_collinear(self, Gup, Gdn):
        """
        Vectorized collinear version of get_all_A for DMFT.
        """
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]
        Delta = [2.0 * self.Pdict[site] for site in magnetic_sites]

        A = {}
        A_orb = {}
        for i, j in product(range(len(magnetic_sites)), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]
            Gij = Gup[:, idx][:, :, jdx]
            Gji = np.flip(Gdn[:, jdx][:, :, idx], axis=0)

            t_tensor = (
                np.einsum(
                    "ab,rbc,cd,rda->rac",
                    Delta[i],
                    Gij,
                    Delta[j],
                    Gji,
                    optimize="optimal",
                )
                / 4.0
            )
            tmp_tensor = np.sum(t_tensor, axis=(1, 2))

            mi, mj = (magnetic_sites[i], magnetic_sites[j])
            for iR, R_vec in enumerate(self.short_Rlist):
                A[(R_vec, mi, mj)] = tmp_tensor[iR]
                if self.orb_decomposition:
                    A_orb[(R_vec, mi, mj)] = t_tensor[iR]
        return A, A_orb

    def get_quantities_per_e(self, e):
        """
        Override to use frequency-dependent P and handle collinear/NCL.
        """
        if getattr(self, "_is_collinear", False):
            # For collinear DMFT, we need G_up and G_dn separately
            Gk_up = self.G.get_Gk_all(e, ispin=0)
            Gk_dn = self.G.get_Gk_all(e, ispin=1)
            GR_up = self.G.get_GR(self.short_Rlist, energy=e, Gk_all=Gk_up)
            GR_dn = self.G.get_GR(self.short_Rlist, energy=e, Gk_all=Gk_dn)

            # Density matrix contribution
            rho_e = np.zeros((2, self.nbasis, self.nbasis), dtype=complex)
            rho_e[0] = GR_up[self.Rvec_to_shortlist_idx[(0, 0, 0)]]
            rho_e[1] = GR_dn[self.Rvec_to_shortlist_idx[(0, 0, 0)]]

            # Update Pdict for this specific frequency
            self.Pdict = {}
            for iatom in self.ind_mag_atoms:
                self.Pdict[iatom] = self.get_P_iatom_e(iatom, e)

            try:
                AijR, AijR_orb = self.get_all_A_vectorized_collinear(GR_up, GR_dn)
            except Exception:
                AijR, AijR_orb = self.get_all_A_collinear(GR_up, GR_dn)
            return dict(AijR=AijR, AijR_orb=AijR_orb, mae=None, rho_e=rho_e)
        else:
            # NCL case
            Gk_all = self.G.get_Gk_all(e)
            GR = self.G.get_GR(self.short_Rlist, energy=e, Gk_all=Gk_all)

            # Update Pdict for this specific frequency
            self.Pdict = {}
            for iatom in self.ind_mag_atoms:
                self.Pdict[iatom] = self.get_P_iatom_e(iatom, e)

            try:
                AijR, AijR_orb = self.get_all_A_vectorized(GR)
            except Exception:
                AijR, AijR_orb = self.get_all_A(GR)
            return dict(
                AijR=AijR,
                AijR_orb=AijR_orb,
                mae=None,
                rho_e=GR[self.Rvec_to_shortlist_idx[(0, 0, 0)]],
            )

    def A_to_Jtensor(self):
        if getattr(self, "_is_collinear", False):
            self.exchange_Jdict = {}
            self.Jiso_orb = {}
            self.B = {}
            for key, val in self.A_ijR.items():
                R, iatom, jatom = key
                is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                if is_nonself:
                    ispin = self.ispin(iatom)
                    jspin = self.ispin(jatom)
                    keyspin = (R, ispin, jspin)
                    sign = np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
                    if sign == 0:
                        sign = 1.0
                    Jij = np.imag(val) / sign
                    self.exchange_Jdict[keyspin] = Jij
                    if self.orb_decomposition:
                        Jorbij = np.imag(self.A_ijR_orb[key]) / sign
                        self.Jiso_orb[keyspin] = self.simplify_orbital_contributions(
                            Jorbij, iatom, jatom
                        )
        else:
            super().A_to_Jtensor()

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
