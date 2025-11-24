from collections import defaultdict
from functools import lru_cache

import numpy as np
from p_tqdm import p_map
from tqdm import tqdm

from TB2J.epwparser import EpmatOneMode
from TB2J.exchange import ExchangeNCL
from TB2J.io_exchange import SpinIO
from TB2J.pauli import pauli_block_all, pauli_block_sigma_norm
from TB2J.utils import simpson_nonuniform, trapezoidal_nonuniform


class ExchangePert2(ExchangeNCL):
    def set_epw(
        self,
        Ru,
        imode=None,
        epmat_up=None,
        epmat_dn=None,
        epmode_up=None,
        epmode_dn=None,
    ):
        # Validate that both spin up and down epmat files are provided
        if (epmat_up is not None) != (epmat_dn is not None):
            raise ValueError("Both epmat_up and epmat_dn must be provided together")
        if (epmode_up is not None) != (epmode_dn is not None):
            raise ValueError("Both epmode_up and epmode_dn must be provided together")
        if (epmat_up is None and epmode_up is None) or (
            epmat_dn is None and epmode_dn is None
        ):
            raise ValueError(
                "Either epmat_up/epmat_dn or epmode_up/epmode_dn must be provided"
            )

        if epmat_up is not None and imode is not None:
            self.epc_up = EpmatOneMode(epmat_up, imode, close_nc=True)
            self.epc_dn = EpmatOneMode(epmat_dn, imode, close_nc=True)
        else:
            self.epc_up = epmode_up
            self.epc_dn = epmode_dn
        self.Ru = Ru

        self.dA_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA2_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA_ijR_orb = {}

    @lru_cache()
    def get_dP_iatom(self, iatom):
        orbs = self.iorb(iatom)
        return pauli_block_sigma_norm(self.dHdxR0[np.ix_(orbs, orbs)])

    def get_A_ijR(self, G, dG, dGrev, R, iatom, jatom):
        """calculate A from G for a energy .
        It take the
        .. math::
           A^{uv} = p T^u p T^v dE / pi

        where u, v are I, x, y, z (index 0, 1,2,3). p(i) = self.get_P_iatom(iatom)
        T^u(ijR)  (u=0,1,2,3) = pauli_block_all(G)

        :param G: Green's function for all R, i, j (spin-resolved).
        :param iatom: i
        :param jatom: j
        :param R:  Rvector o j
        :returns: a matrix of A_ij(u, v), where u, v =(0)0, x(1), y(2), z(3)
        :rtype:  4*4 matrix
        """

        GR_up, GR_dn = G[R]
        Gij_up = self.GR_atom(GR_up, iatom, jatom)
        Gij_dn = self.GR_atom(GR_dn, iatom, jatom)
        Gij_Ixyz_up = pauli_block_all(Gij_up)
        Gij_Ixyz_dn = pauli_block_all(Gij_dn)

        # G(j, i, -R)
        Rm = tuple(-x for x in R)
        GRm_up, GRm_dn = G[Rm]
        Gji_up = self.GR_atom(GRm_up, jatom, iatom)
        Gji_dn = self.GR_atom(GRm_dn, jatom, iatom)
        Gji_Ixyz_up = pauli_block_all(Gji_up)
        Gji_Ixyz_dn = pauli_block_all(Gji_dn)

        dGR_up, dGR_dn = dG[R]
        dGij_up = self.GR_atom(dGR_up, iatom, jatom)
        dGij_dn = self.GR_atom(dGR_dn, iatom, jatom)
        # GijR , I, x, y, z component.
        dGij_Ixyz_up = pauli_block_all(dGij_up)
        dGij_Ixyz_dn = pauli_block_all(dGij_dn)

        # dG(j, i, -R)
        dGRm_up, dGRm_dn = dGrev[R]
        dGji_up = self.GR_atom(dGRm_up, jatom, iatom)
        dGji_dn = self.GR_atom(dGRm_dn, jatom, iatom)
        dGji_Ixyz_up = pauli_block_all(dGji_up)
        dGji_Ixyz_dn = pauli_block_all(dGji_dn)

        # Combine spin up and down contributions
        Gij_Ixyz = tuple((up + dn) / 2.0 for up, dn in zip(Gij_Ixyz_up, Gij_Ixyz_dn))
        Gji_Ixyz = tuple((up + dn) / 2.0 for up, dn in zip(Gji_Ixyz_up, Gji_Ixyz_dn))
        dGij_Ixyz = tuple((up + dn) / 2.0 for up, dn in zip(dGij_Ixyz_up, dGij_Ixyz_dn))
        dGji_Ixyz = tuple((up + dn) / 2.0 for up, dn in zip(dGji_Ixyz_up, dGji_Ixyz_dn))

        tmp = np.zeros((4, 4), dtype=complex)
        dtmp = np.zeros((4, 4), dtype=complex)
        if self.orb_decomposition:
            ni = self.norb_reduced[iatom]
            nj = self.norb_reduced[jatom]
            torb = np.zeros((4, 4, ni, nj), dtype=complex)
            dtorb = np.zeros((4, 4, ni, nj), dtype=complex)
            for a, b in ([0, 0], [3, 3]):
                piGij = self.get_P_iatom(iatom) @ Gij_Ixyz[a]
                pjGji = self.get_P_iatom(jatom) @ Gji_Ixyz[b]

                pidGij = self.get_P_iatom(iatom) @ dGij_Ixyz[a]
                pjdGji = self.get_P_iatom(jatom) @ dGji_Ixyz[a]
                torb[a, b] = self.simplify_orbital_contributions(
                    np.einsum("ij, ji -> ij", piGij, pjGji) / np.pi, iatom, jatom
                )
                d = np.einsum("ij, ji -> ij", pidGij, pjGji) + np.einsum(
                    "ij, ji -> ij", piGij, pjdGji
                )
                dtorb[a, b] = self.simplify_orbital_contributions(
                    d / np.pi, iatom, jatom
                )
                tmp[a, b] = np.sum(torb[a, b])
                dtmp[a, b] = np.sum(dtorb[a, b])
        else:
            torb = None
            dtorb = None
            for a, b in ([0, 0], [3, 3]):
                pGp = self.get_P_iatom(iatom) @ Gij_Ixyz[a] @ self.get_P_iatom(jatom)
                pdGp = self.get_P_iatom(iatom) @ dGij_Ixyz[a] @ self.get_P_iatom(jatom)
                AijRab = pGp @ Gji_Ixyz[b]
                A1 = pdGp @ Gji_Ixyz[b]
                A2 = pGp @ dGji_Ixyz[b]
                tmp[a, b] = np.trace(AijRab) / np.pi
                dtmp[a, b] = np.trace(A1 + A2) / np.pi
        return tmp, dtmp, torb, dtorb

    def get_all_A(self, G, dGij, dGji):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        """
        A_ijR_list = {}
        dAdx_ijR_list = {}
        A_orb_ijR_list = {}
        dAdx_orb_ijR_list = {}
        for iR, R in enumerate(self.R_ijatom_dict):
            for iatom, jatom in self.R_ijatom_dict[R]:
                A, dAdx, A_orb, dAdx_orb = self.get_A_ijR(
                    G, dGij, dGji, R, iatom, jatom
                )
                A_ijR_list[(R, iatom, jatom)] = A
                dAdx_ijR_list[(R, iatom, jatom)] = dAdx
                A_orb_ijR_list[(R, iatom, jatom)] = A_orb
                dAdx_orb_ijR_list[(R, iatom, jatom)] = dAdx_orb
        return A_ijR_list, dAdx_ijR_list, A_orb_ijR_list, dAdx_orb_ijR_list

    def get_AijR_rhoR(self, e):
        GR_up, dGRij_up, dGRji_up, rhoR_up = self.G.get_GR_and_dGRdx_from_epw(
            self.Rlist, self.short_Rlist, energy=e, epc=self.epc_up, Ru=self.Ru
        )
        GR_dn, dGRij_dn, dGRji_dn, rhoR_dn = self.G.get_GR_and_dGRdx_from_epw(
            self.Rlist, self.short_Rlist, energy=e, epc=self.epc_dn, Ru=self.Ru
        )

        # Combine spin up and down results
        GR = {}
        for R in GR_up:
            GR[R] = (GR_up[R], GR_dn[R])
        dGRij = {}
        for R in dGRij_up:
            dGRij[R] = (dGRij_up[R], dGRij_dn[R])
        dGRji = {}
        for R in dGRji_up:
            dGRji[R] = (dGRji_up[R], dGRji_dn[R])
        rhoR = {}
        for R in rhoR_up:
            rhoR[R] = (rhoR_up[R], rhoR_dn[R])

        AijR, dAdx_ijR, A_orb_ijR, dAdx_orb_ijR = self.get_all_A(GR, dGRij, dGRji)
        return AijR, dAdx_ijR, self.get_rho_e_spin(rhoR), A_orb_ijR, dAdx_orb_ijR

    def get_rho_e_spin(self, rhoR):
        """add component to density matrix from spin-resolved green's function
        :param rhoR: Spin-resolved density matrix in real space.
        """
        rho_up, rho_dn = rhoR[(0, 0, 0)]
        return -1.0 / np.pi * (rho_up + rho_dn) / 2.0

    def A_to_Jtensor(self):
        """
        Calculate J tensors from A.
        If we assume the exchange can be written as a bilinear tensor form,
        J_{isotropic} = Tr Im (A^{00} - A^{xx} - A^{yy} - A^{zz})
        """
        super().A_to_Jtensor()
        self.dJdx = {}
        for key, val in self.dA_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            dJiso = np.zeros((3, 3), dtype=float)
            # Heisenberg like J.
            for i in range(3):
                dJiso[i, i] = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])
            if is_nonself:
                self.dJdx[keyspin] = dJiso[0, 0]

        self.dJdx2 = {}
        for key, val in self.dA2_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            dJ2iso = np.zeros((3, 3), dtype=float)
            # Heisenberg like J.
            for i in range(3):
                dJ2iso[i, i] += np.imag(val[0, 0] - val[3, 3])
            if is_nonself:
                self.dJdx2[keyspin] = dJ2iso[0, 0]

    def A_to_Jtensor_orb(self):
        """
        convert the orbital composition of A into J, DMI, Jani
        """
        self.Jiso_orb = {}
        self.Jani_orb = {}
        self.DMI_orb = {}
        self.dJdx_orb = {}

        if self.orb_decomposition:
            for key, val in self.A_ijR_orb.items():
                dval = self.dA_ijR_orb[key]

                R, iatom, jatom = key
                Rm = tuple(-x for x in R)
                valm = self.A_ijR_orb[(Rm, jatom, iatom)]
                # dvalm = self.dA_ijR_orb[(Rm, jatom, iatom)]

                ni = self.norb_reduced[iatom]
                nj = self.norb_reduced[jatom]

                is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                ispin = self.ispin(iatom)
                jspin = self.ispin(jatom)
                keyspin = (R, ispin, jspin)

                # isotropic J
                Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

                dJdx = np.imag(dval[0, 0] - dval[1, 1] - dval[2, 2] - dval[3, 3])

                # off-diagonal anisotropic exchange
                Ja = np.zeros((3, 3, ni, nj), dtype=float)
                for i in range(3):
                    for j in range(3):
                        Ja[i, j] = np.imag(val[i + 1, j + 1] + valm[i + 1, j + 1])
                # DMI

                Dtmp = np.zeros((3, ni, nj), dtype=float)
                for i in range(3):
                    Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

                if is_nonself:
                    self.Jiso_orb[keyspin] = Jiso
                    self.Jani_orb[keyspin] = Ja
                    self.DMI_orb[keyspin] = Dtmp
                    self.dJdx_orb[keyspin] = dJdx

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        rhoRs = []
        # GRs = []
        AijRs = {}
        dAijRs = {}

        AijRs_orb = {}
        dAijRs_orb = {}

        npole = len(self.contour.path)
        if self.np > 1:
            results = p_map(self.get_AijR_rhoR, self.contour.path, num_cpus=self.np)
        else:
            results = map(self.get_AijR_rhoR, tqdm(self.contour.path, total=npole))

        for i, result in enumerate(results):
            for iR, R in enumerate(self.R_ijatom_dict):
                for iatom, jatom in self.R_ijatom_dict[R]:
                    if (R, iatom, jatom) in AijRs:
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom, jatom])
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom, jatom])
                        AijRs_orb[(R, iatom, jatom)].append(result[3][R, iatom, jatom])
                        dAijRs_orb[(R, iatom, jatom)].append(result[4][R, iatom, jatom])

                    else:
                        AijRs[(R, iatom, jatom)] = []
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom, jatom])

                        dAijRs[(R, iatom, jatom)] = []
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom, jatom])

                        AijRs_orb[(R, iatom, jatom)] = []
                        AijRs_orb[(R, iatom, jatom)].append(result[3][R, iatom, jatom])

                        dAijRs_orb[(R, iatom, jatom)] = []
                        dAijRs_orb[(R, iatom, jatom)].append(result[4][R, iatom, jatom])

            rhoRs.append(result[2])
        if self.np > 1:
            pass

        # self.save_AijRs(AijRs)
        self.integrate(rhoRs, AijRs, dAijRs, AijRs_orb, dAijRs_orb)

        self.get_rho_atom()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

    def integrate(
        self, rhoRs, AijRs, dAijRs, AijRs_orb=None, dAijRs_orb=None, method="simpson"
    ):
        """
        AijRs: a list of AijR,
        wherer AijR: array of ((nR, n, n, 4,4), dtype=complex)
        """
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == "simpson":
            integrate = simpson_nonuniform

        self.rho = integrate(self.contour.path, rhoRs)
        for iR, R in enumerate(self.R_ijatom_dict):
            for iatom, jatom in self.R_ijatom_dict[R]:
                f = AijRs[(R, iatom, jatom)]
                df = dAijRs[(R, iatom, jatom)]
                self.A_ijR[(R, iatom, jatom)] = integrate(self.contour.path, f)
                self.dA_ijR[(R, iatom, jatom)] = integrate(self.contour.path, df)
                if self.orb_decomposition:
                    self.A_ijR_orb[(R, iatom, jatom)] = integrate(
                        self.contour.path, AijRs_orb[(R, iatom, jatom)]
                    )
                    self.dA_ijR_orb[(R, iatom, jatom)] = integrate(
                        self.contour.path, dAijRs_orb[(R, iatom, jatom)]
                    )

    def write_output(self, path="TB2J_results"):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            Jiso_orb=self.Jiso_orb,
            dJdx=self.dJdx,
            dJdx_orb=self.dJdx_orb,
            # dmi_ddict=None,
            # NJT_Jdict=None,
            # NJT_ddict=None,
            # bilinear_Jdict=None,
            # biquadratic_Jdict=self.B,
        )
        output.write_all(path=path)
