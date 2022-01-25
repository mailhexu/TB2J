import numpy as np
from TB2J.pauli import pauli_block_all, pauli_block_sigma_norm
from TB2J.io_exchange import SpinIO
from functools import lru_cache
from TB2J.exchange import ExchangeNCL
from collections import defaultdict
from TB2J.utils import simpson_nonuniform, trapezoidal_nonuniform
from tqdm import tqdm
from p_tqdm import p_map


class ExchangePert(ExchangeNCL):
    def set_dHdx(self, dHdx):
        self.dHdx = dHdx
        self.dHdxR0 = dHdx.dHR_0
        print(f"{self.dHdx.nbasis} {self.nbasis}")
        assert (self.dHdx.nbasis == self.nbasis)
        self.dA_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA2_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))

    @lru_cache()
    def get_dP_iatom(self, iatom):
        orbs = self.iorb(iatom)
        return pauli_block_sigma_norm(self.dHdxR0[np.ix_(orbs, orbs)])

    def get_A_ijR(self, G, dG, R, iatom, jatom):
        """ calculate A from G for a energy .
        It take the
        .. math::
           A^{uv} = p T^u p T^v dE / pi

        where u, v are I, x, y, z (index 0, 1,2,3). p(i) = self.get_P_iatom(iatom)
        T^u(ijR)  (u=0,1,2,3) = pauli_block_all(G)

        :param G: Green's function for all R, i, j.
        :param iatom: i
        :param jatom: j
        :param R:  Rvector o j
        :returns: a matrix of A_ij(u, v), where u, v =(0)0, x(1), y(2), z(3)
        :rtype:  4*4 matrix
        """

        # G[i, j, R]
        GR = G[R]
        Gij = self.GR_atom(GR, iatom, jatom)
        Gij_Ixyz = pauli_block_all(Gij)

        # G(j, i, -R)
        Rm = tuple(-x for x in R)
        GRm = G[Rm]
        Gji = self.GR_atom(GRm, jatom, iatom)
        Gji_Ixyz = pauli_block_all(Gji)

        dGR = dG[R]
        dGij = self.GR_atom(dGR, iatom, jatom)
        # GijR , I, x, y, z component.
        dGij_Ixyz = pauli_block_all(dGij)

        # G(j, i, -R)
        Rm = tuple(-np.array(R))
        dGRm = dG[Rm]
        dGji = self.GR_atom(dGRm, jatom, iatom)
        dGji_Ixyz = pauli_block_all(dGji)

        tmp1 = np.zeros((4, 4), dtype=complex)
        tmp2 = np.zeros((4, 4), dtype=complex)
        tmp3 = np.zeros((4, 4), dtype=complex)
        for a in range(4):
            pGp = self.get_P_iatom(iatom) @ Gij_Ixyz[a] @ self.get_P_iatom(
                jatom)
            pdGp = self.get_P_iatom(iatom) @ dGij_Ixyz[a] @ self.get_P_iatom(
                jatom)
            # dpGp = self.get_dP_iatom(iatom) @ Gij_Ixyz[a] @ self.get_P_iatom(
            #    jatom)
            pGdp = self.get_P_iatom(iatom) @ Gij_Ixyz[a] @ self.get_dP_iatom(
                jatom)
            for b in range(4):
                AijRab = pGp @ Gji_Ixyz[b]
                A1 = pdGp @ Gji_Ixyz[b]
                A2 = pGp @ dGji_Ixyz[b]
                #A3 = dpGp @ Gji_Ixyz[b]
                #A4 = pGdp @ Gji_Ixyz[b]
                AOijRab = A1 + A2  # + A3 + A4

                if False:
                    B1 = np.matmul(
                        np.matmul(self.get_dP_iatom(iatom), dGij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), Gji_Ixyz[b]))
                    B2 = np.matmul(
                        np.matmul(self.get_dP_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_dP_iatom(jatom), Gji_Ixyz[b]))
                    B3 = np.matmul(
                        np.matmul(self.get_dP_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), dGji_Ixyz[b]))
                    B4 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), dGij_Ixyz[a]),
                        np.matmul(self.get_dP_iatom(jatom), Gji_Ixyz[b]))
                    B5 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), dGij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), dGji_Ixyz[b]))

                    B6 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_dP_iatom(jatom), dGji_Ixyz[b]))
                    tmp3[a, b] = np.trace(B1 + B2 + B3 + B4 + B5 + B6)

                # trace over orb
                tmp1[a, b] = np.trace(AijRab) / np.pi
                tmp2[a, b] = np.trace(AOijRab) / np.pi
        return tmp1, tmp2

    def get_all_A(self, G, dG):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        """
        A_ijR_list = {}
        dAdx_ijR_list = {}
        for iR, R in enumerate(self.R_ijatom_dict):
            for (iatom, jatom) in self.R_ijatom_dict[R]:
                A, dAdx = self.get_A_ijR(G, dG, R, iatom, jatom)
                A_ijR_list[(R, iatom, jatom)] = A
                dAdx_ijR_list[(R, iatom, jatom)] = dAdx
        return A_ijR_list, dAdx_ijR_list

    def get_AijR_rhoR(self, e):
        GR, dGR, rhoR = self.G.get_GR_and_dGRdx(self.short_Rlist,
                                                energy=e,
                                                dHdx=self.dHdx)
        AijR, dAdx_ijR = self.get_all_A(GR, dGR)
        return AijR, dAdx_ijR, self.get_rho_e(rhoR)

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
                dJiso[i, i] = np.imag(val[0, 0] - val[1, 1] - val[2, 2] -
                                      val[3, 3])
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

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        rhoRs = []
        GRs = []
        AijRs = {}
        dAijRs = {}

        AijRs_orb = {}

        npole = len(self.contour.path)
        if self.np > 1:
            results = p_map(self.get_AijR_rhoR,
                            self.contour.path,
                            num_cpus=self.np)
        else:
            results = map(self.get_AijR_rhoR,
                          tqdm(self.contour.path, total=npole))

        for i, result in enumerate(results):
            for iR, R in enumerate(self.R_ijatom_dict):
                for (iatom, jatom) in self.R_ijatom_dict[R]:
                    if (R, iatom, jatom) in AijRs:
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom,
                                                                  jatom])
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom,
                                                                   jatom])
                    else:
                        AijRs[(R, iatom, jatom)] = []
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom,
                                                                  jatom])

                        dAijRs[(R, iatom, jatom)] = []
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom,
                                                                   jatom])

            rhoRs.append(result[2])
        if self.np > 1:
            pass

        # self.save_AijRs(AijRs)
        self.integrate(rhoRs, AijRs, dAijRs, AijRs_orb)

        self.get_rho_atom()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

    def integrate(self,
                  rhoRs,
                  AijRs,
                  dAijRs,
                  AijRs_orb=None,
                  method='simpson'):
        """
        AijRs: a list of AijR, 
        wherer AijR: array of ((nR, n, n, 4,4), dtype=complex)
        """
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == 'simpson':
            integrate = simpson_nonuniform

        self.rho = integrate(self.contour.path, rhoRs)
        for iR, R in enumerate(self.R_ijatom_dict):
            for (iatom, jatom) in self.R_ijatom_dict[R]:
                f = AijRs[(R, iatom, jatom)]
                df = dAijRs[(R, iatom, jatom)]
                self.A_ijR[(R, iatom, jatom)] = integrate(self.contour.path, f)
                self.dA_ijR[(R, iatom,
                             jatom)] = integrate(self.contour.path, df)

    def write_output(self, path='TB2J_results'):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            dJdx=self.dJdx,
            # dmi_ddict=None,
            # NJT_Jdict=None,
            # NJT_ddict=None,
            # bilinear_Jdict=None,
            # biquadratic_Jdict=self.B,
        )
        output.write_all(path=path)
