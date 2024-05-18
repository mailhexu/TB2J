import numpy as np
from TB2J.pauli import pauli_block_all, pauli_block_sigma_norm
from TB2J.io_exchange import SpinIO
from functools import lru_cache
from TB2J.exchange import ExchangeNCL
from collections import defaultdict


class ExchangePert(ExchangeNCL):
    def set_dHdx(self, dHdx):
        self.dHdx = dHdx
        self.dHdxR0 = dHdx.ham_R0
        assert self.dHdx.nbasis == self.nbasis
        self.AO_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.AT_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))

    @lru_cache()
    def get_dP_iatom(self, iatom):
        orbs = self.iorb(iatom)
        return pauli_block_sigma_norm(self.dHdxR0[np.ix_(orbs, orbs)])

    def get_A_ijR(self, G, dG, iatom, jatom, de):
        """calculate A from G for a energy slice (de).
        It take the
        .. math::
           A^{uv} = p T^u p T^v dE / pi

        where u, v are I, x, y, z (index 0, 1,2,3). p(i) = self.get_P_iatom(iatom)
        T^u(ijR)  (u=0,1,2,3) = pauli_block_all(G)

        :param G: Green's function for all R, i, j.
        :param iatom: i
        :param jatom: j
        :param de:  energy step. used for integeration
        :returns: a matrix of A_ij(u, v), where u, v =(0)0, x(1), y(2), z(3)
        :rtype:  4*4 matrix
        """

        for R in self.Rlist:
            # G[i, j, R]
            GR = G[R]
            Gij = self.GR_atom(GR, iatom, jatom)
            # GijR , I, x, y, z component.
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
                for b in range(4):
                    AijRab = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), Gji_Ixyz[b]),
                    )

                    A1 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), dGij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), Gji_Ixyz[b]),
                    )
                    A2 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), dGji_Ixyz[b]),
                    )
                    A3 = np.matmul(
                        np.matmul(self.get_dP_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_P_iatom(jatom), Gji_Ixyz[b]),
                    )
                    A4 = np.matmul(
                        np.matmul(self.get_P_iatom(iatom), Gij_Ixyz[a]),
                        np.matmul(self.get_dP_iatom(jatom), Gji_Ixyz[b]),
                    )
                    AOijRab = A1 + A2 + A3 + A4

                    if False:
                        B1 = np.matmul(
                            np.matmul(self.get_dP_iatom(iatom), dGij_Ixyz[a]),
                            np.matmul(self.get_P_iatom(jatom), Gji_Ixyz[b]),
                        )
                        B2 = np.matmul(
                            np.matmul(self.get_dP_iatom(iatom), Gij_Ixyz[a]),
                            np.matmul(self.get_dP_iatom(jatom), Gji_Ixyz[b]),
                        )
                        B3 = np.matmul(
                            np.matmul(self.get_dP_iatom(iatom), Gij_Ixyz[a]),
                            np.matmul(self.get_P_iatom(jatom), dGji_Ixyz[b]),
                        )
                        B4 = np.matmul(
                            np.matmul(self.get_P_iatom(iatom), dGij_Ixyz[a]),
                            np.matmul(self.get_dP_iatom(jatom), Gji_Ixyz[b]),
                        )
                        B5 = np.matmul(
                            np.matmul(self.get_P_iatom(iatom), dGij_Ixyz[a]),
                            np.matmul(self.get_P_iatom(jatom), dGji_Ixyz[b]),
                        )

                        B6 = np.matmul(
                            np.matmul(self.get_P_iatom(iatom), Gij_Ixyz[a]),
                            np.matmul(self.get_dP_iatom(jatom), dGji_Ixyz[b]),
                        )
                        tmp3[a, b] = np.trace(B1 + B2 + B3 + B4 + B5 + B6)

                    # trace over orb
                    tmp1[a, b] = np.trace(AijRab)
                    tmp2[a, b] = np.trace(AOijRab)
            # Note: the full complex, rather than Re or Im part is stored into A_ijR.
            self.A_ijR[(R, iatom, jatom)] += tmp1 * de / np.pi
            self.AO_ijR[(R, iatom, jatom)] += tmp2 * de / np.pi
            self.AT_ijR[(R, iatom, jatom)] += tmp3 * de / np.pi

    def get_all_A(self, G, dG, de):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        :param de: energy step.
        """
        for iatom in self.ind_mag_atoms:
            for jatom in self.ind_mag_atoms:
                self.get_A_ijR(G, dG, iatom, jatom, de)

    def A_to_Jtensor(self):
        """
        Calculate J tensors from A.
        If we assume the exchange can be written as a bilinear tensor form,
        J_{isotropic} = Tr Im (A^{00} - A^{xx} - A^{yy} - A^{zz})
        """
        super().A_to_Jtensor()
        self.dJdx = {}
        for key, val in self.AO_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            dJiso = np.zeros((3, 3), dtype=float)
            # Heisenberg like J.
            for i in range(3):
                dJiso[i, i] += np.imag(val[0, 0] - val[3, 3])
            if is_nonself:
                self.dJdx[keyspin] = dJiso[0, 0]

        self.dJdx2 = {}
        for key, val in self.AT_ijR.items():
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

        # widgets = [
        #    " [",
        #    progressbar.Timer(),
        #    "] ",
        #    progressbar.Bar(),
        #    " (",
        #    progressbar.ETA(),
        #    ") ",
        # ]
        # bar = progressbar.ProgressBar(maxval=self.contour.npoints, widgets=widgets)
        # bar.start()
        for ie in range(self.contour.npoints):
            # bar.update(ie)
            e = self.contour.elist[ie]
            de = self.contour.de[ie]
            GR, dGdx = self.G.get_GR_and_dGRdx(self.Rlist, energy=e, dHdx=self.dHdx)
            # self.get_rho_e(GR, de)
            self.get_all_A(GR, dGdx, de)
            if self.calc_NJt:
                self.get_N_e(GR, de)

        self.get_rho_atom()
        self.A_to_Jtensor()
        if self.calc_NJt:
            self.calculate_DMI_NJT()
        # bar.finish()

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
            dJdx=self.dJdx,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            bilinear_Jdict=None,
            biquadratic_Jdict=self.B,
        )
        output.write_all(path=path)
