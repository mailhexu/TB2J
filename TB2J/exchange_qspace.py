import numpy as np
from TB2J.utils import kmesh_to_R
from functools import lru_cache
from .utils import simpson_nonuniform, trapezoidal_nonuniform
from ase.dft.kpoints import monkhorst_pack
from TB2J.exchangeCL2 import ExchangeCL2


def find_index_k(kpts, q):
    """
    for one q point, find the indices of k'=k+q inside the space of k.
    kpts: A list of kpoints, should be a regular grid e.g. Monkhorst-pack
    q: one q point, which should be one point in the regular grid.
    """
    kpts_p = np.mod(kpts + 1e-9, 1)
    jkpts = np.zeros(len(kpts), dtype=int)
    for ik, k in enumerate(kpts):
        jkpts[ik] = np.argmin(
            np.linalg.norm(np.mod(k + q + 1e-9, 1)[None, :] - kpts_p, axis=1)
        )
    if len(jkpts) != len(set(jkpts)):
        raise ValueError(
            "Cannot find all the k+q point. Please check if the k-mesh and the q-point is compatible."
        )
    return jkpts


class ExchangeCLQspace(ExchangeCL2):
    def _prepare(self):
        self.nmagatom = len(self.ind_mag_atoms)
        # self.qmesh=[3,3,3]
        self.qmesh = self.kmesh
        self.qpts = monkhorst_pack(size=self.qmesh)
        self.Rqlist = kmesh_to_R(self.qmesh)

        self.nqpt = len(self.qpts)
        self.ncontour = len(self.contour.path)
        self.Jqe_list = np.zeros(
            (self.ncontour, self.nqpt, self.nmagatom, self.nmagatom), dtype=complex
        )
        self.Kqe_list = np.zeros_like(self.Jqe_list)
        self.Xqe_list = np.zeros_like(self.Jqe_list)
        self.get_rho_atom()

    def get_rho_atom(self):
        """
        charges and spins from density matrices
        """
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)

        rho_up = self.Gup.get_density()
        rho_dn = self.Gdn.get_density()
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tup = np.sum(rho_up[iorb])
            tdn = np.sum(rho_dn[iorb])
            self.charges[iatom] = tup + tdn
            self.spinat[iatom, 2] = tup - tdn

    def Gk_atom(self, Gk, iatom, jatom):
        return self.GR_atom(Gk, iatom, jatom)

    @property
    def kpts(self):
        return self.Gup.kpts

    @property
    def nkpts(self):
        return len(self.Gup.kpts)

    @lru_cache(maxsize=None)
    def get_ikplusq(self, q):
        return find_index_k(self.kpts, q)

    def get_all_A(self):
        # prepare Gk
        Gk_up = np.zeros((self.nkpts, self.Gup.norb, self.Gup.norb), dtype=complex)
        Gk_dn = np.zeros((self.nkpts, self.Gup.norb, self.Gup.norb), dtype=complex)

        for ie, energy in enumerate(self.contour.path):
            for ik, _ in enumerate(self.kpts):
                Gk_up[ik] = self.Gup.get_Gk(ik, energy)
                Gk_dn[ik] = self.Gdn.get_Gk(ik, energy)

            for iq, q in enumerate(self.qpts):
                ikplusq_list = self.get_ikplusq(tuple(q))

                for ik, ikq in enumerate(ikplusq_list):
                    # Kq= Gk_up[ik] @ self.Delta @ Gk_dn[ikq]
                    # for ik, ikq in enumerate(range(self.nkpts)):
                    Guk = Gk_up[ik, :, :]
                    Gdk = Gk_dn[ik, :, :]
                    Gukq = Gk_up[ikq, :, :]
                    Gdkq = Gk_dn[ikq, :, :]
                    for i, iatom in enumerate(self.ind_mag_atoms):
                        Deltai = self.get_Delta(iatom)
                        for j, jatom in enumerate(self.ind_mag_atoms):
                            Deltaj = self.get_Delta(jatom)
                            Gij_up_k = self.Gk_atom(Guk, iatom, jatom)
                            Gji_dn_kq = self.Gk_atom(Gdkq, jatom, iatom)

                            Gij_dn_kq = self.Gk_atom(Gdk, iatom, jatom)
                            Gji_up_k = self.Gk_atom(Gukq, jatom, iatom)
                            A = np.trace(
                                np.linalg.multi_dot(
                                    (Deltai, Gij_up_k, Deltaj, Gji_dn_kq)
                                )
                            )
                            K1 = Gij_up_k @ Deltaj @ Gji_dn_kq

                            K2 = Gij_dn_kq @ Deltaj @ Gji_up_k
                            X = Gij_up_k @ Gji_dn_kq
                            A = (np.trace(Deltai @ K1) + np.trace(Deltai @ K2)) * 0.5
                            K = K1 + K2
                            K += Gij_dn_kq @ Deltaj @ Gji_up_k
                            # K=Kq[self.orb_slice[iatom], self.orb_slice[iatom]]*2
                            # self.Jqe_orb_list[(ikq, iatom, jatom)].append(tmp / (4.0 * np.pi))
                            self.Jqe_list[ie, iq, i, j] += A / (4.0 * np.pi) / self.nqpt
                            self.Kqe_list[ie, iq, i, j] -= (
                                np.trace(K) / (2.0 * np.pi) / self.nqpt
                            )
                            self.Xqe_list[ie, iq, i, j] += (
                                np.trace(X) * (2.0 / np.pi) / self.nqpt
                            )

    def integrate(self, method="simpson"):
        self.Jq = np.zeros((self.nqpt, self.nmagatom, self.nmagatom), dtype=float)
        self.Kq = np.zeros_like(self.Jq)
        self.Xq = np.zeros_like(self.Jq)
        self.Jnorm_q = np.zeros_like(self.Jq)
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == "simpson":
            integrate = simpson_nonuniform
        # self.rho_up = np.imag(integrate(self.contour.path, self.rho_up_list))
        # self.rho_dn = np.imag(integrate(self.contour.path, self.rho_dn_list))
        for iq, q in enumerate(self.qpts):
            for i, iatom in enumerate(self.ind_mag_atoms):
                for j, jatom in enumerate(self.ind_mag_atoms):
                    self.Jq[iq, i, j] = np.imag(
                        integrate(self.contour.path, self.Jqe_list[:, iq, i, j])
                    )
                    self.Kq[iq, i, j] = np.imag(
                        integrate(self.contour.path, self.Kqe_list[:, iq, i, j])
                    )
                    self.Xq[iq, i, j] = np.imag(
                        integrate(self.contour.path, self.Xqe_list[:, iq, i, j])
                    )
            # print(f"{q=}, {self.Kq[iq]}")
        return self.Jq

    def bruno_renormalize(self):
        M = np.diag(self.spinat[self.ind_mag_atoms, 2])

        for iq, q in enumerate(self.qpts):
            self.Jnorm_q[iq] = self.Jq[iq] + 0.5 * (M - self.Kq[iq].T) @ np.linalg.inv(
                self.Xq[iq]
            ) @ (M - self.Kq[iq])
            # print(f"{q}: Jq:{self.Jq[iq]}, Jnorm:{self.Jnorm_q[iq]}")

    def q_to_r(self):
        self.JR = np.zeros((len(self.Rlist), self.nmagatom, self.nmagatom), dtype=float)
        self.Jnorm_R = np.zeros_like(self.JR)
        self.KR = np.zeros_like(self.JR)
        for iR, R in enumerate(self.Rlist):
            for iq, q in enumerate(self.qpts):
                phase = np.exp(-2.0j * np.pi * (R @ q))
                self.JR[iR] += np.real(self.Jq[iq] * phase) / len(self.qpts)
                self.Jnorm_R[iR] += np.real(self.Jnorm_q[iq] * phase) / len(self.qpts)
                self.KR[iR] += np.real(self.Kq[iq] * phase) / len(self.qpts)
            # print(f"{R}: J={self.JR[iR]}, Jnorm{self.Jnorm_R[iR]}")
            # print(f"{R}: {np.sum(self.KR, axis=0)}")
        return self.JR

    def get_Jdict(self):
        for iR, R in enumerate(self.Rlist):
            R = tuple(R)
            for i, iatom in enumerate(self.ind_mag_atoms):
                for j, jatom in enumerate(self.ind_mag_atoms):
                    val = self.JR[iR, i, j]
                    ispin = self.ispin(iatom)
                    jspin = self.ispin(jatom)
                    keyspin = (R, ispin, jspin)
                    is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                    Jij = val / np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
                    # Jorbij = np.imag(self.Jorb[key]) / np.sign(
                    #    np.dot(self.spinat[iatom], self.spinat[jatom]))
                    if is_nonself:
                        self.exchange_Jdict[keyspin] = Jij
                        # self.exchange_Jdict_orb[keyspin] = Jorbij

    def calculate_all(self):
        self._prepare()
        self.get_all_A()
        self.integrate()
        self.bruno_renormalize()
        self.q_to_r()
        self.get_Jdict()
