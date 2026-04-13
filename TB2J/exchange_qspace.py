import numpy as np
from ase.dft.kpoints import monkhorst_pack
from tqdm import tqdm

from TB2J.exchange import ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.external import p_imap
from TB2J.pauli import pauli_block_all


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
        self.qmesh = self.kmesh
        self.qpts = monkhorst_pack(size=self.qmesh)
        # self.Rlist is set by Exchange.__init__() via _prepare_Rlist()
        # as kmesh_to_R(self.kmesh). Since qmesh == kmesh, Rlist covers
        # all R-vectors needed by q_to_r().
        self._ikplusq_cache = {}

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

    def get_ikplusq(self, q):
        q_tuple = tuple(q) if not isinstance(q, tuple) else q
        if q_tuple not in self._ikplusq_cache:
            self._ikplusq_cache[q_tuple] = find_index_k(self.kpts, q_tuple)
        return self._ikplusq_cache[q_tuple]

    def get_all_A(self):
        """
        Calculate A matrix elements in q-space.

        The exchange coupling formula in q-space:
          A_ij(q) = sum_k Tr[Delta_i @ G^up_ij(k) @ Delta_j @ G^dn_ji(k+q)]

        This is the q-space equivalent of the real-space formula:
          A_ij(R) = Tr[Delta_i @ G^up_ij(R) @ Delta_j @ G^dn_ji(-R)]

        where the Fourier transform relation is:
          G(R) = (1/N_k) sum_k G(k) exp(-ikR)
        """
        kweights = self.Gup.kweights

        for ie, energy in enumerate(self.contour.path):
            Gk_up = np.zeros((self.nkpts, self.Gup.norb, self.Gup.norb), dtype=complex)
            Gk_dn = np.zeros((self.nkpts, self.Gup.norb, self.Gup.norb), dtype=complex)
            for ik in range(self.nkpts):
                Gk_up[ik] = self.Gup.get_Gk(ik, energy)
                Gk_dn[ik] = self.Gdn.get_Gk(ik, energy)

            for iq, q in enumerate(self.qpts):
                ikplusq_list = self.get_ikplusq(tuple(q))

                for ik, ikq in enumerate(ikplusq_list):
                    kw = kweights[ik]
                    Guk = Gk_up[ik]
                    Gdkq = Gk_dn[ikq]

                    for i, iatom in enumerate(self.ind_mag_atoms):
                        Deltai = self.get_Delta(iatom)
                        for j, jatom in enumerate(self.ind_mag_atoms):
                            Deltaj = self.get_Delta(jatom)

                            # A_ij(q) = sum_k Tr[Delta_i @ G^up_ij(k) @ Delta_j @ G^dn_ji(k+q)]
                            Gij_up_k = self.Gk_atom(Guk, iatom, jatom)
                            Gji_dn_kq = self.Gk_atom(Gdkq, jatom, iatom)

                            t = np.einsum(
                                "ij, ji-> ij",
                                np.matmul(Deltai, Gij_up_k),
                                np.matmul(Deltaj, Gji_dn_kq),
                            )
                            A = np.sum(t)

                            K = Gij_up_k @ Deltaj @ Gji_dn_kq

                            X = Gij_up_k @ Gji_dn_kq

                            # Accumulate with k-weight; nqpt normalization in q_to_r
                            self.Jqe_list[ie, iq, i, j] += A * kw / (4.0 * np.pi)
                            self.Kqe_list[ie, iq, i, j] -= (
                                np.trace(K) * kw / (2.0 * np.pi)
                            )
                            self.Xqe_list[ie, iq, i, j] += (
                                np.trace(X) * kw * (2.0 / np.pi)
                            )

    def integrate(self):
        """
        Integrate over energy using contour weights, matching the real-space pattern.

        The real-space code accumulates val * weight for each energy point,
        then multiplies by the CFR factor (-pi/2) to get the final result.
        We follow the same pattern here for q-space.
        """
        npole = len(self.contour.path)
        weights = self.contour.weights

        self.Jq = np.zeros((self.nqpt, self.nmagatom, self.nmagatom), dtype=complex)
        self.Kq = np.zeros_like(self.Jq)
        self.Xq = np.zeros_like(self.Jq)
        self.Jnorm_q = np.zeros_like(self.Jq)

        for ie in range(npole):
            w = weights[ie]
            self.Jq += self.Jqe_list[ie] * w
            self.Kq += self.Kqe_list[ie] * w
            self.Xq += self.Xqe_list[ie] * w

        if npole > 0:
            dummy = np.zeros(npole)
            dummy[0] = 1.0
            factor = self.contour.integrate_values(dummy) / weights[0]
            self.Jq *= factor
            self.Kq *= factor
            self.Xq *= factor

        self.Jq = np.imag(self.Jq)
        self.Kq = np.imag(self.Kq)
        self.Xq = np.imag(self.Xq)

        return self.Jq

    def bruno_renormalize(self):
        M = np.diag(self.spinat[self.ind_mag_atoms, 2])

        for iq in range(self.nqpt):
            self.Jnorm_q[iq] = self.Jq[iq] + 0.5 * (M - self.Kq[iq].T) @ np.linalg.inv(
                self.Xq[iq]
            ) @ (M - self.Kq[iq])

    def q_to_r(self):
        """Fourier transform from q-space to real-space.

        Uses exp(-2πiq·R) because J(q) and K(q) are already real (imaginary
        parts were extracted in integrate()).  The NCL q_to_r() uses the
        conjugate phase exp(+2πiq·R) because A(q) remains complex.
        """
        self.JR = np.zeros((len(self.Rlist), self.nmagatom, self.nmagatom), dtype=float)
        self.Jnorm_R = np.zeros_like(self.JR)
        self.KR = np.zeros_like(self.JR)
        for iR, R in enumerate(self.Rlist):
            for iq, q in enumerate(self.qpts):
                phase = np.exp(-2.0j * np.pi * (R @ q))
                self.JR[iR] += np.real(self.Jq[iq] * phase) / self.nqpt
                self.Jnorm_R[iR] += np.real(self.Jnorm_q[iq] * phase) / self.nqpt
                self.KR[iR] += np.real(self.Kq[iq] * phase) / self.nqpt
        return self.JR

    def get_Jdict(self):
        """Convert JR values to exchange J dictionary, matching real-space A_to_Jtensor pattern."""
        for iR, R in enumerate(self.Rlist):
            R = tuple(R)
            for i, iatom in enumerate(self.ind_mag_atoms):
                for j, jatom in enumerate(self.ind_mag_atoms):
                    ispin = self.ispin(iatom)
                    jspin = self.ispin(jatom)
                    keyspin = (R, ispin, jspin)
                    if keyspin not in self.distance_dict:
                        continue
                    val = self.JR[iR, i, j]
                    is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                    Jij = val / np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
                    if is_nonself:
                        self.exchange_Jdict[keyspin] = Jij

        if self.bruno_correction:
            self.exchange_Jdict_bruno = {}
            for iR, R in enumerate(self.Rlist):
                R = tuple(R)
                for i, iatom in enumerate(self.ind_mag_atoms):
                    for j, jatom in enumerate(self.ind_mag_atoms):
                        ispin = self.ispin(iatom)
                        jspin = self.ispin(jatom)
                        keyspin = (R, ispin, jspin)
                        if keyspin not in self.distance_dict:
                            continue
                        val = self.Jnorm_R[iR, i, j]
                        is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                        Jij = val / np.sign(
                            np.dot(self.spinat[iatom], self.spinat[jatom])
                        )
                        if is_nonself:
                            self.exchange_Jdict_bruno[keyspin] = Jij

    def calculate_all(self):
        self._prepare()
        self.get_all_A()
        self.integrate()
        self.bruno_renormalize()
        self.q_to_r()
        self.get_Jdict()


class ExchangeNCLQspace(ExchangeNCL):
    """
    Non-collinear exchange coupling calculated in q-space.

    Computes the full 4×4 A-tensor A^{uv}(q) in reciprocal space using
    the spinor Green's function and Pauli decomposition, then Fourier
    transforms back to real space to extract J_iso, J_ani, and DMI.

    The q-space formula:
      A^{uv}(q, E) = (1/pi) * sum_k Tr[Pi @ G^u_ij(k, E) @ Pj @ G^v_ji(k+q, E)]

    where u, v in {0, x, y, z} are Pauli indices, Pi and Pj are spin
    projection operators, and G^u is the u-th Pauli block of the
    atom-projected spinor Green's function.
    """

    def _prepare(self):
        """Set up q-mesh, P matrices, and accumulator arrays."""
        self.nmagatom = len(self.ind_mag_atoms)
        self.qmesh = self.kmesh
        self.qpts = monkhorst_pack(size=self.qmesh)
        # self.Rlist is set by Exchange.__init__() via _prepare_Rlist()
        # as kmesh_to_R(self.kmesh). Since qmesh == kmesh, Rlist covers
        # all R-vectors needed by q_to_r().
        self._ikplusq_cache = {}

        self.nqpt = len(self.qpts)

        # Prepare spin projection operators P for all magnetic atoms
        self._prepare_Patom()

        # Compute charges and spinat from density matrix
        self.get_rho_atom()

    @property
    def kpts(self):
        return self.G.kpts

    @property
    def nkpts(self):
        return len(self.G.kpts)

    def get_ikplusq(self, q):
        """Find the indices of k+q for all k-points, with caching."""
        q_tuple = tuple(q) if not isinstance(q, tuple) else q
        if q_tuple not in self._ikplusq_cache:
            self._ikplusq_cache[q_tuple] = find_index_k(self.kpts, q_tuple)
        return self._ikplusq_cache[q_tuple]

    def _compute_Aq_for_energy(self, energy):
        """Compute A^{uv}(q, E) for a single energy point.

        Returns dict with "Aq" key mapping to array of shape
        (nqpt, nmagatom, nmagatom, 4, 4).
        """
        kweights = self.G.kweights

        Gk_all = np.zeros((self.nkpts, self.G.nbasis, self.G.nbasis), dtype=complex)
        for ik in range(self.nkpts):
            Gk_all[ik] = self.G.get_Gk(ik, energy)

        Aq_E = np.zeros((self.nqpt, self.nmagatom, self.nmagatom, 4, 4), dtype=complex)

        for iq, q in enumerate(self.qpts):
            ikplusq_list = self.get_ikplusq(tuple(q))

            for ik, ikq in enumerate(ikplusq_list):
                kw = kweights[ik]
                Gk = Gk_all[ik]
                Gkq = Gk_all[ikq]

                for i, iatom in enumerate(self.ind_mag_atoms):
                    Pi = self.get_P_iatom(iatom)
                    orbi = self.iorb(iatom)
                    for j, jatom in enumerate(self.ind_mag_atoms):
                        Pj = self.get_P_iatom(jatom)
                        orbj = self.iorb(jatom)

                        Gij_k = Gk[np.ix_(orbi, orbj)]
                        Gji_kq = Gkq[np.ix_(orbj, orbi)]

                        Gij_pauli = pauli_block_all(Gij_k)
                        Gji_pauli = pauli_block_all(Gji_kq)

                        X = Pi @ Gij_pauli
                        Y = Pj @ Gji_pauli

                        A_uv = np.einsum("uij,vji->uv", X, Y)
                        Aq_E[iq, i, j] += A_uv * kw / np.pi

        return {"Aq": Aq_E}

    def get_all_A(self):
        """Compute A(q) by integrating over the energy contour.

        Follows the same streaming pattern as ExchangeNCL.calculate_all():
        accumulates weighted A(q, E_n) directly into self.Aq without
        pre-allocating the full (ncontour × nqpt × nmagatom² × 4 × 4) tensor.
        Supports parallelization over energy points when nproc > 1.
        """
        npole = len(self.contour.path)
        weights = self.contour.weights

        self.Aq = np.zeros(
            (self.nqpt, self.nmagatom, self.nmagatom, 4, 4), dtype=complex
        )

        if self.nproc > 1:
            results = p_imap(
                self._compute_Aq_for_energy, self.contour.path, num_cpus=self.nproc
            )
        else:
            results = (
                self._compute_Aq_for_energy(e)
                for e in tqdm(self.contour.path, total=npole)
            )

        for i, result in enumerate(results):
            w = weights[i]
            self.Aq += result["Aq"] * w

        # Apply CFR integration factor (e.g. -pi/2 for CFR)
        if npole > 0:
            dummy = np.zeros(npole)
            dummy[0] = 1.0
            factor = self.contour.integrate_values(dummy) / weights[0]
            self.Aq *= factor

    def q_to_r(self):
        """
        Fourier transform the A-tensor from q-space to real-space.

        A^{uv}(R) = (1/N_q) * sum_q A^{uv}(q) * exp(+2*pi*i*q.R)

        Uses exp(+2πiq·R) (conjugate to the CL convention) because A(q) is
        still complex here — imaginary parts are extracted later by
        A_to_Jtensor().
        """
        from collections import defaultdict

        self.A_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))

        for iR, R in enumerate(self.Rlist):
            for iq, q in enumerate(self.qpts):
                phase = np.exp(2.0j * np.pi * (R @ q))
                for i, iatom in enumerate(self.ind_mag_atoms):
                    for j, jatom in enumerate(self.ind_mag_atoms):
                        R_tuple = tuple(R)
                        self.A_ijR[(R_tuple, iatom, jatom)] += (
                            self.Aq[iq, i, j] * phase / self.nqpt
                        )

        return self.A_ijR

    def calculate_all(self):
        """Orchestrate the full NCL q-space calculation pipeline."""
        self._prepare()
        self.get_all_A()
        self.q_to_r()

        self.A_to_Jtensor()
        if self.orb_decomposition:
            self.A_to_Jtensor_orb()
