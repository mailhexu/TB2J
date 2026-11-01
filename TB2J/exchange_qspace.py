from functools import lru_cache

import numpy as np
from ase.dft.kpoints import monkhorst_pack

from TB2J.contour import Contour
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.utils import kmesh_to_R

from .utils import simpson_nonuniform, trapezoidal_nonuniform


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
    def __init__(self, *args, **kwargs):
        # CFR2 is the default integration for the reciprocal-space driver:
        # its stored weights, not pole geometry, define the contour integral.
        # `method` (contour selector) and `integration_method` (integration
        # branch) must select the same contour family: an explicit None is
        # dropped so the base class does not silently build a CFR contour,
        # and conflicting selectors are refused instead of, e.g., applying
        # a geometric rule to CFR pole positions.
        method = kwargs.get("method")
        if method is None:
            kwargs.pop("method", None)
        if kwargs.get("integration_method") is None:
            kwargs["integration_method"] = method or "cfr2"
        if (
            method is not None
            and method.lower() != kwargs["integration_method"].lower()
        ):
            raise ValueError(
                f"conflicting contour selectors: method={method!r} vs "
                f"integration_method={kwargs['integration_method']!r}; pass one"
            )
        super().__init__(*args, **kwargs)

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
        # Longitudinal-relaxation response (MathDoc:
        # tb2j/bruno-renormalization-authority): C = band longitudinal
        # stiffness.  B (mixed transverse-longitudinal response) is zero
        # until the finite-q transverse vertex of story 009 lands; see
        # get_all_A and bruno_renormalize.
        self.Cqe_list = np.zeros_like(self.Jqe_list)
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
                            # MathDoc: tb2j/qspace-unrenormalized-kernel
                            A = (np.trace(Deltai @ K1) + np.trace(Deltai @ K2)) * 0.5
                            K = K1 + K2
                            K += Gij_dn_kq @ Deltaj @ Gji_up_k
                            # Longitudinal-relaxation response: the
                            # longitudinal vertex is the channel sign (+1 up,
                            # -1 down).  C carries equal signs for the two
                            # orderings; its -1/pi measure is pinned against
                            # the exact band stiffness (divided-difference
                            # sum) by tests/test_qspace_raw_response.py.
                            # CR-005: the coded transverse-longitudinal trace
                            # products are NOT the true B vertex (for
                            # collinear block-diagonal G the exact B vanishes
                            # identically, while the coded products give a
                            # false nonzero B for nonreciprocal dispersions);
                            # the true finite-q vertex is the phase
                            # derivative (story 009 basis-phase authority),
                            # so B stays exactly zero until then.
                            C_up = np.trace(Gij_up_k @ Gji_up_k)
                            C_dn = np.trace(Gij_dn_kq @ Gji_dn_kq)
                            # K=Kq[self.orb_slice[iatom], self.orb_slice[iatom]]*2
                            # self.Jqe_orb_list[(ikq, iatom, jatom)].append(tmp / (4.0 * np.pi))
                            self.Jqe_list[ie, iq, i, j] += A / (4.0 * np.pi) / self.nqpt
                            self.Kqe_list[ie, iq, i, j] -= (
                                np.trace(K) / (2.0 * np.pi) / self.nqpt
                            )
                            self.Xqe_list[ie, iq, i, j] += (
                                np.trace(X) * (2.0 / np.pi) / self.nqpt
                            )
                            self.Cqe_list[ie, iq, i, j] -= (
                                (C_up + C_dn) / np.pi / self.nqpt
                            )

    def integrate(self, method="simpson"):
        self.Jq = np.zeros((self.nqpt, self.nmagatom, self.nmagatom), dtype=float)
        self.Kq = np.zeros_like(self.Jq)
        self.Xq = np.zeros_like(self.Jq)
        # B (mixed transverse-longitudinal response) is zero until the
        # finite-q transverse vertex of story 009 lands (CR-005).
        self.Bq = np.zeros_like(self.Jq)
        # CR-006: C is kept complex Hermitian; elementwise .imag would
        # destroy the Hermitian structure a multi-site C(q) needs.  The
        # response is extracted as -i*M (M = raw contour integral): for
        # pole-sum contours M = i x response exactly, and for the
        # geometric contours the Hermitian symmetrization removes the
        # symmetric principal-value partner, leaving the spectral
        # stiffness pinned by tests/test_qspace_raw_response.py.  J/K/X
        # are genuine imaginary-part responses and stay real.
        self.Cq = np.zeros((self.nqpt, self.nmagatom, self.nmagatom), dtype=complex)
        self.Jnorm_q = np.zeros_like(self.Jq)
        if self.integration_method.lower() == "cfr2":
            # MathDoc: tb2j/cfr2-integration-contract. Pole geometry is not
            # a geometric integration measure; kernel prefactors stay above.
            self.Jq = self.contour.integrate_values(self.Jqe_list).imag
            self.Kq = self.contour.integrate_values(self.Kqe_list).imag
            self.Xq = self.contour.integrate_values(self.Xqe_list).imag
            Cq = -1j * self.contour.integrate_values(self.Cqe_list)
            self.Cq = 0.5 * (Cq + np.swapaxes(Cq, 1, 2).conj())
            return self.Jq
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
                    self.Cq[iq, i, j] = integrate(
                        self.contour.path, self.Cqe_list[:, iq, i, j]
                    )
            Cq = -1j * self.Cq[iq]
            self.Cq[iq] = 0.5 * (Cq + Cq.conj().T)
            # print(f"{q=}, {self.Kq[iq]}")
        return self.Jq

    def bruno_renormalize(self):
        """Bruno-renormalized exchange Jnorm = Jq + (B C^+ B^T) / 2 per q.

        MathDoc: tb2j/bruno-renormalization-authority.  With the pinned
        mapping J = -A_12/2, the exchange-space longitudinal-relaxation
        correction is +(B C^-1 B^T)_12 / 2: Delta A_12 = -(B C^-1 B^T)_12
        and Delta J = -Delta A_12/2 (CR-004).  C is complex Hermitian
        (CR-006) and generically invertible at finite temperature; its
        exact uniform null mode exists only at T = 0, so the inverse is a
        declared pseudoinverse on the non-null subspace (a declared gauge
        projection), never np.linalg.inv.

        Scope: the transverse vertex of B is identically zero for
        collinear references; the finite-q twist vertex is a phase
        derivative carried by the multi-site basis-phase authority
        (story 009).  Until that vertex lands, Jnorm equals Jq at every
        q, which is exact for collinear references; the flag
        ``longitudinal_correction_applied`` reports whether any nonzero
        correction was applied.
        """
        self.longitudinal_correction_applied = False
        for iq, q in enumerate(self.qpts):
            C = self.Cq[iq]
            w, V = np.linalg.eigh(0.5 * (C + C.conj().T))
            tol = np.max(np.abs(w)) * 1e-10 if w.size else 0.0
            keep = np.abs(w) > tol
            if not np.any(keep) or np.max(np.abs(self.Bq[iq]), initial=0.0) < 1e-12:
                self.Jnorm_q[iq] = self.Jq[iq]
                continue
            # Declared pseudoinverse on the non-null subspace (FR-020).
            Cinv = (V[:, keep] / w[keep]) @ V[:, keep].conj().T
            # The vertex authority guarantees a real correction; the
            # .real drops the numerical zero imaginary part.
            corr = 0.5 * (self.Bq[iq] @ Cinv @ self.Bq[iq].T).real
            self.Jnorm_q[iq] = self.Jq[iq] + corr
            self.longitudinal_correction_applied = True

    def q_to_r(self):
        self.JR = np.zeros((len(self.Rlist), self.nmagatom, self.nmagatom), dtype=float)
        self.Jnorm_R = np.zeros_like(self.JR)
        self.KR = np.zeros_like(self.JR)
        for iR, R in enumerate(self.Rlist):
            for iq, q in enumerate(self.qpts):
                # MathDoc: tb2j/qspace-finite-grid-transform
                phase = np.exp(-2.0j * np.pi * (R @ q))
                self.JR[iR] += np.real(self.Jq[iq] * phase) / len(self.qpts)
                self.Jnorm_R[iR] += np.real(self.Jnorm_q[iq] * phase) / len(self.qpts)
                self.KR[iR] += np.real(self.Kq[iq] * phase) / len(self.qpts)
            # print(f"{R}: J={self.JR[iR]}, Jnorm{self.Jnorm_R[iR]}")
            # print(f"{R}: {np.sum(self.KR, axis=0)}")
        return self.JR

    def get_Jdict(self):
        self.exchange_Jdict_bruno = {}
        for iR, R in enumerate(self.Rlist):
            R = tuple(R)
            for i, iatom in enumerate(self.ind_mag_atoms):
                for j, jatom in enumerate(self.ind_mag_atoms):
                    val = self.JR[iR, i, j]
                    ispin = self.ispin(iatom)
                    jspin = self.ispin(jatom)
                    keyspin = (R, ispin, jspin)
                    is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                    # The finite-grid transform keeps every R representative;
                    # only the emitted pair table applies the distance cutoff,
                    # matching the real-space route's distance_dict keys.
                    if is_nonself and keyspin in self.distance_dict:
                        sign = np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
                        self.exchange_Jdict[keyspin] = val / sign
                        # CR-007: dual output — the Bruno-renormalized pair
                        # table is emitted separately, never overwriting the
                        # raw J; availability is reported via bruno_meta.
                        self.exchange_Jdict_bruno[keyspin] = (
                            self.Jnorm_R[iR, i, j] / sign
                        )

    def write_output(self, path="TB2J_results"):
        bruno_meta = {
            "longitudinal_correction_applied": getattr(
                self, "longitudinal_correction_applied", False
            ),
            "scope": (
                "Jnorm = Jq + (B C^+ B^T)/2 with B the transverse-longitudinal "
                "response; B is identically zero for collinear references and "
                "its finite-q vertex lands with the basis-phase authority "
                "(story 009), so Jnorm == Jq until then."
            ),
        }
        super().write_output(
            path=path,
            bruno_Jdict=self.exchange_Jdict_bruno,
            bruno_meta=bruno_meta,
        )

    def calculate_raw(self):
        """Return unrenormalized J(q) using a geometric contour, without Bruno."""
        if not isinstance(self.contour, Contour):
            raise ValueError("raw q-space exchange requires a geometric contour")
        self._prepare()
        self.get_all_A()
        raw = self.integrate()
        if not np.isfinite(raw).all():
            raise ValueError(
                "non-finite raw q-space exchange; check the contour for poles"
            )
        return raw

    def calculate_all(self):
        if self.integration_method.lower() == "cfr":
            raise ValueError(
                "CFR is deprecated for the reciprocal-space driver: integrating "
                "Matsubara pole positions with a geometric rule is meaningless. "
                "Use integration_method='cfr2' (the default) or a geometric "
                "contour ('semicircle'/'legendre'); the raw route is "
                "`calculate_raw()`."
            )
        self._prepare()
        self.get_all_A()
        self.integrate()
        self.bruno_renormalize()
        self.q_to_r()
        self.get_Jdict()
