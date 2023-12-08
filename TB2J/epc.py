import numpy as np
from .utils import match_k


class WannierUmat:
    """
    Read Wannier90 Umnk matrix
    For one spin channel only.
    Indices
    k: ikpt
    n: wannier
    m: band

    |mR> = \sum_k exp(-i k R) \sum_n Uknm |nk>
    In matrix language:
      |n, k> is a column vector. Then
      (Uk (n,m) ^T) |nk>
    """

    def __init__(self, Ukmn, kpts):
        self.Uknm = Ukmn
        self.kpts = kpts

    @property
    def nk(self):
        return self.Uknm.shape[0]

    @property
    def nbloch(self):
        return self.Uknm.shape[1]

    @property
    def nwann(self):
        return self.Uknm.shape[2]

    def get_U(self, kpt):
        imatch, _ = match_k(kpt, self.kpts)
        return self.Ukmn[imatch]

    def get_UT(self, kpt):
        imatch, _ = match_k(kpt, self.kpts)
        return self.Ukmn[imatch].T


def EPCMat():
    """
    electron-phonon coupling matrices
    There is no spin index.
    g(ik, iq , v ,m, n)
    where v is phonon band index.
       m and n are electron band index
    kpts : kpoint mesh
    qpts : qpoint mesh
    """

    def __init__(self, g, kpts, qpts):
        self.g = g
        self.kpts = kpts
        self.qpts = qpts
        self.nkpts, self.nqpts, self.nph, self.nband, _ = g.shape

    def _build_electron_Rlist(self):
        # TODO: implement this.
        self.Relec_list = []
        self.nRelec = len(self.Rlist_elec)

    def _build_phonon_Rlist(self):
        # TODO: implement this
        self.Rph_list = []

    def set_electron_Rlist(self, Rlist):
        self.Relec_list = Rlist
        self.nRelec = len(self.Relec_list)

    def set_phonon_Rlist(self, Rlist):
        self.Rph_list = Rlist
        self.nph = len(self.Relec_list)

    @staticmethod
    def read_from_file(self, fname):
        pass

    def to_ewannk_pqv(self, Umat, Rmesh):
        """
        from band representation to wannier representation.
        g_ewann[q, v, m, n, Rn]: <m, 0| q, v| n, R>
        """
        self._build_electron_Rlist(Rmesh)
        nwann = Umat.nwann
        g_ewannk = np.zeros(
            (
                self.nqpts,
                self.nph,
                self.nkpts,
                nwann,
                nwann,
            ),
            dtype=complex,
        )
        for ik, k in enumerate(self.kpts):
            Uk = Umat.get_U(k)
            for iq, q in enumerate(self.qpts):
                Ukqdagger = Umat.get_U(k + q).T.conj()
                for iv, v in enumerate(self.vlist):
                    g_ewannk[iq, iv, ik, :, :] = Ukqdagger.dot(
                        self.g[iq, iv, ik, :, :]
                    ).dot(Uk)
        return g_ewannk

    def to_ewannR_pqv(self, Umat):
        """
        g_wann(k, q, v) =
        U^\dagger(k, Wm, Bk) g(k, q, v, Bo, Bp ) U(k+q, W  )
        """
        nwann = Umat.nwann
        g_ewannR = np.zeros((self.nqpts, self.nph, self.nRelec, nwann, nwann))

        g_ewannk = self.to_ewannk_phbloch()
        for iR, R in enumerate(self.Relect_list):
            for ik, k in enumerate(self.klist):
                phase = np.exp(-2j * np.pi * np.dot(k, R))
                g_ewannR[:, :, iR, :, :] += (
                    self.kweight[:] * phase * g_ewannk[:, :, ik, :, :]
                )

    def to_ewannR_pRv(self, Lmat=None):
        """
        from phonon wannier  function to displacement.
        Lmat: matrix to transfer atomic displacement to lattice wannier function.
        """
        # TODO: implement this
