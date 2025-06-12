import numpy as np
from scipy.spatial.transform import Rotation

from TB2J.io_exchange import SpinIO

from ..mathutils import Hermitize, get_rotation_arrays, uz
from .plot import BandsPlot


class Magnon:
    """ """

    def __init__(self, exc: SpinIO, iso_only=False, asr=False):
        self.exc = exc
        self.nspin = exc.get_nspin()
        self.ind_atoms = exc.ind_atoms
        self.magmom = np.array([exc.spinat[exc.iatom(i)] for i in range(self.nspin)])
        self.Rlist = exc.Rlist
        self.JR = exc.get_full_Jtensor_for_Rlist(asr=asr, iso_only=iso_only)
        self._Q = None
        self._uz = np.array([[0.0, 0.0, 1.0]], dtype=float)
        self._n = np.array([0, 0, 1], dtype=float)

    def set_propagation_vector(self, Q):
        self._Q = np.array(Q)

    @property
    def Q(self):
        if self._Q is None:
            raise ValueError("Propagation vector Q is not set.")
        return self._Q

    @Q.setter
    def Q(self, value):
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("Propagation vector Q must be a list or numpy array.")
        if len(value) != 3:
            raise ValueError("Propagation vector Q must have three components.")
        self._Q = np.array(value)

    def Jq(self, kpoints):
        """
        Compute the exchange interactions in reciprocal space.

        Parameters
        ----------
        kpoints : array_like
            k-points at which to evaluate the exchange interactions
        """
        Rlist = np.array(self.Rlist)
        JR = self.JR

        for iR, R in enumerate(Rlist):
            if self._Q is not None:
                phi = 2 * np.pi * R @ self._Q
                rv = phi * self._n
                Rmat = Rotation.from_rotvec(rv).as_matrix()
                JR[iR] = np.einsum("rijxy, yz -> rixzy", JR[iR], Rmat)

        nkpt = kpoints.shape[0]
        Jq = np.zeros((nkpt, self.nspin, self.nspin, 3, 3), dtype=complex)

        for iR, R in enumerate(Rlist):
            for iqpt, qpt in enumerate(kpoints):
                phase = 2 * np.pi * R @ qpt
                Jq[iqpt] += np.exp(1j * phase) * JR[iR]

                # Hermitian
                Jq[iqpt, :, :, :, :] += np.conj(Jq[iqpt, :, :, :, :].swapaxes(-1, -2))

        # should we divide

        # if self._Q is not None:
        #    phi = 2 * np.pi * vectors.round(3).astype(int) @ self._Q
        #    rv = np.einsum("ij,k->ijk", phi, self._n)
        #    R = (
        #        Rotation.from_rotvec(rv.reshape(-1, 3))
        #        .as_matrix()
        #        .reshape(vectors.shape[:2] + (3, 3))
        #    )
        #    np.einsum("nmij,nmjk->nmik", tensor, R, out=tensor)

        # exp_summand = np.exp(2j * np.pi * vectors @ kpoints.T)
        # Jexp = exp_summand[:, :, :, None, None] * tensor[:, :, None]
        # Jq = np.sum(Jexp, axis=1)

        # pairs = np.array(self._pairs)
        # idx = np.where(pairs[:, 0] == pairs[:, 1])
        # Jq[idx] /= 2

        return Jq

    def Hq(self, kpoints, anisotropic=True, u=uz):
        """
        Compute the magnon Hamiltonian in reciprocal space.

        Parameters
        ----------
        kpoints : array_like
            k-points at which to evaluate the Hamiltonian
        anisotropic : bool, optional
            Whether to include anisotropic interactions, default True
        u : array_like, optional
            Reference direction for spin quantization axis

        Returns
        -------
        numpy.ndarray
            Magnon Hamiltonian matrix at each k-point
        """
        magmoms = self.magmom.copy()
        magmoms /= np.linalg.norm(magmoms, axis=-1)[:, None]

        U, V = get_rotation_arrays(magmoms, u=u)

        J0 = self.Jq(np.zeros((1, 3)), anisotropic=anisotropic)
        J0 = -Hermitize(J0)[:, :, 0]
        Jq = -Hermitize(self.Jq(kpoints, anisotropic=anisotropic))

        C = np.diag(np.einsum("ix,ijxy,jy->i", V, 2 * J0, V))
        B = np.einsum("ix,ijkxy,jy->kij", U, Jq, U)
        A1 = np.einsum("ix,ijkxy,jy->kij", U, Jq, U.conj())
        A2 = np.einsum("ix,ijkxy,jy->kij", U.conj(), Jq, U)

        return np.block([[A1 - C, B], [B.swapaxes(-1, -2).conj(), A2 - C]])

    def _magnon_energies(self, kpoints, anisotropic=True, u=uz):
        H = self.Hq(kpoints, anisotropic=anisotropic, u=u)
        n = H.shape[-1] // 2
        I = np.eye(n)

        min_eig = 0.0
        try:
            K = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            try:
                K = np.linalg.cholesky(H + 1e-6 * np.eye(2 * n))
            except np.linalg.LinAlgError:
                from warnings import warn

                min_eig = np.min(np.linalg.eigvalsh(H))
                K = np.linalg.cholesky(H - (min_eig - 1e-6) * np.eye(2 * n))
                warn(
                    f"WARNING: The system may be far from the magnetic ground-state. Minimum eigenvalue: {min_eig}. The magnon energies might be unphysical."
                )

        g = np.block([[1 * I, 0 * I], [0 * I, -1 * I]])
        KH = K.swapaxes(-1, -2).conj()

        return np.linalg.eigvalsh(KH @ g @ K)[:, n:] + min_eig

    def get_magnon_bands(
        self,
        kpoints: np.array = np.array([]),
        path: str = None,
        npoints: int = 300,
        special_points: dict = None,
        tol: float = 2e-4,
        pbc: tuple = None,
        cartesian: bool = False,
        labels: list = None,
        anisotropic: bool = True,
        u: np.array = uz,
    ):
        pbc = self._pbc if pbc is None else pbc

        if kpoints.size == 0:
            from ase.cell import Cell

            bandpath = Cell(self._cell).bandpath(
                path=path,
                npoints=npoints,
                special_points=special_points,
                eps=tol,
                pbc=pbc,
            )
            kpoints = bandpath.kpts
            spk = bandpath.special_points
            spk[r"$\Gamma$"] = spk.pop("G", np.zeros(3))
            labels = [
                (i, symbol)
                for symbol in spk
                for i in np.where((kpoints == spk[symbol]).all(axis=1))[0]
            ]
        elif cartesian:
            kpoints = np.linalg.solve(self._cell.T, kpoints.T).T

        bands = self._magnon_energies(kpoints, anisotropic=anisotropic, u=u)

        return labels, bands

    def plot_magnon_bands(self, **kwargs):
        """
        Plot magnon band structure.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to get_magnon_bands and plotting functions
        """
        filename = kwargs.pop("filename", None)
        kpath, bands = self.get_magnon_bands(**kwargs)
        bands_plot = BandsPlot(bands, kpath)
        bands_plot.plot(filename=filename)
