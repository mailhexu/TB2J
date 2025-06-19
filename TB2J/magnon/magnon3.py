from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from TB2J.io_exchange import SpinIO
from TB2J.magnon.magnon_math import get_rotation_arrays
from TB2J.magnon.plot import BandsPlot


@dataclass
class Magnon:
    """
    Magnon calculator implementation using dataclass
    """

    nspin: int
    # ind_atoms: list
    magmom: np.ndarray
    Rlist: np.ndarray
    JR: np.ndarray
    cell: np.ndarray
    _Q: np.ndarray
    _uz: np.ndarray
    _n: np.ndarray
    pbc: tuple = (True, True, True)

    def set_reference(self, Q, uz, n):
        """
        Set reference propagation vector and quantization axis

        Parameters
        ----------
        Q : array_like
            Propagation vector
        uz : array_like
            Quantization axis
        n : array_like
            Normal vector for rotation
        """
        self.set_propagation_vector(Q)
        self._uz = np.array(uz, dtype=float)
        self._n = np.array(n, dtype=float)

    def set_propagation_vector(self, Q):
        """Set propagation vector"""
        self._Q = np.array(Q)

    @property
    def Q(self):
        """Get propagation vector"""
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

        The exchange interactions J(q) are computed using the Fourier transform:
        J(q) = ∑_R J(R) exp(iq·R)

        Array shapes and indices:
        - kpoints: (nkpt, 3) array of k-points
        - Rlist: (nR, 3) array of real-space lattice vectors
        - JR: (nR, nspin, nspin, 3, 3) array of exchange tensors in real space
            where nspin is number of magnetic atoms
        - Output Jq: (nkpt, nspin, nspin, 3, 3) array of exchange tensors in q-space

        If propagation vector Q is set, each J(R) is rotated before the Fourier transform:
        J'_mn(R) = R_m(ϕ)^T J(R) R_n(ϕ)
        where ϕ = 2π R·Q and R(ϕ) is rotation matrix around axis n by angle ϕ

        Parameters
        ----------
        kpoints : array_like (nkpt, 3)
            k-points at which to evaluate the exchange interactions

        Returns
        -------
        numpy.ndarray (nkpt, nspin, nspin, 3, 3)
            Exchange interaction tensors J(q) at each k-point
            First two indices are for magnetic atom pairs
            Last two indices are for 3x3 tensor components
        """
        Rlist = np.array(self.Rlist)
        JR = self.JR
        JRprime = JR.copy()

        for iR, R in enumerate(Rlist):
            if self._Q is not None:
                # Rotate exchange tensors based on propagation vector
                phi = 2 * np.pi * R @ self._Q  # angle ϕ = 2π R·Q
                rv = phi * self._n  # rotation vector
                Rmat = Rotation.from_rotvec(rv).as_matrix()
                # J'_mn(R) = R_m(ϕ)^T J(R) R_n(ϕ) using Einstein summation.
                # Here m is always in the R=0, thus the rotation is only applied on the
                # n , so only on the right.
                JRprime[iR] = np.einsum(" ijxy, yb -> ijxb", JR[iR], Rmat)

        nkpt = kpoints.shape[0]
        Jq = np.zeros((nkpt, self.nspin, self.nspin, 3, 3), dtype=complex)

        for iR, R in enumerate(Rlist):
            for iqpt, qpt in enumerate(kpoints):
                # Fourier transform of exchange tensors
                phase = 2 * np.pi * R @ qpt
                Jq[iqpt] += np.exp(1j * phase) * JRprime[iR]

        Jq_copy = Jq.copy()
        Jq.swapaxes(-1, -2)  # swap xyz
        Jq.swapaxes(-3, -4)  # swap ij
        Jq = (Jq.conj() + Jq_copy) / 2.0
        return Jq

    def Hq(self, kpoints):
        """
        Compute the magnon Hamiltonian in reciprocal space.

        Parameters
        ----------
        kpoints : array_like
            k-points at which to evaluate the Hamiltonian
        anisotropic : bool, optional
            Whether to include anisotropic interactions, default True

        Returns
        -------
        numpy.ndarray
            Magnon Hamiltonian matrix at each k-point
        """
        magmoms = self.magmom.copy()
        magmoms /= np.linalg.norm(magmoms, axis=-1)[:, None]

        U, V = get_rotation_arrays(magmoms, u=self._uz)

        J0 = -self.Jq(np.zeros((1, 3)))[0]
        # J0 = -Hermitize(J0)[:, :, 0]
        # Jq = -Hermitize(self.Jq(kpoints, anisotropic=anisotropic))

        Jq = -self.Jq(kpoints)
        print(f"J0 shape: {J0.shape}")

        C = np.diag(np.einsum("ix,ijxy,jy->i", V, 2 * J0, V))
        B = np.einsum("ix,kijxy,jy->kij", U, Jq, U)
        A1 = np.einsum("ix,kijxy,jy->kij", U, Jq, U.conj())
        A2 = np.einsum("ix,kijxy,jy->kij", U.conj(), Jq, U)

        H = np.block([[A1 - C, B], [B.swapaxes(-1, -2).conj(), A2 - C]])
        print(f"H shape: {H.shape}")
        return H

    def _magnon_energies(self, kpoints, u=None):
        """Calculate magnon energies"""
        H = self.Hq(kpoints)
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
        # Why only n:?
        return np.linalg.eigvalsh(KH @ g @ K)[:, n:] + min_eig
        # return np.linalg.eigvalsh(KH @ g @ K)[:, :] + min_eig

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
        u: np.array = None,
    ):
        """Get magnon band structure"""
        pbc = self.pbc if pbc is None else pbc
        pbc = [True, True, True]
        u = self._uz if u is None else u
        if kpoints.size == 0:
            bandpath = self.cell.bandpath(
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
            kpoints = np.linalg.solve(self.cell.T, kpoints.T).T

        bands = self._magnon_energies(kpoints)
        print(f"bands shape: {bands.shape}")

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

    @classmethod
    def load_from_io(cls, exc: SpinIO, **kwargs):
        """
        Create Magnon instance from SpinIO

        Parameters
        ----------
        exc : SpinIO
            SpinIO instance with exchange parameters
        **kwargs : dict
            Additional arguments passed to get_full_Jtensor_for_Rlist

        Returns
        -------
        Magnon
            Initialized Magnon instance
        """
        # magmoms: magnetic moments of atoms with index in ind_mag_atoms
        index_spin = exc.index_spin
        print(index_spin)
        nspin = exc.nspin
        magmoms = np.zeros((nspin, 3))
        ms = exc.magmoms[index_spin]
        print(f"ms: {ms}")
        if ms.ndim == 1:
            magmoms[:, 2] = np.array(ms)
        print(f"magmoms: {magmoms}")

        cell = exc.atoms.get_cell()
        pbc = exc.atoms.get_pbc()

        return cls(
            nspin=exc.nspin,
            magmom=magmoms,
            Rlist=exc.Rlist,
            JR=exc.get_full_Jtensor_for_Rlist(order="ij33", asr=False, **kwargs),
            cell=cell,
            _Q=np.zeros(3),  # Default propagation vector
            _uz=np.array([[0.0, 0.0, 1.0]]),  # Default quantization axis
            _n=np.array([0.0, 0.0, 1.0]),  # Default rotation axis
            pbc=pbc,
        )

    @classmethod
    def from_TB2J_results(cls, path=None, fname="TB2J.pickle", **kwargs):
        """
        Create Magnon instance from TB2J results.

        Parameters
        ----------
        path : str, optional
            Path to the TB2J results file
        fname : str, optional
            Filename of the TB2J results file, default "TB2J.pickle"
        **kwargs : dict
            Additional arguments passed to load_from_io

        Returns
        -------
        Magnon
            Initialized Magnon instance
        """
        exc = SpinIO.load_pickle(path=path, fname=fname)
        return cls.load_from_io(exc, **kwargs)


def test_magnon(path="TB2J_results"):
    """Test the magnon calculator by loading from TB2J_results and computing at high-symmetry points."""
    from pathlib import Path

    import numpy as np

    # Check if TB2J_results exists
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(f"TB2J_results directory not found at {path}")

    # Load magnon calculator from TB2J results
    print(f"Loading exchange parameters from {path}...")
    magnon = Magnon.from_TB2J_results(path=path, Jiso=True, Jani=False, DMI=False)

    # Define high-symmetry points for a cube
    kpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Γ (Gamma)
            [0.5, 0.0, 0.0],  # X
            [0.5, 0.5, 0.0],  # M
            [0.5, 0.5, 0.5],  # R
        ]
    )
    klabels = ["Gamma", "X", "M", "R"]

    print("\nComputing exchange interactions at high-symmetry points...")
    Jq = magnon.Jq(kpoints)

    print(f"\nResults for {len(kpoints)} k-points:")
    print("-" * 50)
    print("Exchange interactions J(q):")
    print(f"Shape of Jq tensor: {Jq.shape}")
    print(
        f"Dimensions: (n_kpoints={Jq.shape[0]}, n_spin={Jq.shape[1]}, n_spin={Jq.shape[2]}, xyz={Jq.shape[3]}, xyz={Jq.shape[4]})"
    )

    print("\nComputing magnon energies...")
    energies = magnon._magnon_energies(kpoints)

    print("\nMagnon energies at high-symmetry points (in meV):")
    print("-" * 50)
    for i, (k, label) in enumerate(zip(kpoints, klabels)):
        print(f"\n{label}-point k={k}:")
        # print(f"Energies: {energies[i] * 1000:.3f} meV")  # Convert to meV
        print(f"Energies: {energies[i] * 1000} meV")  # Convert to meV

    print("\nPlotting magnon bands...")
    magnon.plot_magnon_bands(
        # kpoints=kpoints,
        # labels=klabels,
        path="GHPGPH,PN",
        filename="magnon_bands.png",
    )

    return magnon, Jq, energies


if __name__ == "__main__":
    # test_magnon(path="/Users/hexu/projects/TB2J_examples/Siesta/bccFe/TB2J_results_sym")
    test_magnon(path="/Users/hexu/projects/TB2J_examples/Siesta/bccFe/TB2J_results")
    # /Users/hexu/projects/TB2J_examples/Wannier/bccFe_QE_Wannier90/TB2J_results
    # test_magnon(path="/Users/hexu/projects/TB2J_examples/Wannier/bccFe_QE_Wannier90/TB2J_results")
