from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase.dft.dos import DOS
from ase.units import J, eV
from scipy.spatial.transform import Rotation

from TB2J.io_exchange import SpinIO
from TB2J.kpoints import monkhorst_pack
from TB2J.magnon.magnon_band import MagnonBand
from TB2J.magnon.magnon_math import get_rotation_arrays
from TB2J.magnon.magnon_parameters import (
    MagnonParameters,
    add_band_specific_args,
    add_common_magnon_args,
    prepare_magnon_from_params,
)
from TB2J.mathutils.auto_kpath import auto_kpath

try:
    from ase.cell import Cell as AseCell
except ImportError:  # for compatibility with older ASE layout
    from ase import Cell as AseCell


def _normalize_kpoint_label(label: str) -> str:
    """Normalize Gamma variants to LaTeX display text."""
    if label in {"G", "Gamma", r"$\Gamma$"}:
        return r"$\Gamma$"
    return label


@dataclass
class Magnon:
    """
    Magnon calculator implementation using dataclass
    """

    nspin: int
    magmom: np.ndarray
    Rlist: np.ndarray
    JR: np.ndarray
    cell: np.ndarray
    _Q: np.ndarray
    _uz: np.ndarray
    _n: np.ndarray
    pbc: tuple = (True, True, True)
    # Optional: set when the Magnon is built from a supercell SpinIO so that
    # the primitive-cell BZ k-path can be used and folded back.
    primitive_cell: np.ndarray = None
    supercell_matrix: np.ndarray = None

    def set_reference(self, Q, uz, n, magmoms=None):
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
        axis = np.asarray(n, dtype=float)
        if axis.shape != (3,):
            raise ValueError("Rotation axis n must have three components.")
        if not np.all(np.isfinite(axis)):
            raise ValueError("Rotation axis n must contain only finite values.")
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("Rotation axis n must be non-zero.")
        self._uz = np.array(uz, dtype=float)
        self._n = axis / axis_norm
        if magmoms is not None:
            self.magmom = np.array(magmoms, dtype=float)
        self.Snorm = np.linalg.norm(self.magmom, axis=1) / 2
        # self.Snorm=np.ones(self.nspin)

    def set_propagation_vector(self, Q):
        """Set the known reference ordering/propagation vector."""
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (3,):
            raise ValueError("Propagation vector Q must have three components.")
        if not np.all(np.isfinite(Q)):
            raise ValueError("Propagation vector Q must contain only finite values.")
        self._Q = Q

    @property
    def Q(self):
        """Get propagation vector"""
        if self._Q is None:
            raise ValueError("Propagation vector Q is not set.")
        return self._Q

    @Q.setter
    def Q(self, value):
        self.set_propagation_vector(value)

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
        Snorm_inv = 1 / self.Snorm
        JR = np.einsum("rijxy, i, j-> rijxy", self.JR, Snorm_inv, Snorm_inv)
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
                Jq[iqpt] += np.exp(-1j * phase) * JRprime[iR]

        # Jq_copy = Jq.copy()
        # Jq.swapaxes(-1, -2)  # swap xyz
        # Jq.swapaxes(-3, -4)  # swap ij
        # Jq = (Jq.conj() + Jq_copy) / 2.0
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
        Jmq = Jq.swapaxes(-1, -2).swapaxes(1, 2)

        Ssqrt = np.sqrt(self.Snorm)

        # A1 = UT J(-k) U.conj()
        A1 = np.einsum("ix,kijxy,jy->kij", U, Jmq, U.conj())
        # A2 = [ UT J(-k) U.conj() ] .conj() = U.conj()^T J(k).conj() U
        A2 = np.einsum("ix,kijxy,jy->kij", U.conj(), Jq.conj(), U)
        # B = UT J(-k) U
        B = np.einsum("ix,kijxy,jy->kij", U, Jmq, U)

        # debug: check B(q) == B(-q).T
        # Jmq = -self.Jq(-kpoints)
        # B2 = np.einsum("jx,kjixy,iy->kij", U, Jmq, U)
        # if not np.allclose(B, B2):
        #    print("Warning: B and B2 are not consistent!")
        #    print(f"Max difference: {np.max(np.abs(B - B2))}")

        # C = \delta_ij \sum_l (S_l v_i^T J'_{il} v_l)
        C = np.diag(np.einsum("ix,ijxy,jy, j->i", V, 2 * J0, V, self.Snorm))

        # Apply the sqrt(Si Sj) ans Sl
        A1 = np.einsum("kij, i, j-> kij", A1, Ssqrt, Ssqrt)
        A2 = np.einsum("kij, i, j-> kij", A2, Ssqrt, Ssqrt)
        B = np.einsum("kij, i, j-> kij", B, Ssqrt, Ssqrt)

        H = np.block([[A1 - C, B], [B.swapaxes(-1, -2).conj(), A2 - C]])
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
        use_primitive_kpath: bool = False,
    ):
        """Get magnon band structure.

        Parameters
        ----------
        kpoints : np.array, optional
            Explicit k-points to calculate bands at. If empty, generates k-points from path.
        path : str, optional
            String specifying the k-path. If None, generates automatically using auto_kpath.
        npoints : int, optional
            Number of k-points along the path. Default is 300.
        special_points : dict, optional
            Dictionary of special points coordinates.
        tol : float, optional
            Tolerance for k-point comparisons. Default is 2e-4.
        pbc : tuple, optional
            Periodic boundary conditions. Default is None.
        cartesian : bool, optional
            Whether k-points are in cartesian coordinates. Default is False.
        labels : list, optional
            List of k-point labels. Default is None.
        anisotropic : bool, optional
            Whether to include anisotropic interactions. Default is True.
        u : np.array, optional
            Quantization axis. Default is None.
        use_primitive_kpath : bool, optional
            If True and this Magnon was built from a supercell (i.e.
            ``self.primitive_cell`` and ``self.supercell_matrix`` are set),
            generate the high-symmetry k-path from the primitive cell's BZ
            and fold the k-points into the supercell reciprocal lattice via
            ``k_sc = k_prim @ sc_matrix.T``.  Default is False.

        Returns
        -------
        tuple
            - labels : list of (index, name) tuples for special k-points
            - bands : array of band energies
            - xlist : list of arrays with x-coordinates for plotting (if using auto_kpath)
        """
        pbc = self.pbc if pbc is None else pbc
        pbc = [True, True, True]
        u = self._uz if u is None else u
        xlist = None  # default; overwritten when auto_kpath segments are used

        # Determine whether to use the primitive-cell BZ for k-path generation.
        _use_prim = (
            use_primitive_kpath
            and self.primitive_cell is not None
            and self.supercell_matrix is not None
        )

        if kpoints.size == 0:
            if _use_prim:
                # Build k-path in the primitive BZ, then fold to supercell
                # fractional coordinates.  The primitive reciprocal lattice
                # For a supercell A_sc = S @ A_prim, a supercell translation
                # R_sc has primitive coordinates R_sc @ S.  Since Jq uses the
                # phase R_sc · k_sc, primitive fractional k-points fold as
                # k_sc = S @ k_prim, i.e. row-vector k_prim @ S.T.
                prim_cell_array = np.array(self.primitive_cell.get_cell())
                sc_mat = np.array(self.supercell_matrix, dtype=float)
                fold_matrix = sc_mat.T  # row-vector k_sc = k_prim @ fold_matrix

                if path is None:
                    xlist, kptlist, Xs, knames, spk = auto_kpath(
                        prim_cell_array,
                        None,
                        npoints=npoints,
                        supercell_matrix=fold_matrix,
                    )
                    kpoints = np.concatenate(kptlist)
                    # Use Xs (real distances) as label positions so they align
                    # with the xlist real-distance x-axis.
                    labels = [
                        (x, _normalize_kpoint_label(name))
                        for x, name in zip(Xs.tolist(), knames)
                    ]
                else:
                    prim_ase_cell = AseCell(prim_cell_array)
                    bandpath = prim_ase_cell.bandpath(
                        path=path,
                        npoints=npoints,
                        special_points=special_points,
                        eps=tol,
                        pbc=pbc,
                    )
                    k_prim = bandpath.kpts
                    kpoints = k_prim @ fold_matrix
                    spk_prim = bandpath.special_points
                    spk_prim["Gamma"] = spk_prim.pop("G", np.zeros(3))
                    spk_prim = {
                        _normalize_kpoint_label(name): value
                        for name, value in spk_prim.items()
                    }
                    spk = {name: v @ fold_matrix for name, v in spk_prim.items()}
                    labels = [
                        (i, symbol)
                        for symbol in spk
                        for i in np.where(
                            np.all(np.abs(kpoints - spk[symbol]) < tol, axis=1)
                        )[0]
                    ]
                    labels = [
                        (i, _normalize_kpoint_label(symbol)) for i, symbol in labels
                    ]
                    xlist = None
            elif path is None:
                # Use auto_kpath to generate path automatically
                xlist, kptlist, Xs, knames, spk = auto_kpath(
                    self.cell, None, npoints=npoints
                )
                kpoints = np.concatenate(kptlist)
                # Use Xs (real distances) as label positions so they align
                # with the xlist real-distance x-axis.
                labels = [
                    (x, _normalize_kpoint_label(name))
                    for x, name in zip(Xs.tolist(), knames)
                ]
            else:
                bandpath = self.cell.bandpath(
                    path=path,
                    npoints=npoints,
                    special_points=special_points,
                    eps=tol,
                    pbc=pbc,
                )
                kpoints = bandpath.kpts
                spk = bandpath.special_points
                spk["Gamma"] = spk.pop("G", np.zeros(3))
                spk = {
                    _normalize_kpoint_label(name): value for name, value in spk.items()
                }
                labels = [
                    (i, symbol)
                    for symbol in spk
                    for i in np.where((kpoints == spk[symbol]).all(axis=1))[0]
                ]
                labels = [(i, _normalize_kpoint_label(symbol)) for i, symbol in labels]
                xlist = None
        elif cartesian:
            kpoints = np.linalg.solve(self.cell.T, kpoints.T).T
            xlist = None

        bands = self._magnon_energies(kpoints)
        # print(f"bands shape: {bands.shape}")

        return labels, bands, xlist

    def plot_magnon_bands(self, **kwargs):
        """
        Plot magnon band structure.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to get_magnon_bands and plotting functions.
            Supported plotting options:
            - filename : str, optional
                Output filename for saving the plot
            - ax : matplotlib.axes.Axes, optional
                Axes for plotting. If None, creates new figure
            - show : bool, optional
                Whether to show the plot on screen
        """
        filename = kwargs.pop("filename", None)
        show = kwargs.pop("show", True)
        ax = kwargs.pop("ax", None)
        use_primitive_kpath = kwargs.get("use_primitive_kpath", False)
        kpath_labels, bands, xlist = self.get_magnon_bands(**kwargs)

        # Reconstruct k-points and special points for MagnonBand.
        # Must match exactly what get_magnon_bands used, including the
        # primitive-kpath folding when use_primitive_kpath=True.
        _use_prim = (
            use_primitive_kpath
            and self.primitive_cell is not None
            and self.supercell_matrix is not None
        )
        npoints = kwargs.get("npoints", 300)
        path = kwargs.get("path", None)
        if _use_prim:
            prim_cell_array = np.array(self.primitive_cell.get_cell())
            sc_mat = np.array(self.supercell_matrix, dtype=float)
            fold_matrix = sc_mat.T
            if path is None:
                _, kptlist, _, _, spk_prim = auto_kpath(
                    prim_cell_array,
                    None,
                    npoints=npoints,
                    supercell_matrix=fold_matrix,
                )
                kpoints = np.concatenate(kptlist)
                spk = {
                    _normalize_kpoint_label(name): v @ fold_matrix
                    for name, v in spk_prim.items()
                }
            else:
                bandpath = AseCell(prim_cell_array).bandpath(
                    path=path,
                    npoints=npoints,
                    pbc=[True, True, True],
                )
                spk_prim = bandpath.special_points.copy()
                spk_prim["Gamma"] = spk_prim.pop("G", np.zeros(3))
                spk_prim = {
                    _normalize_kpoint_label(name): value
                    for name, value in spk_prim.items()
                }
                kpoints = bandpath.kpts @ fold_matrix
                spk = {name: v @ fold_matrix for name, v in spk_prim.items()}
        elif path is None:
            _, kptlist, _, _, spk = auto_kpath(self.cell, None, npoints=npoints)
            kpoints = np.concatenate(kptlist)
        else:
            bandpath = self.cell.bandpath(path=path, npoints=npoints)
            kpoints = bandpath.kpts
            spk = bandpath.special_points.copy()
            spk["Gamma"] = spk.pop("G", np.zeros(3))
            spk = {_normalize_kpoint_label(name): value for name, value in spk.items()}

        bands_plot = MagnonBand(
            energies=bands * 1000,  # Convert to meV
            kpoints=kpoints,
            kpath_labels=kpath_labels,
            special_points=spk,
            xcoords=None,  # use integer indices; xlist from auto_kpath not directly usable here
        )

        return bands_plot.plot(filename=filename, show=show, ax=ax)

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
        # print(f"kwargs={kwargs}")
        # Get magnetic moments for magnetic atoms
        magmoms = exc.get_magnetic_moments()
        # nspin = len(magmoms)  # Number of magnetic atoms

        cell = exc.atoms.get_cell()
        pbc = exc.atoms.get_pbc()

        # Extract SIA from kwargs, default True if not present
        include_SIA = kwargs.pop("SIA", True)

        JR = exc.get_full_Jtensor_for_Rlist(
            order="ij33", asr=False, SIA=include_SIA, **kwargs
        )
        # print(JR)

        return cls(
            nspin=exc.nspin,
            magmom=magmoms,
            Rlist=exc.Rlist,
            JR=JR,
            cell=cell,
            _Q=np.zeros(3),  # Default propagation vector
            _uz=np.array([[0.0, 0.0, 1.0]]),  # Default quantization axis
            _n=np.array([1.0, 0.0, 0.0]),  # Default rotation axis
            pbc=pbc,
            primitive_cell=getattr(exc, "primitive_cell", None),
            supercell_matrix=getattr(exc, "supercell_matrix", None),
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


def create_plot_script(base_name: str, output_dir: str = "."):
    """Create a Python script for plotting the saved band structure data.

    Parameters
    ----------
    base_name : str
        Base name (without extension) for the default data and plot files
    output_dir : str
        Directory to write the script to (default: current directory)
    """
    from pathlib import Path

    script_path = Path(output_dir) / "plot_magnon_band.py"
    script = f'''#!/usr/bin/env python3
"""Simple script to plot magnon band structure from saved data."""

from TB2J.magnon.magnon_band import MagnonBand
import matplotlib.pyplot as plt

def plot_magnon_bands(input_file="{base_name}.json", output_file="{base_name}.png", ax=None, color='blue', show=True):
    """Load and plot magnon band structure.
    
    Parameters
    ----------
    input_file : str
        JSON file containing band structure data
    output_file : str, optional
        Output file for saving the plot
    ax : matplotlib.axes.Axes, optional
        Axes for plotting. If None, creates new figure
    color : str, optional
        Color of the band lines (default: blue)
    show : bool, optional
        Whether to show the plot on screen (default: True)
    
    Returns
    -------
    matplotlib.axes.Axes
        The plotting axes
    """
    # Load band structure data
    bands = MagnonBand.load(input_file)
    
    # Create plot
    ax = bands.plot(
        ax=ax,
        filename=output_file,
        color=color,
        show=show
    )
    return ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a figure and axis (optional)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot bands with custom color on given axis
    plot_magnon_bands(
        input_file="{base_name}.json",
        output_file="{base_name}.png",
        ax=ax,
        color='red',
        show=True
    )
'''

    with open(script_path, "w") as f:
        f.write(script)

    import os

    os.chmod(script_path, 0o755)  # Make executable


def save_bands_data(
    kpoints: np.ndarray,
    energies: np.ndarray,
    kpath_labels: List[Tuple[int, str]],
    special_points: dict,
    xcoords: Optional[Union[np.ndarray, List[np.ndarray]]],
    filename: str,
):
    """Save magnon band structure data to a JSON file using MagnonBand class.

    Parameters
    ----------
    kpoints : array_like
        Array of k-points coordinates
    energies : array_like
        Array of band energies (in meV)
    kpath_labels : list of (int, str)
        List of tuples containing k-point indices and their labels
    special_points : dict
        Dictionary of special points and their coordinates
    xcoords : array_like or list of arrays
        x-coordinates for plotting (can be segmented)
    filename : str
        Output filename
    """
    from TB2J.magnon.magnon_band import MagnonBand  # Using same import as above

    bands = MagnonBand(
        energies=energies,
        kpoints=kpoints,
        kpath_labels=kpath_labels,
        special_points=special_points,
        xcoords=xcoords,
    )
    bands.save(filename)

    # Create plotting script in the same directory as the data file
    from pathlib import Path

    filepath = Path(filename)
    script_name = filepath.parent / "plot_magnon_band.py"
    create_plot_script(filepath.stem, str(filepath.parent))

    print("Generated magnon band files:")
    print(f"  data: {filename}")
    print(f"  plotting script: {script_name}")
    print(f"Run {script_name} to replot the saved band data.")

    return bands


def plot_magnon_bands_from_TB2J(
    params: MagnonParameters,
):
    """
    Load TB2J results and plot magnon band structure along a specified k-path.

    Parameters
    ----------
    path : str, optional
        Path to TB2J results directory, default is "TB2J_results"
    kpath : str, optional
        String specifying the k-path, e.g. "GXMR" for Gamma-X-M-R path
        Default is "GXMR"
    npoints : int, optional
        Number of k-points along the path, default is 300
    filename : str, optional
        Output file name for the plot, default is "magnon_bands.png"
    Jiso : bool, optional
        Include isotropic exchange interactions, default is True
    Jani : bool, optional
        Include anisotropic exchange interactions, default is False
    DMI : bool, optional
        Include Dzyaloshinskii-Moriya interactions, default is False
    Q : array-like, optional
        Propagation vector [Qx, Qy, Qz], default is [0, 0, 0]
    uz_file : str, optional
        Path to file containing quantization axes for each spin (natom×3 array)
        If not provided, default [0, 0, 1] will be used for all spins
    n : array-like, optional
        Normal vector for rotation [nx, ny, nz], default is [0, 0, 1]
    show: bool, optional
        whether to show figure.

    Returns
    -------
    magnon : Magnon
        The Magnon instance used for calculations
    """
    magnon = prepare_magnon_from_params(params)

    # Get band structure data
    print(f"\nCalculating bands along path {params.kpath}...")
    kpath_labels, bands, xlist = magnon.get_magnon_bands(
        path=params.kpath,
        npoints=params.npoints,
        special_points=params.qpoints if params.qpoints else None,
        use_primitive_kpath=getattr(params, "use_primitive_kpath", False),
    )

    # Convert energies to meV
    bands_meV = bands * 1000

    # Save band structure data and create plot
    data_file = params.filename.rsplit(".", 1)[0] + ".json"
    print(f"\nSaving band structure data to {data_file}")

    # Get k-points and special points (mirrors logic in get_magnon_bands)
    _use_prim = (
        getattr(params, "use_primitive_kpath", False)
        and magnon.primitive_cell is not None
        and magnon.supercell_matrix is not None
    )
    if _use_prim:
        prim_cell_array = np.array(magnon.primitive_cell.get_cell())
        sc_mat = np.array(magnon.supercell_matrix, dtype=float)
        fold_matrix = sc_mat.T
        if params.kpath is None:
            _, kptlist, _, _, spk_prim = auto_kpath(
                prim_cell_array,
                None,
                npoints=params.npoints,
                supercell_matrix=fold_matrix,
            )
            kpoints = np.concatenate(kptlist)
            spk_prim = {
                _normalize_kpoint_label(name): value for name, value in spk_prim.items()
            }
        else:
            bp = AseCell(prim_cell_array).bandpath(
                path=params.kpath,
                npoints=params.npoints,
                special_points=params.qpoints if params.qpoints else None,
            )
            k_prim = bp.kpts
            kpoints = k_prim @ fold_matrix
            spk_prim = bp.special_points
            spk_prim["Gamma"] = spk_prim.pop("G", np.zeros(3))
            spk_prim = {
                _normalize_kpoint_label(name): value for name, value in spk_prim.items()
            }
        spk = {name: v @ fold_matrix for name, v in spk_prim.items()}
    elif params.kpath is None:
        _, kptlist, _, _, spk = auto_kpath(magnon.cell, None, npoints=params.npoints)
        kpoints = np.concatenate(kptlist)
        spk = {_normalize_kpoint_label(name): value for name, value in spk.items()}
    else:
        bandpath = magnon.cell.bandpath(
            path=params.kpath,
            npoints=params.npoints,
            special_points=params.qpoints if params.qpoints else None,
        )
        kpoints = bandpath.kpts
        spk = bandpath.special_points
        spk["Gamma"] = spk.pop("G", np.zeros(3))
        spk = {_normalize_kpoint_label(name): value for name, value in spk.items()}

    magnon_bands = save_bands_data(
        kpoints=kpoints,
        energies=bands_meV,
        kpath_labels=kpath_labels,
        special_points=spk,
        xcoords=xlist,
        filename=data_file,
    )

    # Plot band structure
    print(f"Plotting bands to {params.filename}")
    magnon_bands.plot(filename=params.filename, show=params.show)

    return magnon


def plot_magnon_bands_cli():
    import argparse
    import warnings

    warnings.warn(
        """ 
        # !!!!!!!!!!!!!!!!!! WARNING: =============================
        # 
        # This functionality is under development and should not be used in production.
        # It is provided for testing and development purposes only.
        # Please use with caution and report any issues to the developers.
        #
        # This warning will be removed in future releases.
        # =====================================

        """,
        UserWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Plot magnon band structure from TB2J results"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=str,
        help="Path to TOML configuration file",
    )
    group.add_argument(
        "--save-config",
        type=str,
        help="Save default configuration to specified TOML file",
    )

    add_common_magnon_args(parser)
    add_band_specific_args(parser)

    args = parser.parse_args()

    if args.save_config:
        params = MagnonParameters()
        params.to_toml(args.save_config)
        print(f"Saved default configuration to {args.save_config}")
        return

    if args.config:
        params = MagnonParameters.from_toml(args.config)
    else:
        params = MagnonParameters(
            path=args.path,
            kpath=args.kpath,
            npoints=args.npoints,
            filename=args.output,
            Jiso=args.Jiso,
            Jani=args.Jani,
            SIA=args.SIA,
            DMI=args.DMI,
            Q=args.ordering_vector,
            uz_file=args.uz_file,
            spin_conf_file=args.spin_conf_file,
            n=args.rotation_axis,
            show=args.show,
        )

    plot_magnon_bands_from_TB2J(params)


def main():
    plot_magnon_bands_cli()


class MagnonASEWrapper:
    def __init__(self, magnon: Magnon):
        self.magnon = magnon
        self.atoms = None
        self.kpts = None
        self.weights = None
        self.dos_args = {}

    def set(self, atoms=None, kmesh=[9, 9, 9], gamma=True, **kwargs):
        self.atoms = atoms
        self.dos_args = {
            "kmesh": kmesh,
            "gamma": gamma,
        }
        self.kpts = monkhorst_pack(
            self.dos_args["kmesh"], gamma_center=self.dos_args["gamma"]
        )
        self.weights = np.ones(len(self.kpts)) / len(self.kpts)

    def get_k_points_and_weights(self):
        return self.kpts, self.weights

    def get_k_point_weights(self):
        return self.weights

    def get_number_of_spins(self):
        return 1

    def get_eigenvalues(self, kpt, spin=0):
        """
        return the eigenvalues at a given k-point. The energy unit is eV
        args:
            kpt: k-point index.
            spin: spin index.
        """
        kpoint = self.kpts[kpt]
        # Magnon energies are already in eV, convert to meV for consistency with plot
        evals = self.magnon._magnon_energies(np.array([kpoint]))[0]
        evals = evals * J / eV  # Convert to eV
        return evals

    def get_fermi_level(self):
        return 0.0

    def get_bz_k_points(self):
        return self.kpts

    def get_dos(self, width=0.1, window=None, npts=401):
        dos = DOS(self, width=width, window=window, npts=npts)
        energies = dos.get_energies()
        tdos = dos.get_dos()
        return energies, tdos

    def plot_dos(
        self,
        smearing_width=0.0001,
        window=None,
        npts=401,
        output="magnon_dos.pdf",
        ax=None,
        show=True,
        dos_filename="magnon_dos.txt",
    ):
        """
        plot total DOS.
        :param width: width of Gaussian smearing
        :param window: energy window
        :param npts: number of points
        :param output: output filename
        :param ax: matplotlib axis
        :return: ax
        """
        if ax is None:
            _fig, ax = plt.subplots()
        energies, tdos = self.get_dos(width=smearing_width, window=window, npts=npts)
        energies = energies * 1000  # Convert to meV
        tdos = tdos / 1000  # Convert to states/meV
        if dos_filename is not None:
            np.savetxt(
                dos_filename,
                np.array([energies, tdos]).T,
                header="Energy(meV) DOS(state/meV)",
            )
        ax.plot(energies, tdos)
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("DOS (states/meV)")
        ax.set_title("Total DOS")
        if output is not None:
            plt.savefig(output)
        if show:
            plt.show()
        return ax
