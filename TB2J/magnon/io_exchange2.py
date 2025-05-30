"""
This module provides functionality for handling magnetic exchange interactions and computing magnon band structures.

It includes classes and functions for:
- Reading and writing exchange interaction data
- Computing exchange tensors and magnon energies
- Plotting magnon band structures
- Converting between different magnetic structure representations
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ..mathutils import generate_grid, get_rotation_arrays, round_to_precision, uz
from .plot import BandsPlot
from .structure import BaseMagneticStructure, get_attribute_array

__all__ = [
    "ExchangeIO",
    "plot_tb2j_magnon_bands",
]


def branched_keys(tb2j_keys, npairs):
    """
    Organize TB2J keys into branches based on magnetic site pairs.

    Parameters
    ----------
    tb2j_keys : list
        List of TB2J dictionary keys containing interaction information
    npairs : int
        Number of magnetic site pairs

    Returns
    -------
    list
        List of branched keys organized by magnetic site pairs
    """
    msites = int((2 * npairs) ** 0.5)
    branch_size = len(tb2j_keys) // msites**2
    new_keys = sorted(tb2j_keys, key=lambda x: -x[1] + x[2])[
        (npairs - msites) * branch_size :
    ]
    new_keys.sort(key=lambda x: x[1:])
    bkeys = [
        new_keys[i : i + branch_size] for i in range(0, len(new_keys), branch_size)
    ]

    return [sorted(branch, key=lambda x: np.linalg.norm(x[0])) for branch in bkeys]


def correct_content(content, quadratic=False):
    """
    Ensure content dictionary has all required entries with proper initialization.

    Parameters
    ----------
    content : dict
        Dictionary containing exchange interaction data
    quadratic : bool, optional
        Whether to include biquadratic interactions, by default False
    """
    n = max(content["index_spin"]) + 1
    data_shape = {"exchange_Jdict": ()}

    if not content["colinear"]:
        data_shape |= {"Jani_dict": (3, 3), "dmi_ddict": (3,)}
    if quadratic:
        data_shape["biquadratic_Jdict"] = (2,)

    for label, shape in data_shape.items():
        content[label] |= {((0, 0, 0), i, i): np.zeros(shape) for i in range(n)}


def Hermitize(array):
    """
    Convert an array into its Hermitian form by constructing a Hermitian matrix.

    A Hermitian matrix H has the property that H = H†, where H† is the conjugate transpose.
    This means H[i,j] = conj(H[j,i]) for all indices i,j. The function takes an input array
    representing the upper triangular part of the matrix and constructs the full Hermitian
    matrix by:
    1. Placing the input values in the upper triangular part
    2. Computing the conjugate transpose of these values for the lower triangular part

    This is commonly used in quantum mechanics and magnetic systems where Hamiltonians
    must be Hermitian to ensure real eigenvalues.

    Parameters
    ----------
    array : numpy.ndarray
        Input array containing the upper triangular elements of the matrix.
        Shape should be (n*(n+1)/2, ...) where n is the dimension of
        the resulting square matrix.

    Returns
    -------
    numpy.ndarray
        Full Hermitian matrix with shape (n, n, ...), where n is computed
        from the input array size. The result satisfies result[i,j] = conj(result[j,i])
        for all indices i,j.

    Example
    -------
    >>> arr = np.array([1+0j, 2+1j, 3+0j])  # Upper triangular elements for 2x2 matrix
    >>> Hermitize(arr)
    array([[1.+0.j, 2.+1.j],
           [2.-1.j, 3.+0.j]])
    """
    n = int((2 * array.shape[0]) ** 0.5)
    result = np.zeros((n, n) + array.shape[1:], dtype=complex)
    u_indices = np.triu_indices(n)

    result[*u_indices] = array
    result.swapaxes(0, 1)[*u_indices] = array.swapaxes(-1, -2).conj()

    return result


class ExchangeIO(BaseMagneticStructure):
    """
    Class for handling magnetic exchange interactions and computing magnon properties.

    This class provides functionality for:
    - Managing magnetic structure information
    - Computing exchange tensors
    - Calculating magnon band structures
    - Reading TB2J format files
    - Visualizing magnon bands

    Parameters
    ----------
    atoms : ase.Atoms, optional
        ASE atoms object containing the structure
    cell : array_like, optional
        3x3 matrix defining the unit cell
    elements : list, optional
        List of chemical symbols for atoms
    positions : array_like, optional
        Atomic positions
    magmoms : array_like, optional
        Magnetic moments for each atom
    pbc : tuple, optional
        Periodic boundary conditions, default (True, True, True)
    magnetic_elements : list, optional
        List of magnetic elements in the structure
    kmesh : list, optional
        k-point mesh dimensions, default [1, 1, 1]
    collinear : bool, optional
        Whether the magnetic structure is collinear, default True
    """

    def __init__(
        self,
        atoms=None,
        cell=None,
        elements=None,
        positions=None,
        magmoms=None,
        pbc=(True, True, True),
        magnetic_elements=[],
        kmesh=[1, 1, 1],
        collinear=True,
    ):
        super().__init__(
            atoms=atoms,
            cell=cell,
            elements=elements,
            positions=positions,
            pbc=pbc,
            magmoms=magmoms,
            collinear=collinear,
        )

        self.magnetic_elements = magnetic_elements
        self.kmesh = kmesh

        num_terms = 4 if collinear else 18
        self._exchange_values = np.empty((0, 0, num_terms), dtype=float)

    @property
    def magnetic_elements(self):
        """List of magnetic elements in the structure."""
        return self._magnetic_elements

    @magnetic_elements.setter
    def magnetic_elements(self, value):
        from .structure import validate_symbols

        symbols = validate_symbols(value)
        for symbol in symbols:
            if symbol not in self.elements:
                raise ValueError(f"Symbol '{symbol}' not in 'elements'.")

        self._magnetic_elements = symbols
        self._set_index_pairs()

    @property
    def interacting_pairs(self):
        """List of pairs of interacting magnetic sites."""
        return self._pairs

    def _set_index_pairs(self):
        from itertools import combinations_with_replacement

        magnetic_elements = self.magnetic_elements
        elements = self.elements
        indices = [
            i for i, symbol in enumerate(elements) if symbol in magnetic_elements
        ]
        index_pairs = list(combinations_with_replacement(indices, 2))
        index_spin = np.sort(np.unique(index_pairs))

        self._pairs = index_pairs
        self._index_spin = index_spin

    @property
    def kmesh(self):
        """K-point mesh dimensions for sampling the Brillouin zone."""
        return self._kmesh

    @kmesh.setter
    def kmesh(self, values):
        try:
            the_kmesh = [int(k) for k in values]
        except (ValueError, TypeError):
            raise ValueError("Argument must be an iterable with 'int' elements.")
        if len(the_kmesh) != 3:
            raise ValueError("Argument must be of length 3.")
        if any(k < 1 for k in the_kmesh):
            raise ValueError("Argument must contain only positive numbers.")

        self._kmesh = the_kmesh

    @property
    def vectors(self):
        """Array of interaction vectors between magnetic sites."""
        return self._exchange_values[:, :, :3]

    def set_vectors(self, values=None, cartesian=False):
        """
        Set the interaction vectors between magnetic sites.

        Parameters
        ----------
        values : array_like, optional
            Array of interaction vectors
        cartesian : bool, optional
            Whether the vectors are in Cartesian coordinates, default False
        """
        try:
            pairs = self._pairs
        except AttributeError:
            raise AttributeError("'magnetic_elements' attribute has not been set yet.")
        else:
            n_pairs = len(pairs)

        if values is None:
            i, j = zip(*pairs)
            positions = self.positions
            base_vectors = positions[i, :] - positions[j, :]
            grid = generate_grid(self.kmesh)
            vectors = base_vectors[:, None, :] + grid[None, :, :]
            m_interactions = np.prod(self.kmesh)
        else:
            vectors = get_attribute_array(values, "vectors", dtype=float)
            if vectors.ndim != 3 or vectors.shape[::2] != (n_pairs, 3):
                raise ValueError(
                    f"'vectors' must have the shape (n, m, ), where n={n_pairs} is the number of\n"
                    "pairs of interacting species."
                )
            if cartesian:
                vectors = np.linalg.solve(self.cell.T, vectors.swapaxes(1, -1))
                vectors = vectors.swapaxes(-1, 1)
            m_interactions = vectors.shape[1]

        shape = (
            (n_pairs, m_interactions, 4)
            if self.collinear
            else (n_pairs, m_interactions, 18)
        )
        exchange_values = np.zeros(shape, dtype=float)
        exchange_values[:, :, :3] = vectors
        self._exchange_values = exchange_values

    def _get_neighbor_indices(self, neighbors, tol=1e-4):
        """
        Get indices of neighbor pairs based on distance.

        Parameters
        ----------
        neighbors : list
            List of neighbor shells to consider
        tol : float, optional
            Distance tolerance for neighbor shell assignment, default 1e-4

        Returns
        -------
        tuple
            Indices corresponding to the specified neighbor shells
        """
        distance = np.linalg.norm(self.vectors @ self.cell, axis=-1)
        distance = round_to_precision(distance, tol)
        neighbors_distance = np.unique(np.sort(distance))
        indices = np.where(
            distance[:, :, None] == neighbors_distance[neighbors][None, None, :]
        )

        return indices

    def set_exchange_array(self, name, values, neighbors=None, tol=1e-4):
        """
        Set exchange interaction values for specified neighbors.

        Parameters
        ----------
        name : str
            Type of exchange interaction ('Jiso', 'Biquad', 'DMI', or 'Jani')
        values : array_like
            Exchange interaction values
        neighbors : list, optional
            List of neighbor shells to assign values to
        tol : float, optional
            Distance tolerance for neighbor shell assignment, default 1e-4
        """
        if self.vectors.size == 0:
            raise AttributeError("The intraction vectors must be set first.")

        array = get_attribute_array(values, name, dtype=float)

        if neighbors is not None:
            if len(array) != len(neighbors):
                raise ValueError(
                    "The number of neighbors and exchange values does not coincide."
                )
            *array_indices, value_indices = self._get_neighbor_indices(
                list(neighbors), tol=tol
            )
        else:
            if self._exchange_values.shape[:2] != array.shape[:2]:
                raise ValueError(
                    f"The shape of the array is incompatible with '{self.exchange_values.shape}'"
                )
            array_indices, value_indices = (
                [slice(None), slice(None)],
                (slice(None), slice(None)),
            )

        if name == "Jiso":
            self._exchange_values[*array_indices, 3] = array[value_indices]
        elif name == "Biquad":
            self._exchange_values[*array_indices, 4:6] = array[value_indices]
        elif name == "DMI":
            self._exchange_values[*array_indices, 6:9] = array[value_indices]
        elif name == "Jani":
            self._exchange_values[*array_indices, 9:] = array[value_indices].reshape(
                array.shape[:2] + (9,)
            )
        else:
            raise ValueError(f"Unrecognized exchange array name: '{name}'.")

    @property
    def Jiso(self):
        return self._exchange_values[:, :, 3]

    @property
    def Biquad(self):
        return self._exchange_values[:, :, 4:6]

    @property
    def DMI(self):
        return self._exchange_values[:, :, 6:9]

    @property
    def Jani(self):
        matrix_shape = self._exchange_values.shape[:2] + (3, 3)
        return self._exchange_values[:, :, 9:].reshape(matrix_shape)

    def exchange_tensor(self, anisotropic=True):
        """
        Compute the exchange interaction tensor.

        Parameters
        ----------
        anisotropic : bool, optional
            Whether to include anisotropic interactions, default True

        Returns
        -------
        numpy.ndarray
            Exchange interaction tensor
        """
        shape = self._exchange_values.shape[:2] + (3, 3)
        tensor = np.zeros(shape, dtype=float)

        if anisotropic and not self.collinear:
            tensor += self._exchange_values[:, :, 9:].reshape(shape)
            pos_indices = ([1, 2, 0], [2, 0, 1])
            neg_indices = ([2, 0, 1], [1, 2, 0])
            tensor[:, :, *pos_indices] += self._exchange_values[:, :, 6:9]
            tensor[:, :, *neg_indices] -= self._exchange_values[:, :, 6:9]
        diag_indices = ([0, 1, 2], [0, 1, 2])
        tensor[:, :, *diag_indices] += self._exchange_values[:, :, 3, None]

        return tensor

    def Jq(self, kpoints, anisotropic=True):
        """
        Compute the exchange interactions in reciprocal space.

        Parameters
        ----------
        kpoints : array_like
            k-points at which to evaluate the exchange interactions
        anisotropic : bool, optional
            Whether to include anisotropic interactions, default True

        Returns
        -------
        numpy.ndarray
            Exchange interactions in reciprocal space
        """
        vectors = self._exchange_values[:, :, :3].copy()
        tensor = self.exchange_tensor(anisotropic=anisotropic)

        if self._Q is not None:
            phi = 2 * np.pi * vectors.round(3).astype(int) @ self._Q
            rv = np.einsum("ij,k->ijk", phi, self._n)
            R = (
                Rotation.from_rotvec(rv.reshape(-1, 3))
                .as_matrix()
                .reshape(vectors.shape[:2] + (3, 3))
            )
            np.einsum("nmij,nmjk->nmik", tensor, R, out=tensor)

        exp_summand = np.exp(2j * np.pi * vectors @ kpoints.T)
        Jexp = exp_summand[:, :, :, None, None] * tensor[:, :, None]
        Jq = np.sum(Jexp, axis=1)

        pairs = np.array(self._pairs)
        idx = np.where(pairs[:, 0] == pairs[:, 1])
        Jq[idx] /= 2

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
        if self.collinear:
            magmoms = np.zeros((self._index_spin.size, 3))
            magmoms[:, 2] = self.magmoms[self._index_spin]
        else:
            magmoms = self.magmoms[self._index_spin]
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

    @classmethod
    def load_tb2j(
        cls,
        pickle_file: str = "TB2J.pickle",
        pbc: tuple = (True, True, True),
        anisotropic: bool = False,
        quadratic: bool = False,
    ):
        from pickle import load

        try:
            with open(pickle_file, "rb") as File:
                content = load(File)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No such file or directory: '{pickle_file}'. Please provide a valid .pickle file."
            )
        else:
            correct_content(content)

        magmoms = content["magmoms"] if content["colinear"] else content["spinat"]
        magnetic_elements = {
            content["atoms"].numbers[i]
            for i, j in enumerate(content["index_spin"])
            if j > -1
        }

        exchange = cls(
            atoms=content["atoms"],
            magmoms=magmoms,
            pbc=pbc,
            collinear=content["colinear"],
            magnetic_elements=magnetic_elements,
        )

        num_pairs = len(exchange.interacting_pairs)
        bkeys = branched_keys(content["distance_dict"].keys(), num_pairs)

        vectors = [
            [content["distance_dict"][key][0] for key in branch] for branch in bkeys
        ]
        exchange.set_vectors(vectors, cartesian=True)
        Jiso = [[content["exchange_Jdict"][key] for key in branch] for branch in bkeys]
        exchange.set_exchange_array("Jiso", Jiso)

        if not content["colinear"] and anisotropic:
            Jani = [[content["Jani_dict"][key] for key in branch] for branch in bkeys]
            exchange.set_exchange_array("Jani", Jani)
            DMI = [[content["dmi_ddict"][key] for key in branch] for branch in bkeys]
            exchange.set_exchange_array("DMI", DMI)
        if quadratic:
            Biquad = [
                [content["biquadratic_Jdict"][key] for key in branch]
                for branch in bkeys
            ]
            exchange.set_exchange_array("Biquad", Biquad)

        return exchange


def plot_tb2j_magnon_bands(
    pickle_file: str = "TB2J.pickle",
    path: str = None,
    npoints: int = 300,
    special_points: dict = None,
    anisotropic: bool = False,
    quadratic: bool = False,
    pbc: tuple = (True, True, True),
    filename: str = None,
):
    """
    Load TB2J data and plot magnon band structure in one step.

    This is a convenience function that combines loading TB2J data and plotting
    magnon bands. It first loads the magnetic structure and exchange interactions
    from a TB2J pickle file, then calculates and plots the magnon band structure.

    Parameters
    ----------
    pickle_file : str, optional
        Path to the TB2J pickle file, default "TB2J.pickle"
    path : str, optional
        High-symmetry k-point path for band structure plot
        (e.g., "GXMG" for a square lattice)
    npoints : int, optional
        Number of k-points for band structure calculation, default 300
    special_points : dict, optional
        Dictionary of special k-points for custom paths
    anisotropic : bool, optional
        Whether to include anisotropic interactions, default False
    quadratic : bool, optional
        Whether to include biquadratic interactions, default False
    pbc : tuple, optional
        Periodic boundary conditions, default (True, True, True)
    filename : str, optional
        If provided, save the plot to this file

    Returns
    -------
    exchange : ExchangeIO
        The ExchangeIO instance containing the loaded data and plot

    Example
    -------
    >>> # Basic usage with default parameters
    >>> plot_tb2j_magnon_bands()

    >>> # Custom path and saving to file
    >>> plot_tb2j_magnon_bands(
    ...     path="GXMG",
    ...     anisotropic=True,
    ...     filename="magnon_bands.png"
    ... )
    """
    # Load the TB2J data
    exchange = ExchangeIO.load_tb2j(
        pickle_file=pickle_file, pbc=pbc, anisotropic=anisotropic, quadratic=quadratic
    )

    # Plot the magnon bands
    exchange.plot_magnon_bands(
        path=path, npoints=npoints, special_points=special_points, filename=filename
    )

    return exchange
