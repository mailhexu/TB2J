import numpy as np
from typing import Tuple
from dataclasses import dataclass
from threadpoolctl import threadpool_limits

from TB2J.pauli import pauli_sigma_norm, pauli_block_all

@dataclass(frozen=True)
class GreenContext:
    efermi: float
    evals: np.ndarray
    evecs_path: str
    atom_indices: tuple
    energies: np.ndarray
    eweights: np.ndarray
    norb: int
    iorbs: Tuple[np.ndarray, ...]
    kpts: np.ndarray
    k2Rfactor: float
    kweights: np.ndarray
    Rvecs: np.ndarray
    Pmatrix: Tuple[np.ndarray, ...]

class GreenRuntime:
    """
    This class implements the Green’s-function formalism used to
    compute the A_ij exchange tensors. The calculations are fully 
    vectorized with respect to energy, real-space lattice vectors,
    and reciprocal-space sampling. To facilitate efficient parallel
    execution, the class stores minimal information required for 
    these operations and is intended to be instantiated as a 
    lightweight runtime object within worker processes.

    Parameters
    ----------
    ctx: GreenContext
        Dataclass to construct attributes
    thlim: int
        Threading upper bound
    
    Attributes
    ----------
    efermi : float
        Fermi energy
    evals : np.ndarray
        Eigenvalues from the TB H(k) Hamiltonian
    atom_indices : tuple
        Indices of magnetic sites
    energies : np.ndarray
        Energy values to evaluate the Green's function
    eweights : np.ndarray
        Energy weights of the integration contour
    norb : int
        Total number of orbitals
    iorbs : tuple(np.ndarray)
        Orbital indices of each magnetic site
    kpts : np.ndarray
        K-vectors inside the first Brilluoin zone
    k2Rfactor : float
        Factor of 2*pi*j
    kweights : np.ndarray
        K-point weights for summations in reciprocal space
    Rvecs : np.ndarray
        Lattice positions
    Pmatrix : tuple(np.ndarray)
        Projector matrix of each magnetic site
    thlim : int
        Maximum number of threads to be used by NumPy/BLAS
    """
    def __init__(self, ctx: GreenContext, thlim=None):

        self.efermi = ctx.efermi
        self.evals = ctx.evals
        self.atom_indices = ctx.atom_indices
        self.norb = ctx.norb
        self.iorbs = ctx.iorbs
        self.kpts = ctx.kpts
        self.k2Rfactor = ctx.k2Rfactor
        self.kweights = ctx.kweights
        self.Rvecs = ctx.Rvecs
        self.energies = ctx.energies
        self.eweights = ctx.eweights
        self.P = ctx.Pmatrix

        self.thlim = thlim
        self.set_eigenvectors(ctx.evecs_path)

    def set_eigenvectors(self, path):

        shape = (len(self.Rvecs), self.norb, self.evals.shape[-1])
        self.evecs = np.memmap(
                path, 
                dtype=np.complex128, 
                shape=shape,
                mode='r'
        )

    def get_Gk(self, energy, idx=slice(None), jdx=slice(None)):
        """
        Green's function G(k) for multiple energies for all kpoints

        Parameters
        ----------
        energy : float or ndarray
            Energy values
        idx : int or slice
            Index (i) of G_ij(k)
        jdx : int or slice
            Index (j) of G_ij(k)

        Returns
        -------
        Gij(k) : ndarray
            Green's function G_ij(k) at given energis,
            projected onto indices (i, j)
        """
        evals = self.evals[..., None]
        V = self.evecs[..., idx, :]
        Vh = self.evecs[..., jdx, :]
        Vh = Vh.swapaxes(-1, -2).conj()
        energy = energy[..., None, None, None]
        middle = 1.0 / (energy + self.efermi - evals)
        Gk = V @ (middle * Vh)

        return Gk

    def get_GR(self, 
            energy, 
            idx=slice(None), 
            jdx=slice(None),
            Gk=None
        ):
        """
        Green's function at multiple energies for all R points.
        G_ij(R, epsilon) = G_ij(k, epsilon) exp(-2\pi i R.dot. k)

        Parameters
        ----------
        energy : ndarray
            Energy values
        idx : int or slice
            Index (i) of G_ij(R)
        jdx : int or slice
            Index (j) of G_ij(R)
        Gk_all : ndarray
            Optional pre-compute G_ij(k) for all k-points

        Returns
        -------
            Green's function G_ij(R) evaluated at given energies
            and real-space points, projected onto indices (i, j).
            Shape of (nenergies, nR, len(idx), len(jdx))
        """
        if Gk is None:
            Gk = self.get_Gk(energy, idx=idx, jdx=jdx)

        phase = np.einsum(
            '...ni,...mi->...nm', self.Rvecs, self.kpts
        )
        expvals = self.kweights * np.exp(self.k2Rfactor * phase)
        GR = np.einsum(
            '...kij,...rk->...rij', Gk, expvals,
            optimize='optimal'
        )

        return GR

    def integrate(self, values, contour_method='cfr'):
        '''Integrate along energy path'''
        result = np.einsum('i...,i->...', values, self.eweights)
        if contour_method == 'cfr':
            result *= -np.pi / 2

        return result

    def _compute_Aij(self, i, j, orb_decomposition=False):
        """
        Computes the A_ij tensor along magnetic site indices 
        i and j. It internally performs the energy integral of
        the Green's function.

        Parameters
        ----------
            i : int
                Magnetic site index i
            j : int 
                Magnetic site index j
            orb_decomposition : bool, optional
                Wether to obtain the orbital decomposed A_ij tensor

        Returns
        -------
            A_ij : ndarray
                A_ij tensor with shape (nR, 4, 4) where nR is the
                number of lattice vectors
            A_ij_orb : ndarray (optional)
                Orbital decomposition of A_ij. Only produced if
                self.orb_decomposition == True
        """

        relative_index_i = self.atom_indices.index(i)
        relative_index_j = self.atom_indices.index(j)

        idx = self.iorbs[relative_index_i]
        jdx = self.iorbs[relative_index_j]
        Gij = self.get_GR(self.energies, idx=idx, jdx=jdx)
        Gji = self.get_GR(self.energies, idx=jdx, jdx=idx)

        Gij = pauli_block_all(Gij)
        Gji = pauli_block_all(Gji)
        # NOTE: becareful: this assumes that short_Rlist is 
        # properly ordered so tha the ith R vector's negative is 
        # at -i index.
        Gji = np.flip(Gji, axis=1)

        Pi = self.P[relative_index_i]
        Pj = self.P[relative_index_j]
        X = Pi @ Gij
        Y = Pj @ Gji

        # Vectorized orbital decomposition over all R vectors
        # X.shape: (nR, 4, ni, nj), Y.shape: (nR, 4, nj, ni)
        if orb_decomposition:
            A_orb_ij = (
                np.einsum("...ruij,...rvji->...ruvij", X, Y) / np.pi
            )  # Shape: (nR, 4, 4, ni, nj)
            A_orb_ij = self.integrate(A_orb_ij)

            # Vectorized sum over orbitals for simplified A values
            A_ij = np.sum(A_orb_ij, axis=(-2, -1))
        else:
            A_orb_ij = None
            A_ij = (
                np.einsum("...uij,...vji->...uv", X, Y) / np.pi
            )  # Shape: (nE, nR, 4, 4)

        # Integrate
        A_ij = self.integrate(A_ij)

        return A_ij, A_orb_ij

    def compute_Aij(self, i, j, orb_decomposition=False):
        '''Computes Aij tensor with threads upper bound'''
        with threadpool_limits(limits=self.thlim):
            A_ij = self._compute_Aij(
                i, j, 
                orb_decomposition=orb_decomposition
            )

        return A_ij
