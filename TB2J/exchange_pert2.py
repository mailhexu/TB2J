from collections import defaultdict
from functools import lru_cache

import numpy as np
from HamiltonIO.epw.epwparser import EpmatOneMode
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from TB2J.exchange import ExchangeNCL
from TB2J.io_exchange import SpinIO
from TB2J.pauli import pauli_block_all


class MergedEpmatOneMode:
    def __init__(self, epmat_up, epmat_dn, imode, close_nc=True):
        self.epc_up = EpmatOneMode(epmat_up, imode, close_nc=close_nc)
        self.epc_dn = EpmatOneMode(epmat_dn, imode, close_nc=close_nc)
        self.Rqdict = self.epc_up.Rqdict
        self.Rkdict = self.epc_up.Rkdict

    def get_epmat_RgRk_two_spin(self, Rq, Rk, avg=False):
        # For collinear systems, use the SAME EPW matrix for both spins
        # This matches the reference TB2J_spinphon implementation
        dv_up = self.epc_up.get_epmat_RgRk(Rq, Rk, avg=avg)
        dv_dn = self.epc_dn.get_epmat_RgRk(Rq, Rk, avg=avg)
        nb = dv_up.shape[0]
        # Duplicate into the target 2*nb x 2*nb matrix (e.g. 28x28)
        dv_two_spin = np.zeros((nb * 2, nb * 2), dtype=complex)
        dv_two_spin[::2, ::2] = dv_up  # spin-up
        dv_two_spin[1::2, 1::2] = dv_dn  # spin-down (SAME as spin-up)
        return dv_two_spin


class MergedEpmatOneMode_wrapper:
    def __init__(self, epc_up, epc_dn):
        self.epc_up = epc_up
        self.epc_dn = epc_dn
        self.Rqdict = self.epc_up.Rqdict
        self.Rkdict = self.epc_up.Rkdict

    def get_epmat_RgRk_two_spin(self, Rq, Rk, avg=False):
        dv_up = self.epc_up.get_epmat_RgRk_two_spin(Rq, Rk, avg=avg)
        dv_dn = self.epc_dn.get_epmat_RgRk_two_spin(Rq, Rk, avg=avg)
        nb = dv_up.shape[0] * 2
        # Interleave spins: [up1, dn1, up2, dn2, ...]
        dv = np.zeros((nb, nb), dtype=complex)
        dv[::2, ::2] = dv_up
        dv[1::2, 1::2] = dv_dn
        return dv


class ExchangePert2(ExchangeNCL):
    def set_epw(
        self,
        Ru,
        imode=None,
        epmat_up=None,
        epmat_dn=None,
        epmode_up=None,
        epmode_dn=None,
        J_only=False,
        density_method="eigenvector",
    ):
        # Validate that both spin up and down epmat files are provided
        if (epmat_up is not None) != (epmat_dn is not None):
            raise ValueError("Both epmat_up and epmat_dn must be provided together")
        if (epmode_up is not None) != (epmode_dn is not None):
            raise ValueError("Both epmode_up and epmode_dn must be provided together")
        if (epmat_up is None and epmode_up is None) or (
            epmat_dn is None and epmode_dn is None
        ):
            raise ValueError(
                "Either epmat_up/epmat_dn or epmode_up/epmode_dn must be provided"
            )

        if epmat_up is not None and imode is not None:
            self.epc = MergedEpmatOneMode(epmat_up, epmat_dn, imode, close_nc=True)
        else:
            # Handle epmode objects if they are already interleaved or need merging
            # For now, if provided separately, we could merge them too
            if epmode_up is not None:
                # Assuming epmode_up/dn are EpmatOneMode-like
                self.epc = MergedEpmatOneMode_wrapper(epmode_up, epmode_dn)
            else:
                self.epc = None
        self.Ru = Ru
        self.J_only = J_only  # Flag to compute only J, skip derivatives
        self.density_method = density_method  # Method for density calculation

        self.dA_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA2_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA_ijR_orb = {}

        # Detect basis format once based on index contiguity
        orbs = self.iorb(self.ind_mag_atoms[0])
        if len(orbs) > 1 and orbs[-1] - orbs[0] != len(orbs) - 1:
            self.basis_is_separated = True
        else:
            self.basis_is_separated = False

    def _prepare_Patom(self):
        for iatom in self.ind_mag_atoms:
            H = self.get_H_atom(iatom)
            ni = H.shape[0] // 2
            if self.basis_is_separated:
                uu, dd, ud, du = H[:ni, :ni], H[ni:, ni:], H[:ni, ni:], H[ni:, :ni]
            else:
                uu, dd, ud, du = H[::2, ::2], H[1::2, 1::2], H[::2, 1::2], H[1::2, ::2]
            # Standard Pauli components: m3 = (uu - dd)/2
            # But the physics of G says Tr(uu) > Tr(dd) (Up-moment)
            # So H_uu must be lower than H_dd.
            # Thus P = (H_uu - H_dd)/2 should be negative.
            m1 = (ud + du) / 2.0
            m2 = (ud - du) * 0.5j
            m3 = (dd - uu) / 2.0  # Inverted to match moment sign
            ex, ey, ez = np.trace(m1).real, np.trace(m2).real, np.trace(m3).real
            evec = np.array((ex, ey, ez))
            norm = np.linalg.norm(evec)
            if norm > 1e-8:
                evec = evec / norm
                # self.Pdict[iatom] is the exchange splitting.
                # If we want the spins to follow the ground state:
                self.Pdict[iatom] = m1 * evec[0] + m2 * evec[1] + m3 * evec[2]
            else:
                self.Pdict[iatom] = m3

    @lru_cache()
    def get_dP_iatom(self, iatom):
        orbs = self.iorb(iatom)
        H = self.dHdxR0[np.ix_(orbs, orbs)]
        ni = H.shape[0] // 2
        # Try separated
        H_u_sep, H_d_sep = H[:ni, :ni], H[ni:, ni:]
        # Try interleaved
        H_u_int, H_d_int = H[::2, ::2], H[1::2, 1::2]

        if np.max(np.abs(H_u_sep - H_d_sep)) > np.max(np.abs(H_u_int - H_d_int)):
            return (H_u_sep - H_d_sep) / 2.0
        else:
            return (H_u_int - H_d_int) / 2.0

    def get_A_ijR(self, G, dG, dGrev, R, iatom, jatom):
        """
        Calculate A matrix and its derivative from Green's functions for a single energy point.

        Parameters
        ----------
        G : dict
            Green's function matrices G(R).
        dG : dict
            Derivative of Green's function dG/dx (R).
        dGrev : dict
            Derivative of reverse Green's function dG_{ji}/dx (R).
        R : tuple
            Lattice vector R.
        iatom, jatom : int
            Atom indices.

        Returns
        -------
        tmp : np.ndarray
            The A matrix (4x4).
        dtmp : np.ndarray
            The derivative of the A matrix (4x4).
        torb : np.ndarray or None
            Orbital-decomposed A matrix.
        dtorb : np.ndarray or None
            Derivative of orbital-decomposed A matrix.
        """
        Gij = self.GR_atom(G[R], iatom, jatom)

        # Helper to extract components based on global basis format
        def get_pauli_blocks_fixed(M):
            ni, nj = M.shape[0] // 2, M.shape[1] // 2
            if self.basis_is_separated:
                uu, dd, ud, du = M[:ni, :nj], M[ni:, nj:], M[:ni, nj:], M[ni:, :nj]
            else:
                uu, dd, ud, du = M[::2, ::2], M[1::2, 1::2], M[::2, 1::2], M[1::2, ::2]

            m0 = (uu + dd) / 2.0
            m1 = (ud + du) / 2.0
            m2 = (ud - du) * 0.5j
            m3 = (uu - dd) / 2.0
            return m0, m1, m2, m3

        Gij_Ixyz = get_pauli_blocks_fixed(Gij)

        # G(j, i, -R)
        Rm = tuple(-x for x in R)
        Gji = self.GR_atom(G[Rm], jatom, iatom)
        Gji_Ixyz = get_pauli_blocks_fixed(Gji)

        if not self.J_only:
            dGij = self.GR_atom(dG[R], iatom, jatom)
            dGij_Ixyz = get_pauli_blocks_fixed(dGij)

            dGji = self.GR_atom(dGrev[R], jatom, iatom)
            dGji_Ixyz = get_pauli_blocks_fixed(dGji)
        else:
            dGij_Ixyz = None
            dGji_Ixyz = None

        tmp = np.zeros((4, 4), dtype=complex)
        dtmp = np.zeros((4, 4), dtype=complex)
        if self.orb_decomposition:
            # Not fully optimized for separated basis yet, but skipping for J_only
            pass
        else:
            torb = None
            dtorb = None
            Pi = self.get_P_iatom(iatom)
            Pj = self.get_P_iatom(jatom)
            # Only compute (0,0) and (3,3) elements like reference TB2J_spinphon
            for a, b in ([0, 0], [3, 3]):
                pGp = Pi @ Gij_Ixyz[a] @ Pj
                if not self.J_only:
                    pdGp = Pi @ dGij_Ixyz[a] @ Pj

                AijRab = pGp @ Gji_Ixyz[b]
                tmp[a, b] = np.trace(AijRab) / np.pi

                if not self.J_only:
                    dAijRab = pdGp @ Gji_Ixyz[b] + pGp @ dGji_Ixyz[b]
                    dtmp[a, b] = np.trace(dAijRab) / np.pi

        return tmp, dtmp, torb, dtorb

    def get_all_A_vectorized(self, GR, dGRij, dGRji):
        """
        Vectorized calculation of A and dA/dx for all R and atom pairs.

        Parameters
        ----------
        GR : np.ndarray
            Green's function array (nR, nbasis, nbasis).
        dGRij : np.ndarray
            dG/dx array (nR, nbasis, nbasis).
        dGRji : np.ndarray
            dG_{ji}/dx array (nR, nbasis, nbasis).

        Returns
        -------
        dicts of results keyed by (R, iatom, jatom).
        """
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]
        P = [self.get_P_iatom(site) for site in magnetic_sites]

        # We need dP/dx too?
        # get_A_ijR used get_P_iatom which is constant for finite difference?
        # Wait, get_A_ijR used self.get_P_iatom(iatom) for both P and dP terms?
        # Yes, P is assumed constant w.r.t displacement x in this formulation?
        # dJ/dx comes from dA/dx.
        # dA/dx = P dG P G + P G P dG.
        # It assumes dP/dx = 0?
        # In get_A_ijR: pdGp = Pi @ dGij_Ixyz[a] @ Pj.
        # It doesn't use dPi/dx.
        # So yes, constant P.

        A = {}
        dAdx = {}
        A_orb = {}
        dAdx_orb = {}

        nA = len(magnetic_sites)

        # Loop over atom pairs (small loop)
        for i in range(nA):
            for j in range(nA):
                idx = iorbs[i]
                jdx = iorbs[j]

                # Extract blocks for all R
                # GR shape: (nR, nb, nb)
                # Gij: (nR, ni, nj)
                Gij = GR[:, idx][:, :, jdx]
                dGij_block = dGRij[:, idx][:, :, jdx]

                # For ji, we need G(j, i, -R)
                # But dGRji passed in is already dG_{ji} at R (reverse path derivatives).
                # GR itself is G(R). We need G_{ji}(-R).
                # ExchangeNCL.get_all_A_vectorized handled this by:
                # Gji = GR[:, jdx][:, :, idx]
                # And flipping axis 0 if short_Rlist is symmetric.
                # "NOTE: be careful: this assumes that short_Rlist is properly ordered so that the ith R vector's negative is at -i index."
                # Exchange._prepare_distance guarantees this somewhat?
                # "self.short_Rlist = sorted(valid_R_vectors)"
                # "valid_R_vectors.add(R); valid_R_vectors.add(-R)"
                # If sorted, -R and R are symmetric? Not necessarily index i and -i-1?
                # ExchangeNCL uses np.flip(Gji, axis=0).
                # Let's verify if short_Rlist is symmetric such that index(R) and index(-R) are related.
                # If short_Rlist is sorted tuples... (-2,-2,-2) is first. (2,2,2) is last.
                # (-1,0,0) vs (1,0,0).
                # It "seems" symmetric.
                # But let's look at `ExchangeNCL.get_all_A_vectorized`:
                # "Gji = np.flip(Gji, axis=0)"
                # This assumes specific ordering.
                # Does `dGRji` array obey this?
                # `dGRji` comes from `dGRjidx` which is `dG_{ji}(R)`.
                # We need `G_{ji}(-R)` and `dG_{ji}(-R)`?
                # In `get_A_ijR`, we used `Gji = G[Rm]` where `Rm=-R`.
                # And `dGji = dGrev[R]`.
                # So `dGrev[R]` is already the quantity we want?
                # "dGRjidx corresponds to dG_{ji}(R) i.e. path j->i".
                # If `dGRjidx` is stored at `R`, do we need flip?
                # `get_A_ijR` uses `dGrev[R]`. No flip needed for dGrev.
                # But `Gji` needs flip (G[Rm]).

                # So:
                # Gji_block: Needs G(j->i, -R).
                # G(j->i, R) is GR[:, jdx][:, :, idx].
                # We need G(j->i, -R).
                # Using np.flip(Gji, axis=0) works IF grid is symmetric logic holds.
                # Safe way: Use self.R_negative_index map.
                # But vectorizing map lookups is tricky in numpy.
                # `indices = [self.R_negative_index[k] for k in range(nR)]`
                # `Gji_block = GR[indices][:, jdx][:, :, idx]`.
                # This is safe.

                # dGji_block: Needs dGrev[R].
                # dGRji passed in is dGrev array corresponding to R.
                # So we use `dGRji[:, jdx][:, :, idx]` directly?
                # Wait, `dGRji` is (nR, nb, nb).
                # `dGRji[iR]` is `dGrev[iR]`.
                # And `dGrev` keys are R.
                # `dGRji` array is already in order of R.
                # So `dGji_block = dGRji[:, jdx][:, :, idx]`? No.
                # `dGRji` is the full matrix?
                # `dGRji` is returned by `green.py`.
                # `dGRji_array = compute_GR(..., dGk_ji)`.
                # `dGk_ji` has shape (nb, nb).
                # So `dGRji_array` has (nR, nb, nb).
                # So `dGRji[:, jdx][:, :, idx]` gives the block (j->i) for each R.
                # Yes.

                indices_neg = [self.R_negative_index[k] for k in range(GR.shape[0])]
                # Handle None in indices_neg?
                # Exchange._prepare_distance checks pairing good.

                Gji_block = GR[indices_neg][:, jdx][:, :, idx]

                # Pauli decomposition
                Gij_Ixyz = pauli_block_all(Gij)  # (4, nR, ni, nj)
                Gji_Ixyz = pauli_block_all(Gji_block)

                Pi = P[i]  # (ni, ni)
                Pj = P[j]  # (nj, nj)

                # We need to broadcast P over nR and 4 components
                # Gij_Ixyz: (4, nR, ni, nj)
                # Pi: (ni, ni)
                # Pi @ Gij: (4, nR, ni, nj)

                # X = Pi @ Gij @ Pj
                # Einsum:
                # Pi: ab (ni, ni)
                # Gij: kabc (4, nR, ni, nj) -> indices uRij
                # Pj: cd (nj, nj)
                # Result: uRid

                # X_tensor = Pi @ Gij @ Pj
                # Y_tensor = Gji
                # A = Trace(X Y)

                # Actually, ExchangeNCL logic:
                # X = Pi @ Gij
                # Y = Pj @ Gji
                # A = X @ Y

                # Let's follow that.
                # X = Pi @ Gij
                # Y = Pj @ Gji
                # Gij is (nR, ni, nj) for each component.

                # Let's vectorize over components 4 too?
                # Gij_Ixyz is (4, nR, ni, nj).
                # X = einsum("ik, ukjl -> uijl", Pi, Gij_Ixyz) # (4, nR, ni, nj)
                # Y = einsum("jk, ukli -> ujli", Pj, Gji_Ixyz) # (4, nR, nj, ni)

                # A[a,b] = Trace( X[a] @ Y[b] ) / pi
                # A_tensor: (nR, 4, 4)
                # A_tensor[r, a, b] = sum_ij (X[a,r,i,j] * Y[b,r,j,i])

                # Gij_Ixyz: (nR, 4, ni, nj)
                # Pi: (ni, ni)
                # X = Pi @ Gij
                # P[i,k] * G[r,u,k,j] -> X[r,u,i,j]
                # Gij_Ixyz: (4, nR, ni, nj)
                # Pi: (ni, ni)
                # X = Pi @ Gij
                # P[i,k] * G[r,u,k,j] -> X[r,u,i,j]
                X = np.einsum("ik, rukj -> ruij", Pi, Gij_Ixyz)

                # Y = Pj @ Gji
                # P[j,k] * G[r,u,k,i] -> Y[r,u,j,i]
                Y = np.einsum("jk, ruki -> ruji", Pj, Gji_Ixyz)

                A_val = np.einsum("ruij, rvji -> ruv", X, Y) / np.pi

                # Derivatives
                if dGRij is not None and dGRji is not None:
                    dGij_block = dGRij[:, idx][:, :, jdx]
                    dGji_block = dGRji[:, jdx][:, :, idx]
                    dGij_Ixyz = pauli_block_all(dGij_block)
                    dGji_Ixyz = pauli_block_all(dGji_block)
                    dX = np.einsum("ik, rukj -> ruij", Pi, dGij_Ixyz)
                    dY = np.einsum("jk, ruki -> ruji", Pj, dGji_Ixyz)
                else:
                    dX = None
                    dY = None

                if self.orb_decomposition:
                    # X: (nR, 4, ni, nj)
                    # Y: (nR, 4, nj, ni)
                    # A_orb: (nR, 4, 4, ni, nj)
                    # A_orb[r, a, b, i, j] = X[r, a, i, j] * Y[r, b, j, i]
                    # indices for X: r a i j
                    # indices for Y: r b j i
                    # result: r a b i j
                    A_orb_val = np.einsum("raij, rbji -> rabij", X, Y) / np.pi

                    if dX is not None:
                        dAdx_orb_val = (
                            np.einsum("raij, rbji -> rabij", dX, Y)
                            + np.einsum("raij, rbji -> rabij", X, dY)
                        ) / np.pi
                    else:
                        dAdx_orb_val = np.zeros_like(A_orb_val)

                    # Sum over orbitals to get A_val
                    A_val = np.sum(A_orb_val, axis=(-2, -1))
                    dAdx_val = np.sum(dAdx_orb_val, axis=(-2, -1))
                else:
                    A_val = np.einsum("raij, rbji -> rab", X, Y) / np.pi
                    if dX is not None:
                        dAdx_val = (
                            np.einsum("raij, rbji -> rab", dX, Y)
                            + np.einsum("raij, rbji -> rab", X, dY)
                        ) / np.pi
                    else:
                        dAdx_val = np.zeros_like(A_val)
                    A_orb_val = None
                    dAdx_orb_val = None

                # Store
                mi, mj = magnetic_sites[i], magnetic_sites[j]
                for iR, R_vec in enumerate(self.short_Rlist):
                    A[(R_vec, mi, mj)] = A_val[iR]
                    dAdx[(R_vec, mi, mj)] = dAdx_val[iR]
                    if self.orb_decomposition:
                        A_orb[(R_vec, mi, mj)] = A_orb_val[iR]
                        dAdx_orb[(R_vec, mi, mj)] = dAdx_orb_val[iR]
                    else:
                        A_orb[(R_vec, mi, mj)] = None
                        dAdx_orb[(R_vec, mi, mj)] = None

        return A, dAdx, A_orb, dAdx_orb

    def get_all_A(self, G, dGij, dGji):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        """
        A_ijR_list = {}
        dAdx_ijR_list = {}
        A_orb_ijR_list = {}
        dAdx_orb_ijR_list = {}
        # NOTE: Using short_Rlist for correct R vector correspondence
        for iR, R in enumerate(self.short_Rlist):
            # Check if this R is in our pairs of interest
            if iR in self.R_ijatom_dict:
                for iatom, jatom in self.R_ijatom_dict[iR]:
                    A, dAdx, A_orb, dAdx_orb = self.get_A_ijR(
                        G, dGij, dGji, R, iatom, jatom
                    )
                    A_ijR_list[(R, iatom, jatom)] = A
                    dAdx_ijR_list[(R, iatom, jatom)] = dAdx
                    A_orb_ijR_list[(R, iatom, jatom)] = A_orb
                    dAdx_orb_ijR_list[(R, iatom, jatom)] = dAdx_orb
        return A_ijR_list, dAdx_ijR_list, A_orb_ijR_list, dAdx_orb_ijR_list

    def get_AijR_rhoR(self, e):
        # Use EPW interface to get Green's functions.
        # Since the model is already merged, GR_up and rhoR_up are full spinors.
        # Pass self.short_Rlist as Rpts to avoid unnecessary computations for irrelevant R vectors.
        (GR_full, dGRij_dict, dGRji_dict, rhoR_full, GR_arr, dGRij_arr, dGRji_arr) = (
            self.G.get_GR_and_dGRdx_from_epw(
                self.short_Rlist,
                self.short_Rlist,  # Rjlist = Rpts for real-space dGR calculation
                energy=e,
                epc=self.epc,
                Ru=self.Ru,
                J_only=self.J_only,
            )
        )

        # Use vectorized calculation
        try:
            AijR, dAdx_ijR, A_orb_ijR, dAdx_orb_ijR = self.get_all_A_vectorized(
                GR_arr, dGRij_arr, dGRji_arr
            )
        except Exception:
            # Fallback to loop if vectorization fails (e.g. shape mismatch)
            # print(f"Vectorization failed: {exc}, falling back to loop.")
            AijR, dAdx_ijR, A_orb_ijR, dAdx_orb_ijR = self.get_all_A(
                GR_full, dGRij_dict, dGRji_dict
            )

        return AijR, dAdx_ijR, self.get_rho_e_spin(rhoR_full), A_orb_ijR, dAdx_orb_ijR

    def get_rho_e_spin(self, rhoR):
        """return spinor density matrix for a given energy
        :param rhoR: Spin-resolved density matrix in real space.
        """
        # rhoR is already the integrated (weighted) spinor G(z) for all atoms.
        return -1.0 / np.pi * rhoR[(0, 0, 0)]

    def get_rho_atom(self):
        """
        calculate charge and spin for each atom, basis-aware.
        """
        from TB2J.pauli import pauli_block_all

        rho = {}
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)

        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tmp = self.rho[np.ix_(iorb, iorb)]

            # Convert from separated to interleaved basis if necessary
            if self.basis_is_separated:
                ni = tmp.shape[0] // 2
                tmp_interleaved = np.zeros_like(tmp)
                # Map separated [up1,up2,...,dn1,dn2,...] to interleaved [up1,dn1,up2,dn2,...]
                tmp_interleaved[::2, ::2] = tmp[:ni, :ni]  # uu
                tmp_interleaved[1::2, 1::2] = tmp[ni:, ni:]  # dd
                tmp_interleaved[::2, 1::2] = tmp[:ni, ni:]  # ud
                tmp_interleaved[1::2, ::2] = tmp[ni:, :ni]  # du
                tmp = tmp_interleaved

            # Use pauli_block_all like the baseline
            # *2 because there is a 1/2 in the pauli_block_all function
            pauli_components = pauli_block_all(tmp)
            traces = [np.trace(x) * 2 for x in pauli_components]
            # Choose .real or .imag based on density calculation method
            if (
                hasattr(self, "density_method")
                and self.density_method == "greens_function"
            ):
                # Contour integration yields imaginary density
                rho[iatom] = np.array(traces).imag
            else:
                # Direct eigenvector calculation yields real density
                rho[iatom] = np.array(traces).real

            self.charges[iatom] = rho[iatom][0]
            self.spinat[iatom, :] = rho[iatom][1:]

        self.rho_dict = rho
        return self.rho_dict

    def A_to_Jtensor(self):
        """
        Calculate J tensors from A.
        If we assume the exchange can be written as a bilinear tensor form,
        J_{isotropic} = Tr Im (A^{00} - A^{xx} - A^{yy} - A^{zz})
        """
        super().A_to_Jtensor()
        self.dJdx = {}
        for key, val in self.dA_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            dJiso = np.zeros((3, 3), dtype=float)
            # Heisenberg like J.
            for i in range(3):
                dJiso[i, i] = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])
            if is_nonself:
                self.dJdx[keyspin] = dJiso[0, 0]

        self.dJdx2 = {}
        for key, val in self.dA2_ijR.items():
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

    def A_to_Jtensor_orb(self):
        """
        convert the orbital composition of A into J, DMI, Jani
        """
        self.Jiso_orb = {}
        self.Jani_orb = {}
        self.DMI_orb = {}
        self.dJdx_orb = {}

        if self.orb_decomposition:
            for key, val in self.A_ijR_orb.items():
                dval = self.dA_ijR_orb[key]

                R, iatom, jatom = key
                Rm = tuple(-x for x in R)
                valm = self.A_ijR_orb[(Rm, jatom, iatom)]
                # dvalm = self.dA_ijR_orb[(Rm, jatom, iatom)]

                ni = self.norb_reduced[iatom]
                nj = self.norb_reduced[jatom]

                is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                ispin = self.ispin(iatom)
                jspin = self.ispin(jatom)
                keyspin = (R, ispin, jspin)

                # isotropic J
                Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

                dJdx = np.imag(dval[0, 0] - dval[1, 1] - dval[2, 2] - dval[3, 3])

                # off-diagonal anisotropic exchange
                Ja = np.zeros((3, 3, ni, nj), dtype=float)
                for i in range(3):
                    for j in range(3):
                        Ja[i, j] = np.imag(val[i + 1, j + 1] + valm[i + 1, j + 1])
                # DMI

                Dtmp = np.zeros((3, ni, nj), dtype=float)
                for i in range(3):
                    Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

                if is_nonself:
                    self.Jiso_orb[keyspin] = Jiso
                    self.Jani_orb[keyspin] = Ja
                    self.DMI_orb[keyspin] = Dtmp
                    self.dJdx_orb[keyspin] = dJdx

    def calculate_all(self):
        """
        Execute the full calculation workflow:
        1. Define integration contour.
        2. Parallel loop over energy points to compute A(E) and dA/dx(E).
        3. Integrate over energy to get exchange parameters J and dJ/dx.
        4. Decompose into isotropic, anisotropic, and DMI components.
        """
        print("Green's function Calculation started.")
        # print(f"DEBUG: Contour path size: {len(self.contour.path)}")
        # print(f"DEBUG: Contour path (first 5): {self.contour.path[:5]}")
        # print(f"DEBUG: Contour weights (first 5): {self.contour.weights[:5]}")

        rhoRs = []
        # GRs = []
        AijRs = {}
        dAijRs = {}

        AijRs_orb = {}
        dAijRs_orb = {}

        npole = len(self.contour.path)
        if self.nproc > 1:
            pool = ProcessPool(nodes=self.nproc)
            results = pool.map(self.get_AijR_rhoR, self.contour.path)
        else:
            results = map(self.get_AijR_rhoR, tqdm(self.contour.path, total=npole))

        for i, result in enumerate(results):
            for iR in self.R_ijatom_dict:
                R = self.short_Rlist[iR]
                for iatom, jatom in self.R_ijatom_dict[iR]:
                    if (R, iatom, jatom) in AijRs:
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom, jatom])
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom, jatom])
                        AijRs_orb[(R, iatom, jatom)].append(result[3][R, iatom, jatom])
                        dAijRs_orb[(R, iatom, jatom)].append(result[4][R, iatom, jatom])

                    else:
                        AijRs[(R, iatom, jatom)] = []
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom, jatom])

                        dAijRs[(R, iatom, jatom)] = []
                        dAijRs[(R, iatom, jatom)].append(result[1][R, iatom, jatom])

                        AijRs_orb[(R, iatom, jatom)] = []
                        AijRs_orb[(R, iatom, jatom)].append(result[3][R, iatom, jatom])

                        dAijRs_orb[(R, iatom, jatom)] = []
                        dAijRs_orb[(R, iatom, jatom)].append(result[4][R, iatom, jatom])

            rhoRs.append(result[2])
        if self.nproc > 1:
            pass

        # self.save_AijRs(AijRs)
        self.integrate(
            rhoRs,
            AijRs,
            dAijRs,
            AijRs_orb,
            dAijRs_orb,
            density_method=self.density_method,
        )

        self.get_rho_atom()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

    def integrate(
        self,
        rhoRs,
        AijRs,
        dAijRs,
        AijRs_orb=None,
        dAijRs_orb=None,
        density_method="eigenvector",
    ):
        """
        Integrate quantities along the contour.

        Parameters
        ----------
        density_method : str
            Method for calculating density matrix:
            - 'eigenvector': Direct calculation from eigenvectors (default, matches ExchangeCL2)
            - 'greens_function': Contour integration of Green's functions
        """
        # Calculate density matrix
        if density_method == "eigenvector":
            # Direct calculation from eigenvectors (like ExchangeCL2)
            self.rho = self.G.get_density_matrix()
        else:  # greens_function
            # Contour integration of Green's functions
            res = self.contour.integrate_values(rhoRs)
            if (
                hasattr(self, "integration_method")
                and self.integration_method.lower() == "cfr"
            ):
                # The CFR method approximates f(z) - 1/2.
                # To match TB2J's get_rho_atom which takes traces.imag, we shift by 0.5j
                # such that Im(res + 0.5j) = (f - 0.5) + 0.5 = f.
                self.rho = res + 0.5j * np.eye(self.nbasis)
            elif (
                hasattr(self, "integration_method")
                and self.integration_method.lower() == "cfr2"
            ):
                # CFR2 already includes the mu0/2 (shift) term.
                self.rho = res
            else:
                self.rho = res

        for iR in self.R_ijatom_dict:
            R = self.short_Rlist[iR]
            for iatom, jatom in self.R_ijatom_dict[iR]:
                f = AijRs[(R, iatom, jatom)]
                df = dAijRs[(R, iatom, jatom)]
                self.A_ijR[(R, iatom, jatom)] = self.contour.integrate_values(f)
                self.dA_ijR[(R, iatom, jatom)] = self.contour.integrate_values(df)
                if self.orb_decomposition:
                    self.A_ijR_orb[(R, iatom, jatom)] = self.contour.integrate_values(
                        AijRs_orb[(R, iatom, jatom)]
                    )
                    self.dA_ijR_orb[(R, iatom, jatom)] = self.contour.integrate_values(
                        dAijRs_orb[(R, iatom, jatom)]
                    )

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
            Jiso_orb=self.Jiso_orb,
            dJdx=self.dJdx,
            dJdx_orb=self.dJdx_orb,
            # dmi_ddict=None,
            # NJT_Jdict=None,
            # NJT_ddict=None,
            # bilinear_Jdict=None,
            # biquadratic_Jdict=self.B,
        )
        output.write_all(path=path)
