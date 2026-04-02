"""
ExchangePert2GPU: GPU-accelerated version of ExchangePert2 using JAX.
"""

from collections import defaultdict

import numpy as np

from TB2J.exchange_pert2 import ExchangePert2
from TB2J.gpu.jax_utils import (
    _compute_dGR_jax,
    _compute_GR_jax,
    _eigen_to_G_jax,
    _pauli_block_all_jax,
    _pauli_block_all_sep_jax,
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)
from TB2J.gpu.jax_utils import (
    _jnp as jnp,
)


def _compute_AdA_tensor_jax(X, Y, dX, dY):
    """
    JAX version: Compute A and dA/dx tensors for all E, R vectors and components.
    X, Y, dX, dY: (ne, nR, 4, ni, nj)
    """
    _require_jax()

    # A[e, r, a, b] = sum_ij (X[e, r, a, i, j] * Y[e, r, b, j, i]) / pi
    A = jnp.einsum("era_ij, erb_ji -> erab", X, Y) / jnp.pi

    if dX is not None and dY is not None:
        # dA[e, r, a, b] = sum_ij (dX[e, r, a, i, j] * Y[e, r, b, j, i] + X[e, r, a, i, j] * dY[e, r, b, j, i]) / pi
        dAdx = (
            jnp.einsum("era_ij, erb_ji -> erab", dX, Y)
            + jnp.einsum("era_ij, erb_ji -> erab", X, dY)
        ) / jnp.pi
    else:
        dAdx = jnp.zeros_like(A)

    return A, dAdx


def _compute_AdA_tensor_orb_jax(X, Y, dX, dY):
    """
    JAX version: Compute A and dA/dx tensors with orbital decomposition for all E, R.
    """
    _require_jax()

    # A_orb[e, r, a, b, i, j] = X[e, r, a, i, j] * Y[e, r, b, j, i] / pi
    A_orb = jnp.einsum("era_ij, erb_ji -> erabij", X, Y) / jnp.pi

    if dX is not None and dY is not None:
        dAdx_orb = (
            jnp.einsum("era_ij, erb_ji -> erabij", dX, Y)
            + jnp.einsum("era_ij, erb_ji -> erabij", X, dY)
        ) / jnp.pi
    else:
        dAdx_orb = jnp.zeros_like(A_orb)

    A = jnp.sum(A_orb, axis=(-2, -1))
    dAdx = jnp.sum(dAdx_orb, axis=(-2, -1))

    return A, dAdx, A_orb, dAdx_orb


class ExchangePert2GPU(ExchangePert2):
    """
    GPU-accelerated version of ExchangePert2 using JAX.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ExchangePert2GPU."""
        _require_jax()
        super().__init__(*args, **kwargs)

    def get_all_A_vectorized_gpu(self, GR, dGRij, dGRji):
        """
        GPU-accelerated vectorized calculation of A and dA/dx for multiple energy points.
        GR, dGRij, dGRji: (ne, nR, nb, nb)
        """
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]
        P = [numpy_to_jax(self.get_P_iatom(site)) for site in magnetic_sites]

        # Convert arrays to JAX
        GR_jax = numpy_to_jax(GR)
        dGRij_jax = numpy_to_jax(dGRij)
        dGRji_jax = numpy_to_jax(dGRji)
        weights_jax = numpy_to_jax(self.contour.weights)

        nA = len(magnetic_sites)
        nR = GR.shape[1]

        indices_neg = jnp.array([self.R_negative_index[k] for k in range(nR)])

        # Loop over atom pairs (small loop, okay for CPU)
        for i in range(nA):
            for j in range(nA):
                idx, jdx = iorbs[i], iorbs[j]

                # Extract matrices on GPU
                Gij = GR_jax[:, :, idx][:, :, :, jdx]
                Gji_block = GR_jax[:, indices_neg][:, :, jdx][:, :, :, idx]
                dGij_block = dGRij_jax[:, :, idx][:, :, :, jdx]
                dGji_block = dGRji_jax[:, :, jdx][:, :, :, idx]

                # Pauli decomposition on GPU
                if self.basis_is_separated:
                    Gij_Ixyz = _pauli_block_all_sep_jax(Gij)
                    Gji_Ixyz = _pauli_block_all_sep_jax(Gji_block)
                    dGij_Ixyz = _pauli_block_all_sep_jax(dGij_block)
                    dGji_Ixyz = _pauli_block_all_sep_jax(dGji_block)
                else:
                    Gij_Ixyz = _pauli_block_all_jax(Gij)
                    Gji_Ixyz = _pauli_block_all_jax(Gji_block)
                    dGij_Ixyz = _pauli_block_all_jax(dGij_block)
                    dGji_Ixyz = _pauli_block_all_jax(dGji_block)

                # Transpose to (ne, nR, 4, ni, nj)
                Gij_Ixyz = jnp.transpose(Gij_Ixyz, (1, 2, 0, 3, 4))
                Gji_Ixyz = jnp.transpose(Gji_Ixyz, (1, 2, 0, 3, 4))
                dGij_Ixyz = jnp.transpose(dGij_Ixyz, (1, 2, 0, 3, 4))
                dGji_Ixyz = jnp.transpose(dGji_Ixyz, (1, 2, 0, 3, 4))

                Pi, Pj = P[i], P[j]
                X = jnp.einsum("ik, eru_kj -> eru_ij", Pi, Gij_Ixyz)
                Y = jnp.einsum("jk, eru_ki -> eru_ji", Pj, Gji_Ixyz)
                dX = jnp.einsum("ik, eru_kj -> eru_ij", Pi, dGij_Ixyz)
                dY = jnp.einsum("jk, eru_ki -> eru_ji", Pj, dGji_Ixyz)

                if self.orb_decomposition:
                    A_val, dAdx_val, A_orb_val, dAdx_orb_val = (
                        _compute_AdA_tensor_orb_jax(X, Y, dX, dY)
                    )
                    A_integrated_jax = jnp.einsum("e, erab -> rab", weights_jax, A_val)
                    dAdx_integrated_jax = jnp.einsum(
                        "e, erab -> rab", weights_jax, dAdx_val
                    )
                    A_orb_integrated_jax = jnp.einsum(
                        "e, erab_ij -> rab_ij", weights_jax, A_orb_val
                    )
                    dAdx_orb_integrated_jax = jnp.einsum(
                        "e, erab_ij -> rab_ij", weights_jax, dAdx_orb_val
                    )

                    A_integrated = jax_to_numpy(A_integrated_jax)
                    dAdx_integrated = jax_to_numpy(dAdx_integrated_jax)
                    A_orb_integrated = jax_to_numpy(A_orb_integrated_jax)
                    dAdx_orb_integrated = jax_to_numpy(dAdx_orb_integrated_jax)
                else:
                    A_val, dAdx_val = _compute_AdA_tensor_jax(X, Y, dX, dY)
                    A_integrated_jax = jnp.einsum("e, erab -> rab", weights_jax, A_val)
                    dAdx_integrated_jax = jnp.einsum(
                        "e, erab -> rab", weights_jax, dAdx_val
                    )

                    A_integrated = jax_to_numpy(A_integrated_jax)
                    dAdx_integrated = jax_to_numpy(dAdx_integrated_jax)
                    A_orb_integrated, dAdx_orb_integrated = None, None

                mi, mj = magnetic_sites[i], magnetic_sites[j]
                for iR, R_vec in enumerate(self.short_Rlist):
                    if (R_vec, i, j) in self.distance_dict:
                        self.A_ijR[(R_vec, mi, mj)] += A_integrated[iR]
                        self.dA_ijR[(R_vec, mi, mj)] += dAdx_integrated[iR]
                        if self.orb_decomposition:
                            if (R_vec, mi, mj) not in self.A_ijR_orb:
                                self.A_ijR_orb[(R_vec, mi, mj)] = A_orb_integrated[iR]
                                self.dA_ijR_orb[(R_vec, mi, mj)] = dAdx_orb_integrated[
                                    iR
                                ]
                            else:
                                self.A_ijR_orb[(R_vec, mi, mj)] += A_orb_integrated[iR]
                                self.dA_ijR_orb[(R_vec, mi, mj)] += dAdx_orb_integrated[
                                    iR
                                ]

    def _prepare_dGR_jax(self, Rpts, Rset, Rjlist, epc, Ru):
        """Prepare indices and matrices for dGR calculation."""
        if self.G._Rmap is None:
            self.G._build_Rmaps(Rpts, Rset, Rjlist, epc, Ru)
        R2idx = {R: i for i, R in enumerate(Rpts)}
        Rmap_idx = []
        dV_list = []
        for Rq, Rk, Rm, Rnj, Rj in self.G._Rmap:
            Rmap_idx.append((R2idx[Rm], R2idx[Rnj], R2idx[Rj]))
            dV_list.append(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T)
        Rmap_rev_idx = []
        dV_rev_list = []
        for Rq, Rk, Rjn, Rmi, Rj in self.G._Rmap_rev:
            Rmap_rev_idx.append((R2idx[Rjn], R2idx[Rmi], R2idx[Rj]))
            dV_rev_list.append(epc.get_epmat_RgRk_two_spin(Rq, Rk, avg=False).T)
        return Rmap_idx, numpy_to_jax(dV_list), Rmap_rev_idx, numpy_to_jax(dV_rev_list)

    def calculate_all(self, use_gpu=True, vectorize_energy=True, e_batch_size=None):
        """Calculate exchange parameters with optional memory management."""
        self.validate()
        npole = len(self.contour.path)

        # Pre-initialize target dictionaries
        self.A_ijR = defaultdict(lambda: 0.0)
        self.dA_ijR = defaultdict(lambda: 0.0)
        self.A_ijR_orb = {}
        self.dA_ijR_orb = {}

        if not vectorize_energy:
            print("Computing ExchangePert2 energy-by-energy on GPU...")
            for ie in range(npole):
                self._calculate_batch([ie], [self.contour.path[ie]])
        else:
            print("Computing ExchangePert2 with vectorized GPU parallelization...")
            if e_batch_size is None or e_batch_size >= npole:
                batches = [(range(npole), self.contour.path)]
            else:
                batches = []
                for i in range(0, npole, e_batch_size):
                    idx = range(i, min(i + e_batch_size, npole))
                    batches.append((idx, self.contour.path[list(idx)]))

            for batch_idx, batch_path in batches:
                self._calculate_batch(batch_idx, batch_path)

        # Apply final integration factor
        weights = self.contour.weights
        if npole > 0:
            dummy = np.zeros(npole)
            dummy[0] = 1.0
            factor = self.contour.integrate_values(dummy) / weights[0]
            for key in self.A_ijR:
                self.A_ijR[key] *= factor
                self.dA_ijR[key] *= factor
            if self.orb_decomposition:
                for key in self.A_ijR_orb:
                    self.A_ijR_orb[key] *= factor
                    self.dA_ijR_orb[key] *= factor

        self.get_rho_atom()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

    def _calculate_batch(self, batch_indices, batch_path):
        """Internal helper to calculate a batch of energy points."""
        evals_jax = numpy_to_jax(self.G.evals)
        evecs_jax = numpy_to_jax(self.G.evecs)
        energy_jax = numpy_to_jax(batch_path)

        # 1. Compute G(k, e)
        Gk_all = _eigen_to_G_jax(
            evals_jax[None, ...],
            evecs_jax[None, ...],
            self.G.efermi,
            energy_jax[:, None, None],
        )

        # 2. Density matrix (integrated)
        kweights_jax = numpy_to_jax(self.G.kweights)
        if self.G.is_orthogonal:
            rhok_all = Gk_all
        else:
            Sk_all = self.G.S if not self.G._use_cache else self.G.get_Sk(slice(None))
            Sk_jax = numpy_to_jax(Sk_all)
            rhok_all = jnp.einsum("kij, ekjl -> ekil", Sk_jax, Gk_all)

        rhoR0_all = jnp.einsum("ekij, k -> eij", rhok_all, kweights_jax)
        batch_weights = self.contour.weights[list(batch_indices)]
        self.rho += jax_to_numpy(
            jnp.einsum("e, eij -> ij", numpy_to_jax(batch_weights), rhoR0_all)
        )

        # 3. Compute G(R, e)
        Rpts_jax = numpy_to_jax(self.short_Rlist)
        kpts_jax = numpy_to_jax(self.G.kpts)
        k2Rfactor = self.G.k2Rfactor
        GR_all = _compute_GR_jax(Rpts_jax, kpts_jax, Gk_all, kweights_jax, k2Rfactor)

        # 4. Compute dG(R, e)
        Rpts = [tuple(R) for R in self.short_Rlist]
        Rset = set(Rpts)
        Rmap_idx, dV_jax, Rmap_rev_idx, dV_rev_jax = self._prepare_dGR_jax(
            Rpts, Rset, self.short_Rlist, self.epc, self.Ru
        )

        dGRij_all, unique_Rj_ij = _compute_dGR_jax(GR_all, Rmap_idx, dV_jax)
        dGRji_all, unique_Rj_ji = _compute_dGR_jax(GR_all, Rmap_rev_idx, dV_rev_jax)

        dGRij_full = jnp.zeros_like(GR_all)
        dGRji_full = jnp.zeros_like(GR_all)
        dGRij_full = dGRij_full.at[:, unique_Rj_ij].set(dGRij_all)
        dGRji_full = dGRji_full.at[:, unique_Rj_ji].set(dGRji_all)

        # 5. Compute A tensor on GPU
        # Temporarily swap weights for vectorized summation in get_all_A...
        orig_weights = self.contour.weights
        self.contour.weights = batch_weights
        self.get_all_A_vectorized_gpu(GR_all, dGRij_full, dGRji_full)
        self.contour.weights = orig_weights

    def run(
        self,
        path="TB2J_results",
        use_gpu=True,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Run calculations with memory options."""
        self.calculate_all(
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.write_output(path=path)
        self.finalize()
