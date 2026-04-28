"""
ExchangePert2GPU: GPU-accelerated version of ExchangePert2 using JAX.

Processes one energy point at a time to avoid GPU memory overflow,
following the same pattern as ExchangeNCLGPU and ExchangeCL2GPU.
"""

from collections import defaultdict
from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchange_pert2 import ExchangePert2
from TB2J.gpu.jax_utils import (
    _compute_A_orb_single_e,
    _compute_A_single_e,
    _compute_A_single_e_no_dA,
    _compute_dGR_single_e,
    _compute_GR_single_e,
    _eigen_to_G_single_e,
    _pauli_block_interleaved,
    _pauli_block_separated,
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)

_require_jax()
import jax.numpy as jnp  # noqa: E402


class ExchangePert2GPU(ExchangePert2):
    """
    GPU-accelerated version of ExchangePert2 using JAX.

    Processes one energy point at a time to keep GPU memory usage bounded,
    following the pattern of ExchangeNCLGPU and ExchangeCL2GPU.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ExchangePert2GPU."""
        _require_jax()
        super().__init__(*args, **kwargs)
        self._evals_jax = None
        self._evecs_jax = None
        self._kpts_jax = None
        self._kweights_jax = None
        self._Rpts_jax = None

    def _prepare_jax_arrays(self):
        """Prepare JAX arrays for GPU computation (called once)."""
        if self._evals_jax is None:
            self._evals_jax = numpy_to_jax(self.G.evals)
            self._evecs_jax = numpy_to_jax(self.G.evecs)
            self._kpts_jax = numpy_to_jax(self.G.kpts)
            self._kweights_jax = numpy_to_jax(self.G.kweights)
            self._Rpts_jax = numpy_to_jax(self.short_Rlist)
            print(
                f"Prepared JAX arrays: evals {self._evals_jax.shape}, "
                f"evecs {self._evecs_jax.shape}, "
                f"kpts {self._kpts_jax.shape}, Rpts {self._Rpts_jax.shape}"
            )

    def _prepare_dGR_jax(self, Rpts, Rset, Rjlist, epc, Ru):
        """Prepare indices and dV matrices for dGR calculation (called once)."""
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

        self._Rmap_indices_Rm = jnp.array([e[0] for e in Rmap_idx])
        self._Rmap_indices_Rnj = jnp.array([e[1] for e in Rmap_idx])
        self._Rmap_indices_Rj = jnp.array([e[2] for e in Rmap_idx])
        self._dV_jax = jnp.array(dV_list)
        self._unique_Rj_ij = jnp.unique(self._Rmap_indices_Rj)

        self._Rmap_rev_indices_Rm = jnp.array([e[0] for e in Rmap_rev_idx])
        self._Rmap_rev_indices_Rnj = jnp.array([e[1] for e in Rmap_rev_idx])
        self._Rmap_rev_indices_Rj = jnp.array([e[2] for e in Rmap_rev_idx])
        self._dV_rev_jax = jnp.array(dV_rev_list)
        self._unique_Rj_ji = jnp.unique(self._Rmap_rev_indices_Rj)

    def _compute_per_energy_gpu(self, e):
        """
        Compute all quantities for a single energy point on GPU.

        Returns dict with A and dA for all atom pairs and R vectors.
        """
        evals = self._evals_jax
        evecs = self._evecs_jax
        efermi = self.G.efermi
        kweights = self._kweights_jax

        Gk_all = _eigen_to_G_single_e(evals, evecs, efermi, e)

        if not self.G.is_orthogonal:
            Sk_all = self.G.S if not self.G._use_cache else self.G.get_Sk(slice(None))
            Sk_jax = numpy_to_jax(Sk_all)
            rhok_all = jnp.einsum("kij,kjl->kil", Sk_jax, Gk_all)
        else:
            rhok_all = Gk_all

        rhoR0 = jnp.einsum("kij,k->ij", rhok_all, kweights)

        GR = _compute_GR_single_e(
            self._Rpts_jax, self._kpts_jax, Gk_all, kweights, self.G.k2Rfactor
        )

        dGRij = _compute_dGR_single_e(
            GR,
            self._Rmap_indices_Rm,
            self._Rmap_indices_Rnj,
            self._Rmap_indices_Rj,
            self._dV_jax,
            self._unique_Rj_ij,
        )
        dGRji = _compute_dGR_single_e(
            GR,
            self._Rmap_rev_indices_Rm,
            self._Rmap_rev_indices_Rnj,
            self._Rmap_rev_indices_Rj,
            self._dV_rev_jax,
            self._unique_Rj_ji,
        )

        nR = GR.shape[0]
        nb = GR.shape[1]

        dGRij_full = jnp.zeros((nR, nb, nb), dtype=GR.dtype)
        dGRji_full = jnp.zeros((nR, nb, nb), dtype=GR.dtype)
        dGRij_full = dGRij_full.at[self._unique_Rj_ij].set(dGRij)
        dGRji_full = dGRji_full.at[self._unique_Rj_ji].set(dGRji)

        magnetic_sites = self.ind_mag_atoms
        nA = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]
        P = [numpy_to_jax(self.get_P_iatom(site)) for site in magnetic_sites]

        indices_neg = jnp.array([self.R_negative_index[k] for k in range(nR)])
        pauli_fn = (
            _pauli_block_separated
            if self.basis_is_separated
            else _pauli_block_interleaved
        )

        A_results = {}
        dA_results = {}
        A_orb_results = {}
        dA_orb_results = {}

        for i, j in product(range(nA), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]

            Gij = GR[:, idx][:, :, jdx]
            Gji_block = GR[indices_neg][:, jdx][:, :, idx]
            dGij_block = dGRij_full[:, idx][:, :, jdx]
            dGji_block = dGRji_full[:, jdx][:, :, idx]

            Gij_blocks = pauli_fn(Gij)
            Gji_blocks = pauli_fn(Gji_block)
            dGij_blocks = pauli_fn(dGij_block)
            dGji_blocks = pauli_fn(dGji_block)

            Pi, Pj = P[i], P[j]

            if self.orb_decomposition:
                A_val, dAdx_val, A_orb_val, dAdx_orb_val = _compute_A_orb_single_e(
                    Gij_blocks,
                    Gji_blocks,
                    dGij_blocks,
                    dGji_blocks,
                    Pi,
                    Pj,
                    self.J_only,
                )
                A_orb_np = jax_to_numpy(A_orb_val)
                dA_orb_np = jax_to_numpy(dAdx_orb_val)
            elif self.J_only:
                A_val = _compute_A_single_e_no_dA(Gij_blocks, Gji_blocks, Pi, Pj)
                A_val.block_until_ready()
                dAdx_val = jnp.zeros_like(A_val)
                A_orb_np = None
                dA_orb_np = None
            else:
                A_val, dAdx_val = _compute_A_single_e(
                    Gij_blocks,
                    Gji_blocks,
                    dGij_blocks,
                    dGji_blocks,
                    Pi,
                    Pj,
                    self.J_only,
                )
                A_orb_np = None
                dA_orb_np = None

            A_val.block_until_ready()

            A_np = jax_to_numpy(A_val)
            dAdx_np = jax_to_numpy(dAdx_val)

            mi, mj = magnetic_sites[i], magnetic_sites[j]
            for iR, R_vec in enumerate(self.short_Rlist):
                if (R_vec, i, j) in self.distance_dict:
                    A_results[(R_vec, mi, mj)] = A_np[iR]
                    dA_results[(R_vec, mi, mj)] = dAdx_np[iR]
                    if self.orb_decomposition:
                        A_orb_results[(R_vec, mi, mj)] = A_orb_np[iR]
                        dA_orb_results[(R_vec, mi, mj)] = dA_orb_np[iR]

        return {
            "A": A_results,
            "dA": dA_results,
            "A_orb": A_orb_results,
            "dA_orb": dA_orb_results,
            "rho": jax_to_numpy(rhoR0),
        }

    def calculate_all(self, use_gpu=True, vectorize_energy=True, e_batch_size=None):
        """Calculate exchange parameters, processing one energy at a time."""
        print(
            "Green's function Calculation started (ExchangePert2GPU, energy-by-energy)."
        )
        self.validate()

        self._prepare_jax_arrays()

        Rpts = [tuple(R) for R in self.short_Rlist]
        Rset = set(Rpts)
        self._prepare_dGR_jax(Rpts, Rset, self.short_Rlist, self.epc, self.Ru)

        npole = len(self.contour.path)
        weights = self.contour.weights

        self.A_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.A_ijR_orb = {}
        self.dA_ijR_orb = {}

        compute_rho_gf = not (
            hasattr(self, "density_method") and self.density_method == "eigenvector"
        )
        if compute_rho_gf:
            self.rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        else:
            self.rho = self.G.get_density_matrix()

        for ie, e in enumerate(
            tqdm(self.contour.path, total=npole, desc="Energy integration")
        ):
            result = self._compute_per_energy_gpu(e)
            w = weights[ie]

            if compute_rho_gf:
                self.rho += result["rho"] * w

            for key, val in result["A"].items():
                self.A_ijR[key] += val * w
            for key, val in result["dA"].items():
                self.dA_ijR[key] += val * w

            if self.orb_decomposition:
                for key, val in result["A_orb"].items():
                    if key in self.A_ijR_orb:
                        self.A_ijR_orb[key] += val * w
                        self.dA_ijR_orb[key] += result["dA_orb"][key] * w
                    else:
                        self.A_ijR_orb[key] = val * w
                        self.dA_ijR_orb[key] = result["dA_orb"][key] * w

        if compute_rho_gf:
            self.rho = -1.0 / np.pi * self.rho
            if (
                hasattr(self, "integration_method")
                and self.integration_method.lower() == "cfr"
            ):
                self.rho = self.rho + 0.5j * np.eye(self.nbasis)

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

    def run(
        self,
        path="TB2J_results",
        use_gpu=True,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Run calculations."""
        self.calculate_all(
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.write_output(path=path)
        self.finalize()
