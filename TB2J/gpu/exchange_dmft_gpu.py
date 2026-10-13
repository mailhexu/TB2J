"""
ExchangeDMFTGPU: GPU-accelerated DMFT exchange calculations using JAX.
"""

from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchange_dmft import ExchangeCLDMFT, ExchangeDMFTNCL
from TB2J.gpu.jax_utils import (
    _compute_GR_single_e,
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)
from TB2J.gpu.jax_utils import (
    _jit as jit,
)
from TB2J.gpu.jax_utils import (
    _jnp as jnp,
)


@jit
def _compute_Gk_with_sigma_jax(Hk, Sk, efermi, energy, sigma):
    """
    Compute G(k, omega) = inv((omega + efermi) * S - H - sigma) on GPU.

    Hk: (nb, nb)
    Sk: (nb, nb) or None
    energy: scalar (complex Matsubara frequency)
    sigma: (nb, nb) self-energy at this frequency
    Returns: (nb, nb)
    """
    if Sk is None:
        Sk = jnp.eye(Hk.shape[0], dtype=Hk.dtype)
    matrix = (energy + efermi) * Sk - Hk - sigma
    return jnp.linalg.inv(matrix)


@jit
def _compute_Gk_with_sigma_spin_jax(
    Hk_up, Hk_dn, Sk, efermi, energy, sigma_up, sigma_dn
):
    """
    Compute G_up(k, omega) and G_dn(k, omega) for collinear case.

    Returns: G_up (nb, nb), G_dn (nb, nb)
    """
    if Sk is None:
        Sk_eye = jnp.eye(Hk_up.shape[0], dtype=Hk_up.dtype)
    else:
        Sk_eye = Sk
    G_up = jnp.linalg.inv((energy + efermi) * Sk_eye - Hk_up - sigma_up)
    G_dn = jnp.linalg.inv((energy + efermi) * Sk_eye - Hk_dn - sigma_dn)
    return G_up, G_dn


@jit
def _compute_Gk_with_sigma_spin_allk_jax(
    Hk_up_all, Hk_dn_all, Sk_all, efermi, energy, sigma_up, sigma_dn
):
    """
    Compute G_up/down for all k-points in one batched linear algebra call.
    """
    matrix_up = (energy + efermi) * Sk_all - Hk_up_all - sigma_up[None, :, :]
    matrix_dn = (energy + efermi) * Sk_all - Hk_dn_all - sigma_dn[None, :, :]
    return jnp.linalg.inv(matrix_up), jnp.linalg.inv(matrix_dn)


@jit
def _compute_Gk_with_sigma_allk_jax(Hk_all, Sk_all, efermi, energy, sigma):
    """
    Compute G for all k-points in one batched linear algebra call.
    """
    matrix = (energy + efermi) * Sk_all - Hk_all - sigma[None, :, :]
    return jnp.linalg.inv(matrix)


@jit
def _compute_Gk_with_sigma_spin_batch_allk_jax(
    Hk_up_all, Hk_dn_all, Sk_all, efermi, energies, sigma_up, sigma_dn
):
    """
    Compute G_up/down for a bounded batch of frequencies and all k-points.

    Shapes:
      energies: (ne,)
      sigma_up/dn: (ne, nb, nb)
      Hk/Sk: (nk, nb, nb)
      returns: (ne, nk, nb, nb) for each spin channel
    """
    z = energies[:, None, None, None] + efermi
    matrix_up = (
        z * Sk_all[None, :, :, :] - Hk_up_all[None, :, :, :] - sigma_up[:, None, :, :]
    )
    matrix_dn = (
        z * Sk_all[None, :, :, :] - Hk_dn_all[None, :, :, :] - sigma_dn[:, None, :, :]
    )
    return jnp.linalg.inv(matrix_up), jnp.linalg.inv(matrix_dn)


@jit
def _compute_GR_batch_jax(Rpts, kpts, Gk_batch, kweights, k2Rfactor):
    """Fourier transform G(k) to G(R) for a frequency batch."""
    phase = jnp.exp(k2Rfactor * jnp.einsum("ri,ki->rk", Rpts, kpts))
    return jnp.einsum("ekij,rk,k->erij", Gk_batch, phase, kweights, optimize="optimal")


@jit
def _compute_collinear_A_batch_sum_jax(
    GR_up, GR_dn, indices_neg, idx, jdx, Delta_i, Delta_j
):
    """Compute summed A tensors over a bounded frequency batch."""
    Gij_up = jnp.take(jnp.take(GR_up, idx, axis=2), jdx, axis=3)
    Gji_dn = jnp.take(
        jnp.take(jnp.take(GR_dn, indices_neg, axis=1), jdx, axis=2), idx, axis=3
    )
    A_orb = jnp.einsum(
        "eab,erbc,ecd,erda->erac",
        Delta_i,
        Gij_up,
        Delta_j,
        Gji_dn,
        optimize="optimal",
    ) / (4.0 * jnp.pi)
    return jnp.sum(A_orb, axis=0), jnp.sum(A_orb, axis=(0, 2, 3))


@jit
def _compute_collinear_A_dmft_jax(Gij_up, Gji_dn, Delta_i, Delta_j):
    """
    Compute collinear DMFT A tensor for all R vectors on GPU.

    Gij_up: (nR, ni, nj)
    Gji_dn: (nR, nj, ni)
    Delta_i: (ni, ni) frequency-dependent splitting for atom i
    Delta_j: (nj, nj) frequency-dependent splitting for atom j

    Returns: A_orb (nR, ni, ni), A_total (nR,)
    """
    t = jnp.einsum(
        "ab,rbc,cd,rda->rac",
        Delta_i,
        Gij_up,
        Delta_j,
        Gji_dn,
        optimize="optimal",
    ) / (4.0 * jnp.pi)
    A_total = jnp.sum(t, axis=(1, 2))
    return t, A_total


def _compute_Hk_jax(Rpts_jax, HR_jax, kpts_jax, R2kfactor):
    """
    Compute H(k) = sum_R H(R) * exp(R2kfactor * k.dot.R) for all k.

    Returns: (nk, nb, nb)
    """
    k_dot_R = jnp.einsum("ki,ri->kr", kpts_jax, Rpts_jax)
    phase = jnp.exp(R2kfactor * k_dot_R)
    Hk = jnp.einsum("rij,kr->kij", HR_jax, phase)
    return Hk


def _compute_Sk_jax(Rpts_jax, SR_jax, kpts_jax, R2kfactor):
    """
    Compute S(k) for all k. Returns None if orthogonal.
    """
    if SR_jax is None:
        return None
    k_dot_R = jnp.einsum("ki,ri->kr", kpts_jax, Rpts_jax)
    phase = jnp.exp(R2kfactor * k_dot_R)
    Sk = jnp.einsum("rij,kr->kij", SR_jax, phase)
    return Sk


class ExchangeCLDMFTGPU(ExchangeCLDMFT):
    """
    GPU-accelerated collinear DMFT exchange using JAX.

    Overrides calculate_all() to compute Green's functions with self-energy
    on GPU, batch processing Matsubara frequencies.
    """

    def __init__(self, tbmodels, atoms, **params):
        _require_jax()
        super().__init__(tbmodels=tbmodels, atoms=atoms, **params)
        self._jax_prepared = False

    def _prepare_jax_arrays(self):
        """Prepare static JAX arrays for GPU computation."""
        if self._jax_prepared:
            return

        tbmodel = self.tbmodel
        static = tbmodel.static_model

        if hasattr(static, "HR") and hasattr(static, "Rlist"):
            HR = static.HR
            Rpts = static.Rlist
            SR = static.SR if hasattr(static, "SR") else None
        elif hasattr(static, "data"):
            Rpts = np.array(list(static.data.keys()))
            HR = np.array([static.data[tuple(R)] for R in Rpts])
            SR = None
        else:
            raise ValueError("Static model must have HR/Rlist or data attributes")

        self._Rpts_static_jax = numpy_to_jax(Rpts)
        self._HR_jax = numpy_to_jax(HR)
        self._SR_jax = numpy_to_jax(SR) if SR is not None else None
        self._R2kfactor = static.R2kfactor

        self._kpts_jax = numpy_to_jax(self.G.kpts)
        self._kweights_jax = numpy_to_jax(self.G.kweights)
        self._Rpts_jax = numpy_to_jax(self.short_Rlist)

        Hk_up = []
        Hk_dn = []
        Sk_all = []
        for ik, kpt in enumerate(self.G.kpts):
            Hk_up.append(tbmodel.get_hamiltonian(kpt, ispin=0))
            Hk_dn.append(tbmodel.get_hamiltonian(kpt, ispin=1))
            Sk = self.G.get_Sk(ik)
            if Sk is None:
                Sk = np.eye(self.nbasis, dtype=complex)
            Sk_all.append(Sk)

        self._Hk_up_all_jax = numpy_to_jax(np.asarray(Hk_up))
        self._Hk_dn_all_jax = numpy_to_jax(np.asarray(Hk_dn))
        self._Sk_all_jax = numpy_to_jax(np.asarray(Sk_all))

        if hasattr(tbmodel, "_sigma_full") and tbmodel._sigma_full is not None:
            self._sigma_full_jax = numpy_to_jax(tbmodel._sigma_full)
        else:
            self._sigma_full_jax = None
        self._mesh_jax = (
            numpy_to_jax(tbmodel.mesh) if tbmodel.mesh is not None else None
        )

        self._efermi_jax = float(self.G.efermi)

        print(
            f"DMFT GPU prepared: Hk {self._Hk_up_all_jax.shape}, nk={len(self.G.kpts)}, nR={len(self.short_Rlist)}"
        )

        self._jax_prepared = True

    def _get_sigma_at_e_jax(self, energy):
        """
        Get self-energy at given Matsubara frequency as JAX arrays.

        Returns: sigma_up (nb, nb), sigma_dn (nb, nb) as JAX arrays
        """
        if self._sigma_full_jax is None:
            return None, None

        sigma_full = self._sigma_full_jax
        if self.tbmodel.is_static:
            if sigma_full.ndim == 3:
                return sigma_full[0], sigma_full[1]
            return sigma_full, sigma_full
        else:
            diffs = jnp.abs(self._mesh_jax - energy)
            idx = jnp.argmin(diffs)
            if sigma_full.ndim == 4:
                return sigma_full[0, idx], sigma_full[1, idx]
            else:
                return sigma_full[idx], sigma_full[idx]

    def _compute_GR_dmft_gpu(self, energy):
        """
        Compute GR_up and GR_dn for a single Matsubara frequency on GPU.
        Includes self-energy in the Green's function.

        Returns: GR_up (nR, nb, nb), GR_dn (nR, nb, nb) as JAX arrays
        """
        sigma_up, sigma_dn = self._get_sigma_at_e_jax(energy)
        if sigma_up is None:
            sigma_up = jnp.zeros_like(self._Hk_up_all_jax[0])
            sigma_dn = jnp.zeros_like(self._Hk_dn_all_jax[0])

        Gk_up_all, Gk_dn_all = _compute_Gk_with_sigma_spin_allk_jax(
            self._Hk_up_all_jax,
            self._Hk_dn_all_jax,
            self._Sk_all_jax,
            self._efermi_jax,
            energy,
            sigma_up,
            sigma_dn,
        )

        GR_up = _compute_GR_single_e(
            self._Rpts_jax,
            self._kpts_jax,
            Gk_up_all,
            self._kweights_jax,
            self.G.k2Rfactor,
        )
        GR_dn = _compute_GR_single_e(
            self._Rpts_jax,
            self._kpts_jax,
            Gk_dn_all,
            self._kweights_jax,
            self.G.k2Rfactor,
        )

        return GR_up, GR_dn

    def _get_sigma_batch_jax(self, energies):
        """Get spin-resolved self-energy for a bounded frequency batch."""
        ne = energies.shape[0]
        if self._sigma_full_jax is None:
            sigma_up = jnp.zeros((ne,) + self._Hk_up_all_jax.shape[1:], dtype=complex)
            sigma_dn = jnp.zeros((ne,) + self._Hk_dn_all_jax.shape[1:], dtype=complex)
            return sigma_up, sigma_dn

        sigma_full = self._sigma_full_jax
        if self.tbmodel.is_static:
            if sigma_full.ndim == 3:
                return (
                    jnp.broadcast_to(sigma_full[0], (ne,) + sigma_full.shape[1:]),
                    jnp.broadcast_to(sigma_full[1], (ne,) + sigma_full.shape[1:]),
                )
            sigma = jnp.broadcast_to(sigma_full, (ne,) + sigma_full.shape)
            return sigma, sigma

        diffs = jnp.abs(self._mesh_jax[None, :] - energies[:, None])
        idx = jnp.argmin(diffs, axis=1)
        if sigma_full.ndim == 4:
            return sigma_full[0, idx], sigma_full[1, idx]
        sigma = sigma_full[idx]
        return sigma, sigma

    def _compute_GR_dmft_gpu_batch(self, energies):
        """Compute GR_up and GR_dn for a bounded batch of Matsubara frequencies."""
        sigma_up, sigma_dn = self._get_sigma_batch_jax(energies)
        Gk_up, Gk_dn = _compute_Gk_with_sigma_spin_batch_allk_jax(
            self._Hk_up_all_jax,
            self._Hk_dn_all_jax,
            self._Sk_all_jax,
            self._efermi_jax,
            energies,
            sigma_up,
            sigma_dn,
        )
        GR_up = _compute_GR_batch_jax(
            self._Rpts_jax, self._kpts_jax, Gk_up, self._kweights_jax, self.G.k2Rfactor
        )
        GR_dn = _compute_GR_batch_jax(
            self._Rpts_jax, self._kpts_jax, Gk_dn, self._kweights_jax, self.G.k2Rfactor
        )
        return GR_up, GR_dn

    def _compute_P_iatom_e_jax(self, iatom, energy):
        """
        Compute frequency-dependent splitting P(iwn) for collinear DMFT on GPU.

        Returns: Delta = 2 * P as JAX array (n_orb_i, n_orb_i)
        """
        P_np = self.get_P_iatom_e(iatom, energy)
        return numpy_to_jax(2.0 * P_np)

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated computation for one Matsubara frequency.
        """
        energy = complex(e)
        GR_up, GR_dn = self._compute_GR_dmft_gpu(energy)

        magnetic_sites = self.ind_mag_atoms
        n_mag = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]

        Delta = [self._compute_P_iatom_e_jax(site, energy) for site in magnetic_sites]

        AijR = {}
        AijR_orb = {}
        rho_e = np.zeros((2, self.nbasis, self.nbasis), dtype=complex)

        GR_up_np = jax_to_numpy(GR_up)
        GR_dn_np = jax_to_numpy(GR_dn)
        iR0 = self.Rvec_to_shortlist_idx.get((0, 0, 0), 0)
        rho_e[0] = GR_up_np[iR0]
        rho_e[1] = GR_dn_np[iR0]

        indices_neg = jnp.array(
            [self.R_negative_index[k] for k in range(len(self.short_Rlist))]
        )

        for i, j in product(range(n_mag), repeat=2):
            idx = jnp.array(iorbs[i])
            jdx = jnp.array(iorbs[j])

            Gij_up = jnp.take(jnp.take(GR_up, idx, axis=1), jdx, axis=2)
            Gji_dn = jnp.take(
                jnp.take(jnp.take(GR_dn, indices_neg, axis=0), jdx, axis=1),
                idx,
                axis=2,
            )

            A_orb, A_total = _compute_collinear_A_dmft_jax(
                Gij_up, Gji_dn, Delta[i], Delta[j]
            )
            A_total.block_until_ready()

            A_orb_np = jax_to_numpy(A_orb)
            A_total_np = jax_to_numpy(A_total)

            mi, mj = magnetic_sites[i], magnetic_sites[j]
            for iR, R_vec in enumerate(self.short_Rlist):
                AijR[(R_vec, mi, mj)] = A_total_np[iR]
                if self.orb_decomposition:
                    AijR_orb[(R_vec, mi, mj)] = A_orb_np[iR]

        return dict(AijR=AijR, AijR_orb=AijR_orb, rho_e=rho_e)

    def get_quantities_per_e_batch_gpu(self, batch):
        """GPU-accelerated summed quantities for a bounded frequency batch."""
        energies_np = np.asarray(batch, dtype=complex)
        energies = numpy_to_jax(energies_np)
        GR_up, GR_dn = self._compute_GR_dmft_gpu_batch(energies)

        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]
        Delta = []
        for site in magnetic_sites:
            Delta.append(
                [
                    2.0 * self.get_P_iatom_e(site, complex(energy))
                    for energy in energies_np
                ]
            )
        Delta = [numpy_to_jax(np.asarray(site_delta)) for site_delta in Delta]

        AijR_sum = {}
        AijR_orb_sum = {}
        indices_neg = jnp.array(
            [self.R_negative_index[k] for k in range(len(self.short_Rlist))]
        )

        for i, j in product(range(len(magnetic_sites)), repeat=2):
            idx = jnp.array(iorbs[i])
            jdx = jnp.array(iorbs[j])
            A_orb_sum, A_total_sum = _compute_collinear_A_batch_sum_jax(
                GR_up, GR_dn, indices_neg, idx, jdx, Delta[i], Delta[j]
            )
            A_total_sum.block_until_ready()

            A_total_np = jax_to_numpy(A_total_sum)
            A_orb_np = jax_to_numpy(A_orb_sum) if self.orb_decomposition else None
            mi, mj = magnetic_sites[i], magnetic_sites[j]
            for iR, R_vec in enumerate(self.short_Rlist):
                AijR_sum[(R_vec, mi, mj)] = A_total_np[iR]
                if self.orb_decomposition:
                    AijR_orb_sum[(R_vec, mi, mj)] = A_orb_np[iR]

        rho_sum = None
        if not getattr(self.tbmodel, "is_static", False):
            iR0 = self.Rvec_to_shortlist_idx.get((0, 0, 0), 0)
            rho_sum = np.zeros((2, self.nbasis, self.nbasis), dtype=complex)
            rho_sum[0] = np.sum(jax_to_numpy(GR_up[:, iR0]), axis=0)
            rho_sum[1] = np.sum(jax_to_numpy(GR_dn[:, iR0]), axis=0)

        return dict(AijR_sum=AijR_sum, AijR_orb_sum=AijR_orb_sum, rho_sum=rho_sum)

    def calculate_all_matsubara_batched(self, e_batch_size=None):
        """Batched Matsubara calculation with bounded frequency memory."""
        AijR_sum = {}
        AijR_orb_sum = {}
        rho_sum = None

        path = list(self.contour.path)
        npole = len(path)
        batch_size = e_batch_size or 8

        for start in tqdm(
            range(0, npole, batch_size), total=(npole + batch_size - 1) // batch_size
        ):
            result = self.get_quantities_per_e_batch_gpu(
                path[start : start + batch_size]
            )
            for key, val in result["AijR_sum"].items():
                AijR_sum[key] = AijR_sum.get(key, 0.0j) + val
            if self.orb_decomposition:
                for key, val in result["AijR_orb_sum"].items():
                    AijR_orb_sum[key] = AijR_orb_sum.get(key, 0.0j) + val
            if result["rho_sum"] is not None:
                if rho_sum is None:
                    rho_sum = result["rho_sum"]
                else:
                    rho_sum += result["rho_sum"]

        T = self.temperature
        for iR in self.R_ijatom_dict:
            R_vec = self.short_Rlist[iR]
            for iatom, jatom in self.R_ijatom_dict[iR]:
                key = (R_vec, iatom, jatom)
                self.A_ijR[key] = -2.0 * np.pi * T * np.real(AijR_sum[key])
                if self.orb_decomposition:
                    self.A_ijR_orb[key] = -2.0 * np.pi * T * np.real(AijR_orb_sum[key])

        if rho_sum is not None:
            eye = np.eye(self.nbasis, dtype=complex)
            self.rho = np.empty_like(rho_sum)
            self.rho[0] = 0.5 * eye + 2.0 * T * np.real(rho_sum[0])
            self.rho[1] = 0.5 * eye + 2.0 * T * np.real(rho_sum[1])

        self.get_rho_atom()
        self.A_to_Jtensor()

    def calculate_all(self, use_gpu=True, vectorize_energy=False, e_batch_size=None):
        """
        GPU-accelerated DMFT exchange calculation.
        Loops over Matsubara frequencies, computing on GPU.
        """
        print("Green's function Calculation started (DMFT Collinear GPU).")
        self._prepare_jax_arrays()

        if self._exchange_method == "matsubara":
            self.calculate_all_matsubara_batched(e_batch_size=e_batch_size)
            return

        AijRs = {}
        AijRs_orb = {}
        rho_list = []

        path = list(self.contour.path)
        npole = len(path)
        batch_size = e_batch_size or (npole if vectorize_energy else 1)

        for start in tqdm(
            range(0, npole, batch_size), total=(npole + batch_size - 1) // batch_size
        ):
            batch = path[start : start + batch_size]
            for result in map(self.get_quantities_per_e_gpu, batch):
                if not getattr(self.tbmodel, "is_static", False):
                    rho_list.append(result["rho_e"])
                for iR in self.R_ijatom_dict:
                    R_vec = self.short_Rlist[iR]
                    for iatom, jatom in self.R_ijatom_dict[iR]:
                        key = (R_vec, iatom, jatom)
                        val = result["AijR"].get(key)
                        if val is not None:
                            if key in AijRs:
                                AijRs[key].append(val)
                            else:
                                AijRs[key] = [val]
                            if self.orb_decomposition:
                                val_orb = result["AijR_orb"].get(key)
                                if val_orb is not None:
                                    if key in AijRs_orb:
                                        AijRs_orb[key].append(val_orb)
                                    else:
                                        AijRs_orb[key] = [val_orb]

        if getattr(self.tbmodel, "is_static", False):
            self.integrate(
                AijRs, AijRs_orb, rho_list=None, method=self._exchange_method
            )
        else:
            self.integrate(
                AijRs, AijRs_orb, rho_list=rho_list, method=self._exchange_method
            )

        self.get_rho_atom()
        self.A_to_Jtensor()

    def run(
        self,
        path="TB2J_results",
        use_gpu=True,
        vectorize_energy=False,
        e_batch_size=None,
    ):
        self.calculate_all(
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.write_output(path=path)
        self.finalize()


class ExchangeDMFTNCLGPU(ExchangeDMFTNCL):
    """
    GPU-accelerated non-collinear DMFT exchange using JAX.
    """

    def __init__(self, tbmodels, atoms, **params):
        _require_jax()
        super().__init__(tbmodels=tbmodels, atoms=atoms, **params)
        self._jax_prepared = False

    def _prepare_jax_arrays(self):
        """Prepare static JAX arrays for GPU computation."""
        if self._jax_prepared:
            return

        tbmodel = self.tbmodel
        static = tbmodel.static_model

        if hasattr(static, "HR") and hasattr(static, "Rlist"):
            HR = static.HR
            Rpts = static.Rlist
            SR = static.SR if hasattr(static, "SR") else None
        elif hasattr(static, "data"):
            Rpts = np.array(list(static.data.keys()))
            HR = np.array([static.data[tuple(R)] for R in Rpts])
            SR = None
        else:
            raise ValueError("Static model must have HR/Rlist or data attributes")

        self._Rpts_static_jax = numpy_to_jax(Rpts)
        self._HR_jax = numpy_to_jax(HR)
        self._SR_jax = numpy_to_jax(SR) if SR is not None else None
        self._R2kfactor = static.R2kfactor

        self._kpts_jax = numpy_to_jax(self.G.kpts)
        self._kweights_jax = numpy_to_jax(self.G.kweights)
        self._Rpts_jax = numpy_to_jax(self.short_Rlist)

        Hk_all = []
        Sk_all = []
        for ik, kpt in enumerate(self.G.kpts):
            Hk_all.append(tbmodel.get_hamiltonian(kpt, ispin=0))
            Sk = self.G.get_Sk(ik)
            if Sk is None:
                Sk = np.eye(self.nbasis, dtype=complex)
            Sk_all.append(Sk)

        self._Hk_all_jax = numpy_to_jax(np.asarray(Hk_all))
        self._Sk_all_jax = numpy_to_jax(np.asarray(Sk_all))

        if hasattr(tbmodel, "_sigma_full") and tbmodel._sigma_full is not None:
            self._sigma_full_jax = numpy_to_jax(tbmodel._sigma_full)
        else:
            self._sigma_full_jax = None
        self._mesh_jax = (
            numpy_to_jax(tbmodel.mesh) if tbmodel.mesh is not None else None
        )

        self._efermi_jax = float(self.G.efermi)

        print(
            f"DMFT NCL GPU prepared: Hk {self._Hk_all_jax.shape}, nk={len(self.G.kpts)}, nR={len(self.short_Rlist)}"
        )

        self._jax_prepared = True

    def _get_sigma_at_e_jax_ncl(self, energy):
        """
        Get NCL self-energy at given Matsubara frequency as JAX array.
        Returns spinor sigma (2n, 2n) or None.
        """
        try:
            return numpy_to_jax(self.tbmodel.get_sigma(energy, ispin=None))
        except TypeError:
            return numpy_to_jax(self.tbmodel.get_sigma(energy))

    def _compute_GR_ncl_gpu(self, energy):
        """
        Compute GR for NCL DMFT at one Matsubara frequency on GPU.
        Returns: GR (nR, nb, nb) as JAX array (spinor basis)
        """
        sigma = self._get_sigma_at_e_jax_ncl(energy)
        if sigma is None:
            sigma = jnp.zeros_like(self._Hk_all_jax[0])

        Gk_all = _compute_Gk_with_sigma_allk_jax(
            self._Hk_all_jax, self._Sk_all_jax, self._efermi_jax, energy, sigma
        )

        GR = _compute_GR_single_e(
            self._Rpts_jax, self._kpts_jax, Gk_all, self._kweights_jax, self.G.k2Rfactor
        )
        return GR

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated computation for one Matsubara frequency (NCL case).
        """
        energy = complex(e)
        GR_jax = self._compute_GR_ncl_gpu(energy)

        self.Pdict = {}
        for iatom in self.ind_mag_atoms:
            self.Pdict[iatom] = self.get_P_iatom_e(iatom, energy)

        try:
            AijR, AijR_orb = self.get_all_A_vectorized(jax_to_numpy(GR_jax))
        except Exception:
            AijR, AijR_orb = self.get_all_A(jax_to_numpy(GR_jax))

        iR0 = self.Rvec_to_shortlist_idx.get((0, 0, 0), 0)
        rho_e = jax_to_numpy(GR_jax)[iR0]

        return dict(AijR=AijR, AijR_orb=AijR_orb, rho_e=rho_e)

    def calculate_all(self, use_gpu=True, vectorize_energy=False, e_batch_size=None):
        """
        GPU-accelerated NCL DMFT exchange calculation.
        """
        print("Green's function Calculation started (DMFT NCL GPU).")
        self._prepare_jax_arrays()

        AijRs = {}
        AijRs_orb = {}
        rho_list = []

        path = list(self.contour.path)
        npole = len(path)
        batch_size = e_batch_size or (npole if vectorize_energy else 1)

        for start in tqdm(
            range(0, npole, batch_size), total=(npole + batch_size - 1) // batch_size
        ):
            batch = path[start : start + batch_size]
            for result in map(self.get_quantities_per_e_gpu, batch):
                if not getattr(self.tbmodel, "is_static", False):
                    rho_list.append(result["rho_e"])
                for iR in self.R_ijatom_dict:
                    R_vec = self.short_Rlist[iR]
                    for iatom, jatom in self.R_ijatom_dict[iR]:
                        key = (R_vec, iatom, jatom)
                        val = result["AijR"].get(key)
                        if val is not None:
                            if key in AijRs:
                                AijRs[key].append(val)
                            else:
                                AijRs[key] = [val]
                            if self.orb_decomposition:
                                val_orb = result["AijR_orb"].get(key)
                                if val_orb is not None:
                                    if key in AijRs_orb:
                                        AijRs_orb[key].append(val_orb)
                                    else:
                                        AijRs_orb[key] = [val_orb]

        if getattr(self.tbmodel, "is_static", False):
            self.integrate(
                AijRs, AijRs_orb, rho_list=None, method=self._exchange_method
            )
        else:
            self.integrate(
                AijRs, AijRs_orb, rho_list=rho_list, method=self._exchange_method
            )

        self.get_rho_atom()
        self.A_to_Jtensor()

    def run(
        self,
        path="TB2J_results",
        use_gpu=True,
        vectorize_energy=False,
        e_batch_size=None,
    ):
        self.calculate_all(
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.write_output(path=path)
        self.finalize()
