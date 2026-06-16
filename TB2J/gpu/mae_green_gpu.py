"""
MAEGreenGPU: GPU-accelerated version of MAEGreen using JAX.
"""

import numpy as np
import tqdm

from TB2J.gpu.jax_utils import (
    _eigen_to_G_jax,
    _require_jax,
    _rotate_spinor_matrix_einsum_jax,
    jax_to_numpy,
    numpy_to_jax,
)
from TB2J.gpu.jax_utils import (
    _jnp as jnp,
)
from TB2J.gpu.jax_utils import (
    _vmap as vmap,
)
from TB2J.MAEGreen import MAEGreen


def _get_perturbed_single_angle_jax(G0K, dHi_k, kweights):
    """
    JAX version: Compute perturbed energy for a single angle over all energy points and k-points.

    The rotation dHi_k = R(theta, phi) @ Hsoc_k must be pre-computed by the caller.

    Parameters
    ----------
    G0K    : jnp.ndarray, shape (ne, nk, nb, nb)
    dHi_k  : jnp.ndarray, shape (nk, nb, nb)   — already rotated SOC Hamiltonian
    kweights : jnp.ndarray, shape (nk,)

    Returns
    -------
    dE : jnp.ndarray, shape (ne,)   — energy contribution for this angle
    """
    _require_jax()

    def compute_one_ek(Gk, dHk, kw):
        """Contribution from one (energy, k) pair — scalar."""
        GdH = Gk @ dHk
        dG2 = GdH * GdH.T
        return jnp.sum(dG2) * kw

    def sum_over_k_one_e(Gk_e):
        """Sum kw-weighted contributions over all k-points for one energy."""
        per_k = vmap(compute_one_ek, in_axes=(0, 0, 0))(Gk_e, dHi_k, kweights)  # (nk,)
        return jnp.sum(per_k)  # scalar

    # Vectorize over energy points -> (ne,)
    return vmap(sum_over_k_one_e, in_axes=(0,))(G0K)


class MAEGreenGPU(MAEGreen):
    """
    GPU-accelerated version of MAEGreen using JAX.
    """

    def __init__(self, angles=None, **kwargs):
        """Initialize MAEGreenGPU."""
        _require_jax()
        super().__init__(angles=angles, **kwargs)

    def _auto_e_batch_size(self):
        """
        Estimate a safe e_batch_size to avoid GPU OOM.

        The dominant allocation is G0K_batch = (ne_batch, nk, nb, nb) complex128.
        We use 10% of currently-available GPU memory to stay well clear of the
        fragmentation boundary left by the exchange calculation.
        """
        bytes_per_e = self.G.nkpts * self.nbasis * self.nbasis * 16
        target_bytes = 500e6  # 500 MB conservative default
        try:
            import jax

            gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
            if gpu_devices:
                stats = gpu_devices[0].memory_stats()
                if stats:
                    total = stats.get("bytes_limit", 0)
                    in_use = stats.get("bytes_in_use", 0)
                    available = total - in_use
                    if available > 0:
                        # 10% target: leaves room for BFC fragmentation + XLA intermediates
                        target_bytes = available * 0.10
        except Exception:
            pass
        batch_size = max(1, int(target_bytes / bytes_per_e))
        npole = len(self.contour.path)
        return min(batch_size, npole)

    @staticmethod
    def _is_oom_error(exc):
        """Return True if exc looks like a GPU out-of-memory error."""
        msg = str(exc)
        return (
            "RESOURCE_EXHAUSTED" in msg
            or "Out of memory" in msg
            or "out of memory" in msg
        )

    def _run_angles_batched(
        self, thetas, phis, e_batch_size, evals_jax, evecs_jax, Hsoc_k, kweights_jax
    ):
        """
        Inner loop: iterate over angles (Python) and energy batches (JAX).

        Returns
        -------
        es_result : np.ndarray, shape (na,)
            Real-valued energy contributions for each angle.
        """
        na = len(thetas)
        npole = len(self.contour.path)

        if e_batch_size >= npole:
            batches = [range(npole)]
        else:
            batches = [
                range(i, min(i + e_batch_size, npole))
                for i in range(0, npole, e_batch_size)
            ]

        es_result = np.zeros(na, dtype=float)
        for ia, (theta, phi) in enumerate(
            tqdm.tqdm(zip(thetas, phis), total=na, desc="angles")
        ):
            theta_jax = numpy_to_jax(np.array(theta))
            phi_jax = numpy_to_jax(np.array(phi))
            dHi_k = vmap(
                lambda Hk: _rotate_spinor_matrix_einsum_jax(Hk, theta_jax, phi_jax)
            )(Hsoc_k)

            es_angle = np.float64(0.0)
            for batch_indices in batches:
                idx = np.array(list(batch_indices))
                energy_jax = numpy_to_jax(self.contour.path[idx])
                batch_weights_jax = numpy_to_jax(self.contour.weights[idx])

                G0K_batch = _eigen_to_G_jax(
                    evals_jax[None, ...],
                    evecs_jax[None, ...],
                    self.G.efermi,
                    energy_jax[:, None, None],
                )  # (ne_batch, nk, nb, nb)

                dE_batch = _get_perturbed_single_angle_jax(
                    G0K_batch, dHi_k, kweights_jax
                )  # (ne_batch,)

                es_contribution = jnp.dot(batch_weights_jax, dE_batch)  # scalar
                es_angle += -np.imag(complex(jax_to_numpy(es_contribution))) / (
                    2 * np.pi
                )

            es_result[ia] = float(es_angle)
        return es_result

    def _get_band_energy_vs_angles_gpu(
        self, thetas, phis, vectorize_energy=True, e_batch_size=None
    ):
        """Internal GPU-accelerated implementation.

        The angle loop runs in Python; JAX only vmaps over (ne_batch, nk).
        Peak intermediate tensor: (ne_batch, nk, nb, nb) — no na factor.
        OOM recovery: if allocation fails, halves e_batch_size and retries.
        """
        self.tbmodel.set_so_strength(0.0)
        Hsoc_k = numpy_to_jax(self.tbmodel.get_Hk_soc(self.G.kpts))  # (nk, nb, nb)
        kweights_jax = numpy_to_jax(self.G.kweights)
        evals_jax = numpy_to_jax(self.G.evals)
        evecs_jax = numpy_to_jax(self.G.evecs)
        na = len(thetas)

        if not vectorize_energy:
            print("Computing MAE energy-by-energy on GPU.")
            for ia, (theta, phi) in enumerate(
                tqdm.tqdm(zip(thetas, phis), total=na, desc="angles")
            ):
                theta_jax = numpy_to_jax(np.array(theta))
                phi_jax = numpy_to_jax(np.array(phi))
                dHi_k = vmap(
                    lambda Hk: _rotate_spinor_matrix_einsum_jax(Hk, theta_jax, phi_jax)
                )(Hsoc_k)
                es_angle = 0.0
                for ie, e in enumerate(self.contour.path):
                    G0K = _eigen_to_G_jax(
                        evals_jax[None, ...],
                        evecs_jax[None, ...],
                        self.G.efermi,
                        numpy_to_jax(np.array([e]))[:, None, None],
                    )  # (1, nk, nb, nb)
                    dE = _get_perturbed_single_angle_jax(
                        G0K, dHi_k, kweights_jax
                    )  # (1,)
                    w = self.contour.weights[ie]
                    es_angle += -np.imag(complex(jax_to_numpy(dE[0])) * w) / (2 * np.pi)
                self.es[ia] += es_angle
        else:
            if e_batch_size is None:
                e_batch_size = self._auto_e_batch_size()
                print(f"Auto-selected e_batch_size={e_batch_size} based on GPU memory.")

            # OOM retry: halve e_batch_size and rerun from scratch until success
            MAX_RETRIES = 8
            for attempt in range(MAX_RETRIES):
                print(
                    f"Computing MAE with vectorized GPU parallelization "
                    f"(e_batch_size={e_batch_size}, na={na})."
                )
                try:
                    es_result = self._run_angles_batched(
                        thetas,
                        phis,
                        e_batch_size,
                        evals_jax,
                        evecs_jax,
                        Hsoc_k,
                        kweights_jax,
                    )
                    self.es += es_result
                    break  # success — exit retry loop
                except Exception as exc:
                    if self._is_oom_error(exc) and e_batch_size > 1:
                        e_batch_size = max(1, e_batch_size // 2)
                        print(
                            f"GPU OOM detected. Retrying with "
                            f"e_batch_size={e_batch_size}."
                        )
                    else:
                        raise

        self.es = np.real(self.es)

    def get_band_energy_vs_angles(
        self,
        thetas,
        phis,
        with_eigen=False,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Calculate band energy vs angles using GPU acceleration."""
        if with_eigen:
            self.es2 = self.get_band_energy_vs_angles_from_eigen(thetas, phis)
        self._get_band_energy_vs_angles_gpu(
            thetas,
            phis,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )

    def run(
        self,
        output_path="TB2J_anisotropy",
        with_eigen=False,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Run the MAE calculation with GPU acceleration."""
        self.get_band_energy_vs_angles(
            self.thetas,
            self.phis,
            with_eigen=with_eigen,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.output(output_path=output_path, with_eigen=with_eigen)


# Keep compatibility with old naming
MAEGreenJAX = MAEGreenGPU
MAE_Green_GPU = MAEGreenGPU
