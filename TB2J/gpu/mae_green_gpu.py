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


def _get_perturbed_all_jax(G0K, Hsoc_k, thetas, phis, kweights, atom_slices):
    """
    JAX version: Compute perturbed energy for all angles and all energy points.
    Includes atom-resolved contributions.
    atom_slices: jnp.array (nb,) mapping each orbital to an atom index.
    """
    _require_jax()
    natoms = jnp.max(atom_slices) + 1

    # Define the core logic for one (energy, angle) pair
    def compute_single(Gk, Hk, theta, phi, kw):
        dHi = _rotate_spinor_matrix_einsum_jax(Hk, theta, phi)
        GdH = Gk @ dHi
        dG2 = GdH * GdH.T  # Element-wise product, then transpose

        # Atom-resolved summation
        # We need to sum dG2[i, j] for i in iatom, j in jatom
        # This is essentially a 2D segment sum.
        # Step 1: Sum over rows (orbital j) for each atom
        # We can use jax.ops.segment_sum if we reshape or use row-wise vmap

        def segment_sum_2d(mat):
            # Sum rows by atom
            row_sum = vmap(lambda row: jnp.zeros(natoms).at[atom_slices].add(row))(
                mat
            )  # (nb, natoms)
            # Sum columns by atom
            total_sum = vmap(lambda col: jnp.zeros(natoms).at[atom_slices].add(col))(
                row_sum.T
            )  # (natoms, natoms)
            return total_sum

        dE_matrix = segment_sum_2d(dG2) * kw
        dE_total = jnp.sum(dE_matrix)

        return dE_total, dE_matrix

    # Vectorize over k-points
    v_k = vmap(compute_single, in_axes=(0, 0, None, None, 0))

    # Vectorize over angles
    v_angle = vmap(
        lambda Gk_all, Hk_all, theta, phi, kw_all: v_k(
            Gk_all, Hk_all, theta, phi, kw_all
        ),
        in_axes=(None, None, 0, 0, None),
    )

    # Vectorize over energies
    v_energy = vmap(
        lambda Gk_all_e, Hk_all, thetas, phis, kw_all: v_angle(
            Gk_all_e, Hk_all, thetas, phis, kw_all
        ),
        in_axes=(0, None, None, None, None),
    )

    return v_energy(G0K, Hsoc_k, thetas, phis, kweights)


class MAEGreenGPU(MAEGreen):
    """
    GPU-accelerated version of MAEGreen using JAX.
    """

    def __init__(self, angles=None, **kwargs):
        """Initialize MAEGreenGPU."""
        _require_jax()
        super().__init__(angles=angles, **kwargs)

        # Prepare orbital to atom mapping for segment sums
        self.atom_slices = np.zeros(self.nbasis, dtype=int)
        for iatom in range(self.natoms):
            iorb = self.iorb(iatom)
            self.atom_slices[iorb] = iatom
        self.atom_slices_jax = numpy_to_jax(self.atom_slices)

    def _get_band_energy_vs_angles_gpu_per_e(
        self, e, thetas, phis, Hsoc_k, kweights_jax, thetas_jax, phis_jax
    ):
        """Process a single energy point on GPU."""
        evals_jax = numpy_to_jax(self.G.evals)
        evecs_jax = numpy_to_jax(self.G.evecs)
        Gk = _eigen_to_G_jax(evals_jax, evecs_jax, self.G.efermi, e)  # (nk, nb, nb)

        # Use _get_perturbed_all_jax logic but for ne=1
        dE_total, dE_matrix = _get_perturbed_all_jax(
            Gk[None, ...],
            Hsoc_k,
            thetas_jax,
            phis_jax,
            kweights_jax,
            self.atom_slices_jax,
        )
        return dE_total[0], dE_matrix[0]  # remove ne dimension

    def _get_band_energy_vs_angles_gpu(
        self, thetas, phis, vectorize_energy=True, e_batch_size=None
    ):
        """Internal GPU-accelerated implementation."""
        self.tbmodel.set_so_strength(0.0)
        Hsoc_k_np = self.tbmodel.get_Hk_soc(self.G.kpts)
        Hsoc_k = numpy_to_jax(Hsoc_k_np)
        kweights_jax = numpy_to_jax(self.G.kweights)
        thetas_jax = numpy_to_jax(thetas)
        phis_jax = numpy_to_jax(phis)
        weights_jax = numpy_to_jax(self.contour.weights)
        npole = len(self.contour.path)

        if not vectorize_energy:
            print("Computing MAE energy-by-energy on GPU (Vectorized Atom-resolved).")
            for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
                dE_total, dE_matrix = self._get_band_energy_vs_angles_gpu_per_e(
                    e, thetas, phis, Hsoc_k, kweights_jax, thetas_jax, phis_jax
                )
                w = self.contour.weights[ie]
                # dE_total: (na,), dE_matrix: (na, natoms, natoms)
                self.es += -np.imag(jax_to_numpy(dE_total) * w) / (2 * np.pi)
                self.es_matrix += -np.imag(jax_to_numpy(dE_matrix) * w) / (2 * np.pi)
        else:
            print(
                "Computing MAE with vectorized GPU parallelization (Vectorized Atom-resolved)."
            )
            if e_batch_size is None or e_batch_size >= npole:
                batches = [range(npole)]
            else:
                batches = [
                    range(i, min(i + e_batch_size, npole))
                    for i in range(0, npole, e_batch_size)
                ]

            evals_jax = numpy_to_jax(self.G.evals)
            evecs_jax = numpy_to_jax(self.G.evecs)

            for batch_indices in batches:
                energy_batch = self.contour.path[list(batch_indices)]
                energy_jax = numpy_to_jax(energy_batch)
                batch_weights_jax = weights_jax[list(batch_indices)]

                G0K_batch = _eigen_to_G_jax(
                    evals_jax[None, ...],
                    evecs_jax[None, ...],
                    self.G.efermi,
                    energy_jax[:, None, None],
                )
                dE_total_batch, dE_matrix_batch = _get_perturbed_all_jax(
                    G0K_batch,
                    Hsoc_k,
                    thetas_jax,
                    phis_jax,
                    kweights_jax,
                    self.atom_slices_jax,
                )

                # Integrated results on GPU
                es_batch = jnp.einsum("e, ea -> a", batch_weights_jax, dE_total_batch)
                es_matrix_batch = jnp.einsum(
                    "e, eam n -> amn", batch_weights_jax, dE_matrix_batch
                )

                self.es += -np.imag(jax_to_numpy(es_batch)) / (2 * np.pi)
                self.es_matrix += -np.imag(jax_to_numpy(es_matrix_batch)) / (2 * np.pi)

        # Orbital decomposition (still on CPU if requested, but MAE usually focuses on atom-resolved)
        # Note: self.es_atom_orb is rarely used in Siesta interface unless specifically requested.

    def get_band_energy_vs_angles(
        self,
        thetas,
        phis,
        with_eigen=False,
        use_gpu=True,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Calculate band energy vs angles using GPU acceleration."""
        if with_eigen:
            self.es2 = self.get_band_energy_vs_angles_from_eigen(thetas, phis)
        if use_gpu:
            self._get_band_energy_vs_angles_gpu(
                thetas,
                phis,
                vectorize_energy=vectorize_energy,
                e_batch_size=e_batch_size,
            )
        else:
            super().get_band_energy_vs_angles(thetas, phis, with_eigen=False)

    def run(
        self,
        output_path="TB2J_anisotropy",
        with_eigen=False,
        use_gpu=True,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """Run the MAE calculation with optional GPU acceleration."""
        self.get_band_energy_vs_angles(
            self.thetas,
            self.phis,
            with_eigen=with_eigen,
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.output(output_path=output_path, with_eigen=with_eigen)


# Keep compatibility with old naming
MAEGreenJAX = MAEGreenGPU
MAE_Green_GPU = MAEGreenGPU
