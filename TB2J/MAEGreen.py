import gc
from pathlib import Path

import numpy as np
import tqdm
from HamiltonIO.abacus.abacus_wrapper import AbacusSplitSOCParser
from HamiltonIO.model.occupations import GaussOccupations
from typing_extensions import DefaultDict

from TB2J.anisotropy import Anisotropy
from TB2J.exchange import ExchangeNCL
from TB2J.external import p_imap
from TB2J.mathutils.fibonacci_sphere import fibonacci_semisphere

# from HamiltonIO.model.rotate_spin import rotate_Matrix_from_z_to_axis, rotate_Matrix_from_z_to_sperical
# from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import (
    rotate_spinor_matrix,
)
from TB2J.sharedmem import attach_shm, detach_shm, free_shm, to_shm


def get_occupation(evals, kweights, nel, width=0.1):
    occ = GaussOccupations(nel=nel, width=width, wk=kweights, nspin=2)
    return occ.occupy(evals)


class MAEGreen(ExchangeNCL):
    def __init__(self, angles=None, **kwargs):
        """
        angles are defined as theta, phi pairs, where theta is the angle between the z-axis and the magnetization direction, and phi is the angle between the x-axis and the projection of the magnetization direction on the x-y plane.
        """
        super().__init__(**kwargs)
        self.natoms = len(self.atoms)
        if angles is None or angles == "xyz":
            self.set_angles_xyz()
        elif angles == "axis":
            self.set_angles_axis()
        elif angles == "scan":
            self.set_angles_scan()
        elif angles == "fib":
            self.set_angles_fib()
        elif angles == "random":
            self.set_angles_random()
        elif angles == "miller":
            self.set_angles_miller()
        elif angles == "ztox":
            self.set_angels_ztox()
        else:
            self.thetas = angles[0]
            self.phis = angles[1]

        nangles = len(self.thetas)
        self.es = np.zeros(nangles, dtype=complex)
        self.es_matrix = np.zeros((nangles, self.natoms, self.natoms), dtype=complex)
        self.es_atom_orb = DefaultDict(lambda: 0)

    def set_angles_xyz(self):
        """theta and phi are defined as the x, y, z, xy, yz, xz, xyz, x-yz, -xyz, -x-yz axis."""
        self.thetas = [np.pi / 2, np.pi / 2, 0.0]
        self.phis = [np.pi / 2, 0, 0.0]

    def set_angles_axis(self):
        """theta and phi are defined as the x, y, z, xy, yz, xz, xyz, x-yz, -xyz, -x-yz axis."""
        self.thetas = [0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi, 0, np.pi / 2, 0, 0, 0]
        self.phis = [0, 0, np.pi / 2, np.pi / 4, 0, 0, 0, np.pi]

    def set_angles_miller(self, nmax=2):
        """theta and angles corresponding to the miller index. remove duplicates.
        e.g. 002 and 001 are the same.
        """
        thetas = []
        phis = []
        for k in range(0, nmax + 1):
            for j in range(-nmax, nmax + 1):
                for i in range(-nmax, nmax + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    thetas.append(np.arccos(k / np.sqrt(i**2 + j**2 + k**2)))
                    if i == 0 and j == 0:
                        phis.append(0)
                    else:
                        phis.append(np.arctan2(j, i))
        self.thetas = thetas
        self.phis = phis
        self.angle_pairs = list(zip(thetas, phis))
        self.angle_pairs = list(set(self.angle_pairs))
        self.thetas, self.phis = zip(*self.angle_pairs)

    def set_angles_scan(self, step=15):
        self.thetas = []
        self.phis = []
        for i in range(0, 181, step):
            for j in range(0, 181, step):
                self.thetas.append(i * np.pi / 180)
                self.phis.append(j * np.pi / 180)

    def set_angels_ztox(self, n=16):
        """Set angles for a scan from z to x"""
        self.thetas = np.linspace(0, np.pi, n)
        self.phis = np.zeros(n)

    def set_angles_random(self, n=16):
        # n random pairs of theta, phi
        self.thetas = np.random.random(n) * np.pi
        self.phis = np.random.random(n) * 2 * np.pi

    def set_angles_fib(self, n=35):
        self.thetas, self.phis = fibonacci_semisphere(n)
        # thetas and phis are np.array
        # add (theta, phi): (pi/2, pi/2) and (pi/2, pi/4)
        # self.thetas += [np.pi / 2, np.pi / 2]
        # self.phis += [np.pi / 2, np.pi / 4]
        for i in range(8):
            self.thetas = np.concatenate([self.thetas, [np.pi / 2]])
            self.phis = np.concatenate([self.phis, [np.pi * i / 8]])

    def get_band_energy_vs_angles_from_eigen(
        self,
        thetas,
        phis,
    ):
        """
        Calculate the band energy for a given set of angles by using the eigenvalues and eigenvectors of the Hamiltonian.
        """
        es = []
        nangles = len(thetas)
        self.tbmodel.set_so_strength(1.0)
        for i in tqdm.trange(nangles):
            theta = thetas[i]
            phi = phis[i]
            self.tbmodel.set_Hsoc_rotation_angle([theta, phi])
            e = self.get_band_energy()
            es.append(e)
        return es

    def get_band_energy(self):
        self.width = 0.1
        evals, _evecs = self.tbmodel.solve_all(self.G.kpts)
        occ = get_occupation(evals, self.G.kweights, self.tbmodel.nel, width=self.width)
        eband = np.sum(evals * occ * self.G.kweights[:, np.newaxis])
        return eband

    def _setup_Hsoc_k_shm(self, Hsoc_k):
        """Copy Hsoc_k into a shared memory block and store descriptor."""
        self._shm_Hsoc_k, self._desc_Hsoc_k = to_shm(np.asarray(Hsoc_k), "Hsoc_k")
        self._use_shm_Hsoc_k = True

    def _teardown_Hsoc_k_shm(self):
        """Release the shared memory block for Hsoc_k."""
        if getattr(self, "_shm_Hsoc_k", None) is not None:
            free_shm(self._shm_Hsoc_k)
            self._shm_Hsoc_k = None
        self._use_shm_Hsoc_k = False

    def get_perturbed(self, e, thetas, phis):
        self.tbmodel.set_so_strength(0.0)
        # Reconstruct Hsoc_k from shared memory if available, otherwise use self.Hsoc_k
        if getattr(self, "_use_shm_Hsoc_k", False):
            Hsoc_k, _shm_Hsoc_k = attach_shm(self._desc_Hsoc_k)
        else:
            Hsoc_k = self.Hsoc_k
            _shm_Hsoc_k = None

        # G0K shape: (Nk, N, N)
        G0K = self.G.get_Gk_all(e)
        na = len(thetas)
        dE_angle = np.zeros(na, dtype=complex)
        dE_angle_matrix = np.zeros((na, self.natoms, self.natoms), dtype=complex)
        dE_angle_atom_orb = DefaultDict(lambda: 0)

        # Decompose is mostly False by default, so we can optimize for that case.
        # If decompose is False, we use vectorized operations over K-points.
        self.decompose = getattr(self, "decompose", False)

        for iangle, (theta, phi) in enumerate(zip(thetas, phis)):
            # Vectorized rotation for all k-points at once
            # Hsoc_k has shape (Nk, 2N, 2N)
            dHi_k = rotate_spinor_matrix(Hsoc_k, theta, phi)

            if not self.decompose:
                # Optimized vectorized calculation for total energy across all K-points
                # GdH_k shape: (Nk, 2N, 2N)
                GdH_k = G0K @ dHi_k
                # Tr(GdH @ GdH) summed over k with weights
                dE_angle[iangle] = np.einsum(
                    "k,kij,kji->", self.G.kweights, GdH_k, GdH_k, optimize=True
                )
            else:
                # Fallback to loop only if decomposition is requested
                for ik, dHi in enumerate(dHi_k):
                    GdH = G0K[ik] @ dHi
                    dG2 = GdH * GdH.T
                    dG2sum = np.sum(dG2)
                    dE_angle[iangle] += dG2sum * self.G.kweights[ik]

                    for iatom in range(self.natoms):
                        iorb = self.iorb(iatom)
                        for jatom in range(self.natoms):
                            jorb = self.iorb(jatom)
                            # Calculate cross terms between atoms i and j
                            dE_ij_orb = dG2[np.ix_(iorb, jorb)] * self.G.kweights[ik]
                            dE_ij_orb = (
                                dE_ij_orb[::2, ::2]
                                + dE_ij_orb[1::2, 1::2]
                                + dE_ij_orb[1::2, ::2]
                                + dE_ij_orb[::2, 1::2]
                            )
                            dE_ij = np.sum(dE_ij_orb)
                            # Transform to local orbital basis
                            mmat_i = self.mmats[iatom]
                            mmat_j = self.mmats[jatom]
                            dE_ij_orb = mmat_i.T @ dE_ij_orb @ mmat_j
                            dE_angle_matrix[iangle, iatom, jatom] += dE_ij
                            # Store orbital-resolved data for diagonal terms
                            if iatom == jatom:
                                dE_angle_atom_orb[(iangle, iatom)] += dE_ij_orb

        if _shm_Hsoc_k is not None:
            detach_shm(_shm_Hsoc_k)
        return dE_angle, dE_angle_matrix, dE_angle_atom_orb

    def get_perturbed_R(self, e, thetas, phis):
        self.tbmodel.set_so_strength(0.0)
        # Here only the first R vector is considered.
        # TODO: consider all R vectors.
        # Rlist = np.zeros((1, 3), dtype=float)
        # G0K = self.G.get_Gk_all(e)
        # G0R = k_to_R(self.G.kpts, Rlist, G0K, self.G.kweights)
        # dE_angle = np.zeros(len(thetas), dtype=complex)
        # dE_angle_atoms = np.zeros((len(thetas), self.natoms), dtype=complex)
        pass

    def get_band_energy_vs_angles(self, thetas, phis, with_eigen=False):
        if with_eigen:
            self.es2 = self.get_band_energy_vs_angles_from_eigen(thetas, phis)

        # for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
        #    dE_angle, dE_angle_atom, dE_angle_atom_orb = self.get_perturbed(
        #        e, thetas, phis
        #    )
        #    self.es += dE_angle * self.contour.weights[ie]
        #    self.es_atom += dE_angle_atom * self.contour.weights[ie]
        #    for key, value in dE_angle_atom_orb.items():
        #        self.es_atom_orb[key] += (
        #            dE_angle_atom_orb[key] * self.contour.weights[ie]
        #        )

        # rewrite the for loop above to use p_map
        self.Hsoc_k = self.tbmodel.get_Hk_soc(self.G.kpts)

        def func(e):
            return self.get_perturbed(e, thetas, phis)

        if self.nproc > 1:
            # Move Hsoc_k and self.G's large arrays into shared memory so dill
            # only serialises shm metadata (names/shapes) into each worker process.
            self._setup_Hsoc_k_shm(self.Hsoc_k)
            # Delete the local copy so dill does not serialise it redundantly
            # into every worker alongside the shm descriptor.
            del self.Hsoc_k
            self.G.enter_parallel()
            try:
                results = p_imap(func, self.contour.path, num_cpus=self.nproc)
                for weight, result in zip(self.contour.weights, results):
                    dE_angle, dE_angle_matrix, dE_angle_atom_orb = result
                    self.es += dE_angle * weight
                    self.es_matrix += dE_angle_matrix * weight
                    for key, value in dE_angle_atom_orb.items():
                        self.es_atom_orb[key] += value * weight
            finally:
                self._teardown_Hsoc_k_shm()
                self.G.exit_parallel()
        else:
            self._use_shm_Hsoc_k = False
            npole = len(self.contour.path)
            results = map(func, tqdm.tqdm(self.contour.path, total=npole))
            for weight, result in zip(self.contour.weights, results):
                dE_angle, dE_angle_matrix, dE_angle_atom_orb = result
                self.es += dE_angle * weight
                self.es_matrix += dE_angle_matrix * weight
                for key, value in dE_angle_atom_orb.items():
                    self.es_atom_orb[key] += value * weight

        self.es = -np.imag(self.es) / (2 * np.pi)
        self.es_matrix = -np.imag(self.es_matrix) / (2 * np.pi)
        for key, value in self.es_atom_orb.items():
            self.es_atom_orb[key] = -np.imag(value) / (2 * np.pi)

    def fit_anisotropy_tensor(self):
        self.ani = Anisotropy.fit_from_data(self.thetas, self.phis, self.es)
        return self.ani

    def output(
        self,
        output_path="TB2J_anisotropy",
        with_eigen=True,
        figure3d="MAE_3d.png",
        figure_contourf="MAE_contourf.png",
    ):
        Path(output_path).mkdir(exist_ok=True)
        fname = f"{output_path}/MAE.dat"
        fname_orb = f"{output_path}/MAE_orb.dat"
        fname_matrix = f"{output_path}/MAE_matrix.dat"
        # fname_tensor = f"{output_path}/MAE_tensor.dat"
        # if figure3d is not None:
        #    fname_fig3d = f"{output_path}/{figure3d}"
        # if figure_contourf is not None:
        #    fname_figcontourf = f"{output_path}/{figure_contourf}"

        # ouput with eigenvalues.
        if with_eigen:
            fname_eigen = f"{output_path}/MAE_eigen.dat"
            with open(fname_eigen, "w") as f:
                f.write("# theta, phi, MAE(total) Unit: meV\n")
                for i, (theta, phi, e) in enumerate(
                    zip(self.thetas, self.phis, self.es2)
                ):
                    f.write(f"{theta:.5f} {phi:.5f} {e*1e3:.8f}\n")

        with open(fname, "w") as f:
            f.write("# theta (rad), phi(rad), MAE(total) Unit: meV\n")
            for i, (theta, phi, e) in enumerate(zip(self.thetas, self.phis, self.es)):
                f.write(f"{theta%np.pi:.5f} {phi%(2*np.pi):.5f} {e*1e3:.8f}\n")

        # Write matrix data to MAE_matrix.dat
        with open(fname_matrix, "w") as fmat:
            fmat.write("# MAE atom-atom interaction matrices\n")
            fmat.write("# Format: angle_index theta phi atom_i atom_j MAE_ij(meV)\n")
            fmat.write("# Units: theta and phi in radians, MAE in meV\n")
            for iangle, (theta, phi) in enumerate(zip(self.thetas, self.phis)):
                for iatom in range(self.natoms):
                    for jatom in range(self.natoms):
                        mae_ij = (
                            self.es_matrix[iangle, iatom, jatom] * 1e3
                        )  # Convert to meV
                        fmat.write(
                            f"{iangle:4d} {theta:.5f} {phi:.5f} {iatom:4d} {jatom:4d} {mae_ij:.8f}\n"
                        )
                fmat.write("\n")  # Empty line between angles for readability

        # self.ani = self.fit_anisotropy_tensor()
        # with open(fname_tensor, "w") as f:
        #    f.write("# Anisotropy tensor in meV\n")
        #    f.write(f"{self.ani.tensor_strings(include_isotropic=False)}\n")

        # if figure3d is not None:
        #    self.ani.plot_3d(figname=fname_fig3d, show=False)

        # if figure_contourf is not None:
        #    self.ani.plot_contourf(figname=fname_figcontourf, show=False)

        # plt.close()
        gc.collect()

        with open(fname_orb, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Orbitals for each atom: \n")
            f.write("=" * 80 + "\n")
            f.write("Note: the energies are in meV\n")
            for iatom in range(self.natoms):
                f.write(f"Atom {iatom:03d}: ")
                for orb in self.orbital_names[iatom]:
                    f.write(f"{orb} ")
                f.write("\n")
            for i, (theta, phi, e) in enumerate(zip(self.thetas, self.phis, self.es)):
                f.write("-" * 60 + "\n")
                f.write(f"Angle {i:03d}:   theta={theta:.5f} phi={phi:.5f} \n ")
                f.write(f"E: {e*1e3:.8f} \n")
                for iatom in range(self.natoms):
                    f.write(f"Atom {iatom:03d} orbital matrix:\n")
                    if (i, iatom) in self.es_atom_orb:
                        eorb = self.es_atom_orb[(i, iatom)]
                        # write numpy matrix to file
                        f.write(
                            np.array2string(
                                eorb * 1e3,
                                precision=4,
                                separator=",",
                                suppress_small=True,
                            )
                        )
                        f.write("\n")

                        if (0, iatom) in self.es_atom_orb:
                            eorb_diff = eorb - self.es_atom_orb[(0, iatom)]
                            f.write("Difference to the first angle: ")
                            f.write(
                                np.array2string(
                                    eorb_diff * 1e3,
                                    precision=4,
                                    separator=",",
                                    suppress_small=True,
                                )
                            )
                            f.write("\n")
                f.write("\n")

    def run(self, output_path="TB2J_anisotropy", with_eigen=False):
        self.get_band_energy_vs_angles(self.thetas, self.phis, with_eigen=with_eigen)
        self.output(output_path=output_path, with_eigen=with_eigen)


def abacus_get_MAE(
    path_nosoc,
    path_soc,
    kmesh,
    thetas,
    phis,
    gamma=True,
    output_path="TB2J_anisotropy",
    nel=None,
    width=0.1,
    with_eigen=False,
    **kwargs,
):
    """Get MAE from Abacus with magnetic force theorem. Two calculations are needed. First we do an calculation with SOC but the soc_lambda is set to 0. Save the density. The next calculatin we start with the density from the first calculation and set the SOC prefactor to 1. With the information from the two calcualtions, we can get the band energy with magnetic moments in the direction, specified in two list, thetas, and phis."""
    parser = AbacusSplitSOCParser(
        outpath_nosoc=path_nosoc, outpath_soc=path_soc, binary=False
    )
    model = parser.parse()
    model.set_so_strength(0.0)
    if nel is not None:
        model.nel = nel
    mae = MAEGreen(
        tbmodels=model,
        atoms=model.atoms,
        kmesh=kmesh,
        efermi=None,
        basis=model.basis,
        angles=[thetas, phis],
        **kwargs,
    )
    mae.run(output_path=output_path, with_eigen=with_eigen)
