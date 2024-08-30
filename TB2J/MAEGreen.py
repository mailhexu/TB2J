from pathlib import Path

import numpy as np
import tqdm
from HamiltonIO.abacus.abacus_wrapper import AbacusSplitSOCParser
from HamiltonIO.model.occupations import Occupations
from typing_extensions import DefaultDict

from TB2J.exchange import ExchangeNCL

# from HamiltonIO.model.rotate_spin import rotate_Matrix_from_z_to_axis, rotate_Matrix_from_z_to_sperical
# from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import (
    rotate_spinor_matrix,
)


def get_occupation(evals, kweights, nel, width=0.1):
    occ = Occupations(nel=nel, width=width, wk=kweights, nspin=2)
    return occ.occupy(evals)


class MAEGreen(ExchangeNCL):
    def __init__(self, angles=None, **kwargs):
        super().__init__(**kwargs)
        self.natoms = len(self.atoms)
        if angles is None or angles == "axis":
            self.set_angles_axis()
        elif angles == "scan":
            self.set_angles_scan()
        else:
            self.thetas = angles[0]
            self.phis = angles[1]

        nangles = len(self.thetas)
        self.es = np.zeros(nangles, dtype=complex)
        self.es_atom = np.zeros((nangles, self.natoms), dtype=complex)
        self.es_atom_orb = DefaultDict(lambda: 0)

    def set_angles_axis(self):
        """theta and phi are defined as the x, y, z, axis."""
        self.thetas = [0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi]
        self.phis = [0, 0, np.pi / 2, np.pi / 4, 0]

    def set_angles_scan(self, step=15):
        self.thetas = []
        self.phis = []
        for i in range(0, 181, step):
            for j in range(0, 181, step):
                self.thetas.append(i * np.pi / 180)
                self.phis.append(j * np.pi / 180)

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

    def get_perturbed(self, e, thetas, phis):
        self.tbmodel.set_so_strength(0.0)
        G0K = self.G.get_Gk_all(e)
        Hsoc_k = self.tbmodel.get_Hk_soc(self.G.kpts)
        na = len(thetas)
        dE_angle = np.zeros(na, dtype=complex)
        dE_angle_atom = np.zeros((na, self.natoms), dtype=complex)
        # dE_angle_orbitals = np.zeros((na, self.natoms, self.norb, self.norb), dtype=complex)
        # dE_angle_orbitals = DefaultDict(lambda: 0)
        dE_angle_atom_orb = DefaultDict(lambda: 0)
        for iangle, (theta, phi) in enumerate(zip(thetas, phis)):
            for ik, dHk in enumerate(Hsoc_k):
                dHi = rotate_spinor_matrix(dHk, theta, phi)
                GdH = G0K[ik] @ dHi
                # dE += np.trace(GdH @ G0K[i].T.conj() @ dHi) * self.kweights[i]
                # diagonal of second order perturbation.
                dG2diag = np.diag(GdH @ GdH)
                # dG2 = np.einsum("ij,ji->ij", GdH,   GdH)
                dG2 = GdH * GdH.T
                dG2sum = np.sum(dG2)
                # print(f"dG2sum-sum: {dG2sum}")

                # dG2sum = np.trace(GdH @ GdH)
                # print(f"dG2sum-Tr: {dG2sum}")
                # dG1sum = np.trace(GdH)
                # print(f"dG1sum-Tr: {dG1sum}")

                # dG2diag = np.diag(GdH @G0K[i].T.conj() @ dHi)
                # dE_angle[iangle] += np.trace(GdH@GdH) * self.G.kweights[ik]
                # dE_angle[iangle] += np.trace(GdH@G0K[ik].T.conj()@dHi ) * self.G.kweights[ik]
                dE_angle[iangle] += dG2sum * self.G.kweights[ik]
                for iatom in range(self.natoms):
                    iorb = self.iorb(iatom)
                    # dG2= dG2[::2, ::2] + dG2[1::2, 1::2] + dG2[1::2, ::2] + dG2[::2, 1::2]
                    dE_atom_orb = dG2[np.ix_(iorb, iorb)] * self.G.kweights[ik]
                    dE_atom_orb = (
                        dE_atom_orb[::2, ::2]
                        + dE_atom_orb[1::2, 1::2]
                        + dE_atom_orb[1::2, ::2]
                        + dE_atom_orb[::2, 1::2]
                    )
                    mmat = self.mmats[iatom]
                    dE_atom_orb = mmat.T @ dE_atom_orb @ mmat

                    dE_angle_atom_orb[(iangle, iatom)] += dE_atom_orb

                    dE_atom = np.sum(dG2diag[iorb]) * self.G.kweights[ik]
                    # dE_atom = np.sum(dE_atom_orb)
                    dE_angle_atom[iangle, iatom] += dE_atom
        return dE_angle, dE_angle_atom, dE_angle_atom_orb

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

        for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
            dE_angle, dE_angle_atom, dE_angle_atom_orb = self.get_perturbed(
                e, thetas, phis
            )
            self.es += dE_angle * self.contour.weights[ie]
            self.es_atom += dE_angle_atom * self.contour.weights[ie]
            for key, value in dE_angle_atom_orb.items():
                self.es_atom_orb[key] += (
                    dE_angle_atom_orb[key] * self.contour.weights[ie]
                )

        self.es = -np.imag(self.es) / (2 * np.pi)
        self.es_atom = -np.imag(self.es_atom) / (2 * np.pi)
        for key, value in self.es_atom_orb.items():
            self.es_atom_orb[key] = -np.imag(value) / (2 * np.pi)

    def output(self, output_path="TB2J_anisotropy", with_eigen=False):
        Path(output_path).mkdir(exist_ok=True)
        fname = f"{output_path}/MAE.dat"
        fname_orb = f"{output_path}/MAE_orb.dat"

        # ouput with eigenvalues.
        if with_eigen:
            fname_eigen = f"{output_path}/MAE_eigen.dat"
            with open(fname_eigen, "w") as f:
                f.write("# theta, phi, MAE(total), MAE(atom-wise) Unit: meV\n")
                for i, (theta, phi, e, es) in enumerate(
                    zip(self.thetas, self.phis, self.es2, self.es_atom)
                ):
                    f.write(f"{theta:.5f} {phi:.5f} {e*1e3:.8f} ")
                    for ea in es:
                        f.write(f"{ea*1e3:.8f} ")
                    f.write("\n")

        with open(fname, "w") as f:
            f.write("# theta, phi, MAE(total), MAE(atom-wise) Unit: meV\n")
            for i, (theta, phi, e, es) in enumerate(
                zip(self.thetas, self.phis, self.es, self.es_atom)
            ):
                f.write(f"{theta:.5f} {phi:.5f} {e*1e3:.8f} ")
                for ea in es:
                    f.write(f"{ea*1e3:.8f} ")
                f.write("\n")

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
            for i, (theta, phi, e, eatom) in enumerate(
                zip(self.thetas, self.phis, self.es, self.es_atom)
            ):
                f.write("-" * 60 + "\n")
                f.write(f"Angle {i:03d}:   theta={theta:.5f} phi={phi:.5f} \n ")
                f.write(f"E: {e*1e3:.8f} \n")
                for iatom, ea in enumerate(eatom):
                    f.write(f"Atom {iatom:03d}: {ea*1e3:.8f} \n")
                    f.write("Orbital: ")
                    eorb = self.es_atom_orb[(i, iatom)]

                    # write numpy matrix to file
                    f.write(
                        np.array2string(
                            eorb * 1e3, precision=4, separator=",", suppress_small=True
                        )
                    )

                    eorb_diff = eorb - self.es_atom_orb[(0, iatom)]
                    f.write("Diference to the first angle: ")
                    f.write(
                        np.array2string(
                            eorb_diff * 1e3,
                            precision=4,
                            separator=",",
                            suppress_small=True,
                        )
                    )
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
    magnetic_elements=None,
    nel=None,
    width=0.1,
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
        magnetic_elements=magnetic_elements,
        **kwargs,
    )
    mae.run(output_path=output_path)
