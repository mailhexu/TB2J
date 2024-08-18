from pathlib import Path

import numpy as np
import tqdm
from typing_extensions import DefaultDict

# from TB2J.abacus.abacus_wrapper import AbacusSplitSOCParser
from TB2J.exchange import ExchangeNCL

# from HamiltonIO.model.rotate_spin import rotate_Matrix_from_z_to_axis, rotate_Matrix_from_z_to_sperical
# from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import (
    rotate_spinor_matrix,
)


class MAEGreen(ExchangeNCL):
    def __init__(self, angles=[], **kwargs):
        super().__init__(**kwargs)
        self.natoms = len(self.atoms)
        nangles = len(angles)
        self.es = np.zeros(nangles, dtype=float)
        self.es_atom = np.zeros((nangles, self.natoms), dtype=float)
        self.es_atom_orb = DefaultDict(lambda: 0)
        if angles is None or angles == "axis":
            self.set_angles_axis()
        elif angles == "scan":
            self.set_angles_scan()

    def set_angles_axis(self):
        """theta and phi are defined as the x, y, z, axis."""
        self.thetas = [0, np.pi / 2, np.pi / 2, np.pi / 2]
        self.phis = [0, 0, np.pi / 2, np.pi / 4]

    def set_angles_scan(self, step=15):
        self.thetas = []
        self.phis = []
        for i in range(0, 181, step):
            for j in range(0, 181, step):
                self.thetas.append(i * np.pi / 180)
                self.phis.append(j * np.pi / 180)

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
                # dG2diag = np.diag(GdH @ GdH)
                dG2 = np.einsum("ij,ji->ij", GdH, G0K[ik].T.conj() @ dHi)
                dG2sum = np.sum(dG2)
                # dG2diag = np.diag(GdH @G0K[i].T.conj() @ dHi)
                dE_angle[iangle] += dG2sum * self.G.kweights[ik]
                dE_angle = -np.imag(dE_angle) / np.pi
                for iatom in range(self.natoms):
                    dE_atom_orb = (
                        dG2[self.iorb(iatom), self.iorb(iatom)] * self.G.kweights[ik]
                    )
                    dE_atom_orb = -np.imag(dE_atom_orb) / np.pi
                    dE_angle_atom_orb[iangle, iatom] += dE_atom_orb

                    dE_atom = np.sum(dE_atom_orb)
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

    def get_band_energy_vs_angles(self, thetas, phis):
        for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
            dE_angle, dE_angle_atom, dE_angle_atom_orb = self.get_perturbed(
                e, thetas, phis
            )
            self.es += dE_angle * self.contour.weights[ie]
            self.es_atom += dE_angle_atom * self.contour.weights[ie]
            self.es_atom_orb += dE_angle_atom_orb * self.contour.weights[ie]

    def output(self, output_path="TB2J_anisotropy"):
        Path(output_path).mkdir(exist_ok=True)
        fname = f"{output_path}/MAE.dat"
        fname_orb = f"{output_path}/MAE_orb.dat"
        with open(fname, "w") as f:
            f.write("# theta phi MAE, MAEatom1, atom2, ...\n")
            for i, (theta, phi, e, es) in enumerate(
                zip(self.thetas, self.phis, self.es, self.es_atoms)
            ):
                f.write(f"{theta:.5f} {phi:.5f} {e:.8f} ")
                for ea in es:
                    f.write(f"{ea:.8f} ")
                f.write("\n")

        with open(fname_orb, "w") as f:
            f.write("=" * 80 + "\n")
            for i, (theta, phi, e, eatom, eorb) in enumerate(
                zip(self.thetas, self.phis, self.es, self.es_atoms, self.es_atom_orb)
            ):
                print("-" * 60)
                f.write(f"Angle {i:03d}:   theta={theta:.5f} phi={phi:.5f} \n ")
                f.write(f"E: {e:.8f} \n")
                for iatom, ea in enumerate(eatom):
                    f.write(f"Atom {iatom:03d}: {ea:.8f} \n")
                    f.write("Orbital: ")
                    # write numpy matrix to file
                    f.write(
                        np.array2string(
                            eorb[iatom], precision=4, separator=",", suppress_small=True
                        )
                    )
                f.write("\n")

    def run(self, output_path="TB2J_anisotropy"):
        self.get_band_energy_vs_angles(self.thetas, self.phis)
        self.output()
