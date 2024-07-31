from pathlib import Path

import numpy as np
import tqdm

# from TB2J.abacus.abacus_wrapper import AbacusSplitSOCParser
from TB2J.exchange import ExchangeNCL

# from HamiltonIO.model.rotate_spin import rotate_Matrix_from_z_to_axis, rotate_Matrix_from_z_to_sperical
# from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import (
    rotate_spinor_matrix,
)


class MAEGreen(ExchangeNCL):
    def __init__(self, angles=None, **kwargs):
        super().__init__(**kwargs)
        self.es = None
        self.natoms = len(self.atoms)
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
        dE_angle_atoms = np.zeros((na, self.natoms), dtype=complex)
        for iangle, (theta, phi) in enumerate(zip(thetas, phis)):
            for ik, dHk in enumerate(Hsoc_k):
                dHi = rotate_spinor_matrix(dHk, theta, phi)
                GdH = G0K[ik] @ dHi
                # dE += np.trace(GdH @ G0K[i].T.conj() @ dHi) * self.kweights[i]
                # diagonal of second order perturbation.
                dG2diag = np.diag(GdH @ GdH)
                # dG2diag = np.diag(GdH @G0K[i].T.conj() @ dHi)
                dE_angle[iangle] += np.sum(dG2diag) * self.G.kweights[ik]
                for iatom in range(self.natoms):
                    dE_atom = np.sum(dG2diag[self.iorb(iatom)]) * self.G.kweights[ik]
                    dE_angle_atoms[iangle, iatom] += dE_atom
        return dE_angle, dE_angle_atoms

    def get_band_energy_vs_angles(self, thetas, phis):
        nangles = len(thetas)
        self.es = np.zeros(nangles, dtype=float)
        self.es_atoms = np.zeros((nangles, self.natoms), dtype=float)
        for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
            dE_angle, dE_angle_atoms = self.get_perturbed(e, thetas, phis)
            self.es -= np.imag(dE_angle * self.contour.weights[ie]) / np.pi
            self.es_atoms -= np.imag(dE_angle_atoms * self.contour.weights[ie]) / np.pi

    def output(self, output_path="TB2J_anisotropy"):
        Path(output_path).mkdir(exist_ok=True)
        fname = f"{output_path}/MAE.dat"
        with open(fname, "w") as f:
            f.write("# theta phi MAE, MAEatom1, atom2, ...\n")
            for i, (theta, phi, e, es) in enumerate(
                zip(self.thetas, self.phis, self.es, self.es_atoms)
            ):
                f.write(f"{theta:.5f} {phi:.5f} {e:.8f} ")
                for ea in es:
                    f.write(f"{ea:.8f} ")
                f.write("\n")

    def run(self, output_path="TB2J_anisotropy"):
        self.get_band_energy_vs_angles(self.thetas, self.phis)
        self.output(output_path=output_path)
