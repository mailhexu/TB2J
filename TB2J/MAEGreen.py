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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thetas = None
        self.phis = None
        self.es = None

    def set_angles(self, thetas, phis):
        self.thetas = thetas
        self.phis = phis

    def get_perturbed(self, e, thetas, phis):
        self.model.set_soc_strength(0.0)
        G0K = self.G.get_Gk_all(e)
        Hsoc_k = self.model.get_Hk_soc(self.kpts)
        na = len(thetas)
        natom = self.model.natoms
        dE_angle = np.zeros(na, dtype=complex)
        dE_angle_atoms = np.zeros((na, natom), dtype=complex)
        for iangle, (theta, phi) in enumerate(zip(thetas, phis)):
            for i, dHk in enumerate(Hsoc_k):
                dHi = rotate_spinor_matrix(dHk, theta, phi)
                GdH = G0K[i] @ dHi
                # dE += np.trace(GdH @ G0K[i].T.conj() @ dHi) * self.kweights[i]
                # diagonal of second order perturbation.
                dG2diag = np.diag(GdH @ GdH)
                dE_angle[iangle] += np.sum(dG2diag) * self.kweights[i]
                for iatom in range(self.model.natoms):
                    dE_atom = np.sum(dG2diag[self.iorb(iatom)]) * self.kweights[i]
                    dE_angle_atoms[iangle, iatom] += dE_atom
        return dE_angle, dE_angle_atoms

    def get_band_energy_vs_angles(self, thetas, phis):
        nangles = len(thetas)
        es = np.zeros(nangles, dtype=float)
        es_atoms = np.zeros((nangles, self.model.natoms), dtype=float)
        for ie, e in enumerate(tqdm.tqdm(self.contour.path)):
            dE_angle, dE_angle_atoms = self.get_perturbed(e, thetas, phis)
            es += np.imag(dE_angle * self.contour.weights[ie])
            es_atoms += np.imag(dE_angle_atoms * self.contour.weights[ie])
        return -es / np.pi

    def output(self, fname="TB2J_anisotropy/MAE.dat"):
        with open(fname, "w") as f:
            f.write("# theta phi MAE\n")
            for i, (theta, phi, e) in enumerate(
                zip(self.contour.thetas, self.contour.phis, self.es)
            ):
                f.write(f"{theta} {phi} {e}\n")

    def run(self, thetas=None, phis=None, output_path="TB2J_anisotropy"):
        self.es = self.get_band_energy_vs_angles(thetas, phis)
        self.output()
