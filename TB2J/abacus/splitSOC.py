import numpy as np
from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import rotate_Matrix_from_z_to_axis
from TB2J.kpoints import monkhorst_pack
from TB2J.mathutils.fermi import fermi
from copy import deepcopy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# TODO List:
# - [x] Add the class AbacusSplitSOCWrapper
# - [x] Add the function to rotate the XC part
# - [x] Compute the band energy at arbitrary


class AbacusSplitSOCWrapper(AbacusWrapper):
    """
    Abacus wrapper with Hamiltonian split to SOC and non-SOC parts
    """

    def __init__(self, *args, **kwargs):
        HR_soc = kwargs.pop("HR_soc", None)
        super().__init__(*args, **kwargs)
        self._HR_copy = deepcopy(self._HR)
        self.HR_soc = HR_soc

    @property
    def HR(self):
        return self._HR + self.HR_soc

    def rotate_Hxc(self, axis):
        """
        Rotate SOC part of Hamiltonian
        """
        # print("Before rotation:")
        # print(self._HR[0][:2, :2])
        for iR, R in enumerate(self.Rlist):
            self._HR[iR] = rotate_Matrix_from_z_to_axis(self._HR_copy[iR], axis)

        # print("After rotation:")
        # print(self._HR[0][:2, :2])


class RotateHam:
    def __init__(self, model, kmesh, gamma=True):
        self.model = model
        self.kpts = monkhorst_pack(kmesh, gamma_center=gamma)
        self.kweights = np.ones(len(self.kpts), dtype=float) / len(self.kpts)

    def get_band_energy(self, dm=False):
        evals, evecs = self.model.solve_all(self.kpts)
        eband = np.sum(
            evals
            * fermi(evals, self.model.efermi, width=0.01)
            * self.kweights[:, np.newaxis]
        )
        if dm:
            density_matrix = self.model.get_density_matrix(evecs)
        return eband

    def get_band_energy_vs_theta(
        self,
        angle_range=(0, np.pi * 2),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=21,
    ):
        thetas = np.linspace(*angle_range)
        es = []
        for theta in thetas:
            axis = Rotation.from_euler(rotation_axis, theta).apply(initial_direction)
            self.model.rotate_Hxc(axis)
            e = self.get_band_energy()
            es.append(e)
        return thetas, es


def get_model_energy(model, kmesh, gamma=True):
    ham = RotateHam(model, kmesh, gamma=gamma)
    return ham.get_band_energy()


class AbacusSplitSOCParser:
    """
    Abacus parser with Hamiltonian split to SOC and non-SOC parts
    """

    def __init__(self, outpath_nosoc=None, outpath_soc=None, binary=False):
        self.outpath_nosoc = outpath_nosoc
        self.outpath_soc = outpath_soc
        self.binary = binary
        self.parser_nosoc = AbacusParser(outpath=outpath_nosoc, binary=binary)
        self.parser_soc = AbacusParser(outpath=outpath_soc, binary=binary)
        spin1 = self.parser_nosoc.read_spin()
        spin2 = self.parser_soc.read_spin()
        if spin1 != "noncollinear" or spin2 != "noncollinear":
            raise ValueError("Spin should be noncollinear")

    def parse(self):
        nbasis, Rlist, HR, SR = self.parser_nosoc.Read_HSR_noncollinear()
        nbasis2, Rlist2, HR2, SR2 = self.parser_soc.Read_HSR_noncollinear()
        # print(HR[0])
        HR_soc = HR2 - HR
        model = AbacusSplitSOCWrapper(HR, SR, Rlist, nbasis, nspin=2, HR_soc=HR_soc)
        model.efermi = self.parser_nosoc.efermi
        model.basis = self.parser_nosoc.basis
        model.atoms = self.parser_nosoc.atoms
        return model


def test_AbacusSplitSOCWrapper():
    path = "/Users/hexu/projects/2D_Fe/2D_Fe"
    outpath_nosoc = f"{path}/Fe_SOC0/OUT.ABACUS"
    outpath_soc = f"{path}/Fe_SOC1/OUT.ABACUS"
    parser = AbacusSplitSOCParser(
        outpath_nosoc=outpath_nosoc, outpath_soc=outpath_soc, binary=False
    )
    model = parser.parse()
    kmesh = [4, 4, 1]
    e_z = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_z)

    model.rotate_Hxc([0, 0, 1])
    e_z = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_z)

    model.rotate_Hxc([1, 0, 0])
    e_x = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_x)

    r = RotateHam(model, kmesh)
    # thetas, es = r.get_band_energy_vs_theta(angle_range=(0, np.pi*2), rotation_axis="z", initial_direction=(1,0,0),  npoints=21)
    thetas, es = r.get_band_energy_vs_theta(
        angle_range=(0, np.pi * 2),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=21,
    )

    plt.plot(thetas / np.pi, es, marker="o")
    plt.savefig("E_along_z_x_z.png")
    plt.show()


if __name__ == "__main__":
    test_AbacusSplitSOCWrapper()
