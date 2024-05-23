import numpy as np
from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import rotate_Matrix_from_z_to_axis
from TB2J.kpoints import monkhorst_pack
from TB2J.mathutils.fermi import fermi
from TB2J.mathutils.kR_convert import k_to_R, R_to_k
from copy import deepcopy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from pathlib import Path

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

    def rotate_HR_xc(self, axis):
        """
        Rotate SOC part of Hamiltonian
        """
        for iR, R in enumerate(self.Rlist):
            self._HR[iR] = rotate_Matrix_from_z_to_axis(self._HR_copy[iR], axis)

    def rotate_Hk_xc(self, axis):
        """
        Rotate SOC part of Hamiltonian
        """
        for ik in range(len(self._Hk)):
            self._Hk[ik] = rotate_Matrix_from_z_to_axis(self._Hk_copy[ik], axis)

    def get_density_matrix(self, kpts, kweights):
        evals, evecs = self.solve_all(kpts)
        nkpt = len(kpts)
        rho = np.einsum(
            "kb,kib,kjb->kij",
            fermi(evals, self.efermi, width=0.05),
            evecs,
            evecs.conj(),
        )
        rho = np.zeros((nkpt, self.nbasis, self.nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            rho[ik] = (
                evecs[ik]
                * fermi(evals[ik], self.efermi, width=0.05)
                @ evecs[ik].T.conj()
                * kweights[ik]
            )
        print(np.trace(np.sum(rho, axis=0)))
        return rho

    def rotate_DM(self, rho, axis):
        """
        Rotate the density matrix
        """
        for ik in range(len(rho)):
            rho[ik] = rotate_Matrix_from_z_to_axis(rho[ik], axis)
        return rho


class RotateHam:
    def __init__(self, model, kmesh, gamma=True):
        self.model = model
        self.kpts = monkhorst_pack(kmesh, gamma_center=gamma)
        self.kweights = np.ones(len(self.kpts), dtype=float) / len(self.kpts)

    def get_band_energy(self, dm=False):
        evals, evecs = self.model.solve_all(self.kpts)
        eband = np.sum(
            evals
            * fermi(evals, self.model.efermi, width=0.05)
            * self.kweights[:, np.newaxis]
        )
        if dm:
            density_matrix = self.model.get_density_matrix(evecs)
            return eband, density_matrix
        else:
            return eband

    def calc_ref(self):
        # calculate the Hk_ref, Sk_ref, Hk_soc_ref, and rho_ref
        self.rho_ref = self.model.get_density_matrix(self.kpts, self.kweights)
        self.Hk_xc_ref = R_to_k(self.kpts, self.model.Rlist, self.model.HR)
        self.Hk_soc_ref = R_to_k(self.kpts, self.model.Rlist, self.model.HR_soc)

    def get_band_energy_from_rho(self, axis):
        eband = 0.0
        for ik, k in enumerate(self.kpts):
            rho = rotate_Matrix_from_z_to_axis(self.rho_ref[ik], axis)
            Hk_xc = rotate_Matrix_from_z_to_axis(self.Hk_xc_ref[ik], axis)
            Hk_soc = self.Hk_soc_ref[ik]
            eband += np.trace(rho @ (Hk_xc + Hk_soc))
        return eband

    def get_band_energy_vs_theta(
        self,
        angle_range=(0, np.pi * 2),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=21,
    ):
        es = []
        es2 = []
        # e,rho = self.model.get_band_energy(dm=True)
        self.calc_ref()
        thetas = np.linspace(*angle_range)
        for theta in thetas:
            axis = Rotation.from_euler(rotation_axis, theta).apply(initial_direction)
            self.model.rotate_HR_xc(axis)
            e = self.get_band_energy()
            e2 = self.get_band_energy_from_rho(axis)
            es.append(e)
            es2.append(e2)
            print(f"{e=}, {e2=}")
        return thetas, es, es2


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
        model.efermi = self.parser_soc.efermi
        model.basis = self.parser_nosoc.basis
        model.atoms = self.parser_nosoc.atoms
        return model


def test_AbacusSplitSOCWrapper():
    # path = Path("~/projects/2D_Fe").expanduser()
    path = Path("/home/hexu/projects/TB2Jflows/examples/2D_Fe")
    outpath_nosoc = f"{path}/Fe_soc0/OUT.ABACUS"
    outpath_soc = f"{path}/Fe_soc1_nscf/OUT.ABACUS"
    parser = AbacusSplitSOCParser(
        outpath_nosoc=outpath_nosoc, outpath_soc=outpath_soc, binary=False
    )
    model = parser.parse()
    kmesh = [4, 4, 1]
    e_z = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_z)

    model.rotate_HR_xc([0, 0, 1])
    e_z = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_z)

    model.rotate_HR_xc([1, 0, 0])
    e_x = get_model_energy(model, kmesh=kmesh, gamma=True)
    print(e_x)

    r = RotateHam(model, kmesh)
    # thetas, es = r.get_band_energy_vs_theta(angle_range=(0, np.pi*2), rotation_axis="z", initial_direction=(1,0,0),  npoints=21)
    thetas, es, es2 = r.get_band_energy_vs_theta(
        angle_range=(0, np.pi * 2),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=21,
    )

    # plt.plot(thetas / np.pi, es, marker="o")
    plt.plot(thetas / np.pi, es2, marker=".")
    plt.savefig("E_along_z_x_z.png")
    plt.show()


if __name__ == "__main__":
    test_AbacusSplitSOCWrapper()
