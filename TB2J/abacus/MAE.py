import numpy as np
from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import rotate_Matrix_from_z_to_axis
from TB2J.kpoints import monkhorst_pack
from TB2J.mathutils.fermi import fermi
from TB2J.mathutils.kR_convert import R_to_k
from scipy.linalg import eigh
from copy import deepcopy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from pathlib import Path
from TB2J.abacus.occupations import Occupations

# TODO List:
# - [x] Add the class AbacusSplitSOCWrapper
# - [x] Add the function to rotate the XC part
# - [x] Compute the band energy at arbitrary angle


def get_occupation(evals, kweights, nel, width=0.1):
    occ = Occupations(nel=nel, width=width, wk=kweights, nspin=2)
    return occ.occupy(evals)


def get_density_matrix(evals=None, evecs=None, kweights=None, nel=None, width=0.1):
    occ = get_occupation(evals, kweights, nel, width=width)
    rho = np.einsum("kib, kb, kjb -> kij", evecs, occ, evecs.conj())
    return rho


def spherical_to_cartesian(theta, phi, normalize=True):
    """
    Convert spherical coordinates to cartesian
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x, y, z])
    if normalize:
        vec = vec / np.linalg.norm(vec)
    return vec


class AbacusSplitSOCWrapper(AbacusWrapper):
    """
    Abacus wrapper with Hamiltonian split to SOC and non-SOC parts
    """

    def __init__(self, *args, **kwargs):
        HR_soc = kwargs.pop("HR_soc", None)
        # nbasis = HR_soc.shape[1]
        # kwargs["nbasis"] = nbasis
        super().__init__(*args, **kwargs)
        self._HR_copy = deepcopy(self._HR)
        self.HR_soc = HR_soc
        self.soc_lambda = 1.0
        self.nel = 16
        self.width = 0.1

    @property
    def HR(self):
        return self._HR + self.HR_soc * self.soc_lambda

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

    def get_density_matrix(self, kpts, kweights=None):
        rho = np.zeros((len(kpts), self.nbasis, self.nbasis), dtype=complex)
        evals, evecs = self.solve_all(kpts)
        occ = get_occupation(evals, kweights, self.nel, width=self.width)
        rho = np.einsum(
            "kib, kb, kjb -> kij", evecs, occ, evecs.conj()
        )  # should multiply S to the the real DM.
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

    def get_band_energy2(self):
        for ik, kpt in enumerate(self.kpts):
            Hk, Sk = self.model.gen_ham(kpt)
            evals, evecs = eigh(Hk, Sk)
            rho = np.einsum(
                "ib, b, jb -> ij",
                evecs,
                fermi(evals, self.model.efermi, width=0.05),
                evecs.conj(),
            )
            eband1 = np.sum(evals * fermi(evals, self.model.efermi, width=0.05))
            eband2 = np.trace(Hk @ rho)
            print(eband1, eband2)

    def get_band_energy(self, dm=False):
        evals, evecs = self.model.solve_all(self.kpts)
        occ = get_occupation(
            evals, self.kweights, self.model.nel, width=self.model.width
        )
        eband = np.sum(evals * occ * self.kweights[:, np.newaxis])
        # * fermi(evals, self.model.efermi, width=0.05)
        if dm:
            density_matrix = self.model.get_density_matrix(evecs)
            return eband, density_matrix
        else:
            return eband

    def calc_ref(self):
        # calculate the Hk_ref, Sk_ref, Hk_soc_ref, and rho_ref
        self.Sk_ref = R_to_k(self.kpts, self.model.Rlist, self.model.SR)
        self.Hk_xc_ref = R_to_k(self.kpts, self.model.Rlist, self.model._HR_copy)
        self.Hk_soc_ref = R_to_k(self.kpts, self.model.Rlist, self.model.HR_soc)
        self.rho_ref = np.zeros(
            (len(self.kpts), self.model.nbasis, self.model.nbasis), dtype=complex
        )

        evals = np.zeros((len(self.kpts), self.model.nbasis), dtype=float)
        evecs = np.zeros(
            (len(self.kpts), self.model.nbasis, self.model.nbasis), dtype=complex
        )

        for ik, kpt in enumerate(self.kpts):
            # evals, evecs = eigh(self.Hk_xc_ref[ik]+self.Hk_soc_ref[ik], self.Sk_ref[ik])
            evals[ik], evecs[ik] = eigh(self.Hk_xc_ref[ik], self.Sk_ref[ik])
        occ = get_occupation(
            evals, self.kweights, self.model.nel, width=self.model.width
        )
        # occ = fermi(evals, self.model.efermi, width=self.model.width)
        self.rho_ref = np.einsum("kib, kb, kjb -> kij", evecs, occ, evecs.conj())

    def get_band_energy_from_rho(self, axis):
        """
        This is wrong!! Should use second order perturbation theory to get the band energy instead.
        """
        eband = 0.0
        for ik, k in enumerate(self.kpts):
            rho = rotate_Matrix_from_z_to_axis(self.rho_ref[ik], axis)
            Hk_xc = rotate_Matrix_from_z_to_axis(self.Hk_xc_ref[ik], axis)
            Hk_soc = self.Hk_soc_ref[ik]
            Htot = Hk_xc + Hk_soc * self.model.soc_lambda
            # Sk = self.Sk_ref[ik]
            # evals, evecs = eigh(Htot, Sk)
            # rho2= np.einsum("ib, b, jb -> ij", evecs, fermi(evals, self.model.efermi, width=0.05), evecs.conj())
            if ik == 0 and False:
                pass
                # print(f"{evecs[:4,0:4].real=}")
                # print(f"{evals[:4]=}")
                # print(f"{Hk_xc[:4,0:4].real=}")
                # print(f"{Htot[:4,0:4].real=}")
                # print(f"{Sk[:4,0:4].real=}")
                # print(f"{rho[:4,0:4].real=}")
                # print(f"{rho2[:4,0:4].real=}")
            # eband1 = np.sum(evals * fermi(evals, self.model.efermi, width=0.05))
            # eband2 = np.trace(Htot @ rho2).real
            # eband3 = np.trace(Htot @ rho).real
            # print(eband1, eband2, eband3)
            e_soc = np.trace(Hk_soc @ rho) * self.kweights[ik] * self.model.soc_lambda
            eband += e_soc
        return eband

    def get_band_energy_vs_angles(
        self,
        thetas,
        psis,
    ):
        es = []
        # es2 = []
        # e,rho = self.model.get_band_energy(dm=True)
        # self.calc_ref()
        # thetas = np.linspace(*angle_range, npoints)
        for i, theta, phi in enumerate(zip(thetas, psis)):
            axis = spherical_to_cartesian(theta, phi)
            self.model.rotate_HR_xc(axis)
            # self.get_band_energy2()
            e = self.get_band_energy()
            es.append(e)
            # es2.append(e2)
        return es


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


def abacus_get_MAE(
    path_nosoc, path_soc, kmesh, thetas, psis, gamma=True, outfile="MAE.txt"
):
    """Get MAE from Abacus with magnetic force theorem. Two calculations are needed. First we do an calculation with SOC but the soc_lambda is set to 0. Save the density. The next calculatin we start with the density from the first calculation and set the SOC prefactor to 1. With the information from the two calcualtions, we can get the band energy with magnetic moments in the direction, specified in two list, thetas, and phis."""
    parser = AbacusSplitSOCParser(
        outpath_nosoc=path_nosoc, outpath_soc=path_soc, binary=False
    )
    model = parser.parse()
    ham = RotateHam(model, kmesh, gamma=gamma)
    es = []
    for theta, psi in zip(thetas, psis):
        axis = spherical_to_cartesian(theta, psi)
        model.rotate_HR_xc(axis)
        e = ham.get_band_energy()
        es.append(ham.get_band_energy())
    if outfile:
        with open(outfile, "w") as f:
            f.write("theta, psi, energy\n")
            for theta, psi, e in zip(thetas, psis, es):
                f.write(f"{theta}, {psi}, {e}\n")
    return es


def test_AbacusSplitSOCWrapper():
    # path = Path("~/projects/2D_Fe").expanduser()
    path = Path("~/projects/TB2Jflows/examples/2D_Fe/Fe_z").expanduser()
    outpath_nosoc = f"{path}/soc0/OUT.ABACUS"
    outpath_soc = f"{path}/soc1/OUT.ABACUS"
    parser = AbacusSplitSOCParser(
        outpath_nosoc=outpath_nosoc, outpath_soc=outpath_soc, binary=False
    )
    model = parser.parse()
    kmesh = [6, 6, 1]

    r = RotateHam(model, kmesh)
    # thetas, es = r.get_band_energy_vs_theta(angle_range=(0, np.pi*2), rotation_axis="z", initial_direction=(1,0,0),  npoints=21)
    thetas, es, es2 = r.get_band_energy_vs_theta(
        angle_range=(0, np.pi * 2),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=11,
    )
    # print the table of thetas and es, es2
    print("theta, e, e2")
    for theta, e, e2 in zip(thetas, es, es2):
        print(f"{theta=}, {e=}, {e2=}")

    plt.plot(thetas / np.pi, es - es[0], marker="o")
    plt.plot(thetas / np.pi, es2 - es2[0], marker=".")
    plt.savefig("E_along_z_x_z.png")
    plt.show()


def abacus_get_MAE_cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get MAE from Abacus with magnetic force theorem. Two calculations are needed. First we do an calculation with SOC but the soc_lambda is set to 0. Save the density. The next calculatin we start with the density from the first calculation and set the SOC prefactor to 1. With the information from the two calcualtions, we can get the band energy with magnetic moments in the direction, specified in two list, thetas, and phis. "
    )
    parser.add_argument("path_nosoc", type=str, help="Path to the  calculation with ")
    parser.add_argument("path_soc", type=str, help="Path to the SOC calculation")
    parser.add_argument("thetas", type=float, nargs="+", help="Thetas")
    parser.add_argument("psis", type=float, nargs="+", help="Phis")
    parser.add_argument("kmesh", type=int, nargs=3, help="K-mesh")
    parser.add_argument(
        "--gamma", action="store_true", help="Use Gamma centered kpoints"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="The angles and the energey will be saved in this file.",
    )
    args = parser.parse_args()
    abacus_get_MAE(
        args.path_nosoc,
        args.path_soc,
        args.kmesh,
        args.thetas,
        args.psis,
        gamma=args.gamma,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    abacus_get_MAE_cli()
