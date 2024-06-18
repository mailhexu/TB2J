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
from TB2J.mathutils.rotate_spin import spherical_to_cartesian
from HamiltonIO.model.occupations import Occupations

# from TB2J.abacus.abacus_wrapper import AbacusSplitSOCParser
from HamiltonIO.abacus.abacus_wrapper import AbacusSplitSOCParser
from HamiltonIO.siesta import SislParser, SiestaHamiltonian
import tqdm


def get_occupation(evals, kweights, nel, width=0.1):
    occ = Occupations(nel=nel, width=width, wk=kweights, nspin=2)
    return occ.occupy(evals)


def get_density_matrix(evals=None, evecs=None, kweights=None, nel=None, width=0.1):
    occ = get_occupation(evals, kweights, nel, width=width)
    rho = np.einsum("kib, kb, kjb -> kij", evecs, occ, evecs.conj())
    return rho


class MAE:
    def __init__(self, model, kmesh, gamma=True, width=0.1, nel=None):
        self.model = model
        if nel is not None:
            self.model.nel = nel
        self.kpts = monkhorst_pack(kmesh, gamma_center=gamma)
        self.kweights = np.ones(len(self.kpts), dtype=float) / len(self.kpts)
        self.width = width

    def get_band_energy(self):
        evals, evecs = self.model.solve_all(self.kpts)
        occ = get_occupation(evals, self.kweights, self.model.nel, width=self.width)
        eband = np.sum(evals * occ * self.kweights[:, np.newaxis])
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

    def get_band_energy_vs_angles(
        self,
        thetas,
        phis,
    ):
        es = []
        # es2 = []
        # e,rho = self.model.get_band_energy(dm=True)
        # self.calc_ref()
        # thetas = np.linspace(*angle_range, npoints)
        nangles = len(thetas)
        for i in tqdm.trange(nangles):
            theta = thetas[i]
            phi = phis[i]
            self.model.set_Hsoc_rotation_angle([theta, phi])
            e = self.get_band_energy()
            es.append(e)
            # es2.append(e2)
        return es


def get_model_energy(model, kmesh, gamma=True):
    ham = MAE(model, kmesh, gamma=gamma)
    return ham.get_band_energy()


def abacus_get_MAE(
    path_nosoc,
    path_soc,
    kmesh,
    thetas,
    psis,
    gamma=True,
    outfile="MAE.txt",
    nel=None,
    width=0.1,
):
    """Get MAE from Abacus with magnetic force theorem. Two calculations are needed. First we do an calculation with SOC but the soc_lambda is set to 0. Save the density. The next calculatin we start with the density from the first calculation and set the SOC prefactor to 1. With the information from the two calcualtions, we can get the band energy with magnetic moments in the direction, specified in two list, thetas, and phis."""
    parser = AbacusSplitSOCParser(
        outpath_nosoc=path_nosoc, outpath_soc=path_soc, binary=False
    )
    model = parser.parse()
    if nel is not None:
        model.nel = nel
    ham = MAE(model, kmesh, gamma=gamma, width=width)
    es = ham.get_band_energy_vs_angles(thetas, psis)
    if outfile:
        with open(outfile, "w") as f:
            f.write("#theta, psi, energy\n")
            for theta, psi, e in zip(thetas, psis, es):
                f.write(f"{theta:5.3f}, {psi:5.3f}, {e:10.9f}\n")
    return es


def siesta_get_MAE(
    fdf_fname, kmesh, thetas, phis, gamma=True, outfile="MAE.txt", nel=None, width=0.1
):
    """ """
    model = SislParser(fdf_fname=fdf_fname, read_H_soc=True).get_model()
    if nel is not None:
        model.nel = nel
    ham = MAE(model, kmesh, gamma=gamma, width=width)
    es = ham.get_band_energy_vs_angles(thetas, phis)
    if outfile:
        with open(outfile, "w") as f:
            f.write("#theta, psi, energy\n")
            for theta, psi, e in zip(thetas, phis, es):
                # f.write(f"{theta}, {psi}, {e}\n")
                f.write(f"{theta:5.3f}, {psi:5.3f}, {e:10.9f}\n")
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

    r = MAE(model, kmesh, gamma=True)
    # thetas, es = r.get_band_energy_vs_theta(angle_range=(0, np.pi*2), rotation_axis="z", initial_direction=(1,0,0),  npoints=21)
    thetas, es, es2 = r.get_band_energy_vs_theta(
        angle_range=(0, np.pi),
        rotation_axis="y",
        initial_direction=(0, 0, 1),
        npoints=11,
    )
    # print the table of thetas and es, es2
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
