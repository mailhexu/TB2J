import copy
from pathlib import Path

import numpy as np
from sympair import SymmetryPairFinder

from TB2J.io_exchange import SpinIO
from TB2J.versioninfo import print_license


class TB2JSymmetrizer:
    def __init__(self, exc, symprec=1e-8, verbose=True, Jonly=False):
        # list of pairs with the index of atoms
        ijRs = exc.ijR_list_index_atom()
        finder = SymmetryPairFinder(atoms=exc.atoms, pairs=ijRs, symprec=symprec)
        self.verbose = verbose
        self.Jonly = Jonly

        if verbose:
            print("=" * 30)
            print_license()
            print("-" * 30)
            print(
                "WARNING: The symmetry detection is based on the crystal symmetry, not the magnetic symmetry. Make sure if this is what you want."
            )
            print("-" * 30)
            if exc.has_dmi:
                print(
                    "WARNING: Currently only the isotropic exchange is symmetrized. Symmetrization of DMI and anisotropic exchange are not yet implemented."
                )

            print(f"Finding crystal symmetry with symprec of {symprec} Angstrom.")
            print("Symmetry found:")
            print(finder.spacegroup)
            print("-" * 30)
        self.pgdict = finder.get_symmetry_pair_group_dict()
        self.exc = exc
        self.new_exc = copy.deepcopy(exc)
        self.Jonly = Jonly

    def print_license(self):
        print_license()

    def symmetrize_J(self):
        """
        Symmetrize the exchange parameters J.
        """
        symJdict = {}
        # Jdict = self.exc.exchange_Jdict
        # ngroup = self.pgdict
        for pairgroup in self.pgdict.groups:
            ijRs = pairgroup.get_all_ijR()
            ijRs_spin = [self.exc.ijR_index_atom_to_spin(*ijR) for ijR in ijRs]
            Js = [self.exc.get_J(*ijR_spin) for ijR_spin in ijRs_spin]
            Javg = np.average(Js)
            for i, j, R in ijRs_spin:
                symJdict[(R, i, j)] = Javg
        self.new_exc.exchange_Jdict = symJdict
        if self.Jonly:
            self.new_exc.has_dmi = False
            self.new_exc.dmi_ddict = None
            self.new_exc.has_bilinear = False
            self.new_exc.Jani_dict = None
            self.has_uniaxial_anisotropy = False
            self.k1 = None
            self.k1dir = None

    def output(self, path="TB2J_symmetrized"):
        if path is None:
            path = Path(".")
        self.new_exc.write_all(path=path)

    def run(self, path=None):
        print("** Symmetrizing exchange parameters.")
        self.symmetrize_J()
        print("** Outputing the symmetrized exchange parameters.")
        print(f"** Output path: {path} .")
        self.output(path=path)
        print("** Finished.")


def symmetrize_J(
    exc=None,
    path=None,
    fname="TB2J.pickle",
    symprec=1e-5,
    output_path="TB2J_symmetrized",
    Jonly=False,
):
    """
    symmetrize the exchange parameters
    parameters:
    exc: exchange
    """
    if exc is None:
        exc = SpinIO.load_pickle(path=path, fname=fname)
    symmetrizer = TB2JSymmetrizer(exc, symprec=symprec, Jonly=Jonly)
    symmetrizer.run(path=output_path)


def symmetrize_J_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Symmetrize exchange parameters. Currently, it take the crystal symmetry into account and  not the magnetic moment into account."
    )
    parser.add_argument(
        "-i",
        "--inpath",
        default=None,
        help="input path to the exchange parameters",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        default="TB2J_results_symmetrized",
        help="output path to the symmetrized exchange parameters",
    )
    parser.add_argument(
        "-s",
        "--symprec",
        type=float,
        default=1e-5,
        help="precision for symmetry detection. default is 1e-5 Angstrom",
    )

    parser.add_argument(
        "--Jonly",
        action="store_true",
        help="symmetrize only the exchange parameters and discard the DMI and anisotropic exchange",
        default=False,
    )

    args = parser.parse_args()
    if args.inpath is None:
        parser.print_help()
        raise ValueError("Please provide the input path to the exchange.")
    symmetrize_J(
        path=args.inpath,
        output_path=args.outpath,
        symprec=args.symprec,
        Jonly=args.Jonly,
    )


if __name__ == "__main__":
    symmetrize_J_cli()
