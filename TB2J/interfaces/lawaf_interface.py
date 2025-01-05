import argparse
import os
import sys

from HamiltonIO.lawaf import LawafHamiltonian as Ham

from TB2J.exchange_params import add_exchange_args_to_parser, parser_argument_to_dict
from TB2J.versioninfo import print_license

from .manager import Manager


class LaWaFManager(Manager):
    def __init__(
        self,
        path,
        prefix_up,
        prefix_dn,
        prefix_SOC,
        colinear=True,
        wannier_type="LaWaF",
        **kwargs,
    ):
        # models and basis
        if colinear:
            tbmodels, basis, atoms = self.prepare_model_colinear(
                path, prefix_up, prefix_dn
            )
        else:
            tbmodels, basis, atoms = self.prepare_model_ncl(path, prefix_SOC)

        description = self.description(path, prefix_up, prefix_dn, prefix_SOC, colinear)
        kwargs["description"] = description

        super().__init__(atoms, tbmodels, basis, colinear=colinear, **kwargs)

    def prepare_model_colinear(self, path, prefix_up, prefix_dn):
        print("Reading LawaF hamiltonian: spin up.")
        tbmodel_up = Ham.load_pickle(os.path.join(path, f"{prefix_up}.pickle"))
        print("Reading LaWaF hamiltonian: spin down.")
        tbmodel_dn = Ham.load_pickle(os.path.join(path, f"{prefix_dn}.pickle"))
        tbmodels = (tbmodel_up, tbmodel_dn)
        basis = tbmodel_up.wann_names
        atoms = tbmodel_up.atoms
        return tbmodels, basis, atoms

    def prepare_model_ncl(self, path, prefix_SOC):
        print("Reading LaWaF hamiltonian: non-colinear spin.")
        tbmodel = Ham.load_pickle(os.path.join(path, f"{prefix_SOC}.pickle"))
        basis = tbmodel.wann_names
        atoms = tbmodel.atoms
        return tbmodel, basis, atoms

    def description(self, path, prefix_up, prefix_dn, prefix_SOC, colinear=True):
        if colinear:
            description = f""" Input from collinear LaWaF data.
 Tight binding data from {path}.
 Prefix of wannier function files:{prefix_up} and {prefix_dn}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
\n"""

        else:
            description = f""" Input from non-collinear LaWaF data.
Tight binding data from {path}.
Prefix of wannier function files:{prefix_SOC}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
The DMI component parallel to the spin orientation, the Jani which has the component of that orientation should be disregarded
e.g. if the spins are along z, the xz, yz, zz, zx, zy components and the z component of DMI.
If you need these component, try to do three calculations with spin along x, y, z,  or use structure with z rotated to x, y and z. And then use TB2J_merge.py to get the full set of parameters.\n"""

        return description


def lawaf2J_cli():
    print_license()
    parser = argparse.ArgumentParser(
        description="lawaf2J: Using magnetic force theorem to calculate exchange parameter J from wannier functions in LaWaF"
    )
    parser.add_argument(
        "--path", help="path to the wannier files", default="./", type=str
    )

    parser.add_argument(
        "--spinor",
        help="if the calculation is spinor, default is False",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--prefix_spinor",
        help="prefix to the spinor wannier files",
        default="lawaf_spinor.pickle",
        type=str,
    )
    parser.add_argument(
        "--prefix_up",
        help="prefix to the spin up wannier files",
        default="lawaf_up",
        type=str,
    )
    parser.add_argument(
        "--prefix_down",
        help="prefix to the spin down wannier files",
        default="lawaf_dn",
        type=str,
    )
    add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    if args.efermi is None:
        print("Please input fermi energy using --efermi ")
        sys.exit()
    if args.elements is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        sys.exit()

    kwargs = parser_argument_to_dict(args)

    manager = LaWaFManager(
        path=args.path,
        prefix_up=args.prefix_up,
        prefix_dn=args.prefix_down,
        prefix_SOC=args.prefix_spinor,
        colinear=not args.spinor,
        **kwargs,
    )
    return manager
