#!/usr/bin/env python3
import argparse
import sys

from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces import gen_exchange
from TB2J.versioninfo import print_license


def run_wann2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="wann2J: Using magnetic force theorem to calculate exchange parameter J from wannier functions"
    )
    parser.add_argument(
        "--path", help="path to the wannier files", default="./", type=str
    )
    parser.add_argument(
        "--posfile", help="name of the position file", default="POSCAR", type=str
    )
    parser.add_argument(
        "--prefix_spinor",
        help="prefix to the spinor wannier files",
        default="wannier90",
        type=str,
    )
    parser.add_argument(
        "--prefix_up",
        help="prefix to the spin up wannier files",
        default="wannier90.up",
        type=str,
    )
    parser.add_argument(
        "--prefix_down",
        help="prefix to the spin down wannier files",
        default="wannier90.dn",
        type=str,
    )
    parser.add_argument(
        "--groupby",
        help="In the spinor case, the order of the orbitals have two conventions: 1: group by spin (orb1_up, orb2_up,... orb1_down, ...), 2,group by orbital (orb1_up, orb1_down, orb2_up, ...,). Use 'spin' in the former case and 'orbital' in the latter case. The default is spin.",
        default="spin",
        type=str,
    )
    parser.add_argument(
        "--wannier_type",
        help="The type of Wannier function, either Wannier90 or banddownfolder",
        type=str,
        default="Wannier90",
    )

    # parser.add_argument("--qspace",
    #                    action="store_true",
    #                    help="Whether to calculate J in qspace first and transform to real space.",
    #                    default=False)

    add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    if args.efermi is None:
        print("Please input fermi energy using --efermi ")
        sys.exit()
    if args.elements is None and args.index_magnetic_atoms is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        sys.exit()

    index_magnetic_atoms = args.index_magnetic_atoms
    if index_magnetic_atoms is not None:
        index_magnetic_atoms = [i - 1 for i in index_magnetic_atoms]

    gen_exchange(
        path=args.path,
        colinear=(not args.spinor),
        groupby=args.groupby,
        posfile=args.posfile,
        efermi=args.efermi,
        kmesh=args.kmesh,
        magnetic_elements=args.elements,
        Rcut=args.rcut,
        prefix_SOC=args.prefix_spinor,
        prefix_up=args.prefix_up,
        prefix_dn=args.prefix_down,
        emin=args.emin,
        emax=args.emax,
        nz=args.nz,
        use_cache=args.use_cache,
        nproc=args.np,
        description=args.description,
        output_path=args.output_path,
        exclude_orbs=args.exclude_orbs,
        wannier_type=args.wannier_type,
        # qspace=args.qspace,
        write_density_matrix=args.write_dm,
        orb_decomposition=args.orb_decomposition,
        index_magnetic_atoms=index_magnetic_atoms,
    )


if __name__ == "__main__":
    run_wann2J()
