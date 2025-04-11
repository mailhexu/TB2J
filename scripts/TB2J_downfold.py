#!/usr/bin/env python3
import argparse
import os
import sys
from TB2J.Jdownfolder import JDownfolder_pickle


def main():
    parser = argparse.ArgumentParser(
        description="TB2J_downfold: Downfold the exchange parameter with ligand spin as independent variable to that only with metal site spin as independent variables."
    )
    parser.add_argument(
        "--inpath",
        "-i",
        type=str,
        help="The path of the input TB2J results before the downfolding.",
        default="TB2J_results",
    )

    parser.add_argument(
        "--outpath",
        "-o",
        type=str,
        help="The path of the output TB2J results path after the downfolding",
        default="TB2J_results_downfolded",
    )

    parser.add_argument(
        "--metals",
        "-m",
        type=str,
        help="List of magnetic cation elements",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--ligands",
        "-l",
        type=str,
        help="List of ligand elements",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--qmesh",
        help="kmesh in the format of kx ky kz",
        type=int,
        nargs="*",
        default=[5, 5, 5],
    )

    parser.add_argument(
        "--iso_only",
        help="whether to downfold only the isotropic part. The other parts will be neglected.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--method",
        help="The method to downfold the exchange parameter. Options are lowdin and PWF (projected Wannier function). ",
        type=str,
        default="lowdin",
    )

    args = parser.parse_args()

    if len(args.metals) == []:
        print("List of magnetic cation elements cannot be empty")

    if len(args.ligands) == []:
        print("List of ligand elements cannot be empty")

    JDownfolder_pickle(
        inpath=args.inpath,
        metals=args.metals,
        ligands=args.ligands,
        outpath=args.outpath,
        qmesh=args.qmesh,
        iso_only=args.iso_only,
        method=args.method,
    )


main()
