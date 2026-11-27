#!/usr/bin/env python3
import argparse

from TB2J.Jdownfolder import JDownfolder_pickle
from TB2J.versioninfo import print_license


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
        help="Downfolding method: Lowdin, PWF (projected Wannier function), or Mryasov (isotropic).",
        type=str.lower,
        choices=("lowdin", "pwf", "mryasov"),
        default="lowdin",
    )
    parser.add_argument(
        "--dressing",
        type=float,
        help="Mryasov energy enhancement factor lambda (default: 1, bare model).",
    )

    args = parser.parse_args()

    if not args.metals:
        parser.error("List of magnetic metal elements cannot be empty.")
    if not args.ligands:
        parser.error("List of ligand elements cannot be empty.")
    if args.dressing is not None and args.method != "mryasov":
        parser.error("--dressing is only supported with --method mryasov.")
    if args.method == "mryasov":
        args.iso_only = True
    model_options = {} if args.dressing is None else {"dressing": args.dressing}

    print_license()
    print("Input path:", args.inpath)
    print("Output path:", args.outpath)
    print("Magnetic cation elements:", args.metals)
    print("Ligand elements:", args.ligands)
    print("k-mesh:", args.qmesh)
    print("Downfolding method:", args.method)
    print("Downfolding only isotropic part:", args.iso_only)
    print("Begining downfolding the exchange parameters:")
    JDownfolder_pickle(
        inpath=args.inpath,
        metals=args.metals,
        ligands=args.ligands,
        outpath=args.outpath,
        qmesh=args.qmesh,
        iso_only=args.iso_only,
        method=args.method,
        **model_options,
    )
    print("Downfolding finished. Results are saved in:", args.outpath)


if __name__ == "__main__":
    main()
