#!/usr/bin/env python3
"""CLI for TB2J VASP projector XML exchange calculations."""

from __future__ import annotations

import argparse

from TB2J.interfaces.vasp_projector_xml import gen_exchange_vasp_projector_xml
from TB2J.versioninfo import print_license


def run_vasp_projector2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J-style exchange parameters from a VASP projector "
            "Green XML file."
        )
    )
    parser.add_argument("--input", required=True, help="VASP projector Green XML file")
    parser.add_argument(
        "--output_path", default="TB2J_results_vasp_xml", help="output directory"
    )
    parser.add_argument(
        "--Rmax",
        type=int,
        default=None,
        help="optional maximum integer lattice-vector component for the R grid",
    )
    parser.add_argument(
        "--Rcut",
        type=float,
        default=None,
        help="optional spin-pair distance cutoff in Angstrom",
    )
    parser.add_argument(
        "--nz", type=int, default=30, help="number of continued-fraction poles"
    )
    parser.add_argument(
        "--smearing",
        type=float,
        default=0.05,
        help="CFR smearing in eV for the projector exchange trace",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=None,
        help="magnetic elements to include, for example Ni or Fe",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        type=int,
        nargs="*",
        default=None,
        help="1-based magnetic atom indices to include",
    )
    parser.add_argument(
        "--outcar",
        default=None,
        help="optional VASP OUTCAR with LORBIT site charges and moments",
    )
    parser.add_argument(
        "--population_source",
        choices=("green", "outcar", "none"),
        default="green",
        help="source for exchange.out atom charge/moment fields",
    )
    parser.add_argument(
        "--allow_symmetry_expanded",
        action="store_true",
        help="allow diagnostic output from VASP XML generated with ISYM>0",
    )
    parser.add_argument(
        "--allow_basis_mismatch",
        action="store_true",
        help=(
            "allow diagnostic output from VASP XML pairing LPRJ_COVL coefficients "
            "with native CDIJ operators"
        ),
    )
    args = parser.parse_args()
    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]
    exchange_out, _ = gen_exchange_vasp_projector_xml(
        args.input,
        output_path=args.output_path,
        Rmax=args.Rmax,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
        outcar_filename=args.outcar,
        population_source=args.population_source,
        allow_symmetry_expanded=args.allow_symmetry_expanded,
        allow_basis_mismatch=args.allow_basis_mismatch,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_vasp_projector2J()
