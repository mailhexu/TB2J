#!/usr/bin/env python3
"""CLI for TB2J ABINIT savetb2j projector exchange calculations."""

from __future__ import annotations

import argparse

from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_projector
from TB2J.versioninfo import print_license


def run_abinit_projector2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J-style exchange parameters from an ABINIT savetb2j "
            "PAW projector NetCDF file. The default operator component is delta_total."
        ),
        epilog=(
            "Typical workflow: run ABINIT with savetb2j 1, then pass the generated "
            "*_SAVETB2J.nc file to this command. Version 1 files must come from "
            "collinear PAW calculations with nsppol=2, nspinor=1, and a full-BZ "
            "explicit k-point list (kptopt=0)."
        ),
    )
    parser.add_argument("--input", required=True, help="ABINIT savetb2j NetCDF file")
    parser.add_argument(
        "--output_path", default="TB2J_results_abinit", help="output directory"
    )
    parser.add_argument(
        "--Rmax",
        type=int,
        default=1,
        help="maximum integer lattice-vector component for the R grid",
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
        help="magnetic elements to include, for example Fe or Mn",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        type=int,
        nargs="*",
        default=None,
        help="1-based magnetic atom indices to include",
    )
    parser.add_argument(
        "--operator_component",
        default="delta_total",
        help="ABINIT operator component to use for exchange; default: delta_total",
    )
    parser.add_argument(
        "--population_mode",
        choices=("none", "projector"),
        default="none",
        help=(
            "source for exchange.out atom charge/moment fields; ABINIT savetb2j "
            "v1 supports only 'none' because PAW-complete populations are not exported"
        ),
    )
    parser.add_argument(
        "--shell_charge_threshold",
        type=float,
        default=None,
        help="exclude local-operator PAW projector shells with projected charge below this value",
    )
    parser.add_argument(
        "--shell_moment_threshold",
        type=float,
        default=None,
        help="exclude local-operator PAW projector shells with projected moment norm below this value",
    )
    args = parser.parse_args()
    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]
    exchange_out, _ = gen_exchange_abinit_projector(
        args.input,
        output_path=args.output_path,
        Rmax=args.Rmax,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
        operator_component=args.operator_component,
        population_mode=args.population_mode,
        shell_charge_threshold=args.shell_charge_threshold,
        shell_moment_threshold=args.shell_moment_threshold,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_abinit_projector2J()
