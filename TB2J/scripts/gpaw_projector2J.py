#!/usr/bin/env python3
"""CLI for TB2J projector Green NetCDF exchange calculations."""

from __future__ import annotations

import argparse

from TB2J.interfaces.gpaw_projector import gen_exchange_projector_netcdf
from TB2J.versioninfo import print_license


def run_gpaw_projector2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J-style exchange parameters from a GPAW PAW projector "
            "Green NetCDF file."
        )
    )
    parser.add_argument("--input", required=True, help="projector Green NetCDF file")
    parser.add_argument(
        "--output_path", default="TB2J_results", help="output directory"
    )
    parser.add_argument(
        "--Rmax",
        type=int,
        default=1,
        help="maximum integer lattice-vector component for the R grid",
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
    args = parser.parse_args()
    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]
    exchange_out, _ = gen_exchange_projector_netcdf(
        args.input,
        output_path=args.output_path,
        Rmax=args.Rmax,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_gpaw_projector2J()
