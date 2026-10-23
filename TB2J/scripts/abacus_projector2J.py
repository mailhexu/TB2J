#!/usr/bin/env python3
"""Exchange from canonical ABACUS PW UPF atomic-projector NetCDF data."""

from __future__ import annotations

import argparse

from TB2J.interfaces.gpaw_projector import gen_exchange_projector_netcdf
from TB2J.versioninfo import print_license


def run_abacus_projector2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate exchange from ABACUS plane-wave UPF atomic-projector "
            "Green NetCDF data using delta_total and the inverse overlap metric."
        )
    )
    parser.add_argument("--input", required=True, help="projector Green NetCDF file")
    parser.add_argument(
        "--output_path", default="TB2J_results", help="output directory"
    )
    parser.add_argument(
        "--Rcut", type=float, default=10.0, help="spin-pair distance cutoff in Angstrom"
    )
    parser.add_argument(
        "--nz", type=int, default=30, help="number of continued-fraction poles"
    )
    parser.add_argument(
        "--smearing", type=float, default=0.05, help="CFR smearing in eV"
    )
    parser.add_argument(
        "--elements", nargs="*", default=None, help="magnetic elements, for example Fe"
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
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
        operator_component="delta_total",
        overlap_mode="inverse",
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_abacus_projector2J()
