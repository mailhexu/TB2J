#!/usr/bin/env python3
"""CLI for TB2J VASP native PAW projector exchange calculations.

Reads a VASP native binary (tb2j_native.bin, v4–v6) and computes
exchange coupling constants.  Version 6 streams are symmetry-reduced
IBZ exports that TB2J expands to the full BZ before exchange.
"""

from __future__ import annotations

import argparse

from TB2J.interfaces.vasp_native import read_vasp_native
from TB2J.versioninfo import print_license


def run_vasp_native2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J exchange from a VASP native PAW export "
            "(tb2j_native.bin).  Supports v4/v5 full-BZ and v6 "
            "symmetry-reduced IBZ streams."
        ),
        epilog=(
            "Typical workflow: run VASP with the tb2j_native export "
            "patch, then pass tb2j_native.bin to this command."
        ),
    )
    parser.add_argument(
        "--input", required=True, help="VASP native binary (tb2j_native.bin)"
    )
    parser.add_argument(
        "--output_path", default="TB2J_results_vasp", help="output directory"
    )
    parser.add_argument(
        "--Rcut",
        type=float,
        default=None,
        help="spin-pair distance cutoff in Angstrom",
    )
    parser.add_argument(
        "--nz", type=int, default=80, help="number of continued-fraction poles"
    )
    parser.add_argument(
        "--smearing",
        type=float,
        default=0.05,
        help="CFR smearing in eV",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=None,
        help="magnetic elements, e.g. Fe or Mn",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        type=int,
        nargs="*",
        default=None,
        help="1-based magnetic atom indices",
    )
    args = parser.parse_args()

    from TB2J.interfaces.gpaw_projector import (
        _R_grid_for_cutoff,
        write_projector_exchange_out,
    )
    from TB2J.mycfr import CFR2
    from TB2J.paw_projector import build_projector_green_data
    from TB2J.projector_green import (
        ProjectorGreen,
        projector_charge_moments_from_green,
    )
    from ase.units import kB

    snapshot = read_vasp_native(args.input)
    data = build_projector_green_data(snapshot)

    # Resolve magnetic atoms
    magnetic_atoms = None
    if args.index_magnetic_atoms is not None:
        magnetic_atoms = [i - 1 for i in args.index_magnetic_atoms]
    elif args.elements is not None:
        symbols = {s.strip().capitalize() for s in args.elements}
        magnetic_atoms = [
            i
            for i, site in enumerate(snapshot.site_layout)
            if site.species.capitalize() in symbols
        ]
    if not magnetic_atoms:
        magnetic_atoms = list(range(len(snapshot.site_layout)))

    rcut = args.Rcut if args.Rcut is not None else 10.0
    Rpts = _R_grid_for_cutoff(data, magnetic_atoms, rcut, None)

    contour = CFR2(nz=args.nz, T=args.smearing / kB)
    population = projector_charge_moments_from_green(
        ProjectorGreen(data), contour
    )

    exchange_out, _ = write_projector_exchange_out(
        data,
        path=args.output_path,
        Rpts=Rpts,
        nz=args.nz,
        smearing_eV=args.smearing,
        index_magnetic_atoms=magnetic_atoms,
        Rcut=rcut,
        charges=population["charges"],
        spinat=population["spinat"],
        description=(
            "VASP-native W%CPROJ/CDIJ projector exchange trace. "
            "Charges and moments are QTOT-contracted PAW partial-wave "
            "populations."
        ),
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_vasp_native2J()
