#!/usr/bin/env python3
"""CLI for the VASP KS-basis split-SOC exchange workflow (story 011).

Consumes the two artifacts of one collinear strength-0 VASP run patched
with the story-010 dump hooks — ``tb2j_native.bin`` (v5/v6 collinear
native export) and ``tb2j_cso.bin`` (one-center CSO operator + COCC) —
runs the three-direction (x, y, z) split-SOC legs through the shared
KS-band kernel, merges them, and persists per-leg provenance.
"""

from __future__ import annotations

import argparse

import numpy as np

from TB2J.interfaces.vasp_split_soc import LEGS, gen_exchange_vasp_split_soc
from TB2J.versioninfo import print_license


def run_vasp_split_soc2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J exchange from one collinear VASP strength-0 "
            "run patched with the story-010 CSO dump: three-direction "
            "(x, y, z) split-SOC legs from tb2j_native.bin + tb2j_cso.bin, "
            "rotated to the lattice frame and merged."
        ),
        epilog=(
            "Typical workflow: run patched collinear VASP (ISPIN=2, "
            "LSORBIT=.FALSE., story-010 dump hooks) once, then pass the "
            "two dumps to this command."
        ),
    )
    parser.add_argument(
        "--native-input",
        required=True,
        help="VASP collinear native export (tb2j_native.bin, v5/v6)",
    )
    parser.add_argument(
        "--cso-dump",
        required=True,
        help="story-010 CSO dump from the same run (tb2j_cso.bin)",
    )
    parser.add_argument(
        "--output_path",
        default="TB2J_results_vasp_split_soc",
        help="merged output directory (per-leg results in leg_x/leg_y/leg_z)",
    )
    parser.add_argument(
        "--Rcut",
        type=float,
        default=10.0,
        help="spin-pair distance cutoff in Angstrom",
    )
    parser.add_argument(
        "--nz", type=int, default=60, help="number of continued-fraction poles"
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
        help="magnetic elements, e.g. Fe (oxygen and other ligands stay in "
        "the all-atom W_SO)",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        type=int,
        nargs="*",
        default=None,
        help="1-based magnetic atom indices",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="SOC strength scaling (1.0 = physical; 0.0 = SOC-off anchor)",
    )
    parser.add_argument(
        "--mode",
        default="second_variation",
        choices=("second_variation", "first_order_insertion"),
        help="split-SOC kernel mode (second_variation is production)",
    )
    parser.add_argument(
        "--legs",
        default="xyz",
        help="leg axes as a string of x/y/z characters (default xyz)",
    )
    args = parser.parse_args()

    legs = []
    for char in args.legs:
        axis = "xyz".index(char.lower())
        direction = np.zeros(3)
        direction[axis] = 1.0
        legs.append(tuple(float(x) for x in direction))
    if not legs:
        parser.error("--legs must select at least one of x/y/z")

    out = gen_exchange_vasp_split_soc(
        native_input=args.native_input,
        cso_dump=args.cso_dump,
        output_path=args.output_path,
        rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=args.index_magnetic_atoms,
        lam=args.lam,
        mode=args.mode,
        legs=legs or LEGS,
    )
    print(f"Split-SOC exchange written to {out}")
    print(f"Provenance: {out}/split_soc_provenance.json")


if __name__ == "__main__":
    run_vasp_split_soc2J()
