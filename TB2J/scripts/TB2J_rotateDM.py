#!/usr/bin/env python3
"""CLI for rotating a SIESTA density matrix in spin and orbital space."""

import argparse

from TB2J.rotate_siestaDM import rotate_DM, rotate_file


def main():
    parser = argparse.ArgumentParser(
        description="Rotate a SIESTA density matrix in spin and/or orbital space."
    )
    sub = parser.add_subparsers(dest="mode")

    # --- legacy mode (backward compatible) ---
    legacy = sub.add_parser("legacy", help="Original fixed-angle spin rotation.")
    legacy.add_argument("--fdf_fname", required=True, help="Name of the *.fdf file.")
    legacy.add_argument(
        "--noncollinear",
        action="store_true",
        help="Generate 6 instead of 3 rotated DMs.",
    )

    # --- general mode (arbitrary angle + orbital) ---
    gen = sub.add_parser("rotate", help="Rotate to a specific (theta, phi) direction.")
    gen.add_argument("--dm", required=True, help="Path to the input .DM file.")
    gen.add_argument("--fdf", required=True, help="Path to the matching .fdf file.")
    gen.add_argument("--output", required=True, help="Path for the output .DM file.")
    gen.add_argument(
        "--theta", type=float, default=None, help="Polar angle in degrees."
    )
    gen.add_argument(
        "--phi", type=float, default=None, help="Azimuthal angle in degrees."
    )
    gen.add_argument(
        "--orb_indx",
        default=None,
        help="Path to ORB_INDX for orbital rotation. "
        "If omitted, only the spin part is rotated.",
    )
    gen.add_argument(
        "--no-orbital",
        action="store_true",
        help="Disable orbital rotation even if ORB_INDX is provided.",
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    if args.mode == "legacy":
        rotate_DM(args.fdf_fname, noncollinear=args.noncollinear)

    elif args.mode == "rotate":
        import math

        theta = math.radians(args.theta) if args.theta is not None else None
        phi = math.radians(args.phi) if args.phi is not None else None

        rotate_file(
            dm_path=args.dm,
            fdf_path=args.fdf,
            output_path=args.output,
            orbital=not args.no_orbital,
            orb_indx_path=args.orb_indx,
            theta=theta,
            phi=phi,
        )
        print(f"Rotated DM written to {args.output}")


if __name__ == "__main__":
    main()
