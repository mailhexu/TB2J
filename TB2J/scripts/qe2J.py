#!/usr/bin/env python3
"""CLI for TB2J exchange calculations from a QE projector dump file."""

from __future__ import annotations

import argparse

from TB2J.interfaces.qe_projector import gen_exchange_qe
from TB2J.versioninfo import print_license


def run_qe2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J-style exchange parameters from a Quantum "
            "ESPRESSO projector dump file (QE fork branch TB2J). Supports "
            "KB/beta becp dumps (TB2JQEDUMPV1*) and atomic-wavefunction "
            "dumps (TB2JQEATWFC1.0, TB2J_PROJECTORS=atomic). KB spinor "
            "dumps (TB2JQEDUMPNC1.0/NC1.1) require independent x/y/z dumps."
        )
    )
    parser.add_argument("--input", help="collinear KB or atomic QE projector dump")
    for axis in ("x", "y", "z"):
        parser.add_argument(
            f"--input_{axis}", help=f"KB spinor dump with magnetization along {axis}"
        )
    parser.add_argument(
        "--output_path", default="TB2J_results_qe", help="output directory"
    )
    parser.add_argument(
        "--Rcut",
        type=float,
        default=10.0,
        help="spin-pair distance cutoff in Angstrom",
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
        "--overlap_mode",
        default=None,
        choices=["inverse", "svd", "lowdin", "tikhonov", "plain"],
        help=(
            "how the runtime dresses primal atomic coefficients with the "
            "k-dependent M(k)^-1 overlap (atomic-wavefunction dumps only; "
            "default inverse)"
        ),
    )
    parser.add_argument(
        "--overlap_rcond",
        type=float,
        default=None,
        help=(
            "cutoff for small singular values of M(k) in svd/lowdin/"
            "tikhonov overlap modes (atomic-wavefunction dumps only)"
        ),
    )
    parser.add_argument(
        "--leg_consistency_atol_eV",
        type=float,
        default=5e-8,
        help=(
            "bounded repeated-row consistency gate in eV for the spinor "
            "three-leg merge (spinor --input_x/y/z only; default 5e-8, "
            "capped at 1e-7); the measured worst repeated-row spread is "
            "reported in the output description"
        ),
    )
    args = parser.parse_args()
    spinor_inputs = {axis: getattr(args, f"input_{axis}") for axis in ("x", "y", "z")}
    if any(spinor_inputs.values()):
        if args.input or not all(spinor_inputs.values()):
            parser.error("use --input alone or --input_x --input_y --input_z together")
        if args.overlap_mode is not None or args.overlap_rcond is not None:
            parser.error(
                "overlap controls are invalid for already-dual KB spinor dumps"
            )
        input_files = spinor_inputs
    elif args.input:
        input_files = args.input
    else:
        parser.error("supply --input or all --input_x --input_y --input_z")
    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]
    exchange_out, _ = gen_exchange_qe(
        input_files,
        output_path=args.output_path,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
        overlap_mode=args.overlap_mode,
        overlap_rcond=args.overlap_rcond,
        leg_consistency_atol_eV=(
            args.leg_consistency_atol_eV if any(spinor_inputs.values()) else None
        ),
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_qe2J()
