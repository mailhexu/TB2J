#!/usr/bin/env python3
"""CLI for the VASP KS-basis split-SOC exchange workflow (story 011).

Consumes the artifacts of THREE independent collinear strength-0 VASP
runs patched with the story-010 dump hooks — one per SAXIS reference
axis (x, y, z), each providing ``tb2j_native.bin`` (v5/v6 collinear
native export) and ``tb2j_cso.bin`` (one-center CSO operator + COCC) —
measures each reference's transverse block through the shared tangent
kernel, rotates the blocks into the lattice frame, and merges them with
the rank-nine raw-tensor solve.
"""

from __future__ import annotations

import argparse

from TB2J.interfaces.vasp_split_soc import (
    LEG_TAGS,
    _parse_leg_argument,
    gen_exchange_vasp_split_soc,
)
from TB2J.versioninfo import print_license


def run_vasp_split_soc2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J exchange from three collinear VASP strength-0 "
            "runs patched with the story-010 CSO dump (SAXIS = 1 0 0 / "
            "0 1 0 / 0 0 1): per-axis transverse legs through the tangent "
            "kernel, lattice-frame rotation, and the rank-nine raw-tensor "
            "merge."
        ),
        epilog=(
            "Typical workflow: run the patched collinear VASP (ISPIN=2, "
            "LSORBIT=.FALSE., story-010 dump hooks) three times with "
            "SAXIS = 1 0 0, 0 1 0, 0 0 1, then pass the three run "
            "directories to this command."
        ),
    )
    parser.add_argument(
        "--leg",
        action="append",
        required=True,
        metavar="TAG=RUN_DIR",
        help=(
            "one reference per axis, TAG in x/y/z; RUN_DIR is a directory "
            "containing tb2j_native.bin and tb2j_cso.bin (or an explicit "
            "TAG=native_path:cso_path pair). Repeat for x, y and z."
        ),
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
        "--merge_consistency_atol",
        type=float,
        default=1.0e-8,
        help=(
            "tolerance for the repeated-diagonal consistency gate of the "
            "rank-nine merge (eV)"
        ),
    )
    parser.add_argument(
        "--no-band-window-study",
        action="store_true",
        help="skip the ADR-8 band-window convergence report",
    )
    args = parser.parse_args()

    artifacts = {}
    for spec in args.leg:
        parsed = _parse_leg_argument(spec)
        tag = next(iter(parsed))
        if tag in artifacts:
            parser.error(f"--leg {tag!r} given more than once")
        artifacts.update(parsed)
    missing = [tag for tag in LEG_TAGS if tag not in artifacts]
    if missing:
        parser.error(
            "--leg must provide all three references %s; missing %s"
            % (LEG_TAGS, missing)
        )

    out = gen_exchange_vasp_split_soc(
        artifacts,
        output_path=args.output_path,
        rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=args.index_magnetic_atoms,
        lam=args.lam,
        mode=args.mode,
        merge_consistency_atol=args.merge_consistency_atol,
        band_window_study=not args.no_band_window_study,
    )
    print(f"Split-SOC exchange written to {out}")
    print(f"Provenance: {out}/split_soc_provenance.json")


if __name__ == "__main__":
    run_vasp_split_soc2J()
