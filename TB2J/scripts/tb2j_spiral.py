"""Command-line entry point for frozen spin-spiral state bundles (*.spiral.nc).

Runs :class:`TB2J.exchange_spiral.ExchangeSpiral`: MFT curvature kernels
about the torque-free spiral reference, tensor mapping through the
ExchangeNCL conventions, and the standard TB2J output tree plus the
``spiral_diagnostics.json`` report.

Runnable as ``python -m TB2J.scripts.tb2j_spiral`` (the pyproject entry
point registration is deferred by the spiral-bundle contract):

    python -m TB2J.scripts.tb2j_spiral \
        --spiral-state crI3.spiral.nc --ncell 6 \
        --output TB2J_spiral_results --eq-table
"""

from __future__ import annotations

import argparse

import numpy as np

from TB2J.exchange_spiral import ExchangeSpiral


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb2j_spiral",
        description=(
            "Exchange parameters from a frozen spin-spiral state bundle "
            "(*.spiral.nc) via the magnetic-force-theorem curvature kernels, "
            "written as a standard TB2J results directory plus "
            "spiral_diagnostics.json."
        ),
    )
    parser.add_argument(
        "--spiral-state", required=True, help="Path to the *.spiral.nc bundle"
    )
    parser.add_argument(
        "--output",
        default="TB2J_results",
        help="Output directory (default: TB2J_results)",
    )
    parser.add_argument(
        "--kmesh",
        nargs=3,
        type=int,
        default=None,
        metavar=("N", "K2", "K3"),
        help="Folded (N, 1, 1) mesh for the E(q) table (default: planar mesh)",
    )
    parser.add_argument(
        "--ncell",
        type=int,
        default=0,
        help="Ring size; 0 derives it from q commensurability (default: 0)",
    )
    parser.add_argument(
        "--kernel",
        choices=("contour", "eigenbasis"),
        default="contour",
        help="Curvature kernel path (default: contour)",
    )
    parser.add_argument(
        "--n-matsubara",
        type=int,
        default=3000,
        help="Matsubara points of the contour kernel (default: 3000)",
    )
    parser.add_argument(
        "--width", type=float, default=None, help="Smearing width override in eV"
    )
    parser.add_argument(
        "--Rcut", type=float, default=None, help="Pair distance cutoff in Angstrom"
    )
    parser.add_argument(
        "--gate-override",
        action="store_true",
        help="Report gate violations instead of failing",
    )
    parser.add_argument(
        "--eq-table", action="store_true", help="Write the E(q) diagnostic table"
    )
    parser.add_argument(
        "--q-set",
        nargs=3,
        type=float,
        action="append",
        metavar=("Q1", "Q2", "Q3"),
        default=None,
        help="q point of the E(q) table; repeatable (default: 0, +q, -q)",
    )
    parser.add_argument(
        "--cell",
        nargs=9,
        type=float,
        default=None,
        metavar=("a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "a3x", "a3y", "a3z"),
        help="Primitive cell vectors (row-major); default: unit cell",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Chemical symbols of the norb sites (default: X placeholders)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = {}
    if args.kmesh is not None:
        params["kmesh"] = list(args.kmesh)
    for key in (
        "ncell",
        "kernel",
        "n_matsubara",
        "width",
        "Rcut",
        "gate_override",
        "eq_table",
        "symbols",
    ):
        val = getattr(args, key)
        if val is not None:
            params[key] = val
    if args.q_set is not None:
        params["q_set"] = [list(qv) for qv in args.q_set]
    if args.cell is not None:
        params["cell"] = np.asarray(args.cell, dtype=float).reshape(3, 3).tolist()

    calc = ExchangeSpiral.from_spiral_state(args.spiral_state, **params)
    report = calc.run()
    calc.write_output(path=args.output)

    print("ExchangeSpiral finished.")
    print(f"  q_frac      : {report['q_frac']}")
    print(f"  ncell       : {report['ncell']}")
    print(f"  pairs       : {report['n_pairs']}")
    for name, rep in report["gates"].items():
        tag = "FLAGGED" if rep["flagged"] else ("passed" if rep["passed"] else "FAILED")
        print(f"  gate {name:<17s}: {tag} (max residual {rep['max_residual']:.3e})")
    nh = report["nonheisenberg_Cbb"]
    print(
        "  ||C^bb + J o cosTheta|| : "
        f"{nh['max_abs_offdiag']:.3e} (off-diag, scale {nh['scale']:.3e})"
    )
    print(f"  output      : {args.output}")
    return calc


if __name__ == "__main__":
    main()
