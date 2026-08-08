#!/usr/bin/env python3
"""CLI for TB2J ABINIT norm-conserving PAO exchange calculations."""

from __future__ import annotations

import argparse
import warnings

from TB2J.interfaces.abinit_savetb2j import (
    ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT,
    gen_exchange_abinit_nc_pao,
)
from TB2J.versioninfo import print_license


def run_abinit_nc_pao2J():
    warnings.warn(
        "abinit_nc_pao2J (native ABINIT NC-PAO exchange) is deprecated. For new "
        "projector/Green-function exchange from ABINIT output, use the maintained "
        "ABINIT + abinao handoff (abinao.exchange.gen_exchange_from_orbitals -> "
        "TB2J projector core) or abinit_projector2J.py instead. This CLI is "
        "retained only for backward compatibility.",
        DeprecationWarning,
        stacklevel=2,
    )
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J-style exchange parameters from an ABINIT "
            "norm-conserving PAO or NC spherical-window savetb2j NetCDF file. "
            "For spherical-window files, the combined XC+U component is loaded "
            "as delta_total. The default operator component is "
            f"{ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT}, falling back to "
            "delta_total when spectral_spin_split is unavailable."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="ABINIT NC PAO or NC spherical-window savetb2j file",
    )
    parser.add_argument(
        "--output_path",
        default="TB2J_results_abinit_nc_pao",
        help="output directory",
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
        default=ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT,
        help=(
            "ABINIT NC PAO or NC spherical-window operator component to use; "
            "spherical-window exchange-ready files provide delta_total from "
            "delta_spherical_xc_u. Default: "
            f"{ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT}"
        ),
    )
    parser.add_argument(
        "--population_mode",
        choices=("none", "green", "projector"),
        default="projector",
        help=(
            "source for exchange.out atom charge/moment fields; projector uses "
            "occupied bands with the k-dependent S(k)^-1 dual metric"
        ),
    )
    parser.add_argument(
        "--shell_charge_threshold",
        type=float,
        default=0.01,
        help="exclude local-operator PAO shells with projected charge below this value",
    )
    parser.add_argument(
        "--shell_moment_threshold",
        type=float,
        default=0.01,
        help="exclude local-operator PAO shells with projected moment norm below this value",
    )
    parser.add_argument(
        "--no_shell_filter",
        action="store_true",
        help="disable projected-charge shell filtering",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
        help="absolute maximum eigenvalue in eV included in Green spectral sums",
    )
    parser.add_argument(
        "--emax_relative_to_fermi",
        type=float,
        default=None,
        help="maximum eigenvalue relative to Fermi level in eV",
    )
    parser.add_argument(
        "--n_empty",
        type=int,
        default=None,
        help="include all occupied bands plus this fixed number of empty bands per spin/k-point",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="optional Markdown diagnostics report path",
    )
    parser.add_argument(
        "--overlap_mode",
        choices=("inverse", "svd", "lowdin", "tikhonov", "plain"),
        default="inverse",
        help="overlap treatment for projected Green functions",
    )
    parser.add_argument(
        "--overlap_rcond",
        type=float,
        default=1.0e-10,
        help="relative cutoff or Tikhonov lambda for regularized overlap modes",
    )
    args = parser.parse_args()
    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]
    exchange_out, _ = gen_exchange_abinit_nc_pao(
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
        shell_charge_threshold=(
            None if args.no_shell_filter else args.shell_charge_threshold
        ),
        shell_moment_threshold=(
            None if args.no_shell_filter else args.shell_moment_threshold
        ),
        emax_eV=args.emax,
        emax_relative_to_fermi_eV=args.emax_relative_to_fermi,
        n_empty=args.n_empty,
        report_path=args.report,
        overlap_mode=args.overlap_mode,
        overlap_rcond=args.overlap_rcond,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_abinit_nc_pao2J()
