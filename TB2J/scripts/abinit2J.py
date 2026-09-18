#!/usr/bin/env python3
"""Unified TB2J CLI for ABINIT exchange calculations.

Three subcommands cover the three ABINIT output regimes:

``savetb2j``
    Exchange from an ABINIT ``savetb2j``/``write_pao_HS_nc`` NetCDF file.
    The backend (PAW ``abinit.savetb2j.projector`` vs norm-conserving
    PAO / spherical-window schemas) is detected from the file's
    ``schema_name`` attribute.

``orbitals``
    One-step workflow: project an ABINIT WFK (+VXC) onto arbitrary
    local orbitals (UPF, SIESTA ``.ion.nc``, ABACUS ``.orb``) via
    abinao and compute exchange.

``pawlog``
    PAW workflow from an ABINIT WFK, PAW-XML pseudopotentials, and the
    spin-resolved on-site D_ij printed in the ABINIT log
    (``pawprtvol -1``).
"""

from __future__ import annotations

import argparse

from TB2J.versioninfo import print_license


def _add_common_arguments(parser):
    parser.add_argument("--output_path", default=None, help="output directory")
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
        "--Rcut",
        type=float,
        default=10.0,
        help="spin-pair distance cutoff in Angstrom",
    )
    parser.add_argument(
        "--nz", type=int, default=30, help="number of continued-fraction poles"
    )
    parser.add_argument(
        "--smearing", type=float, default=0.05, help="CFR smearing in eV"
    )


def _indices(args):
    if args.index_magnetic_atoms is None:
        return None
    return [i - 1 for i in args.index_magnetic_atoms]


_PAW_SAVETB2J_ONLY = ("population_mode",)
_NC_ONLY = (
    "no_shell_filter",
    "emax",
    "emax_relative_to_fermi",
    "n_empty",
    "report",
    "overlap_mode",
    "overlap_rcond",
    "dftu_file",
)


def _check_applicable(args, kind):
    def _is_set(name):
        value = getattr(args, name, None)
        if isinstance(value, bool):
            return value
        return value is not None

    nc_only = [name for name in _NC_ONLY if _is_set(name)]
    paw_only = [name for name in _PAW_SAVETB2J_ONLY if _is_set(name)]
    if kind == "paw_savetb2j" and nc_only:
        raise SystemExit(
            f"Option(s) {nc_only} only apply to norm-conserving PAO files, "
            "but this file is a PAW savetb2j export."
        )
    if kind == "nc_pao" and paw_only:
        raise SystemExit(
            f"Option(s) {paw_only} only apply to PAW savetb2j files, "
            "but this file is a norm-conserving PAO export."
        )


def _run_savetb2j(args):
    from TB2J.interfaces.abinit_savetb2j import (
        abinit_netcdf_kind,
        gen_exchange_abinit_nc_pao,
        gen_exchange_abinit_projector,
    )

    kind = abinit_netcdf_kind(args.input)
    _check_applicable(args, kind)
    common = dict(
        output_path=args.output_path,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        magnetic_elements=args.elements,
        index_magnetic_atoms=_indices(args),
        operator_component=args.operator_component,
    )
    if kind == "paw_savetb2j":
        exchange_out, _ = gen_exchange_abinit_projector(
            args.input,
            population_mode=args.population_mode or "projector",
            shell_charge_threshold=args.shell_charge_threshold,
            shell_moment_threshold=args.shell_moment_threshold,
            **common,
        )
    else:
        exchange_out, _ = gen_exchange_abinit_nc_pao(
            args.input,
            population_mode=args.population_mode or "none",
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
            overlap_mode=args.overlap_mode or "inverse",
            overlap_rcond=args.overlap_rcond,
            dftu_file=args.dftu_file,
            **common,
        )
    print(f"Wrote {exchange_out}")


def _run_orbitals(args):
    from abinao.exchange import gen_exchange_from_orbitals

    gen_exchange_from_orbitals(
        wfk_path=args.wfk,
        vxc_path=args.vxc,
        orbital_paths=args.orb,
        orb_format=args.orb_format,
        output_path=args.output_path or "TB2J_results_orbital",
        magnetic_elements=args.elements,
        index_magnetic_atoms=_indices(args),
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        operator_component=args.operator_component,
        shell_charge_threshold=args.shell_charge_threshold,
        shell_moment_threshold=args.shell_moment_threshold,
        keep_pao_hs=args.keep_pao_hs,
        report_path=args.report,
        fermi_energy_eV=args.fermi_eV,
        population_mode=args.population_mode or "projector",
        n_empty=args.n_empty,
        emax_relative_to_fermi_eV=args.emax_relative_to_fermi,
    )


def _run_pawlog(args):
    from TB2J.interfaces.abinit_paw import gen_exchange_abinit_paw

    exchange_out, _ = gen_exchange_abinit_paw(
        wfk_path=args.wfk,
        paw_xml_path=args.paw_xml,
        log_path=args.log,
        projected_data_path=args.projected_data,
        magnetic_elements=args.elements,
        index_magnetic_atoms=_indices(args),
        output_path=args.output_path or "TB2J_results_abinit_paw",
        nz=args.nz,
        smearing_eV=args.smearing,
        Rcut=args.Rcut,
        delta_unit=args.delta_unit,
        snapshot_cache=args.snapshot_cache,
        write_snapshot_cache=args.write_snapshot_cache,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="abinit2J",
        description=(
            "Calculate TB2J exchange from ABINIT outputs. Subcommands select "
            "the ABINIT output regime; --Rcut/--nz/--smearing/--elements are "
            "shared."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "savetb2j",
        help="ABINIT savetb2j / write_pao_HS_nc NetCDF (PAW or NC, auto-detected)",
    )
    _add_common_arguments(p)
    p.add_argument("--input", required=True, help="ABINIT NetCDF export")
    p.add_argument(
        "--operator_component",
        default=None,
        help="operator component for the spin splitting, e.g. delta_total (default: backend default)",
    )
    p.add_argument(
        "--population_mode",
        choices=("none", "green", "projector"),
        default=None,
        help="source for exchange.out charge/moment fields",
    )
    p.add_argument(
        "--shell_charge_threshold",
        type=float,
        default=None,
        help="PAO shell charge filter",
    )
    p.add_argument(
        "--shell_moment_threshold",
        type=float,
        default=None,
        help="PAO shell moment filter",
    )
    p.add_argument(
        "--no_shell_filter",
        action="store_true",
        help="disable default NC shell filtering (NC files only)",
    )
    p.add_argument(
        "--emax", type=float, default=None, help="band energy window (NC only)"
    )
    p.add_argument(
        "--emax_relative_to_fermi",
        type=float,
        default=None,
        help="band window above E_F (NC only)",
    )
    p.add_argument(
        "--n_empty", type=int, default=None, help="empty bands added (NC only)"
    )
    p.add_argument(
        "--report", default=None, help="write a band-window report (NC only)"
    )
    p.add_argument(
        "--overlap_mode",
        choices=("inverse", "svd", "lowdin", "tikhonov", "plain"),
        default=None,
        help="overlap treatment (NC only)",
    )
    p.add_argument(
        "--overlap_rcond",
        type=float,
        default=None,
        help="regularization parameter (NC only)",
    )
    p.add_argument(
        "--dftu_file",
        default=None,
        help="ABINIT *_DFTU.nc for the Hubbard component (NC only)",
    )
    p.set_defaults(func=_run_savetb2j)

    p = sub.add_parser(
        "orbitals",
        help="WFK + VXC projected onto arbitrary local orbitals via abinao",
    )
    _add_common_arguments(p)
    p.add_argument("--wfk", required=True, help="ABINIT _WFK.nc (NetCDF, iomode 3)")
    p.add_argument("--vxc", required=True, help="ABINIT _VXC.nc (prtvxc 1)")
    p.add_argument(
        "--orb",
        nargs="+",
        required=True,
        help="orbital files per species: UPF, SIESTA .ion.nc, or ABACUS .orb",
    )
    p.add_argument(
        "--orb-format",
        default="auto",
        choices=["auto", "upf", "ionnc", "abacus-orb"],
        help="force orbital format (default: auto-detect)",
    )
    p.add_argument(
        "--operator_component", default="delta_total", help="spin-splitting operator"
    )
    p.add_argument(
        "--shell_charge_threshold",
        type=float,
        default=0.01,
        help="PAO shell charge filter",
    )
    p.add_argument(
        "--shell_moment_threshold",
        type=float,
        default=0.01,
        help="PAO shell moment filter",
    )
    p.add_argument(
        "--keep_pao_hs", action="store_true", help="keep the intermediate PAO_HS.nc"
    )
    p.add_argument("--report", default=None, help="write a band-window report")
    p.add_argument("--fermi_eV", type=float, default=None, help="override Fermi energy")
    p.add_argument(
        "--population_mode",
        choices=("none", "green", "projector"),
        default=None,
        help="source for exchange.out charge/moment fields",
    )
    p.add_argument("--n_empty", type=int, default=None, help="empty bands added")
    p.add_argument(
        "--emax_relative_to_fermi",
        type=float,
        default=None,
        help="band window above E_F",
    )
    p.set_defaults(func=_run_orbitals)

    p = sub.add_parser(
        "pawlog",
        help="WFK + PAW-XML + pawprt D_ij from the ABINIT log",
    )
    _add_common_arguments(p)
    p.add_argument("--wfk", required=True, help="ABINIT _WFK.nc (NetCDF)")
    p.add_argument(
        "--log", required=True, help="ABINIT log with pawprtvol -1 D_ij blocks"
    )
    p.add_argument(
        "--paw_xml",
        nargs="+",
        required=True,
        help="PAW-XML pseudopotential per species",
    )
    p.add_argument(
        "--projected_data", default=None, help="precomputed abinao projection NetCDF"
    )
    p.add_argument(
        "--snapshot_cache", default=None, help="reuse a cached PAW projection snapshot"
    )
    p.add_argument(
        "--write_snapshot_cache", default=None, help="write a projection snapshot cache"
    )
    p.add_argument(
        "--delta_unit",
        default=None,
        choices=[None, "eV", "hartree"],
        help="override delta energy unit (auto-detected by default)",
    )
    p.set_defaults(func=_run_pawlog)

    return parser


def run_abinit2J():
    print_license()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    run_abinit2J()
