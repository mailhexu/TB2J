#!/usr/bin/env python3
"""CLI for TB2J exchange from ABINIT WFK + arbitrary local orbitals.

This is the orbital-source-agnostic entry point (ADR-0006). It accepts any
pypao-supported orbital file (UPF, SIESTA ``.ion.nc``, ABACUS ``.orb``)
together with an ABINIT WFK and VXC, projects the plane-wave states onto
those orbitals via abinao, and computes TB2J exchange in one step.

Example::

    python -m TB2J.scripts.abinit2J \\
        --wfk cri3_paoo_WFK.nc \\
        --vxc cri3_paoo_VXC.nc \\
        --orb Cr.ion.nc I.ion.nc \\
        --elements Cr --Rcut 9.5 --nz 40
"""

from __future__ import annotations

import argparse

from TB2J.versioninfo import print_license


def run_abinit2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate TB2J exchange from ABINIT plane-wave wavefunctions "
            "projected onto arbitrary local orbitals (UPF, SIESTA .ion.nc, "
            "ABACUS .orb). This wraps the abinao projection step and the "
            "TB2J exchange step into a single call."
        )
    )
    parser.add_argument(
        "--wfk",
        required=True,
        help="ABINIT _WFK.nc (NetCDF, iomode 3)",
    )
    parser.add_argument(
        "--vxc",
        required=True,
        help="ABINIT _VXC.nc (NetCDF, prtvxc 1)",
    )
    parser.add_argument(
        "--orb",
        nargs="+",
        required=True,
        help="Orbital files per species: UPF, SIESTA .ion.nc, or ABACUS .orb",
    )
    parser.add_argument(
        "--orb-format",
        default="auto",
        choices=["auto", "upf", "siesta-ionnc", "abacus-orb"],
        help="Force orbital format (default: auto-detect by extension)",
    )
    parser.add_argument(
        "--output_path",
        default="TB2J_results_orbital",
        help="Output directory (default: TB2J_results_orbital)",
    )
    parser.add_argument("--elements", nargs="+", default=None, help="Magnetic elements")
    parser.add_argument(
        "--index_magnetic_atoms",
        nargs="+",
        type=int,
        default=None,
        help="Indices of magnetic atoms (1-based, as in exchange.out)",
    )
    parser.add_argument(
        "--Rmax",
        type=int,
        default=None,
        help="Real-space R-grid radius; inferred from --Rcut",
    )
    parser.add_argument("--Rcut", type=float, default=None, help="Exchange cutoff (Å)")
    parser.add_argument(
        "--nz", type=int, default=30, help="Green-function integration points"
    )
    parser.add_argument(
        "--smearing", type=float, default=0.05, help="Smearing width (eV)"
    )
    parser.add_argument(
        "--operator_component",
        default="delta_total",
        help="Spin-splitting operator: delta_total or spectral_spin_split",
    )
    parser.add_argument(
        "--shell_charge_threshold",
        type=float,
        default=0.01,
        help="Exclude shells with projected charge below this",
    )
    parser.add_argument(
        "--shell_moment_threshold",
        type=float,
        default=0.01,
        help="Exclude shells with projected moment below this",
    )
    parser.add_argument(
        "--keep_pao_hs",
        action="store_true",
        help="Keep the intermediate PAO_HS.nc file",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write diagnostics report to this path",
    )
    parser.add_argument(
        "--fermi_eV",
        type=float,
        default=None,
        help="Fermi energy in eV (required when WFK has no valid value, e.g. wfk_fullbz output)",
    )
    parser.add_argument(
        "--population_mode",
        choices=("none", "green", "projector"),
        default="projector",
        help="Source for atom charge/moment fields (default: projector)",
    )
    parser.add_argument(
        "--n_empty",
        type=int,
        default=None,
        help="Number of empty bands above Fermi to include",
    )
    parser.add_argument(
        "--emax_relative_to_fermi_eV",
        type=float,
        default=None,
        help="Energy cutoff above Fermi (eV)",
    )
    args = parser.parse_args()

    from abinao.exchange import gen_exchange_from_orbitals

    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]

    gen_exchange_from_orbitals(
        wfk_path=args.wfk,
        vxc_path=args.vxc,
        orbital_paths=args.orb,
        orb_format=args.orb_format,
        output_path=args.output_path,
        magnetic_elements=args.elements,
        index_magnetic_atoms=indices,
        Rmax=args.Rmax,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        operator_component=args.operator_component,
        shell_charge_threshold=args.shell_charge_threshold,
        shell_moment_threshold=args.shell_moment_threshold,
        report_path=args.report,
        keep_pao_hs=args.keep_pao_hs,
        fermi_energy_eV=args.fermi_eV,
        population_mode=args.population_mode,
        n_empty=args.n_empty,
        emax_relative_to_fermi_eV=args.emax_relative_to_fermi_eV,
    )
    print(f"\nAll calculation finished. Results in {args.output_path}/")


if __name__ == "__main__":
    run_abinit2J()
