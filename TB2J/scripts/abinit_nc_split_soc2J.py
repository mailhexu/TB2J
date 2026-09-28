"""Command line driver for ABINIT NC split-SOC three-leg exchange (story-009).

Consumes the abinao ``abinit.nc_pao_hs`` v2 projections and the
``abinao.nc_soc_ks`` v1 SOC sidecar (story-008), runs the x/y/z split-SOC
legs, attaches even-prefix window convergence studies, rotates/merges into
one lattice-frame output, and enforces the FR-032 one-shot equivalence and
the SOC-off collinear anchor gates by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from TB2J.versioninfo import print_license


def run_abinit_nc_split_soc2J(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Three-direction split-SOC exchange from an abinao PAO_HS v2 file "
            "and an abinao.nc_soc_ks v1 SOC sidecar"
        )
    )
    parser.add_argument(
        "--pao_hs",
        required=True,
        help="abinao abinit.nc_pao_hs v2 strength-0 projection file",
    )
    parser.add_argument(
        "--soc_kernel",
        required=True,
        help="abinao.nc_soc_ks v1 sidecar (three x/y/z SOC legs)",
    )
    parser.add_argument(
        "--wfk",
        default=None,
        help="strength-0 WFK path; when given its SHA-256 is checked against "
        "the sidecar provenance",
    )
    parser.add_argument("--output_path", default="TB2J_results_abinit_nc_split_soc")
    parser.add_argument("--Rcut", type=float, default=10.0)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument(
        "--smearing", type=float, default=0.05, help="CFR smearing in eV"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="dimensionless SOC scaling lam of the second-variation spectrum",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        nargs="*",
        type=int,
        default=None,
        help="1-based indices of magnetic sites; required for multi-atom files "
        "with nonmagnetic ligands",
    )
    parser.add_argument(
        "--vertex_component",
        choices=("delta_total", "delta_xc_smooth", "spectral_spin_split"),
        default="delta_total",
    )
    parser.add_argument(
        "--window_prefixes",
        nargs="*",
        type=int,
        default=None,
        help="even composite-band prefixes for the per-leg window convergence "
        "study (default: 2b-2 and 2b)",
    )
    parser.add_argument(
        "--no-verify-tangent-projection",
        action="store_true",
        help="disable the FR-032 projection-only (tangent block) gate "
        "(not recommended)",
    )
    parser.add_argument(
        "--tangent_tol",
        type=float,
        default=1e-2,
        help="tangent-block tolerance in eV; the gate compares the merged "
        "raw tensor[:2,:2] against the z one-shot leg (default 1e-2, "
        "respects reference-state differences)",
    )
    parser.add_argument(
        "--no-soc-off-anchor",
        action="store_true",
        help="disable the SOC-off collinear anchor gate (not recommended)",
    )
    parser.add_argument(
        "--anchor_rtol",
        type=float,
        default=1e-7,
        help="SOC-off anchor relative Jiso tolerance (default 1e-7)",
    )
    args = parser.parse_args(argv)
    print_license()

    from TB2J.interfaces.abinit_nc_split_soc import gen_exchange_abinit_nc_split_soc

    sites = (
        None
        if args.index_magnetic_atoms is None
        else [i - 1 for i in args.index_magnetic_atoms]
    )
    result = gen_exchange_abinit_nc_split_soc(
        args.pao_hs,
        args.soc_kernel,
        output_path=args.output_path,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        lam=args.scale,
        index_magnetic_atoms=sites,
        vertex_component=args.vertex_component,
        window_prefixes=args.window_prefixes,
        verify_tangent_projection=not args.no_verify_tangent_projection,
        tangent_tol_eV=args.tangent_tol,
        soc_off_anchor=not args.no_soc_off_anchor,
        anchor_jiso_rtol=args.anchor_rtol,
        wfk=args.wfk,
    )
    anchor = result["soc_off_anchor"]
    print(
        f"SOC-off anchor: passed={anchor.get('passed')} "
        f"(max rel Jiso dev {anchor.get('max_rel_Jiso_dev', float('nan')):.3e})"
    )
    tangent = result["tangent_projection_check"]
    if tangent is not None:
        print(
            f"FR-032 tangent projection gate: passed={tangent['passed']} over "
            f"{tangent['pairs_compared']} pairs "
            f"(max |dT[:2,:2]|={tangent['max_transverse_dev_eV']:.3e} eV, tol "
            f"{tangent['tol_eV']:.1e} eV; full Jiso is final only after the "
            "rank-9 merge)"
        )
    print(f"Merged exchange: {Path(result['output_path']) / 'exchange.out'}")
    return result


if __name__ == "__main__":
    run_abinit_nc_split_soc2J()
