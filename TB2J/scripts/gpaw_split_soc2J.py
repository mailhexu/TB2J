"""Command line driver for one-strength-zero GPAW split-SOC exchange and MAE."""

from __future__ import annotations

import argparse

from TB2J.interfaces.gpaw_split_soc import gen_exchange_gpaw_split_soc
from TB2J.versioninfo import print_license


def run_gpaw_split_soc2J(argv=None):
    parser = argparse.ArgumentParser(
        description="Three-direction SOC exchange and MAE from one collinear old-API GPAW checkpoint"
    )
    parser.add_argument(
        "--input", required=True, help="converged collinear no-SOC legacy .gpw"
    )
    parser.add_argument("--output_path", default="TB2J_results_gpaw_split_soc")
    parser.add_argument("--Rcut", type=float, default=10.0)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument(
        "--smearing", type=float, default=0.05, help="CFR smearing in eV"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="frozen GPAW SOC operator strength"
    )
    parser.add_argument(
        "--mae-contour-tolerance",
        type=float,
        default=5e-6,
        help="reported second-order MAE comparison bound in eV",
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        nargs="*",
        type=int,
        default=None,
        help="1-based indices; default sites with nonzero frozen moments",
    )
    parser.add_argument(
        "--vertex_component", choices=("delta_xc", "delta_total"), default="delta_total"
    )
    args = parser.parse_args(argv)
    print_license()
    sites = (
        None
        if args.index_magnetic_atoms is None
        else [i - 1 for i in args.index_magnetic_atoms]
    )
    result = gen_exchange_gpaw_split_soc(
        args.input,
        output_path=args.output_path,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        scale=args.scale,
        mae_contour_tolerance_eV=args.mae_contour_tolerance,
        magnetic_sites=sites,
        vertex_component=args.vertex_component,
    )
    print(f"Merged exchange: {result['merged_path'] / 'exchange.out'}")
    for name, row in result["mae"].items():
        print(
            f"{name}: E_band={row['band_energy_eV']:.12f} eV; "
            f"MAE_z={row['relative_to_z_eV']:.12f} eV; "
            f"contour2_z={row['contour_relative_to_z_eV']:.12f} eV; "
            f"residual={row['contour_residual_eV']:.3g} eV; "
            f"within={row['contour_within_tolerance']}"
        )
    return result


if __name__ == "__main__":
    run_gpaw_split_soc2J()
