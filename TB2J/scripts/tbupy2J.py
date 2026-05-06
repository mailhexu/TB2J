"""Command-line entry point for TBUpy result files."""

from __future__ import annotations

import argparse

from TB2J.interfaces.tbupy_interface import gen_exchange_tbupy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute exchange from a TBUpy .tbupy.nc result file."
    )
    parser.add_argument("--input", required=True, help="Path to .tbupy.nc file")
    parser.add_argument("--output_path", default="TB2J_results")
    parser.add_argument("--kmesh", nargs=3, type=int, default=[5, 5, 5])
    parser.add_argument("--efermi", type=float, default=None)
    parser.add_argument("--elements", nargs="+", default=None, dest="magnetic_elements")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--description", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    kwargs = vars(args)
    tbupy_result_file = kwargs.pop("input")
    return gen_exchange_tbupy(tbupy_result_file=tbupy_result_file, **kwargs)


if __name__ == "__main__":
    main()
