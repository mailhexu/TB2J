#!/usr/bin/env python
"""Create a supercell TB2J result directory.

Usage examples:
    python srmno3_gafm_supercell.py --input TB2J_results --output output_new_matrix \
        --matrix 1 1 0 1 0 1 0 1 1

    python srmno3_gafm_supercell.py --input TB2J_results --output output --matrix 2 1 1
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from TB2J.io_exchange.edit import load, make_supercell, save


def _parse_supercell_matrix(values: list[int]) -> np.ndarray:
    """Build a 3x3 integer supercell matrix from 3 or 9 values."""
    if len(values) == 3:
        return np.diag(values)
    if len(values) == 9:
        return np.array(values, dtype=int).reshape(3, 3)
    raise ValueError(
        "Supercell matrix must be 3 diagonal integers or 9 matrix entries."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a TB2J supercell from an existing TB2J result."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input TB2J results directory or TB2J.pickle path",
    )
    parser.add_argument(
        "--output",
        default="output_new_matrix",
        help="Output directory to write the supercell TB2J results",
    )
    parser.add_argument(
        "--matrix",
        nargs="+",
        type=int,
        required=True,
        help="Supercell matrix as 3 diagonal integers or 9 full-matrix integers",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Use centered lattice vectors when building the supercell",
    )

    args = parser.parse_args()

    try:
        sc_matrix = _parse_supercell_matrix(args.matrix)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loading TB2J result from: {args.input}")
    spinio = load(args.input)

    print(f"Building supercell with matrix: {sc_matrix.tolist()}")
    print(f"Output directory: {args.output}")
    sc_spinio = make_supercell(spinio, sc_matrix, center=args.center)
    save(sc_spinio, args.output)

    print("Done: supercell data written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
