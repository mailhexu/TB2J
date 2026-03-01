#!/usr/bin/env python3
"""
Example: Magnon band structure with custom q-points.

This example demonstrates how to specify custom q-points that differ from
the ASE default special points. This is useful when:
1. You want to use non-standard high-symmetry points
2. You want to define a custom k-path for your specific crystal structure
3. The ASE defaults don't match your needs

Usage:
    cd TB2J
    source .venv/bin/activate
    python examples/magnon/magnon_bands_custom_qpoints.py
"""

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters


def main():
    # Method 1: Define custom q-points in Python
    custom_qpoints = {
        "G": [0.0, 0.0, 0.0],  # Gamma point
        "X": [0.5, 0.0, 0.0],  # X point
        "M": [0.5, 0.5, 0.0],  # M point
        "R": [0.5, 0.5, 0.5],  # R point
    }

    params = MagnonParameters(
        path="TB2J_results",
        kpath="GXMRG",  # Path connecting the custom q-points
        npoints=200,
        filename="magnon_bands_custom.png",
        qpoints=custom_qpoints,  # Pass custom q-points
        show=False,
    )

    magnon = plot_magnon_bands_from_TB2J(params)
    print(f"\nMagnon calculation complete. nspin = {magnon.nspin}")
    print(f"Custom q-points used: {list(custom_qpoints.keys())}")


if __name__ == "__main__":
    main()
