#!/usr/bin/env python3
"""
Magnon band structure calculation with default parameters.

Usage:
    cd TB2J
    source .venv/bin/activate
    python examples/magnon/magnon_bands.py
"""

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters


def main():
    params = MagnonParameters(
        path="TB2J_results",
        kpath="GMKG",
        npoints=200,
        filename="magnon_bands.png",
        show=False,
    )

    magnon = plot_magnon_bands_from_TB2J(params)
    print(f"\nMagnon calculation complete. nspin = {magnon.nspin}")


if __name__ == "__main__":
    main()
