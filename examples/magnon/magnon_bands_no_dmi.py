#!/usr/bin/env python3
"""
Magnon band structure calculation without DMI and anisotropic exchange.

This example shows how to calculate magnon bands using only the isotropic
Heisenberg exchange interaction, which is useful for understanding the
basic magnetic coupling without spin-orbit effects.

Usage:
    cd TB2J
    source .venv/bin/activate
    python examples/magnon/magnon_bands_no_dmi.py
"""

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters


def main():
    params = MagnonParameters(
        path="TB2J_results",
        kpath="GMKG",
        npoints=200,
        filename="magnon_bands_no_dmi.png",
        DMI=False,
        Jani=False,
        show=False,
    )

    magnon = plot_magnon_bands_from_TB2J(params)
    print(f"\nMagnon calculation complete (isotropic only). nspin = {magnon.nspin}")


if __name__ == "__main__":
    main()
