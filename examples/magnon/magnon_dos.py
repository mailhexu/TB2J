#!/usr/bin/env python3
"""
Magnon density of states calculation.

Usage:
    cd TB2J
    source .venv/bin/activate
    python examples/magnon/magnon_dos.py
"""

from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters


def main():
    params = MagnonParameters(
        path="TB2J_results",
        kmesh=[20, 20, 20],
        gamma=True,
        width=0.001,
        npts=401,
        filename="magnon_dos.png",
        DMI=False,
        Jani=False,
        show=False,
    )

    plot_magnon_dos_from_TB2J(params)
    print("\nMagnon DOS calculation complete.")


if __name__ == "__main__":
    main()
