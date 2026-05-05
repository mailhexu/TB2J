#!/usr/bin/env python3
"""
Export magnon eigenstate data and prepare a Three.js scene.

Usage:
    cd TB2J
    python examples/magnon/magnon_export_animation.py
"""

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters
from TB2J.magnon.streamlit_viewer import build_scene_from_file


def main():
    params = MagnonParameters(
        path="TB2J_results",
        kpath="GMKG",
        npoints=50,
        filename="magnon_bands.png",
        export_formats=["json", "netcdf"],
        export_prefix="magnon_bands_full",
        save_wavefunctions=True,
        show=False,
    )
    plot_magnon_bands_from_TB2J(params)

    scene = build_scene_from_file(
        "magnon_bands_full.json",
        kpoint_index=0,
        band_index=0,
        amplitude=0.2,
        nframes=40,
        repetitions=(2, 2, 1),
    )
    print(f"Prepared Three.js scene with {len(scene['frames'])} frames.")


if __name__ == "__main__":
    main()
