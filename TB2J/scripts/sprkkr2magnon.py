from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from TB2J.interfaces.sprkkr import (
    SprkkrParseError,
    read_sprkkr_exchange,
    sprkkr_to_spinio,
)
from TB2J.magnon.magnon3 import Magnon, save_bands_data
from TB2J.magnon.magnon_parameters import parse_qpoints_string
from TB2J.mathutils.auto_kpath import auto_kpath

try:
    from ase.cell import Cell as AseCell
except ImportError:
    from ase import Cell as AseCell


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute magnon bands from SPRKKR exchange output files"
    )
    parser.add_argument(
        "-s", "--structure", required=True, help="SPRKKR .str structure file"
    )
    parser.add_argument(
        "-e", "--exchange", required=True, help="SPRKKR Jij.dat exchange file"
    )
    parser.add_argument(
        "-S",
        "--magnetic-species",
        nargs="+",
        default=None,
        help="Chemical symbols to treat as magnetic sites, for example Ru",
    )
    parser.add_argument(
        "-i",
        "--magnetic-sites",
        type=int,
        nargs="+",
        default=None,
        help="1-based SPRKKR site IDs to treat as magnetic",
    )
    parser.add_argument(
        "-m",
        "--moment",
        type=float,
        nargs="+",
        help=(
            "Magnetic moment values: one scalar, N z moments, or 3N vector "
            "components for N magnetic sites"
        ),
    )
    parser.add_argument(
        "-t",
        "--tensor-policy",
        choices=("transverse-block", "transverse-block-jzz", "isotropic"),
        default="transverse-block",
        help="How to convert SPRKKR J_xx/J_yy/J_xy/J_yx columns",
    )
    parser.add_argument(
        "-b", "--bands", action="store_true", help="Compute magnon bands"
    )
    parser.add_argument("-k", "--kpath", default=None, help="Band path such as GXMG")
    parser.add_argument(
        "-n", "--npoints", type=int, default=300, help="Number of band points"
    )
    parser.add_argument(
        "--qpoints",
        default=None,
        help='Custom q-points as name:coord pairs, e.g. "G:0,0,0,X:0.5,0,0"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default="sprkkr_magnon.png",
        help="Output image path for band structure",
    )
    parser.add_argument("--show", action="store_true", help="Show the band plot")
    parser.add_argument(
        "-w",
        "--write-tb2j-results",
        default=None,
        help="Write a TB2J-compatible pickle directory",
    )
    return parser


def _build_spinio(args: argparse.Namespace):
    if args.magnetic_species is None and args.magnetic_sites is None:
        raise SystemExit("Provide --magnetic-species or --magnetic-sites")
    if args.moment is None:
        raise SystemExit("Provide --moment for SPRKKR magnon normalization")
    data = read_sprkkr_exchange(
        structure_file=args.structure,
        exchange_file=args.exchange,
        magnetic_species=args.magnetic_species,
        magnetic_sites=args.magnetic_sites,
        moment=args.moment,
    )
    return sprkkr_to_spinio(data, tensor_policy=args.tensor_policy)


def _band_data(
    magnon: Magnon,
    path: str | None,
    npoints: int,
    special_points: dict | None = None,
) -> tuple[list, np.ndarray, Any, np.ndarray, dict]:
    if path is None:
        labels, bands, xcoords = magnon.get_magnon_bands(npoints=npoints)
        _, kptlist, _, _, band_special_points = auto_kpath(
            magnon.cell,
            None,
            npoints=npoints,
        )
        kpoints = np.concatenate(kptlist)
    else:
        if special_points is None:
            labels, bands, xcoords = magnon.get_magnon_bands(
                path=path,
                npoints=npoints,
            )
        else:
            labels, bands, xcoords = magnon.get_magnon_bands(
                path=path,
                npoints=npoints,
                special_points=special_points,
            )
        bandpath = AseCell(magnon.cell).bandpath(
            path=path,
            npoints=npoints,
            special_points=special_points,
        )
        kpoints = bandpath.kpts
        band_special_points = dict(bandpath.special_points)
        band_special_points["Gamma"] = band_special_points.pop("G", np.zeros(3))
    return labels, bands * 1000, xcoords, kpoints, band_special_points


def main(argv: list[str] | None = None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.bands and args.write_tb2j_results is None:
        parser.error("Please specify --bands or --write-tb2j-results")

    try:
        spinio = _build_spinio(args)
    except (SprkkrParseError, ValueError) as exc:
        parser.error(str(exc))
    if args.write_tb2j_results is not None:
        spinio.write_all(path=args.write_tb2j_results)

    if args.bands:
        try:
            special_points = parse_qpoints_string(args.qpoints)
        except ValueError as exc:
            parser.error(str(exc))
        magnon = Magnon.load_from_io(
            spinio,
            Jiso=True,
            Jani=args.tensor_policy in {"transverse-block", "transverse-block-jzz"},
            DMI=False,
            SIA=False,
        )
        magnon.set_reference(
            Q=np.zeros(3),
            uz=np.array([[0.0, 0.0, 1.0]]),
            n=np.array([0.0, 0.0, 1.0]),
            magmoms=magnon.magmom,
        )
        labels, energies, xcoords, kpoints, special_points = _band_data(
            magnon,
            args.kpath,
            args.npoints,
            special_points=special_points,
        )
        output = Path(args.output)
        data_file = output.with_suffix(".json")
        bands = save_bands_data(
            kpoints=kpoints,
            energies=energies,
            kpath_labels=labels,
            special_points=special_points,
            xcoords=xcoords,
            filename=str(data_file),
        )
        bands.plot(filename=str(output), show=args.show)
    return None


if __name__ == "__main__":
    main()
