"""Shared parameters for magnon band and DOS calculations."""

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import tomli
import tomli_w


@dataclass
class MagnonParameters:
    """Common parameters for magnon band structure and DOS calculations."""

    path: str = "TB2J_results"
    filename: str = None

    Jiso: bool = True
    Jani: bool = True
    DMI: bool = True
    SIA: bool = True

    Q: Optional[List[float]] = None
    uz_file: Optional[str] = None
    n: Optional[List[float]] = None
    spin_conf_file: Optional[str] = None
    show: bool = False

    kpath: str = None
    npoints: int = 300

    kmesh: List[int] = field(default_factory=lambda: [20, 20, 20])
    gamma: bool = True
    width: float = 0.001
    window: Optional[Tuple[float, float]] = None
    npts: int = 401

    @classmethod
    def from_toml(cls, filename: str) -> "MagnonParameters":
        """Load parameters from a TOML file."""
        with open(filename, "rb") as f:
            data = tomli.load(f)
        return cls(**data)

    def to_toml(self, filename: str):
        """Save parameters to a TOML file."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        with open(filename, "wb") as f:
            tomli_w.dump(data, f)

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.Q is not None and len(self.Q) != 3:
            raise ValueError("Q must be a list of 3 numbers")
        if self.n is not None and len(self.n) != 3:
            raise ValueError("n must be a list of 3 numbers")
        if self.kmesh is not None and len(self.kmesh) != 3:
            raise ValueError("kmesh must be a list of 3 integers")

        if self.uz_file and not Path(self.uz_file).is_absolute():
            self.uz_file = str(Path(self.path) / self.uz_file)
        if self.spin_conf_file and not Path(self.spin_conf_file).is_absolute():
            self.spin_conf_file = str(Path(self.path) / self.spin_conf_file)


def add_common_magnon_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for magnon band/DOS CLIs."""
    parser.add_argument(
        "-p",
        "--path",
        default="TB2J_results",
        help="Path to TB2J results directory (default: TB2J_results)",
    )

    parser.add_argument(
        "--no-Jiso",
        action="store_false",
        dest="Jiso",
        help="Exclude isotropic exchange interactions",
    )
    parser.add_argument(
        "--no-Jani",
        action="store_false",
        dest="Jani",
        help="Exclude anisotropic exchange interactions",
    )
    parser.add_argument(
        "--no-DMI",
        action="store_false",
        dest="DMI",
        help="Exclude Dzyaloshinskii-Moriya interactions",
    )
    parser.add_argument(
        "--no-SIA",
        action="store_false",
        dest="SIA",
        help="Exclude single-ion anisotropy",
    )

    parser.add_argument(
        "-c",
        "--spin-conf-file",
        type=str,
        help="Path to file containing magnetic moments for each spin (nspin×3 array)",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show figure on screen",
    )


def add_band_specific_args(parser: argparse.ArgumentParser) -> None:
    """Add band-specific arguments."""
    parser.add_argument(
        "-k",
        "--kpath",
        default=None,
        help="k-path specification (default: auto-detected from cell type)",
    )
    parser.add_argument(
        "-n",
        "--npoints",
        type=int,
        default=300,
        help="Number of k-points along the path (default: 300)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="magnon_bands.png",
        help="Output file name (default: magnon_bands.png)",
    )


def add_dos_specific_args(parser: argparse.ArgumentParser) -> None:
    """Add DOS-specific arguments."""
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=[20, 20, 20],
        metavar=("nx", "ny", "nz"),
        help="k-point mesh dimensions (default: 20, 20, 20)",
    )
    parser.add_argument(
        "--no-gamma",
        action="store_false",
        dest="gamma",
        help="Exclude Gamma point from k-mesh",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.001,
        help="Gaussian smearing width in eV (default: 0.001)",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("emin", "emax"),
        help="Energy window in meV (optional)",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=401,
        help="Number of energy points (default: 401)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="magnon_dos.png",
        help="Output filename for plot (default: magnon_dos.png)",
    )


def parse_common_args(args) -> MagnonParameters:
    """Extract common parameters from parsed args."""
    return MagnonParameters(
        path=args.path,
        Jiso=getattr(args, "Jiso", True),
        Jani=getattr(args, "Jani", True),
        DMI=getattr(args, "DMI", True),
        SIA=getattr(args, "SIA", True),
        Q=args.Q,
        uz_file=args.uz_file,
        n=getattr(args, "n", None),
        spin_conf_file=args.spin_conf_file,
        show=args.show,
    )
