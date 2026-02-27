"""Shared parameters for magnon band and DOS calculations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import tomli
import tomli_w

if TYPE_CHECKING:
    from TB2J.magnon.magnon3 import Magnon


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
    spin_conf: Optional[List[List[float]]] = None
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

        if self.spin_conf is not None and self.spin_conf_file is not None:
            raise ValueError(
                "spin_conf and spin_conf_file are mutually exclusive. "
                "Please specify only one."
            )

        if self.spin_conf is not None:
            for i, vec in enumerate(self.spin_conf):
                if len(vec) != 3:
                    raise ValueError(
                        f"spin_conf[{i}] must have 3 elements (mx, my, mz), got {len(vec)}"
                    )

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
        help=(
            "Path to file containing magnetic moments for each spin (nspin×3 array). "
            "Default: Uses magnetic moments from TB2J results. "
            "Use cases: (1) Quantum theory - use integer 2S values (e.g., 3 for Cr³⁺ instead of 3.1). "
            "(2) Different state - explore FM from AFM DFT. "
            "(3) Rotated spins - change spin direction. "
            "Mutually exclusive with --spin-conf."
        ),
    )
    parser.add_argument(
        "--spin-conf",
        type=float,
        nargs="+",
        metavar="M",
        help=(
            "Spin configuration as flat list: mx1 my1 mz1 mx2 my2 mz2 ... "
            "Values are in μB (Bohr magneton). For spin S, use 2S μB. "
            "Example: --spin-conf 0 0 3 0 0 -3 for two antiparallel spins. "
            "Default: Uses magnetic moments from TB2J results. "
            "Mutually exclusive with --spin-conf-file."
        ),
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
    spin_conf = None
    if hasattr(args, "spin_conf") and args.spin_conf:
        if len(args.spin_conf) % 3 != 0:
            raise ValueError(
                f"--spin-conf must have 3n values (mx my mz for each spin), "
                f"got {len(args.spin_conf)} values"
            )
        spin_conf = [
            args.spin_conf[i : i + 3] for i in range(0, len(args.spin_conf), 3)
        ]

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
        spin_conf=spin_conf,
        show=args.show,
    )


def _load_uz(params: MagnonParameters, nspin: int) -> np.ndarray:
    """Load quantization axes from file or return default.

    Parameters
    ----------
    params : MagnonParameters
        Parameters containing uz_file path
    nspin : int
        Number of spins in the system (for validation)

    Returns
    -------
    np.ndarray
        Quantization axes array of shape (n, 3)
    """
    if params.uz_file is not None:
        uz_file = params.uz_file
        if not Path(uz_file).is_absolute():
            uz_file = str(Path(params.path) / uz_file)
        uz = np.loadtxt(uz_file)
        if uz.shape[1] != 3:
            raise ValueError(
                f"Quantization axes file should contain a natom×3 array. Got shape {uz.shape}"
            )
        if uz.shape[0] != nspin:
            raise ValueError(
                f"Number of spins in uz file ({uz.shape[0]}) does not match the system ({nspin})"
            )
        return uz
    else:
        return np.array([[0, 0, 1]], dtype=float)


def _load_spin_conf(params: MagnonParameters, nspin: int) -> Optional[np.ndarray]:
    """Load spin configuration from params, file, or return None.

    Parameters
    ----------
    params : MagnonParameters
        Parameters containing spin_conf or spin_conf_file
    nspin : int
        Number of spins in the system (for validation)

    Returns
    -------
    Optional[np.ndarray]
        Spin configuration array of shape (nspin, 3) or None
    """
    if params.spin_conf is not None:
        magmoms = np.array(params.spin_conf, dtype=float)
        if magmoms.shape[1] != 3:
            raise ValueError(
                f"spin_conf must have 3 columns (mx, my, mz). Got shape {magmoms.shape}"
            )
        if magmoms.shape[0] != nspin:
            raise ValueError(
                f"Number of spins in spin_conf ({magmoms.shape[0]}) does not match the system ({nspin})"
            )
        return magmoms
    elif params.spin_conf_file is not None:
        spin_conf_file = params.spin_conf_file
        if not Path(spin_conf_file).is_absolute():
            spin_conf_file = str(Path(params.path) / spin_conf_file)
        magmoms = np.loadtxt(spin_conf_file)
        if magmoms.shape[1] != 3:
            raise ValueError(
                f"Spin configuration file should contain a nspin×3 array. Got shape {magmoms.shape}"
            )
        if magmoms.shape[0] != nspin:
            raise ValueError(
                f"Number of spins in spin configuration file ({magmoms.shape[0]}) does not match the system ({nspin})"
            )
        return magmoms
    else:
        return None


def prepare_magnon_from_params(params: MagnonParameters) -> "Magnon":
    """Load Magnon object from TB2J results and configure with parameters.

    This function handles:
    - Loading exchange parameters (Jiso, Jani, DMI, SIA)
    - Setting propagation vector Q and rotation axis n
    - Loading quantization axes uz (from file or default)
    - Loading spin configuration (from params, file, or None)
    - Calling set_reference() to configure the Magnon object

    Parameters
    ----------
    params : MagnonParameters
        Parameters for the calculation

    Returns
    -------
    Magnon
        Configured Magnon instance ready for band/DOS calculations

    Raises
    ------
    FileNotFoundError
        If TB2J results path does not exist
    ValueError
        If spin configuration or uz file has wrong shape
    """
    from TB2J.magnon.magnon3 import Magnon

    if not Path(params.path).exists():
        raise FileNotFoundError(f"TB2J results not found at {params.path}")

    print(f"Loading exchange parameters from {params.path}...")
    magnon = Magnon.from_TB2J_results(
        path=params.path,
        Jiso=params.Jiso,
        Jani=params.Jani,
        DMI=params.DMI,
        SIA=params.SIA,
    )

    Q = [0, 0, 0] if params.Q is None else params.Q
    n = [0, 0, 1] if params.n is None else params.n

    uz = _load_uz(params, magnon.nspin)
    magmoms = _load_spin_conf(params, magnon.nspin)

    magnon.set_reference(Q, uz, n, magmoms)

    return magnon
