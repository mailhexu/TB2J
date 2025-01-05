import argparse
from dataclasses import dataclass

import yaml

__all__ = ["ExchangeParams", "add_exchange_args_to_parser", "parser_argument_to_dict"]


@dataclass
class ExchangeParams:
    """
    A class to store the parameters for exchange calculation.
    """

    efermi: float
    basis = []
    magnetic_elements = []
    include_orbs = {}
    _kmesh = [4, 4, 4]
    emin: float = -15
    emax: float = 0.05
    nz: int = 100
    exclude_orbs = []
    ne: int = 0
    Rcut: float = None
    _use_cache: bool = False
    nproc: int = 1
    description: str = ""
    write_density_matrix: bool = False
    orb_decomposition: bool = False
    output_path: str = "TB2J_results"
    mae_angles = None
    orth = False
    ibz = False

    def __init__(
        self,
        efermi=-10.0,
        basis=None,
        magnetic_elements=None,
        include_orbs=None,
        kmesh=[4, 4, 4],
        emin=-15,
        emax=0.00,
        nz=100,
        ne=None,
        Rcut=None,
        use_cache=False,
        nproc=1,
        description="",
        write_density_matrix=False,
        orb_decomposition=False,
        output_path="TB2J_results",
        exclude_orbs=[],
        mae_angles=None,
        orth=False,
        ibz=False,
    ):
        self.efermi = efermi
        self.basis = basis
        # self.magnetic_elements = magnetic_elements
        # self.include_orbs = include_orbs
        self.magnetic_elements, self.include_orbs = self.set_magnetic_elements(
            magnetic_elements, include_orbs
        )
        self._kmesh = kmesh
        self.emin = emin
        self.emax = emax
        self.nz = nz
        self.exclude_orbs = exclude_orbs
        self.ne = ne
        self.Rcut = Rcut
        self._use_cache = use_cache
        self.nproc = nproc
        self.description = description
        self.write_density_matrix = write_density_matrix
        self.orb_decomposition = orb_decomposition
        self.output_path = output_path
        self.mae_angles = mae_angles
        self.orth = orth
        self.ibz = ibz

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def save_to_yaml(self, fname):
        with open(fname, "w") as myfile:
            yaml.dump(self.__dict__, myfile)

    def set_magnetic_elements(self, magnetic_elements, include_orbs):
        # magnetic_elements = exargs.pop("magnetic_elements")
        # include_orbs = exargs.pop("include_orbs")
        if include_orbs is None:
            include_orbs = {}
        if isinstance(magnetic_elements, str):
            magnetic_elements = [magnetic_elements]
        for element in magnetic_elements:
            if "_" in element:
                elem = element.split("_")[0]
                orb = element.split("_")[1:]
                include_orbs[elem] = orb
            else:
                include_orbs[element] = None

        magnetic_elements = list(include_orbs.keys())
        return magnetic_elements, include_orbs


def add_exchange_args_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--elements",
        help="elements to be considered in Heisenberg model",
        default=None,
        type=str,
        nargs="*",
    )

    parser.add_argument(
        "--spinor",
        help="whether the Wannier functions are spinor. Default: False",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--rcut",
        help="cutoff of spin pair distance. The default is to calculate all commensurate R point to the k mesh.",
        default=None,
        type=float,
    )
    parser.add_argument("--efermi", help="Fermi energy in eV", default=None, type=float)
    parser.add_argument(
        "--ne",
        help="number of electrons in the unit cell. If not given, TB2J will use the fermi energy to compute it.",
    )
    parser.add_argument(
        "--kmesh",
        help="kmesh in the format of kx ky kz",
        type=int,
        nargs="*",
        default=[5, 5, 5],
    )
    parser.add_argument(
        "--emin",
        help="energy minimum below efermi, default -14 eV",
        type=float,
        default=-14.0,
    )
    parser.add_argument(
        "--emax",
        help="energy maximum above efermi, default 0.0 eV",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--nz",
        help="number of steps for semicircle contour, default: 100",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--cutoff",
        help="The minimum of J amplitude to write, (in eV), default is 1e-5 eV",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--exclude_orbs",
        help="the indices of wannier functions to be excluded from magnetic site. counting start from 0",
        default=[],
        type=int,
        nargs="+",
    )

    parser.add_argument(
        "--np",
        "--nproc",
        help="number of cpu cores to use in parallel, default: 1",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--use_cache",
        help="whether to use disk file for temporary storing wavefunctions and hamiltonian to reduce memory usage. Default: False",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--description",
        help="add description of the calculatiion to the xml file. Essential information, like the xc functional, U values, magnetic state should be given.",
        type=str,
        default="Calculated with TB2J.",
    )

    parser.add_argument(
        "--orb_decomposition",
        default=False,
        action="store_true",
        help="whether to do orbital decomposition in the non-collinear mode.",
    )

    parser.add_argument(
        "--output_path",
        help="The path of the output directory, default is TB2J_results",
        type=str,
        default="TB2J_results",
    )
    parser.add_argument(
        "--write_dm",
        help="whether to write density matrix",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--orth",
        help="whether to use lowdin orthogonalization before diagonalization (for testing only)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--ibz",
        help=" use irreducible k-points in the Brillouin zone. (Note: only for computing total MAE).",
        action="store_true",
        default=False,
    )

    return parser


def parser_argument_to_dict(args) -> dict:
    return {
        "efermi": args.efermi,
        "magnetic_elements": args.elements,
        "kmesh": args.kmesh,
        "emin": args.emin,
        "emax": args.emax,
        "nz": args.nz,
        "exclude_orbs": args.exclude_orbs,
        "ne": args.ne,
        "Rcut": args.rcut,
        "use_cache": args.use_cache,
        "nproc": args.np,
        "description": args.description,
        "write_density_matrix": args.write_dm,
        "orb_decomposition": args.orb_decomposition,
        "output_path": args.output_path,
        "orth": args.orth,
    }
