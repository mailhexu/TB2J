from dataclasses import dataclass

import yaml


@dataclass
class ExchangeParams:
    """
    A class to store the parameters for exchange calculation.
    """

    efermi: float
    basis: list = []
    magnetic_elements: list = []
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

    def __init__(
        self,
        efermi=-10.0,
        basis=None,
        magnetic_elements=None,
        include_orbs=None,
        kmesh=[4, 4, 4],
        emin=-15,
        emax=0.05,
        nz=100,
        exclude_orbs=[],
        ne=None,
        Rcut=None,
        use_cache=False,
        nproc=1,
        description="",
        write_density_matrix=False,
        orb_decomposition=False,
        output_path="TB2J_results",
    ):
        self.efermi = efermi
        self.basis = basis
        self.magnetic_elements = magnetic_elements
        self.include_orbs = include_orbs
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

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def save_to_yaml(self, fname):
        with open(fname, "w") as myfile:
            yaml.dump(self.__dict__, myfile)
