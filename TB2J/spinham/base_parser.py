import numpy as np
from ase.atoms import Atoms

class BaseSpinModelParser(object):
    """
    SpinModelParser: a general model for spin model file parser.
    """

    def __init__(self, fname):
        self._fname = fname
        self.atoms = None
        self.damping_factors = []
        self.gyro_ratios = []
        self.index_spin = []
        self.cell = None
        self.zions = []
        self.masses = []
        self.positions = []
        self.spinat = []
        self._exchange = {}
        self._dmi = {}
        self._bilinear = {}
        self._parse(fname)
        self.lattice = Atoms(
            positions=self.positions, masses=self.masses, cell=self.cell)

    def _parse(self, fname):
        raise NotImplementedError("parse function not implemented yet")

    def get_atoms(self):
        return self.atoms

    def _spin_property(self, prop):
        return [
            prop[i] for i in range(len(self.index_spin))
            if self.index_spin[i] > 0
        ]

    @property
    def spin_positions(self):
        return np.array(self._spin_property(self.positions), dtype='float')

    @property
    def spin_zions(self):
        return np.array(self._spin_property(self.zions), dtype='int')

    @property
    def spin_spinat(self):
        return np.array(self._spin_property(self.spinat), dtype='float')

    @property
    def spin_damping_factors(self):
        return np.array(
            self._spin_property(self.damping_factors), dtype='float')

    @property
    def spin_gyro_ratios(self):
        return np.array(self._spin_property(self.gyro_ratios), dtype='float')

    def get_index_spin(self):
        return self.index_spin

    def exchange(self, isotropic=True):
        if isotropic:
            iso_jdict = {}
            for key, val in self._exchange.items():
                iso_jdict[key] = val[0]
            return iso_jdict
        else:
            return self._exchange

    @property
    def dmi(self):
        return self._dmi

    @property
    def has_exchange(self):
        return bool(len(self._exchange))

    @property
    def has_dmi(self):
        return bool(len(self._dmi))

    @property
    def has_bilinear(self):
        return bool(len(self._bilinear))

