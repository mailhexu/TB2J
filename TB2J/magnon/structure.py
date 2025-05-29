from warnings import warn

import numpy as np

valid_symbols = [
    # 0
    "Ae",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def get_attribute_array(array, attribute, dtype=float):
    try:
        the_array = np.array(array, dtype=dtype)
    except (IndexError, ValueError, TypeError) as err:
        typename = dtype.__name__
        raise type(err)(
            f"'{attribute}' must be an arraylike object of '{typename}' elements."
        )

    return the_array


def validate_symbols(value):
    try:
        values_list = value.split()
    except AttributeError:
        try:
            values_list = list(value)
        except (ValueError, TypeError):
            raise TypeError("'elements' must be an iterable of 'str' or 'int' entries.")

    if all(isinstance(s, str) for s in values_list):
        symbols = values_list
        for symbol in symbols:
            if symbol not in valid_symbols:
                raise ValueError(f"Unrecognized element '{symbol}'.")
    elif all(isinstance(i, (int, np.integer)) for i in values_list):
        if any(i < 0 for i in values_list):
            raise ValueError("Atomic numbers must be positive.")
        try:
            symbols = [valid_symbols[i] for i in values_list]
        except IndexError:
            raise ValueError("Atomic number exceeds 118.")
    else:
        raise ValueError("'elements' must be an iterable of 'str' or 'int' entries.")

    return symbols


class BaseMagneticStructure:
    def __init__(
        self,
        atoms=None,
        cell=None,
        elements=None,
        positions=None,
        magmoms=None,
        pbc=None,
        collinear=True,
    ):
        if atoms is not None:
            if any(arg is not None for arg in [cell, elements, positions]):
                warn(
                    "WARNING: 'atoms' overrides the 'cell', 'elements', and 'positions' arguments."
                )
            cell = atoms.cell.array
            elements = atoms.numbers
            positions = atoms.get_scaled_positions()
            pbc = atoms.pbc if pbc is None else pbc
        else:
            if cell is None:
                cell = np.zeros((3, 3))
            if elements is None:
                elements = ()
            if positions is None:
                positions = np.zeros((len(elements), 3))
            if pbc is None:
                pbc = (True, True, True)
        if magmoms is None:
            magmoms_shape = positions.shape[0] if collinear else positions.shape
            magmoms = np.zeros(magmoms_shape)

        self.cell = cell
        self.elements = elements
        self.positions = positions
        self.collinear = collinear
        self.magmoms = magmoms
        self.pbc = pbc

        self._Q = None
        self._n = np.array([0, 0, 1])

    @property
    def cell(self):
        return self._cell

    @property
    def reciprocal_cell(self):
        return 2 * np.pi * np.linalg.inv(self._cell)

    @cell.setter
    def cell(self, value):
        cell_array = get_attribute_array(value, "cell")
        if cell_array.shape != (3, 3):
            raise ValueError("'cell' must have a (3, 3) shape.")

        self._cell = cell_array

    @property
    def elements(self):
        return self._symbols

    @elements.setter
    def elements(self, value):
        symbols = validate_symbols(value)
        self._symbols = symbols

    @property
    def numbers(self):
        return [valid_symbols.index(symbol) for symbol in self._symbols]

    @property
    def positions(self):
        return self._positions

    @property
    def cartesian_positions(self):
        return self._positions @ self._cell

    @positions.setter
    def positions(self, value):
        natoms = len(self.elements)
        posarray = get_attribute_array(value, "positions")
        if posarray.shape != (natoms, 3):
            raise ValueError(
                "'positions' must have a (natoms, 3) shape, where natoms is the length of 'elements'. Make sure to set 'elements' first."
            )

        self._positions = posarray

    @property
    def magmoms(self):
        return self._magmoms

    @magmoms.setter
    def magmoms(self, value):
        magarray = get_attribute_array(value, "magmoms")
        if self.collinear and magarray.shape != (len(self._positions),):
            raise ValueError(
                "'magmoms' must be a 1dim array with the same length as 'positions'. Make sure to set 'positions' first."
            )
        elif not self.collinear and magarray.shape != self._positions.shape:
            raise ValueError(
                "'magmoms' must have the same shape as 'positions'. Make sure to set 'positions' first."
            )

        self._magmoms = magarray

    @property
    def pbc(self):
        return self._pbc

    @pbc.setter
    def pbc(self, value):
        try:
            periodic_boundary_conditions = tuple(bool(b) for b in value)
        except (ValueError, TypeError):
            raise TypeError("'pbc' must be an iterable of 'bool' entries.")

        if len(periodic_boundary_conditions) != 3:
            raise ValueError("'pbc' must be of length 3.")

        self._pbc = periodic_boundary_conditions

    @property
    def propagation_vector(self):
        return self._Q

    @propagation_vector.setter
    def propagation_vector(self, value):
        Q = get_attribute_array(value, "propagation_vector")
        if Q.shape != (3,):
            raise ValueError(
                "The 'propagation_vector' must be a vector with 3 numbers."
            )

        self._Q = Q

    @property
    def normal_vector(self):
        return self._n

    @normal_vector.setter
    def normal_vector(self, value):
        n = get_attribute_array(value, "normal_vector")
        if n.shape != (3,):
            raise ValueError("The 'normal_vector' must be a vector with 3 numbers.")

        self._n = n

    def to_ase(self):
        from ase.atoms import Atoms

        atoms = Atoms(
            cell=self.cell,
            positions=self.cartesian_positions,
            numbers=[valid_symbols.index(symbol) for symbol in self.elements],
            pbc=self.pbc,
        )

        return atoms

    @classmethod
    def from_structure_file(
        cls, filename, magmoms=None, collinear=True, pbc=(True, True, True)
    ):
        from ase.io import read

        atoms = read(filename)
        atoms.pbc = pbc
        obj = cls(atoms=atoms, magmoms=magmoms, collinear=collinear)

        return obj
