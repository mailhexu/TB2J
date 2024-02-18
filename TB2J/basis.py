#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class for basis set
"""
from typing import Any
import numpy as np
import dataclasses
from TB2J.utils import symbol_number


@dataclasses.dataclass
class Basis:
    iatom: int = 0
    symbol: str = "unamed"
    spin: int = 0

    def __str__(self):
        return f"{self.iatom}|{self.symbol}|{self.spin}"

    def to_symnum_type(self, atoms=None, symbol_number=None):
        if symbol_number is None:
            symbol_number = symbol_number(atoms)
        return f"{self.iatom}|{self.symbol}|{self.spin}"


@dataclasses.dataclass
class NAOBasis(Basis):
    n: int = 0
    l: int = 0
    m: int = 0
    zeta: int = 0
    element: str = "unknown"

    def __str__(self) -> str:
        return super().__str__()


# @dataclasses.dataclass
#  inherit from tuple, the elements are basis.


class BasisSet(list):
    def set_atoms(self, atoms):
        self._atoms = atoms

    def get_iorbs_of_atom(self, iatom: int):
        """
        get the index of orbitals of an atom
        """
        return [i for i, basis in enumerate(self) if basis.iatom == iatom]

    def get_iorbs_of_atom_spin(self, iatom: int, spin: int):
        """
        get the index of orbitals of an atom
        """
        return [
            i
            for i, basis in enumerate(self)
            if basis.iatom == iatom and basis.spin == spin
        ]

    def find_iorbs(self, key=None):
        """
        find the index of orbitals of an atom
        """
        return [i for i, basis in enumerate(self) if key(basis)]
