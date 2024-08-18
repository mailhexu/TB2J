#!/usr/vin/env python3
# -*- coding: utf-8 -*-
"""
Parser for the abacus orbital file
"""

from dataclasses import dataclass
from pathlib import Path

from TB2J.utils import symbol_number_list


@dataclass
class AbacusOrbital:
    """
    Orbital class
    """

    iatom: int = 0
    sym: str = ""
    spin: int = 0
    element: str = ""
    n: int = 0
    l: int = 0
    m: int = 0
    z: int = 0


def parse_abacus_orbital(fname):
    """
    parse the abacus orbital file
    """
    orbs = []
    with open(fname, "r") as myfile:
        line = myfile.readline()
        line = myfile.readline()
        while line.strip() != "":
            seg = line.split()
            iatom, element, l, m, z, sym = seg
            iatom = int(iatom)
            ispin = 0
            n = 0
            l = int(l)
            m = int(m)
            z = int(z)
            orbs.append(AbacusOrbital(iatom, sym, ispin, element, n, l, m, z))
            line = myfile.readline()
    return orbs


def bset_to_symnum_type(bset, atoms):
    """
    convert the basis set to symbol number type
    """
    slist = symbol_number_list(atoms)
    result = []
    for b in bset:
        result.append((slist[b.iatom], b.sym, b.spin))
    return result


def test_parse_abacus_orbital():
    """
    test the parser
    """
    outpath = "/Users/hexu/projects/TB2J_abacus/abacus-tb2j-master/abacus_example/case_Sr2Mn2O6/1_no_soc/OUT.Sr2Mn2O6"
    orbs = parse_abacus_orbital(Path(outpath) / "Orbital")
    print(orbs)


if __name__ == "__main__":
    test_parse_abacus_orbital()
