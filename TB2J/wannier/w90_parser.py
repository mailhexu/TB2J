from collections import defaultdict
import math
import re
import numpy as np
from ase.io import read
from ase.atoms import Atoms
from ase.units import Angstrom, Bohr
from .w90_tb_parser import parse_tb_file
from TB2J.utils import split_symbol_number

unit_dict = {"ANG": Angstrom, "BOHR": Bohr}


def parse_xyz(fname):
    """
    wannier90 xyz file parser.

    :param fname: relative or absolute path to the xyz file.  str.
    """
    atoms = read(fname)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    wann_pos = []
    atoms_pos = []
    atoms_symbols = []
    for s, x in zip(symbols, pos):
        if s == "X":
            wann_pos.append(x)
        else:
            atoms_symbols.append(s)
            atoms_pos.append(x)
    return np.array(wann_pos), atoms_symbols, np.array(atoms_pos)


def parse_ham(fname="wannier90_hr.dat", cutoff=None):
    """
    wannier90 hr file parser.

    :param fname: relative or absolute path to the hamiltonian file.  str.

    :param cutoff: the energy cutoff.  None | number | list (of Emin, Emax).
    """
    with open(fname, "r") as myfile:
        lines = myfile.readlines()
    n_wann = int(lines[1].strip())
    n_R = int(lines[2].strip())

    # The lines of degeneracy of each R point. 15 per line.
    nline = int(math.ceil(n_R / 15.0))
    dlist = []
    for i in range(3, 3 + nline):
        d = map(float, lines[i].strip().split())
        dlist += d
    R_degens = np.array(dlist, dtype=int)
    H_mnR = defaultdict(lambda: np.zeros((n_wann, n_wann), dtype=complex))
    for i in range(3 + nline, 3 + nline + n_wann**2 * n_R):
        t = lines[i].strip().split()
        R = tuple(map(int, t[:3]))
        m, n = map(int, t[3:5])
        m = m - 1
        n = n - 1
        H_real, H_imag = map(float, t[5:])
        val = H_real + 1j * H_imag
        if m == n and np.linalg.norm(R) < 0.001:
            H_mnR[R][m, n] = val / 2.0
        elif cutoff is not None:
            if abs(val) > cutoff:
                H_mnR[R][m, n] = val / 2.0
        else:
            H_mnR[R][m, n] = val / 2.0
    return n_wann, H_mnR, R_degens


def parse_cell(fname, unit=Angstrom):
    """
    wannier90 hr cell parser.

    :param fname: relative or absolute path to the file.  str.
    """

    uc_regex = re.compile(
        r"BEGIN\s+UNIT_CELL_CART\s+"
        r"(?P<units>BOHR|ANG)?"
        r"(?P<cell>.+)"
        r"END\s+UNIT_CELL_CART\s+",
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    with open(fname) as myfile:
        text = myfile.read()
    match = uc_regex.search(text)
    if match is None:
        raise Exception(f"Cannot find unit cell information from {fname}")

    cell = np.fromstring(match.group("cell").strip(), sep="\n").reshape((3, 3))

    if match.group("units") is not None:
        factor = {"ANG": Angstrom, "BOHR": Bohr}[match.group("units").upper()]
    else:
        factor = Angstrom
    cell = cell * factor / unit
    return cell


def parse_atoms(fname):
    """
    wannier90 hr atoms parser.

    :param fname: relative or absolute path to the file.  str.
    """
    cell = parse_cell(fname)
    atoms_regex = re.compile(
        r"BEGIN\s+ATOMS_(?P<suffix>(FRAC)|(CART))\s+"
        r"(?P<units>BOHR|ANG)?"
        r"(?P<atoms>.+)"
        r"END\s+ATOMS_(?P=suffix)\s+",
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )

    with open(fname, "r") as f:
        match = atoms_regex.search(f.read())
        if match is None:
            raise Exception(f"Cannot read atomic structure from {fname}")

    symbols = []
    taus = []
    tags = []

    for line in match.group("atoms").strip().splitlines():
        symbol = line.split()[0].strip()
        symbol = symbol[0].upper() + symbol[1:]
        symbol, tag = split_symbol_number(symbol)
        symbols.append(symbol)
        tags.append(tag)
        taus.append(np.array(list(map(float, line.split()[1:]))))

    taus = np.asarray(taus)
    if match.group("suffix").upper() == "FRAC":
        atoms = Atoms(symbols=symbols, cell=cell, scaled_positions=taus)
        atoms.set_tags(tags)
    else:
        if match.group("units") is not None:
            factor = unit_dict[match.group("units").upper()]
        else:
            factor = Angstrom
        taus = taus * factor
        atoms = Atoms(symbols=symbols, cell=cell, positions=taus)
    return atoms


def parse_tb(fname):
    """
    return the wannier center, number of wannier functions and the Hamiltonian from the wannier90 _tb.dat file.
    """
    result = parse_tb_file(fname)
    return result["centers"], result["n_wann"], result["H"], result["Rdegens"]
