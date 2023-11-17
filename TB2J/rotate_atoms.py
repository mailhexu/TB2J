#!/usr/bin/env python3
import copy
from ase.io import read, write
import numpy as np
from TB2J.tensor_rotate import Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz


def rotate_atom_xyz(atoms):
    """
    given a atoms, return:
    - 'z'->'x'
    - 'z'->'y'
    - 'z'->'z'
    """
    atoms_x = copy.deepcopy(atoms)
    atoms_x.rotate(90, "y", rotate_cell=True)
    atoms_y = copy.deepcopy(atoms)
    atoms_y.rotate(90, "x", rotate_cell=True)
    atoms_z = atoms
    return atoms_x, atoms_y, atoms_z


def rotate_atom_spin_one_rotation(atoms, Rotation):
    """
    roate the spin of atoms with one rotation operator
    """
    magmoms = np.array(atoms.get_initial_magnetic_moments())
    if len(magmoms.shape) == 1:
        m = np.zeros((len(magmoms), 3), dtype=float)
        m[:, 2] = magmoms
    else:
        m = magmoms
    rotated_m = Rotation.apply(m)
    atoms_copy = copy.deepcopy(atoms)
    atoms_copy.set_initial_magnetic_moments(None)
    atoms_copy.set_initial_magnetic_moments(rotated_m)
    return atoms_copy


def rotate_atom_spin(atoms):
    """
    given a atoms, return the atoms with rotated spin:
    - 'z'->'x'
    - 'z'->'y'
    - 'z'->'z'
    """
    return [rotate_atom_spin_one_rotation(atoms, R) for R in [Rzx, Rzy, Rzz]]


def rotate_atom_spin_legacy(atoms):
    """
    given a atoms, return the atoms with rotated spin:
    - 'z'->'x'
    - 'z'->'y'
    - 'z'->'z'
    """
    magmoms = np.array(atoms.get_initial_magnetic_moments())
    if len(magmoms.shape) == 1:
        m = np.zeros((len(magmoms), 3), dtype=float)
        m[:, 2] = magmoms
    else:
        m = magmoms
    atoms_copy = copy.deepcopy(atoms)
    atoms.set_initial_magnetic_moments(None)

    m_x = copy.deepcopy(m)
    m_x[:, 0] = m[:, 2]
    m_x[:, 1] = m[:, 1]
    m_x[:, 2] = m[:, 0]
    atoms_x = copy.deepcopy(atoms_copy)
    atoms_x.set_initial_magnetic_moments(m_x)

    m_y = copy.deepcopy(m)
    m_y[:, 0] = m[:, 0]
    m_y[:, 1] = m[:, 2]
    m_y[:, 2] = m[:, 1]
    atoms_y = copy.deepcopy(atoms_copy)
    atoms_y.set_initial_magnetic_moments(m_y)

    m_z = copy.deepcopy(m)
    m_z[:, 0] = m[:, 0]
    m_z[:, 1] = m[:, 1]
    m_z[:, 2] = m[:, 2]
    atoms_z = copy.deepcopy(atoms_copy)
    atoms_z.set_initial_magnetic_moments(m_z)
    return atoms_x, atoms_y, atoms_z


def check_ftype(ftype):
    if ftype in ["cif"]:
        print("=" * 40)
        print("WARNING!!!!!")
        print(
            f"{ftype} type does not contains the cell matrix explicitly. Therefore the outputted files does not give the rotation properly. Please use another format."
        )
        print("=" * 40)


def rotate_xyz(fname, ftype="xyz"):
    check_ftype(ftype)
    atoms = read(fname)
    atoms.set_pbc(True)

    atoms_x, atoms_y, atoms_z = rotate_atom_xyz(atoms)

    fname_x = f"atoms_x.{ftype}"
    fname_y = f"atoms_y.{ftype}"
    fname_z = f"atoms_z.{ftype}"

    write(fname_x, atoms_x)
    write(fname_y, atoms_y)
    write(fname_z, atoms_z)
    print(f"The output has been written to {fname_x}, {fname_y}, {fname_z}")
