#!/usr/bin/env python3
import copy
from ase.io import read, write

def rotate_atom_xyz(atoms):
    """
    given a atoms, return:
    - 'z'->'x'
    - 'z'->'y'
    - 'z'->'z'
    """
    atoms_x = copy.deepcopy(atoms)
    atoms_x.rotate(90, 'y', rotate_cell=True)
    atoms_y = copy.deepcopy(atoms)
    atoms_y.rotate(90, 'x', rotate_cell=True)
    atoms_z = atoms
    return atoms_x, atoms_y, atoms_z


def check_ftype(ftype):
    if ftype in ['cif']:
        print("="*40)
        print("WARNING!!!!!")
        print(f'{ftype} type does not contains the cell matrix explicitly. Therefore the outputted files does not give the rotation properly. Please use another format.')
        print("="*40)

def rotate_xyz(fname, ftype='xyz'):
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


