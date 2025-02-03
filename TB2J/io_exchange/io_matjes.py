import os

import numpy as np

from TB2J.utils import symbol_number


def write_matjes(cls, path="TB2J_results/Matjes"):
    if not os.path.exists(path):
        os.makedirs(path)
    inputfname = os.path.join(path, "matjes.in")
    with open(inputfname, "w") as myfile:
        _write_lattice_supercell(cls, myfile)
        _write_atoms(cls, myfile)
        _write_magnetic_interactions(cls, myfile)
        _write_magnetic_anisotropy(cls, myfile)
        # _write_dmi(cls, myfile)
        _write_exchange_tensor(cls, myfile)


def _write_lattice_supercell(cls, myfile):
    myfile.write("# Lattice and supercell\n")
    myfile.write(
        "Periodic_log .T. .T. .T.       periodic boundary conditions along vector 1, 2 and 3\n"
    )
    myfile.write("Nsize 8 8 8                  size of the supercell\n")
    myfile.write("alat 1.0 1.0 1.0               lattice parameter\n")
    myfile.write(
        "lattice                         lattice vector should be orthogonal or expressed in cartesian\n"
    )
    try:
        unitcell = cls.atoms.get_cell().reshape((3, 3))
    except Exception:
        unitcell = cls.atoms.get_cell().array.reshape((3, 3))
    myfile.write(
        f"{unitcell[0][0]} {unitcell[0][1]} {unitcell[0][2]}     # a_11 a_12 a_1      first lattice vector in line (does not need to be normalize)\n"
    )
    myfile.write(
        f"{unitcell[1][0]} {unitcell[1][1]} {unitcell[1][2]}      # a_21 a_22 a_23    second lattice vector in line (does not need to be normalize)\n"
    )
    myfile.write(
        f"{unitcell[2][0]} {unitcell[2][1]} {unitcell[2][2]}                          third lattice vector in line\n"
    )


def get_atoms_info(atoms, spinat, symmetry=False):
    if symmetry:
        raise NotImplementedError("Symmetry not implemented yet")
    else:
        symnum = symbol_number(atoms)
        atom_types = list(symnum.keys())
        magmoms = np.linalg.norm(spinat, axis=1)
        masses = atoms.get_masses()
        tags = [i for i in range(len(atom_types))]
    return atom_types, magmoms, masses, tags


def _write_atoms(cls, myfile, symmetry=False):
    myfile.write("\n")
    myfile.write("# Atoms\n")
    atom_types, magmoms, masses, tags = get_atoms_info(
        cls.atoms, cls.spinat, symmetry=symmetry
    )

    myfile.write(f"atomtypes  {len(atom_types)}       Number of types atom\n")
    for i, atom_type in enumerate(atom_types):
        m = magmoms[i]
        mass = masses[i]
        myfile.write(
            f"{atom_type} {m} {mass} F 0 # atom type: (name, mag. moment, mass, charge, displacement, number TB-orb.)\n"
        )

    myfile.write(
        f"\natoms {len(cls.atoms)}          positions of the atom in the unit cell\n"
    )
    for i, atom in enumerate(cls.atoms):
        spos = cls.atoms.get_scaled_positions()[i]
        myfile.write(f"{atom_types[tags[i]]} {spos[0]}, {spos[1]}, {spos[2]}  \n")


def _write_magnetic_interactions(cls, myfile):
    myfile.write("\nThe Hamiltonian\n")
    myfile.write("# Magnetic interactions\n")
    myfile.write("\nmagnetic_J\n")
    for key, val in cls.exchange_Jdict.items():
        R, i, j = key
        myfile.write(
            f"{i+1} {j+1} {R[0]} {R[1]} {R[2]} {val}     # between atoms type {i+1} and {j+1}, shell {R}, amplitude in eV  \n"
        )
    myfile.write(
        "\nc_H_J 0.5        apply 1/2 in front of the sum of the exchange energy - default is -1\n"
    )


def _write_magnetic_anisotropy(cls, myfile):
    myfile.write("\nmagnetic_anisotropy \n")
    if cls.k1 is None:
        return
    else:
        for i, k1 in enumerate(cls.k1):
            myfile.write(
                f"{i+1} {cls.k1dir[i][0]} {cls.k1dir[i][1]} {cls.k1dir[i][2]} {k1}     anisotropy of atoms type {i+1}, direction {cls.k1dir[i][0]} {cls.k1dir[i][1]} {cls.k1dir[i][2]} and amplitude in eV\n"
            )
        myfile.write("\nc_H_ani 1.0\n")


def _write_dmi(cls, myfile):
    myfile.write("\nmagnetic_D\n")
    for key, val in cls.dmi_ddict.items():
        R, i, j = key
        myfile.write(
            f"{i+1} {j+1} {R[0]} {R[1]} {R[2]} {val}         #between atoms type {i+1} and {j+1}, mediated by atom type {R}, shell {R}, amplitude in eV\n"
        )
    myfile.write(
        "\nc_H_D   -1.0             # coefficients to put in from of the sum - default is -1\n"
    )


def _write_exchange_tensor(cls, myfile):
    myfile.write(
        "\nmagnetic_r2_tensor #Exchange tensor elements, middle 9 entries: xx, xy, xz, yx, etc. (in units of eV) and direction in which it should be applied\n"
    )
    for key, val in cls.Jani_dict.items():
        R, i, j = key
        myfile.write(
            f"{i+1} {j+1} {R[0]} {R[1]} {R[2]} {' '.join([str(x) for x in val.flatten()])} 1.0 0.0 0.0 #First NN (J1intra: 0 GPa)\n"
        )
    myfile.write(
        "\nc_H_Exchten -1         apply 1/2 in front of the sum of the exchange tensor energy - default is -1\n"
    )
