import os

import numpy as np
from ase.units import nm

from TB2J.utils import symbol_number


def get_ind_shell(distance_dict, symprec=1e-5):
    """
    return a dictionary of shell index for each pair of atoms.
    The index of shell is the ith shortest distances between all magnetic atom pairs.
    """
    shell_dict = {}
    distances = np.array([x[1] for x in distance_dict.values()])
    distances_int = np.round(distances / symprec).astype(int)
    dint = sorted(np.unique(distances_int))
    dintdict = dict(zip(dint, range(len(dint))))
    for key, val in distance_dict.items():
        di = np.round(val[1] / symprec).astype(int)
        shell_dict[key] = dintdict[di]
    return shell_dict


def _write_symmetry(cls, path, fname="symmetry.in", symmetry=True):
    fname = os.path.join(path, fname)
    if not symmetry:
        with open(fname, "w") as myfile:
            myfile.write("1 \n")
            myfile.write("\n")
            myfile.write("! identity operation\n")
            myfile.write("1.0000000000 0.0000000000 0.0000000000\n")
            myfile.write("0.0000000000 1.0000000000 0.0000000000\n")
            myfile.write("0.0000000000 0.0000000000 1.0000000000\n")
            myfile.write("0.0000000000 0.0000000000 0.0000000000\n")
    else:
        raise NotImplementedError("Symmetry not implemented yet")


def write_matjes(cls, path="TB2J_results/Matjes", symmetry=False):
    if not os.path.exists(path):
        os.makedirs(path)
    inputfname = os.path.join(path, "input")
    with open(inputfname, "w") as myfile:
        _write_lattice_supercell(cls, myfile)
        _write_atoms(cls, myfile)
        _write_magnetic_interactions(cls, myfile)
        # _write_isotropic_exchange(cls, myfile, symmetry=symmetry)
        _write_magnetic_anisotropy(cls, myfile)
        # _write_dmi(cls, myfile)
        _write_exchange_tensor(cls, myfile, symmetry=symmetry)
        _write_parameters(cls, myfile, symmetry=symmetry)
    print("writting symmetries ")
    _write_symmetry(cls, path, fname="symmetries.in", symmetry=False)


def _write_parameters(cls, myfile, symmetry=False):
    if symmetry:
        myfile.write("cal_sym T\n")
        myfile.write("sym_mode 2\n")
    else:
        myfile.write("cal_sym F \n")
        myfile.write("sym_mode 0\n")


def _write_lattice_supercell(cls, myfile):
    myfile.write("# Lattice and supercell\n")
    myfile.write(
        "Periodic_log .T. .T. .T.      # periodic boundary conditions along vector 1, 2 and 3\n"
    )
    myfile.write("Nsize 8 8 8                 # size of the supercell\n")
    try:
        unitcell = cls.atoms.get_cell().reshape((3, 3))
    except Exception:
        unitcell = cls.atoms.get_cell().array.reshape((3, 3))
    uc_lengths = np.linalg.norm(unitcell, axis=1)
    unitcell /= uc_lengths[:, None]
    myfile.write(
        f"alat {uc_lengths[0]/nm}  {uc_lengths[1]/nm} {uc_lengths[2]/nm}              #lattice constant lengths\n"
    )
    myfile.write(
        "lattice                         #lattice vector should be orthogonal or expressed in cartesian\n"
    )
    myfile.write(
        f"{unitcell[0][0]} {unitcell[0][1]} {unitcell[0][2]}     # a_11 a_12 a_1      first lattice vector in line (does not need to be normalize)\n"
    )
    myfile.write(
        f"{unitcell[1][0]} {unitcell[1][1]} {unitcell[1][2]}      # a_21 a_22 a_23    second lattice vector in line (does not need to be normalize)\n"
    )
    myfile.write(
        f"{unitcell[2][0]} {unitcell[2][1]} {unitcell[2][2]}     # a_31 a_32 a_33                     third lattice vector in line\n"
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

    myfile.write(f"atomtypes  {len(atom_types)}       #Number of types atom\n")
    for i, atom_type in enumerate(atom_types):
        m = magmoms[i]
        mass = masses[i]
        charge = 0
        myfile.write(
            f"{atom_type} {m} {mass} {charge} F 0 # atom type: (name, mag. moment, mass, charge, displacement, number TB-orb.)\n"
        )

    myfile.write(
        f"\natoms {len(cls.atoms)}          positions of the atom in the unit cell\n"
    )
    for i, atom in enumerate(cls.atoms):
        spos = cls.atoms.get_scaled_positions()[i]
        myfile.write(f"{atom_types[tags[i]]} {spos[0]}, {spos[1]}, {spos[2]}  \n")


def _write_magnetic_interactions(cls, myfile, symmetry=False):
    myfile.write("\nThe Hamiltonian\n")
    myfile.write("# Magnetic interactions\n")


def _write_isotropic_exchange(cls, myfile, symmetry=False):
    myfile.write("\nmagnetic_J\n")
    shell_dict = get_ind_shell(cls.distance_dict)
    if symmetry:
        Jdict = cls.reduced_exchange_Jdict
    else:
        Jdict = cls.exchange_Jdict
    for key, val in Jdict.items():
        R, i, j = key
        ishell = shell_dict[(R, i, j)]
        myfile.write(
            f"{i+1} {j+1} {ishell} {val}     # between atoms type {i+1} and {j+1}, shell {R}, amplitude in eV  \n"
        )
    myfile.write(
        "\nc_H_J -1        apply 1/2 in front of the sum of the exchange energy - default is -1\n"
    )


def _write_magnetic_anisotropy(cls, myfile):
    if cls.k1 is None:
        return
    else:
        myfile.write("\nmagnetic_anisotropy \n")
        for i, k1 in enumerate(cls.k1):
            myfile.write(
                f"{i+1} {cls.k1dir[i][0]} {cls.k1dir[i][1]} {cls.k1dir[i][2]} {k1}     anisotropy of atoms type {i+1}, direction {cls.k1dir[i][0]} {cls.k1dir[i][1]} {cls.k1dir[i][2]} and amplitude in eV\n"
            )
        myfile.write("\nc_H_ani 1.0\n")


def _write_dmi(cls, myfile):
    if cls.has_dmi:
        myfile.write("\nmagnetic_D\n")
        for key, val in cls.dmi_ddict.items():
            R, i, j = key
            myfile.write(
                f"{i+1} {j+1} {R[0]} {R[1]} {R[2]} {val}         #between atoms type {i+1} and {j+1}, mediated by atom type {R}, shell {R}, amplitude in eV\n"
            )
        myfile.write(
            "\nc_H_D   -1.0             # coefficients to put in from of the sum - default is -1\n"
        )


def _write_exchange_tensor(cls, myfile, symmetry=False):
    myfile.write(
        "\nmagnetic_r2_tensor #Exchange tensor elements, middle 9 entries: xx, xy, xz, yx, etc. (in units of eV) and direction in which it should be applied\n"
    )

    Jtensor = cls.get_J_tensor_dict()
    shelldict = get_ind_shell(cls.distance_dict)
    unitcell = cls.atoms.get_cell().array.reshape((3, 3))
    uc_lengths = np.linalg.norm(unitcell, axis=1)
    if Jtensor is not None:
        for key, val in Jtensor.items():
            i, j, R = key
            # distance vector
            dvec = cls.distance_dict[(R, i, j)][0] / nm / uc_lengths
            # dscalar = np.linalg.norm(dvec)
            val = np.real(val)
            ishell = shelldict[(R, i, j)]
            myfile.write(
                f"{i+1} {j+1} {ishell} {' '.join([str(x) for x in val.flatten()])}  {dvec[0]} {dvec[1]} {dvec[2]} \n"
                # f"{i+1} {j+1} {dscalar} {' '.join([str(x) for x in val.flatten()])}  {dvec[0]} {dvec[1]} {dvec[2]} \n"
            )
        myfile.write(
            "\nc_H_Exchten -1         apply 1/2 in front of the sum of the exchange tensor energy - default is -1\n"
        )


def rattle_atoms_and_cell(atoms, stdev=0.001, cell_stdev=0.001):
    """
    Rattle both atomic positions and cell parameters.

    Parameters:
    -----------
    atoms: ASE atoms object
        The atoms to be rattled
    stdev: float
        Standard deviation for atomic displacement in Angstrom
    cell_stdev: float
        Standard deviation for cell parameter variation (fractional)

    Returns:
    --------
    None
        The atoms object is modified in-place
    """
    # Rattle atomic positions
    positions = atoms.get_positions()
    displacement = np.random.normal(0, stdev, positions.shape)
    atoms.set_positions(positions + displacement)

    # Rattle cell parameters
    cell = atoms.get_cell()
    cell_noise = np.random.normal(0, cell_stdev, cell.shape)
    atoms.set_cell(cell * (1 + cell_noise), scale_atoms=True)
