import numpy as np
import os
from ase.units import Ry


def write_uppasd(cls, path="TB2J_results/UppASD"):
    if not os.path.exists(path):
        os.makedirs(path)
    cls.write_uppasd_posfile(os.path.join(path, "posfile"))
    cls.write_uppasd_momfile(os.path.join(path, "momfile"))
    cls.write_uppasd_exchange(os.path.join(path, "jASD1"))
    cls.write_uppasd_infile(os.path.join(path, "input"))


def write_uppasd_posfile(cls, fname):
    with open(fname, "w") as myfile:
        natom = len(cls.atoms)
        for i in range(natom):
            text = ""
            id_spin = cls.index_spin[i]
            if id_spin > -1:
                pos = cls.atoms.get_scaled_positions()[i]
                text = "{id_atom} {id_spin} {pos_x} {pos_y} {pos_z}\n".format(
                    id_atom=id_spin + 1,
                    id_spin=id_spin + 1,
                    pos_x=pos[0],
                    pos_y=pos[1],
                    pos_z=pos[2],
                )
                myfile.write(text)


def write_uppasd_momfile(cls, fname):
    with open(fname, "w") as myfile:
        natom = len(cls.atoms)
        for i in range(natom):
            text = ""
            id_spin = cls.index_spin[i]
            if id_spin > -1:
                pos = cls.atoms.get_scaled_positions()[i]
                ms = np.sqrt(np.sum(np.array(cls.spinat[i]) ** 2))
                spin = np.array(cls.spinat[i]) / ms
                text = "{id_atom} {id_spin} {ms} 0.0 0.0 1.0\n".format(
                    id_atom=id_spin + 1, id_spin=id_spin + 1, ms=ms
                )
                myfile.write(text)


def write_uppasd_exchange(cls, fname):
    with open(fname, "w") as myfile:
        nexch = len(cls.exchange_Jdict.items())
        myfile.write(
            "{num_interactions} {type_exchange}\n".format(
                num_interactions=nexch, type_exchange=0
            )
        )

        counter = -1
        for key, val in cls.exchange_Jdict.items():
            counter += 1  # starts at 0
            R, i, j = key
            pos = cls.atoms.get_positions()
            d = np.dot(np.array(R), cls.atoms.get_cell()) + pos[j] - pos[i]
            myfile.write(
                "{i} {j} {Rx} {Ry} {Rz} {Jij}\n".format(
                    IID=counter,
                    i=i + 1,
                    j=j + 1,
                    Rx=d[0],
                    Ry=d[1],
                    Rz=d[2],
                    Jij=val * 1e3 / Ry,
                )
            )  # mRy


def write_uppasd_infile(cls, fname):
    tmpl = """ simid Unamed
    ncell 12 12 12
    BC P P P
    cell
    Sym 0
    posfile ./posfile
    momfile ./momfile
    exchange ./jASD1
    #anistropy ./kfile

    do_ralloy 0
    Mensemble 1
    tseed 4499
    maptype 1
    SDEalgh 1
    Initmag 1
    ip_mode M
    ip_mcanneal 1
    10000 300 1.00e-16 0.3

    mode M
    temp 300
    mcNstep 50000
    Nstep 50000
    damping 0.1
    timestep 1.0e-16

    do_avrg Y

    do_cumu Y
    cumu_step 50
    cumu_buff 10

    do_tottraj N
    tottraj_step 1000

    plotenergy 1

    do_sc C
    do_ams Y
    do_magdos Y
    magdos_freq 200
    magdos_sigma 30
    qpoints C

    do_stiffness Y
    eta_max 12
    eta_min 6
    alat 2.83e-10
    """
    with open(fname, "w") as myfile:
        myfile.write(tmpl)
