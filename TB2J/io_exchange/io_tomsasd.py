import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
from ase.data import atomic_masses
from ase.units import eV, Hartree, Bohr, Ry, J
import os
from collections import Iterable, namedtuple
from itertools import groupby
from TB2J.utils import symbol_number
import pickle


def write_tom_ucf(cls, fname):
    """
    n1 TODO: what is this
    n2 TODO : what is n2
    id pos_x pos_y pos_z ms damping gyroratio symbol spin_x spin_y spin_z abs(K) K_x K_y K_z
    """
    natom = len(cls.atoms)
    nmatom = len(list(filter(lambda x: x > -1, cls.index_spin)))
    with open(fname, 'w') as myfile:
        myfile.write("%s\n" % nmatom)
        myfile.write("%s\n" % nmatom)
        for i in range(natom):
            text = ""
            id_spin = cls.index_spin[i]
            if id_spin > -1:
                pos = cls.atoms.get_scaled_positions()[i]
                ms = np.sqrt(np.sum(np.array(cls.spinat[i])**2))
                spin = np.array(cls.spinat[i]) / ms
                damping = cls.damping[i]
                gyro_ratio = cls.gyro_ratio[i]
                symbol = cls.atoms.get_chemical_symbols()[i]
                if cls.has_uniaxial_anistropy:
                    #abs(K)
                    k1 = cls.k1[i]
                    k1dir = cls.k1dir[i]
                else:
                    k1 = 0.0
                    k1dir = [0.0, 0.0, 1.0]
                text = "{id_spin} {pos_x} {pos_y} {pos_z} {ms} {damping} {gyro_ratio} {symbol} {spin_x} {spin_y} {spin_z} {k1} {kx} {ky} {kz}\n".format(
                    id_spin=id_spin,
                    pos_x=pos[0],
                    pos_y=pos[1],
                    pos_z=pos[2],
                    ms=ms,
                    damping=damping,
                    gyro_ratio=gyro_ratio / 1.76e11,
                    symbol=symbol,
                    spin_x=spin[0],
                    spin_y=spin[1],
                    spin_z=spin[2],
                    k1=k1,
                    kx=k1dir[0],
                    ky=k1dir[1],
                    kz=k1dir[2],
                )
                myfile.write(text)

def write_tom_exch(cls, fname):
    """
    write exchange_J to exch file.
    """
    # TODO convert DMI to matrix form and

    # prepare
    maxInt = 0
    if cls.has_exchange:
        for key, group in groupby(
                cls.exchange_Jdict.keys(), lambda x: (x[1], x[2])):
            l = len(list(group))
            if l > maxInt:
                maxInt = l

    with open(fname, 'w') as myfile:
        myfile.write("""exchange:
{
FourSpin = FALSE;
Scale = [ 1.0 , 1.0 , 1.0 ];
MaxInteractions = %s;
TruncateExchange = FALSE;
};

        """ % maxInt)
        if cls.has_exchange:
            for key, group in groupby(
                    cls.exchange_Jdict.keys(), lambda x: (x[1], x[2])):
                group = list(group)
                myfile.write("exchange_{}_{}:\n".format(key[0], key[1]))
                myfile.write("{ \n")
                myfile.write("Num_Interactions = {};\n".format(
                    len(list(group))))
                myfile.write('units = "eV";\n')
                for i, k in enumerate(group):
                    myfile.write(
                        "UnitCell{} = [{:.5e} , {:.5e} , {:.5e} ];\n".
                        format(i + 1, k[0][0], k[0][1], k[0][2]))
                    myfile.write(
                        "J{}_{} = [{:.8e} , {:.8e}, {:.8e}];\n".format(
                            i + 1, 1, cls.exchange_Jdict[k], 0.0, 0.0))
                    myfile.write(
                        "J{}_{} = [{:.8e} , {:.8e}, {:.8e}];\n".format(
                            i + 1, 2, 0.0, cls.exchange_Jdict[k], 0.0))
                    myfile.write(
                        "J{}_{} = [{:.8e} , {:.8e}, {:.8e}];\n".format(
                            i + 1, 3, 0.0, 0.0, cls.exchange_Jdict[k]))
                myfile.write("};\n")

# Tom's ASD code
def write_tom_format(cls, path='TB2J_results/TomASD', prefix='exchange'):
    if not os.path.exists(path):
        os.makedirs(path)
    exch_fname = os.path.join(path, "%s.exch" % prefix)
    ucf_fname = os.path.join(path, "%s.ucf" % prefix)
    write_tom_ucf(cls,ucf_fname)
    write_tom_exch(cls, exch_fname)

