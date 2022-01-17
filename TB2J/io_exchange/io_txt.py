import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
from ase.data import atomic_masses
from ase.units import eV, Hartree, Bohr, Ry, J
import os
from collections import Iterable, namedtuple
from itertools import groupby
from TB2J import __version__
from TB2J.utils import symbol_number
from numpy import array_str
from TB2J.spinham.spin_api import SpinModel
from datetime import datetime
import pickle


def write_info_section(cls, myfile):
    myfile.write('=' * 90 + '\n')
    myfile.write("Information: \n")
    now = datetime.now()
    #myfile.write("Exchange parameters generated by TB2J %s\n" % (__version__))
    #myfile.write("Generation time: %s\n" % (now.strftime("%y/%m/%d %H:%M:%S")))
    myfile.write(cls.description)


def write_atom_section(cls, myfile):
    """
    write the atom section
    including the cell and the atomic positions.
    """
    myfile.write('=' * 90 + '\n')
    myfile.write("Cell (Angstrom):\n")
    cell = cls.atoms.get_cell()
    for c in cell:
        myfile.write("{:6.3f}  {:6.3f}  {:6.3f}\n".format(c[0], c[1], c[2]))

    # write atom information
    myfile.write('\n')
    myfile.write('=' * 90 + '\n')
    myfile.write('Atoms:  \n')
    myfile.write(
        '(Note: charge and magmoms only count the wannier functions.)\n')
    if cls.colinear:
        myfile.write("{:^12s} {:^9s} {:^9s} {:^9s} {:^9s} {:^9s}\n".format(
            'Atom number', 'x', 'y', 'z', 'w_charge', 'w_magmom'))
    else:
        myfile.write(
            "{:^12s} {:^9s} {:^9s} {:^9s} {:^9s} {:^9s} {:^9s} {:^9s}\n".
            format('Atom_number', 'x', 'y', 'z', 'w_charge', 'M(x)', 'M(y)',
                   'M(z)'))

    symnum = symbol_number(cls.atoms)
    sns = list(symnum.keys())
    poses = cls.atoms.get_positions()

    tchg, tmx, tmy, tmz = 0, 0, 0, 0
    for i, s in enumerate(symnum):
        if cls.colinear:
            mag = cls.magmoms[i]
            chg = cls.charges[i]
            myfile.write(
                "{:<12s} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f}\n".format(
                    s, poses[i, 0], poses[i, 1], poses[i, 2], chg, mag))
        else:
            chg = cls.charges[i]
            mx, my, mz = cls.spinat[i, :]
            myfile.write(
                "{:<12s} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f}\n"
                .format(s, poses[i, 0], poses[i, 1], poses[i, 2], chg, mx, my,
                        mz))
            tchg += chg
            tmx += mx
            tmy += my
            tmz += mz
    if cls.colinear:
        myfile.write("{:<12s} {:9s} {:9s} {:9s} {:9.4f} {:9.4f}\n".format(
            'Total', '', '', '', np.sum(cls.charges), np.sum(cls.magmoms)))
    else:
        myfile.write(
            "{:<12s} {:9s} {:9s} {:9s} {:9.4f} {:9.4f} {:9.4f} {:9.4f}\n".
            format('Total', '', '', '', tchg, tmx, tmy, tmz))

    myfile.write('\n')


def write_orbital_section(cls, myfile):
    if cls.Jiso_orb:
        myfile.write('=' * 90 + '\n')
        myfile.write('Orbitals used in decomposition: \n')
        myfile.write("The name of the orbitals for the decomposition: \n")
        symnum = symbol_number(cls.atoms)
        sns = list(symnum.keys())
        for iatom in cls.orbital_names:
            if cls.index_spin[iatom] != -1:
                myfile.write(f"{sns[iatom]} : {cls.orbital_names[iatom]}\n")
        myfile.write("\n")


def write_exchange_section(cls,
                           myfile,
                           order='distance',
                           write_experimental: bool = True,
                           write_orb_decomposition: bool = False,
                           ):
    symnum = symbol_number(cls.atoms)
    sns = list(symnum.keys())
    poses = cls.atoms.get_positions()

    myfile.write('=' * 90 + '\n')
    myfile.write('Exchange: \n')

    # l = [x for x in cls.exchange_Jlist if abs(x.J) > cutoff]
    keys = cls.exchange_Jdict
    if order == 'amplitude':
        l = sorted(keys,
                   key=lambda x: abs(cls.exchange_Jdict[x]),
                   reverse=True)
    elif order == 'distance':
        l = sorted(keys, key=lambda x: cls.distance_dict[x][1])
    else:
        l = keys

    myfile.write("{:6s} {:5s} {:15s} {:7s} {:24s} {:11s} \n".format(
        "    i", "    j", "         R", "  J_iso(meV)", "         vector",
        "distance(A)"))

    for ll in l:
        myfile.write('-' * 88 + '\n')
        R, i, j = ll
        J = cls.exchange_Jdict[ll]
        vec, distance = cls.distance_dict[ll]
        myfile.write(
            "   {:5s} {:5s} ({:3d}, {:3d}, {:3d}) {:7.4f}   ({:6.3f}, {:6.3f}, {:6.3f}) {:6.3f} \n"
            .format(sns[cls.ind_atoms[i]], sns[cls.ind_atoms[j]], R[0], R[1],
                    R[2], J * 1e3, vec[0], vec[1], vec[2], distance))

        Jiso = cls.exchange_Jdict[ll] * 1e3
        myfile.write(f'J_iso: {Jiso:7.4f} \n')

        if cls.has_biquadratic and write_experimental:
            Jprime, B = cls.biquadratic_Jdict[ll]
            myfile.write(
                f"[Testing!] Jprime: {Jprime*1e3:.3f},  B: {B*1e3:.3f}\n")

        if cls.dJdx is not None:
            dJdx = cls.dJdx[ll]
            myfile.write(f"dJ/dx: {dJdx*1e3:.3f}\n")

        if cls.dJdx2 is not None:
            dJdx2 = cls.dJdx2[ll]
            myfile.write(f"d2J/dx2: {dJdx2*1e3:.3f}\n")

        if cls.dmi_ddict is not None:
            DMI = cls.dmi_ddict[ll] * 1e3
            myfile.write(
                '[Testing!] DMI: ({:7.4f} {:7.4f} {:7.4f})\n'.format(
                    DMI[0], DMI[1], DMI[2]))

        if write_experimental:
            try:
                DMI2 = cls.debug_dict['DMI2'][ll] * 1e3
                myfile.write('[Debug!] DMI2: ({:7.4f} {:7.4f} {:7.4f})\n'.format(
                    DMI2[0], DMI2[1], DMI2[2]))
                pass
            except:
                pass

        if cls.Jani_dict is not None and write_experimental:
            J = cls.Jani_dict[ll] * 1e3
            myfile.write(
                f"[Testing!]J_ani:\n{array_str(J, precision=3, suppress_small=True)}\n"
            )

        if cls.NJT_ddict is not None:
            DMI = cls.NJT_ddict[ll] * 1e3
            myfile.write(
                '[Experimental!] DMI_NJt: ({:7.4f} {:7.4f} {:7.4f})\n'.format(
                    DMI[0], DMI[1], DMI[2]))

        if cls.NJT_Jdict is not None:
            J = cls.NJT_Jdict[ll] * 1e3
            myfile.write(
                '[Testing!] Jani_NJt: ({:7.4f} {:7.4f} {:7.4f})\n'.format(
                    J[0], J[1], J[2]))

        if write_orb_decomposition:
            if cls.Jiso_orb:
                myfile.write("Orbital contributions:\n isotropic J:\n {} \n".format(
                    np.array_str(cls.Jiso_orb[ll] * 1e3,
                                 precision=3,
                                 suppress_small=True)))

            xyz = 'xyz'
            if cls.DMI_orb:
                for i in range(3):
                    myfile.write(f"DMI {xyz[i]}:\n")
                    myfile.write(np.array_str(cls.DMI_orb[ll][i] * 1e3,
                                              precision=3,
                                              suppress_small=True))
                    myfile.write("\n")

            if cls.Jani_orb:
                for i in range(3):
                    for j in range(3):
                        myfile.write(f"Jani {xyz[i]}{xyz[j]}:\n")
                        myfile.write(np.array_str(cls.Jani_orb[ll][i, j] * 1e3,
                                                  precision=3,
                                                  suppress_small=True))
                        myfile.write("\n")


def write_Jq_info(cls, kpts, evals, evecs, myfile, special_kpoints={}):
    """
    kpts: the list of kpoints
    evals: eigen values
    evecs: eigen vectors
    fname: the name of the output file
    spk: special kpoints
    """
    symnum = symbol_number(cls.atoms)
    sns = list(symnum.keys())
    imin = np.argmin(evals[:, 0])
    #emin = np.min(evals[:, 0])
    nspin = evals.shape[1] // 3
    evec_min = evecs[imin, :, 0].reshape(nspin, 3)

    # write information to file
    if myfile is not None:
        myfile.write("\nThe energy minimum is at the q-point:")
        myfile.write("%s\n" % kpts[np.argmin(evals[:, 0])])
        myfile.write("The eigenstate at the energy minimum is:\n")
        for i, ev in enumerate(evec_min):
            v = ev.real / np.linalg.norm(ev)
            myfile.write(" > %s: (%.3f, %.3f, %.3f)\n" %
                         (sns[cls.ind_atoms[i]], v[0], v[1], v[2]))
    print("\nThe energy minimum is at:")
    print("%s\n" % kpts[np.argmin(evals[:, 0])])
    print("The eigenstate at the energy minimum is:\n")
    for i, ev in enumerate(evec_min):
        v = ev.real / np.linalg.norm(ev)
        print("spin %s: (%.3f, %.3f, %.3f)" %
              (sns[cls.ind_atoms[i]], v[0], v[1], v[2]))


def write_txt(cls,
              path='TB2J_results',
              fname='exchange.out',
              order='distance',
              write_orb_decomposition=False,
              write_experimental=True,
              cutoff=1e-4):
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, 'w') as myfile:
        write_info_section(cls, myfile)
        write_atom_section(cls, myfile)
        write_orbital_section(cls, myfile)
        write_exchange_section(cls,
                               myfile,
                               write_experimental=write_experimental,
                               write_orb_decomposition=write_orb_decomposition
                               )
