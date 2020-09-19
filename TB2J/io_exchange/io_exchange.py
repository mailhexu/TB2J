"""
IO module. called io_xml for historical reason.
write not only xml output.
- xml
- multibinit input template
- vampire inputs
- uppasd inputs (not tested yet)
- tom's asd inputs.
"""

import os
import numpy as np
from ase.dft.kpoints import monkhorst_pack
import pickle
from ase.cell import Cell
from TB2J import __version__
from datetime import datetime


class SpinIO(object):
    def __init__(
        self,
        atoms,
        spinat,
        charges,
        index_spin,
        colinear=True,
        distance_dict=None,
        exchange_Jdict=None,
        exchange_Jdict_orb=None,
        dJdx=None,
        dmi_ddict=None,
        Jani_dict=None,
        biquadratic_Jdict=None,
        debug_dict=None,
        k1=None,
        k1dir=None,
        NJT_Jdict=None,
        NJT_ddict=None,
        damping=None,
        gyro_ratio=None,
        write_experimental=True,
        description=None,
    ):
        """
        :param atoms: Ase atoms structure.
        :param spinat: spin for each atom. (3*natom)
        :param charges: charges for each atom (natom)
        :param index_spin: index of spin in the spin potential for each atom.
        :param colinear: whether the parameters are for a colinear wannier calculation.
        :param distance_dict: {(R, i,j ): distance}
        :param exchange_Jdict: {(R, i,j): J}
        :param exchange_Jdict_orb: {(R, i,j): J}
        :param dJdx: {(R, i,j): dJdx}
        :param dim_ddict:{(R, i,j): DMI}
        :param Jani_dict: {(R, i,j): Jani'}, Jani is a 3*3 matrix
        :param biqudratic_Jdict: {(R, i,j ): J}
        :param k1: single ion anisotropy amplitude.o
        :param kdir: directio of k1
        :param NJT_Jdict: exhange calculated using NJT method
        :param NJT_ddict: DMI calculated using NJT method
        :param damping: damping factor
        :param gyro_ratio:  gyromagnetic ratio
        :param write_experimental: write_experimental data to output files
        :param description: add some description into the xml file.
        """
        self.atoms = atoms  #: atomic structures, ase.Atoms object
        self.index_spin = index_spin  #: index of spin linked to atoms. -1 if non-magnetic
        self.spinat = spinat  #: spin for each atom. shape of (natom, 3)
        self.colinear = colinear  #: If the calculation is collinear or not
        if self.colinear and self.spinat != []:
            self.magmoms = np.array(self.spinat)[:, 2]
        self.charges = charges
        self.distance_dict = distance_dict  #: A dictionary of distances, the keys are (i,j, R), where i and j are spin index and R is the cell index, a tuple of three integers.
        self.ind_atoms = {}  #: The index of atom for each spin.
        for iatom, ispin in enumerate(self.index_spin):
            if ispin >= 0:
                self.ind_atoms[ispin] = iatom

        if exchange_Jdict is not None:
            self.has_exchange = True  #: whether there is isotropic exchange
            self.exchange_Jdict = exchange_Jdict  #: The dictionary of :math:`J_{ij}(R)`, the keys are (i,j, R), where R is a tuple, and the value is the isotropic exchange
        else:
            self.has_exchange = False
            self.exchange_Jdict = None

        self.exchange_Jdict_orb = exchange_Jdict_orb

        self.dJdx = dJdx

        if dmi_ddict is not None:
            self.has_dmi = True  #: Whether there is DMI.
            self.dmi_ddict = dmi_ddict  #: The dictionary of DMI. the key is the same as exchange_Jdict, the values are 3-d vectors (Dx, Dy, Dz).
        else:
            self.has_dmi = False
            self.dmi_ddict = None

        if Jani_dict is not None:
            self.has_bilinear = True  #: Whether there is anisotropic exchange term
            self.Jani_dict = Jani_dict  #: The dictionary of anisotropic exchange. The vlaues are matrices of shape (3,3).
        else:
            self.has_bilinear = False
            self.Jani_dict = None

        if k1 is not None and k1dir is not None:
            self.has_uniaxial_anistropy = True
            self.k1 = k1
            self.k1dir = k1dir
        else:
            self.has_uniaxial_anistropy = False
            self.k1 = None
            self.k1dir = None

        self.has_bilinear = not (Jani_dict == {} or Jani_dict is None)

        self.has_biquadratic = not (biquadratic_Jdict == {}
                                    or biquadratic_Jdict is None)
        self.biquadratic_Jdict = biquadratic_Jdict

        if NJT_ddict is not None:
            self.has_NJT_dmi = True
        self.NJT_ddict = NJT_ddict

        if NJT_Jdict is not None:
            self.has_NJT_exchange = True
        self.NJT_Jdict = NJT_Jdict

        natom = len(self.atoms)
        if gyro_ratio is None:
            self.gyro_ratio = [1.0
                               ] * natom  #: Gyromagnetic ratio for each atom
        elif isinstance(gyro_ratio, Iterable):
            self.gyro_ratio = gyro_ratio
        else:
            self.gyro_ratio = [gyro_ratio] * natom

        if damping is None:
            self.damping = [1.0] * natom  # damping factor for each atom
        elif isinstance(damping, Iterable):
            self.damping = damping
        else:
            self.damping = [damping] * natom

        self.debug_dict = debug_dict

        self.write_experimental = write_experimental

        now=datetime.now()
        self.description = f"""Exchange parameters generated by TB2J {__version__}
Generation time: {now.strftime("%y/%m/%d %H:%M:%S")}
"""
        if description is not None:
            self.description += description


    def write_pickle(self, path='TB2J_results', fname='TB2J.pickle'):
        if not os.path.exists(path):
            os.makedirs(path)
        fname = os.path.join(path, fname)
        with open(fname, 'wb') as myfile:
            try:
                pickle.dump(self.__dict__, myfile)
            except:
                print("Pickle not written.")

    @staticmethod
    def load_pickle(path='TB2J_resutls', fname='TB2J.pickle'):
        fname = os.path.join(path, fname)
        with open(fname, 'rb') as myfile:
            d = pickle.load(myfile)
        obj = SpinIO(atoms=[], spinat=[], charges=[], index_spin=[])
        obj.__dict__.update(d)
        return obj

    def write_all(self, path='TB2J_results'):
        self.write_pickle(path=path)
        self.write_txt(path=path)
        self.write_multibinit(path=os.path.join(path, 'Multibinit'))
        self.write_tom_format(path=os.path.join(path, 'TomASD'))
        self.write_vampire(path=os.path.join(path, 'Vampire'))
        self.write_Jq(kmesh=[9, 9, 9], path=path)

    def write_txt(self, path):
        from TB2J.io_exchange.io_txt import write_txt
        write_txt(self, path=path, write_experimental=self.write_experimental)

    def write_multibinit(self, path):
        from TB2J.io_exchange.io_multibinit import write_multibinit
        write_multibinit(self, path=path)

    def write_Jq(self, kmesh, path,**kwargs):
        from TB2J.spinham.spin_api import SpinModel
        from TB2J.io_exchange.io_txt import write_Jq_info
        m = SpinModel(fname=os.path.join(path, 'Multibinit', 'exchange.xml'))
        m.set_ham(**kwargs)
        kpts1 = monkhorst_pack(kmesh)
        bp = Cell(self.atoms.cell).bandpath(npoints=400)
        kpts2 = bp.kpts
        kpts = np.vstack([kpts1, kpts2])

        evals, evecs = m.ham.solve_k(kpts, Jq=True)
        with open(os.path.join(path, 'summary.txt'), 'w') as myfile:
            myfile.write("=" * 60)
            myfile.write("\n")
            myfile.write("Generated by TB2J %s.\n" % (__version__))
            myfile.write("=" * 60 + '\n')
            myfile.write(
                "The spin ground state is estimated by calculating\n the eigenvalues and eigen vectors of J(q):\n"
            )
            write_Jq_info(self, kpts, evals, evecs, myfile, special_kpoints={})

    def write_tom_format(self, path):
        from TB2J.io_exchange.io_tomsasd import write_tom_format
        write_tom_format(self, path=path)

    def write_vampire(self, path):
        from TB2J.io_exchange.io_vampire import write_vampire
        write_vampire(self, path=path)

    def write_uppasd(self, path):
        from TB2J.io_exchange.io_uppasd import write_uppasd
        write_uppasd(self, path=path)


def gen_distance_dict(ind_mag_atoms, atoms, Rlist):
    distance_dict = {}
    ind_matoms = ind_mag_atoms
    for ispin, iatom in enumerate(ind_matoms):
        for jspin, jatom in enumerate(ind_matoms):
            for R in Rlist:
                pos_i = atoms.get_positions()[iatom]
                pos_jR = atoms.get_positions()[jatom] + np.dot(
                    R, atoms.get_cell())
                vec = pos_jR - pos_i
                distance = np.sqrt(np.sum(vec**2))
                distance_dict[(tuple(R), ispin, jspin)] = (vec, distance)
    return distance_dict


def test_spin_io():
    from ase import Atoms
    import numpy as np
    atoms = Atoms('SrMnO3',
                  cell=np.eye(3) * 3.8,
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0, .5, .5],
                                    [.5, 0, .5], [.5, .5, 0]])
    spinat = [[0, 0, x] for x in [0, 3, 0, 0, 0]]
    charges = [2, 4, 5, 5, 5]
    index_spin = [-1, 0, -1, -1, -1]
    colinear = True
    Rlist = [[0, 0, 0], [0, 0, 1]]
    ind_mag_atoms = [1]
    distance_dict = gen_distance_dict(ind_mag_atoms, atoms, Rlist)

    R0 = (0, 0, 0)
    R1 = (0, 0, 1)
    exchange_Jdict = {(R0, 0, 0): 1.2, (R1, 0, 0): 1.1}

    sio = SpinIO(atoms,
                 spinat,
                 charges,
                 index_spin,
                 colinear=True,
                 distance_dict=distance_dict,
                 exchange_Jdict=exchange_Jdict)

    sio.write_all()
