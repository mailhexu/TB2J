"""
IO module. called io_xml for historical reason.
write not only xml output.
- xml
- multibinit input template
- vampire inputs
- uppasd inputs (not tested yet)
- tom's asd inputs.
"""

# matplotlib.use("Agg")
import gc
import os
import pickle
from collections.abc import Iterable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from TB2J import __version__
from TB2J.io_exchange.io_txt import write_Jq_info
from TB2J.Jtensor import combine_J_tensor
from TB2J.kpoints import monkhorst_pack
from TB2J.spinham.spin_api import SpinModel
from TB2J.utils import symbol_number


class SpinIO(object):
    def __init__(
        self,
        atoms,
        spinat,
        charges,
        index_spin,
        orbital_names={},
        colinear=True,
        distance_dict=None,
        exchange_Jdict=None,
        Jiso_orb=None,
        DMI_orb=None,
        Jani_orb=None,
        dJdx=None,
        dJdx2=None,
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
        :param distance_dict: {(R, i,j ): distance} , here i and j are spin indices
        :param exchange_Jdict: {(R, i,j): J}
        :param Jiso_orb: {(R, i,j): J_orb}
        :param DMI_orb: {(R, i,j): D_orb}
        :param Jani_orb: {(R, i,j): Jani_orb}
        :param dJdx: {(R, i,j): dJdx}
        :param dJdx2: {(R, i,j): dJdx2}
        :param dmi_ddict:{(R, i,j): DMI}
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
        self.index_spin = index_spin
        #: index of spin linked to atoms. -1 if non-magnetic
        self.spinat = spinat  #: spin for each atom. shape of (natom, 3)
        self.colinear = colinear  #: If the calculation is collinear or not
        if (
            self.colinear
            and isinstance(self.spinat, np.ndarray)
            and self.spinat.shape[1] == 3
        ):
            self.magmoms = np.array(self.spinat)[:, 2]
        self.charges = charges
        #: A dictionary of distances, the keys are (i,j, R),
        # where i and j are spin index and R is the cell index,
        # a tuple of three integers.
        self.distance_dict = distance_dict
        self._ind_atoms = {}  #: The index of atom for each spin.
        for iatom, ispin in enumerate(self.index_spin):
            if ispin >= 0:
                self._ind_atoms[ispin] = iatom

        if exchange_Jdict is not None:
            self.has_exchange = True  #: whether there is isotropic exchange
            #: The dictionary of :math:`J_{ij}(R)`, the keys are (i,j, R),
            # where R is a tuple, and the value is the isotropic exchange
            self.exchange_Jdict = exchange_Jdict
        else:
            self.has_exchange = False
            self.exchange_Jdict = None

        self.Jiso_orb = Jiso_orb

        self.DMI_orb = DMI_orb
        self.Jani_orb = Jani_orb

        self.dJdx = dJdx
        self.dJdx2 = dJdx2

        if dmi_ddict is not None:
            self.has_dmi = True  #: Whether there is DMI.
            #: The dictionary of DMI. the key is the same as exchange_Jdict, the values are 3-d vectors (Dx, Dy, Dz).
            self.dmi_ddict = dmi_ddict
        else:
            self.has_dmi = False
            self.dmi_ddict = None

        if Jani_dict is not None:
            self.has_bilinear = True  #: Whether there is anisotropic exchange term
            #: The dictionary of anisotropic exchange. The vlaues are matrices of shape (3,3).
            self.Jani_dict = Jani_dict
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

        self.has_biquadratic = not (
            biquadratic_Jdict == {} or biquadratic_Jdict is None
        )
        self.biquadratic_Jdict = biquadratic_Jdict

        if NJT_ddict is not None:
            self.has_NJT_dmi = True
        self.NJT_ddict = NJT_ddict

        if NJT_Jdict is not None:
            self.has_NJT_exchange = True
        self.NJT_Jdict = NJT_Jdict

        natom = len(self.atoms)
        if gyro_ratio is None:
            self.gyro_ratio = [1.0] * natom  #: Gyromagnetic ratio for each atom
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

        now = datetime.now()
        self.description = f"""Exchange parameters generated by TB2J {__version__}
Generation time: {now.strftime("%y/%m/%d %H:%M:%S")}
"""
        if description is not None:
            self.description += description

        self.orbital_names = orbital_names
        self.TB2J_version = __version__

    def _build_Rlist(self):
        Rset = set()
        ispin_set = set()
        for R, i, j in self.exchange_Jdict:
            Rset.add(R)
            ispin_set.add(i)
            ispin_set.add(j)
        self.Rlist = list(Rset)
        self.ispin_list = list(ispin_set)
        self.nspin = len(self.ispin_list)
        assert self.nspin == max(self.ispin_list) + 1

    def _build_ind_atoms(self):
        self._ind_atoms = {}  #: The index of atom for each spin.
        for iatom, ispin in enumerate(self.index_spin):
            if ispin >= 0:
                self._ind_atoms[ispin] = iatom

    @property
    def ind_atoms(self):
        if not self._ind_atoms:
            self._build_ind_atoms()
        return self._ind_atoms

    def iatom(self, i):
        return self.ind_atoms[i]

    def get_spin_ispin(self, i):
        return self.spinat[self.iatom(i)]

    def get_symbol_number_ispin(self, symnum):
        """
        Return the spin index for a given symbol number.
        """
        symdict = symbol_number(self.atoms)
        return self.index_spin[symdict[symnum]]

    def i_spin(self, i):
        if isinstance(i, int):
            return i
        elif isinstance(i, str):
            return self.get_symbol_number_ispin(i)
        else:
            raise ValueError("i must be either an integer or a string.")

    def get_charge_ispin(self, i):
        i = self.i_spin(i)
        return self.charges[self.iatom(i)]

    def get_spin_iatom(self, iatom):
        return self.spinat[iatom]

    def get_charge_iatom(self, iatom):
        return self.charges[iatom]

    def ijR_index_spin_to_atom(self, i, j, R):
        return (self.iatom(i), self.iatom(j), R)

    def ijR_index_atom_to_spin(self, iatom, jatom, R):
        return (self.index_spin[iatom], self.index_spin[jatom], R)

    def ijR_list(self):
        return [(i, j, R) for R, i, j in self.exchange_Jdict]

    def ijR_list_index_atom(self):
        return [self.ijR_index_spin_to_atom(i, j, R) for R, i, j in self.exchange_Jdict]

    def get_J(self, i, j, R, default=None):
        i = self.i_spin(i)
        j = self.i_spin(j)
        key = (
            tuple(R),
            i,
            j,
        )
        if self.exchange_Jdict is not None and key in self.exchange_Jdict:
            return self.exchange_Jdict[key]
        else:
            return default

    def get_Jiso(self, i, j, R, default=None):
        i = self.i_spin(i)
        j = self.i_spin(j)
        key = (
            tuple(R),
            i,
            j,
        )
        if self.exchange_Jdict is not None and key in self.exchange_Jdict:
            return self.exchange_Jdict[key]
        else:
            return default

    def get_DMI(self, i, j, R, default=None):
        i = self.i_spin(i)
        j = self.i_spin(j)
        key = (
            tuple(R),
            i,
            j,
        )
        if self.dmi_ddict is not None and key in self.dmi_ddict:
            return self.dmi_ddict[(tuple(R), i, j)]
        else:
            return default

    def get_Jani(self, i, j, R, default=None):
        """
        Return the anisotropic exchange tensor for atom i and j, and cell R.
        param i : spin index i
        param j: spin index j
        param R (tuple of integers): cell index R
        """
        i = self.i_spin(i)
        j = self.i_spin(j)
        key = (
            tuple(R),
            i,
            j,
        )
        if self.Jani_dict is not None and key in self.Jani_dict:
            return self.Jani_dict[(tuple(R), i, j)]
        else:
            return default

    def get_J_tensor(self, i, j, R, iso_only=False):
        """
        Return the full exchange tensor for atom i and j, and cell R.
        param i : spin index i
        param j: spin index j
        param R (tuple of integers): cell index R
        """
        i = self.i_spin(i)
        j = self.i_spin(j)
        if iso_only:
            J = self.get_Jiso(i, j, R)
            if J is not None:
                Jtensor = np.eye(3) * self.get_J(i, j, R)
            else:
                Jtensor = np.eye(3) * 0
        else:
            Jtensor = combine_J_tensor(
                Jiso=self.get_J(i, j, R),
                D=self.get_DMI(i, j, R),
                Jani=self.get_Jani(i, j, R),
            )
        return Jtensor

    def get_full_Jtensor_for_one_R(self, R, iso_only=False):
        """
        Return the full exchange tensor of all i and j for cell R.
        param R (tuple of integers): cell index R
        returns:
            Jmat: (3*nspin,3*nspin) matrix.
        """
        n3 = self.nspin * 3
        Jmat = np.zeros((n3, n3), dtype=float)
        for i in range(self.nspin):
            for j in range(self.nspin):
                Jmat[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] = self.get_J_tensor(
                    i, j, R, iso_only=iso_only
                )
        return Jmat

    def get_full_Jtensor_for_Rlist(self, asr=False, iso_only=True):
        n3 = self.nspin * 3
        nR = len(self.Rlist)
        Jmat = np.zeros((nR, n3, n3), dtype=float)
        for iR, R in enumerate(self.Rlist):
            Jmat[iR] = self.get_full_Jtensor_for_one_R(R, iso_only=iso_only)
        if asr:
            iR0 = np.argmin(np.linalg.norm(self.Rlist, axis=1))
            assert np.linalg.norm(self.Rlist[iR0]) == 0
            for i in range(n3):
                sum_JRi = np.sum(np.sum(Jmat, axis=0)[i])
                Jmat[iR0][i, i] -= sum_JRi
        return Jmat

    def write_pickle(self, path="TB2J_results", fname="TB2J.pickle"):
        if not os.path.exists(path):
            os.makedirs(path)
        fname = os.path.join(path, fname)
        with open(fname, "wb") as myfile:
            try:
                pickle.dump(self.__dict__, myfile)
            except Exception as ex:
                print(f"Pickle not written due to {ex}")

    @classmethod
    def load_pickle(cls, path="TB2J_results", fname="TB2J.pickle"):
        fname = os.path.join(path, fname)
        with open(fname, "rb") as myfile:
            d = pickle.load(myfile)
        obj = cls(
            atoms=d["atoms"],
            spinat=d["spinat"],
            charges=d["charges"],
            index_spin=d["index_spin"],
        )
        obj.__dict__.update(d)
        obj._build_Rlist()
        return obj

    def write_all(self, path="TB2J_results"):
        self.write_pickle(path=path)
        self.write_txt(path=path)
        if self.Jiso_orb:
            self.write_txt(
                path=path,
                fname="exchange_orb_decomposition.out",
                write_orb_decomposition=True,
            )
        self.write_multibinit(path=os.path.join(path, "Multibinit"))
        self.write_tom_format(path=os.path.join(path, "TomASD"))
        self.write_vampire(path=os.path.join(path, "Vampire"))

        self.plot_all(savefile=os.path.join(path, "JvsR.pdf"))
        # self.write_Jq(kmesh=[9, 9, 9], path=path)

    def write_txt(self, *args, **kwargs):
        from TB2J.io_exchange.io_txt import write_txt

        write_txt(self, *args, **kwargs)

    # def write_txt_with_orb(self, path):
    #    from TB2J.io_exchange.io_txt import write_txt
    #    write_txt_with_orb(
    #        self, path=path, write_experimental=self.write_experimental)

    def write_multibinit(self, path):
        from TB2J.io_exchange.io_multibinit import write_multibinit

        write_multibinit(self, path=path)

    def write_Jq(self, kmesh, path, gamma=True, output_fname="EigenJq.txt", **kwargs):
        m = SpinModel(fname=os.path.join(path, "Multibinit", "exchange.xml"))
        m.set_ham(**kwargs)
        kpts = monkhorst_pack(kmesh, gamma_center=gamma)

        evals, evecs = m.ham.solve_k(kpts, Jq=True)
        with open(os.path.join(path, output_fname), "w") as myfile:
            myfile.write("=" * 60)
            myfile.write("\n")
            myfile.write("Generated by TB2J %s.\n" % (__version__))
            myfile.write("=" * 60 + "\n")
            myfile.write(
                "The spin ground state is estimated by calculating\n the eigenvalues and eigen vectors of J(q):\n"
            )
            write_Jq_info(self, kpts, evals, evecs, myfile, special_kpoints={})

    def model(self, path):
        m = SpinModel(fname=os.path.join(path, "Multibinit", "exchange.xml"))
        return m

    def plot_JvsR(
        self,
        ax=None,
        color="blue",
        marker="o",
        fname=None,
        show=False,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        ds = []
        Js = []
        for key, val in self.exchange_Jdict.items():
            d = self.distance_dict[key][1]
            ds.append(d)
            Js.append(val * 1e3)
        ax.scatter(ds, Js, marker=marker, color=color, **kwargs)
        ax.axhline(color="gray")
        ax.set_xlabel("Distance ($\AA$)")
        ax.set_ylabel("J (meV)")
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        return ax

    def plot_DvsR(self, ax=None, fname=None, show=False):
        if ax is None:
            fig, ax = plt.subplots()
        ds = []
        Ds = []
        for key, val in self.dmi_ddict.items():
            d = self.distance_dict[key][1]
            ds.append(d)
            Ds.append(val * 1e3)
        Ds = np.array(Ds)
        ax.scatter(ds, Ds[:, 0], marker="s", color="r", label="Dx")
        ax.scatter(
            ds, Ds[:, 1], marker="o", edgecolors="g", facecolors="none", label="Dy"
        )
        ax.scatter(
            ds, Ds[:, 2], marker="D", edgecolors="b", facecolors="none", label="Dz"
        )
        ax.axhline(color="gray")
        ax.legend(loc=1)
        ax.set_ylabel("D (meV)")
        ax.set_xlabel("Distance ($\AA$)")
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        return ax

    def plot_JanivsR(self, ax=None, fname=None, show=False):
        if ax is None:
            fig, ax = plt.subplots()
        ds = []
        Jani = []
        for key, val in self.Jani_dict.items():
            d = self.distance_dict[key][1]
            ds.append(d)
            # val = val - np.diag([np.trace(val) / 3] * 3)
            Jani.append(val * 1e3)
        Jani = np.array(Jani)
        s = "xyz"
        for i in range(3):
            ax.scatter(ds, Jani[:, i, i], marker="s", label=f"J{s[i]}{s[i]}")
        c = "rgb"
        for ic, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
            ax.scatter(
                ds,
                Jani[:, i, j],
                edgecolors=c[ic],
                facecolors="none",
                label=f"J{s[i]}{s[j]}",
            )
        ax.axhline(color="gray")
        ax.legend(loc=1, ncol=2)
        ax.set_xlabel("Distance ($\AA$)")
        ax.set_ylabel("Jani (meV)")
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        return ax

    def plot_all(self, title=None, savefile=None, show=False):
        if self.has_dmi and self.has_bilinear:
            naxis = 3
        else:
            naxis = 1
        fig, axes = plt.subplots(naxis, 1, sharex=True, figsize=(5, 2.2 * naxis))

        if self.has_dmi and self.has_bilinear:
            self.plot_JvsR(axes[0])
            self.plot_DvsR(axes[1])
            self.plot_JanivsR(axes[2])
        else:
            self.plot_JvsR(axes)

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, top=0.93)
        if savefile is not None:
            plt.savefig(savefile)
        if show:
            plt.show()
        plt.clf()
        plt.close()
        gc.collect()  # This is to fix the tk error if multiprocess is used.
        return fig, axes

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
                pos_jR = atoms.get_positions()[jatom] + np.dot(R, atoms.get_cell())
                vec = pos_jR - pos_i
                distance = np.sqrt(np.sum(vec**2))
                distance_dict[(tuple(R), ispin, jspin)] = (vec, distance)
    return distance_dict


def test_spin_io():
    import numpy as np
    from ase import Atoms

    atoms = Atoms(
        "SrMnO3",
        cell=np.eye(3) * 3.8,
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ],
    )
    spinat = [[0, 0, x] for x in [0, 3, 0, 0, 0]]
    charges = [2, 4, 5, 5, 5]
    index_spin = [-1, 0, -1, -1, -1]
    Rlist = [[0, 0, 0], [0, 0, 1]]
    ind_mag_atoms = [1]
    distance_dict = gen_distance_dict(ind_mag_atoms, atoms, Rlist)

    R0 = (0, 0, 0)
    R1 = (0, 0, 1)
    exchange_Jdict = {(R0, 0, 0): 1.2, (R1, 0, 0): 1.1}

    sio = SpinIO(
        atoms,
        spinat,
        charges,
        index_spin,
        colinear=True,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_Jdict,
    )

    sio.write_all()


if __name__ == "__main__":
    test_spin_io()
