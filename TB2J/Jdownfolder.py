import os
from collections import defaultdict
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from TB2J.io_exchange import SpinIO
from TB2J.Jtensor import decompose_J_tensor


def ind_to_indn(ind, n=3):
    """index to index. e.g. 1, 2, 4 to 1*3, 1*3+1, 1*3+2, ... 4*3, 4*3+1, 4*3+2
    :param ind: the input indices
    :returns: ind3, the output indices
    """
    indn = np.repeat(ind, n) * n
    for i in range(n):
        indn[i::n] += i
    return indn


class JDownfolder:
    def __init__(self, JR, Rlist, iM, iL, qmesh, iso_only=False):
        self.JR = JR
        self.Rlist = Rlist
        self.nR = len(Rlist)
        self.nM = len(iM)
        self.nL = len(iL)
        self.nsite = self.nM + self.nL
        self.iM = iM
        self.iL = iL
        self.qmesh = qmesh
        self.qpts = monkhorst_pack(qmesh)
        self.nqpt = len(self.qpts)
        self.nMn = self.nM * 3
        self.nLn = self.nL * 3
        self.iso_only = iso_only

    def get_Jq(self, q):
        Jq = np.zeros(self.JR[0].shape, dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2.0j * np.pi * np.dot(q, R))
            Jq += self.JR[iR] * phase
        return Jq

    def get_JR(self):
        JR_downfolded = np.zeros((self.nR, self.nMn, self.nMn), dtype=float)
        Jq_downfolded = np.zeros((self.nqpt, self.nMn, self.nMn), dtype=complex)
        self.iMn = ind_to_indn(self.iM, n=3)
        self.iLn = ind_to_indn(self.iL, n=3)
        for iq, q in enumerate(self.qpts):
            Jq = self.get_Jq(q)
            Jq_downfolded[iq] = self.downfold_oneq(Jq)
            for iR, R in enumerate(self.Rlist):
                phase = np.exp(-2.0j * np.pi * np.dot(q, R))
                JR_downfolded[iR] += np.real(Jq_downfolded[iq] * phase / self.nqpt)
        return JR_downfolded

    def downfold_oneq(self, J):
        JMM = J[np.ix_(self.iMn, self.iMn)]
        JLL = J[np.ix_(self.iLn, self.iLn)]
        JLM = J[np.ix_(self.iLn, self.iMn)]
        JML = J[np.ix_(self.iMn, self.iLn)]
        Jn = JMM - JML @ np.linalg.inv(JLL) @ JLM
        return Jn


class JDownfolder_pickle:
    def __init__(
        self, inpath, metals, ligands, outpath, qmesh=[7, 7, 7], iso_only=False
    ):
        self.exc = SpinIO.load_pickle(path=inpath, fname="TB2J.pickle")

        self.iso_only = (self.exc.dmi_ddict is None) or iso_only

        self.metals = metals
        self.ligands = ligands
        self.outpath = outpath

        # read atomic structure
        self.atoms = self.exc.atoms
        self.nspin = self.exc.nspin
        self.qmesh = qmesh
        self.natom = len(self.atoms)
        self.Rcut = None
        self._build_atom_index()
        self._prepare_distance()
        self._downfold()

    def _build_atom_index(self):
        self.magnetic_elements = self.metals
        self.iM = []
        self.iL = []
        self.ind_mag_atoms = []
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                self.iM.append(self.exc.index_spin[i])
                self.ind_mag_atoms.append(i)
            elif sym in self.ligands:
                self.iL.append(self.exc.index_spin[i])

        self.nM = len(self.iM)
        self.nL = len(self.iL)
        self.nsite = self.nM + self.nL

    def _downfold(self):
        JR2 = self.exc.get_full_Jtensor_for_Rlist(asr=True)
        d = JDownfolder(
            JR2,
            self.exc.Rlist,
            iM=self.iM,
            iL=self.iL,
            qmesh=self.qmesh,
            iso_only=self.iso_only,
        )
        Jd = d.get_JR()

        self._prepare_distance()
        self._prepare_index_spin()
        self.Jdict = {}
        if self.iso_only:
            self.DMIdict = None
            self.Janidict = None
        else:
            self.DMIdict = {}
            self.Janidict = {}

        for iR, R in enumerate(d.Rlist):
            for i, ispin in enumerate(self.index_spin):
                for j, jspin in enumerate(self.index_spin):
                    if ispin >= 0 and jspin >= 0:
                        if not (tuple(R) == (0, 0, 0) and ispin == jspin):
                            # self interaction.
                            J33 = Jd[
                                iR, ispin * 3 : ispin * 3 + 3, jspin * 3 : jspin * 3 + 3
                            ]
                            J, DMI, Jani = decompose_J_tensor(J33)
                            self.Jdict[(tuple(R), ispin, jspin)] = J
                            if not self.iso_only:
                                self.DMIdict[(tuple(R), ispin, jspin)] = DMI
                                self.Janidict[(tuple(R), ispin, jspin)] = Jani

        io = SpinIO(
            atoms=self.atoms,
            spinat=self.exc.spinat,
            charges=self.exc.charges,
            index_spin=self.index_spin,
            colinear=self.iso_only,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.Jdict,
            dmi_ddict=self.DMIdict,
            Jani_dict=self.Janidict,
        )

        io.write_all(self.outpath)

    def _prepare_distance(self):
        self.distance_dict = {}
        self.short_Rlist = []
        self.R_ijatom_dict = defaultdict(lambda: [])
        ind_matoms = self.ind_mag_atoms
        for ispin, iatom in enumerate(ind_matoms):
            for jspin, jatom in enumerate(ind_matoms):
                for R in self.exc.Rlist:
                    pos_i = self.atoms.get_positions()[iatom]
                    pos_jR = self.atoms.get_positions()[jatom] + np.dot(
                        R, self.atoms.get_cell()
                    )
                    vec = pos_jR - pos_i
                    distance = np.sqrt(np.sum(vec**2))
                    if self.Rcut is None or distance < self.Rcut:
                        self.distance_dict[(tuple(R), ispin, jspin)] = (vec, distance)
                        self.R_ijatom_dict[tuple(R)].append((iatom, jatom))
        self.short_Rlist = list(self.R_ijatom_dict.keys())

    def _prepare_index_spin(self):
        # index_spin: index in spin hamiltonian of atom. starts from 1. -1 means not considered.
        ind_matoms = []
        self.index_spin = []
        ispin = 0
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                ind_matoms.append(i)
                self.index_spin.append(ispin)
                ispin += 1
            else:
                self.index_spin.append(-1)


def test():
    # pass
    # inpath = "/home/hexu/projects/NiCl2/vasp_inputs/TB2J_results"
    # inpath = "/home/hexu/projects/TB2J_example/CrI3/TB2J_results"
    inpath = "/home/hexu/projects/TB2J_projects/NiCl2/TB2J_NiCl/TB2J_results"
    fname = os.path.join(inpath, "TB2J.pickle")
    p = JDownfolder_pickle(
        inpath=inpath, metals=["Ni"], ligands=["Cl"], outpath="TB2J_results_downfolded"
    )


if __name__ == "__main__":
    test()
