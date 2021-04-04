import os
import pickle
import copy
from collections import defaultdict
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from TB2J.io_exchange import SpinIO


class JDownfolder():
    def __init__(self, JR, Rlist, iM, iL, qmesh):
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

    def get_Jq(self, q):
        Jq = np.zeros((self.nsite, self.nsite), dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2.0j * np.pi * np.dot(q, R))
            Jq += self.JR[iR] * phase
        return Jq

    def get_JR(self):
        JR_downfolded = np.zeros((self.nR, self.nM, self.nM), dtype=float)
        Jq_downfolded = np.zeros((self.nqpt, self.nM, self.nM), dtype=complex)
        for iq, q in enumerate(self.qpts):
            Jq = self.get_Jq(q)
            Jq_downfolded[iq] = self.downfold_oneq(Jq)

            for iR, R in enumerate(self.Rlist):
                phase = np.exp(-2.0j * np.pi * np.dot(q, R))
                JR_downfolded[iR] += np.real(Jq_downfolded[iq] * phase /
                                             self.nqpt)
        return JR_downfolded

    def downfold_oneq(self, J):
        JMM = J[np.ix_(self.iM, self.iM)]
        JLL = J[np.ix_(self.iL, self.iL)]
        JLM = J[np.ix_(self.iL, self.iM)]
        JML = J[np.ix_(self.iM, self.iL)]
        Jn = JMM - JML @ np.linalg.inv(JLL) @ JLM
        return Jn


class JDownfolder_pickle():
    def __init__(self, inpath, metals, ligands, outpath):
        fname=os.path.join(inpath, 'TB2J.pickle')
        with open(fname, 'rb') as myfile:
            self.obj = pickle.load(myfile)

        self.atoms = self.obj['atoms']
        self.magnetic_elements= metals
        self.iM=[]
        self.iL=[]
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                self.iM.append(i)
            elif sym in ligands:
                self.iL.append(i)
        self.ind_mag_atoms=self.iM

        self.nM=len(self.iM)
        self.nL=len(self.iL)
        self.nsite=self.nM+self.nL

        atoms = self.obj['atoms']
        index_spin = self.obj['index_spin']
        ind_atoms = self.obj['ind_atoms']
        Jdict = self.obj['exchange_Jdict']
        nspin = len(index_spin)
        JR = defaultdict(lambda: np.zeros((nspin, nspin), dtype=float))
        for key, val in Jdict.items():
            R, i, j = key
            JR[R][i, j] = val

        Rlist = list(JR.keys())

        # build matrix form of JR
        nR = len(Rlist)
        JR2 = np.zeros((nR, nspin, nspin), dtype=float)
        for i, (key, val) in enumerate(JR.items()):
            JR2[i] = val

        # sum rule
        iR0 = np.argmin(np.linalg.norm(Rlist, axis=1))
        for i in range(self.nsite):
            sum_JRi=np.sum(np.sum(JR2, axis=0)[i])
            JR2[iR0][i, i] -= sum_JRi

        d = JDownfolder(JR2, Rlist, iM=self.iM, iL=self.iL, qmesh=[7, 7, 7])
        Jd = d.get_JR()

        self.Rlist=Rlist
        self.Rcut=None

        self._prepare_distance()
        self._prepare_index_spin()
        self.Jdict={}
        for iR, R in enumerate(d.Rlist):
            for i, ispin in enumerate(self.index_spin):
                for j, jspin in enumerate(self.index_spin):
                    if ispin>=0 and jspin>=0:
                        self.Jdict[(tuple(R), i, j)]=Jd[iR, i, j]


        io = SpinIO(atoms=atoms,
                    spinat=self.obj['spinat'],
                    charges=self.obj['charges'],
                    index_spin=self.index_spin,
                    colinear=True,
                    distance_dict=self.distance_dict,
                    exchange_Jdict=self.Jdict)

        io.write_all(outpath)

    def _prepare_distance(self):
        self.distance_dict = {}
        self.short_Rlist = []
        self.R_ijatom_dict = defaultdict(lambda: [])
        ind_matoms = self.ind_mag_atoms
        for ispin, iatom in enumerate(ind_matoms):
            for jspin, jatom in enumerate(ind_matoms):
                for R in self.Rlist:
                    pos_i = self.atoms.get_positions()[iatom]
                    pos_jR = self.atoms.get_positions()[jatom] + np.dot(
                        R, self.atoms.get_cell())
                    vec = pos_jR - pos_i
                    distance = np.sqrt(np.sum(vec**2))
                    if self.Rcut is None or distance < self.Rcut:
                        self.distance_dict[(tuple(R), ispin,
                                            jspin)] = (vec, distance)
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




def test(inpath, outpath, metals, ligands):
    #path = "/home/hexu/projects/NiCl2/vasp_inputs/TB2J_results"
    path = "/home/hexu/projects/NiCl2/AFM/TB2J_results"
    #fname = os.path.join(path, "TB2J.pickle")
    p=JDownfolder_pickle(inpath=inpath, metals=['Ni'], ligands=['Cl'], outpath='TB2J_results_downfolded')


if __name__=="__main__":
    test()
