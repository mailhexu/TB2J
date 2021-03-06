"""
Exchange from Green's function
"""

from collections import defaultdict
import os
import shutil
import numpy as np
from TB2J.green import TBGreen
from TB2J.utils import symbol_number, read_basis
from TB2J.myTB import MyTB
from ase.io import read
from TB2J.utils import auto_assign_basis_name
from TB2J.io_exchange import SpinIO
import progressbar
from functools import lru_cache
from .exchange import ExchangeCL
from .utils import simpson_nonuniform


class ExchangeCL2(ExchangeCL):
    def set_tbmodels(self, tbmodels):
        """
        only difference is a colinear tag.
        """
        self.tbmodel_up, self.tbmodel_dn = tbmodels
        self.Gup = TBGreen(self.tbmodel_up, self.kmesh, self.efermi, use_cache=self._use_cache, cache_path='TB2J_results/cache/spinup')
        self.Gdn = TBGreen(self.tbmodel_dn, self.kmesh, self.efermi, use_cache=self._use_cache, cache_path='TB2J_results/cache/spindn')
        self.norb = self.Gup.norb
        self.nbasis = self.Gup.nbasis + self.Gdn.nbasis
        self.rho_up_list=[]
        self.rho_dn_list=[]
        self.rho_up = np.zeros((self.norb, self.norb), dtype=float)
        self.rho_dn = np.zeros((self.norb, self.norb), dtype=float)
        self.Jorb_list=defaultdict(lambda:[])
        self.JJ_list=defaultdict(lambda:[])
        self.JJ = defaultdict(lambda: 0.0j)
        self.Jorb = defaultdict(lambda: 0.0j)
        self.HR0_up = self.Gup.H0
        self.HR0_dn = self.Gdn.H0
        self.Delta = self.HR0_up - self.HR0_dn
        if self.Gup.is_orthogonal and self.Gdn.is_orthogonal:
            self.is_orthogonal =True
        else:
            self.is_orthogonal = False
            #self.S0=self.Gup.S0
        self._is_colinear = True

        self.exchange_Jdict = {}
        self.exchange_Jdict_orb={}

        self.biquadratic = False

    def get_Delta(self, iatom):
        orbs = self.iorb(iatom)
        return self.Delta[np.ix_(orbs, orbs)]

    def GR_atom(self, GR, iatom, jatom):
        """Given a green's function matrix, return the [iatom, jatom] component.

        :param GR:  Green's function matrix
        :param iatom:  index of atom i
        :param jatom:  index of atom j
        :returns:  G_ij
        :rtype:  complex matrix.
        """
        orbi = self.iorb(iatom)
        orbj = self.iorb(jatom)
        return GR[np.ix_(orbi, orbj)]

    def get_A_ijR(self, Gup, Gdn, iatom, jatom):
        Rij_done=set()
        for R, ijpairs in self.R_ijatom_dict.items():
            if (iatom, jatom) in ijpairs and (R, iatom, jatom) not in Rij_done:
                Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                Rm = tuple(-x for x in R)
                Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                tmp = 0.0j
                #t = self.get_Delta(iatom) @ Gij_up @ self.get_Delta(jatom) @ Gji_dn
                t = np.einsum('ij, ji-> ij',
                              np.matmul(self.get_Delta(iatom), Gij_up),
                              np.matmul(self.get_Delta(jatom), Gji_dn))
                if self.biquadratic:
                    A = np.einsum('ij, ji-> ij',
                              np.matmul(self.get_Delta(iatom), Gij_up),
                              np.matmul(self.get_Delta(jatom), Gji_up))
                    C = np.einsum('ij, ji-> ij',
                              np.matmul(self.get_Delta(iatom), Gij_down),
                              np.matmul(self.get_Delta(jatom), Gji_down))
                tmp = np.sum(t)
                self.Jorb_list[(R, iatom, jatom)].append( t / (4.0 * np.pi))
                self.JJ_list[(R, iatom, jatom)].append(tmp  / (4.0 * np.pi))
                Rij_done.add((R,iatom, jatom))
                if (Rm,jatom, iatom) not in Rij_done:
                    self.Jorb_list[(Rm, jatom, iatom)].append( t / (4.0 * np.pi))
                    self.JJ_list[(Rm, jatom, iatom)].append(tmp  / (4.0 * np.pi))
                    Rij_done.add((Rm,jatom, iatom))

    def get_all_A(self, Gup, Gdn):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        :param de: energy step.
        """
        Rij_done=set()
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                if (R, iatom, jatom) not in Rij_done:
                    Rm = tuple(-x for x in R)
                    if (Rm,jatom, iatom) in Rij_done:
                        raise KeyError(f"Strange (Rm, jatom, iatom) has already been calculated! {(Rm, jatom, iatom)}")
                    Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                    Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                    tmp = 0.0j
                    #t = self.get_Delta(iatom) @ Gij_up @ self.get_Delta(jatom) @ Gji_dn
                    t = np.einsum('ij, ji-> ij',
                                  np.matmul(self.get_Delta(iatom), Gij_up),
                                  np.matmul(self.get_Delta(jatom), Gji_dn))
                    tmp = np.sum(t)
                    self.Jorb_list[(R, iatom, jatom)].append( t/ (4.0 * np.pi))
                    self.JJ_list[(R, iatom, jatom)].append(tmp/ (4.0 * np.pi))
                    Rij_done.add((R,iatom, jatom))
                    if (Rm,jatom, iatom)  not in Rij_done:
                        self.Jorb_list[(Rm, jatom, iatom)].append(t/(4.0 * np.pi))
                        self.JJ_list[(Rm, jatom, iatom)].append(tmp/(4.0 * np.pi))
                        Rij_done.add((Rm,jatom, iatom))
                            

    def A_to_Jtensor(self):
        for key, val in self.JJ.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            Jij = np.imag(val) / np.sign(
                np.dot(self.spinat[iatom], self.spinat[jatom]))
            Jorbij=np.imag(self.Jorb[key])/np.sign(
                np.dot(self.spinat[iatom], self.spinat[jatom]))
            if is_nonself:
                self.exchange_Jdict[keyspin] = Jij
                self.exchange_Jdict_orb[keyspin] =Jorbij

    def get_rho_e(self, rho_up, rho_dn):
        #self.rho_up_list.append(-1.0 / np.pi * np.imag(rho_up[(0,0,0)]))
        #self.rho_dn_list.append(-1.0 / np.pi * np.imag(rho_dn[(0,0,0)]))
        self.rho_up_list.append(-1.0 / np.pi * rho_up[(0,0,0)])
        self.rho_dn_list.append(-1.0 / np.pi * rho_dn[(0,0,0)])


    def get_rho_atom(self):
        """
        charges and spins from density matrices
        """
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tup = np.real(np.trace(self.rho_up[np.ix_(iorb, iorb)]))
            tdn = np.real(np.trace(self.rho_dn[np.ix_(iorb, iorb)]))
            # *2 because there is a 1/2 in the paui_block_all function
            self.charges[iatom] = tup + tdn
            self.spinat[iatom, 2] = tup - tdn

    def finalize(self):
        self.Gup.clean_cache()
        self.Gdn.clean_cache()
        path='TB2J_results/cache'
        if os.path.exists(path):
            shutil.rmtree(path)

    def integrate(self):
        self.rho_up=np.imag(simpson_nonuniform(self.contour.path, self.rho_up_list))
        self.rho_dn=np.imag(simpson_nonuniform(self.contour.path, self.rho_dn_list))
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                self.Jorb[(R, iatom, jatom)]=simpson_nonuniform(self.contour.path,
                             self.Jorb_list[(R, iatom, jatom)])
                self.JJ[(R, iatom, jatom)]=simpson_nonuniform(self.contour.path,
                    self.JJ_list[(R, iatom, jatom)])

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        widgets = [
            ' [',
            progressbar.Timer(),
            '] ',
            progressbar.Bar(),
            ' (',
            progressbar.ETA(),
            ') ',
        ]
        bar = progressbar.ProgressBar(
            maxval=len(self.contour.path), widgets=widgets)
        bar.start()
        for ie,e in enumerate(self.contour.path):
            bar.update(ie)
            e = self.contour.path[ie]
            GR_up, rho_up = self.Gup.get_GR(self.short_Rlist, energy=e, get_rho=True)
            GR_dn, rho_dn = self.Gdn.get_GR(self.short_Rlist, energy=e, get_rho=True)
            self.get_rho_e(rho_up, rho_dn)
            self.get_all_A(GR_up, GR_dn)
        self.integrate()
        self.get_rho_atom()
        self.A_to_Jtensor()
        bar.finish()

    def write_output(self, path='TB2J_results'):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            exchange_Jdict_orb = self.exchange_Jdict_orb,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            Jani_dict=None,
            biquadratic_Jdict=None,
            description=self.description,
        )
        output.write_all(path=path)
