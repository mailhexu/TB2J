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
from tqdm import tqdm
from p_tqdm import p_map
from functools import lru_cache
from .exchange import ExchangeCL
from .utils import simpson_nonuniform, trapezoidal_nonuniform
from pathos.multiprocessing import ProcessPool
from functools import partial


class ExchangeCL2(ExchangeCL):
    def set_tbmodels(self, tbmodels):
        """
        only difference is a colinear tag.
        """
        self.tbmodel_up, self.tbmodel_dn = tbmodels
        self.backend_name = self.tbmodel_up.name
        self.Gup = TBGreen(self.tbmodel_up,
                           self.kmesh,
                           self.efermi,
                           use_cache=self._use_cache,
                           cache_path='TB2J_results/cache/spinup',
                           nproc=self.np)
        self.Gdn = TBGreen(self.tbmodel_dn,
                           self.kmesh,
                           self.efermi,
                           use_cache=self._use_cache,
                           cache_path='TB2J_results/cache/spindn',
                           nproc=self.np)
        self.norb = self.Gup.norb
        self.nbasis = self.Gup.nbasis + self.Gdn.nbasis
        self.rho_up_list = []
        self.rho_dn_list = []
        self.rho_up = np.zeros((self.norb, self.norb), dtype=float)
        self.rho_dn = np.zeros((self.norb, self.norb), dtype=float)
        self.Jorb_list = defaultdict(lambda: [])
        self.JJ_list = defaultdict(lambda: [])
        self.JJ = defaultdict(lambda: 0.0j)
        self.Jorb = defaultdict(lambda: 0.0j)
        self.HR0_up = self.Gup.H0
        self.HR0_dn = self.Gdn.H0
        self.Delta = self.HR0_up - self.HR0_dn
        if self.Gup.is_orthogonal and self.Gdn.is_orthogonal:
            self.is_orthogonal = True
        else:
            self.is_orthogonal = False
            # self.S0=self.Gup.S0

        self.exchange_Jdict = {}
        self.exchange_Jdict_orb = {}

        self.biquadratic = False

        self._is_collinear=True

    def _clean_tbmodels(self):
        del self.tbmodel_up
        del self.tbmodel_dn
        del self.Gup.tbmodel
        del self.Gdn.tbmodel

    def _adjust_emin(self):
        emin_up = self.Gup.find_energy_ingap(rbound=self.efermi -
                                             5.0) - self.efermi
        emin_dn = self.Gdn.find_energy_ingap(rbound=self.efermi -
                                             5.0) - self.efermi
        self.emin = min(emin_up, emin_dn)
        print(f"A gap is found at {self.emin}, set emin to it.")

    def get_Delta(self, iatom):
        orbs = self.iorb(iatom)
        return self.Delta[np.ix_(orbs, orbs)]
        #s = self.orb_slice[iatom]
        #return self.Delta[s, s]

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
        #return GR[self.orb_slice[iatom], self.orb_slice[jatom]]

    def get_A_ijR(self, Gup, Gdn, iatom, jatom):
        Rij_done = set()
        Jorb_list = dict()
        JJ_list = dict()
        for R, ijpairs in self.R_ijatom_dict.items():
            if (iatom, jatom) in ijpairs and (R, iatom, jatom) not in Rij_done:
                Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                Rm = tuple(-x for x in R)
                Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                tmp = 0.0j
                Deltai = self.get_Delta(iatom)
                Deltaj = self.get_Delta(jatom)
                t = np.einsum('ij, ji-> ij', np.matmul(Deltai, Gij_up),
                              np.matmul(Deltaj, Gji_dn))

                if self.biquadratic:
                    A = np.einsum('ij, ji-> ij', np.matmul(Deltai, Gij_up),
                                  np.matmul(Deltaj, Gji_up))
                    C = np.einsum('ij, ji-> ij', np.matmul(Deltai, Gij_down),
                                  np.matmul(Deltaj, Gji_down))
                tmp = np.sum(t)
                self.Jorb_list[(R, iatom, jatom)].append(t / (4.0 * np.pi))
                self.JJ_list[(R, iatom, jatom)].append(tmp / (4.0 * np.pi))
                Rij_done.add((R, iatom, jatom))
                if (Rm, jatom, iatom) not in Rij_done:
                    Jorb_list[(Rm, jatom, iatom)] = t.T / (4.0 * np.pi)
                    JJ_list[(Rm, jatom, iatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((Rm, jatom, iatom))
        return Jorb_list, JJ_list

    def get_all_A(self, Gup, Gdn):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        :param de: energy step.
        """
        Rij_done = set()
        Jorb_list = dict()
        JJ_list = dict()
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                if (R, iatom, jatom) not in Rij_done:
                    Rm = tuple(-x for x in R)
                    if (Rm, jatom, iatom) in Rij_done:
                        raise KeyError(
                            f"Strange (Rm, jatom, iatom) has already been calculated! {(Rm, jatom, iatom)}"
                        )
                    Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                    Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                    tmp = 0.0j
                    #t = self.get_Delta(iatom) @ Gij_up @ self.get_Delta(jatom) @ Gji_dn
                    t = np.einsum('ij, ji-> ij',
                                  np.matmul(self.get_Delta(iatom), Gij_up),
                                  np.matmul(self.get_Delta(jatom), Gji_dn))
                    tmp = np.sum(t)
                    Jorb_list[(R, iatom, jatom)] = t / (4.0 * np.pi)
                    JJ_list[(R, iatom, jatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((R, iatom, jatom))
                    if (Rm, jatom, iatom) not in Rij_done:
                        Jorb_list[(Rm, jatom, iatom)] = t / (4.0 * np.pi)
                        JJ_list[(Rm, jatom, iatom)] = tmp / (4.0 * np.pi)
                        Rij_done.add((Rm, jatom, iatom))
        return Jorb_list, JJ_list

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
            Jorbij = np.imag(self.Jorb[key]) / np.sign(
                np.dot(self.spinat[iatom], self.spinat[jatom]))
            if is_nonself:
                self.exchange_Jdict[keyspin] = Jij
                self.exchange_Jdict_orb[
                    keyspin] = self.simplify_orbital_contributions(Jorbij, iatom, jatom)

    def simplify_orbital_contributions(self, Jorbij, iatom, jatom):
        """
        sum up the contribution of all the orbitals with same (n,l,m)
        """
        if self.backend_name == 'SIESTA':
            mmat_i = self.mmats[iatom]
            mmat_j = self.mmats[jatom]
            Jorbij = mmat_i.T @ Jorbij @ mmat_j
        return Jorbij

    def get_rho_e(self, rho_up, rho_dn):
        #self.rho_up_list.append(-1.0 / np.pi * np.imag(rho_up[(0,0,0)]))
        #self.rho_dn_list.append(-1.0 / np.pi * np.imag(rho_dn[(0,0,0)]))
        rup = -1.0 / np.pi * rho_up[(0, 0, 0)]
        rdn = -1.0 / np.pi * rho_dn[(0, 0, 0)]
        return rup, rdn

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
            self.charges[iatom] = tup + tdn
            self.spinat[iatom, 2] = tup - tdn

    def finalize(self):
        self.Gup.clean_cache()
        self.Gdn.clean_cache()
        path = 'TB2J_results/cache'
        if os.path.exists(path):
            shutil.rmtree(path)

    def integrate(self, method="simpson"):
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == 'simpson':
            integrate = simpson_nonuniform
        self.rho_up = np.imag(integrate(self.contour.path, self.rho_up_list))
        self.rho_dn = np.imag(integrate(self.contour.path, self.rho_dn_list))
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                self.Jorb[(R, iatom, jatom)] = integrate(
                    self.contour.path, self.Jorb_list[(R, iatom, jatom)])
                self.JJ[(R, iatom,
                         jatom)] = integrate(self.contour.path,
                                             self.JJ_list[(R, iatom, jatom)])

    def get_AijR_rhoR(self, e):
        GR_up, rho_up = self.Gup.get_GR(self.short_Rlist,
                                        energy=e,
                                        get_rho=True)
        GR_dn, rho_dn = self.Gdn.get_GR(self.short_Rlist,
                                        energy=e,
                                        get_rho=True)
        rup, rdn = self.get_rho_e(rho_up, rho_dn)
        Jorb_list, JJ_list = self.get_all_A(GR_up, GR_dn)
        return rup, rdn, Jorb_list, JJ_list

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        npole=len(self.contour.path)
        if self.np == 1:
            results = map(self.get_AijR_rhoR, tqdm(self.contour.path, total=npole))
        else:
            #pool = ProcessPool(nodes=self.np)
            #results = pool.map(self.get_AijR_rhoR ,self.contour.path)
            results = p_map(self.get_AijR_rhoR, self.contour.path, num_cpus=self.np)
        for i, result in enumerate(results):
            rup, rdn, Jorb_list, JJ_list = result
            self.rho_up_list.append(rup)
            self.rho_dn_list.append(rdn)
            for iR, R in enumerate(self.R_ijatom_dict):
                for (iatom, jatom) in self.R_ijatom_dict[R]:
                    key = (R, iatom, jatom)
                    self.Jorb_list[key].append(Jorb_list[key])
                    self.JJ_list[key].append(JJ_list[key])
        if self.np > 1:
            pass
            #pool.close()
            #pool.join()
            #pool.clear()
        self.integrate()
        self.get_rho_atom()
        self.A_to_Jtensor()

    def write_output(self, path='TB2J_results'):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            orbital_names=self.orbital_names,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            exchange_Jdict_orb=self.exchange_Jdict_orb,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            Jani_dict=None,
            biquadratic_Jdict=None,
            description=self.description,
        )
        output.write_all(path=path)
