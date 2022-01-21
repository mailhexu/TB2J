from collections import defaultdict, OrderedDict

import os
import numpy as np
from TB2J.green import TBGreen
from TB2J.pauli import (pauli_block_all, pauli_block_sigma_norm, pauli_mat)
from TB2J.utils import symbol_number, read_basis, kmesh_to_R
from TB2J.myTB import MyTB
from ase.io import read
from TB2J.utils import auto_assign_basis_name
from TB2J.io_exchange import SpinIO
from tqdm import tqdm
from p_tqdm import p_map
from functools import lru_cache
from TB2J.contour import Contour
from TB2J.orbmap import map_orbs_matrix
from pathos.multiprocessing import ProcessPool
import pickle
from TB2J.utils import simpson_nonuniform, trapezoidal_nonuniform


class Exchange():
    def __init__(
            self,
            tbmodels,
            atoms,
            efermi,
            basis=None,
            magnetic_elements=[],
            include_orbs={},
            kmesh=[4, 4, 4],
            emin=-15,  # integration lower bound, relative to fermi energy
            # integration upper bound. Should be 0 (fermi energy). But DFT codes define Fermi energy in various ways.
        emax=0.05,
            nz=100,
            # the delta in the (i delta) in green's function to prevent divergence
            height=0.5,
            nz1=150,  # grid from emin to emin+(i delta)
            nz2=300,  # grid from emin +(i delta) to emax+(i delta)
            nz3=150,  # grid from emax + (i delta) to emax
            exclude_orbs=[],  #
            ne=None,  # number of electrons in Wannier function.
            Rcut=None,  # Rcut.
            use_cache=False,
            np=1,
            description='',
            orb_decomposition=False):

        self.atoms = atoms
        self.efermi = efermi
        self.emin = emin
        self.emax = emax
        self.nz = nz
        self.height = height
        self.nz1 = nz1
        self.nz2 = nz2
        self.nz3 = nz3
        if nz is None:
            self.nz = nz1 + nz2 + nz3
        self._prepare_kmesh(kmesh)
        self.Rcut = Rcut
        self.basis = basis
        self.magnetic_elements = magnetic_elements
        self.include_orbs = include_orbs

        self.exclude_orbs = exclude_orbs
        self.ne = ne
        self._use_cache = use_cache
        self.np = np

        self.orb_decomposition = orb_decomposition

        self.set_tbmodels(tbmodels)
        self._adjust_emin()
        self._prepare_elist(method='legendre')
        self._prepare_Rlist()
        self._prepare_basis()
        self._prepare_orb_dict()
        self._prepare_distance()

        # whether to calculate J and DMI with NJt method.
        self.calc_NJt = True
        # self._prepare_NijR()
        self.Ddict_NJT = None
        self.Jdict_NJT = None
        self._is_collinear = True
        self.has_elistc = False
        self.description = description
        self._clean_tbmodels()

        # self._prepare_Jorb_file()

    def _prepare_Jorb_file(self):
        os.makedirs(self.output_path, exist_ok=True)
        self.orbpath = os.path.join(self.output_path, 'OrbResolve')
        os.makedirs(self.orbpath, exist_ok=True)

    def _adjust_emin(self):
        self.emin = self.G.find_energy_ingap(rbound=self.efermi -
                                             5.0) - self.efermi

    def set_tbmodels(self, tbmodels):
        pass

    def _clean_tbmodels(self):
        del self.tbmodel
        del self.G.tbmodel

    def _prepare_kmesh(self, kmesh):
        for k in kmesh:
            self.kmesh = list(map(lambda x: x // 2 * 2 + 1, kmesh))

    def _prepare_elist(self, method='legendre'):
        """
        prepare list of energy for integration.
        The path has three segments:
         emin --1-> emin + 1j*height --2-> emax+1j*height --3-> emax
        """
        self.contour = Contour(self.emin, self.emax)
        if method.lower() == 'rectangle':
            self.contour.build_path_rectangle(height=self.height,
                                              nz1=self.nz1,
                                              nz2=self.nz2,
                                              nz3=self.nz3)
        elif method.lower() == 'semicircle':
            self.contour.build_path_semicircle(npoints=self.nz, endpoint=True)
        elif method.lower() == 'legendre':
            self.contour.build_path_legendre(npoints=self.nz, endpoint=True)
        else:
            raise ValueError(f"The path cannot be of type {method}.")

    def _prepare_Rlist(self):
        """
        prepare R list for J(i, j, R)
        [-Rx, Rx] * [-Ry, Ry] * [-Rz, Rz]
        """
        self.Rlist = kmesh_to_R(self.kmesh)

    def _prepare_basis(self):
        if self.basis is None:
            pass

    def _prepare_orb_dict(self):
        """
        generate self.ind_mag_atoms and self.orb_dict
        """
        # adict: dictionary of {'Fe': ['dxy', 'dyz', ...], ...}
        adict = OrderedDict()
        # orb_dict: {ind_atom:[ind_orb,1,2], ...}
        self.orb_dict = {}
        # labels: {0:{dxy, ...}}
        self.labels = {}
        # magnetic atoms index
        self.ind_mag_atoms = []

        sdict = symbol_number(self.atoms)

        for i, base in enumerate(self.basis):
            if i not in self.exclude_orbs:
                # e.g. Fe2, dxy, _, _
                if isinstance(base, str):
                    atom_sym, orb_sym = base.split('|')[:2]
                else:
                    atom_sym, orb_sym = base[:2]

                if atom_sym in adict:
                    adict[atom_sym].append(orb_sym)
                else:
                    adict[atom_sym] = [orb_sym]
                iatom = sdict[atom_sym]
                if iatom not in self.orb_dict:
                    self.orb_dict[iatom] = [i]
                    self.labels[iatom] = [orb_sym]
                else:
                    self.orb_dict[iatom] += [i]
                    self.labels[iatom] += [orb_sym]

        #self.orb_slice = []

        # for iatom in range(len(self.atoms)):
        #    if iatom in self.orb_dict:
        #        self.orb_slice.append(
        #            slice(
        #                self.orb_dict[iatom][0],
        #                self.orb_dict[iatom][-1] + 1,
        #            ))
        #    else:
        #        self.orb_slice.append(slice(0, 0))

        #self.orb_slice = np.array(self.orb_slice)

        # index of magnetic atoms
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                self.ind_mag_atoms.append(i)

        # sanity check to see if some magnetic atom has no orbital.
        for iatom in self.ind_mag_atoms:
            if iatom not in self.orb_dict:
                raise ValueError(
                    f"""Cannot find any orbital for atom {iatom}, which is supposed to be magnetic. Please check the Wannier functions."""
                )

        self._spin_dict = {}
        self._atom_dict = {}
        for ispin, iatom in enumerate(self.ind_mag_atoms):
            self._spin_dict[iatom] = ispin
            self._atom_dict[ispin] = iatom

        self._prepare_orb_mmat()

    def _prepare_orb_mmat(self):
        self.mmats = {}
        self.orbital_names = {}
        self.norb_reduced = {}
        if self.backend_name == "SIESTA":
            syms = self.atoms.get_chemical_symbols()
            for iatom, orbs in self.labels.items():
                if (self.include_orbs
                        is not None) and syms[iatom] in self.include_orbs:
                    mmat, reduced_orbs = map_orbs_matrix(
                        orbs,
                        spinor=not (self._is_collinear),
                        include_only=self.include_orbs[syms[iatom]])
                else:
                    mmat, reduced_orbs = map_orbs_matrix(
                        orbs,
                        spinor=not (self._is_collinear),
                        include_only=None)

                self.mmats[iatom] = mmat
                self.orbital_names[iatom] = reduced_orbs
                self.norb_reduced[iatom] = len(reduced_orbs) // 2
        else:
            self.orbital_names = self.labels
            for iatom, orbs in self.labels.items():
                self.norb_reduced[iatom] = len(orbs) // 2

    def ispin(self, iatom):
        return self._spin_dict[iatom]

    def iatom(self, ispin):
        return self._atom_dict[ispin]

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

    def iorb(self, iatom):
        """
        given an index of atom, return all the orbital indices of it.
        :param iatom:  index of atom.
        """
        return np.array(self.orb_dict[iatom], dtype=int)

    def simplify_orbital_contributions(self, Jorbij, iatom, jatom):
        """
        sum up the contribution of all the orbitals with same (n,l,m)
        """
        if self.backend_name == 'SIESTA':
            mmat_i = self.mmats[iatom]
            mmat_j = self.mmats[jatom]
            Jorbij = mmat_i.T @ Jorbij @ mmat_j
        return Jorbij

    def calculate_all(self):
        raise NotImplementedError(
            "calculate all is not implemented in a abstract class. ")

    def write_output(self):
        raise NotImplementedError(
            "write_output is not implemented in a abstract class. ")


class ExchangeNCL(Exchange):
    """
    Non-collinear exchange
    """
    def set_tbmodels(self, tbmodels):
        """
        tbmodels should be in spinor form.
        The basis should be orb1_up, orb2_up,...orbn_up, orb1_dn, orb2_dn....
        """
        self.tbmodel = tbmodels
        self.backend_name = self.tbmodel.name
        # TODO: check if tbmodels are really a tbmodel with SOC.
        self.G = TBGreen(self.tbmodel,
                         self.kmesh,
                         self.efermi,
                         use_cache=self._use_cache,
                         nproc=self.np)
        self.norb = self.G.norb
        self.nbasis = self.G.nbasis
        self.rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        self.A_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.A_ijR_orb = dict()
        self.HR0 = self.G.H0
        self._is_collinear = False
        self.Pdict = {}

    def _prepare_NijR(self):
        self.N = {}
        for R in self.Rlist:
            self.N[R] = np.zeros((self.nbasis, self.nbasis), dtype=complex)

    def _prepare_Patom(self):
        for iatom in self.ind_mag_atoms:
            self.Pdict[iatom] = pauli_block_sigma_norm(self.get_H_atom(iatom))

    def get_H_atom(self, iatom):
        orbs = self.iorb(iatom)
        # return self.HR0[self.orb_slice[iatom], self.orb_slice[iatom]]
        return self.HR0[np.ix_(orbs, orbs)]

    def get_P_iatom(self, iatom):
        """ Calculate the norm of the Hamiltonian vector.
        For each atom, the local hamiltonian of each orbital H(2*2 matrix)
        can be written as H0* I + H1* sigma1 + H2*sigma2 + H3 *sigma3
        where sigma is a Pauli matrix. return the norm of (H1, H2, H3) vector
        :param iatom: index of atom
        :returns: a matrix of norms P. P[i, j] is for orbital i and j.
        :rtype: complex matrix of shape norb_i * norb_i.
        """
        if self.Pdict == {}:
            self._prepare_Patom()
        return self.Pdict[iatom]

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
        # return GR[self.orb_slice[iatom], self.orb_slice[jatom]]

    def get_A_ijR(self, G, R, iatom, jatom):
        """ calculate A from G for a energy slice (de).
        It take the
        .. math::
           A^{uv} = p T^u p T^v dE / pi

        where u, v are I, x, y, z (index 0, 1,2,3). p(i) = self.get_P_iatom(iatom)
        T^u(ijR)  (u=0,1,2,3) = pauli_block_all(G)

        :param G: Green's function for all R, i, j.
        :param iatom: i
        :param jatom: j
        :param de:  energy step. used for integeration
        :returns: a matrix of A_ij(u, v), where u, v =(0)0, x(1), y(2), z(3)
        :rtype:  4*4 matrix
        """
        GR = G[R]
        Gij = self.GR_atom(GR, iatom, jatom)
        Gij_Ixyz = pauli_block_all(Gij)

        # G(j, i, -R)
        Rm = tuple(-x for x in R)
        GRm = G[Rm]
        Gji = self.GR_atom(GRm, jatom, iatom)
        Gji_Ixyz = pauli_block_all(Gji)

        tmp = np.zeros((4, 4), dtype=complex)
        if self.orb_decomposition:
            ni = self.norb_reduced[iatom]
            nj = self.norb_reduced[jatom]
            torb = np.zeros((4, 4, ni, nj), dtype=complex)
            # a, b in (0,x,y,z)
            for a in range(4):
                piGij = self.get_P_iatom(iatom) @ Gij_Ixyz[a]
                for b in range(4):
                    pjGji = self.get_P_iatom(jatom) @ Gji_Ixyz[b]
                    torb[a, b] = self.simplify_orbital_contributions(
                        np.einsum('ij, ji -> ij', piGij, pjGji) / np.pi, iatom,
                        jatom)
                    tmp[a, b] = np.sum(torb[a, b])
        else:
            for a in range(4):
                pGp = self.get_P_iatom(iatom) @ Gij_Ixyz[a] @ self.get_P_iatom(
                    jatom)
                for b in range(4):
                    AijRab = pGp @ Gji_Ixyz[b]
                    tmp[a, b] = np.trace(AijRab) / np.pi
            torb = None
        return tmp, torb

    def get_all_A(self, G):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        """
        A_ijR_list = {}
        Aorb_ijR_list = {}
        for iR, R in enumerate(self.R_ijatom_dict):
            for (iatom, jatom) in self.R_ijatom_dict[R]:
                A, A_orb = self.get_A_ijR(G, R, iatom, jatom)
                A_ijR_list[(R, iatom, jatom)] = A
                Aorb_ijR_list[(R, iatom, jatom)] = A_orb
        return A_ijR_list, Aorb_ijR_list

    def A_to_Jtensor_orb(self):
        """
        convert the orbital composition of A into J, DMI, Jani
        """
        self.Jiso_orb = {}
        self.Jani_orb = {}
        self.DMI_orb = {}

        if self.orb_decomposition:
            for key, val in self.A_ijR_orb.items():
                R, iatom, jatom = key
                Rm = tuple(-x for x in R)
                valm = self.A_ijR_orb[(Rm, jatom, iatom)]
                ni = self.norb_reduced[iatom]
                nj = self.norb_reduced[jatom]

                is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                ispin = self.ispin(iatom)
                jspin = self.ispin(jatom)
                keyspin = (R, ispin, jspin)

                # isotropic J
                Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

                # off-diagonal anisotropic exchange
                Ja = np.zeros((3, 3, ni, nj), dtype=float)
                for i in range(3):
                    for j in range(3):
                        Ja[i,
                           j] = np.imag(val[i + 1, j + 1] + valm[i + 1, j + 1])
                # DMI

                Dtmp = np.zeros((3, ni, nj), dtype=float)
                for i in range(3):
                    Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

                if is_nonself:
                    self.Jiso_orb[keyspin] = Jiso
                    self.Jani_orb[keyspin] = Ja
                    self.DMI_orb[keyspin] = Dtmp

    def A_to_Jtensor(self):
        """
        Calculate J tensors from A.
        If we assume the exchange can be written as a bilinear tensor form,
        J_{isotropic} = Tr Im (A^{00} - A^{xx} - A^{yy} - A^{zz})
        J_{anisotropic}_uv = Tr Im (2A)
        DMI =  Tr Re (A^{0z} - A^{z0} )
        """
        self.Jani = {}
        self.DMI = {}

        self.Jprime = {}
        self.B = {}
        self.exchange_Jdict = {}
        self.Jiso_orb = {}

        self.debug_dict = {'DMI2': {}}

        for key, val in self.A_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key

            Rm = tuple(-x for x in R)
            valm = self.A_ijR[(Rm, jatom, iatom)]

            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)

            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            Jiso = 0.0
            Ja = np.zeros((3, 3), dtype=float)
            Dtmp = np.zeros(3, dtype=float)
            Dtmp2 = np.zeros(3, dtype=float)
            # Heisenberg like J.
            Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

            if is_nonself:
                self.exchange_Jdict[keyspin] = Jiso

            # off-diagonal anisotropic exchange
            for i in range(3):
                for j in range(3):
                    Ja[i, j] = np.imag(val[i + 1, j + 1] + valm[i + 1, j + 1])
            if is_nonself:
                self.Jani[keyspin] = Ja

            # DMI
            for i in range(3):
                Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

            # Dx = Jyz-Jzy
            # Dy = Jzx-Jxz
            # Dz = Jxy-Jyx
            Dtmp2[0] = np.imag(val[2, 3] - val[3, 2])
            Dtmp2[1] = np.imag(val[3, 1] - val[1, 3])
            Dtmp2[2] = np.imag(val[1, 2] - val[2, 1])
            if is_nonself:
                self.DMI[keyspin] = Dtmp
                self.debug_dict['DMI2'][keyspin] = Dtmp2

            # isotropic exchange into bilinear and biqudratic parts:
            # Jprime SiSj and B (SiSj)^2
            if is_nonself:
                Si = self.spinat[iatom]
                Sj = self.spinat[jatom]
                Jprime = np.imag(val[0, 0] - val[3, 3]) - 2 * np.sign(
                    np.dot(Si, Sj)) * np.imag(val[3, 3])
                # Jprime = np.imag(val[0, 0] - 3*val[3, 3])
                B = np.imag(val[3, 3])
                self.B[keyspin] = Jprime, B

    def get_N_e(self, GR, de):
        """
        calcualte density matrix for all R,i, j
        """
        self.N = defaultdict(lambda: 0.0)
        for R, G in GR.items():
            self.N[R] += -1.0 / np.pi * np.imag(G * de)

    def get_rho_e(self, rhoR):
        """ add component to density matrix from a green's function
        :param GR: Green's funciton in real space.
        """
        return -1.0 / np.pi * rhoR[0, 0, 0]

    def get_total_charges(self):
        return np.sum(np.imag(np.diag(self.rho)))

    def get_rho_atom(self):
        """
        calculate charge and spin for each atom.
        """
        rho = {}
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tmp = self.rho[np.ix_(iorb, iorb)]
            # *2 because there is a 1/2 in the paui_block_all function
            rho[iatom] = np.array(
                [np.trace(x) * 2 for x in pauli_block_all(tmp)])
            self.charges[iatom] = np.imag(rho[iatom][0])
            self.spinat[iatom, :] = np.imag(rho[iatom][1:])
        self.rho_dict = rho
        return self.rho_dict

    def calculate_DMI_NJT(self):
        """
        calculate exchange and DMI with the
        D(i,j) =
        """
        Ddict_NJT = {}
        Jdict_NJT = {}
        for R in self.short_Rlist:
            N = self.N[tuple(-np.array(R))]  # density matrix
            t = self.tbmodel.get_hamR(R)  # hopping parameter
            for iatom in self.ind_mag_atoms:
                orbi = self.iorb(iatom)
                ni = len(orbi)
                for jatom in self.ind_mag_atoms:
                    orbj = self.iorb(jatom)
                    nj = len(orbj)
                    Nji = N[np.ix_(orbj, orbi)]
                    tij = t[np.ix_(orbi, orbj)]
                    D = np.zeros(3, dtype=float)
                    J = np.zeros(3, dtype=float)
                    for dim in range(3):
                        # S_i = pauli_mat(ni, dim +
                        #                1)  #*self.rho[np.ix_(orbi, orbi)]
                        # S_j = pauli_mat(nj, dim +
                        #                1)  #*self.rho[np.ix_(orbj, orbj)]
                        # TODO: Note that rho is complex, not the imaginary part
                        S_i = pauli_mat(ni, dim + 1) * self.rho[np.ix_(
                            orbi, orbi)]
                        S_j = pauli_mat(nj, dim + 1) * self.rho[np.ix_(
                            orbj, orbj)]

                        # [S, t]+  = Si tij + tij Sj, where
                        # Si and Sj are the spin operator
                        # Here we do not have L operator, so J-> S
                        Jt = np.matmul(S_i, tij) + np.matmul(tij, S_j)

                        Jtminus = np.matmul(S_i, tij) - np.matmul(tij, S_j)
                        # D = -1/2 Tr Nji [J, tij]
                        # Trace over spin and orb
                        D[dim] = -0.5 * np.imag(np.trace(np.matmul(Nji, Jt)))
                        J[dim] = -0.5 * np.imag(
                            np.trace(np.matmul(Nji, Jtminus)))
                    ispin = self.ispin(iatom)
                    jspin = self.ispin(jatom)
                    Ddict_NJT[(R, ispin, jspin)] = D
                    Jdict_NJT[(R, ispin, jspin)] = J
        self.Jdict_NJT = Jdict_NJT
        self.Ddict_NJT = Ddict_NJT
        return Ddict_NJT

    def integrate(self, rhoRs, AijRs, AijRs_orb=None, method='simpson'):
        """
        AijRs: a list of AijR, 
        wherer AijR: array of ((nR, n, n, 4,4), dtype=complex)
        """
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == 'simpson':
            integrate = simpson_nonuniform

        self.rho = integrate(self.contour.path, rhoRs)
        for iR, R in enumerate(self.R_ijatom_dict):
            for (iatom, jatom) in self.R_ijatom_dict[R]:
                f = AijRs[(R, iatom, jatom)]
                self.A_ijR[(R, iatom, jatom)] = integrate(self.contour.path, f)
                if self.orb_decomposition:
                    self.A_ijR_orb[(R, iatom, jatom)] = integrate(
                        self.contour.path, AijRs_orb[(R, iatom, jatom)])

    def get_AijR_rhoR(self, e):
        GR, rhoR = self.G.get_GR(self.short_Rlist, energy=e, get_rho=True)
        AijR, AijR_orb = self.get_all_A(GR)
        return AijR, AijR_orb, self.get_rho_e(rhoR)

    def save_AijR(self, AijRs, fname):
        result = dict(path=self.contour.path, AijRs=AijRs)
        with open(fname, 'wb') as myfile:
            pickle.dump(result, myfile)

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        rhoRs = []
        AijRs = {}

        AijRs_orb = {}

        npole = len(self.contour.path)
        if self.np > 1:
            results = p_map(self.get_AijR_rhoR,
                            self.contour.path,
                            num_cpus=self.np)
        else:
            results = map(self.get_AijR_rhoR,
                          tqdm(self.contour.path, total=npole))

        for i, result in enumerate(results):
            for iR, R in enumerate(self.R_ijatom_dict):
                for (iatom, jatom) in self.R_ijatom_dict[R]:
                    if (R, iatom, jatom) in AijRs:
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom,
                                                                  jatom])
                        if self.orb_decomposition:
                            AijRs_orb[(R, iatom,
                                       jatom)].append(result[1][R, iatom,
                                                                jatom])

                    else:
                        AijRs[(R, iatom, jatom)] = []
                        AijRs[(R, iatom, jatom)].append(result[0][R, iatom,
                                                                  jatom])
                        if self.orb_decomposition:
                            AijRs_orb[(R, iatom, jatom)] = []
                            AijRs_orb[(R, iatom,
                                       jatom)].append(result[1][R, iatom,
                                                                jatom])
            rhoRs.append(result[2])

        # self.save_AijRs(AijRs)
        self.integrate(rhoRs, AijRs, AijRs_orb)

        self.get_rho_atom()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

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

    def write_output(self, path='TB2J_results'):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=False,
            orbital_names=self.orbital_names,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            Jiso_orb=self.Jiso_orb,
            dmi_ddict=self.DMI,
            Jani_dict=self.Jani,
            DMI_orb=self.DMI_orb,
            Jani_orb=self.Jani_orb,
            NJT_Jdict=self.Jdict_NJT,
            NJT_ddict=self.Ddict_NJT,
            biquadratic_Jdict=self.B,
            debug_dict=self.debug_dict,
            description=self.description,
        )
        output.write_all(path=path)
        # with open("TB2J_results/J_orb.pickle", 'wb') as myfile:
        #    pickle.dump({'Jiso_orb': self.Jiso_orb,
        #                 'DMI_orb': self.DMI_orb, 'Jani_orb': self.Jani_orb}, myfile)

    def finalize(self):
        self.G.clean_cache()

    def run(self, path='TB2J_results'):
        self.calculate_all()
        self.write_output(path=path)
        self.finalize()


class ExchangeCL(ExchangeNCL):
    def set_tbmodels(self, tbmodel):
        """
        only difference is a colinear tag.
        """
        super().set_tbmodels(tbmodel)
        self._is_collinear = True

    def write_output(self, path='TB2J_results'):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            orbital_names=self.orbital_names,
            colinear=True,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            biquadratic_Jdict=self.B,
            description=self.description,
        )
        output.write_all(path=path)
