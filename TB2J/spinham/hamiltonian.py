#!/usr/bin/env python
import numpy as np
from collections import Iterable, defaultdict
from .hamiltonian_terms import (ZeemanTerm, UniaxialMCATerm,
                                              ExchangeTerm, DMITerm, BilinearTerm)
from .constants import mu_B, gyromagnetic_ratio
from .supercell import SupercellMaker

from .spin_xml import SpinXmlParser, SpinXmlWriter
from .plot import plot_magnon_band


class SpinHamiltonian(object):
    def __init__(self,
                 cell=None,
                 pos=None,
                 xcart=None,
                 spinat=None,
                 zion=None,
                 Rlist=None,
                 iprim=None):
        if xcart is None:
            self.xcart = np.dot(pos, cell)
            self.pos=pos
        elif pos is None:
            self.pos = np.dot( xcart, np.linalg.inv(cell))
            self.xcart=xcart
        self.cell = cell
        self.xcart = pos
        self.spinat = spinat
        self.zion = zion

        self._spin = np.array(spinat) * mu_B
        self._map_to_magnetic_only()
        # spin interaction parameters

        # Homogeneous H eff_field
        self.has_external_hfield = False
        self.zeeman_H = np.zeros(3, dtype='float64')

        # uniaxial single ion anisotropy
        self.has_uniaxial_anistropy = False

        # cubic Anistropy (Not implemented yet)
        self.has_cubic_anistropy = False

        # Exchange interaction
        self.has_exchange = False
        self.exchange_Jdict = None

        # DM interaction
        self.has_dmi = False
        self.dmi_ddict = None

        # Dipole dipole interaction
        self.has_dipdip = False

        # bilinear term
        self.has_bilinear = False
        self.bilinear_dict = None

        # calculation parameters

        # calculation results

        # hamiltonian list
        self.hamiltonians = {}

        self._total_hessian_ijR = None

        self.gilbert_damping = np.ones(self.nspin) * 1.0
        self.gyro_ratio = np.ones(self.nspin) * gyromagnetic_ratio

        # internal variables:
        self._langevin_tmp = None

        self.sublattice = list(range(self.nspin))

        self.index_spin = None
        self.lattice = None

        self.iprim = iprim
        self.Rlist = Rlist

    def set_lattice(self, atoms, index_spin):
        self.lattice = atoms
        self.index_spin = index_spin

    @property
    def spin_positions(self):
        return self.pos

    def _map_to_magnetic_only(self):
        """
        select the sites with spin and re-index.
        """
        ms = np.linalg.norm(self._spin, axis=1)
        self.magsites = np.where(ms > 0.00001 * mu_B)[0]
        self.ms = ms[self.magsites]
        S = self._spin[self.magsites]
        self.nspin = len(S)
        # self.S = np.array(
        #    [S[i] / self.ms[i] for i in range(self.nspin)],
        #    dtype='float64')
        self.s = S / self.ms[:, None]

    @property
    def spin(self):
        self._spin[self.magsites] = np.multiply(self.ms[:, np.newaxis],
                                                self.s) / mu_B
        return self._spin

    @spin.setter
    def spin(self, spin):
        self._spin = np.array(spin) * mu_B
        self._map_to_magnetic_only()

    def normalize_S(self):
        """
        normalize so the norm of self.S[i] is 1
        """
        snorm = np.linalg.norm(self.s, axis=1)
        self.s /= np.expand_dims(snorm, axis=1)

    def set(self, gilbert_damping=None, gyro_ratio=None):
        """
        set parameters for simulation:
        args:
        ====================
        timestep: in ps
        temperature: in K, default 0K.
        damping factor: Gilbert damping factor, default: 0.01
        gyromagnetic_ratio: default 1.0.
        """
        if gilbert_damping is not None:
            self.gilbert_damping = np.array(gilbert_damping)
        if gyro_ratio is not None:
            self.gyro_ratio = np.array(gyro_ratio)

    def randomize_spin(self):
        raise NotImplementedError("not yet implemented")

    def set_exchange_ijR(self, exchange_Jdict):
        """
        J: [(i,j, R, J_{ijR})] J_{ijR} is a scalar
        """
        self.has_exchange = True
        self.exchange_Jdict = exchange_Jdict
        exchange = ExchangeTerm(self.exchange_Jdict, ms=self.ms)
        self.hamiltonians['exchange'] = exchange

    def set_dmi_ijR(self, dmi_ddict):
        """
        D: [(i,j, R, D_{ijR})], D_{ijR} is a vector
        """
        self.has_dmi = True
        self.dmi_ddict = dmi_ddict
        DMI = DMITerm(self.dmi_ddict, self.ms)
        self.hamiltonians['DMI'] = DMI

    def set_bilinear_ijR(self, bilinear_dict):
        self.has_bilinear= True
        self.bilinear_dict = bilinear_dict
        Bi = BilinearTerm(self.bilinear_dict, self.ms)
        self.hamiltonians['Bilinear'] = Bi


    def set_dipdip(self):
        """
        add the dipole dipole interaction term.
        """
        pass

    def set_external_hfield(self, H):
        """
        add external magnetic field. If H is a vector , it is homogenoues.
            Otherwise H should be given as a nspin*3 matrix.
        """
        self.has_external_hfield = True
        if isinstance(H, Iterable):
            self.H_ext = np.asarray(H)
        else:
            self.H_ext = np.ones([self.nspin, 3]) * H
        zeeman = ZeemanTerm(H=self.H_ext, ms=self.ms)
        self.hamiltonians['zeeman'] = zeeman

    def set_uniaxial_mca(self, k1, k1dir):
        """
        Add homogenoues uniaxial anisotropy
        """
        self.has_uniaxial_anistropy = True

        self.k1 = k1
        self.k1dir = k1dir
        umcaterm = UniaxialMCATerm(k1, k1dir, ms=self.ms)
        self.hamiltonians['UMCA'] = umcaterm

    def add_Hamiltonian_term(self, Hamiltonian_term, name=None):
        """
        add Hamiltonian term which is not pre_defined.
        """
        if name in self.hamiltonians:
            raise ValueError(
                'Hamiltonian name %s already defined. The defined names are %s'
                % (name, self.hamiltonians.keys()))
        else:
            self.hamiltonians[name] = Hamiltonian_term

    #@profile
    def get_effective_field(self, S):
        """
        calculate the effective field Heff=-1/ms * \partial H / \partial S
        Langevin term not included.
        """
        Heff = 0.0
        for ham in self.hamiltonians.values():
            Heff += ham.eff_field(S)
        return Heff

    def make_supercell(self, sc_matrix=None, supercell_maker=None):
        if supercell_maker is None:
            smaker = SupercellMaker(sc_matrix)
        else:
            smaker = supercell_maker

        sc_cell = smaker.sc_cell(np.array(self.cell))
        sc_pos = np.array(smaker.sc_pos(np.array(self.pos)))
        sc_zion = smaker.sc_trans_invariant(np.array(self.zion))

        if self.index_spin is not None:
            sc_index_spin = smaker.sc_trans_invariant(self.index_spin)

        sc_Rlist = np.repeat(smaker.R_sc, self.nspin, axis=0)
        sc_iprim = smaker.sc_trans_invariant(list(range(self.nspin)))

        sc_spinat = np.array(smaker.sc_trans_invariant(self.spinat))

        sc_ham = SpinHamiltonian(
            cell=sc_cell,
            pos=sc_pos,
            spinat=sc_spinat,
            zion=sc_zion,
            Rlist=sc_Rlist,
            iprim=sc_iprim)

        sc_gyro_ratio = np.array(smaker.sc_trans_invariant(self.gyro_ratio))
        sc_ham.gyro_ratio = sc_gyro_ratio

        sc_gilbert_damping = np.array(
            smaker.sc_trans_invariant(self.gilbert_damping))
        sc_ham.gilbert_damping = sc_gilbert_damping

        if self.has_external_hfield:
            sc_Hext = smaker.sc_trans_invariant(self.H_ext)
            sc_ham.set_external_hfield(sc_Hext)

        if self.has_uniaxial_anistropy:
            sc_k1 = smaker.sc_trans_invariant(self.k1)
            sc_k1dir = smaker.sc_trans_invariant(self.k1dir)
            sc_ham.set_uniaxial_mca(sc_k1, np.array(sc_k1dir))

        if self.has_exchange:
            sc_Jdict = smaker.sc_ijR(
                self.exchange_Jdict, n_basis=len(self.pos))
            sc_ham.set_exchange_ijR(exchange_Jdict=sc_Jdict)

        if self.has_dmi:
            sc_dmi_ddict = smaker.sc_ijR(self.dmi_ddict, n_basis=len(self.pos))
            sc_ham.set_dmi_ijR(sc_dmi_ddict)

        if self.has_bilinear:
            sc_bilinear_dict = smaker.sc_ijR(self.bilinear_dict, n_basis=len(self.pos))
            sc_ham.set_bilinear_ijR(sc_bilinear_dict)


        return sc_ham

    def calc_total_HijR(self):
        self._total_hessian_ijR = defaultdict(lambda: np.zeros((3, 3),
                                                               dtype=float))
        for tname, term in self.hamiltonians.items():
            if term.is_twobody_term():
                for key, val in term.hessian_ijR().items():
                    self._total_hessian_ijR[key] += val
        return self._total_hessian_ijR

    def get_total_hessian_ijR(self):
        if self._total_hessian_ijR is None:
            self.calc_total_HijR()
        return self._total_hessian_ijR

    def plot_magnon_band(self,
                         lattice_type=None,
                         kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                            [0.5, 0.5, 0], [0, 0, 0],
                                            [.5, .5, .5]]),
                         knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                         supercell_matrix=None,
                         npoints=100,
                         color='red',
                         kpath_fname=None,
                         ax=None):
        plot_magnon_band(
            self,
            lattice_type=lattice_type,
            kvectors=kvectors,
            knames=knames,
            supercell_matrix=supercell_matrix,
            npoints=npoints,
            color=color,
            kpath_fname=kpath_fname,
            ax=ax)

    def write_xml(self, fname):
        writer = SpinXmlWriter()
        writer._write(self, fname)


def read_spin_ham_from_file(fname):
    parser = SpinXmlParser(fname)
    ham = SpinHamiltonian(
        cell=parser.cell,
        xcart=parser.spin_positions,
        spinat=parser.spin_spinat,
        zion=parser.spin_zions)
    ham.set(
        gilbert_damping=parser.spin_damping_factors,
        gyro_ratio=parser.spin_gyro_ratios)
    if parser.has_exchange:
        ham.set_exchange_ijR(parser.exchange(isotropic=True))
    if parser.has_dmi:
        ham.set_dmi_ijR(parser.dmi)
    if parser.has_bilinear:
        ham.set_bilinear_ijR(parser.bilinear)
    return ham
