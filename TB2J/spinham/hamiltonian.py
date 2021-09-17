#!/usr/bin/env python
import numpy as np
from collections import Iterable, defaultdict
import matplotlib.pyplot as plt
from ase.dft.kpoints import bandpath, monkhorst_pack
from .hamiltonian_terms import (ZeemanTerm, UniaxialMCATerm, ExchangeTerm,
                                DMITerm, BilinearTerm)
from .constants import mu_B, gyromagnetic_ratio
from .supercell import SupercellMaker
from .spin_xml import SpinXmlParser, SpinXmlWriter
from .plot import group_band_path
from ase.cell import Cell
#from minimulti.spin.hamiltonian import SpinHamiltonian
#from minimulti.spin.mover import SpinMover
from .qsolver import QSolver


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
            self.pos = pos
        elif pos is None:
            self.pos = np.dot(xcart, np.linalg.inv(cell))
            self.xcart = xcart
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
        self.has_uniaxial_anisotropy = False

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

    def set(self, gilbert_damping=None, gyro_ratio=None, 
            has_exchange=None, has_dmi=None, has_bilinear=None,
            has_external_hfield=None, has_uniaxial_anisotropy=None):
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
        self.has_exchange=self.has_exchange and bool(has_exchange)
        self.has_dmi=self.has_dmi and bool(has_dmi)
        self.has_bilinear=self.has_bilinear and bool(has_bilinear)
        self.has_external_hfield=self.has_external_hfield and bool(has_external_hfield)
        self.has_uniaxial_anisotropy=self.has_uniaxial_anisotropy and bool(has_uniaxial_anisotropy)

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
        self.has_bilinear = True
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
        self.has_uniaxial_anisotropy = True

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

        sc_ham = SpinHamiltonian(cell=sc_cell,
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

        if self.has_uniaxial_anisotropy:
            sc_k1 = smaker.sc_trans_invariant(self.k1)
            sc_k1dir = smaker.sc_trans_invariant(self.k1dir)
            sc_ham.set_uniaxial_mca(sc_k1, np.array(sc_k1dir))

        if self.has_exchange:
            sc_Jdict = smaker.sc_ijR(self.exchange_Jdict,
                                     n_basis=len(self.pos))
            sc_ham.set_exchange_ijR(exchange_Jdict=sc_Jdict)

        if self.has_dmi:
            sc_dmi_ddict = smaker.sc_ijR(self.dmi_ddict, n_basis=len(self.pos))
            sc_ham.set_dmi_ijR(sc_dmi_ddict)

        if self.has_bilinear:
            sc_bilinear_dict = smaker.sc_ijR(self.bilinear_dict,
                                             n_basis=len(self.pos))
            sc_ham.set_bilinear_ijR(sc_bilinear_dict)

        return sc_ham

    def calc_total_HijR(self):
        self._total_hessian_ijR = defaultdict(lambda: np.zeros(
            (3, 3), dtype=float))
        for tname, term in self.hamiltonians.items():
            if term.is_twobody_term():
                for key, val in term.hessian_ijR().items():
                    self._total_hessian_ijR[key] += val
        return self._total_hessian_ijR

    def get_total_hessian_ijR(self):
        if self._total_hessian_ijR is None:
            self.calc_total_HijR()
        return self._total_hessian_ijR

    def solve_k(self, kpts, Jq=False):
        """
        Get the eigenvalues and eigenvectors for the kpoints
        """
        qsolver = QSolver(hamiltonian=self)
        evals, evecs = qsolver.solve_all(kpts, eigen_vectors=True, Jq=Jq)
        return evals, evecs

    def find_ground_state_from_kmesh(self, kmesh, myfile):
        kpts = monkhorst_pack(kmesh)
        evals, evecs = self.solve_k(kpts, Jq=True)
        #write_magnon_info(self, kpts, evals, evecs, myfile)

    def plot_magnon_band(self,
                         kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                            [0.5, 0.5, 0], [0, 0, 0],
                                            [.5, .5, .5]]),
                         knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                         supercell_matrix=None,
                         npoints=100,
                         color='red',
                         kpath_fname=None,
                         Jq=False,
                         ax=None,
                         ):
        if ax is None:
            fig, ax = plt.subplots()
        if knames is None and kvectors is None:
            # fully automatic k-path
            bp = Cell(self.cell).bandpath(npoints=npoints)
            spk = bp.special_points
            xlist, kptlist, Xs, knames=group_band_path(bp)
        elif knames is not None and kvectors is None:
            # user specified kpath by name
            bp = Cell(self.cell).bandpath(knames, npoints=npoints)
            spk = bp.special_points
            kpts = bp.kpts
            xlist, kptlist, Xs, knames=group_band_path(bp)
        else:
            # user spcified kpath and kvector.
            kpts, x, Xs = bandpath(kvectors, self.cell, npoints)
            spk = dict(zip(knames, kvectors))
            xlist=[x]
            kptlist=[kpts]

        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        print("High symmetry k-points:")
        for name, k in spk.items():
            if name=='G':
                name='Gamma'
            print(f"{name}: {k}")


        for kpts, xs in zip(kptlist, xlist):
            evals, evecs = self.solve_k(kpts,Jq=Jq)
            # Plot band structure
            nbands = evals.shape[1]
            emin = np.min(evals[:, 0])
            for i in range(nbands):
                ax.plot(xs, (evals[:, i]) / 1.6e-22, color=color)

        ax.set_ylabel('Energy (meV)')
        ax.set_xlim(xlist[0][0], xlist[-1][-1])
        ax.set_xticks(Xs)
        knames=[x if x!='G' else '$\Gamma$' for x in knames]
        ax.set_xticklabels(knames)
        for x in Xs:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def write_xml(self, fname):
        writer = SpinXmlWriter()
        writer._write(self, fname)


def read_spin_ham_from_file(fname):
    parser = SpinXmlParser(fname)
    ham = SpinHamiltonian(cell=parser.cell,
                          xcart=parser.spin_positions,
                          spinat=parser.spin_spinat,
                          zion=parser.spin_zions)
    ham.set(gilbert_damping=parser.spin_damping_factors,
            gyro_ratio=parser.spin_gyro_ratios)
    if parser.has_exchange:
        ham.set_exchange_ijR(parser.exchange(isotropic=True))
    if parser.has_dmi:
        ham.set_dmi_ijR(parser.dmi)
    if parser.has_bilinear:
        ham.set_bilinear_ijR(parser.bilinear)
    return ham
