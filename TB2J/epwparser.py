"""
Parser for EPW (Electron-Phonon Wannier) files.

This module provides functionality to parse and handle various EPW-related files
generated from Quantum Espresso's EPW code. It includes parsers for:
- WSVec files containing Wigner-Seitz grid information
- Crystal format files containing structural information
- EPW data format files
- EPW matrix elements

The module provides both low-level parsing functions and high-level classes
for handling the EPW data structure.
"""

import os
import sys
import re
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
from netCDF4 import Dataset
from TB2J.wannier.w90_parser import parse_ham
from ase.units import Ry, Bohr


def line_to_array(line, fmt=float):
    """Convert a string line to a numpy array with given format.

    Parameters
    ----------
    line : str
        The input string containing space-separated values
    fmt : type, optional
        The format to convert elements to, by default float

    Returns
    -------
    numpy.ndarray
        Array containing the converted values
    """
    return np.array([fmt(x) for x in line.split()])


def line2vec(line):
    """Convert a string line to a vector of integers.

    Parameters
    ----------
    line : str
        The input string containing space-separated integers

    Returns
    -------
    list
        List of integers parsed from the input line
    """
    return [int(x) for x in line.strip().split()]


def read_WSVec(fname):
    """Read a Wigner-Seitz vector file.

    Parameters
    ----------
    fname : str
        Path to the WSVec file

    Returns
    -------
    tuple
        Contains:
        - dims : int
            First dimension parameter
        - dims2 : int
            Second dimension parameter
        - nRk : int
            Number of R vectors for k-space
        - nRq : int
            Number of R vectors for q-space
        - nRg : int
            Number of R vectors for g-space
        - Rk : numpy.ndarray
            R vectors in k-space
        - Rq : numpy.ndarray
            R vectors in q-space
        - Rg : numpy.ndarray
            R vectors in g-space
        - ndegen_k : numpy.ndarray
            Degeneracy of k-space vectors
        - ndegen_q : numpy.ndarray
            Degeneracy of q-space vectors
        - ndegen_g : numpy.ndarray
            Degeneracy of g-space vectors
    """
    with open(fname) as myfile:
        lines = myfile.readlines()

    (dims, dims2, nRk, nRq,
     nRg) = tuple(int(line.split("=")[1]) for line in lines[:5])

    Rk = np.zeros((nRk, 3), dtype=int)
    Rq = np.zeros((nRq, 3), dtype=int)
    Rg = np.zeros((nRg, 3), dtype=int)
    ndegen_k = np.zeros(nRk, dtype=int)
    ndegen_q = np.zeros(nRq, dtype=int)
    ndegen_g = np.zeros(nRg, dtype=int)

    start = 6
    end = start + nRk
    for i, line in enumerate(lines[start:end]):
        Rk[i] = line2vec(line)

    start = end + 1
    end = start + nRq
    for i, line in enumerate(lines[start:end]):
        Rq[i] = line2vec(line)

    start = end + 1
    end = start + nRg
    for i, line in enumerate(lines[start:end]):
        Rg[i] = line2vec(line)

    start = end + 1
    end = start + nRk
    for i, line in enumerate(lines[start:end]):
        ndegen_k[i] = int(line.strip())

    start = end + 1
    end = start + nRq
    for i, line in enumerate(lines[start:end]):
        ndegen_q[i] = int(line.strip())

    start = end + 1
    end = start + nRg
    for i, line in enumerate(lines[start:end]):
        ndegen_g[i] = int(line.strip())

    return (dims, dims2, nRk, nRq, nRg, Rk, Rq, Rg, ndegen_k, ndegen_q,
            ndegen_g)


@dataclass
class Crystal:
    """Class storing crystal structure information from EPW calculation.

    Attributes
    ----------
    natom : int
        Number of atoms in the unit cell
    nmode : int
        Number of phonon modes
    nelect : float
        Number of electrons
    at : numpy.ndarray
        Real space lattice vectors
    bg : numpy.ndarray
        Reciprocal space lattice vectors
    omega : float
        Unit cell volume
    alat : float
        Lattice parameter
    tau : numpy.ndarray
        Atomic positions
    amass : numpy.ndarray
        Atomic masses
    ityp : numpy.ndarray
        Atomic types
    noncolin : bool
        Whether the calculation is non-collinear
    w_centers : numpy.ndarray
        Wannier function centers
    """
    natom: int = 0
    nmode: int = 0
    nelect: float = 0.0
    at: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    omega: float = 0.0
    alat: float = 0.0
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))
    amass: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ityp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noncolin: bool = False
    w_centers: np.ndarray = field(default_factory=lambda: np.zeros(0))


def is_text_True(s):
    """Check if a string represents a boolean True value.

    Parameters
    ----------
    s : str
        Input string to check

    Returns
    -------
    bool
        True if string starts with 't' or 'T', False otherwise
    """
    return s.strip().lower().startswith("t")


def read_crystal_fmt(fname="crystal.fmt"):
    """Parse the crystal.fmt file containing crystal structure information.

    Parameters
    ----------
    fname : str, optional
        Path to the crystal.fmt file, by default "crystal.fmt"

    Returns
    -------
    Crystal
        Crystal object containing the parsed structural information
    """
    """
    parser to the crystal.fmt file
    see line 114 (qe version 6.8) epw_write in ephwann_shuffle.f90.
    """
    d = Crystal()
    with open(fname) as myfile:
        d.natom = int(next(myfile))
        d.nmode = int(next(myfile))
        d.nelect = float(next(myfile))
        d.at = line_to_array(next(myfile), float)
        d.bg = line_to_array(next(myfile), float)
        d.omega = float(next(myfile))
        d.alat = float(next(myfile))
        d.tau = line_to_array(next(myfile), float)
        d.amass = line_to_array(next(myfile), float)
        d.ityp = line_to_array(next(myfile), int)
        d.noncolin = is_text_True(next(myfile))
        d.w_centers = line_to_array(next(myfile), float)
    return d


def read_epwdata_fmt(fname="epwdata.fmt"):
    """Read the EPW data format file containing basic dimensions.

    Parameters
    ----------
    fname : str, optional
        Path to the epwdata.fmt file, by default "epwdata.fmt"

    Returns
    -------
    tuple
        Contains:
        - nbndsub : int
            Number of bands
        - nrr_k : int
            Number of R vectors for k-space
        - nmodes : int
            Number of phonon modes
        - nrr_q : int
            Number of R vectors for q-space
        - nrr_g : int
            Number of R vectors for g-space
    """
    with open(fname) as myfile:
        efermi = float(next(myfile))
        nbndsub, nrr_k, nmodes, nrr_q, nrr_g = [
            int(x) for x in next(myfile).split()
        ]
    return nbndsub, nrr_k, nmodes, nrr_q, nrr_g


def read_epmatwp(fname="./sic.epmatwp", path='./'):
    """Read the EPW matrix elements file.

    Parameters
    ----------
    fname : str, optional
        Name of the epmatwp file, by default "./sic.epmatwp"
    path : str, optional
        Path to the directory containing the file, by default './'

    Returns
    -------
    numpy.ndarray
        5D array containing the EPW matrix elements with shape
        (nrr_g, nmodes, nrr_k, nbndsub, nbndsub)
    """
    nbndsub, nrr_k, nmodes, nrr_q, nrr_g = read_epwdata_fmt(
        os.path.join(path, "epwdata.fmt"))
    # Is this H or H.T?
    mat = np.fromfile(os.path.join(path, fname), dtype=complex)
    return np.reshape(mat, (nrr_g, nmodes, nrr_k, nbndsub, nbndsub), order='C')


@dataclass
class EpmatOneMode:
    """Class for handling EPW matrix elements for a single phonon mode.

    This class provides functionality to store and manipulate electron-phonon
    matrix elements for a specific phonon mode.

    Attributes
    ----------
    nwann : int
        Number of Wannier functions
    nRk : int
        Number of R vectors in k-space
    nRq : int
        Number of R vectors in q-space
    nRg : int
        Number of R vectors in g-space
    Rk : numpy.ndarray
        R vectors in k-space
    Rq : numpy.ndarray
        R vectors in q-space
    Rg : numpy.ndarray
        R vectors in g-space
    ndegen_k : numpy.ndarray
        Degeneracy of k-space vectors
    ndegen_q : numpy.ndarray
        Degeneracy of q-space vectors
    ndegen_g : numpy.ndarray
        Degeneracy of g-space vectors
    """
    nwann: int = 0
    nRk: int = None
    nRq: int = None
    nRg: int = None
    Rk: np.ndarray = field(default=None)
    Rq: np.ndarray = field(default=None)
    Rg: np.ndarray = field(default=None)
    ndegen_k: np.ndarray = field(default=None)
    ndegen_q: np.ndarray = field(default=None)
    ndegen_g: np.ndarray = field(default=None)

    def __init__(self, epmat, imode, close_nc=False):
        """Initialize EPW matrix elements for a single mode.

        Parameters
        ----------
        epmat : Epmat
            EPW matrix elements object containing data for all modes
        imode : int
            Index of the phonon mode to extract
        close_nc : bool, optional
            Whether to close the netCDF file after reading, by default False
        """
        self.nwann = epmat.nwann
        self.nRk = epmat.nRk
        self.nRq = epmat.nRq
        self.nRg = epmat.nRg
        self.Rk = epmat.Rk
        self.Rq = epmat.Rq
        self.Rg = epmat.Rg

        self.Rkdict = epmat.Rkdict
        self.Rqdict = epmat.Rqdict
        self.Rgdict = epmat.Rgdict

        self.ndegen_k = epmat.ndegen_k
        self.ndegen_q = epmat.ndegen_q
        self.ndegen_g = epmat.ndegen_g

        self.data = np.zeros(
            (self.nRg, self.nRk, self.nwann, self.nwann), dtype=complex)
        for Rg, iRg in self.Rgdict.items():
            dreal = epmat.epmatfile.variables['epmat_real'][iRg,
                                                            imode, :, :, :] * (Ry/Bohr)
            dimag = epmat.epmatfile.variables['epmat_imag'][iRg,
                                                            imode, :, :, :] * (Ry/Bohr)
            self.data[iRg] = (dreal+1.0j * dimag)
            # self.data = np.swapaxes(self.data, 2, 3)

        if close_nc:
            epmat.epmatfile.close()

    def get_epmat_RgRk(self, Rg, Rk, avg=False):
        """Get EPW matrix elements for given R vectors in k and g space.

        Parameters
        ----------
        Rg : tuple
            R vector in g-space
        Rk : tuple
            R vector in k-space
        avg : bool, optional
            Whether to average with the time-reversed counterpart, by default False

        Returns
        -------
        numpy.ndarray
            Complex array of EPW matrix elements
        """
        iRg = self.Rgdict[tuple(Rg)]
        iRk = self.Rkdict[tuple(Rk)]
        ret = np.copy(self.data[iRg, iRk, :, :])

        if avg:
            Rg2 = tuple(np.array(Rg)-np.array(Rk))
            Rk2 = tuple(-np.array(Rk))
            if Rg2 in self.Rgdict and Rk in self.Rkdict:
                iRg2 = self.Rgdict[tuple(Rg2)]
                iRk2 = self.Rkdict[tuple(Rk2)]
                ret += self.data[iRg2, iRk2, :, :].T
                ret /= 2.0
        return ret

    def get_epmat_RgRk_two_spin(self, Rg, Rk, avg=False):
        """Get EPW matrix elements for two-spin case.

        Parameters
        ----------
        Rg : tuple
            R vector in g-space
        Rk : tuple
            R vector in k-space
        avg : bool, optional
            Whether to average with the time-reversed counterpart, by default False

        Returns
        -------
        numpy.ndarray
            Complex array of EPW matrix elements for two-spin case
        """
        H = self.get_epmat_RgRk(Rg, Rk, avg=avg)
        ret = np.zeros((self.nwann*2, self.nwann*2), dtype=complex)
        ret[::2, ::2] = H
        ret[1::2, 1::2] = H
        return ret


@dataclass
class Epmat:
    """Class for handling the complete set of EPW matrix elements.

    This class provides functionality to read, store, and manipulate
    the full set of electron-phonon matrix elements for all modes.

    Attributes
    ----------
    crystal : Crystal
        Crystal structure information
    nwann : int
        Number of Wannier functions
    nmodes : int
        Number of phonon modes
    nRk : int
        Number of R vectors in k-space
    nRq : int
        Number of R vectors in q-space
    nRg : int
        Number of R vectors in g-space
    Rk : numpy.ndarray
        R vectors in k-space
    Rq : numpy.ndarray
        R vectors in q-space
    Rg : numpy.ndarray
        R vectors in g-space
    ndegen_k : numpy.ndarray
        Degeneracy of k-space vectors
    ndegen_q : numpy.ndarray
        Degeneracy of q-space vectors
    ndegen_g : numpy.ndarray
        Degeneracy of g-space vectors
    Hwann : numpy.ndarray
        Wannier Hamiltonian
    Rlist : numpy.ndarray
        List of R vectors
    epmat_wann : numpy.ndarray
        EPW matrix elements in Wannier gauge
    epmat_ncfile : Dataset
        NetCDF file containing EPW matrix elements
    """
    crystal: Crystal = field(default_factory=Crystal)
    nwann: int = 0
    nmodes: int = 0
    nRk: int = None
    nRq: int = None
    nRg: int = None
    Rk: np.ndarray = field(default=None)
    Rq: np.ndarray = field(default=None)
    Rg: np.ndarray = field(default=None)
    ndegen_k: np.ndarray = field(default=None)
    ndegen_q: np.ndarray = field(default=None)
    ndegen_g: np.ndarray = field(default=None)
    Hwann: np.ndarray = field(default=None)
    Rlist: np.ndarray = field(default=None)
    epmat_wann: np.ndarray = field(default=None)
    epmat_ncfile: Dataset = field(default=None)

    Rk_dict: dict = field(default_factory=dict)
    Rq_dict: dict = field(default_factory=dict)
    Rg_dict: dict = field(default_factory=dict)

    def read_Rvectors(self, path, fname="WSVecDeg.dat"):
        """Read R vectors from a WS vector file.

        Parameters
        ----------
        path : str
            Directory containing the file
        fname : str, optional
            Name of the WS vector file, by default "WSVecDeg.dat"
        """
        fullfname = os.path.join(path, fname)
        (dims, dims2, self.nRk, self.nRq, self.nRg, self.Rk, self.Rq, self.Rg,
         self.ndegen_k, self.ndegen_q, self.ndegen_g) = read_WSVec(fullfname)
        self.Rkdict = {tuple(self.Rk[i]): i for i in range(self.nRk)}
        self.Rqdict = {tuple(self.Rq[i]): i for i in range(self.nRq)}
        self.Rgdict = {tuple(self.Rg[i]): i for i in range(self.nRg)}

    def read_Wannier_Hamiltonian(self, path, fname):
        """Read Wannier Hamiltonian from file.

        Parameters
        ----------
        path : str
            Directory containing the file
        fname : str
            Name of the Hamiltonian file
        """
        nwann, HR = parse_ham(fname=os.path.join(path, fname))
        nR = len(HR)
        self.Rlist = np.array(list(HR.keys()), dtype=int)
        self.Hwann = np.zeros((nR, nwann, nwann), dtype=complex)
        for i, H in enumerate(HR.values()):
            self.Hwann[i] = H

    def read_epmat(self, path, prefix):
        """Read EPW matrix elements from binary file.

        Parameters
        ----------
        path : str
            Directory containing the file
        prefix : str
            Prefix for the EPW files
        """
        mat = np.fromfile(os.path.join(path, f"{prefix}.epmatwp"),
                          dtype=complex)
        # mat = mat.reshape((self.nRg, self.nmodes, self.nRk,
        #                   self.nwann, self.nwann), order='F')
        mat.shape = (self.nRg, self.nmodes, self.nRk,
                     self.nwann, self.nwann)
        self.epmat_wann = mat

    def read(self, path, prefix, epmat_ncfile=None):
        """Read all necessary EPW data.

        Parameters
        ----------
        path : str
            Directory containing the files
        prefix : str
            Prefix for the EPW files
        epmat_ncfile : str, optional
            Name of netCDF file containing EPW matrix elements, by default None
        """
        # self.crystal = read_crystal_fmt(
        #    fname=os.path.join(path, 'crystal.fmt'))
        #self.read_Wannier_Hamiltonian(path,  f"{prefix}_hr.dat")
        (self.nwann, self.nRk, self.nmodes, self.nRq,
         self.nRg) = read_epwdata_fmt(os.path.join(path, 'epwdata.fmt'))
        self.read_Rvectors(path, "WSVecDeg.dat")
        if epmat_ncfile is not None:
            self.epmatfile = Dataset(os.path.join(path, epmat_ncfile), 'r')
        else:
            self.read_epmat(path, prefix)

    def get_epmat_Rv_from_index(self, imode, iRg, iRk):
        """Get EPW matrix elements for given mode and R vector indices.

        Parameters
        ----------
        imode : int
            Mode index
        iRg : int
            Index of R vector in g-space
        iRk : int
            Index of R vector in k-space

        Returns
        -------
        numpy.ndarray
            Complex array of EPW matrix elements
        """
        dreal = self.epmatfile.variables['epmat_real'][iRg, imode, iRk, :, :]
        dimag = self.epmatfile.variables['epmat_imag'][iRg, imode, iRk, :, :]
        return (dreal + 1.0j * dimag)*(Ry/Bohr)

    def get_epmat_Rv_from_RgRk(self, imode, Rg, Rk):
        """Get EPW matrix elements for given mode and R vectors.

        Parameters
        ----------
        imode : int
            Mode index
        Rg : tuple
            R vector in g-space
        Rk : tuple
            R vector in k-space

        Returns
        -------
        numpy.ndarray
            Complex array of EPW matrix elements
        """
        iRg = self.Rgdict[tuple(Rg)]
        iRk = self.Rkdict[tuple(Rk)]
        dreal = self.epmatfile.variables['epmat_real'][iRg, imode, iRk, :, :]
        dimag = self.epmatfile.variables['epmat_imag'][iRg, imode, iRk, :, :]
        return (dreal + 1.0j * dimag)*(Ry/Bohr)

    def get_epmat_Rv_from_R(self, imode, Rg):
        """Get EPW matrix elements for given mode and g-space R vector.

        Parameters
        ----------
        imode : int
            Mode index
        Rg : tuple
            R vector in g-space

        Returns
        -------
        dict
            Dictionary mapping k-space R vectors to EPW matrix elements
        """
        iRg = self.Rgdict[tuple(Rg)]
        g = defaultdict(lambda: np.zeros((self.nwann, self.nwann)))
        for Rk in self.Rkdict:
            iRk = self.Rkdict[tuple(Rk)]
            g[tuple(Rk)] = self.get_epmat_Rv_from_index(imode, iRg, iRk)
        return g

    def print_info(self):
        """Print basic information about the EPW matrix elements."""
        print(
            f"nwann: {self.nwann}\n nRk:{self.nRk}, nmodes:{self.nmodes}, nRg:{self.nRg}"
        )

    def save_to_netcdf(self, path, fname):
        """Save EPW matrix elements to a netCDF file.

        Parameters
        ----------
        path : str
            Directory to save the file in
        fname : str
            Name of the output netCDF file
        """
        root = Dataset(os.path.join(path, fname), "w")
        root.createDimension("nwann", self.nwann)
        root.createDimension("nRk", self.nRk)
        root.createDimension("nRq", self.nRq)
        root.createDimension("nRg", self.nRg)
        root.createDimension("nmodes", self.nmodes)
        root.createDimension("three", 3)
        root.createVariable("epmat_real",
                            "double",
                            dimensions=("nRg", "nmodes", "nRk", "nwann",
                                        "nwann"), zlib=False)
        root.createVariable("epmat_imag",
                            "double",
                            dimensions=("nRg", "nmodes", "nRk", "nwann",
                                        "nwann"), zlib=False)

        root.variables["epmat_real"][:] = self.epmat_wann.real
        root.variables["epmat_imag"][:] = self.epmat_wann.imag
        print("ncfile written")
        root.close()


def save_epmat_to_nc(path, prefix, ncfile='epmat.nc'):
    """Save EPW matrix elements to a NetCDF file.

    Parameters
    ----------
    path : str
        Path to the directory containing EPW files
    prefix : str
        Prefix for the EPW files
    ncfile : str, optional
        Name of the output NetCDF file, by default 'epmat.nc'
    """
    ep = Epmat()
    ep.read(path=path, prefix=prefix)
    ep.save_to_netcdf(path=path, fname=ncfile)


def test():
    """Test function for saving EPW matrix elements to netCDF format.
    
    Uses a sample SiC dataset to demonstrate the file conversion process.
    """
    path = "/home/hexu/projects/epw_test/sic_small_kq/NM/epw"
    save_epmat_to_nc(path=path, prefix='sic', ncfile='epmat.nc')


def test_read_data():
    """Test function for reading and manipulating EPW data.
    
    Demonstrates loading EPW data from a netCDF file and performing
    various operations on the matrix elements, including time-reversal
    symmetry checks.
    """
    path = "/home/hexu/projects/SrMnO3/spinphon_data/epw555_10"
    prefix = 'SrMnO3'
    #path = "/home/hexu/projects/epw_test/sic_small_kq/NM/epw"
    #prefix = 'sic'
    ep = Epmat()
    ep.read(path=path, prefix=prefix, epmat_ncfile='epmat.nc')
    # ep.get_epmat_Rv_from_index(0, 0)
    #d = ep.get_epmat_Rv_from_R(0, (0, 0, 0))
    # print(d[(0, 0, 2)].imag)

    ep1mode = EpmatOneMode(ep, imode=3)
    Rg = (0, 0, 0)
    Rk = (0, 1, 0)
    Rg2 = tuple(np.array(Rg)-np.array(Rk))
    Rk2 = tuple(-np.array(Rk))
    dv1 = ep1mode.get_epmat_RgRk(Rg=Rg, Rk=Rk, avg=False).real
    dv2 = ep1mode.get_epmat_RgRk(Rg=Rg2, Rk=Rk2, avg=False).real.T
    print(dv1)
    print("-"*10)
    print(dv2)
    print("-"*10)
    print(dv1-dv2)


if __name__ == "__main__":
    # test()
    test_read_data()
