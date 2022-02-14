"""
Parser for EPW files.
"""

import os
import sys
import re
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from netCDF4 import Dataset
from TB2J.wannier.w90_parser import parse_ham
from ase.units import Ry, Bohr


def line_to_array(line, fmt=float):
    return np.array([fmt(x) for x in line.split()])


def line2vec(line):
    return [int(x) for x in line.strip().split()]


def read_WSVec(fname):
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


@dataclass()
class Crystal():
    natom: int = 0
    nmode: int = 0
    nelect: float = 0.0
    at: np.ndarray = np.zeros(0)
    bg: np.ndarray = np.zeros(0)
    omega: float = 0.0
    alat: float = 0.0
    tau: np.ndarray = np.zeros(0)
    amass: np.ndarray = np.zeros(0)
    ityp: np.ndarray = np.zeros(0)
    noncolin: bool = False
    w_centers: np.ndarray = np.zeros(0)


def is_text_True(s):
    return s.strip().lower().startswith("t")


def read_crystal_fmt(fname="crystal.fmt"):
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
    with open(fname) as myfile:
        efermi = float(next(myfile))
        nbndsub, nrr_k, nmodes, nrr_q, nrr_g = [
            int(x) for x in next(myfile).split()
        ]
    return nbndsub, nrr_k, nmodes, nrr_q, nrr_g


def read_epmatwp(fname="./sic.epmatwp", path='./'):
    nbndsub, nrr_k, nmodes, nrr_q, nrr_g = read_epwdata_fmt(
        os.path.join(path, "epwdata.fmt"))
    # Is this H or H.T?
    mat = np.fromfile(os.path.join(path, fname), dtype=complex)
    return np.reshape(mat, (nrr_g, nmodes, nrr_k, nbndsub, nbndsub), order='C')


@dataclass()
class EpmatOneMode():
    nwann: int = 0
    nRk: int = None
    nRq: int = None
    nRg: int = None
    Rk: np.ndarray = None
    Rq: np.ndarray = None
    Rg: np.ndarray = None
    ndegen_k: np.ndarray = None
    ndegen_q: np.ndarray = None
    ndegen_g: np.ndarray = None

    def __init__(self, epmat, imode, close_nc=False):
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
        H = self.get_epmat_RgRk(Rg, Rk, avg=avg)
        ret = np.zeros((self.nwann*2, self.nwann*2), dtype=complex)
        ret[::2, ::2] = H
        ret[1::2, 1::2] = H
        return ret


@ dataclass()
class Epmat():
    crystal: Crystal = Crystal()
    nwann: int = 0
    nmodes: int = 0
    nRk: int = None
    nRq: int = None
    nRg: int = None
    Rk: np.ndarray = None
    Rq: np.ndarray = None
    Rg: np.ndarray = None
    ndegen_k: np.ndarray = None
    ndegen_q: np.ndarray = None
    ndegen_g: np.ndarray = None
    Hwann: np.ndarray = None
    Rlist: np.ndarray = None
    epmat_wann: np.ndarray = None
    epmat_ncfile: Dataset = None

    Rk_dict = {}
    Rq_dict = {}
    Rg_dict = {}

    def read_Rvectors(self, path, fname="WSVecDeg.dat"):
        fullfname = os.path.join(path, fname)
        (dims, dims2, self.nRk, self.nRq, self.nRg, self.Rk, self.Rq, self.Rg,
         self.ndegen_k, self.ndegen_q, self.ndegen_g) = read_WSVec(fullfname)
        self.Rkdict = {tuple(self.Rk[i]): i for i in range(self.nRk)}
        self.Rqdict = {tuple(self.Rq[i]): i for i in range(self.nRq)}
        self.Rgdict = {tuple(self.Rg[i]): i for i in range(self.nRg)}

    def read_Wannier_Hamiltonian(self, path, fname):
        nwann, HR = parse_ham(fname=os.path.join(path, fname))
        nR = len(HR)
        self.Rlist = np.array(list(HR.keys()), dtype=int)
        self.Hwann = np.zeros((nR, nwann, nwann), dtype=complex)
        for i, H in enumerate(HR.values()):
            self.Hwann[i] = H

    def read_epmat(self, path, prefix):
        mat = np.fromfile(os.path.join(path, f"{prefix}.epmatwp"),
                          dtype=complex)
        # mat = mat.reshape((self.nRg, self.nmodes, self.nRk,
        #                   self.nwann, self.nwann), order='F')
        mat.shape = (self.nRg, self.nmodes, self.nRk,
                     self.nwann, self.nwann)
        self.epmat_wann = mat

    def read(self, path, prefix, epmat_ncfile=None):
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
        dreal = self.epmatfile.variables['epmat_real'][iRg, imode, iRk, :, :]
        dimag = self.epmatfile.variables['epmat_imag'][iRg, imode, iRk, :, :]
        return (dreal + 1.0j * dimag)*(Ry/Bohr)

    def get_epmat_Rv_from_RgRk(self, imode, Rg, Rk):
        iRg = self.Rgdict[tuple(Rg)]
        iRk = self.Rkdict[tuple(Rk)]
        dreal = self.epmatfile.variables['epmat_real'][iRg, imode, iRk, :, :]
        dimag = self.epmatfile.variables['epmat_imag'][iRg, imode, iRk, :, :]
        return (dreal + 1.0j * dimag)*(Ry/Bohr)

    def get_epmat_Rv_from_R(self, imode, Rg):
        iRg = self.Rgdict[tuple(Rg)]
        g = defaultdict(lambda: np.zeros((self.nwann, self.nwann)))
        for Rk in self.Rkdict:
            iRk = self.Rkdict[tuple(Rk)]
            g[tuple(Rk)] = self.get_epmat_Rv_from_index(imode, iRg, iRk)
        return g

    def print_info(self):
        print(
            f"nwann: {self.nwann}\n nRk:{self.nRk}, nmodes:{self.nmodes}, nRg:{self.nRg}"
        )

    def save_to_netcdf(self, path, fname):
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
    ep = Epmat()
    ep.read(path=path, prefix=prefix)
    ep.save_to_netcdf(path=path, fname=ncfile)


def test():
    path = "/home/hexu/projects/epw_test/sic_small_kq/NM/epw"
    save_epmat_to_nc(path=path, prefix='sic', ncfile='epmat.nc')


def test_read_data():
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
