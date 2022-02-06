"""
This module connect the epw data to generate tightbinding+ep models
"""

import copy
from TB2J.myTB import MyTB
from TB2J.FDTB import dHdx
from supercellmap.supercell import SupercellMaker
from TB2J.epwparser import Epmat
from TB2J.exchange_pert import ExchangePert
import numpy as np
from collections import defaultdict


def generate_TB_with_distortion(
    tb: MyTB,
    amplitude: float,
    dH: dHdx,
):
    dtb = copy.deepcopy(tb)
    for R in tb.Rlist:
        if R in dH.Rdict:
            dtb.data[R] += dH.get_dHR(R) * amplitude / dH.get_wsdeg(R)
    return dtb


def epw_to_dHdx_old(epw_path, epw_prefix, imode, Rq):
    ep = Epmat()
    ep.read(path=epw_path, prefix=epw_prefix, epmat_ncfile='epmat.nc')
    dHR_dict = ep.get_epmat_Rv_from_R(imode, Rq)
    dH = dHdx(ep.Rk,
              nbasis=ep.nwann,
              dHR=dHR_dict,
              dHR2=None,
              wsdeg=ep.ndegen_k)
    return dH


def epw_to_dHdx(epw_path, epw_prefix, imode, Rpprime=(0, 0, 0), scmaker=SupercellMaker(np.eye(3))):
    """
    For a given perturbation at Rp',
    <Rm|Rp'=Rp+Rm|Rk+Rm>
    =H(Rp,Rk)=<0|Rp|Rk> is a matrix of nbasis*nbasis
    First: Rm = Rp'-Rp, Rk+Rm = Rp'-Rp+Rm
    Input: Rplist, Rklist, H
    H: [iRg, iRk, ibasis, ibasis]
    """
    ep = Epmat()
    ep.read(path=epw_path, prefix=epw_prefix, epmat_ncfile='epmat.nc')
    n_basis = ep.nwann
    sc_nbasis = n_basis*scmaker.ncell
    sc_RHdict = defaultdict(lambda: np.zeros(
        (sc_nbasis, sc_nbasis), dtype=complex))
    for iRp, Rp in enumerate(ep.Rq):
        Rm = np.array(Rpprime)-np.array(Rp)
        sc_part_i, pair_ind_i = scmaker._sc_R_to_pair_ind(
            tuple(np.array(Rm)))
        ii = pair_ind_i*n_basis
        for iRk, Rk in enumerate(ep.Rk):
            Rn = np.array(Rk) + np.array(Rm)
            sc_part_j, pair_ind_j = scmaker._sc_R_to_pair_ind(
                tuple(Rn))
            if tuple(sc_part_i) == (0, 0, 0):
                jj = pair_ind_j * n_basis
                sc_RHdict[tuple(sc_part_j)][ii:ii + n_basis,
                                            jj:jj + n_basis] += ep.get_epmat_Rv_from_RgRk(imode, Rm, Rk)
            # NOTE: for sc_part_i /=0. the data are thrown away.
            # elif tuple(sc_part_j) == (0, 0, 0):
            #    sc_RHdict[tuple(-sc_part_j)][jj:jj + n_basis,
            #                                 ii:ii + n_basis] += H[iRp, iRk, :, :].T.conj()

    Rlist = np.array(list(sc_RHdict.keys()))
    dH = dHdx(Rlist,
              nbasis=sc_nbasis,
              dHR=sc_RHdict,
              dHR2=None,
              wsdeg=None)

    return dH


def test_epw_to_dHdx():
    epw_path = "/home/hexu/projects/SrMnO3/epw"
    prefix = "SrMnO3"
    scmaker = SupercellMaker([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    dH = epw_to_dHdx(epw_path, prefix, imode=0,
                     Rpprime=(0, 0, 0), scmaker=scmaker)
    dH = dH.duplicate_spin()


# test_epw_to_dHdx()


class ExchangeEPWSupercell(ExchangePert):
    def build_supercell(self, ):
        pass

    def set_phonon_perturbtaion(self, epw_path, epw_prefix, imode, Rp):
        pass

    def read_epw(self, epw_path, epw_prefix, imode, Rp):
        pass

    def dHdx_to_supercell(self):
        pass

    def run_all(self):
        self.build_supercell()
        self.set_phonon_perturbation()


class MyTBEPW(MyTB):
    pass
