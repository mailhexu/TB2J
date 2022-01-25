"""
This module connect the epw data to generate tightbinding+ep models
"""

import copy
from TB2J.myTB import MyTB
from TB2J.FDTB import dHdx
from supercellmap.supercell import SupercellMaker
from epwparser import Epmat
from TB2J.exchange_pert import ExchangePert


def generate_TB_with_distortion(
    tb: MyTB,
    amplitude: float,
    dH: dHdx,
):
    dtb = copy.deepcopy(tb)
    for R in tb.Rlist:
        if R in dH.Rdict:
            dtb.data[R] += dH.get_dHR(R) * amplitude / dH.get_wsdeg(R)


def epw_to_dHdx(epw_path, epw_prefix, imode, Rp):
    ep = Epmat()
    ep.read(path=epw_path, prefix=epw_prefix, epmat_ncfile='epmat.nc')
    dHR_dict = ep.get_epmat_Rv_from_R(imode, Rp)
    dH = dHdx(ep.Rk,
              nbasis=ep.nwann,
              dHR=dHR_dict,
              dHR2=None,
              wsdeg=ep.ndegen_k)
    return dH


def test_epw_to_dHdx():
    epw_path = "/home/hexu/projects/SrMnO3/epw"
    prefix = "SrMnO3"
    dH = epw_to_dHdx(epw_path, prefix, imode=0, Rp=(0, 0, 0))
    scmaker = SupercellMaker([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    scdH = dH.make_supercell(scmaker=scmaker)
    scdH = scdH.duplicate_spin()
    print(scdH.get_dHR((0, 0, 0)))


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
