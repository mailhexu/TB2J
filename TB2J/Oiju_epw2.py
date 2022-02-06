import os
import numpy as np
import copy
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.exchange_pert2 import ExchangePert2
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read
from TB2J.FDTB import dHdx
from TB2J.epw import epw_to_dHdx, generate_TB_with_distortion
from TB2J.epwparser import Epmat, EpmatOneMode
from ase.io import write


def gen_exchange_Oiju_epw(path,
                          colinear=True,
                          posfile='POSCAR',
                          prefix_up='wannier90.up',
                          prefix_dn='wannier90.dn',
                          prefix_SOC='wannier90',
                          epw_path='./',
                          epw_prefix='SrMnO3',
                          imode=0,
                          Ru=(0, 0, 0),
                          min_hopping_norm=1e-6,
                          Rcut=None,
                          efermi=3.,
                          magnetic_elements=[],
                          kmesh=[5, 5, 5],
                          emin=-12.0,
                          emax=0.0,
                          nz=50,
                          np=1,
                          exclude_orbs=[],
                          description='',
                          list_iatom=None,
                          output_path='TB2J_results'):
    atoms = read(os.path.join(path, posfile))
    if colinear:
        tbmodel_up = MyTB.read_from_wannier_dir(path=path,
                                                prefix=prefix_up,
                                                atoms=atoms,
                                                nls=False)
        tbmodel_dn = MyTB.read_from_wannier_dir(path=path,
                                                prefix=prefix_dn,
                                                atoms=atoms,
                                                nls=False)
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        epw = Epmat()
        epw.read(path=epw_path, prefix=epw_prefix, epmat_ncfile='epmat.nc')

        basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)

        exchange = ExchangePert2(tbmodels=tbmodel,
                                 atoms=atoms,
                                 basis=basis,
                                 efermi=efermi,
                                 magnetic_elements=magnetic_elements,
                                 kmesh=kmesh,
                                 emin=emin,
                                 emax=emax,
                                 nz=nz,
                                 np=np,
                                 Rcut=Rcut,
                                 exclude_orbs=exclude_orbs,
                                 list_iatom=list_iatom,
                                 description=description)
        exchange.set_epw(Ru, imode=imode, epmat=epw)
        exchange.run(output_path)


if __name__ == '__main__':
    # for imode in range(15):
    # for imode in range(3, 15):
    for imode in [3, 6, 7]:
        gen_exchange_Oiju_epw(
            # path="./U3_SrMnO3_111_slater0.00",
            # path="/home/hexu/projects/SrMnO3/origin_wannier/SrMnO3_FM/center_SrMnO3_slater0.00",
            path="/home/hexu/projects/TB2J_examples/Wannier/SrMnO3_QE_Wannier90/W90",
            colinear=True,
            posfile='SrMnO3.scf.pwi',
            # prefix_up='wannier90.up',
            # prefix_dn='wannier90.dn',
            prefix_up="SrMnO3_up",
            prefix_dn="SrMnO3_down",
            prefix_SOC='wannier90',
            epw_path='/home/hexu/projects/SrMnO3/epw',
            epw_prefix='SrMnO3',
            imode=imode,
            Ru=(0, 0, 0),
            Rcut=8,
            efermi=10.67,
            magnetic_elements=['Mn'],
            kmesh=[5, 5, 5],
            emin=-7.3363330034071295,
            emax=0.0,
            nz=70,
            np=1,
            exclude_orbs=[],
            description='',
            # list_iatom=[1],
            output_path=f"VT_{imode}_withcutoff0.1")
