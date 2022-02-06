import os
import numpy as np
import copy
from supercellmap import SupercellMaker
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.exchange_pert import ExchangePert
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read
from TB2J.FDTB import dHdx
from TB2J.epw import epw_to_dHdx, generate_TB_with_distortion


def gen_exchange_distorted_epw(path,
                               colinear=True,
                               posfile='POSCAR',
                               prefix_up='wannier90.up',
                               prefix_dn='wannier90.dn',
                               prefix_SOC='wannier90',
                               epw_path='./',
                               epw_prefix='SrMnO3',
                               imode=0,
                               Rp=(0, 0, 0),
                               supercell_matrix=np.eye(3),
                               amplitude=0.0,
                               min_hopping_norm=1e-6,
                               max_distance=None,
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
        scmaker = SupercellMaker(supercell_matrix, center=True)

        dH = epw_to_dHdx(epw_path, epw_prefix, imode, Rp, scmaker=scmaker)
        dH = dH.duplicate_spin()

        tbmodel = tbmodel.make_supercell(scmaker)
        atoms = scmaker.sc_atoms(atoms)
        basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        #dH = dH.make_supercell(scmaker)

        dtb = generate_TB_with_distortion(tbmodel, amplitude, dH)

        exchange = ExchangeNCL(tbmodels=tbmodel,
                               atoms=atoms,
                               basis=basis,
                               efermi=efermi,
                               magnetic_elements=magnetic_elements,
                               kmesh=kmesh,
                               emin=emin,
                               emax=emax,
                               nz=nz,
                               np=np,
                               exclude_orbs=exclude_orbs,
                               list_iatom=list_iatom,
                               description=description)
        exchange.run(os.path.join(output_path, 'original'))

        exchange2 = ExchangeNCL(tbmodels=dtb,
                                atoms=atoms,
                                basis=basis,
                                efermi=efermi,
                                magnetic_elements=magnetic_elements,
                                kmesh=kmesh,
                                emin=emin,
                                emax=emax,
                                nz=nz,
                                np=np,
                                exclude_orbs=exclude_orbs,
                                list_iatom=list_iatom,
                                description=description)
        exchange2.run(os.path.join(
            output_path, f"imode{imode}_Rp{Rp[0]}_{Rp[1]}_{Rp[2]}"))


if __name__ == '__main__':
    for imode in range(15):
        gen_exchange_distorted_epw(
            path="/home/hexu/projects/TB2J_examples/Wannier/SrMnO3_QE_Wannier90/W90",
            colinear=True,
            posfile='SrMnO3.scf.pwi',
            prefix_up="SrMnO3_up",
            prefix_dn="SrMnO3_down",
            prefix_SOC='wannier90',
            epw_path='/home/hexu/projects/SrMnO3/epw',
            epw_prefix='SrMnO3',
            imode=imode,
            Rp=(0, 0, 0),
            supercell_matrix=np.eye(3) * 3,
            amplitude=0.01,
            max_distance=None,
            efermi=10.67,
            magnetic_elements=['Mn'],
            kmesh=[3, 3, 3],
            emin=-7.3363330034071295,
            emax=0.0,
            nz=70,
            np=1,
            exclude_orbs=[],
            description='',
            list_iatom=[13*5+1],
            output_path=f"FD333_imode{imode}")
