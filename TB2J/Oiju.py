import os
import copy
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.exchange_pert import ExchangePert
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read
from TB2J.FDTB import dHdx


class PolyTB():
    def __init__(self, ref_model, m0, m1, m2):
        self.ref_model = ref_model
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        pass

    def gen_model(self, amp):
        m = copy.deepcopy(self.ref_model)
        for R in m.data:
            m.data[R] = self.m0.data[R] + self.m1.data[R] * amp + self.m2.data[
                R] * (amp * amp * 0.5)
        return m


def gen_exchange_Oiju(path,
                      colinear=True,
                      posfile='POSCAR',
                      prefix_up='wannier90.up',
                      prefix_dn='wannier90.dn',
                      prefix_SOC='wannier90',
                      dHdx=None,
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
                      ):
    atoms = read(os.path.join(path, posfile))
    basis_fname = os.path.join(path, 'basisb.txt')
    if colinear:
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, atoms=atoms, nls=False)
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, atoms=atoms, nls=False)
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        if os.path.exists(basis_fname):
            basis = read_basis(basis_fname)
        else:
            basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        exchange = ExchangePert(
            tbmodels=tbmodel,
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
        exchange.set_dHdx(dHdx)
        exchange.run('Oiju')


if __name__ == '__main__':
    dHdx = dHdx.load_from_pickle("dHdx_shiftz.pickle")
    gen_exchange_Oiju(
        path="./U3_SrMnO3_111_slater0.00",
        colinear=True,
        posfile='POSCAR',
        prefix_up='wannier90.up',
        prefix_dn='wannier90.dn',
        prefix_SOC='wannier90',
        dHdx=dHdx,
        max_distance=None,
        efermi=4.5,
        magnetic_elements=['Mn'],
        kmesh=[5, 5, 5],
        emin=-7.3363330034071295,
        emax=0.0,
        nz=100,
        np=3,
        exclude_orbs=[],
        description='',
        list_iatom=[2],
    )
