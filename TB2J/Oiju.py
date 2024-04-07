import os
import copy
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.exchange_pert import ExchangePert
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read


class PolyTB:
    def __init__(self, ref_model, m0, m1, m2):
        self.ref_model = ref_model
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2

    def gen_model(self, amp):
        m = copy.deepcopy(self.ref_model)
        for R in m.data:
            m.data[R] = (
                self.m0.data[R]
                + self.m1.data[R] * amp
                + self.m2.data[R] * (amp * amp * 0.5)
            )
        return m


def gen_exchange_Oiju(
    path,
    amp,
    colinear=True,
    poscar="POSCAR",
    prefix_up="wannier90.up",
    prefix_dn="wannier90.dn",
    prefix_SOC="wannier90",
    d0_up_ncfile="m0_up.nc",
    d0_dn_ncfile="m0_dn.nc",
    dHdx_up_ncfile="m1_up.nc",
    dHdx_dn_ncfile="m1_dn.nc",
    d2_up_ncfile="m2_up.nc",
    d2_dn_ncfile="m2_dn.nc",
    min_hopping_norm=1e-6,
    max_distance=None,
    efermi=3.0,
    magnetic_elements=[],
    kmesh=[5, 5, 5],
    emin=-12.0,
    emax=0.0,
    exclude_orbs=[],
    Rmesh=[1, 1, 1],
    description="",
):
    atoms = read(os.path.join(path, poscar))
    basis_fname = os.path.join(path, "basisb.txt")
    if colinear:
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, poscar=poscar, nls=False
        )
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, poscar=poscar, nls=False
        )
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        dHdx_up = MyTB.load_MyTB(dHdx_up_ncfile)
        dHdx_dn = MyTB.load_MyTB(dHdx_dn_ncfile)
        dHdx = merge_tbmodels_spin(dHdx_up, dHdx_dn)

        d0_up = MyTB.load_MyTB(d0_up_ncfile)
        d0_dn = MyTB.load_MyTB(d0_dn_ncfile)
        d0 = merge_tbmodels_spin(d0_up, d0_dn)

        d2_up = MyTB.load_MyTB(d2_up_ncfile)
        d2_dn = MyTB.load_MyTB(d2_dn_ncfile)
        d2 = merge_tbmodels_spin(d2_up, d2_dn)

        tb = PolyTB(ref_model=tbmodel, m0=d0, m1=dHdx, m2=d2)

        tbmodel = tb.gen_model(amp)

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
            exclude_orbs=exclude_orbs,
            Rmesh=Rmesh,
            description=description,
        )
        exchange.set_dHdx(dHdx)
        exchange.run("DJ_amp%.3f" % amp)


if __name__ == "__main__":
    gen_exchange_Oiju(
        path="/media/hexu/Backup/materials/SOC/SrMnO3_FM/U3_SrMnO3_slater0.02",
        amp=0.02,
        colinear=True,
        poscar="POSCAR",
        prefix_up="wannier90.up",
        prefix_dn="wannier90.dn",
        prefix_SOC="wannier90",
        dHdx_up_ncfile="m1_up.nc",
        dHdx_dn_ncfile="m1_dn.nc",
        min_hopping_norm=1e-4,
        max_distance=None,
        efermi=3.95,
        magnetic_elements=["Mn"],
        kmesh=[4, 4, 4],
        emin=-12.0,
        emax=0.0,
        exclude_orbs=[],
        Rmesh=[1, 1, 1],
        description="",
    )
