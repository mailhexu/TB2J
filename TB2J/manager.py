import os
from TB2J.myTB import MyTB, merge_tbmodels_spin
import numpy as np
from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read
from TB2J.sisl_wrapper import SislWrapper
from TB2J.gpaw_wrapper import GPAWWrapper

def gen_exchange(path,
                 colinear=True,
                 orb_order=1,
                 posfile='POSCAR',
                 prefix_up='wannier90.up',
                 prefix_dn='wannier90.dn',
                 prefix_SOC='wannier90',
                 min_hopping_norm=1e-4,
                 max_distance=None,
                 efermi=0,
                 magnetic_elements=[],
                 kmesh=[4, 4, 4],
                 emin=-12.0,
                 emax=0.0,
                 nz=100,
                 height=0.2,
                 nz1=50,
                 nz2=200,
                 nz3=50,
                 exclude_orbs=[],
                 Rcut=None,
                 ne=None,
                 description=''):
    atoms = read(os.path.join(path, posfile))
    basis_fname = os.path.join(path, 'basis.txt')
    if colinear:
        print("Reading Wannier90 hamiltonian: spin up.")
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, posfile=posfile, nls=False)
        print("Reading Wannier90 hamiltonian: spin down.")
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, posfile=posfile, nls=False)
        if os.path.exists(basis_fname):
            basis = read_basis(basis_fname)
        else:
            basis, _ = auto_assign_basis_name(tbmodel_up.xred, atoms)
        print("Starting to calculate exchange.")
        exchange = ExchangeCL2(
            tbmodels=(tbmodel_up, tbmodel_dn),
            atoms=atoms,
            basis=basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            height=height,
            nz1=nz1,
            nz2=nz2,
            nz3=nz3,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        exchange.run()
        print("All calculation finsihed. The results are in TB2J_results directory.")

    elif colinear and 0:
        print("Reading Wannier90 hamiltonian: spin up.")
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, posfile=posfile, nls=False)
        print("Reading Wannier90 hamiltonian: spin down.")
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, posfile=posfile, nls=False)
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)
        if os.path.exists(basis_fname):
            basis = read_basis(basis_fname)
        else:
            basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        print("Starting to calculate exchange.")
        exchange = ExchangeCL(
            tbmodels=tbmodel,
            atoms=atoms,
            basis=basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            height=height,
            nz1=nz1,
            nz2=nz2,
            nz3=nz3,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        exchange.run()
        print("All calculation finsihed. The results are in TB2J_results directory.")
    else:
        print("Reading Wannier90 hamiltonian: non-colinear spin.")
        tbmodel = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_SOC, posfile=posfile, nls=True)
        if orb_order==1:
            pass
        if orb_order==2:
            tbmodel=tbmodel.reorder()
        if os.path.exists(basis_fname):
            print("The use of basis file is deprecated. It will be ignored.")
            #basis = read_basis(basis_fname)
        else:
            basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=atoms,
            basis=basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            height=height,
            nz1=nz1,
            nz2=nz2,
            nz3=nz3,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        print("\n")
        exchange.run()
        print("All calculation finsihed. The results are in TB2J_results directory.")


def gen_exchange_siesta(
                        fdf_fname,
                        magnetic_elements=[],
                        kmesh=[4, 4, 4],
                        emin=-12.0,
                        emax=0.0,
                        nz=50,
                        #height=0.2,
                        #nz1=50,
                        #nz2=200,
                        #nz3=50,
                        exclude_orbs=[],
                        Rcut=None,
                        ne=None,
                        description=''):

    try:
        import sisl
    except:
        raise ImportError("sisl cannot be imported. Please install sisl first.")
    fdf = sisl.get_sile(fdf_fname)
    H = fdf.read_hamiltonian()
    if H.spin.is_colinear:
        print("Reading Siesta hamiltonian: colinear spin.")
        tbmodel_up = SislWrapper(H, spin=0)
        tbmodel_dn = SislWrapper(H, spin=1)
        basis = dict(zip(tbmodel_up.orbs, list(range(tbmodel_up.norb))))
        print("Starting to calculate exchange.")
        exchange = ExchangeCL2(
            tbmodels=(tbmodel_up, tbmodel_dn),
            atoms=tbmodel_up.atoms,
            basis=basis,
            efermi=0.0,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            #height=height,
            #nz1=nz1,
            #nz2=nz2,
            #nz3=nz3,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")

    elif H.spin.is_colinear:
        print("Reading Siesta hamiltonian: colinear spin. Treat as non-colinear")
        tbmodel = SislWrapper(H, spin='merge')
        basis = dict(zip(tbmodel.orbs, list(range(tbmodel.nbasis))))
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=basis,
            efermi=0.0,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            #height=height,
            #nz1=nz1,
            #nz2=nz2,
            #nz3=nz3,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")

    elif H.spin.is_spinorbit:
        print("Reading Siesta hamiltonian: non-colinear spin.")
        tbmodel = SislWrapper(H, spin=None)
        basis = dict(zip(tbmodel.orbs, list(range(tbmodel.nbasis))))
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=basis,
            efermi=0.0,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            description=description)
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")


def gen_exchange_gpaw(
                        gpw_fname,
                        magnetic_elements=[],
                        kmesh=[3, 3, 3],
                        emin=-12.0,
                        emax=0.0,
                        nz=50,
                        exclude_orbs=[],
                        Rcut=None,
                        description=''):
    print("Reading from GPAW data and calculate electronic structure.")
    model=GPAWWrapper(gpw_fname=gpw_fname)
    efermi=model.calc.get_fermi_level()
    print(f"Fermi Energy: {efermi}")
    poses=np.vstack([model.positions, model.positions])
    basis, _ = auto_assign_basis_name(poses, model.atoms)
    if model.calc.get_spin_polarized():
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(
            tbmodels=model,
            atoms=model.atoms,
            efermi=efermi,
            basis=basis,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            description=description)
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")

