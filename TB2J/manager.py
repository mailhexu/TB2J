import os
from TB2J.myTB import MyTB, merge_tbmodels_spin
import numpy as np
from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.utils import read_basis, auto_assign_basis_name
from ase.io import read
from TB2J.sisl_wrapper import SislWrapper
from TB2J.gpaw_wrapper import GPAWWrapper
from TB2J.wannier import parse_atoms

def gen_exchange(path,
                 colinear=True,
                 groupby='spin',
                 posfile=None,
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
                 use_cache=False,
                 np=1,
                 output_path='TB2J_results',
                 wannier_type="wannier90",
                 qspace=False,
                 description=''):
    try:
        fname=os.path.join(path, posfile)
        print(f"Reading atomic structure from file {fname}.")
        atoms = read(os.path.join(path, posfile))
    except Exception:

        print(f"Cannot read atomic structure from file {fname}. Trying to read from Wannier input.")
        if colinear:
            fname=os.path.join(path, f"{prefix_up}.win")
        else:
            fname=os.path.join(path, f"{prefix_SOC}.win")

        print(f"Reading atomic structure from file {fname}.")
        atoms= parse_atoms(fname)

    basis_fname = os.path.join(path, 'basis.txt')
    if colinear:
        if wannier_type.lower() == "wannier90":
            print("Reading Wannier90 hamiltonian: spin up.")
            tbmodel_up = MyTB.read_from_wannier_dir(path=path,
                                                    prefix=prefix_up,
                                                    atoms=atoms, 
                                                    nls=False)
            print("Reading Wannier90 hamiltonian: spin down.")
            tbmodel_dn = MyTB.read_from_wannier_dir(path=path,
                                                    prefix=prefix_dn,
                                                    atoms=atoms,
                                                    nls=False)
            if os.path.exists(basis_fname):
                basis = read_basis(basis_fname)
            else:
                basis, _ = auto_assign_basis_name(tbmodel_up.xred, atoms)
        elif wannier_type.lower() == "banddownfolder":
            print("Reading Banddownfolder hamiltonian: spin up.")
            tbmodel_up = MyTB.load_banddownfolder(path=path,
                                                  prefix=prefix_up,
                                                  atoms=atoms,                                                   nls=False)
            print("Reading Banddownfolder hamiltonian: spin down.")
            tbmodel_dn = MyTB.load_banddownfolder(path=path,
                                                  prefix=prefix_dn,
                                                  atoms=atoms, 
                                                  nls=False)

            basis, _ = auto_assign_basis_name(tbmodel_up.xred, atoms)
        else:
            raise ValueError(
                "wannier_type should be Wannier90 or banddownfolder.")

        print("Starting to calculate exchange.")
        description = f""" Input from collinear Wannier90 data.
 Tight binding data from {path}. 
 Prefix of wannier function files:{prefix_up} and {prefix_dn}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
\n"""
        if not qspace:
            exchange = ExchangeCL2(tbmodels=(tbmodel_up, tbmodel_dn),
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
                                   np=np,
                                   use_cache=use_cache,
                                   description=description)
        else:
            exchange = ExchangeCLQspace(tbmodels=(tbmodel_up, tbmodel_dn),
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
                                        np=np,
                                        use_cache=use_cache,
                                        description=description)

        exchange.run(path=output_path)
        print(
            "All calculation finsihed. The results are in TB2J_results directory."
        )

    elif colinear and wannier_type.lower() == "banddownfolder":
        print("Reading Wannier90 hamiltonian: spin up.")
        tbmodel_up = MyTB.read_from_wannier_dir(path=path,
                                                prefix=prefix_up,
                                                atoms=atoms,
                                                groupby=None,
                                                nls=False)
        print("Reading Wannier90 hamiltonian: spin down.")
        tbmodel_dn = MyTB.read_from_wannier_dir(path=path,
                                                prefix=prefix_dn,
                                                atoms=atoms,
                                                groupby=None,
                                                nls=False)
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)
        basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        description = f""" Input from collinear BandDownfolder data.
 Tight binding data from {path}. 
 Prefix of wannier function files:{prefix_up} and {prefix_dn}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
\n"""
        print("Starting to calculate exchange.")
        exchange = ExchangeCL2(tbmodels=tbmodel,
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
                               np=np,
                               use_cache=use_cache,
                               description=description)
        exchange.run(path=output_path)
        print(
            "All calculation finsihed. The results are in TB2J_results directory."
        )
    else:
        print("Reading Wannier90 hamiltonian: non-colinear spin.")
        groupby = groupby.lower().strip()
        if groupby not in ['spin', 'orbital']:
            raise ValueError("groupby can only be spin or orbital.")
        tbmodel = MyTB.read_from_wannier_dir(path=path,
                                             prefix=prefix_SOC,
                                             atoms=atoms,
                                             groupby=groupby,
                                             nls=True)
        if os.path.exists(basis_fname):
            print("The use of basis file is deprecated. It will be ignored.")
            #basis = read_basis(basis_fname)
        else:
            basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)
        description = f""" Input from non-collinear Wannier90 data.
 Tight binding data from {path}. 
 Prefix of wannier function files:{prefix_SOC}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
 The DMI component parallel to the spin orientation, the Jani which has the component of that orientation should be disregarded
 e.g. if the spins are along z, the xz, yz, zz, zx, zy components and the z component of DMI.
 If you need these component, try to do three calculations with spin along x, y, z,  or use structure with z rotated to x, y and z. And then use TB2J_merge.py to get the full set of parameters.

\n"""
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(tbmodels=tbmodel,
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
                               np=np,
                               use_cache=use_cache,
                               description=description)
        print("\n")
        exchange.run(path=output_path)
        print(
            f"All calculation finsihed. The results are in {output_path} directory."
        )


def gen_exchange_siesta(fdf_fname,
                        magnetic_elements=[],
                        include_orbs=None,
                        kmesh=[5, 5, 5],
                        emin=-12.0,
                        emax=0.0,
                        nz=100,
                        exclude_orbs=[],
                        Rcut=None,
                        ne=None,
                        np=1,
                        use_cache=False,
                        output_path='TB2J_results',
                        description=''):

    try:
        import sisl
    except:
        raise ImportError(
            "sisl cannot be imported. Please install sisl first.")

    from packaging import version
    if version.parse(sisl.__version__) <= version.parse("0.10.0"):
        raise ImportError(
            f"sisl version is {sisl.__version__}, but should be larger than 0.10.0."
        )
    fdf = sisl.get_sile(fdf_fname)
    H = fdf.read_hamiltonian()
    if H.spin.is_colinear:
        print("Reading Siesta hamiltonian: colinear spin.")
        tbmodel_up = SislWrapper(H, spin=0)
        tbmodel_dn = SislWrapper(H, spin=1)
        basis = dict(zip(tbmodel_up.orbs, list(range(tbmodel_up.norb))))
        print("Starting to calculate exchange.")
        description = f""" Input from collinear Siesta data.
 working directory: {os.getcwd()}
 fdf_fname: {fdf_fname}.
\n"""
        exchange = ExchangeCL2(
            tbmodels=(tbmodel_up, tbmodel_dn),
            atoms=tbmodel_up.atoms,
            basis=basis,
            efermi=0.0,
            magnetic_elements=magnetic_elements,
            include_orbs=include_orbs,
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
            np=np,
            use_cache=use_cache,
            description=description)
        exchange.run(path=output_path)
        print("\n")
        print(
            f"All calculation finsihed. The results are in {output_path} directory."
        )

    elif H.spin.is_colinear:
        print(
            "Reading Siesta hamiltonian: colinear spin. Treat as non-colinear")
        tbmodel = SislWrapper(H, spin='merge')
        basis = dict(zip(tbmodel.orbs, list(range(tbmodel.nbasis))))
        print("Starting to calculate exchange.")
        description = f""" Input from collinear Siesta data.
 working directory: {os.getcwd()}
 fdf_fname: {fdf_fname}.
\n"""
        exchange = ExchangeNCL(tbmodels=tbmodel,
                               atoms=tbmodel.atoms,
                               basis=basis,
                               efermi=0.0,
                               magnetic_elements=magnetic_elements,
                               include_orbs=include_orbs,
                               kmesh=kmesh,
                               emin=emin,
                               emax=emax,
                               nz=nz,
                               exclude_orbs=exclude_orbs,
                               Rcut=Rcut,
                               ne=ne,
                               np=np,
                               use_cache=use_cache,
                               description=description)
        exchange.run(path=output_path)
        print("\n")
        print(
            f"All calculation finsihed. The results are in {output_path} directory."
        )

    elif H.spin.is_spinorbit or H.spin.is_noncolinear:

        print("Reading Siesta hamiltonian: non-colinear spin.")
        tbmodel = SislWrapper(H, spin=None)
        basis = dict(zip(tbmodel.orbs, list(range(tbmodel.nbasis))))
        print("Starting to calculate exchange.")
        description = f""" Input from non-collinear Siesta data.
 working directory: {os.getcwd()}
 fdf_fname: {fdf_fname}.
Warning: The DMI component parallel to the spin orientation, the Jani which has the component of that orientation should be disregarded
 e.g. if the spins are along z, the xz, yz, zz, zx, zy components and the z component of DMI.
 If you need these component, try to do three calculations with spin along x, y, z,  or use structure with z rotated to x, y and z. And then use TB2J_merge.py to get the full set of parameters.
\n"""
        exchange = ExchangeNCL(tbmodels=tbmodel,
                               atoms=tbmodel.atoms,
                               basis=basis,
                               efermi=0.0,
                               magnetic_elements=magnetic_elements,
                               include_orbs=include_orbs,
                               kmesh=kmesh,
                               emin=emin,
                               emax=emax,
                               nz=nz,
                               exclude_orbs=exclude_orbs,
                               Rcut=Rcut,
                               ne=ne,
                               np=np,
                               use_cache=use_cache,
                               description=description)
        exchange.run(path=output_path)
        print("\n")
        print(
            f"All calculation finsihed. The results are in {output_path} directory."
        )


def gen_exchange_gpaw(gpw_fname,
                      magnetic_elements=[],
                      kmesh=[3, 3, 3],
                      emin=-12.0,
                      emax=0.0,
                      nz=50,
                      exclude_orbs=[],
                      Rcut=None,
                      use_cache=False,
                      output_path='TB2J_results',
                      description=''):
    print("Reading from GPAW data and calculate electronic structure.")
    model = GPAWWrapper(gpw_fname=gpw_fname)
    efermi = model.calc.get_fermi_level()
    print(f"Fermi Energy: {efermi}")
    poses = np.vstack([model.positions, model.positions])
    basis, _ = auto_assign_basis_name(poses, model.atoms)
    if model.calc.get_spin_polarized():
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(tbmodels=model,
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
                               use_cache=use_cache,
                               description=description)
        exchange.run(path=output_path)
        print("\n")
        print(
            f"All calculation finsihed. The results are in {output_path} directory."
        )
