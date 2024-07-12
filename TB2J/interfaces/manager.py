import os

import numpy as np
from ase.io import read

from TB2J.exchange import ExchangeNCL
from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.myTB import MyTB
from TB2J.utils import auto_assign_basis_name

# from TB2J.sisl_wrapper import SislWrapper

# from HamiltonIO.siesta import SislParser

try:
    from HamiltonIO.siesta import SislParser
except ImportError:
    print(
        "Cannot import SislWrapper from HamiltonIO.siesta. Please install HamiltonIO first."
    )

from TB2J.gpaw_wrapper import GPAWWrapper
from TB2J.wannier import parse_atoms


class Manager:
    def __init__(self, atoms, models, basis, colinear, **kwargs):
        # computing exchange
        print("Starting to calculate exchange.")
        ExchangeClass = self.select_exchange(colinear)
        exchange = ExchangeClass(tbmodels=models, atoms=atoms, basis=basis, **kwargs)
        output_path = kwargs.get("output_path", "TB2J_results")
        exchange.run(path=output_path)
        print(f"All calculation finished. The results are in {output_path} directory.")

    def select_exchange(self, colinear, qspace=False):
        if colinear:
            if qspace:
                return ExchangeCLQspace
            else:
                return ExchangeCL2
        else:
            return ExchangeNCL


class WannierManager(Manager):
    def __init__(
        self,
        path,
        prefix_up,
        prefix_dn,
        prefix_SOC,
        colinear=True,
        groupby="spin",
        posfile=None,
        basis_fname=None,
        qspace=False,
        wannier_type="wannier90",
        **kwargs,
    ):
        # atoms
        atoms = self.prepare_atoms(path, posfile, prefix_up, prefix_SOC, colinear)
        output_path = kwargs.get("output_path", "TB2J_results")
        # models and basis
        if colinear:
            tbmodels, basis = self.prepare_model_colinear(
                path, prefix_up, prefix_dn, atoms, output_path=output_path
            )
        else:
            tbmodels, basis = self.prepare_model_ncl(
                path, prefix_SOC, atoms, groupby, output_path=output_path
            )

        description = self.description(path, prefix_up, prefix_dn, prefix_SOC, colinear)
        kwargs["description"] = description

        super().__init__(atoms, tbmodels, basis, colinear=colinear, **kwargs)

    def prepare_atoms(self, path, posfile, prefix_up, prefix_SOC, colinear=True):
        fname = os.path.join(path, posfile)
        try:
            print(f"Reading atomic structure from file {fname}.")
            atoms = read(os.path.join(path, posfile))
        except Exception:
            print(
                f"Cannot read atomic structure from file {fname}. Trying to read from Wannier input."
            )
            if colinear:
                fname = os.path.join(path, f"{prefix_up}.win")
            else:
                fname = os.path.join(path, f"{prefix_SOC}.win")

            print(f"Reading atomic structure from file {fname}.")
            atoms = parse_atoms(fname)
        return atoms

    def prepare_model_colinear(self, path, prefix_up, prefix_dn, atoms, output_path):
        print("Reading Wannier90 hamiltonian: spin up.")
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, atoms=atoms, nls=False
        )
        print("Reading Wannier90 hamiltonian: spin down.")
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, atoms=atoms, nls=False
        )
        basis, _ = auto_assign_basis_name(
            tbmodel_up.xred,
            atoms,
            write_basis_file=os.path.join(output_path, "assigned_basis.txt"),
        )
        tbmodels = (tbmodel_up, tbmodel_dn)
        return tbmodels, basis

    def prepare_model_ncl(self, path, prefix_SOC, atoms, groupby, output_path):
        print("Reading Wannier90 hamiltonian: non-colinear spin.")
        groupby = groupby.lower().strip()
        if groupby not in ["spin", "orbital"]:
            raise ValueError("groupby can only be spin or orbital.")
        tbmodel = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_SOC, atoms=atoms, groupby=groupby, nls=True
        )
        basis, _ = auto_assign_basis_name(
            tbmodel.xred,
            atoms,
            write_basis_file=os.path.join(output_path, "assigned_basis.txt"),
        )
        return tbmodel, basis

    def description(self, path, prefix_up, prefix_dn, prefix_SOC, colinear=True):
        if colinear:
            description = f""" Input from collinear Wannier90 data.
 Tight binding data from {path}.
 Prefix of wannier function files:{prefix_up} and {prefix_dn}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
\n"""

        else:
            description = f""" Input from non-collinear Wannier90 data.
Tight binding data from {path}.
Prefix of wannier function files:{prefix_SOC}.
Warning: Please check if the noise level of Wannier function Hamiltonian to make sure it is much smaller than the exchange values.
The DMI component parallel to the spin orientation, the Jani which has the component of that orientation should be disregarded
e.g. if the spins are along z, the xz, yz, zz, zx, zy components and the z component of DMI.
If you need these component, try to do three calculations with spin along x, y, z,  or use structure with z rotated to x, y and z. And then use TB2J_merge.py to get the full set of parameters.\n"""

        return description


gen_exchange = WannierManager


def gen_exchange_siesta(
    fdf_fname,
    read_H_soc=False,
    magnetic_elements=[],
    include_orbs=None,
    kmesh=[5, 5, 5],
    emin=-12.0,
    emax=0.0,
    nz=100,
    exclude_orbs=[],
    Rcut=None,
    ne=None,
    nproc=1,
    use_cache=False,
    output_path="TB2J_results",
    orb_decomposition=False,
    description="",
):
    try:
        import sisl
    except ImportError:
        raise ImportError("sisl cannot be imported. Please install sisl first.")

    from packaging import version

    if version.parse(sisl.__version__) <= version.parse("0.10.0"):
        raise ImportError(
            f"sisl version is {sisl.__version__}, but should be larger than 0.10.0."
        )

    include_orbs = {}
    if isinstance(magnetic_elements, str):
        magnetic_elements = [magnetic_elements]
    for element in magnetic_elements:
        if "_" in element:
            elem = element.split("_")[0]
            orb = element.split("_")[1:]
            include_orbs[elem] = orb
        else:
            include_orbs[element] = None
    magnetic_elements = list(include_orbs.keys())

    parser = SislParser(fdf_fname=fdf_fname, ispin=None, read_H_soc=read_H_soc)
    if parser.spin.is_colinear:
        print("Reading Siesta hamiltonian: colinear spin.")
        # tbmodel_up = SislWrapper(fdf_fname=None, sisl_hamiltonian=H, spin=0, geom=geom)
        # tbmodel_dn = SislWrapper(fdf_fname=None, sisl_hamiltonian=H, spin=1, geom=geom)
        tbmodel_up, tbmodel_dn = parser.get_model()
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
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            ne=ne,
            nproc=nproc,
            use_cache=use_cache,
            output_path=output_path,
            description=description,
        )
        exchange.run(path=output_path)
        print("\n")
        print(f"All calculation finished. The results are in {output_path} directory.")

    elif parser.spin.is_spinorbit:
        print("Reading Siesta hamiltonian: non-colinear spin.")
        model = parser.get_model()
        basis = dict(zip(model.orbs, list(range(model.nbasis))))
        print("Starting to calculate exchange.")
        description = f""" Input from non-collinear Siesta data.
 working directory: {os.getcwd()}
 fdf_fname: {fdf_fname}.
Warning: The DMI component parallel to the spin orientation, the Jani which has the component of that orientation should be disregarded
 e.g. if the spins are along z, the xz, yz, zz, zx, zy components and the z component of DMI.
 If you need these component, try to do three calculations with spin along x, y, z,  or use structure with z rotated to x, y and z. And then use TB2J_merge.py to get the full set of parameters.
\n"""
        if not model.split_soc:
            exchange = ExchangeNCL(
                tbmodels=model,
                atoms=model.atoms,
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
                nproc=nproc,
                use_cache=use_cache,
                description=description,
                output_path=output_path,
                orb_decomposition=orb_decomposition,
            )
            exchange.run(path=output_path)
            print("\n")
            print(
                f"All calculation finished. The results are in {output_path} directory."
            )
        else:
            angle = {"x": (np.pi / 2, 0), "y": (np.pi / 2, np.pi / 2), "z": (0, 0)}
            for key, val in angle.items():
                # model = parser.get_model()
                theta, phi = val
                model.set_Hsoc_rotation_angle([theta, phi])
                basis = dict(zip(model.orbs, list(range(model.nbasis))))
                output_path_full = f"{output_path}_{key}"
                exchange = ExchangeNCL(
                    tbmodels=model,
                    atoms=model.atoms,
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
                    nproc=nproc,
                    use_cache=use_cache,
                    description=description,
                    output_path=output_path_full,
                    orb_decomposition=orb_decomposition,
                )
                exchange.run(path=output_path_full)
                print("\n")
                print(
                    f"All calculation finished. The results are in {output_path_full} directory."
                )


def gen_exchange_gpaw(
    gpw_fname,
    magnetic_elements=[],
    kmesh=[3, 3, 3],
    emin=-12.0,
    emax=0.0,
    nz=50,
    exclude_orbs=[],
    Rcut=None,
    use_cache=False,
    output_path="TB2J_results",
    description="",
):
    print("Reading from GPAW data and calculate electronic structure.")
    model = GPAWWrapper(gpw_fname=gpw_fname)
    efermi = model.calc.get_fermi_level()
    print(f"Fermi Energy: {efermi}")
    poses = np.vstack([model.positions, model.positions])
    basis, _ = auto_assign_basis_name(
        poses,
        model.atoms,
        write_basis_file=os.path.join(output_path, "assigned_basis.txt"),
    )

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
            use_cache=use_cache,
            output_path=output_path,
            description=description,
        )
        exchange.run(path=output_path)
        print("\n")
        print(f"All calculation finished. The results are in {output_path} directory.")
