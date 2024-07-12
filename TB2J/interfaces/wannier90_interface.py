import os

from ase.io import read

from TB2J.myTB import MyTB
from TB2J.utils import auto_assign_basis_name
from TB2J.wannier import parse_atoms

from .manager import Manager


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
