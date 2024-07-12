import os

from HamiltonIO.lawaf import LawafHamiltonian as Ham

from .manager import Manager


class LaWaFManager(Manager):
    def __init__(
        self,
        path,
        prefix_up,
        prefix_dn,
        prefix_SOC,
        colinear=True,
        groupby="spin",
        basis_fname=None,
        qspace=False,
        wannier_type="LaWaF",
        **kwargs,
    ):
        # models and basis
        if colinear:
            tbmodels, basis, atoms = self.prepare_model_colinear(
                path, prefix_up, prefix_dn
            )
        else:
            tbmodels, basis, atoms = self.prepare_model_ncl(path, prefix_SOC)

        description = self.description(path, prefix_up, prefix_dn, prefix_SOC, colinear)
        kwargs["description"] = description

        super().__init__(atoms, tbmodels, basis, colinear=colinear, **kwargs)

    def prepare_model_colinear(self, path, prefix_up, prefix_dn):
        print("Reading LawaF hamiltonian: spin up.")
        tbmodel_up = Ham.load_pickle(os.path.join(path, f"{prefix_up}.pickle"))
        print("Reading LaWaF hamiltonian: spin down.")
        tbmodel_dn = Ham.load_pickle(os.path.join(path, f"{prefix_dn}.pickle"))
        tbmodels = (tbmodel_up, tbmodel_dn)
        basis = tbmodel_up.wann_names
        atoms = tbmodel_up.atoms
        return tbmodels, basis, atoms

    def prepare_model_ncl(self, path, prefix_SOC):
        print("Reading LaWaF hamiltonian: non-colinear spin.")
        tbmodel = Ham.load_pickle(os.path.join(path, f"{prefix_SOC}.pickle"))
        basis = tbmodel.wann_names
        atoms = tbmodel.atoms
        return tbmodel, basis, atoms

    def description(self, path, prefix_up, prefix_dn, prefix_SOC, colinear=True):
        if colinear:
            description = f""" Input from collinear LaWaF data.
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
