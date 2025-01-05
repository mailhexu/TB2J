import os

import numpy as np

from TB2J.exchange import ExchangeNCL
from TB2J.gpaw_wrapper import GPAWWrapper
from TB2J.utils import auto_assign_basis_name


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
