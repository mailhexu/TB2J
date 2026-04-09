#!/usr/bin/env python3
import os
from pathlib import Path

from ase.io import read
from HamiltonIO.openmx import OpenMXParser

from TB2J.exchange import ExchangeCL, ExchangeNCL
from TB2J.utils import symbol_number


def _build_basis(atoms, orb_dict, nspin):
    if atoms is None or orb_dict is None:
        return []

    symbols = atoms.get_chemical_symbols()
    sn = list(symbol_number(symbols).keys())

    basis = []
    for ia, orbs in orb_dict.items():
        for orb_name in orbs:
            if "|up" in orb_name:
                spin = "up"
            elif "|down" in orb_name:
                spin = "down"
            else:
                spin = None

            parts = orb_name.split("|")
            orb_sym = parts[1] if len(parts) > 1 else orb_name
            basis.append((sn[ia], orb_sym, spin))

    return basis


def gen_exchange_openmx(
    path,
    prefix="openmx",
    magnetic_elements=None,
    include_orbs=None,
    kmesh=[7, 7, 7],
    emin=-13.0,
    emax=0.00,
    nz=100,
    exclude_orbs=None,
    Rcut=None,
    use_cache=False,
    nproc=1,
    output_path="TB2J_results",
    orb_decomposition=False,
    index_magnetic_atoms=None,
    description=None,
):
    scfout_path = Path(path) / f"{prefix}.scfout"
    xyz_path = Path(path) / f"{prefix}.xyz"

    if not scfout_path.exists():
        raise ValueError(
            f"The scfout file {scfout_path} does not exist. "
            "Please check if the path and prefix are correct."
        )

    if magnetic_elements is None:
        magnetic_elements = []
    if include_orbs is None:
        include_orbs = {}
    if exclude_orbs is None:
        exclude_orbs = []

    parser = OpenMXParser(str(scfout_path))
    spin = parser.get_spin_type()
    efermi = parser.get_fermi_energy()
    cell = parser.get_cell()

    auto_desc = f"""Using OpenMX data via HamiltonIO:
 path: {os.path.abspath(path)}
 prefix: {prefix}
"""
    user_desc = description or ""
    description = auto_desc + user_desc

    if spin == "collinear":
        model_up, model_down = parser.get_model()

        if xyz_path.exists():
            atoms = read(str(xyz_path))
            atoms.set_cell(cell, scale_atoms=False)
            atoms.set_pbc(True)
            model_up.atoms = atoms
            model_down.atoms = atoms

        model_up.basis = _build_basis(model_up.atoms, model_up.orb_dict, nspin=2)
        model_down.basis = model_up.basis

        print("Starting to calculate exchange (collinear spin).")

        exchange = ExchangeCL(
            tbmodels=(model_up, model_down),
            atoms=model_up.atoms,
            basis=model_up.basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            include_orbs=include_orbs,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            nproc=nproc,
            use_cache=use_cache,
            output_path=output_path,
            orb_decomposition=orb_decomposition,
            index_magnetic_atoms=index_magnetic_atoms,
            description=description,
        )
        exchange.run(path=output_path)
    else:
        model = parser.get_model()

        if xyz_path.exists():
            atoms = read(str(xyz_path))
            atoms.set_cell(cell, scale_atoms=False)
            atoms.set_pbc(True)
            model.atoms = atoms

        model.basis = _build_basis(
            model.atoms, model.orb_dict, nspin=2 if spin == "soc" else 1
        )

        print("Starting to calculate exchange (SOC/non-collinear).")

        exchange = ExchangeNCL(
            tbmodels=model,
            atoms=model.atoms,
            basis=model.basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            include_orbs=include_orbs,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            nproc=nproc,
            use_cache=use_cache,
            output_path=output_path,
            orb_decomposition=orb_decomposition,
            index_magnetic_atoms=index_magnetic_atoms,
            description=description,
        )
        exchange.run(path=output_path)

    print(f"\nAll calculation finished. The results are in {output_path} directory.")


if __name__ == "__main__":
    gen_exchange_openmx(
        path="/Users/hexu/projects/TB2J_dev/TB2J_OpenMX/examples/SrMnO3_FM_SOC",
        prefix="openmx",
        magnetic_elements=["Mn"],
        nz=50,
        Rcut=8,
        kmesh=[7, 7, 7],
    )
