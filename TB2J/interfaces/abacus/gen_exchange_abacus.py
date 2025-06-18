#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main function to compute exchange interaction from abacus data
"""

import os
from pathlib import Path

# from TB2J.abacus.abacus_wrapper import AbacusParser
from HamiltonIO.abacus import AbacusParser

from TB2J.exchange import ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2


def gen_exchange_abacus(
    path,
    suffix="Abacus",
    binary=False,
    magnetic_elements=[],
    include_orbs=[],
    kmesh=[7, 7, 7],
    emin=-13.0,
    emax=0.00,
    nz=100,
    exclude_orbs=[],
    Rcut=None,
    use_cache=False,
    nproc=1,
    output_path="TB2J_results",
    orb_decomposition=False,
    index_magnetic_atoms=None,
    description=None,
):
    outpath = Path(path) / f"OUT.{suffix}"

    if not os.path.exists(outpath):
        raise ValueError(
            f"The path {outpath} does not exist. Please check if the path and the suffix is correct"
        )
    parser = AbacusParser(outpath=outpath, spin=None, binary=binary)
    spin = parser.read_spin()
    if spin == "collinear":
        tbmodel_up, tbmodel_dn = parser.get_models()
        efermi = parser.read_efermi()
        print("Starting to calculate exchange.")
        description = f""" Input from collinear Abacus data.
data directory: {outpath}
\n"""
        exchange = ExchangeCL2(
            tbmodels=(tbmodel_up, tbmodel_dn),
            atoms=tbmodel_up.atoms,
            basis=tbmodel_up.basis,
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
        print("\n")
        print(f"All calculation finished. The results are in {output_path} directory.")
    else:
        tbmodel = parser.get_models()
        print("Starting to calculate exchange.")
        description = f""" Input from non-collinear Abacus data.
data directory: {outpath}
\n"""
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=tbmodel.basis,
            efermi=tbmodel.efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            nproc=nproc,
            use_cache=use_cache,
            orb_decomposition=orb_decomposition,
            index_magnetic_atoms=index_magnetic_atoms,
            description=description,
        )
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")


if __name__ == "__main__":
    gen_exchange_abacus(
        path="/Users/hexu/projects/TB2J_abacus/abacus-tb2j-master/abacus_example/case_Fe/2_soc",
        suffix="Fe",
        magnetic_elements=["Fe"],
        nz=50,
        Rcut=8,
        kmesh=[7, 7, 7],
    )
