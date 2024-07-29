import os

import numpy as np

from TB2J.exchange import ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2

try:
    from HamiltonIO.siesta import SislParser
except ImportError:
    print(
        "Cannot import SislWrapper from HamiltonIO.siesta. Please install HamiltonIO first."
    )

np.seterr(all="raise", under="ignore")


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
    orth=False,
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

    # fdf = sisl.get_sile(fdf_fname)
    # geom = fdf.read_geometry()
    # H = fdf.read_hamiltonian()
    # geom = H.geometry
    parser = SislParser(
        fdf_fname=fdf_fname, ispin=None, read_H_soc=read_H_soc, orth=orth
    )
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

    elif parser.spin.is_spinorbit or parser.spin.is_noncolinear:
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
                # FIXME:
                # efermi=None,
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
                    efermi=None,  # set to None, compute from efermi.
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
