import os

import numpy as np

from TB2J.exchange import ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.io_merge import merge
from TB2J.MAEGreen import MAEGreen

try:
    from HamiltonIO.siesta import SislParser
except ImportError:
    print(
        "Cannot import SislWrapper from HamiltonIO.siesta. Please install HamiltonIO first."
    )


def siesta_anisotropy(**kwargs):
    pass


def gen_exchange_siesta(fdf_fname, read_H_soc=False, **kwargs):
    """
    parameters:
        fdf_fname: str
            The fdf file for the calculation.
        read_H_soc: bool
            Whether to read the SOC Hamiltonian. Default is False.

    parameters in **kwargs:
        magnetic_elements: list of str
            The magnetic elements to calculate the exchange. e.g. ["Fe", "Co", "Ni"]
        include_orbs: dict
            The included orbitals for each magnetic element. e.g. {"Fe": ["d"]}. Default is None.
        kmesh: list of int
            The k-point mesh for the calculation. e.g. [5, 5, 5]. Default is [5, 5, 5].
        emin: float
            The minimum energy for the calculation. Default is -12.0.
        emax: float
            The maximum energy for the calculation. Default is 0.0.
        nz: int
            The number of points for the energy mesh. Default is 100.
        exclude_orbs: list of str
            The excluded orbitals for the calculation. Default is [].
        Rcut: float
            The cutoff radius for the exchange calculation in angstrom. Default is None.
        ne: int
            The number of electrons for the calculation. Default is None.
        nproc: int
            The number of processors for the calculation. Default is 1.
        use_cache: bool
            Whether to use the cache for the calculation. Default is False.
        output_path: str
            The output path for the calculation. Default is "TB2J_results".
        orb_decomposition: bool
            Whether to calculate the orbital decomposition. Default is False.
        orth: bool
            Whether to orthogonalize the orbitals. Default is False.
        description: str
            The description for the calculation. Default is "".
    """

    exargs = dict(
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
        ibz=False,
        description="",
    )
    exargs.update(kwargs)
    try:
        import sisl
    except ImportError:
        raise ImportError("sisl cannot be imported. Please install sisl first.")

    from packaging import version

    if version.parse(sisl.__version__) <= version.parse("0.10.0"):
        raise ImportError(
            f"sisl version is {sisl.__version__}, but should be larger than 0.10.0."
        )

    output_path = exargs.pop("output_path")

    parser = SislParser(
        fdf_fname=fdf_fname, ispin=None, read_H_soc=read_H_soc, orth=exargs["orth"]
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
            # magnetic_elements=exargs['magnetic_elements'],
            # include_orbs=ex
            **exargs,
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
        exargs["description"] = description
        if not model.split_soc:
            exchange = ExchangeNCL(
                tbmodels=model,
                atoms=model.atoms,
                basis=basis,
                efermi=0.0,
                # magnetic_elements=magnetic_elements,
                # include_orbs=include_orbs,
                output_path=output_path,
                **exargs,
            )
            exchange.run(path=output_path)
            print("\n")
            print(
                f"All calculation finished. The results are in {output_path} directory."
            )
        else:
            print("Starting to calculate MAE.")
            model.set_so_strength(0.0)
            MAE = MAEGreen(
                tbmodels=model,
                atoms=model.atoms,
                basis=basis,
                efermi=None,
                angles="axis",
                # magnetic_elements=magnetic_elements,
                # include_orbs=include_orbs,
                **exargs,
            )
            # thetas = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            # phis = [0, 0, 0, 0]
            # MAE.set_angles(thetas=thetas, phis=phis)
            MAE.run(output_path=f"{output_path}_anisotropy", with_eigen=False)
            # print(
            #    f"MAE calculation finished. The results are in {output_path} directory."
            # )

            angle = {"x": (np.pi / 2, 0), "y": (np.pi / 2, np.pi / 2), "z": (0, 0)}
            for key, val in angle.items():
                # model = parser.get_model()
                theta, phi = val
                model.set_so_strength(1.0)
                model.set_Hsoc_rotation_angle([theta, phi])
                basis = dict(zip(model.orbs, list(range(model.nbasis))))
                output_path_full = f"{output_path}_{key}"
                exchange = ExchangeNCL(
                    tbmodels=model,
                    atoms=model.atoms,
                    basis=basis,
                    efermi=None,  # set to None, compute from efermi.
                    # magnetic_elements=magnetic_elements,
                    # include_orbs=include_orbs,
                    **exargs,
                )
                exchange.run(path=output_path_full)
                print("\n")
                print(
                    f"All calculation finished. The results are in {output_path_full} directory."
                )

            merge(
                "TB2J_results_x",
                "TB2J_results_y",
                "TB2J_results_z",
                main_path=None,
                save=True,
                write_path="TB2J_results",
            )
            print("Final TB2J_results written to TB2J_results directory.")
