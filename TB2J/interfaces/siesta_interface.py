import os

import numpy as np

from TB2J.exchange import ExchangeNCL
from TB2J.exchangeCL2 import ExchangeCL2

# GPU exchange/MAE classes are imported lazily (only when use_gpu=True) so the
# default CPU import path does not pull in TB2J.gpu / JAX.
from TB2J.io_merge import merge
from TB2J.MAEGreen import MAEGreen

try:
    from HamiltonIO.siesta import SislParser
except ImportError:
    print(
        "Cannot import SislParser from HamiltonIO.siesta. Please install HamiltonIO first."
    )


def siesta_anisotropy(**kwargs):
    pass


def prepare_exchange_siesta_wfsx(
    source, *, reduce_scalar=False, operation="exchange", **params
):
    """Prepare the existing exchange kernel from complete states, without H/S.

    ``source`` is a validated HamiltonIO ``SiestaWFSXParser``. Its full ``scf_mesh``
    establishes electron provenance independently of optional requested
    ``kpts``/``kweights``. The caller owns ``calculate_all`` and ``finalize``.
    ``kmesh`` retains its ordinary R-list meaning; explicit points determine the
    actual integration domain. Only full-operator CPU exchange is supported.
    """
    from types import SimpleNamespace

    from ase import Atoms
    from ase.units import kB
    from HamiltonIO.siesta.wfsx import (
        InconsistentMetadata,
        MissingMetadata,
        SiestaWFSXParser,
        UnsupportedOperatorMode,
        UnsupportedSpinMode,
    )

    from TB2J.utils import symbol_number

    if not isinstance(source, SiestaWFSXParser):
        raise MissingMetadata("source: supply a validated SiestaWFSXParser")
    bool_options = (
        "read_H_soc",
        "orth",
        "use_gpu",
        "ibz",
        "split_soc",
        "spin_spiral",
        "dmft",
    )
    operator_options = ("mae_angles", "so_strength", "qspiral", "self_energy")
    if (
        operation != "exchange"
        or any(params.get(key, False) for key in bool_options)
        or any(key in params and params[key] is not None for key in operator_options)
        or params.get("nproc", 1) != 1
    ):
        raise UnsupportedOperatorMode(
            f"file {source.wfsx_path}: plain WFSX supports single-process CPU full-operator "
            "exchange, not split-SOC/MAE/rescaling, spiral/GBT, GPU or DMFT operations"
        )
    metadata = source.metadata
    if metadata.nspin not in (2, 4, 8):
        raise UnsupportedSpinMode(
            f"file {source.wfsx_path}: nspin={metadata.nspin}: scalar nspin=1 is a "
            "reader/Green interface; supply magnetic collinear or full spinor states for exchange"
        )
    spinor = metadata.nspin in (4, 8)
    if spinor and reduce_scalar:
        raise UnsupportedSpinMode(
            f"file {source.wfsx_path}: scalar reduction is forbidden for full spinor exchange"
        )
    if params.get("efermi") not in (None, 0):
        raise InconsistentMetadata(
            "efermi: WFSX energies are already Ef-relative; use zero"
        )
    params["efermi"] = 0.0
    params.setdefault("ne", metadata.electron_count)
    params.setdefault("smearing", metadata.electronic_temperature_k * kB)
    method = params.get("method", params.get("integration_method", "CFR")) or "CFR"
    if method.lower() in ("cfr", "cfr2") and params["smearing"] <= 0:
        raise UnsupportedOperatorMode(
            f"file {source.wfsx_path}: CFR requires positive integration broadening; "
            "explicitly supply positive smearing or select a non-CFR contour"
        )
    atoms = Atoms(
        symbols=metadata.atom_symbols,
        positions=metadata.positions_angstrom,
        cell=metadata.cell_angstrom,
        pbc=True,
    )
    atom_names = {index: name for name, index in symbol_number(atoms).items()}
    # MathDoc: tb2j/siesta-wfsx-spectral
    # Basis labels follow the validated AO/interleaved-spin row order unchanged.
    orbs = [
        f"{atom_names[iatom]}|{orbital}" + (f"|{spin}" if spinor else "")
        for iatom, orbitals in enumerate(metadata.orbital_names_by_atom)
        for orbital in orbitals
        for spin in (("up", "down") if spinor else ("",))
    ]
    models = [
        SimpleNamespace(
            name="SIESTA",
            atoms=atoms,
            orbs=orbs,
            norb=metadata.no_u,
            nbasis=metadata.nbasis,
            eigenstate_source=source,
            spin_channel=spin,
            reduce_scalar=reduce_scalar,
        )
        for spin in ((0,) if spinor else (0, 1))
    ]
    cls = ExchangeNCL if spinor else ExchangeCL2
    # Keep the partially initialized owner reachable for failure-path cleanup.
    exchange = cls.__new__(cls)
    try:
        cls.__init__(
            exchange,
            tbmodels=models[0] if spinor else tuple(models),
            atoms=atoms,
            basis=dict(zip(orbs, range(metadata.nbasis))),
            **params,
        )
    except BaseException:
        for name in ("G", "Gup", "Gdn"):
            green = getattr(exchange, name, None)
            if green is not None:
                green.clean_cache()
        raise
    return exchange


def gen_exchange_siesta(
    fdf_fname, read_H_soc=False, use_gpu=False, skip=False, **kwargs
):
    """
    parameters:
        fdf_fname: str
            The fdf file for the calculation.
        read_H_soc: bool
            Whether to read the SOC Hamiltonian. Default is False.
        use_gpu: bool
            Whether to use GPU acceleration for MAE calculation. Default is False.

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
        use_gpu=False,
        vectorize_energy=False,
        e_batch_size=None,
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
        exargs["description"] = description
        exargs["use_gpu"] = use_gpu  # Pass use_gpu through exargs

        # Choose Exchange class based on use_gpu flag
        if use_gpu:
            from TB2J.gpu.exchangeCL_gpu import ExchangeCL2GPU

            ExchangeClass = ExchangeCL2GPU
        else:
            ExchangeClass = ExchangeCL2
        exchange = ExchangeClass(
            tbmodels=(tbmodel_up, tbmodel_dn),
            atoms=tbmodel_up.atoms,
            basis=basis,
            efermi=0.0,
            # magnetic_elements=exargs['magnetic_elements'],
            # include_orbs=ex
            **exargs,
        )
        if use_gpu:
            exchange.run(
                path=output_path,
                use_gpu=use_gpu,
                vectorize_energy=exargs.get("vectorize_energy", False),
                e_batch_size=exargs.get("e_batch_size", None),
            )
        else:
            exchange.calculate_all()
            exchange.write_output(path=output_path)
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
        exargs["use_gpu"] = use_gpu  # Pass use_gpu through exargs
        if not model.split_soc:
            # Choose Exchange class based on use_gpu flag
            if use_gpu:
                from TB2J.gpu.exchange_ncl_gpu import ExchangeNCLGPU

                ExchangeClass = ExchangeNCLGPU
            else:
                ExchangeClass = ExchangeNCL
            exchange = ExchangeClass(
                tbmodels=model,
                atoms=model.atoms,
                basis=basis,
                efermi=0.0,
                # magnetic_elements=magnetic_elements,
                # include_orbs=include_orbs,
                output_path=output_path,
                **exargs,
            )
            if use_gpu:
                exchange.run(
                    path=output_path,
                    use_gpu=use_gpu,
                    vectorize_energy=exargs.get("vectorize_energy", False),
                    e_batch_size=exargs.get("e_batch_size", None),
                )
            else:
                exchange.calculate_all()
                exchange.write_output(path=output_path)
            print("\n")
            print(
                f"All calculation finished. The results are in {output_path} directory."
            )
        else:
            print("Starting to calculate MAE.")
            model.set_so_strength(0.0)

            # Choose MAE class based on use_gpu flag
            if use_gpu:
                from TB2J.gpu.mae_green_gpu import MAEGreenGPU

                MAEClass = MAEGreenGPU
            else:
                MAEClass = MAEGreen

            mae_output_path = f"{output_path}_anisotropy"
            if skip and os.path.exists(os.path.join(mae_output_path, "MAE.dat")):
                print(f"Skipping MAE: {mae_output_path}/MAE.dat already exists.")
            else:
                MAE = MAEClass(
                    tbmodels=model,
                    atoms=model.atoms,
                    basis=basis,
                    efermi=None,
                    angles="xyz",
                    # magnetic_elements=magnetic_elements,
                    # include_orbs=include_orbs,
                    **exargs,
                )
                # thetas = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
                # phis = [0, 0, 0, 0]
                # MAE.set_angles(thetas=thetas, phis=phis)
                # MAE.set_xyz_angles()
                MAE.run(
                    output_path=mae_output_path,
                    with_eigen=False,
                )

            angle = {"x": (np.pi / 2, 0), "y": (np.pi / 2, np.pi / 2), "z": (0, 0)}
            for key, val in angle.items():
                output_path_full = f"{output_path}_{key}"
                pickle_path = os.path.join(output_path_full, "TB2J.pickle")
                if skip and os.path.exists(pickle_path):
                    print(
                        f"Skipping exchange {key}: {output_path_full}/TB2J.pickle already exists."
                    )
                    continue
                theta, phi = val
                model.set_so_strength(1.0)
                model.set_Hsoc_rotation_angle([theta, phi])
                basis = dict(zip(model.orbs, list(range(model.nbasis))))
                # Choose Exchange class based on use_gpu flag
                if use_gpu:
                    from TB2J.gpu.exchange_ncl_gpu import ExchangeNCLGPU

                    ExchangeClass = ExchangeNCLGPU
                else:
                    ExchangeClass = ExchangeNCL
                exchange = ExchangeClass(
                    tbmodels=model,
                    atoms=model.atoms,
                    basis=basis,
                    efermi=None,  # set to None, compute from efermi.
                    **exargs,
                )
                if use_gpu:
                    exchange.run(
                        path=output_path_full,
                        use_gpu=use_gpu,
                        vectorize_energy=exargs.get("vectorize_energy", False),
                        e_batch_size=exargs.get("e_batch_size", None),
                    )
                else:
                    exchange.calculate_all()
                    exchange.write_output(path=output_path_full)
                print("\n")
                print(
                    f"All calculation finished. The results are in {output_path_full} directory."
                )

            merge(
                f"{output_path}_x",
                f"{output_path}_y",
                f"{output_path}_z",
                main_path=None,
                save=True,
                write_path=output_path,
            )
            print(f"Final TB2J_results written to {output_path} directory.")
