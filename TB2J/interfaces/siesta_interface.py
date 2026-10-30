import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

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


def _require_wfsx_exchange_operations(path, operation, params):
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    bool_options = (
        "read_H_soc",
        "orth",
        "use_gpu",
        "ibz",
        "split_soc",
        "spin_spiral",
        "dmft",
        "skip",
        "vectorize_energy",
    )
    operator_options = (
        "mae_angles",
        "so_strength",
        "qspiral",
        "self_energy",
        "e_batch_size",
    )
    if (
        operation != "exchange"
        or any(params.get(key, False) for key in bool_options)
        or any(params.get(key) is not None for key in operator_options)
        or params.get("nproc", 1) != 1
    ):
        raise UnsupportedOperatorMode(
            f"file {path}: plain WFSX supports single-process CPU full-operator exchange; "
            "remove split-SOC/MAE/rescaling, spiral/GBT, GPU batching, IBZ, "
            "orthogonalization, DMFT or split-SOC skip options"
        )


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
    _require_wfsx_exchange_operations(source.wfsx_path, operation, params)
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


@contextmanager
def _measure_wfsx_eigensolvers():
    """Measure actual public electronic solver calls for single-process runs."""
    import scipy.linalg
    from HamiltonIO import lcao_hamiltonian
    from HamiltonIO.siesta import sisl_wrapper

    stats = {
        "calls": 0,
        "seconds": 0.0,
        "method": "NumPy/SciPy eig/eigh/eigvals/eigvalsh and HamiltonIO SIESTA/LCAO eigh bindings; outermost calls",
    }
    originals = []
    depth = 0

    def instrument(original):
        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal depth
            outermost = depth == 0
            if outermost:
                stats["calls"] += 1
                start = time.perf_counter()
            depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                depth -= 1
                if outermost:
                    stats["seconds"] += time.perf_counter() - start

        return measured

    try:
        for module, names in (
            (np.linalg, ("eig", "eigh", "eigvals", "eigvalsh")),
            (scipy.linalg, ("eig", "eigh", "eigvals", "eigvalsh")),
            (sisl_wrapper, ("eigh",)),
            (lcao_hamiltonian, ("eigh",)),
        ):
            for name in names:
                original = getattr(module, name)
                originals.append((module, name, original))
                setattr(module, name, instrument(original))
        yield stats
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _run_exchange_siesta_wfsx(wfsx_file, wfsx_metadata, **params):
    """Execute the existing kernel, publish measured provenance, and clean up."""
    from collections.abc import Mapping

    from HamiltonIO.siesta.wfsx import (
        IncompleteSpectrum,
        InconsistentMetadata,
        MissingMetadata,
        SiestaWFSXParser,
        UnsupportedOperatorMode,
    )

    started = time.perf_counter()
    _require_wfsx_exchange_operations(
        wfsx_file, params.get("operation", "exchange"), params
    )
    try:
        import resource
    except ImportError as exc:
        raise UnsupportedOperatorMode(
            "WFSX peak RSS measurement requires the Unix resource module; "
            "run on a supported Unix platform; ordinary H/S imports remain available"
        ) from exc
    if wfsx_file is None or wfsx_metadata is None:
        raise MissingMetadata(
            "wfsx_file/wfsx_metadata: supply complete states and explicit metadata"
        )
    metadata_path = None
    if isinstance(wfsx_metadata, (str, os.PathLike)):
        metadata_path = Path(wfsx_metadata)
        try:
            metadata = json.loads(metadata_path.read_text())
        except OSError as exc:
            raise MissingMetadata(
                f"file {metadata_path}: supply readable metadata JSON"
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise InconsistentMetadata(
                f"file {metadata_path}: correct malformed metadata JSON"
            ) from exc
    elif isinstance(wfsx_metadata, Mapping):
        metadata = dict(wfsx_metadata)
    else:
        raise MissingMetadata(
            "wfsx_metadata: supply a metadata mapping or readable JSON path"
        )
    source = SiestaWFSXParser(wfsx_file, metadata)
    inputs = []
    for role, path in (("wavefunctions", Path(wfsx_file)), ("metadata", metadata_path)):
        if path is not None:
            digest = hashlib.sha256()
            try:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                category = (
                    IncompleteSpectrum if role == "wavefunctions" else MissingMetadata
                )
                raise category(
                    f"file {path}: supply readable complete {role} input"
                ) from exc
            inputs.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": digest.hexdigest(),
                }
            )
    if metadata_path is None:
        inputs.append(
            {
                "role": "metadata",
                "path": None,
                "sha256": hashlib.sha256(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        allow_nan=False,
                        default=lambda value: value.tolist(),
                    ).encode()
                ).hexdigest(),
                "method": "canonical JSON mapping, sorted keys",
            }
        )
    output_path = Path(params.get("output_path", "TB2J_results"))
    write_density_matrix = params.pop("write_density_matrix", False)
    with _measure_wfsx_eigensolvers() as solvers:
        exchange = prepare_exchange_siesta_wfsx(source, **params)
        try:
            greens = (
                [exchange.G] if hasattr(exchange, "G") else [exchange.Gup, exchange.Gdn]
            )
            stats = [green.spectral_stats for green in greens]
            first = greens[0]
            before_integration = time.perf_counter()
            exchange.calculate_all()
            integration_seconds = time.perf_counter() - before_integration
            before_output = time.perf_counter()
            exchange.write_output(path=output_path)
            if write_density_matrix:
                filenames = (
                    ["rhoR.pickle"]
                    if len(greens) == 1
                    else ["rho_up.pickle", "rho_dn.pickle"]
                )
                for green, filename in zip(greens, filenames):
                    green.write_rho_R(exchange.Rlist, fname=output_path / filename)
            output_seconds = time.perf_counter() - before_output
            channels = 2 if source.metadata.nspin == 2 else 1
            supplied = len(first._spectral_map.record_representatives)
            summary = {
                "schema_version": 1,
                "input_mode": "wfsx",
                "timing_seconds": {
                    "import": sum(item["import_seconds"] for item in stats),
                    "preparation": sum(item["preparation_seconds"] for item in stats),
                    "integration": integration_seconds,
                    "output": output_seconds,
                },
                "electronic_eigensolver": solvers,
                "inputs": inputs,
                "metadata": {
                    **{
                        key: getattr(source.metadata, key)
                        for key in (
                            "source_provenance",
                            "fermi_source",
                            "electron_source",
                            "electron_count",
                            "occupation_convention",
                            "electronic_temperature_k",
                            "weight_semantics",
                            "gauge",
                        )
                    },
                    "scalar_symmetry": dict(source.metadata.scalar_symmetry),
                },
                "symmetry": [
                    {
                        "provenance": green._spectral_map.symmetry_provenance,
                        "operator_validation_required": bool(
                            green._spectral_map.requires_operator_validation
                        ),
                        "partner_expansion_events": green.spectral_stats[
                            "scalar_partner_expansions"
                        ],
                    }
                    for green in greens
                ],
                "spin_mode": source.metadata.nspin,
                "source_precision": sorted(
                    {tuple(pair) for item in stats for pair in item["source_precision"]}
                ),
                "matrix_sizes": {
                    "basis": first.nbasis,
                    "orbitals": first.norb,
                    "channels": channels,
                },
                "states": {
                    "supplied": supplied * channels * first.nbasis,
                    "requested": first.nkpts * channels * first.nbasis,
                    "per_record": first.nbasis,
                },
                "kpoints": {
                    "supplied": supplied,
                    "requested": first.nkpts,
                    "representatives_per_channel": [
                        item["representative_kpoints"] for item in stats
                    ],
                    "requested_fractional": first.kpts.tolist(),
                    "weights": first.kweights.tolist(),
                    "rlist_mesh": [int(value) for value in exchange.kmesh],
                },
                "energy_reference": {
                    "shifted_fermi_ev": first.efermi,
                    "source_absolute_fermi_ev": source.metadata.fermi_energy_ev,
                },
                "integration_width_ev": float(exchange.smearing),
                "green_mode": (
                    "contravariant"
                    if os.environ.get("TB2J_SIESTA_GREEN_MODE") == "contravariant"
                    else "default"
                ),
                "cache": {
                    "mode": "disk-spectral",
                    "owned_children_removed": False,
                    "spectral_disk_bytes": sum(
                        item["spectral_disk_bytes"] for item in stats
                    ),
                },
                "nproc": 1,
            }
        finally:
            exchange.finalize()
    summary["cache"]["owned_children_removed"] = all(
        not Path(green.cache_path).exists() for green in greens
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary["peak_rss"] = {
        "value": int(rss * (1 if sys.platform == "darwin" else 1024)),
        "unit": "bytes",
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; process-lifetime high-water; macOS bytes, Linux KiB converted; no children",
    }
    summary["timing_seconds"]["total"] = time.perf_counter() - started
    (output_path / "wfsx_run.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def gen_exchange_siesta(
    fdf_fname=None,
    read_H_soc=False,
    use_gpu=False,
    skip=False,
    *,
    input_mode="hs",
    wfsx_file=None,
    wfsx_metadata=None,
    **kwargs,
):
    """
    parameters:
        fdf_fname: str
            The fdf file for the calculation.
        read_H_soc: bool
            Whether to read the SOC Hamiltonian. Default is False.
        use_gpu: bool
            Whether to use GPU acceleration for MAE calculation. Default is False.
        input_mode: str
            "hs" (ordinary default) or "wfsx" (complete wavefunctions, no H/S).
        wfsx_file, wfsx_metadata:
            Complete WFSX and explicit metadata mapping or user-authored JSON path.
            WFSX mode returns a measured summary and writes wfsx_run.json.

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
    if input_mode == "wfsx":
        return _run_exchange_siesta_wfsx(
            wfsx_file,
            wfsx_metadata,
            read_H_soc=read_H_soc,
            use_gpu=use_gpu,
            skip=skip,
            **kwargs,
        )
    if input_mode != "hs" or wfsx_file is not None or wfsx_metadata is not None:
        from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

        raise UnsupportedOperatorMode(
            "input_mode: select 'wfsx' with wavefunction/metadata inputs, or 'hs' without them"
        )

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
