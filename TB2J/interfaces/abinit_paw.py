"""ABINIT PAW projector-space exchange assembly and CLI.

This module implements Stories 004+005 of the PAW WFK+Vxc exchange pipeline:

* :func:`assemble_paw_exchange_data` (Story 004) — assemble ABINIT PAW
  projection coefficients and the spin-difference ``Delta_ij`` read from the
  ABINIT ``pawprt`` log block into a :class:`~TB2J.projector_green.ProjectorGreenData`
  ready for the controlled projector exchange trace.

* :func:`gen_exchange_abinit_paw` (Story 005) — end-to-end entry point that
  projects a WFK file (or loads pre-projected data), reads ``Delta_ij`` from the
  log, assembles the spectral data, evaluates the exchange trace, and writes
  ``exchange.out``.

Architecture decisions (P2/P3/P4):

* **P2 — dual projector coefficients, no S^-1.**  The stored projection
  coefficients are bare dual-projector overlaps ``<~p|psi_n>``.  ``overlap_k``
  is ``None`` so the Green function is used as-is (dual-dual metric) without any
  inverse-overlap dressing.

* **P3 — pawprt Dij as complete on-site Delta.**  The spin-difference of the
  ABINIT total Dij (``D^up - D^down``) is stored as the ``delta_xc`` operator
  component.  For collinear DFT this is the complete on-site exchange splitting
  because Hartree and kinetic PAW terms are spin independent.

* **P4 — abinao projects, TB2J consumes.**  The projection runs in abinao
  (Story 003) and produces the ``cprj`` coefficients; TB2J consumes them here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from TB2J.projector_green import ProjectorGreenData

# Hartree → eV conversion (same value as ase.units.Ha / abinao wfk.py).
HARTREE_TO_EV = 27.211386245988

__all__ = [
    "assemble_paw_exchange_data",
    "gen_exchange_abinit_paw",
    "run_gen_exchange_abinit_paw",
    "save_projected_data",
    "load_projected_data",
]


# ---------------------------------------------------------------------------
# Story 004 — Assembly
# ---------------------------------------------------------------------------


def assemble_paw_exchange_data(
    cprj_per_kpt: list,
    delta_ij: dict[int, np.ndarray],
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    natom: int,
    nproj_per_atom: int,
    *,
    delta_unit: str = "eV",
    occupations: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
    projector_l: np.ndarray | None = None,
    projector_m: np.ndarray | None = None,
    projector_radial: np.ndarray | None = None,
    efermi_spin: np.ndarray | None = None,
    metadata: dict | None = None,
) -> ProjectorGreenData:
    """Assemble ABINIT PAW projection results into exchange-ready spectral data.

    Parameters
    ----------
    cprj_per_kpt:
        ``[nkpt]`` list of ``[nsppol]`` lists of ``complex128`` arrays with shape
        ``(natom, nproj_per_atom, nband)`` — the dual-projector coefficients
        ``<~p|psi_n>`` produced by abinao (Story 003).  A flat
        ``(nproj_total, nband)`` layout per spin is also accepted.
    delta_ij:
        ``{atom_index: Delta_matrix}`` from
        :func:`abinao.pawprt_parser.compute_delta_ij` (``D^up - D^down`` per atom).
    eigenvalues:
        ``(nsppol, nkpt, nband)`` band energies in eV.
    kweights:
        ``(nkpt,)`` k-point weights.
    kpoints:
        ``(nkpt, 3)`` reduced k-point coordinates.
    efermi:
        Fermi energy in eV.
    natom:
        Number of atoms.
    nproj_per_atom:
        Number of PAW projector channels per atom.
    delta_unit:
        Energy unit of ``delta_ij``: ``"eV"`` (default) or ``"hartree"``.
        The stored ``delta_xc`` is always converted to eV.
    occupations, cell, positions, atomic_numbers, projector_l/m/radial, efermi_spin:
        Optional metadata forwarded to :class:`ProjectorGreenData`.
    metadata:
        Extra metadata dict merged into the data record.

    Returns
    -------
    ProjectorGreenData
        A validated spectral-data container with ``overlap_k=None`` (dual
        no-dressing), a block-diagonal ``delta_xc`` operator component, and
        site-projector indexing.  Wrap it in
        :class:`~TB2J.projector_green.ProjectorGreen` and feed to
        :func:`~TB2J.projector_green.projector_exchange_trace`.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError("eigenvalues must have shape (nsppol, nkpt, nband)")
    nsppol, nkpt, nband = eigenvalues.shape

    kpoints = np.asarray(kpoints, dtype=float)
    kweights = np.asarray(kweights, dtype=float)
    nproj_total = natom * nproj_per_atom

    if kpoints.shape != (nkpt, 3):
        raise ValueError(f"kpoints shape {kpoints.shape} inconsistent with nkpt={nkpt}")
    if kweights.shape != (nkpt,):
        raise ValueError(
            f"kweights shape {kweights.shape} inconsistent with nkpt={nkpt}"
        )
    if len(cprj_per_kpt) != nkpt:
        raise ValueError(
            f"cprj_per_kpt has {len(cprj_per_kpt)} k-points, expected {nkpt}"
        )

    # -- Build coefficients [nsppol, nkpt, nband, nproj_total] ----------------
    coefficients = np.empty((nsppol, nkpt, nband, nproj_total), dtype=complex)
    for ik in range(nkpt):
        kpt_cprj = cprj_per_kpt[ik]
        if len(kpt_cprj) != nsppol:
            raise ValueError(
                f"k-point {ik} has {len(kpt_cprj)} spins, expected {nsppol}"
            )
        for ispin in range(nsppol):
            cprj = np.asarray(kpt_cprj[ispin], dtype=complex)
            if cprj.ndim == 3:
                if cprj.shape != (natom, nproj_per_atom, nband):
                    raise ValueError(
                        f"k-point {ik} spin {ispin}: cprj shape {cprj.shape} "
                        f"!= ({natom}, {nproj_per_atom}, {nband})"
                    )
                cprj = cprj.reshape(nproj_total, nband)
            elif cprj.ndim == 2:
                if cprj.shape != (nproj_total, nband):
                    raise ValueError(
                        f"k-point {ik} spin {ispin}: cprj shape {cprj.shape} "
                        f"!= ({nproj_total}, {nband})"
                    )
            else:
                raise ValueError(
                    f"k-point {ik} spin {ispin}: cprj must be 2-D or 3-D, "
                    f"got {cprj.ndim}-D"
                )
            coefficients[ispin, ik] = cprj.T  # [nband, nproj_total]

    # -- Site / projector indexing (contiguous, zero-based) -------------------
    projector_site = np.repeat(np.arange(natom, dtype=int), nproj_per_atom)
    projector_atom = projector_site.copy()
    site_nproj = np.full(natom, nproj_per_atom, dtype=int)
    site_projector_indices = np.arange(nproj_total, dtype=int).reshape(
        natom, nproj_per_atom
    )

    # -- Block-diagonal delta_xc operator component ---------------------------
    delta_xc = np.zeros((natom, nproj_per_atom, nproj_per_atom), dtype=complex)
    for atom_key, delta in delta_ij.items():
        atom = int(atom_key)
        if atom < 0 or atom >= natom:
            raise ValueError(f"delta_ij atom index {atom} out of range [0, {natom})")
        delta = np.asarray(delta, dtype=complex)
        if delta.shape != (nproj_per_atom, nproj_per_atom):
            raise ValueError(
                f"delta_ij[{atom}] shape {delta.shape} != "
                f"({nproj_per_atom}, {nproj_per_atom})"
            )
        delta_xc[atom] = delta

    # Unit normalisation → eV (TB2J convention; eigenvalues are in eV).
    unit_factor = 1.0
    if delta_unit.strip().lower().startswith("ha"):
        unit_factor = HARTREE_TO_EV
    if unit_factor != 1.0:
        delta_xc = delta_xc * unit_factor

    # Store as both delta_xc (preferred by get_local_operator) and delta_total
    # (satisfies validate(exchange_ready=True)).  Same data, two names.
    operator_components = {"delta_xc": delta_xc, "delta_total": delta_xc}
    operator_component_metadata = {
        "delta_xc": {
            "units": "eV",
            "definition": (
                "ABINIT pawprt total Dij spin difference (D^up - D^down) as "
                "the complete on-site exchange splitting (P3)"
            ),
            "source": "ABINIT pawprt Dij via abinao.pawprt_parser",
            "operator_basis": "paw_partial_wave_channel",
            "completeness": "complete",
            "exchange_ready": "true",
            "input_unit": str(delta_unit),
        },
        "delta_total": {
            "units": "eV",
            "definition": (
                "Complete on-site Delta (pawprt Dij spin difference), alias "
                "of delta_xc for validate(exchange_ready=True)"
            ),
            "source": "ABINIT pawprt Dij",
            "completeness": "complete",
            "exchange_ready": "true",
        },
    }

    meta = {} if metadata is None else dict(metadata)
    meta.update(
        {
            "source": "ABINIT PAW projector workflow",
            "projector_basis_type": "paw",
            "source_code": "abinit",
            "coefficient_convention": "dual_projector_no_inverse (P2)",
            "delta_source": "pawprt Dij spin difference (P3)",
            "pipeline": "abinao projects (Story 003), TB2J consumes (Story 004)",
        }
    )

    return ProjectorGreenData(
        kpoints=kpoints,
        weights=kweights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=float(efermi),
        efermi_spin=efermi_spin,
        projector_site=projector_site,
        projector_atom=projector_atom,
        occupations=occupations,
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        projector_l=projector_l,
        projector_m=projector_m,
        projector_radial=projector_radial,
        overlap_k=None,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        operator_components=operator_components,
        operator_component_metadata=operator_component_metadata,
        coefficient_source="abinit_wfk_projection",
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_partial_wave_channel",
        operator_basis="native_paw_projector_hamiltonian",
        population_metric="projector_trace",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Projected-data persistence (pickle)
# ---------------------------------------------------------------------------

_PROJECTED_DATA_KEYS = (
    "cprj_per_kpt",
    "eigenvalues",
    "kweights",
    "kpoints",
    "efermi",
    "natom",
    "nproj_per_atom",
)


def save_projected_data(
    path: str | Path,
    *,
    cprj_per_kpt: list,
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    natom: int,
    nproj_per_atom: int,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
    occupations: np.ndarray | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist PAW projection results for later assembly without abinao.

    The file is a pickled dict carrying the raw projection outputs plus
    structural metadata, so :func:`gen_exchange_abinit_paw` can skip the
    in-process WFK projection step.
    """
    payload: dict[str, Any] = {
        "cprj_per_kpt": cprj_per_kpt,
        "eigenvalues": np.asarray(eigenvalues),
        "kweights": np.asarray(kweights),
        "kpoints": np.asarray(kpoints),
        "efermi": float(efermi),
        "natom": int(natom),
        "nproj_per_atom": int(nproj_per_atom),
    }
    if cell is not None:
        payload["cell"] = np.asarray(cell)
    if positions is not None:
        payload["positions"] = np.asarray(positions)
    if atomic_numbers is not None:
        payload["atomic_numbers"] = np.asarray(atomic_numbers)
    if occupations is not None:
        payload["occupations"] = np.asarray(occupations)
    if extra:
        payload["extra"] = dict(extra)
    path = Path(path)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_projected_data(path: str | Path) -> dict:
    """Load a pickle produced by :func:`save_projected_data`."""
    path = Path(path)
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    missing = [k for k in _PROJECTED_DATA_KEYS if k not in payload]
    if missing:
        raise ValueError(f"projected-data file {path} is missing keys: {missing}")
    return payload


# ---------------------------------------------------------------------------
# Story 005 — CLI / high-level entry point
# ---------------------------------------------------------------------------


def _read_delta_from_log(log_path: str | Path) -> tuple[dict[int, np.ndarray], str]:
    """Parse spin-resolved Dij from *log_path* and return (delta, unit).

    Delegates to ``abinao.pawprt_parser``.  If abinao is not importable a clear
    error is raised.
    """
    try:
        from abinao.pawprt_parser import (
            compute_delta_ij,
            detect_dij_unit,
            parse_pawprt_dij,
        )
    except ImportError as exc:
        raise ImportError(
            "abinao.pawprt_parser is required to read the pawprt Dij block. "
            "Install abinao or pre-compute delta_ij."
        ) from exc

    dij_by_spin = parse_pawprt_dij(log_path)
    if not dij_by_spin:
        raise ValueError(f"no pawprt Dij block found in {log_path}")
    delta = compute_delta_ij(dij_by_spin)
    unit = detect_dij_unit(log_path)
    return delta, unit


def _project_wfk_in_process(
    wfk_path: str | Path,
    paw_xml_path: str | Path,
) -> dict:
    """Project a WFK file with PAW projectors using abinao (Story 003).

    Returns a dict with keys matching :func:`save_projected_data`.
    """
    try:
        from abinao.wfk import read_wfk
    except ImportError as exc:
        raise ImportError(
            "abinao is required for in-process WFK projection. "
            "Either install abinao or pass projected_data_path."
        ) from exc

    wfk = read_wfk(wfk_path)

    # Load the PAW pseudo from XML via pypao.
    paw_pseudo = _load_paw_pseudo(paw_xml_path)

    try:
        from abinao.paw_projection import project_wfk_paw
    except ImportError as exc:
        raise ImportError(
            "abinao.paw_projection.project_wfk_paw is not available "
            "(Story 003). Use projected_data_path instead."
        ) from exc

    result = project_wfk_paw(wfk, paw_pseudo)

    # Reshape eigenvalues[ik][ispin] → [nsppol, nkpt, nband].
    nsppol = len(result.eigenvalues[0])
    nkpt = len(result.eigenvalues)
    eigenvalues = np.array(
        [
            [np.asarray(result.eigenvalues[ik][ispin]) for ik in range(nkpt)]
            for ispin in range(nsppol)
        ]
    )  # [nsppol, nkpt, nband]

    kpoints = np.array([kp.kred for kp in wfk.kpoints], dtype=float)
    kweights = np.asarray(result.kweights, dtype=float)

    # Structural metadata from the WFK.
    cell = np.asarray(wfk.rprimd, dtype=float)
    positions = np.asarray(wfk.xred, dtype=float) @ cell
    atomic_numbers = _atomic_numbers_from_symbols(wfk.atom_species)

    return {
        "cprj_per_kpt": result.cprj,
        "eigenvalues": eigenvalues,
        "kweights": kweights,
        "kpoints": kpoints,
        "efermi": float(result.efermi) if result.efermi is not None else 0.0,
        "natom": int(result.natom),
        "nproj_per_atom": int(result.nproj_per_atom),
        "cell": cell,
        "positions": positions,
        "atomic_numbers": atomic_numbers,
    }


def _load_paw_pseudo(paw_xml_path: str | Path):
    """Load a PAW-XML pseudopotential via pypao.

    Uses :meth:`pypao.libpsp.PawXmlPseudo.from_file`, which returns a
    :class:`~pypao.libpsp.PawPseudo` subclass accepted by abinao's
    ``project_wfk_paw``.
    """
    try:
        from pypao.libpsp import PawXmlPseudo
    except ImportError as exc:
        raise ImportError(
            "pypao is required to read PAW-XML pseudopotentials."
        ) from exc
    return PawXmlPseudo.from_file(str(paw_xml_path))


def _atomic_numbers_from_symbols(symbols: list[str]) -> np.ndarray:
    """Convert element symbols to atomic numbers via ASE data tables."""
    from ase.data import atomic_numbers

    return np.array([atomic_numbers[sym] for sym in symbols], dtype=int)


def gen_exchange_abinit_paw(
    wfk_path: str | None = None,
    paw_xml_path: str | None = None,
    log_path: str | None = None,
    projected_data_path: str | None = None,
    *,
    natom: int | None = None,
    nproj_per_atom: int | None = None,
    delta_ij: dict[int, np.ndarray] | None = None,
    delta_unit: str | None = None,
    magnetic_elements: list[str] | None = None,
    index_magnetic_atoms: list[int] | None = None,
    output_path: str = "TB2J_results_abinit_paw",
    nz: int = 30,
    smearing_eV: float = 0.05,
    Rcut: float | None = None,
    Rmax: int | None = None,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
    efermi: float | None = None,
    description: str | None = None,
    population_mode: str = "none",
    **kwargs: Any,
) -> tuple[Path, dict]:
    """Run PAW exchange calculation from ABINIT outputs.

    Two data-source modes:

    * **In-process projection** — pass ``wfk_path``, ``paw_xml_path``, and
      ``log_path``.  Requires abinao (Story 003) to project the WFK.

    * **Pre-projected data** — pass ``projected_data_path`` (a pickle written by
      :func:`save_projected_data`) and ``log_path``.  Avoids the abinao
      dependency at runtime.

    ``delta_ij`` may be passed directly to bypass log parsing entirely.

    Parameters
    ----------
    log_path:
        Path to the ABINIT log/``.abo`` file containing the ``pawprt`` Dij
        block.  Required unless ``delta_ij`` is given directly.
    delta_ij:
        Pre-computed ``{atom: Delta_matrix}``.  Overrides ``log_path``.
    delta_unit:
        Force the delta energy unit (``"eV"`` or ``"hartree"``).  When
        ``None`` (default) the unit is auto-detected from the log, or assumed
        ``"eV"`` when ``delta_ij`` is passed directly.

    Returns
    -------
    (Path, dict)
        Path to ``exchange.out`` and the exchange ``J`` dictionary.
    """
    # -- Step 1-4: obtain projection data ------------------------------------
    if projected_data_path is not None:
        proj = load_projected_data(projected_data_path)
    elif wfk_path is not None:
        if paw_xml_path is None:
            raise ValueError("paw_xml_path is required when wfk_path is given")
        proj = _project_wfk_in_process(wfk_path, paw_xml_path)
    else:
        raise ValueError(
            "Provide either projected_data_path or wfk_path (+paw_xml_path)."
        )

    # Structural metadata: CLI args override projected-data payload.
    proj_natom = int(natom if natom is not None else proj["natom"])
    proj_nproj = int(
        nproj_per_atom if nproj_per_atom is not None else proj["nproj_per_atom"]
    )
    proj_cell = cell if cell is not None else proj.get("cell")
    proj_positions = positions if positions is not None else proj.get("positions")
    proj_atomic_numbers = (
        atomic_numbers if atomic_numbers is not None else proj.get("atomic_numbers")
    )
    proj_efermi = float(efermi) if efermi is not None else float(proj["efermi"])

    # -- Step 3: obtain delta_ij ---------------------------------------------
    if delta_ij is not None:
        resolved_delta = delta_ij
        resolved_unit = delta_unit or "eV"
    elif log_path is not None:
        resolved_delta, detected_unit = _read_delta_from_log(log_path)
        resolved_unit = delta_unit or detected_unit
    else:
        raise ValueError(
            "Either log_path or delta_ij must be provided to obtain Delta_ij."
        )

    # -- Step 5: assemble ----------------------------------------------------
    data = assemble_paw_exchange_data(
        cprj_per_kpt=proj["cprj_per_kpt"],
        delta_ij=resolved_delta,
        eigenvalues=proj["eigenvalues"],
        kweights=proj["kweights"],
        kpoints=proj["kpoints"],
        efermi=proj_efermi,
        natom=proj_natom,
        nproj_per_atom=proj_nproj,
        delta_unit=resolved_unit,
        occupations=proj.get("occupations"),
        cell=proj_cell,
        positions=proj_positions,
        atomic_numbers=proj_atomic_numbers,
        metadata=proj.get("extra"),
    )

    # -- Step 6-7: run exchange trace and write output -----------------------
    from TB2J.interfaces.gpaw_projector import (
        _magnetic_sites,
        _R_grid_for_cutoff,
        write_projector_exchange_out,
    )

    sites = _magnetic_sites(
        data,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
    )
    rpts = _R_grid_for_cutoff(
        data,
        sites,
        Rcut=Rcut,
        Rmax=Rmax,
    )

    if description is None:
        description = (
            "PAW projector Green workflow using ABINIT WFK projections "
            "(dual-projector coefficients, P2) and pawprt Dij spin difference "
            "as the complete on-site Delta_xc (P3). Values are from the "
            "controlled projector exchange-like trace.\n"
        )

    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        population_mode=population_mode,
        Rcut=Rcut,
    )


# ---------------------------------------------------------------------------
# Argparse CLI wrapper
# ---------------------------------------------------------------------------


def run_gen_exchange_abinit_paw() -> None:
    """Command-line entry point for ``abinit_paw2J.py``."""
    import argparse

    from TB2J.versioninfo import print_license

    print_license()
    parser = argparse.ArgumentParser(
        description=("Compute exchange from ABINIT PAW WFK projections + pawprt Dij."),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wfk", help="ABINIT WFK NetCDF file")
    src.add_argument(
        "--projected_data",
        help="pickle of pre-projected PAW data (from abinao or save_projected_data)",
    )
    parser.add_argument("--paw_xml", help="PAW-XML pseudopotential file")
    parser.add_argument("--log", required=True, help="ABINIT log/.abo with pawprt Dij")
    parser.add_argument(
        "--output_path", default="TB2J_results_abinit_paw", help="output directory"
    )
    parser.add_argument(
        "--magnetic_elements", nargs="*", default=None, help="magnetic element symbols"
    )
    parser.add_argument(
        "--index_magnetic_atoms",
        nargs="*",
        type=int,
        default=None,
        help="1-based atom indices to treat as magnetic",
    )
    parser.add_argument(
        "--nz", type=int, default=30, help="number of continued-fraction poles"
    )
    parser.add_argument(
        "--smearing_eV", type=float, default=0.05, help="smearing temperature in eV"
    )
    parser.add_argument(
        "--Rcut", type=float, default=None, help="real-space cutoff (Å)"
    )
    parser.add_argument("--Rmax", type=int, default=None, help="max R-grid shell")
    parser.add_argument(
        "--delta_unit",
        default=None,
        choices=[None, "eV", "hartree"],
        help="override delta energy unit (auto-detected from log by default)",
    )
    args = parser.parse_args()

    indices = None
    if args.index_magnetic_atoms is not None:
        indices = [i - 1 for i in args.index_magnetic_atoms]

    exchange_out, _ = gen_exchange_abinit_paw(
        wfk_path=args.wfk,
        paw_xml_path=args.paw_xml,
        log_path=args.log,
        projected_data_path=args.projected_data,
        magnetic_elements=args.magnetic_elements,
        index_magnetic_atoms=indices,
        output_path=args.output_path,
        nz=args.nz,
        smearing_eV=args.smearing_eV,
        Rcut=args.Rcut,
        Rmax=args.Rmax,
        delta_unit=args.delta_unit,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_gen_exchange_abinit_paw()
