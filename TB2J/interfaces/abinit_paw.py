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

import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from TB2J.paw_projector import (
    PawOperatorComponent,
    PawOperatorComponents,
    PawProjectorChannel,
    PawProjectorSnapshot,
    PawSiteLayout,
    build_projector_green_data,
)
from TB2J.projector_green import ProjectorGreenData

# Atomic length conversion. ABINIT WFK ``primitive_vectors`` are in Bohr;
# TB2J writes structural metadata and applies ``Rcut`` in Å.
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988

__all__ = [
    "assemble_paw_exchange_data",
    "gen_exchange_abinit_paw",
    "run_gen_exchange_abinit_paw",
    "save_projected_data",
    "load_projected_data",
]


def normalize_paw_xml_mapping(
    atom_species: Sequence[str],
    paw_xml_path: str | Path | Mapping[str, str | Path],
) -> dict[str, Path]:
    """Resolve and validate the physical species-to-PAW-XML input mapping."""
    species = tuple(str(symbol) for symbol in atom_species)
    unique_species = tuple(dict.fromkeys(species))
    if isinstance(paw_xml_path, Mapping):
        mapping = {str(symbol): Path(path) for symbol, path in paw_xml_path.items()}
        extras = set(mapping) - set(unique_species)
        if extras:
            raise ValueError("unknown species mapping for " + ", ".join(sorted(extras)))
        for site, symbol in enumerate(species):
            if symbol not in mapping:
                raise ValueError(
                    f"missing PAW XML for source site {site} species {symbol!r}; "
                    "supply --paw_xml SPECIES=PATH"
                )
        return mapping
    if len(unique_species) != 1:
        raise ValueError(
            "single PAW XML is valid only for a single-species WFK; "
            "supply --paw_xml SPECIES=PATH for every species"
        )
    return {unique_species[0]: Path(paw_xml_path)}


def build_abinit_paw_site_layout(
    atom_species: Sequence[str],
    paw_xml_by_species: Mapping[str, str | Path],
    paw_by_species: Mapping[str, Any],
    *,
    site_slices: Sequence[slice] | None = None,
) -> tuple[PawSiteLayout, ...]:
    """Build and validate every physical ABINIT PAW site block."""
    resolved_paths = normalize_paw_xml_mapping(atom_species, paw_xml_by_species)
    inferred_widths = []
    expanded_channels = {}
    for symbol, pseudo in paw_by_species.items():
        channels = []
        for radial, (_n, l) in enumerate(pseudo.channel_labels):
            channels.extend(
                PawProjectorChannel(
                    l=int(l), m=m, radial=radial, label=f"n{_n}l{l}m{m}"
                )
                for m in range(-int(l), int(l) + 1)
            )
        expanded_channels[str(symbol)] = tuple(channels)
    for site, symbol in enumerate(atom_species):
        if symbol not in expanded_channels:
            raise ValueError(
                f"missing loaded PAW XML for source site {site} species {symbol!r}"
            )
        inferred_widths.append(len(expanded_channels[symbol]))
    if site_slices is None:
        starts = np.cumsum([0, *inferred_widths[:-1]])
        site_slices = tuple(
            slice(int(start), int(start + width))
            for start, width in zip(starts, inferred_widths, strict=True)
        )
    if len(site_slices) != len(atom_species):
        raise ValueError("site_slices must contain every source site")
    layout = []
    for site, (symbol, projector_slice, expected_width) in enumerate(
        zip(atom_species, site_slices, inferred_widths, strict=True)
    ):
        observed_width = projector_slice.stop - projector_slice.start
        if observed_width != expected_width:
            raise ValueError(
                f"source site {site} species {symbol!r}: expected {expected_width} "
                f"channels from {resolved_paths[symbol]}, observed shape "
                f"({observed_width}, {observed_width}); correct the PAW XML mapping or Dij block"
            )
        path = resolved_paths[str(symbol)]
        setup_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        layout.append(
            PawSiteLayout(
                source_site=site,
                species=str(symbol),
                atomic_number=int(_atomic_numbers_from_symbols([str(symbol)])[0]),
                projector_slice=projector_slice,
                channels=expanded_channels[str(symbol)],
                setup_hash=setup_hash,
            )
        )
    return tuple(layout)


def build_abinit_paw_snapshot(
    cprj_per_kpt: list,
    delta_ij: Mapping[int, np.ndarray],
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    *,
    site_layout: Sequence[PawSiteLayout],
    cell: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    delta_unit: str,
    provenance: Mapping[str, object],
    selected_source_sites: Sequence[int] | None = None,
) -> PawProjectorSnapshot:
    """Construct the shared immutable snapshot from ABINIT PAW source data."""
    layout = tuple(site_layout)
    nproj_total = layout[-1].projector_slice.stop
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError("eigenvalues must have shape (nspin, nkpt, nband)")
    nspin, nkpt, nband = eigenvalues.shape
    if len(cprj_per_kpt) != nkpt:
        raise ValueError("cprj_per_kpt must contain every k-point")
    coefficients = np.empty((nspin, nkpt, nband, nproj_total), dtype=complex)
    for ik, by_spin in enumerate(cprj_per_kpt):
        if len(by_spin) != nspin:
            raise ValueError(f"k-point {ik} has incompatible spin count")
        for spin, values in enumerate(by_spin):
            values = np.asarray(values, dtype=complex)
            if values.shape != (nproj_total, nband):
                raise ValueError(
                    f"k-point {ik} spin {spin}: coefficient shape {values.shape} "
                    f"!= ({nproj_total}, {nband})"
                )
            coefficients[spin, ik] = values.T
    nmax = max(
        site.projector_slice.stop - site.projector_slice.start for site in layout
    )
    blocks = np.zeros((len(layout), nmax, nmax), dtype=complex)
    for site, physical_site in enumerate(layout):
        width = physical_site.projector_slice.stop - physical_site.projector_slice.start
        try:
            dij = np.asarray(delta_ij[site], dtype=complex)
        except KeyError as exc:
            raise ValueError(
                f"missing Dij for source site {site} species {physical_site.species!r}"
            ) from exc
        if dij.shape != (width, width):
            raise ValueError(
                f"source site {site} species {physical_site.species!r}: expected "
                f"{width} channels, observed shape {dij.shape}; correct the PAW XML "
                "mapping or Dij block"
            )
        blocks[site, :width, :width] = dij
    if not delta_unit.strip().lower().startswith(("ha", "hartree")):
        raise ValueError("ABINIT PAW snapshot requires Hartree Dij input")
    metadata = dict(provenance)
    metadata.setdefault("source_code", "abinit")
    metadata.setdefault("source_version", "unknown")
    metadata.setdefault("functional", "unknown")
    metadata["setup_hashes"] = [site.setup_hash for site in layout]
    metadata.setdefault("u_eV", 0.0)
    metadata.setdefault("j_eV", 0.0)
    metadata.setdefault("correlated_shells", [])
    return PawProjectorSnapshot(
        kpoints=kpoints,
        weights=kweights,
        eigenvalues=eigenvalues,
        occupations=None,
        coefficients=coefficients,
        efermi=efermi,
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        site_layout=layout,
        operators=PawOperatorComponents(
            components=(
                PawOperatorComponent(
                    name="total",
                    values=blocks,
                    units="Hartree",
                    basis_id="native_paw_projector_hamiltonian",
                    definition="ABINIT pawprt total Dij spin difference",
                    source="ABINIT pawprt Dij",
                ),
            ),
            policy="authoritative_total",
            selected_names=("total",),
        ),
        kpoint_mode="full_bz",
        selected_source_sites=tuple(
            range(len(layout))
            if selected_source_sites is None
            else selected_source_sites
        ),
        provenance=metadata,
    )


# ---------------------------------------------------------------------------
# Story 004 — Assembly
# ---------------------------------------------------------------------------


def _resolve_site_slices(
    natom: int,
    nproj_per_atom: int | None,
    site_slices: Sequence[slice] | None,
) -> tuple[tuple[slice, ...], int]:
    """Validate contiguous global projector slices for every atomic site."""
    if natom < 1:
        raise ValueError("natom must be positive")
    if site_slices is None:
        if nproj_per_atom is None or nproj_per_atom < 1:
            raise ValueError(
                "nproj_per_atom must be positive when site_slices is not provided"
            )
        return (
            tuple(
                slice(atom * nproj_per_atom, (atom + 1) * nproj_per_atom)
                for atom in range(natom)
            ),
            natom * nproj_per_atom,
        )

    if len(site_slices) != natom:
        raise ValueError(f"site_slices has {len(site_slices)} sites, expected {natom}")
    resolved: list[slice] = []
    cursor = 0
    for atom, site_slice in enumerate(site_slices):
        if not isinstance(site_slice, slice):
            raise TypeError(f"site_slices[{atom}] must be a slice")
        if (
            site_slice.start is None
            or site_slice.stop is None
            or site_slice.step not in (None, 1)
            or site_slice.start != cursor
            or site_slice.stop <= cursor
        ):
            raise ValueError(
                "site_slices must be non-empty, contiguous, forward slices "
                "covering the flattened projector axis"
            )
        resolved.append(slice(site_slice.start, site_slice.stop))
        cursor = site_slice.stop
    if nproj_per_atom is not None and any(
        projector_slice.stop - projector_slice.start != nproj_per_atom
        for projector_slice in resolved
    ):
        raise ValueError("nproj_per_atom conflicts with unequal site_slices")
    return tuple(resolved), cursor


def assemble_paw_exchange_data(
    cprj_per_kpt: list,
    delta_ij: dict[int, np.ndarray],
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    natom: int,
    nproj_per_atom: int | None = None,
    *,
    site_slices: Sequence[slice] | None = None,
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
    site_layout: Sequence[PawSiteLayout] | None = None,
) -> ProjectorGreenData:
    """Assemble ABINIT PAW projections into exchange-ready spectral data.

    ``cprj_per_kpt[ik][ispin]`` is normally a flat
    ``(nproj_total, nband)`` array and ``site_slices`` identifies each site's
    contiguous portion of that global axis.  This permits distinct PAW datasets
    with unequal channel counts without fake projector padding.  The original
    uniform ``(natom, nproj_per_atom, nband)`` representation remains accepted
    when ``site_slices`` is omitted.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError("eigenvalues must have shape (nsppol, nkpt, nband)")
    nsppol, nkpt, nband = eigenvalues.shape
    if site_layout is not None:
        if len(site_layout) != natom:
            raise ValueError("site_layout must contain every source site")
        layout_slices = tuple(site.projector_slice for site in site_layout)
        if site_slices is not None and tuple(site_slices) != layout_slices:
            raise ValueError("site_slices must match the physical PAW site_layout")
        site_slices = layout_slices
    site_slices, nproj_total = _resolve_site_slices(natom, nproj_per_atom, site_slices)

    kpoints = np.asarray(kpoints, dtype=float)
    kweights = np.asarray(kweights, dtype=float)
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

    coefficients = np.empty((nsppol, nkpt, nband, nproj_total), dtype=complex)
    for ik, kpt_cprj in enumerate(cprj_per_kpt):
        if len(kpt_cprj) != nsppol:
            raise ValueError(
                f"k-point {ik} has {len(kpt_cprj)} spins, expected {nsppol}"
            )
        for ispin, cprj_spin in enumerate(kpt_cprj):
            cprj = np.asarray(cprj_spin, dtype=complex)
            if cprj.ndim == 3:
                expected_shape = (natom, nproj_per_atom, nband)
                if site_slices is not None and nproj_per_atom is None:
                    raise ValueError(
                        "ragged site_slices require flat cprj arrays; "
                        "3-D cprj is only valid for uniform projector counts"
                    )
                if cprj.shape != expected_shape:
                    raise ValueError(
                        f"k-point {ik} spin {ispin}: cprj shape {cprj.shape} "
                        f"!= {expected_shape}"
                    )
                cprj = cprj.reshape(nproj_total, nband)
            elif cprj.ndim != 2 or cprj.shape != (nproj_total, nband):
                raise ValueError(
                    f"k-point {ik} spin {ispin}: cprj shape {cprj.shape} "
                    f"!= ({nproj_total}, {nband})"
                )
            coefficients[ispin, ik] = cprj.T

    site_nproj = np.array(
        [
            projector_slice.stop - projector_slice.start
            for projector_slice in site_slices
        ],
        dtype=int,
    )
    nproj_site_max = int(site_nproj.max())
    site_projector_indices = -np.ones((natom, nproj_site_max), dtype=int)
    projector_site = np.empty(nproj_total, dtype=int)
    for atom, projector_slice in enumerate(site_slices):
        indices = np.arange(projector_slice.start, projector_slice.stop)
        site_projector_indices[atom, : len(indices)] = indices
        projector_site[indices] = atom
    projector_atom = projector_site.copy()

    # The new flattened path keeps the full delta on the global projector axis,
    # so each unequal local block lands exactly in its own contiguous slice.
    # Keep the old padded-site component for the legacy uniform representation.
    global_delta = np.zeros((nproj_total, nproj_total), dtype=complex)
    local_delta = np.zeros((natom, nproj_site_max, nproj_site_max), dtype=complex)
    for atom_key, delta in delta_ij.items():
        atom = int(atom_key)
        if atom < 0 or atom >= natom:
            raise ValueError(f"delta_ij atom index {atom} out of range [0, {natom})")
        delta = np.asarray(delta, dtype=complex)
        projector_slice = site_slices[atom]
        nsite_proj = site_nproj[atom]
        if delta.shape != (nsite_proj, nsite_proj):
            context = ""
            if site_layout is not None:
                physical_site = site_layout[atom]
                context = (
                    f" for source site {atom} species {physical_site.species!r}; "
                    f"expected {nsite_proj} channels, observed shape {delta.shape}; "
                    "correct the PAW XML mapping or Dij block"
                )
            raise ValueError(
                f"delta_ij[{atom}] shape {delta.shape} != "
                f"({nsite_proj}, {nsite_proj}){context}"
            )
        global_delta[projector_slice, projector_slice] = delta
        local_delta[atom, :nsite_proj, :nsite_proj] = delta

    if delta_unit.strip().lower().startswith("ha"):
        global_delta *= HARTREE_TO_EV
        local_delta *= HARTREE_TO_EV
    delta_xc = global_delta if nproj_per_atom is None else local_delta
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
            "site_projector_slices": [
                (projector_slice.start, projector_slice.stop)
                for projector_slice in site_slices
            ],
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
    nproj_per_atom: int | None = None,
    site_slices: Sequence[slice] | None = None,
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
    if nproj_per_atom is None and site_slices is None:
        raise ValueError(
            "provide nproj_per_atom or site_slices for the projector layout"
        )

    payload: dict[str, Any] = {
        "cprj_per_kpt": cprj_per_kpt,
        "eigenvalues": np.asarray(eigenvalues),
        "kweights": np.asarray(kweights),
        "kpoints": np.asarray(kpoints),
        "efermi": float(efermi),
        "natom": int(natom),
    }
    if nproj_per_atom is not None:
        payload["nproj_per_atom"] = int(nproj_per_atom)
    if site_slices is not None:
        payload["site_slices"] = tuple(
            slice(projector_slice.start, projector_slice.stop)
            for projector_slice in site_slices
        )
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
    if payload.get("nproj_per_atom") is None and payload.get("site_slices") is None:
        raise ValueError(
            f"projected-data file {path} is missing projector layout metadata"
        )
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
    paw_xml_path: str | Path | Mapping[str, str | Path],
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

    xml_by_species = normalize_paw_xml_mapping(wfk.atom_species, paw_xml_path)

    # Load the PAW pseudo from XML via pypao.
    paw_pseudo = _load_paw_pseudo(xml_by_species)

    try:
        from abinao.paw_projection import project_wfk_paw
    except ImportError as exc:
        raise ImportError(
            "abinao.paw_projection.project_wfk_paw is not available "
            "(Story 003). Use projected_data_path instead."
        ) from exc
    result = project_wfk_paw(wfk, paw_pseudo)

    site_layout = build_abinit_paw_site_layout(
        wfk.atom_species,
        xml_by_species,
        paw_pseudo,
        site_slices=getattr(result, "site_slices", None),
    )

    # Reshape eigenvalues[ik][ispin] → [nsppol, nkpt, nband].
    nsppol = len(result.eigenvalues[0])
    nkpt = len(result.eigenvalues)
    eigenvalues = np.array(
        [
            [np.asarray(result.eigenvalues[ik][ispin]) for ik in range(nkpt)]
            for ispin in range(nsppol)
        ]
    )  # [nsppol, nkpt, nband]

    kpoints = np.asarray(result.kpoints, dtype=float)
    kweights = np.asarray(result.kweights, dtype=float)

    # ABINIT stores primitive vectors in Bohr; TB2J's public structural
    # metadata and real-space cutoffs use Å.
    cell = np.asarray(wfk.rprimd, dtype=float) * BOHR_TO_ANGSTROM
    positions = np.asarray(wfk.xred, dtype=float) @ cell
    atomic_numbers = _atomic_numbers_from_symbols(wfk.atom_species)

    payload = {
        "cprj_per_kpt": result.cprj,
        "eigenvalues": eigenvalues,
        "kweights": kweights,
        "kpoints": kpoints,
        "efermi": float(result.efermi) if result.efermi is not None else 0.0,
        "natom": int(result.natom),
        "cell": cell,
        "positions": positions,
        "atomic_numbers": atomic_numbers,
    }
    payload["site_layout"] = site_layout
    payload["atom_species"] = tuple(wfk.atom_species)
    site_slices = getattr(result, "site_slices", None)
    if site_slices is not None:
        payload["site_slices"] = tuple(site_slices)
    nproj_per_atom = getattr(result, "nproj_per_atom", None)
    if nproj_per_atom is not None:
        payload["nproj_per_atom"] = int(nproj_per_atom)
    if "site_slices" not in payload and "nproj_per_atom" not in payload:
        raise ValueError(
            "abinao PAW projection must provide site_slices or nproj_per_atom"
        )
    return payload


def _load_paw_pseudo(
    paw_xml_path: str | Path | Mapping[str, str | Path],
):
    """Load one PAW-XML pseudo or a species-to-PAW-XML mapping via pypao."""
    try:
        from pypao.libpsp import PawXmlPseudo
    except ImportError as exc:
        raise ImportError(
            "pypao is required to read PAW-XML pseudopotentials."
        ) from exc
    if isinstance(paw_xml_path, Mapping):
        return {
            str(species): PawXmlPseudo.from_file(str(path))
            for species, path in paw_xml_path.items()
        }
    return PawXmlPseudo.from_file(str(paw_xml_path))


def _atomic_numbers_from_symbols(symbols: list[str]) -> np.ndarray:
    """Convert element symbols to atomic numbers via ASE data tables."""
    from ase.data import atomic_numbers

    return np.array([atomic_numbers[sym] for sym in symbols], dtype=int)


def gen_exchange_abinit_paw(
    wfk_path: str | None = None,
    paw_xml_path: str | Mapping[str, str | Path] | None = None,
    log_path: str | None = None,
    projected_data_path: str | None = None,
    *,
    natom: int | None = None,
    nproj_per_atom: int | None = None,
    site_slices: Sequence[slice] | None = None,
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
    snapshot_cache: str | None = None,
    write_snapshot_cache: str | None = None,
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
    # -- Cache read: load a previously validated snapshot and skip projection -
    snapshot = None
    if snapshot_cache is not None:
        import json

        from TB2J.paw_snapshot_cache import read_paw_snapshot_netcdf

        identity_path = Path(snapshot_cache).with_suffix(".identity.json")
        if not identity_path.exists():
            raise ValueError(
                f"expected identity sidecar {identity_path} beside snapshot cache"
            )
        expected_identity = json.loads(identity_path.read_text())
        snapshot = read_paw_snapshot_netcdf(
            snapshot_cache, expected_identity=expected_identity
        )

    if snapshot is not None:
        data = build_projector_green_data(snapshot)
    else:
        # -- Step 1-4: obtain projection data --------------------------------
        if projected_data_path is not None:
            proj = load_projected_data(projected_data_path)
        elif wfk_path is not None:
            if paw_xml_path is None:
                raise ValueError("paw_xml_path is required when wfk_path is given")
            proj = _project_wfk_in_process(wfk_path, paw_xml_path)
        else:
            raise ValueError(
                "Provide either snapshot_cache, projected_data_path, "
                "or wfk_path (+paw_xml_path)."
            )

        # Structural metadata
        proj_efermi = float(efermi) if efermi is not None else float(proj["efermi"])
        proj_cell = cell if cell is not None else proj.get("cell")
        proj_positions = positions if positions is not None else proj.get("positions")
        proj_atomic_numbers = (
            atomic_numbers if atomic_numbers is not None else proj.get("atomic_numbers")
        )

        # -- Step 3: obtain delta_ij ----------------------------------------
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

        # -- Step 5: validate/build the source-neutral PAW seam -------------
        if proj.get(
            "site_layout"
        ) is not None and resolved_unit.strip().lower().startswith(("ha", "hartree")):
            snapshot = build_abinit_paw_snapshot(
                cprj_per_kpt=proj["cprj_per_kpt"],
                delta_ij=resolved_delta,
                eigenvalues=proj["eigenvalues"],
                kweights=proj["kweights"],
                kpoints=proj["kpoints"],
                efermi=proj_efermi,
                site_layout=proj["site_layout"],
                cell=proj_cell,
                positions=proj_positions,
                atomic_numbers=proj_atomic_numbers,
                delta_unit=resolved_unit,
                provenance=proj.get("extra", {}),
            )
            data = build_projector_green_data(snapshot)
        else:
            data = assemble_paw_exchange_data(
                cprj_per_kpt=proj["cprj_per_kpt"],
                delta_ij=resolved_delta,
                eigenvalues=proj["eigenvalues"],
                kweights=proj["kweights"],
                kpoints=proj["kpoints"],
                efermi=proj_efermi,
                natom=int(natom if natom is not None else proj["natom"]),
                nproj_per_atom=(
                    nproj_per_atom
                    if nproj_per_atom is not None
                    else proj.get("nproj_per_atom")
                ),
                delta_unit=resolved_unit,
                site_slices=(
                    site_slices if site_slices is not None else proj.get("site_slices")
                ),
                occupations=proj.get("occupations"),
                cell=proj_cell,
                positions=proj_positions,
                atomic_numbers=proj_atomic_numbers,
                metadata=proj.get("extra"),
                site_layout=proj.get("site_layout"),
            )

    # -- Optional: write validated snapshot cache ---------------------------
    if write_snapshot_cache is not None and snapshot is not None:
        import json

        from TB2J.paw_snapshot_cache import write_paw_snapshot_netcdf

        identity = write_paw_snapshot_netcdf(write_snapshot_cache, snapshot)
        identity_path = Path(write_snapshot_cache).with_suffix(".identity.json")
        identity_path.write_text(json.dumps(identity, sort_keys=True, indent=2))

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
    src.add_argument(
        "--snapshot_cache",
        help="load a previously validated PAW snapshot NetCDF cache",
    )
    parser.add_argument(
        "--paw_xml",
        action="append",
        default=None,
        help="PAW XML path for a single-species WFK, or repeat SPECIES=PATH",
    )
    parser.add_argument(
        "--write_snapshot_cache",
        default=None,
        help="write a validated PAW snapshot NetCDF cache before exchange",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="ABINIT log/.abo with pawprt Dij (not needed with --snapshot_cache)",
    )
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

    paw_xml_path = None
    if args.paw_xml:
        if len(args.paw_xml) == 1 and "=" not in args.paw_xml[0]:
            paw_xml_path = args.paw_xml[0]
        else:
            paw_xml_path = {}
            for entry in args.paw_xml:
                if "=" not in entry:
                    parser.error("--paw_xml mappings must use SPECIES=PATH")
                species, path = entry.split("=", 1)
                if not species or not path or species in paw_xml_path:
                    parser.error(
                        "--paw_xml mappings require unique SPECIES=PATH entries"
                    )
                paw_xml_path[species] = path

    exchange_out, _ = gen_exchange_abinit_paw(
        wfk_path=args.wfk,
        paw_xml_path=paw_xml_path,
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
        snapshot_cache=args.snapshot_cache,
        write_snapshot_cache=args.write_snapshot_cache,
    )
    print(f"Wrote {exchange_out}")


if __name__ == "__main__":
    run_gen_exchange_abinit_paw()
