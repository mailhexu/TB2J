"""Validated source-neutral PAW projector state before Green construction.

This module keeps producer validation separate from :mod:`TB2J.projector_green`.
Its input operators are Hartree-valued PAW spin differences; the builder converts
those values once to the eV convention of :class:`ProjectorGreenData`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from TB2J.projector_green import ProjectorGreenData

HARTREE_TO_EV = 27.211386245988

_OPERATOR_NAMES = frozenset({"xc", "hubbard", "soc", "total"})
_REQUIRED_PROVENANCE = (
    "source_code",
    "source_version",
    "functional",
    "setup_hashes",
    "u_eV",
    "j_eV",
    "correlated_shells",
)


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Return an owned, read-only array so snapshots cannot be mutated in place."""
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _freeze_value(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return _frozen_array(value)
    return value


def _thaw_value(value):
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PawProjectorChannel:
    """One ordered PAW projector channel in a physical site block."""

    l: int
    m: int
    radial: int
    label: str


@dataclass(frozen=True)
class PawSiteLayout:
    """Physical PAW projector layout for one source atom."""

    source_site: int
    species: str
    atomic_number: int
    projector_slice: slice
    channels: tuple[PawProjectorChannel, ...]
    setup_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))


@dataclass(frozen=True)
class PawOperatorComponent:
    """One Hartree-valued local PAW spin-difference component."""

    name: Literal["xc", "hubbard", "soc", "total"]
    values: np.ndarray
    units: str
    basis_id: str
    definition: str
    source: str
    included_in_total: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_array(self.values))


@dataclass(frozen=True)
class PawOperatorComponents:
    """The inclusion-once policy for local PAW operator components."""

    components: tuple[PawOperatorComponent, ...]
    policy: Literal["authoritative_total", "compose"]
    selected_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "selected_names", tuple(self.selected_names))


@dataclass(frozen=True)
class PawProjectorSnapshot:
    """Producer-neutral PAW state that must validate before Green construction."""

    kpoints: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray | None
    coefficients: np.ndarray
    efermi: float
    cell: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray
    site_layout: tuple[PawSiteLayout, ...]
    operators: PawOperatorComponents
    kpoint_mode: Literal["full_bz", "expanded_from_ibz"]
    selected_source_sites: tuple[int, ...]
    provenance: Mapping[str, object]
    overlap_metric: np.ndarray | None = None
    population_metric_matrix: np.ndarray | None = None
    hij: np.ndarray | None = None
    efermi_spin: np.ndarray | None = None
    population_metric: str | None = None
    hij_definition: str | None = None
    hij_units: str | None = None
    hij_source: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "kpoints",
            "weights",
            "eigenvalues",
            "coefficients",
            "cell",
            "positions",
            "atomic_numbers",
            "overlap_metric",
            "population_metric_matrix",
            "hij",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _frozen_array(value))
        if self.occupations is not None:
            object.__setattr__(self, "occupations", _frozen_array(self.occupations))
        object.__setattr__(self, "site_layout", tuple(self.site_layout))
        object.__setattr__(
            self, "selected_source_sites", tuple(self.selected_source_sites)
        )
        object.__setattr__(self, "provenance", _freeze_value(dict(self.provenance)))


@dataclass(frozen=True)
class PawSnapshotValidationReport:
    """Validated facts consumed by the source-neutral Green-data builder."""

    selected_component_names: tuple[str, ...]
    operator_basis: str
    site_nproj: tuple[int, ...]


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def validate_full_bz_mesh(kpoints: np.ndarray, weights: np.ndarray) -> None:
    """Reject incomplete, duplicate, or nonuniform periodic full-BZ meshes."""
    kpoints = _require_finite("kpoints", np.asarray(kpoints, dtype=float))
    weights = _require_finite("weights", np.asarray(weights, dtype=float))
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("kpoints must have shape (nkpt, 3)")
    if weights.shape != (len(kpoints),):
        raise ValueError("weights must have shape (nkpt,)")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("full-BZ weights must be non-negative and normalized to one")
    canonical = np.mod(np.round(kpoints, decimals=8), 1.0)
    if len(np.unique(canonical, axis=0)) != len(canonical):
        raise ValueError("full-BZ mesh contains duplicate periodic k-points")
    # Cartesian-product completeness with tolerance grouping (handles
    # non-trivial divisions like 1/7 that produce float near-duplicates).
    tol = 1e-6
    product = 1
    for ax in range(3):
        vals = np.sort(canonical[:, ax])
        groups = 1
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] > tol:
                groups += 1
        product *= groups
    if product != len(canonical):
        raise ValueError(
            "full-BZ mesh is incomplete; generate an explicitly unsymmetrized "
            "full-BZ WFK instead of using magnetic IBZ reconstruction"
        )

def _validate_spectral_data(
    snapshot: PawProjectorSnapshot,
) -> tuple[int, int, int, int]:
    kpoints = _require_finite("kpoints", np.asarray(snapshot.kpoints, dtype=float))
    weights = _require_finite("weights", np.asarray(snapshot.weights, dtype=float))
    eigenvalues = _require_finite(
        "eigenvalues", np.asarray(snapshot.eigenvalues, dtype=float)
    )
    coefficients = _require_finite(
        "coefficients", np.asarray(snapshot.coefficients, dtype=complex)
    )

    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("kpoints must have shape (nkpt, 3)")
    if weights.shape != (kpoints.shape[0],):
        raise ValueError("weights must have shape (nkpt,)")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("weights must be non-negative and normalized to one")
    if eigenvalues.ndim != 3:
        raise ValueError("eigenvalues must have shape (nspin, nkpt, nband)")
    nspin, nkpt, nband = eigenvalues.shape
    if nkpt != kpoints.shape[0]:
        raise ValueError("eigenvalues and kpoints have inconsistent nkpt")
    if coefficients.shape[:3] != (nspin, nkpt, nband) or coefficients.ndim != 4:
        raise ValueError("coefficients must have shape (nspin, nkpt, nband, nproj)")
    if snapshot.occupations is not None:
        occupations = _require_finite(
            "occupations", np.asarray(snapshot.occupations, dtype=float)
        )
        if occupations.shape != (nspin, nkpt, nband):
            raise ValueError("occupations must match eigenvalues shape")
    if not np.isfinite(float(snapshot.efermi)):
        raise ValueError("efermi must be finite")
    return nspin, nkpt, nband, coefficients.shape[-1]


def _validate_layout(snapshot: PawProjectorSnapshot, nproj: int) -> tuple[int, ...]:
    layout = snapshot.site_layout
    if not layout:
        raise ValueError("site_layout must contain every source site")
    source_sites = tuple(site.source_site for site in layout)
    if source_sites != tuple(range(len(layout))):
        raise ValueError("site_layout source_site values must be contiguous from zero")

    cursor = 0
    site_nproj = []
    for site in layout:
        projector_slice = site.projector_slice
        if (
            projector_slice.step not in (None, 1)
            or projector_slice.start is None
            or projector_slice.stop is None
            or projector_slice.start != cursor
            or projector_slice.stop <= cursor
        ):
            raise ValueError(
                "site_layout projector slices must be contiguous and nonempty"
            )
        nsite_proj = projector_slice.stop - projector_slice.start
        if len(site.channels) != nsite_proj:
            raise ValueError("site_layout channel count must match projector slice")
        if not site.species or not site.setup_hash:
            raise ValueError("site_layout requires species and setup_hash")
        if site.atomic_number <= 0:
            raise ValueError("site_layout atomic_number must be positive")
        cursor = projector_slice.stop
        site_nproj.append(nsite_proj)
    if cursor != nproj:
        raise ValueError("site_layout projector slices must cover every coefficient")

    positions = _require_finite(
        "positions", np.asarray(snapshot.positions, dtype=float)
    )
    atomic_numbers = np.asarray(snapshot.atomic_numbers, dtype=int)
    cell = _require_finite("cell", np.asarray(snapshot.cell, dtype=float))
    if positions.shape != (len(layout), 3):
        raise ValueError("positions must have shape (nsite, 3)")
    if atomic_numbers.shape != (len(layout),):
        raise ValueError("atomic_numbers must have shape (nsite,)")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if not np.array_equal(atomic_numbers, [site.atomic_number for site in layout]):
        raise ValueError("atomic_numbers must match site_layout")
    selected = tuple(snapshot.selected_source_sites)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_source_sites must be nonempty and unique")
    if any(site not in source_sites for site in selected):
        raise ValueError("selected_source_sites contains a site outside site_layout")
    return tuple(site_nproj)


def _validate_provenance(snapshot: PawProjectorSnapshot) -> None:
    for name in _REQUIRED_PROVENANCE:
        value = snapshot.provenance.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"provenance requires {name}")
    setup_hashes = tuple(snapshot.provenance["setup_hashes"])
    expected_hashes = tuple(site.setup_hash for site in snapshot.site_layout)
    if setup_hashes != expected_hashes:
        raise ValueError("provenance setup_hashes must match site_layout order")


def _validate_operators(
    snapshot: PawProjectorSnapshot, site_nproj: tuple[int, ...]
) -> tuple[tuple[str, ...], str]:
    operators = snapshot.operators
    if operators.policy not in {"authoritative_total", "compose"}:
        raise ValueError("operator policy must be authoritative_total or compose")
    components = {component.name: component for component in operators.components}
    if len(components) != len(operators.components):
        raise ValueError("operator component names must be unique")
    if not components or set(components) - _OPERATOR_NAMES:
        raise ValueError("operator components must use known PAW component names")

    basis_ids = {component.basis_id for component in components.values()}
    if basis_ids != {"native_paw_projector_hamiltonian"}:
        raise ValueError(
            "operator components must use native_paw_projector_hamiltonian"
        )
    nsite = len(site_nproj)
    nmax = max(site_nproj)
    for component in components.values():
        if component.units.strip().lower() not in {"hartree", "ha"}:
            raise ValueError("PAW operator components must use Hartree units")
        if not component.definition or not component.source:
            raise ValueError("operator components require definition and source")
        values = _require_finite(
            f"operator component {component.name}",
            np.asarray(component.values, dtype=complex),
        )
        if values.shape != (nsite, nmax, nmax):
            raise ValueError(
                f"operator component {component.name} must have shape "
                "(nsite, nproj_site_max, nproj_site_max)"
            )
        for site, nsite_proj in enumerate(site_nproj):
            block = values[site, :nsite_proj, :nsite_proj]
            if not np.allclose(block, block.conj().T, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"operator component {component.name} site {site} must be Hermitian"
                )

    selected = tuple(operators.selected_names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("operator selection must be nonempty and unique")
    if any(name not in components for name in selected):
        raise ValueError("operator selection names must exist in components")
    if operators.policy == "authoritative_total":
        if selected != ("total",) or "total" not in components:
            raise ValueError("authoritative_total policy must select only total")
        if any(
            not component.included_in_total
            for name, component in components.items()
            if name != "total"
        ):
            raise ValueError(
                "authoritative total requires every other component included"
            )
    else:
        if "total" in selected:
            raise ValueError("compose policy cannot select total")
        if any(components[name].included_in_total for name in selected):
            raise ValueError(
                "compose policy cannot select a component included in total"
            )
    return selected, next(iter(basis_ids))


def validate_paw_projector_snapshot(
    snapshot: PawProjectorSnapshot,
) -> PawSnapshotValidationReport:
    """Validate all source-neutral PAW state invariants before Green construction."""
    _nspin, _nkpt, _nband, nproj = _validate_spectral_data(snapshot)
    if snapshot.kpoint_mode == "full_bz":
        validate_full_bz_mesh(snapshot.kpoints, snapshot.weights)
    elif snapshot.kpoint_mode == "expanded_from_ibz":
        raise ValueError(
            "IBZ expansion lacks validated spatial, anti-unitary spin, and "
            "coefficient transformations; generate a full-BZ source calculation"
        )
    else:
        raise ValueError("kpoint_mode must be full_bz or expanded_from_ibz")
    site_nproj = _validate_layout(snapshot, nproj)
    _validate_provenance(snapshot)
    selected_names, operator_basis = _validate_operators(snapshot, site_nproj)
    return PawSnapshotValidationReport(selected_names, operator_basis, site_nproj)


def _component_data_name(name: str) -> str:
    return "delta_total" if name == "total" else f"delta_{name}"


def build_projector_green_data(snapshot: PawProjectorSnapshot) -> ProjectorGreenData:
    """Build eV-valued dual-PAW Green data from a validated snapshot."""
    report = validate_paw_projector_snapshot(snapshot)
    nsite = len(snapshot.site_layout)
    nmax = max(report.site_nproj)
    nproj = snapshot.coefficients.shape[-1]
    site_projector_indices = -np.ones((nsite, nmax), dtype=int)
    projector_site = np.empty(nproj, dtype=int)
    projector_l = np.empty(nproj, dtype=int)
    projector_m = np.empty(nproj, dtype=int)
    projector_radial = np.empty(nproj, dtype=int)
    for site, layout in enumerate(snapshot.site_layout):
        projector_slice = layout.projector_slice
        indices = np.arange(projector_slice.start, projector_slice.stop)
        site_projector_indices[site, : len(indices)] = indices
        projector_site[indices] = layout.source_site
        projector_l[indices] = [channel.l for channel in layout.channels]
        projector_m[indices] = [channel.m for channel in layout.channels]
        projector_radial[indices] = [channel.radial for channel in layout.channels]

    components_by_name = {
        component.name: component for component in snapshot.operators.components
    }
    operator_components = {}
    operator_component_metadata = {}
    for name, component in components_by_name.items():
        data_name = _component_data_name(name)
        operator_components[data_name] = (
            np.asarray(component.values, dtype=complex) * HARTREE_TO_EV
        )
        operator_component_metadata[data_name] = {
            "units": "eV",
            "input_units": "Hartree",
            "definition": component.definition,
            "source": component.source,
            "operator_basis": component.basis_id,
            "included_in_total": str(component.included_in_total).lower(),
            "completeness": "complete" if name == "total" else "component",
            "exchange_ready": "true" if name == "total" else "false",
        }

    selected_values = sum(
        (
            np.asarray(components_by_name[name].values, dtype=complex)
            for name in report.selected_component_names
        ),
        np.zeros((nsite, nmax, nmax), dtype=complex),
    )
    operator_components["delta_total"] = selected_values * HARTREE_TO_EV
    operator_component_metadata["delta_total"] = {
        "units": "eV",
        "input_units": "Hartree",
        "definition": " + ".join(
            components_by_name[name].definition
            for name in report.selected_component_names
        ),
        "source": " + ".join(
            components_by_name[name].source for name in report.selected_component_names
        ),
        "operator_basis": report.operator_basis,
        "completeness": "complete",
        "exchange_ready": "true",
        "selection_policy": snapshot.operators.policy,
        "selected_components": list(report.selected_component_names),
    }
    source_operator_metadata = snapshot.provenance.get(
        "operator_component_metadata", {}
    )
    for name, source_metadata in source_operator_metadata.items():
        if name in operator_component_metadata:
            operator_component_metadata[name].update(dict(source_metadata))

    metadata = _thaw_value(snapshot.provenance)
    metadata.update(
        {
            "kpoint_mode": snapshot.kpoint_mode,
            "projector_basis_type": "paw",
            "coefficient_convention": "dual_projector_no_inverse",
            "paw_operator_policy": snapshot.operators.policy,
            "selected_operator_components": list(report.selected_component_names),
            "selected_source_sites": list(snapshot.selected_source_sites),
            "site_projector_slices": [
                (layout.projector_slice.start, layout.projector_slice.stop)
                for layout in snapshot.site_layout
            ],
            "site_layout": [
                {
                    "source_site": layout.source_site,
                    "species": layout.species,
                    "setup_hash": layout.setup_hash,
                    "channels": [channel.label for channel in layout.channels],
                }
                for layout in snapshot.site_layout
            ],
        }
    )
    return ProjectorGreenData(
        kpoints=snapshot.kpoints,
        weights=snapshot.weights,
        eigenvalues=snapshot.eigenvalues,
        occupations=snapshot.occupations,
        coefficients=snapshot.coefficients,
        efermi=snapshot.efermi,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=snapshot.cell,
        positions=snapshot.positions,
        atomic_numbers=snapshot.atomic_numbers,
        projector_l=projector_l,
        projector_m=projector_m,
        projector_radial=projector_radial,
        overlap_k=None,
        overlap_metric=snapshot.overlap_metric,
        population_metric_matrix=snapshot.population_metric_matrix,
        hij=snapshot.hij,
        hij_definition=snapshot.hij_definition,
        hij_units=snapshot.hij_units,
        hij_source=snapshot.hij_source,
        efermi_spin=snapshot.efermi_spin,
        site_nproj=np.asarray(report.site_nproj, dtype=int),
        site_projector_indices=site_projector_indices,
        operator_components=operator_components,
        operator_component_metadata=operator_component_metadata,
        coefficient_source=str(snapshot.provenance["source_code"]),
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_partial_wave_channel",
        operator_basis=report.operator_basis,
        population_metric=snapshot.population_metric,
        metadata=metadata,
    )
