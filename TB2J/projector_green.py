"""Projector-space spectral data and Green-function reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_NAME = "tb2j.projector_green"
SCHEMA_VERSION = "1.0"
COMPLEX_COMPONENT = ["real", "imag"]
GREEN_BACKEND_REQUIRED_ATTRIBUTES = ("kpts", "kweights", "efermi", "nbasis", "norb")
GREEN_BACKEND_REQUIRED_METHODS = ("get_Gk", "get_Gk_all", "get_GR")
SUPPORTED_HIJ_EXCHANGE_DEFINITIONS = (
    "projector_hamiltonian",
    "projector_potential",
    "projected_spin_dependent_potential",
    "paw_dij_projector_hamiltonian",
    "paw_dh_asp_projector_hamiltonian",
    "spin-resolved real(CDIJ) in native PAW projector basis",
    "spin-resolved real(CDIJ) in LPRJ function basis",
    "spin-splitting matrix in VASP LOCPROJ trial-function basis",
    "spin-dependent projector hamiltonian matrix",
    "spin-dependent projector potential matrix",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def encode_complex(array):
    """Encode a complex array with final real/imag dimension."""
    array = np.asarray(array, dtype=complex)
    out = np.empty(array.shape + (2,), dtype=float)
    out[..., 0] = array.real
    out[..., 1] = array.imag
    return out


def decode_complex(array):
    """Decode a final-dimension real/imag array into complex values."""
    array = np.asarray(array)
    if array.shape[-1] != 2:
        raise ValueError("complex arrays must have final dimension of size 2")
    return array[..., 0] + 1j * array[..., 1]


def validate_green_backend(backend):
    """Validate the minimal Green backend protocol used by exchange helpers."""
    missing = [
        name for name in GREEN_BACKEND_REQUIRED_ATTRIBUTES if not hasattr(backend, name)
    ]
    missing.extend(
        name
        for name in GREEN_BACKEND_REQUIRED_METHODS
        if not callable(getattr(backend, name, None))
    )
    if missing:
        raise TypeError(
            "Green backend is missing required protocol members: " + ", ".join(missing)
        )
    return True


def _definition_is_supported(definition, supported_definitions):
    normalized = str(definition).strip().lower()
    return normalized in {str(item).strip().lower() for item in supported_definitions}


def project_potential_to_hij(projectors, potential, weights=None):
    """Project a local potential onto projectors to form H_ij.

    Parameters
    ----------
    projectors : array-like, shape (nproj, ngrid)
        Projector values on the grid used by the DFT exporter.
    potential : array-like, shape (nspin, ngrid) or (ngrid,)
        Spin-resolved local potential/Hamiltonian values on the same grid.
    weights : array-like, shape (ngrid,), optional
        Integration weights. Unit weights are used when omitted.
    """
    projectors = np.asarray(projectors, dtype=complex)
    potential = np.asarray(potential, dtype=complex)
    if projectors.ndim != 2:
        raise ValueError("projectors must have shape (nproj, ngrid)")
    if potential.ndim == 1:
        potential = potential[None, :]
    if potential.ndim != 2:
        raise ValueError("potential must have shape (nspin, ngrid) or (ngrid,)")
    ngrid = projectors.shape[1]
    if potential.shape[1] != ngrid:
        raise ValueError("potential and projectors must use the same grid size")
    if weights is None:
        weights = np.ones(ngrid, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    if weights.shape != (ngrid,):
        raise ValueError("weights must have shape (ngrid,)")
    return np.einsum(
        "ig,sg,jg,g->sij",
        projectors.conj(),
        potential,
        projectors,
        weights,
        optimize="optimal",
    )


def build_site_projector_indices(projector_site):
    """Build padded site-projector indexing arrays from per-projector site ids."""
    projector_site = np.asarray(projector_site, dtype=int)
    if projector_site.ndim != 1:
        raise ValueError("projector_site must have shape (nproj,)")
    sites = sorted(int(site) for site in np.unique(projector_site))
    if sites != list(range(len(sites))):
        raise ValueError("projector_site values must be contiguous and zero-based")
    site_projectors = [np.where(projector_site == site)[0] for site in sites]
    site_nproj = np.array([len(indices) for indices in site_projectors], dtype=int)
    nmax = int(site_nproj.max(initial=0))
    indices = -np.ones((len(sites), nmax), dtype=int)
    for site, projectors in enumerate(site_projectors):
        indices[site, : len(projectors)] = projectors
    return site_nproj, indices


def pack_site_hij(hij_global, site_projector_indices, site_nproj=None):
    """Pack global H_ij matrices into padded site-local blocks."""
    hij_global = np.asarray(hij_global, dtype=complex)
    if hij_global.ndim != 3:
        raise ValueError("hij_global must have shape (nspin, nproj, nproj)")
    site_projector_indices = np.asarray(site_projector_indices, dtype=int)
    if site_projector_indices.ndim != 2:
        raise ValueError("site_projector_indices must have shape (nsite, nmax)")
    if site_nproj is None:
        site_nproj = np.sum(site_projector_indices >= 0, axis=1)
    site_nproj = np.asarray(site_nproj, dtype=int)
    if site_nproj.shape != (site_projector_indices.shape[0],):
        raise ValueError("site_nproj must have shape (nsite,)")
    nspin, nproj, _ = hij_global.shape
    if hij_global.shape[1] != hij_global.shape[2]:
        raise ValueError("hij_global site blocks must be square")
    valid = site_projector_indices >= 0
    if np.any(site_projector_indices[valid] >= nproj):
        raise ValueError("site_projector_indices contains projector out of range")
    nsite, nmax = site_projector_indices.shape
    hij = np.zeros((nspin, nsite, nmax, nmax), dtype=complex)
    for site in range(nsite):
        projectors = site_projector_indices[site, : site_nproj[site]]
        hij[:, site, : site_nproj[site], : site_nproj[site]] = hij_global[
            :, projectors[:, None], projectors
        ]
    return hij


@dataclass
class ProjectorGreenData:
    """Spectral projector data used to reconstruct Green functions in TB2J.

    The NetCDF v1 schema stores full-BZ spectral ingredients only. Green
    functions are derived at requested energies by :class:`ProjectorGreen`.
    """

    kpoints: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    efermi: float
    projector_site: np.ndarray
    projector_atom: np.ndarray
    cell: np.ndarray | None = None
    positions: np.ndarray | None = None
    atomic_numbers: np.ndarray | None = None
    occupations: np.ndarray | None = None
    band_mask: np.ndarray | None = None
    projector_l: np.ndarray | None = None
    projector_m: np.ndarray | None = None
    projector_radial: np.ndarray | None = None
    overlap_metric: np.ndarray | None = None
    overlap_k: np.ndarray | None = None
    population_metric_matrix: np.ndarray | None = None
    site_nproj: np.ndarray | None = None
    site_projector_indices: np.ndarray | None = None
    hij: np.ndarray | None = None
    hij_definition: str | None = None
    hij_units: str | None = None
    hij_source: str | None = None
    hij_projection: str | None = None
    operator_components: dict[str, np.ndarray] | None = None
    operator_component_metadata: dict[str, dict] | None = None
    coefficient_source: str | None = None
    coefficient_projector: str | None = None
    channel_interpretation: str | None = None
    overlap_metric_definition: str | None = None
    population_metric: str | None = None
    operator_basis: str | None = None
    metadata: dict = field(default_factory=dict)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION
    efermi_spin: np.ndarray | None = None

    def __post_init__(self):
        self.kpoints = np.asarray(self.kpoints, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.eigenvalues = np.asarray(self.eigenvalues, dtype=float)
        self.coefficients = np.asarray(self.coefficients, dtype=complex)
        self.projector_site = np.asarray(self.projector_site, dtype=int)
        self.projector_atom = np.asarray(self.projector_atom, dtype=int)
        self.efermi = float(self.efermi)
        if self.efermi_spin is not None:
            self.efermi_spin = np.asarray(self.efermi_spin, dtype=float)
            if self.efermi_spin.shape == ():
                self.efermi_spin = self.efermi_spin.reshape(1)
            self.efermi = float(np.mean(self.efermi_spin))

        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float)
        if self.positions is not None:
            self.positions = np.asarray(self.positions, dtype=float)
        if self.atomic_numbers is not None:
            self.atomic_numbers = np.asarray(self.atomic_numbers, dtype=int)
        if self.occupations is not None:
            self.occupations = np.asarray(self.occupations, dtype=float)
        if self.band_mask is not None:
            self.band_mask = np.asarray(self.band_mask, dtype=bool)
        if self.projector_l is not None:
            self.projector_l = np.asarray(self.projector_l, dtype=int)
        if self.projector_m is not None:
            self.projector_m = np.asarray(self.projector_m, dtype=int)
        if self.projector_radial is not None:
            self.projector_radial = np.asarray(self.projector_radial, dtype=int)
        if self.overlap_metric is not None:
            self.overlap_metric = np.asarray(self.overlap_metric, dtype=complex)
        if self.overlap_k is not None:
            self.overlap_k = np.asarray(self.overlap_k, dtype=complex)
        if self.population_metric_matrix is not None:
            self.population_metric_matrix = np.asarray(
                self.population_metric_matrix, dtype=complex
            )
        if self.site_nproj is not None:
            self.site_nproj = np.asarray(self.site_nproj, dtype=int)
        if self.site_projector_indices is not None:
            self.site_projector_indices = np.asarray(
                self.site_projector_indices, dtype=int
            )
        if self.hij is not None:
            self.hij = np.asarray(self.hij, dtype=complex)
        if self.operator_components is not None:
            self.operator_components = {
                str(name): np.asarray(value, dtype=complex)
                for name, value in self.operator_components.items()
            }
        if self.operator_component_metadata is not None:
            self.operator_component_metadata = {
                str(name): dict(value)
                for name, value in self.operator_component_metadata.items()
            }

        defaults = {
            "units": {
                "cell": "Angstrom",
                "positions": "Angstrom",
                "eigenvalues": "eV",
                "efermi": "eV",
            },
            "kpoint_convention": "fractional_reciprocal",
            "phase_convention": "exp(-2*pi*i*k.R)",
            "complex_component": COMPLEX_COMPONENT,
            "storage_level": "spectral",
        }
        self.metadata = {**defaults, **self.metadata}
        self._sync_metadata_fields()
        self.validate()

    @property
    def has_spin_resolved_fermi(self):
        return self.efermi_spin is not None

    def _sync_metadata_fields(self):
        for name in (
            "coefficient_source",
            "coefficient_projector",
            "channel_interpretation",
            "overlap_metric_definition",
            "population_metric",
            "operator_basis",
        ):
            value = getattr(self, name)
            if value is None:
                value = self.metadata.get(name)
                setattr(self, name, value)
            elif str(value).strip():
                self.metadata[name] = value

    @property
    def nspin(self):
        return self.eigenvalues.shape[0]

    @property
    def nkpt(self):
        return self.kpoints.shape[0]

    @property
    def nband(self):
        return self.eigenvalues.shape[2]

    @property
    def nproj(self):
        return self.coefficients.shape[3]

    def validate(self, exchange_ready=False):
        if self.schema_name != SCHEMA_NAME:
            raise ValueError(f"unsupported schema_name: {self.schema_name}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nkpt, 3)")
        if self.weights.shape != (self.kpoints.shape[0],):
            raise ValueError("weights must have shape (nkpt,)")
        if self.eigenvalues.ndim != 3:
            raise ValueError("eigenvalues must have shape (nspin, nkpt, nband)")
        nspin, nkpt, nband = self.eigenvalues.shape
        if self.efermi_spin is not None and self.efermi_spin.shape != (nspin,):
            raise ValueError("efermi_spin must have shape (nspin,)")
        if nkpt != self.kpoints.shape[0]:
            raise ValueError("eigenvalues and kpoints have inconsistent nkpt")
        if self.coefficients.ndim != 4:
            raise ValueError("coefficients must have shape (nspin, nkpt, nband, nproj)")
        if self.coefficients.shape[:3] != (nspin, nkpt, nband):
            raise ValueError("coefficients must have shape (nspin, nkpt, nband, nproj)")
        nproj = self.coefficients.shape[3]
        if self.projector_site.shape != (nproj,):
            raise ValueError("projector_site must have shape (nproj,)")
        if self.projector_atom.shape != (nproj,):
            raise ValueError("projector_atom must have shape (nproj,)")
        if self.occupations is not None and self.occupations.shape != (
            nspin,
            nkpt,
            nband,
        ):
            raise ValueError("occupations must match eigenvalues shape")
        if self.band_mask is not None and self.band_mask.shape != (
            nspin,
            nkpt,
            nband,
        ):
            raise ValueError("band_mask must match eigenvalues shape")
        if self.cell is not None and self.cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        if self.positions is not None:
            if self.positions.ndim != 2 or self.positions.shape[1] != 3:
                raise ValueError("positions must have shape (natom, 3)")
        if self.atomic_numbers is not None and self.positions is not None:
            if self.atomic_numbers.shape != (self.positions.shape[0],):
                raise ValueError("atomic_numbers must have shape (natom,)")
        for name in ("projector_l", "projector_m", "projector_radial"):
            value = getattr(self, name)
            if value is not None and value.shape != (nproj,):
                raise ValueError(f"{name} must have shape (nproj,)")
        if self.overlap_metric is not None and self.overlap_metric.shape != (
            nproj,
            nproj,
        ):
            raise ValueError("overlap_metric must have shape (nproj, nproj)")
        if self.overlap_k is not None and self.overlap_k.shape != (
            nkpt,
            nproj,
            nproj,
        ):
            raise ValueError("overlap_k must have shape (nkpt, nproj, nproj)")
        if self.population_metric_matrix is not None and (
            self.population_metric_matrix.shape != (nproj, nproj)
        ):
            raise ValueError("population_metric_matrix must have shape (nproj, nproj)")
        self._validate_site_projector_indices(nproj)
        if self.hij is not None:
            self._validate_hij()
        if self.operator_components is not None:
            self._validate_operator_components()
        if (
            exchange_ready
            and self.hij is None
            and not self.has_operator_component("delta_total")
        ):
            raise ValueError(
                "exchange-ready projector data requires spin-resolved hij "
                "or delta_total operator component"
            )
        return True

    def _validate_site_projector_indices(self, nproj):
        if self.site_nproj is None and self.site_projector_indices is None:
            return
        if self.site_nproj is None or self.site_projector_indices is None:
            raise ValueError(
                "site_nproj and site_projector_indices must be provided together"
            )
        if self.site_projector_indices.ndim != 2:
            raise ValueError("site_projector_indices must have shape (nsite, nmax)")
        if self.site_nproj.shape != (self.site_projector_indices.shape[0],):
            raise ValueError("site_nproj must have shape (nsite,)")
        valid = self.site_projector_indices >= 0
        if np.any(self.site_projector_indices[valid] >= nproj):
            raise ValueError("site_projector_indices contains projector out of range")

    def _validate_hij(self):
        if self.hij_definition is None or not str(self.hij_definition).strip():
            raise ValueError("hij requires an explicit definition")
        if self.hij_units is None or not str(self.hij_units).strip():
            raise ValueError("hij requires explicit units")
        if self.hij.ndim != 4:
            raise ValueError(
                "hij must have shape (nspin, nsite, nproj_site_max, " "nproj_site_max)"
            )
        if self.hij.shape[0] != 2:
            raise ValueError("collinear spin-resolved hij requires nspin=2")
        if self.hij.shape[2] != self.hij.shape[3]:
            raise ValueError("hij site blocks must be square")
        if (
            self.site_nproj is not None
            and self.hij.shape[1] != self.site_nproj.shape[0]
        ):
            raise ValueError("hij nsite does not match site_nproj")
        if self.site_nproj is not None and np.any(self.site_nproj > self.hij.shape[2]):
            raise ValueError("hij nproj_site_max is smaller than site_nproj")

    def _validate_operator_components(self):
        if self.site_nproj is None or self.site_projector_indices is None:
            raise ValueError("operator components require site projector indices")
        nsite = len(self.site_nproj)
        nmax = self.site_projector_indices.shape[1]
        for name, value in self.operator_components.items():
            if value.shape != (nsite, nmax, nmax):
                raise ValueError(
                    f"operator component {name!r} must have shape "
                    "(nsite, nproj_site_max, nproj_site_max)"
                )

    def has_operator_component(self, name):
        return self.operator_components is not None and name in self.operator_components

    def get_operator_component(self, name, site=None):
        if not self.has_operator_component(name):
            raise ValueError(f"missing operator component: {name}")
        component = self.operator_components[name]
        if site is None:
            return component
        block = component[int(site)]
        if self.site_nproj is not None:
            nproj = self.site_nproj[int(site)]
            block = block[:nproj, :nproj]
        return block

    def get_hij_spin_difference(self, site=None):
        """Return H_ij(up) - H_ij(down), optionally for one site."""
        if self.hij is None:
            raise ValueError("hij is not available")
        self._validate_hij()
        diff = self.hij[0] - self.hij[1]
        if site is None:
            return diff
        return diff[int(site)]

    def save_netcdf(self, filename):
        """Save spectral projector data to NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for projector Green NetCDF export"
            ) from exc

        with Dataset(Path(filename), "w") as nc:
            nspin, nkpt, nband = self.eigenvalues.shape
            nproj = self.nproj
            nc.createDimension("nspin", nspin)
            nc.createDimension("nkpt", nkpt)
            nc.createDimension("nband", nband)
            nc.createDimension("nproj", nproj)
            nc.createDimension("three", 3)
            nc.createDimension("complex", 2)
            nc.schema_name = self.schema_name
            nc.schema_version = self.schema_version
            nc.metadata_json = json.dumps(_jsonable(self.metadata))
            nc.complex_component = json.dumps(COMPLEX_COMPONENT)

            structure = nc.createGroup("structure")
            if self.cell is not None:
                structure.createVariable("cell", "f8", ("three", "three"))[:] = (
                    self.cell
                )
            if self.positions is not None:
                structure.createDimension("natom", self.positions.shape[0])
                structure.createVariable("positions", "f8", ("natom", "three"))[:] = (
                    self.positions
                )
                if self.atomic_numbers is not None:
                    structure.createVariable("atomic_numbers", "i4", ("natom",))[:] = (
                        self.atomic_numbers
                    )

            kgrp = nc.createGroup("kpoints")
            kgrp.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = self.kpoints
            kgrp.createVariable("weights", "f8", ("nkpt",))[:] = self.weights

            bands = nc.createGroup("bands")
            bands.efermi = self.efermi
            if self.efermi_spin is not None:
                bands.createVariable("efermi_spin", "f8", ("nspin",))[:] = (
                    self.efermi_spin
                )
            bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = (
                self.eigenvalues
            )
            if self.occupations is not None:
                bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[
                    :
                ] = self.occupations

            projectors = nc.createGroup("projectors")
            if self.coefficient_source is not None:
                projectors.coefficient_source = self.coefficient_source
            if self.coefficient_projector is not None:
                projectors.coefficient_projector = self.coefficient_projector
            if self.channel_interpretation is not None:
                projectors.channel_interpretation = self.channel_interpretation
            if self.population_metric is not None:
                projectors.population_metric = self.population_metric
            projectors.createVariable(
                "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
            )[:] = encode_complex(self.coefficients)
            projectors.createVariable("projector_site", "i4", ("nproj",))[:] = (
                self.projector_site
            )
            projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = (
                self.projector_atom
            )
            self._write_optional_projector_array(
                projectors, "projector_l", self.projector_l
            )
            self._write_optional_projector_array(
                projectors, "projector_m", self.projector_m
            )
            self._write_optional_projector_array(
                projectors, "projector_radial", self.projector_radial
            )
            if self.overlap_metric is not None:
                metric = projectors.createVariable(
                    "overlap_metric", "f8", ("nproj", "nproj", "complex")
                )
                metric[:] = encode_complex(self.overlap_metric)
                metric.definition = (
                    self.overlap_metric_definition or "projector overlap metric"
                )
            if self.overlap_k is not None:
                overlap_k = projectors.createVariable(
                    "overlap_k", "f8", ("nkpt", "nproj", "nproj", "complex")
                )
                overlap_k[:] = encode_complex(self.overlap_k)
                overlap_k.definition = (
                    self.overlap_metric_definition or "k-dependent projector overlap"
                )
            if self.population_metric_matrix is not None:
                metric = projectors.createVariable(
                    "population_metric_matrix", "f8", ("nproj", "nproj", "complex")
                )
                metric[:] = encode_complex(self.population_metric_matrix)
                metric.definition = (
                    self.population_metric or "projector population metric"
                )
            if self.site_nproj is not None:
                nsite, nmax = self.site_projector_indices.shape
                projectors.createDimension("nsite", nsite)
                projectors.createDimension("nproj_site_max", nmax)
                projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = (
                    self.site_nproj
                )
                projectors.createVariable(
                    "site_projector_indices", "i4", ("nsite", "nproj_site_max")
                )[:] = self.site_projector_indices

            if self.hij is not None or self.operator_components is not None:
                operators = nc.createGroup("operators")

            if self.hij is not None:
                if "nsite" not in nc.dimensions:
                    nc.createDimension("nsite", self.hij.shape[1])
                if "nproj_site_max" not in nc.dimensions:
                    nc.createDimension("nproj_site_max", self.hij.shape[2])
                hij = operators.createVariable(
                    "hij",
                    "f8",
                    (
                        "nspin",
                        "nsite",
                        "nproj_site_max",
                        "nproj_site_max",
                        "complex",
                    ),
                )
                hij[:] = encode_complex(self.hij)
                hij.definition = self.hij_definition
                hij.units = self.hij_units
                if self.hij_source is not None:
                    hij.source = self.hij_source
                if self.hij_projection is not None:
                    hij.projection = self.hij_projection
                if self.operator_basis is not None:
                    hij.operator_basis = self.operator_basis

            if self.operator_components is not None:
                if "nsite" not in nc.dimensions:
                    nc.createDimension(
                        "nsite", next(iter(self.operator_components.values())).shape[0]
                    )
                if "nproj_site_max" not in nc.dimensions:
                    nc.createDimension(
                        "nproj_site_max",
                        next(iter(self.operator_components.values())).shape[1],
                    )
                components = operators.createGroup("operator_components")
                for name, component in self.operator_components.items():
                    variable = components.createVariable(
                        name,
                        "f8",
                        ("nsite", "nproj_site_max", "nproj_site_max", "complex"),
                    )
                    variable[:] = encode_complex(component)
                    if self.operator_component_metadata is not None:
                        for key, value in self.operator_component_metadata.get(
                            name, {}
                        ).items():
                            setattr(variable, key, value)

    @staticmethod
    def _write_optional_projector_array(group, name, value):
        if value is not None:
            group.createVariable(name, "i4", ("nproj",))[:] = value

    @classmethod
    def load_netcdf(cls, filename):
        """Load spectral projector data from NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for projector Green NetCDF import"
            ) from exc

        with Dataset(filename) as nc:
            if getattr(nc, "schema_name", None) != SCHEMA_NAME:
                raise ValueError("unsupported projector Green schema")
            if "greens_k" in nc.groups or "greens_R" in nc.groups:
                raise ValueError("v1 projector Green files must not store Green arrays")
            metadata = json.loads(getattr(nc, "metadata_json", "{}"))
            structure = nc.groups.get("structure")
            cell = positions = atomic_numbers = None
            if structure is not None:
                if "cell" in structure.variables:
                    cell = structure.variables["cell"][:]
                if "positions" in structure.variables:
                    positions = structure.variables["positions"][:]
                if "atomic_numbers" in structure.variables:
                    atomic_numbers = structure.variables["atomic_numbers"][:]
            kgrp = nc.groups["kpoints"]
            bands = nc.groups["bands"]
            projectors = nc.groups["projectors"]
            occupations = None
            if "occupations" in bands.variables:
                occupations = bands.variables["occupations"][:]
            efermi_spin = None
            if "efermi_spin" in bands.variables:
                efermi_spin = bands.variables["efermi_spin"][:]
            projector_l = cls._optional_var(projectors, "projector_l")
            projector_m = cls._optional_var(projectors, "projector_m")
            projector_radial = cls._optional_var(projectors, "projector_radial")
            coefficient_source = getattr(
                projectors, "coefficient_source", metadata.get("coefficient_source")
            )
            coefficient_projector = getattr(
                projectors,
                "coefficient_projector",
                metadata.get("coefficient_projector"),
            )
            channel_interpretation = getattr(
                projectors,
                "channel_interpretation",
                metadata.get("channel_interpretation"),
            )
            population_metric = getattr(
                projectors, "population_metric", metadata.get("population_metric")
            )
            overlap_metric = None
            overlap_k = None
            population_metric_matrix = None
            overlap_metric_definition = metadata.get("overlap_metric_definition")
            if "overlap_metric" in projectors.variables:
                overlap_var = projectors.variables["overlap_metric"]
                overlap_metric = decode_complex(overlap_var[:])
                overlap_metric_definition = getattr(
                    overlap_var, "definition", overlap_metric_definition
                )
            if "overlap_k" in projectors.variables:
                overlap_k_var = projectors.variables["overlap_k"]
                overlap_k = decode_complex(overlap_k_var[:])
                overlap_metric_definition = getattr(
                    overlap_k_var, "definition", overlap_metric_definition
                )
            if "population_metric_matrix" in projectors.variables:
                population_metric_matrix = decode_complex(
                    projectors.variables["population_metric_matrix"][:]
                )
            site_nproj = cls._optional_var(projectors, "site_nproj")
            site_projector_indices = cls._optional_var(
                projectors, "site_projector_indices"
            )
            hij = hij_definition = hij_units = hij_source = hij_projection = None
            operator_components = None
            operator_component_metadata = None
            operator_basis = metadata.get("operator_basis")
            operators = nc.groups.get("operators")
            if operators is not None and "hij" in operators.variables:
                hij_var = operators.variables["hij"]
                hij = decode_complex(hij_var[:])
                hij_definition = getattr(hij_var, "definition", None)
                hij_units = getattr(hij_var, "units", None)
                hij_source = getattr(hij_var, "source", None)
                hij_projection = getattr(hij_var, "projection", None)
                operator_basis = getattr(hij_var, "operator_basis", operator_basis)
            if operators is not None and "operator_components" in operators.groups:
                components_group = operators.groups["operator_components"]
                operator_components = {}
                operator_component_metadata = {}
                for name, variable in components_group.variables.items():
                    operator_components[name] = decode_complex(variable[:])
                    operator_component_metadata[name] = {
                        key: getattr(variable, key) for key in variable.ncattrs()
                    }
            return cls(
                kpoints=kgrp.variables["kpoints"][:],
                weights=kgrp.variables["weights"][:],
                eigenvalues=bands.variables["eigenvalues"][:],
                coefficients=decode_complex(projectors.variables["coefficients"][:]),
                efermi=getattr(bands, "efermi"),
                efermi_spin=efermi_spin,
                projector_site=projectors.variables["projector_site"][:],
                projector_atom=projectors.variables["projector_atom"][:],
                cell=cell,
                positions=positions,
                atomic_numbers=atomic_numbers,
                occupations=occupations,
                band_mask=None,
                projector_l=projector_l,
                projector_m=projector_m,
                projector_radial=projector_radial,
                overlap_metric=overlap_metric,
                overlap_k=overlap_k,
                population_metric_matrix=population_metric_matrix,
                site_nproj=site_nproj,
                site_projector_indices=site_projector_indices,
                hij=hij,
                hij_definition=hij_definition,
                hij_units=hij_units,
                hij_source=hij_source,
                hij_projection=hij_projection,
                operator_components=operator_components,
                operator_component_metadata=operator_component_metadata,
                coefficient_source=coefficient_source,
                coefficient_projector=coefficient_projector,
                channel_interpretation=channel_interpretation,
                overlap_metric_definition=overlap_metric_definition,
                population_metric=population_metric,
                operator_basis=operator_basis,
                metadata=metadata,
            )

    @staticmethod
    def _optional_var(group, name):
        if name in group.variables:
            return group.variables[name][:]
        return None

    @classmethod
    def load_nc_pao_netcdf(cls, filename):
        """Load a minimal norm-conserving PAO NetCDF fixture.

        This adapter is intentionally narrow: it maps root-level spectral arrays
        from a simple NC PAO file into the canonical ProjectorGreenData model.
        """
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError("netCDF4 is required for NC PAO NetCDF import") from exc

        with Dataset(filename) as nc:
            metadata = json.loads(getattr(nc, "metadata_json", "{}"))
            metadata.setdefault("coefficient_projector", "nc_pao")
            metadata.setdefault("channel_interpretation", "norm_conserving_pao")

            overlap_k = None
            overlap_metric_definition = metadata.get(
                "overlap_metric_definition", "k-dependent NC PAO overlap"
            )
            if "overlap_k" in nc.variables:
                overlap_var = nc.variables["overlap_k"]
                overlap_k = decode_complex(overlap_var[:])
                overlap_metric_definition = getattr(
                    overlap_var, "definition", overlap_metric_definition
                )

            return cls(
                kpoints=nc.variables["kpoints"][:],
                weights=nc.variables["weights"][:],
                eigenvalues=nc.variables["eigenvalues"][:],
                coefficients=decode_complex(nc.variables["coefficients"][:]),
                efermi=float(getattr(nc, "efermi")),
                projector_site=nc.variables["projector_site"][:],
                projector_atom=nc.variables["projector_atom"][:],
                cell=nc.variables["cell"][:] if "cell" in nc.variables else None,
                positions=(
                    nc.variables["positions"][:]
                    if "positions" in nc.variables
                    else None
                ),
                atomic_numbers=(
                    nc.variables["atomic_numbers"][:]
                    if "atomic_numbers" in nc.variables
                    else None
                ),
                projector_l=(
                    nc.variables["projector_l"][:]
                    if "projector_l" in nc.variables
                    else None
                ),
                projector_m=(
                    nc.variables["projector_m"][:]
                    if "projector_m" in nc.variables
                    else None
                ),
                projector_radial=(
                    nc.variables["projector_radial"][:]
                    if "projector_radial" in nc.variables
                    else None
                ),
                overlap_k=overlap_k,
                coefficient_source=getattr(
                    nc, "coefficient_source", metadata.get("coefficient_source")
                ),
                coefficient_projector=getattr(
                    nc, "coefficient_projector", metadata.get("coefficient_projector")
                ),
                channel_interpretation=getattr(
                    nc,
                    "channel_interpretation",
                    metadata.get("channel_interpretation"),
                ),
                overlap_metric_definition=overlap_metric_definition,
                metadata=metadata,
            )


class ProjectorGreen:
    """Runtime projector-space Green-function backend."""

    def __init__(self, data: ProjectorGreenData):
        self.data = data
        self.kpts = data.kpoints
        self.kweights = data.weights
        self.efermi = data.efermi
        self.efermi_spin = data.efermi_spin
        self.nbasis = data.nproj
        self.norb = data.nproj
        self.k2Rfactor = -2.0j * np.pi
        # PAW projector overlaps are dual coefficients: the spectral sum already
        # yields the dual-dual Green matrix, so no S^-1 G S^-1 dressing is applied
        # here (derivation [D], PAW_LKAG_derivation). Only the k-dependent NC-PAO
        # overlap (overlap_k) triggers the contravariant transform in _contravariant_Gk.
        self.is_orthogonal = data.overlap_k is None
        self.overlap_condition_threshold = float(
            data.metadata.get("overlap_condition_threshold", 1.0e12)
        )
        self.adjusted_emin = float(np.min(data.eigenvalues) - data.efermi)
        import os

        self.use_contravariant = (
            os.environ.get("TB2J_GREEN_MODE", "contravariant") != "plain"
        )

    def _fermi(self, ispin):
        if self.efermi_spin is not None:
            return float(self.efermi_spin[int(ispin)])
        return self.efermi

    def get_Gk(self, ik, energy, ispin=0):
        evals = self.data.eigenvalues[ispin, ik]
        coeff = self.data.coefficients[ispin, ik]
        if self.data.band_mask is not None:
            mask = self.data.band_mask[ispin, ik]
            evals = evals[mask]
            coeff = coeff[mask]
        inv_denom = 1.0 / (energy + self._fermi(ispin) - evals)
        Gk = np.einsum("np,nq,n->pq", coeff, coeff.conj(), inv_denom)
        return self._contravariant_Gk(Gk, ik)

    def get_Gk_all(self, energy, ispin=0):
        if self.data.band_mask is not None:
            return np.asarray(
                [self.get_Gk(ik, energy, ispin=ispin) for ik in range(self.data.nkpt)],
                dtype=complex,
            )
        evals = self.data.eigenvalues[ispin]
        coeff = self.data.coefficients[ispin]
        inv_denom = 1.0 / (energy + self._fermi(ispin) - evals)
        Gk_all = np.einsum("knp,knq,kn->kpq", coeff, coeff.conj(), inv_denom)
        if self.data.overlap_k is None:
            return Gk_all
        return np.asarray(
            [self._contravariant_Gk(Gk, ik) for ik, Gk in enumerate(Gk_all)],
            dtype=complex,
        )

    def get_Sk(self, ik):
        """Return the projector overlap for one k point."""
        if self.data.overlap_k is not None:
            return self.data.overlap_k[ik]
        if self.data.overlap_metric is not None:
            return self.data.overlap_metric
        return np.eye(self.nbasis, dtype=complex)

    def _contravariant_Gk(self, Gk, ik):
        if self.data.overlap_k is None or not self.use_contravariant:
            return Gk
        Sk = self.get_Sk(ik)
        condition = np.linalg.cond(Sk)
        if not np.isfinite(condition) or condition > self.overlap_condition_threshold:
            raise ValueError(
                "overlap_k is singular or ill-conditioned at k-point "
                f"{ik}: condition={condition:.3e}, "
                f"threshold={self.overlap_condition_threshold:.3e}"
            )
        Sinv = np.linalg.inv(Sk)
        return Sinv @ Gk @ Sinv

    def compute_GR(self, Rpts, kpts, Gks):
        Rvecs = np.asarray(Rpts, dtype=float)
        kpts = np.asarray(kpts, dtype=float)
        Gks = np.asarray(Gks, dtype=complex)
        if Rvecs.ndim != 2 or Rvecs.shape[1] != 3:
            raise ValueError("Rpts must have shape (nR, 3)")
        if kpts.ndim != 2 or kpts.shape[1] != 3:
            raise ValueError("kpts must have shape (nkpt, 3)")
        if Gks.shape != (kpts.shape[0], self.nbasis, self.nbasis):
            raise ValueError("Gks must have shape (nkpt, nproj, nproj)")
        if self.kweights.shape != (kpts.shape[0],):
            raise ValueError("k-point weights must match kpts")
        phase = np.exp(self.k2Rfactor * np.einsum("ri,ki->rk", Rvecs, kpts))
        phase *= self.kweights[None, :]
        return np.einsum("kpq,rk->rpq", Gks, phase, optimize="optimal")

    def get_GR(self, Rpts, energy, Gk_all=None, ispin=0):
        if Gk_all is None:
            Gk_all = self.get_Gk_all(energy, ispin=ispin)
        return self.compute_GR(Rpts, self.kpts, Gk_all)

    def get_site_projectors(self, site):
        site = int(site)
        if self.data.site_projector_indices is not None:
            nproj = self.data.site_nproj[site]
            return self.data.site_projector_indices[site, :nproj]
        return np.where(self.data.projector_site == site)[0]

    def get_site_block(self, matrix, iatom, jatom):
        """Return the projector block connecting two sites."""
        matrix = np.asarray(matrix)
        if matrix.shape != (self.nbasis, self.nbasis):
            raise ValueError("matrix must have shape (nproj, nproj)")
        iproj = self.get_site_projectors(iatom)
        jproj = self.get_site_projectors(jatom)
        return matrix[np.ix_(iproj, jproj)]

    def get_local_operator(
        self,
        site,
        operator="hij_spin_difference",
        supported_definitions=SUPPORTED_HIJ_EXCHANGE_DEFINITIONS,
    ):
        """Return a validated site-local operator for exchange-like traces."""
        if operator != "hij_spin_difference":
            raise ValueError(f"unsupported local operator source: {operator}")
        # Prefer the explicit XC exchange field (V_xc^up - V_xc^down) partial-wave
        # matrix when exported (GPAW eq:pseudo-partial-delta); fall back to the
        # assembled delta_total, then to the hij spin splitting.
        if self.data.has_operator_component("delta_xc"):
            return self.data.get_operator_component("delta_xc", site=site)
        if self.data.has_operator_component("delta_total"):
            return self.data.get_operator_component("delta_total", site=site)
        if not _definition_is_supported(
            self.data.hij_definition, supported_definitions
        ):
            raise ValueError(
                "unsupported hij definition for exchange trace: "
                f"{self.data.hij_definition!r}"
            )
        block = self.data.get_hij_spin_difference(site=site)
        if self.data.site_nproj is not None:
            nproj = self.data.site_nproj[int(site)]
            block = block[:nproj, :nproj]
        return block

    def get_local_operators(
        self,
        sites=None,
        operator="hij_spin_difference",
        supported_definitions=SUPPORTED_HIJ_EXCHANGE_DEFINITIONS,
    ):
        """Return site-local operators keyed by site index."""
        if sites is None:
            sites = self.get_sites()
        return {
            int(site): self.get_local_operator(
                site,
                operator=operator,
                supported_definitions=supported_definitions,
            )
            for site in sites
        }

    def get_sites(self):
        """Return sorted site indices represented by projector metadata."""
        if self.data.site_nproj is not None:
            return list(range(len(self.data.site_nproj)))
        return sorted(int(site) for site in np.unique(self.data.projector_site))


def projector_exchange_trace(
    green,
    Rpts,
    energy,
    local_operators=None,
    sites=None,
    supported_definitions=SUPPORTED_HIJ_EXCHANGE_DEFINITIONS,
):
    """Compute a collinear projector exchange-like trace for one energy.

    This validates the projector mechanics only. It evaluates
    Tr[Delta_i G_up_ij(R) Delta_j G_down_ji(-R)] / (4*pi) using explicit
    site-local spin-dependent projector Hamiltonian/potential operators.
    """
    validate_green_backend(green)
    if not callable(getattr(green, "get_site_block", None)):
        raise TypeError("Projector exchange trace requires get_site_block()")
    Rpts = np.asarray(Rpts, dtype=int)
    if Rpts.ndim != 2 or Rpts.shape[1] != 3:
        raise ValueError("Rpts must have shape (nR, 3)")
    Rkeys = [tuple(int(x) for x in R) for R in Rpts]
    R_index = {R: i for i, R in enumerate(Rkeys)}
    missing_negative = [R for R in Rkeys if tuple(-x for x in R) not in R_index]
    if missing_negative:
        raise ValueError("Rpts must include each negative R vector")

    if sites is None:
        if not hasattr(green, "get_sites"):
            raise TypeError(
                "Green backend must provide get_sites() or sites must be set"
            )
        sites = green.get_sites()
    sites = [int(site) for site in sites]

    local_operator_source = "explicit"
    if local_operators is None:
        if not hasattr(green, "get_local_operators"):
            raise TypeError(
                "Green backend must provide get_local_operators() when "
                "local_operators is not set"
            )
        local_operator_source = "hij_spin_difference"
        local_operators = green.get_local_operators(
            sites=sites, supported_definitions=supported_definitions
        )
    local_operators = {
        int(site): np.asarray(op) for site, op in local_operators.items()
    }
    missing_operators = [site for site in sites if site not in local_operators]
    if missing_operators:
        raise ValueError(
            "local_operators is missing site(s): "
            + ", ".join(str(site) for site in missing_operators)
        )

    Gup = green.get_GR(Rpts, energy=energy, ispin=0)
    Gdn = green.get_GR(Rpts, energy=energy, ispin=1)
    traces = {}
    orbital_traces = {}
    for iR, R in enumerate(Rkeys):
        iRm = R_index[tuple(-x for x in R)]
        for iatom in sites:
            for jatom in sites:
                Delta_i = local_operators[iatom]
                Delta_j = local_operators[jatom]
                Gij_up = green.get_site_block(Gup[iR], iatom, jatom)
                Gji_dn = green.get_site_block(Gdn[iRm], jatom, iatom)
                if Delta_i.shape != (Gij_up.shape[0], Gij_up.shape[0]):
                    raise ValueError(
                        "local operator shape does not match site projectors"
                    )
                if Delta_j.shape != (Gji_dn.shape[0], Gji_dn.shape[0]):
                    raise ValueError(
                        "local operator shape does not match site projectors"
                    )
                orbital = np.einsum("ij,ji->ij", Delta_i @ Gij_up, Delta_j @ Gji_dn) / (
                    4.0 * np.pi
                )
                key = (R, iatom, jatom)
                orbital_traces[key] = orbital
                traces[key] = np.sum(orbital)
    return {
        "trace": traces,
        "orbital_trace": orbital_traces,
        "method": "projector_exchange_trace",
        "local_operator": local_operator_source,
        "normalization": "1/(4*pi)",
    }


def projector_charge_moments_from_green(green, contour, sites=None):
    """Compute projector-space charges and collinear moments from Green functions.

    The convention follows TB2J's existing contour-density path.  For NC PAOs,
    the Green function is contravariant, so its density is left-contracted by
    the k-dependent overlap on the right before taking the site trace.
    """
    validate_green_backend(green)
    if not callable(getattr(green, "get_site_block", None)):
        raise TypeError("projector charge integration requires get_site_block()")
    if not hasattr(contour, "path") or not callable(
        getattr(contour, "integrate_values", None)
    ):
        raise TypeError("contour must provide path and integrate_values()")
    if sites is None:
        if not hasattr(green, "get_sites"):
            raise TypeError(
                "Green backend must provide get_sites() or sites must be set"
            )
        sites = green.get_sites()
    sites = [int(site) for site in sites]
    nsites = max(sites, default=-1) + 1
    charges = np.zeros(nsites, dtype=float)
    spinat = np.zeros((nsites, 3), dtype=float)
    density_by_spin = np.zeros((green.data.nspin, nsites), dtype=float)

    R0 = np.array([[0, 0, 0]], dtype=int)
    for ispin in range(green.data.nspin):
        site_diags = {site: [] for site in sites}
        for energy in contour.path:
            if green.data.overlap_k is None:
                GR0 = green.get_GR(R0, energy=energy, ispin=ispin)[0]
            else:
                Gk = green.get_Gk_all(energy, ispin=ispin)
                GR0 = np.einsum(
                    "k,kpq,kqr->pr",
                    green.kweights,
                    Gk,
                    green.data.overlap_k,
                    optimize="optimal",
                )
            for site in sites:
                block = green.get_site_block(GR0, site, site)
                site_diags[site].append(np.diag(block))
        for site in sites:
            integrated = (
                -np.imag(contour.integrate_values(np.asarray(site_diags[site]))) / np.pi
            )
            density_by_spin[ispin, site] = float(np.sum(integrated))

    charges[:] = np.sum(density_by_spin, axis=0)
    if green.data.nspin >= 2:
        spinat[:, 2] = density_by_spin[0] - density_by_spin[1]
    return {
        "charges": charges,
        "spinat": spinat,
        "density_by_spin": density_by_spin,
        "method": "projector_green_contour_diagonal",
    }
