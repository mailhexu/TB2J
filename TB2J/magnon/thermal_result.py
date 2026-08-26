"""Versioned result data model for thermal-magnon calculations.

Schema ``tb2j.magnon.thermal`` version 1.0: JSON by default with an optional
NetCDF representation for dense band data, following the conventions of
``tb2j.magnon.eigenstates`` (core energies in eV, k-points in fractional
reciprocal coordinates, complex values split into real/imag).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

SCHEMA_NAME = "tb2j.magnon.thermal"
SCHEMA_VERSION = "1.0"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class TransitionRecord:
    """Method-specific transition (or breakdown) outcome."""

    kind: str  # curie_temperature | neel_temperature | temperature_hp_breakdown
    temperature_K: float
    converged: bool
    method_validity: str  # nominal | limited
    detail: Optional[str] = None
    breakdown_magnetization: Optional[float] = None
    validity_reason: Optional[str] = None

    def to_dict(self):
        return _jsonable(
            {
                "kind": self.kind,
                "temperature_K": self.temperature_K,
                "converged": self.converged,
                "method_validity": self.method_validity,
                "detail": self.detail,
                "breakdown_magnetization": self.breakdown_magnetization,
                "validity_reason": self.validity_reason,
            }
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            kind=data["kind"],
            temperature_K=float(data["temperature_K"]),
            converged=bool(data["converged"]),
            method_validity=data["method_validity"],
            detail=data.get("detail"),
            breakdown_magnetization=data.get("breakdown_magnetization"),
            validity_reason=data.get("validity_reason"),
        )


@dataclass
class MeshHistoryEntry:
    """One q-mesh estimate in the transition convergence sequence."""

    qmesh: List[int]
    estimate_K: float
    residual: float
    iterations: int
    min_energy_eV: float
    status: str  # refined | converged | unconverged

    def to_dict(self):
        return _jsonable(
            {
                "qmesh": self.qmesh,
                "estimate_K": self.estimate_K,
                "residual": self.residual,
                "iterations": self.iterations,
                "min_energy_eV": self.min_energy_eV,
                "status": self.status,
            }
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            qmesh=[int(m) for m in data["qmesh"]],
            estimate_K=float(data["estimate_K"]),
            residual=float(data["residual"]),
            iterations=int(data["iterations"]),
            min_energy_eV=float(data["min_energy_eV"]),
            status=data["status"],
        )


@dataclass
class ThermalBandBlock:
    """Magnon bands evaluated at one temperature after self-consistency."""

    temperature_K: float
    kpoints: np.ndarray
    energies_eV: np.ndarray
    order_parameters: np.ndarray
    status: str  # ordered | zero_transition | hp_breakdown | unstable_reference
    magnetization: Optional[np.ndarray] = None  # per-site <S^z> in local frames

    def __post_init__(self):
        self.kpoints = np.asarray(self.kpoints, dtype=float)
        self.energies_eV = np.asarray(self.energies_eV, dtype=float)
        self.order_parameters = np.asarray(self.order_parameters, dtype=float)
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nkpt, 3)")
        if self.energies_eV.ndim != 2:
            raise ValueError("energies_eV must have shape (nkpt, nmode)")
        if self.energies_eV.shape[0] != self.kpoints.shape[0]:
            raise ValueError("kpoints and energies_eV must share nkpt")

    def to_dict(self):
        return _jsonable(
            {
                "temperature_K": self.temperature_K,
                "kpoints": self.kpoints,
                "energies_eV": self.energies_eV,
                "order_parameters": self.order_parameters,
                "status": self.status,
                "magnetization": self.magnetization,
            }
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            temperature_K=float(data["temperature_K"]),
            kpoints=np.array(data["kpoints"], dtype=float),
            energies_eV=np.array(data["energies_eV"], dtype=float),
            order_parameters=np.array(data["order_parameters"], dtype=float),
            status=data["status"],
            magnetization=(
                np.array(data["magnetization"], dtype=float)
                if data.get("magnetization") is not None
                else None
            ),
        )


@dataclass
class ThermalMagnonResult:
    """Complete thermal-magnon calculation result.

    Core energies are in eV; temperatures in K; k-points in fractional
    reciprocal coordinates. Physical outcomes (zero transition, unstable
    reference, HP breakdown) are recorded as statuses, never as silent
    numeric failures.
    """

    method: str
    spin_regime: str
    spin_interpretation: str  # physical_quantum_spin | effective_quantum_spin
    spins: List[float]
    order_mode: str
    dimensionality: int
    status: str  # ok | zero_transition | unstable_reference | unconverged
    transition: Optional[TransitionRecord] = None
    mesh_history: List[MeshHistoryEntry] = field(default_factory=list)
    bands: List[ThermalBandBlock] = field(default_factory=list)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "metadata": {
                "method": self.method,
                "spin_regime": self.spin_regime,
                "spin_interpretation": self.spin_interpretation,
                "spins": list(self.spins),
                "order_mode": self.order_mode,
                "dimensionality": self.dimensionality,
                "units": {"energies": "eV", "temperatures": "K"},
                "kpoint_convention": "fractional_reciprocal",
            },
            "status": self.status,
            "transition": self.transition.to_dict() if self.transition else None,
            "mesh_history": [e.to_dict() for e in self.mesh_history],
            "bands": [b.to_dict() for b in self.bands],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_name") != SCHEMA_NAME:
            raise ValueError(
                f"Unsupported thermal-magnon schema: {data.get('schema_name')!r}"
            )
        metadata = data.get("metadata", data)
        return cls(
            method=metadata["method"],
            spin_regime=metadata["spin_regime"],
            spin_interpretation=metadata["spin_interpretation"],
            spins=[float(s) for s in metadata["spins"]],
            order_mode=metadata["order_mode"],
            dimensionality=int(metadata["dimensionality"]),
            status=data["status"],
            transition=(
                TransitionRecord.from_dict(data["transition"])
                if data.get("transition")
                else None
            ),
            mesh_history=[
                MeshHistoryEntry.from_dict(e) for e in data.get("mesh_history", [])
            ],
            bands=[ThermalBandBlock.from_dict(b) for b in data.get("bands", [])],
            schema_name=data.get("schema_name", SCHEMA_NAME),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save_json(self, filename):
        """Save the thermal result to versioned JSON."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filename):
        """Load a thermal result from versioned JSON."""
        with open(filename) as f:
            return cls.from_dict(json.load(f))

    def save_netcdf(self, filename):
        """Save the thermal result to NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for thermal-magnon NetCDF export"
            ) from exc

        filename = Path(filename)
        with Dataset(filename, "w") as nc:
            nc.schema_name = self.schema_name
            nc.schema_version = self.schema_version
            nc.method = self.method
            nc.status = self.status
            metadata = self.to_dict()["metadata"]
            nc.metadata_json = json.dumps(_jsonable(metadata))
            nc.transition_json = (
                json.dumps(self.transition.to_dict()) if self.transition else "null"
            )
            nc.mesh_history_json = json.dumps([e.to_dict() for e in self.mesh_history])

            nband = len(self.bands)
            nc.createDimension("nband", nband if nband else 1)
            for iband, band in enumerate(self.bands):
                group = nc.createGroup(f"band{iband:03d}")
                group.temperature_K = band.temperature_K
                group.status = band.status
                nkpt = band.kpoints.shape[0]
                group.createDimension("nkpt", nkpt)
                group.createDimension("xyz", 3)
                nmode = band.energies_eV.shape[1]
                group.createDimension("nmode", nmode if nmode else 1)
                group.createDimension("nsite", len(self.spins))
                group.createVariable("kpoints", "f8", ("nkpt", "xyz"))[:] = band.kpoints
                group.createVariable("energies_eV", "f8", ("nkpt", "nmode"))[:] = (
                    band.energies_eV
                )
                op_var = group.createVariable("order_parameters", "f8", ("nkpt",))
                op_var[:] = band.order_parameters
                mag = group.createVariable(
                    "magnetization", "f8", ("nsite",), fill_value=False
                )
                if band.magnetization is not None:
                    mag[:] = band.magnetization
                else:
                    mag[:] = np.full(len(self.spins), np.nan)

    @classmethod
    def load_netcdf(cls, filename):
        """Load a thermal result from NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for thermal-magnon NetCDF import"
            ) from exc

        with Dataset(filename) as nc:
            if getattr(nc, "schema_name", None) != SCHEMA_NAME:
                raise ValueError(
                    f"Unsupported thermal-magnon schema: "
                    f"{getattr(nc, 'schema_name', None)!r}"
                )
            metadata = json.loads(getattr(nc, "metadata_json", "{}"))
            transition = None
            raw = json.loads(getattr(nc, "transition_json", "null"))
            if raw:
                transition = TransitionRecord.from_dict(raw)
            mesh_history = [
                MeshHistoryEntry.from_dict(e)
                for e in json.loads(getattr(nc, "mesh_history_json", "[]"))
            ]
            bands = []
            for name, group in nc.groups.items():
                if not name.startswith("band"):
                    continue
                mag = np.asarray(group.variables["magnetization"][:], dtype=float)
                bands.append(
                    ThermalBandBlock(
                        temperature_K=float(group.temperature_K),
                        kpoints=np.array(group.variables["kpoints"][:]),
                        energies_eV=np.array(group.variables["energies_eV"][:]),
                        order_parameters=np.array(
                            group.variables["order_parameters"][:]
                        ),
                        status=group.status,
                        magnetization=None if np.all(np.isnan(mag)) else mag,
                    )
                )
            return cls(
                method=nc.method,
                spin_regime=metadata["spin_regime"],
                spin_interpretation=metadata["spin_interpretation"],
                spins=[float(s) for s in metadata["spins"]],
                order_mode=metadata["order_mode"],
                dimensionality=int(metadata["dimensionality"]),
                status=nc.status,
                transition=transition,
                mesh_history=mesh_history,
                bands=bands,
                schema_name=nc.schema_name,
                schema_version=nc.schema_version,
            )

    @classmethod
    def load(cls, filename):
        """Load a thermal result from JSON or NetCDF by extension."""
        suffix = Path(filename).suffix.lower()
        if suffix in {".nc", ".netcdf"}:
            return cls.load_netcdf(filename)
        return cls.load_json(filename)
