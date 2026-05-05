"""Data model for magnon eigenstate calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


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


@dataclass
class MagnonEigenstateData:
    """Container for magnon energies, optional wavefunctions, and metadata.

    Energies are stored in eV. K-points use fractional reciprocal coordinates
    unless the metadata records a different convention.
    """

    calculation_type: str
    kpoints: np.ndarray
    energies: np.ndarray
    wavefunctions: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    plot: Optional[dict] = None
    schema_name: str = "tb2j.magnon.eigenstates"
    schema_version: str = "1.0"

    def __post_init__(self):
        self.kpoints = np.asarray(self.kpoints, dtype=float)
        self.energies = np.asarray(self.energies, dtype=float)
        if self.wavefunctions is not None:
            self.wavefunctions = np.asarray(self.wavefunctions, dtype=complex)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)

        self._validate_shapes()

        defaults = {
            "units": {"energies": "eV"},
            "kpoint_convention": "fractional_reciprocal",
        }
        self.metadata = {**defaults, **self.metadata}

    def _validate_shapes(self):
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (nkpt, 3)")
        if self.energies.ndim != 2:
            raise ValueError("energies must have shape (nkpt, nmode)")
        if self.energies.shape[0] != self.kpoints.shape[0]:
            raise ValueError(
                "kpoints and energies must have the same number of k-points"
            )
        if self.wavefunctions is not None:
            expected = self.energies.shape
            if self.wavefunctions.ndim != 3:
                raise ValueError(
                    "wavefunctions must have shape (nkpt, nmode, ncomponent)"
                )
            if self.wavefunctions.shape[:2] != expected:
                raise ValueError(
                    "wavefunctions and energies must have matching k-point and mode dimensions"
                )
        if self.weights is not None and self.weights.shape != (self.kpoints.shape[0],):
            raise ValueError("weights must have shape (nkpt,)")

    def to_dict(self):
        """Return a JSON-serializable representation."""
        data = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "calculation_type": self.calculation_type,
            "metadata": _jsonable(self.metadata),
            "kpoints": self.kpoints.tolist(),
            "energies": self.energies.tolist(),
            "weights": self.weights.tolist() if self.weights is not None else None,
            "wavefunctions": None,
            "plot": _jsonable(self.plot) if self.plot is not None else None,
        }
        if self.wavefunctions is not None:
            data["wavefunctions"] = {
                "encoding": "complex_split",
                "real": self.wavefunctions.real.tolist(),
                "imag": self.wavefunctions.imag.tolist(),
            }
        return data

    @classmethod
    def from_dict(cls, data):
        """Build eigenstate data from a schema dictionary."""
        if data.get("schema_name") != "tb2j.magnon.eigenstates":
            raise ValueError("Unsupported magnon eigenstate schema")
        wavefunctions = data.get("wavefunctions")
        if wavefunctions is not None:
            if wavefunctions.get("encoding") != "complex_split":
                raise ValueError("Unsupported wavefunction encoding")
            wavefunctions = np.array(wavefunctions["real"]) + 1j * np.array(
                wavefunctions["imag"]
            )
        return cls(
            calculation_type=data["calculation_type"],
            kpoints=np.array(data["kpoints"]),
            energies=np.array(data["energies"]),
            wavefunctions=wavefunctions,
            weights=np.array(data["weights"])
            if data.get("weights") is not None
            else None,
            metadata=data.get("metadata", {}),
            plot=data.get("plot"),
            schema_name=data.get("schema_name", "tb2j.magnon.eigenstates"),
            schema_version=data.get("schema_version", "1.0"),
        )

    def save_json(self, filename):
        """Save magnon eigenstate data to versioned JSON."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filename):
        """Load magnon eigenstate data from versioned JSON."""
        with open(filename) as f:
            return cls.from_dict(json.load(f))

    def save_netcdf(self, filename):
        """Save magnon eigenstate data to NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError("netCDF4 is required for magnon NetCDF export") from exc

        filename = Path(filename)
        with Dataset(filename, "w") as nc:
            nkpt, nmode = self.energies.shape
            nc.createDimension("nkpt", nkpt)
            nc.createDimension("xyz", 3)
            nc.createDimension("nmode", nmode)
            nc.schema_name = self.schema_name
            nc.schema_version = self.schema_version
            nc.calculation_type = self.calculation_type
            metadata = {**self.metadata, "complex_component": ["real", "imag"]}
            nc.metadata_json = json.dumps(_jsonable(metadata))
            nc.plot_json = (
                json.dumps(_jsonable(self.plot)) if self.plot is not None else "null"
            )

            nc.createVariable("kpoints", "f8", ("nkpt", "xyz"))[:] = self.kpoints
            nc.createVariable("energies", "f8", ("nkpt", "nmode"))[:] = self.energies
            if self.weights is not None:
                nc.createVariable("weights", "f8", ("nkpt",))[:] = self.weights
            if self.wavefunctions is not None:
                nc.createDimension("ncomponent", self.wavefunctions.shape[2])
                nc.createDimension("complex", 2)
                wf = nc.createVariable(
                    "wavefunctions", "f8", ("nkpt", "nmode", "ncomponent", "complex")
                )
                wf[:, :, :, 0] = self.wavefunctions.real
                wf[:, :, :, 1] = self.wavefunctions.imag

    @classmethod
    def load_netcdf(cls, filename):
        """Load magnon eigenstate data from NetCDF4."""
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ImportError("netCDF4 is required for magnon NetCDF import") from exc

        with Dataset(filename) as nc:
            if getattr(nc, "schema_name", None) != "tb2j.magnon.eigenstates":
                raise ValueError("Unsupported magnon eigenstate schema")
            wavefunctions = None
            if "wavefunctions" in nc.variables:
                wf = nc.variables["wavefunctions"][:]
                wavefunctions = wf[:, :, :, 0] + 1j * wf[:, :, :, 1]
            weights = None
            if "weights" in nc.variables:
                weights = nc.variables["weights"][:]
            return cls(
                calculation_type=getattr(nc, "calculation_type"),
                kpoints=nc.variables["kpoints"][:],
                energies=nc.variables["energies"][:],
                wavefunctions=wavefunctions,
                weights=weights,
                metadata=json.loads(getattr(nc, "metadata_json", "{}")),
                plot=json.loads(getattr(nc, "plot_json", "null")),
                schema_name=getattr(nc, "schema_name"),
                schema_version=getattr(nc, "schema_version"),
            )

    @classmethod
    def load(cls, filename):
        """Load magnon eigenstate data from JSON or NetCDF by extension."""
        suffix = Path(filename).suffix.lower()
        if suffix in {".nc", ".netcdf"}:
            return cls.load_netcdf(filename)
        return cls.load_json(filename)

    def spin_rotation(
        self,
        kpoint_index,
        band_index,
        amplitude=1.0,
        nframes=40,
        repetitions=(1, 1, 1),
    ):
        """Generate site-resolved spin rotations for a selected eigenstate."""
        if self.wavefunctions is None:
            raise ValueError("wavefunctions are required for spin rotation generation")
        if kpoint_index < 0 or kpoint_index >= self.kpoints.shape[0]:
            raise IndexError("kpoint_index out of range")
        if band_index < 0 or band_index >= self.energies.shape[1]:
            raise IndexError("band_index out of range")

        nspin = int(self.metadata.get("nspin", self.energies.shape[1]))
        coeffs = self.wavefunctions[kpoint_index, band_index, :nspin]
        magmoms = np.asarray(
            self.metadata.get("magmoms", np.tile([0.0, 0.0, 1.0], (nspin, 1))),
            dtype=float,
        )
        positions = np.asarray(
            self.metadata.get(
                "positions",
                np.column_stack([np.arange(nspin, dtype=float), np.zeros((nspin, 2))]),
            ),
            dtype=float,
        )
        cell = np.asarray(self.metadata.get("cell", np.eye(3)), dtype=float)
        supercell = cell * np.asarray(repetitions, dtype=float)[:, None]
        symbols = self.metadata.get("symbols", ["X"] * nspin)
        atom_positions = np.asarray(
            self.metadata.get("atom_positions", positions),
            dtype=float,
        )
        atom_symbols = self.metadata.get("atom_symbols", symbols)

        base_spins = _normalize_rows(magmoms)
        e1, e2 = _transverse_axes(base_spins)
        base_amplitudes = amplitude * coeffs[:, None] * (e1 + 1j * e2)

        site_positions, reference_spins, rotation_amplitudes, site_symbols = (
            _repeat_spin_data(
                positions,
                base_spins,
                base_amplitudes,
                symbols,
                cell,
                repetitions,
                self.kpoints[kpoint_index],
            )
        )
        structure_positions, atom_symbols = _repeat_structure_data(
            atom_positions,
            atom_symbols,
            cell,
            repetitions,
        )
        frames = _build_frames(
            reference_spins,
            rotation_amplitudes,
            nframes=nframes,
            added_phase=0.0,
        )
        return SpinRotationData(
            kpoint_index=kpoint_index,
            band_index=band_index,
            kpoint=self.kpoints[kpoint_index],
            frequency=self.energies[kpoint_index, band_index],
            site_positions=site_positions,
            reference_spins=reference_spins,
            rotation_amplitudes=np.real(rotation_amplitudes),
            frames=frames,
            metadata={
                "normalization": "boson_1",
                "added_phase": 0.0,
                "bloch_phase": "exp(i 2pi q.R)",
                "amplitude": amplitude,
                "repetitions": list(repetitions),
                "cell": cell.tolist(),
                "supercell": supercell.tolist(),
                "symbols": site_symbols,
                "atom_positions": structure_positions.tolist(),
                "atom_symbols": atom_symbols,
            },
        )


@dataclass
class SpinRotationData:
    """Site-resolved spin-wave rotation frames for one selected eigenstate."""

    kpoint_index: int
    band_index: int
    kpoint: np.ndarray
    frequency: float
    site_positions: np.ndarray
    reference_spins: np.ndarray
    rotation_amplitudes: np.ndarray
    frames: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.kpoint = np.asarray(self.kpoint, dtype=float)
        self.frequency = float(self.frequency)
        self.site_positions = np.asarray(self.site_positions, dtype=float)
        self.reference_spins = np.asarray(self.reference_spins, dtype=float)
        self.rotation_amplitudes = np.asarray(self.rotation_amplitudes, dtype=float)
        self.frames = np.asarray(self.frames, dtype=float)

    def to_threejs_scene(self, display=None):
        """Return a Three.js-ready scene dictionary."""
        scene_display = {
            "vectors": True,
            "cell": True,
            "atoms": True,
            "amplitude": self.metadata.get("amplitude", 1.0),
            "speed": 1.0,
            "camera": "auto",
            "repetitions": self.metadata.get("repetitions", [1, 1, 1]),
        }
        if display is not None:
            scene_display.update(display)
        return {
            "schema_name": "tb2j.magnon.threejs_scene",
            "schema_version": "1.0",
            "mode": {
                "kpoint_index": int(self.kpoint_index),
                "band_index": int(self.band_index),
                "kpoint": self.kpoint.tolist(),
                "frequency": self.frequency,
                "frequency_unit": "eV",
            },
            "sites": {
                "positions": self.site_positions.tolist(),
                "reference_spins": self.reference_spins.tolist(),
                "rotation_amplitudes": self.rotation_amplitudes.tolist(),
                "symbols": self.metadata.get(
                    "symbols", ["X"] * len(self.site_positions)
                ),
            },
            "structure": {
                "cell": self.metadata.get(
                    "supercell",
                    self.metadata.get("cell", np.eye(3).tolist()),
                ),
                "unit_cell": self.metadata.get("cell", np.eye(3).tolist()),
                "positions": self.metadata.get(
                    "atom_positions",
                    self.site_positions.tolist(),
                ),
                "symbols": self.metadata.get(
                    "atom_symbols",
                    self.metadata.get("symbols", ["X"] * len(self.site_positions)),
                ),
            },
            "frames": self.frames.tolist(),
            "display": scene_display,
            "metadata": _jsonable(self.metadata),
        }

    def save_threejs_scene(self, filename, display=None):
        """Save Three.js-ready scene JSON."""
        with open(filename, "w") as f:
            json.dump(self.to_threejs_scene(display=display), f, indent=2)


def _normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe[:, None]


def _transverse_axes(spins):
    trial = np.tile([0.0, 0.0, 1.0], (len(spins), 1))
    parallel = np.abs(np.sum(spins * trial, axis=1)) > 0.9
    trial[parallel] = [1.0, 0.0, 0.0]
    e1 = _normalize_rows(np.cross(spins, trial))
    e2 = _normalize_rows(np.cross(spins, e1))
    return e1, e2


def _repeat_spin_data(positions, spins, amplitudes, symbols, cell, repetitions, kpoint):
    reps = [range(int(n)) for n in repetitions]
    repeated_positions = []
    repeated_spins = []
    repeated_amplitudes = []
    repeated_symbols = []
    for i in reps[0]:
        for j in reps[1]:
            for k in reps[2]:
                image = np.array([i, j, k], dtype=float)
                shift = np.array([i, j, k], dtype=float) @ cell
                phase = np.exp(2j * np.pi * np.dot(kpoint, image))
                repeated_positions.append(positions + shift)
                repeated_spins.append(spins)
                repeated_amplitudes.append(amplitudes * phase)
                repeated_symbols.extend(symbols)
    return (
        np.concatenate(repeated_positions, axis=0),
        np.concatenate(repeated_spins, axis=0),
        np.concatenate(repeated_amplitudes, axis=0),
        repeated_symbols,
    )


def _repeat_structure_data(positions, symbols, cell, repetitions):
    reps = [range(int(n)) for n in repetitions]
    repeated_positions = []
    repeated_symbols = []
    for i in reps[0]:
        for j in reps[1]:
            for k in reps[2]:
                shift = np.array([i, j, k], dtype=float) @ cell
                repeated_positions.append(positions + shift)
                repeated_symbols.extend(symbols)
    return np.concatenate(repeated_positions, axis=0), repeated_symbols


def _build_frames(reference_spins, rotation_amplitudes, nframes, added_phase):
    times = np.linspace(0.0, 2.0 * np.pi, int(nframes), endpoint=False)
    frames = []
    for time in times:
        phase = np.exp(1j * (time + added_phase))
        frames.append(reference_spins + np.real(rotation_amplitudes * phase))
    return np.array(frames)
