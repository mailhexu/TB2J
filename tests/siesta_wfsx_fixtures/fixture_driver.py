"""Generate and independently verify the small SIESTA WFSX fixture set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = "SHA256SUMS"
SCHEMA = "tb2j.siesta_wfsx_fixture/1"
K_T_EV_PER_K = 8.617333262145e-5
KPOINT_COORD_TOL = 1e-8
# Same-run KP text is printed to six significant digits; retained fixtures
# show at most 1.832899072e-7 fractional-coordinate quantization error.
KP_PRINTING_TOL = 5e-7
ENERGY_TOL_EV = 1e-5
RESIDUAL_TOL_EV = 1e-5
ORTHOGONALITY_TOL = 1e-5

CASE_INFO = {
    "scalar_nspin1": {
        "nspin": 1,
        "kind": "scalar",
        "magnetic": False,
        "weight_semantics": "scf-folded",
        "wfsx": "o2.fullBZ.WFSX",
        "fdf": "o2_scalar.fdf",
        "symmetry": {
            "real_hamiltonian": True,
            "no_soc": True,
            "no_orbital_field": True,
            "no_spin_spiral": True,
        },
    },
    "collinear_nspin2_full": {
        "nspin": 2,
        "kind": "collinear",
        "magnetic": True,
        "weight_semantics": "selected-list",
        "wfsx": "o2.selected.WFSX",
        "fdf": "../collinear_nspin2_shared/o2_collinear.fdf",
        "symmetry": {
            "real_hamiltonian": True,
            "no_soc": True,
            "no_orbital_field": True,
            "no_spin_spiral": True,
        },
    },
    "collinear_nspin2_reduced": {
        "nspin": 2,
        "kind": "collinear",
        "magnetic": True,
        "weight_semantics": "scf-folded",
        "wfsx": "o2.fullBZ.WFSX",
        "fdf": "../collinear_nspin2_shared/o2_collinear.fdf",
        "symmetry": {
            "real_hamiltonian": True,
            "no_soc": True,
            "no_orbital_field": True,
            "no_spin_spiral": True,
        },
    },
    "noncollinear_nspin4_full": {
        "nspin": 4,
        "kind": "full-spinor",
        "magnetic": True,
        "weight_semantics": "scf-folded",
        "wfsx": "o2.fullBZ.WFSX",
        "fdf": "o2_noncollinear.fdf",
        "symmetry": None,
    },
}

SOURCE_FILES = (
    "Src/writewave.F",
    "Src/siesta_analysis.F90",
    "Src/kpoint_t.F90",
    "Src/read_options.F90",
)


class FixtureError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: str, message: str) -> None:
    raise FixtureError(code, message)


def _require_finite(values: Any, field: str) -> None:
    if not np.all(np.isfinite(values)):
        _error("NONFINITE", f"non-finite values in {field}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_file(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def _write_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST and not _runtime_file(path)
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_manifest(root: Path) -> set[str]:
    manifest = root / MANIFEST
    if not manifest.is_file():
        _error("MISSING_FILE", f"required manifest is absent: {MANIFEST}")
    entries: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            _error("INVALID_MANIFEST", f"malformed line {number} in {MANIFEST}")
        relative = fields[1].lstrip("*")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in entries:
            _error(
                "INVALID_MANIFEST",
                f"unsafe or duplicate path on line {number}: {relative}",
            )
        entries[relative] = fields[0].lower()
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            _error("MISSING_FILE", f"manifest asset is absent: {relative}")
        actual = _sha256(path)
        if actual != expected:
            _error(
                "HASH_MISMATCH",
                f"SHA256 mismatch for {relative}: {actual} != {expected}",
            )
    return set(entries)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error("INVALID_METADATA", f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _error("INVALID_METADATA", f"metadata root is not an object: {path}")
    return value


def _relative_file(root: Path, case_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        _error("INVALID_METADATA", f"file path is not a string in {case_dir}")
    path = (case_dir / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        _error("INVALID_METADATA", f"file path escapes fixture root: {relative}")
    if not path.is_file():
        _error(
            "MISSING_FILE",
            f"required fixture asset is absent: {path.relative_to(root)}",
        )
    return path


def _sisl():
    try:
        import sisl
    except ImportError as exc:
        _error("DEPENDENCY_MISSING", f"fixture verification requires sisl: {exc}")
    return sisl


def _as_float_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _basis_order(basis) -> dict[str, Any]:
    return {
        "names_by_atom": [
            [str(orbital.name()) for orbital in atom.orbitals] for atom in basis
        ],
        "quantum_numbers_by_atom": [
            [
                {
                    "n": int(orbital.n),
                    "l": int(orbital.l),
                    "m": int(orbital.m),
                    "zeta": int(orbital.zeta),
                }
                for orbital in atom.orbitals
            ]
            for atom in basis
        ],
    }


def _fractional_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return (delta + 0.5) % 1.0 - 0.5


def _fractional_close(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    return bool(np.all(np.abs(_fractional_delta(first, second)) <= tolerance))


def _match_kpoint(
    point: np.ndarray,
    mesh: np.ndarray,
    allow_inversion: bool = False,
    *,
    tolerance: float = KPOINT_COORD_TOL,
):
    mesh = np.asarray(mesh, dtype=float)
    if mesh.size == 0:
        return None, False

    def matching_indices(candidate: np.ndarray) -> set[int]:
        deltas = np.asarray([_fractional_delta(candidate, item) for item in mesh])
        return {
            int(index)
            for index in np.flatnonzero(np.all(np.abs(deltas) <= tolerance, axis=1))
        }

    direct = matching_indices(np.asarray(point, dtype=float))
    inverted = (
        matching_indices(-np.asarray(point, dtype=float)) if allow_inversion else set()
    )
    matches = direct | inverted
    if len(matches) > 1:
        _error(
            "KPOINT_AMBIGUOUS",
            f"fractional k-point has multiple canonical matches: {point.tolist()}",
        )
    if not matches:
        return None, False
    index = next(iter(matches))
    return index, index not in direct


def _parse_fdf(fdf_path: Path) -> tuple[list[int], bool, float]:
    text = fdf_path.read_text(encoding="utf-8")
    mesh_match = re.search(
        r"%block\s+kgrid_Monkhorst_Pack\s*(.*?)%endblock\s+kgrid_Monkhorst_Pack",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if mesh_match is None:
        _error("INVALID_INPUT", f"missing kgrid_Monkhorst_Pack block in {fdf_path}")
    rows = []
    for row in mesh_match.group(1).splitlines():
        fields = row.split()
        if len(fields) >= 3:
            rows.append([int(value) for value in fields[:3]])
    if len(rows) != 3:
        _error("INVALID_INPUT", f"invalid three-vector k mesh in {fdf_path}")
    dimensions = [rows[index][index] for index in range(3)]
    tr_match = re.search(
        r"^\s*TimeReversalSymmetryForKpoints\s+(T|F|true|false)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if tr_match is None:
        _error("INVALID_INPUT", f"missing TimeReversalSymmetryForKpoints in {fdf_path}")
    time_reversal = tr_match.group(1).lower() in {"t", "true"}
    temperature_match = re.search(
        r"^\s*ElectronicTemperature\s+([-+0-9.eEdD]+)\s*K\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if temperature_match is None:
        _error("INVALID_INPUT", f"missing ElectronicTemperature in {fdf_path}")
    temperature = float(temperature_match.group(1).replace("D", "E").replace("d", "e"))
    for key in ("WFS.BandMin", "WFS.BandMax", "WFS.EnergyMin", "WFS.EnergyMax"):
        if re.search(rf"^\s*{re.escape(key)}\b", text, re.IGNORECASE | re.MULTILINE):
            _error(
                "INCOMPLETE_STATES",
                f"band/energy filter {key} is present in {fdf_path}",
            )
    return dimensions, time_reversal, temperature


def _parse_output(stdout_path: Path) -> dict[str, Any]:
    text = stdout_path.read_text(encoding="utf-8", errors="replace")

    def one(pattern: str, label: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match is None:
            _error(
                "INVALID_PRODUCER_OUTPUT", f"could not read {label} from {stdout_path}"
            )
        return match.group(1).strip()

    version = one(r"^Version\s*:\s*(.+)$", "SIESTA version")
    electrons = float(
        one(r"^Total number of electrons:\s*([-+0-9.eEdD]+)", "electron count")
        .replace("D", "E")
        .replace("d", "e")
    )
    total_energy = float(
        one(r"^\s*siesta:\s+Total\s*=\s*([-+0-9.eEdD]+)", "total energy")
        .replace("D", "E")
        .replace("d", "e")
    )
    output_fermi = float(
        one(r"^\s*siesta:\s+Fermi\s*=\s*([-+0-9.eEdD]+)", "printed Fermi energy")
        .replace("D", "E")
        .replace("d", "e")
    )
    iter_lines = re.findall(r"^\s*scf:\s+(\d+)\s+", text, re.IGNORECASE | re.MULTILINE)
    criterion_match = re.search(r"^SCF Convergence by\s+(.+)$", text, re.MULTILINE)
    if not iter_lines or criterion_match is None:
        _error(
            "INVALID_PRODUCER_OUTPUT",
            f"missing converged SCF evidence in {stdout_path}",
        )
    moments = re.findall(
        r"spin moment:\s*\{S\}\s*,\s*\|S\|\s*=\s*\{\s*([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
        text,
        re.IGNORECASE,
    )
    moment = (
        [float(value.replace("D", "E").replace("d", "e")) for value in moments[-1]]
        if moments
        else [0.0, 0.0, 0.0]
    )
    return {
        "version": version,
        "architecture": one(r"^Architecture\s*:\s*(.+)$", "architecture"),
        "compiler": one(r"^Compiler version:\s*(.+)$", "compiler"),
        "compiler_flags": one(r"^Compiler flags\s*:\s*(.+)$", "compiler flags"),
        "parallelizations": one(r"^Parallelisations:\s*(.+)$", "parallelization"),
        "electron_count": electrons,
        "total_energy_ev": total_energy,
        "printed_fermi_ev": output_fermi,
        "iteration_count": int(iter_lines[-1]),
        "convergence_criterion": criterion_match.group(1).strip(),
        "final_spin_moment_mu_b": moment,
    }


def _occupation_count(
    eigenvalues: np.ndarray, kp_weights: np.ndarray, nspin: int, temp_k: float
) -> float:
    if temp_k <= 0:
        _error(
            "INVALID_METADATA",
            "ElectronicTemperature must be positive for recorded occupations",
        )
    exponent = np.clip(eigenvalues / (K_T_EV_PER_K * temp_k), -700.0, 700.0)
    occupations = 1.0 / (1.0 + np.exp(exponent))
    if nspin == 1:
        return float(2.0 * np.sum(kp_weights[:, None] * occupations[0]))
    if nspin == 2:
        return float(np.sum(kp_weights[None, :, None] * occupations))
    return float(np.sum(kp_weights[:, None] * occupations[0]))


def _read_oracle(root: Path, case_dir: Path, files: dict[str, str]):
    sisl = _sisl()
    paths = {
        name: _relative_file(root, case_dir, relative)
        for name, relative in files.items()
        if name in {"wfsx", "hsx", "eig", "kp", "stdout", "fdf", "orb_indx", "pseudo"}
    }
    hsx_sile = sisl.get_sile(str(paths["hsx"]))
    geom = hsx_sile.read_geometry()
    hsx_basis = hsx_sile.read_basis()
    kp_points, kp_weights = sisl.get_sile(str(paths["kp"])).read_data(geom.lattice)
    eig_sile = sisl.get_sile(str(paths["eig"]))
    eig_values = np.asarray(eig_sile.read_data(), dtype=float)
    fermi_ev = float(eig_sile.read_fermi_level())
    wfsx = sisl.get_sile(str(paths["wfsx"]), lattice=geom.lattice)
    wfsx_basis = wfsx.read_basis()
    sizes = wfsx.read_sizes()
    records = list(wfsx.yield_eigenstate())
    for field, values in (
        ("KP coordinates", kp_points),
        ("KP weights", kp_weights),
        ("EIG energies", eig_values),
        ("Fermi energy", fermi_ev),
    ):
        _require_finite(values, field)
    for index, record in enumerate(records):
        for field, values in (
            ("coordinates", record.info["k"]),
            ("energies", record.c),
            ("coefficients", record.state),
            ("weight", record.info["weight"]),
        ):
            _require_finite(values, f"WFSX group {index} {field}")
    output = _parse_output(paths["stdout"])
    dimensions, time_reversal, temp_k = _parse_fdf(paths["fdf"])
    return {
        "sisl": sisl,
        "paths": paths,
        "geom": geom,
        "kp_points": np.asarray(kp_points, dtype=float),
        "kp_weights": np.asarray(kp_weights, dtype=float),
        "eig_values": eig_values,
        "fermi_ev": fermi_ev,
        "wfsx": wfsx,
        "sizes": sizes,
        "records": records,
        "hsx_basis": hsx_basis,
        "wfsx_basis": wfsx_basis,
        "output": output,
        "mesh_dimensions": dimensions,
        "time_reversal": time_reversal,
        "temperature_k": temp_k,
    }


def _oracle_metadata(data: dict[str, Any], nspin: int) -> dict[str, float]:
    ham = data["sisl"].get_sile(str(data["paths"]["hsx"])).read_hamiltonian()
    max_residual = 0.0
    max_orthogonality = 0.0
    max_eig_difference = 0.0
    mesh = data["kp_points"]
    eig_values = data["eig_values"]
    for record in data["records"]:
        info = record.info
        kpoint = np.asarray(info["k"], dtype=float)
        spin_id = int(info.get("spin", 0))
        index, inverted = _match_kpoint(
            kpoint, mesh, allow_inversion=nspin == 2, tolerance=KP_PRINTING_TOL
        )
        eig_spin = spin_id
        if eig_values.shape[0] == 1:
            eig_spin = 0
        wfsx_relative = np.asarray(record.c, dtype=float) - data["fermi_ev"]
        eig_relative = eig_values[eig_spin, index]
        if wfsx_relative.shape != eig_relative.shape:
            _error(
                "INCOMPLETE_STATES",
                f"WFSX/EIG state counts differ at k={kpoint.tolist()}",
            )
        _require_finite(wfsx_relative - eig_relative, "WFSX/EIG energy difference")
        max_eig_difference = max(
            max_eig_difference,
            float(np.max(np.abs(wfsx_relative - eig_relative))),
        )
        if max_eig_difference > ENERGY_TOL_EV:
            _error(
                "ENERGY_MISMATCH",
                f"WFSX/EIG absolute-vs-relative energy mismatch at k={kpoint.tolist()}",
            )
        if nspin == 2:
            h = ham.Hk(kpoint, spin=spin_id, gauge="r", format="array")
        else:
            h = ham.Hk(kpoint, gauge="r", format="array")
        overlap = ham.Sk(kpoint, gauge="r", format="array")
        _require_finite(h, "H(k)")
        _require_finite(overlap, "S(k)")
        vectors = np.asarray(record.state).T
        if h.shape != overlap.shape or vectors.shape[0] != h.shape[0]:
            _error(
                "BASIS_MISMATCH", f"WFSX/H/S dimensions differ at k={kpoint.tolist()}"
            )
        residual = h @ vectors - (overlap @ vectors) * wfsx_relative[None, :]
        _require_finite(residual, "generalized eigenpair residual")
        max_residual = max(max_residual, float(np.max(np.abs(residual))))
        orthogonality = vectors.conj().T @ overlap @ vectors - np.eye(vectors.shape[1])
        _require_finite(orthogonality, "S-orthogonality error")
        max_orthogonality = max(max_orthogonality, float(np.max(np.abs(orthogonality))))
    if max_residual > RESIDUAL_TOL_EV:
        _error(
            "EIGENPAIR_RESIDUAL",
            f"generalized eigenpair residual {max_residual:.6g} eV exceeds {RESIDUAL_TOL_EV:g} eV",
        )
    if max_orthogonality > ORTHOGONALITY_TOL:
        _error(
            "ORTHOGONALITY_ERROR",
            f"S-orthogonality error {max_orthogonality:.6g} exceeds {ORTHOGONALITY_TOL:g}",
        )
    return {
        "max_generalized_eigenpair_residual_ev": max_residual,
        "max_s_orthogonality_error": max_orthogonality,
        "max_wfsx_vs_eig_relative_energy_difference_ev": max_eig_difference,
    }


def _metadata_for_case(
    fixture_id: str,
    case_dir: Path,
    run_id: str,
    producer: dict[str, Any],
    files: dict[str, str],
) -> dict[str, Any]:
    spec = CASE_INFO[fixture_id]
    data = _read_oracle(case_dir.parent, case_dir, files)
    nspin, no_u, nk, gamma = data["sizes"]
    if int(nspin) != spec["nspin"]:
        _error(
            "SPIN_MISMATCH",
            f"{fixture_id} WFSX nspin={nspin}; expected {spec['nspin']}",
        )
    points: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    unique: list[np.ndarray] = []
    for group_index, record in enumerate(data["records"]):
        info = record.info
        kpoint = np.asarray(info["k"], dtype=float)
        point_index, _ = (
            _match_kpoint(kpoint, np.asarray(unique), allow_inversion=False)
            if unique
            else (None, False)
        )
        if point_index is None:
            point_index = len(unique)
            unique.append(kpoint)
            points.append(
                {
                    "fractional": _as_float_list(kpoint),
                    "source_weight": float(info["weight"]),
                }
            )
        groups.append(
            {
                "group_index": group_index,
                "kpoint_index": int(point_index),
                "spin_id": int(info["spin"]) if "spin" in info else None,
                "state_ids": [
                    int(value) for value in np.asarray(info["index"]).tolist()
                ],
                "state_count": int(len(record.c)),
                "energy_dtype": str(np.asarray(record.c).dtype),
                "coefficient_dtype": str(np.asarray(record.state).dtype),
                "coefficient_shape": [
                    int(value) for value in np.asarray(record.state).shape
                ],
                "gauge": str(info.get("gauge", "unknown")),
            }
        )
    kp_records = [
        {"fractional": _as_float_list(kpoint), "weight": float(weight)}
        for kpoint, weight in zip(data["kp_points"], data["kp_weights"])
    ]
    cell = np.asarray(data["geom"].cell, dtype=float)
    positions = np.asarray(data["geom"].xyz, dtype=float)
    symbols = [str(atom.symbol) for atom in data["geom"].atoms]
    hsx_basis = _basis_order(data["hsx_basis"])
    wfsx_basis = _basis_order(data["wfsx_basis"])
    output = data["output"]
    temperature = data["temperature_k"]
    electron_count = _occupation_count(
        data["eig_values"], data["kp_weights"], spec["nspin"], temperature
    )
    metadata = {
        "schema": SCHEMA,
        "fixture_id": fixture_id,
        "producer_run_id": run_id,
        "spin": {
            "nspin": int(nspin),
            "kind": spec["kind"],
            "magnetic": spec["magnetic"],
            "mode": "non-polarized"
            if nspin == 1
            else "collinear"
            if nspin == 2
            else "non-collinear",
            "scalar_symmetry": spec["symmetry"],
            "channel_index_origin": 0,
            "spinor_components": 2 if nspin == 4 else 1,
            "scalar_reduction": nspin != 4,
            "soc": False,
        },
        "structure": {
            "cell_angstrom": np.asarray(cell).tolist(),
            "atom_symbols": symbols,
            "positions_angstrom": np.asarray(positions).tolist(),
            "basis_orbitals_by_atom": hsx_basis["names_by_atom"],
            "wfsx_basis_orbitals_by_atom": wfsx_basis["names_by_atom"],
            "basis_quantum_numbers_by_atom": hsx_basis["quantum_numbers_by_atom"],
            "wfsx_basis_quantum_numbers_by_atom": wfsx_basis["quantum_numbers_by_atom"],
            "basis_size": int(data["geom"].no),
        },
        "mesh": {
            "dimensions": data["mesh_dimensions"],
            "time_reversal_symmetry_for_kpoints": data["time_reversal"],
            "scf_kpoint_count": int(len(data["kp_points"])),
        },
        "electrons": {
            "total_electrons": float(output["electron_count"]),
            "total_electrons_source": "same-run SIESTA stdout: Total number of electrons",
            "fermi_energy_ev": float(data["fermi_ev"]),
            "fermi_energy_source": "first record of same-run EIG (absolute eV)",
            "electronic_temperature_k": float(temperature),
            "temperature_source": "ElectronicTemperature from same-run FDF",
            "occupation_convention": "Fermi-Dirac",
            "occupation_count_from_eig_and_kp": electron_count,
            "occupation_count_method": "sisl EIG relative to E_F, same-run KP integration weights, 100 K Fermi-Dirac; scalar nspin=1 includes spin degeneracy 2",
        },
        "energy_conventions": {
            "wfsx_eigenvalues": "absolute eV",
            "eig_eigenvalues": "relative to E_F (sisl EIG read_data convention)",
            "ordinary_hsx_hamiltonian": "sisl HSX H(k) energies relative to E_F",
            "wfsx_minus_fermi_matches_eig": True,
        },
        "kpoints": {
            "coordinate_system": "fractional reciprocal-lattice coordinates from sisl; WFSX binary values converted once using the same-run cell",
            "weight_semantics": spec["weight_semantics"],
            "source_weight_note": "selected-list weights are raw WFSX values 1.0 and are not Brillouin-zone integration weights; scf-folded weights come from the SCF export and sum to one over distinct k-points",
            "points": points,
            "integration_mesh": {
                "source": "same-run KP file",
                "normalized": True,
                "points": kp_records,
            },
            "fold_map": [],
        },
        "states": {
            "no_u": int(no_u),
            "nk": int(nk),
            "gamma_only": bool(gamma),
            "all_states_retained": True,
            "state_ids_zero_based": True,
            "groups": groups,
        },
        "scf": {
            "converged": True,
            "convergence_criterion": output["convergence_criterion"],
            "iteration_count": output["iteration_count"],
            "total_energy_ev": output["total_energy_ev"],
            "final_spin_moment_mu_b": output["final_spin_moment_mu_b"],
            "fermi_energy_printed_ev": output["printed_fermi_ev"],
            "free_energy_only_convergence": "FreeE" in output["convergence_criterion"],
        },
        "producer": producer,
        "files": files,
    }
    if fixture_id == "noncollinear_nspin4_full":
        metadata["ordinary_tb2j"] = {
            "evidence_scope": "Dated 2026-10-30 observations for this retained reference bundle and the source commits cited here; ordinary-parser comparisons are not recomputed by fixture_driver.py generate, which runs SIESTA and independent direct-sisl checks.",
            "historical_pre_fix_observation": {
                "hamiltonio_baseline": "829fc0f",
                "error": "IndexError: index 5 is out of bounds for axis 3 with size 5",
                "hsx_csr_shape": [18, 54, 5],
            },
            "post_fix_commit": {
                "hamiltonio_commit": "e806c29",
                "review_base": "0e1551c",
                "quick_task": "tb2j/quick-task/fix-four-component-noncollinear-siesta-h-s-parsing",
                "source_state": "committed as e806c29; independent review and successful qualified checked finish passed, and two-axis test-hook cleanup review approved with the seven-line numerical branch unchanged",
                "authorization": "User selected Authorize parser fix",
                "tb2j_module": "/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/TB2J/__init__.py",
                "hamiltonio_module": "/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio/HamiltonIO/__init__.py",
                "operator_smoke": {
                    "fractional_k_z": [0.0, 0.2, -0.2, 0.4, -0.4],
                    "fixture_spin_modes": ["nspin1", "nspin2", "nspin4"],
                    "max_hamiltonian_difference_ev": 3.469446951953614e-18,
                    "max_overlap_difference": 4.235164736271502e-22,
                },
                "nspin4_full_spectrum_vs_wfsx": {
                    "states_compared": 180,
                    "max_absolute_difference_ev": 3.7032847899354238e-09,
                },
                "scope": "Four-component noncollinear H/S parsing only; no SOC or HSX-offset change.",
            },
        }
    metadata["independent_checks"] = _oracle_metadata(data, spec["nspin"])
    return metadata


def _fold_map(
    full_points: list[dict[str, Any]], folded_points: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    representatives = np.asarray(
        [point["fractional"] for point in folded_points], dtype=float
    )
    weights = [float(point["source_weight"]) for point in folded_points]
    mapping = []
    for point in full_points:
        selected = np.asarray(point["fractional"], dtype=float)
        index, inverted = _match_kpoint(selected, representatives, allow_inversion=True)
        if index is None:
            _error(
                "KPOINT_MISMATCH",
                f"full selected k-point has no inversion-reduced representative: {selected.tolist()}",
            )
        mapping.append(
            {
                "selected_fractional": selected.tolist(),
                "integration_representative_fractional": representatives[
                    index
                ].tolist(),
                "inverted": bool(inverted),
                "selected_source_weight": float(point["source_weight"]),
                "scf_folded_weight": weights[index],
            }
        )
    return mapping


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_provenance(metadata_by_case: dict[str, dict[str, Any]]) -> str:
    first = next(iter(metadata_by_case.values()))["producer"]
    lines = [
        "# SIESTA WFSX exchange reference fixtures",
        "",
        "This directory contains small, complete producer outputs for independent WFSX checks. `fixture_driver.py verify --root .` checks hashes, metadata, complete state IDs, spin/k-point/weight/energy conventions, EIG agreement, and generalized eigenpair residuals using sisl directly. It is not a WFSX importer and does not reconstruct an ordinary Hamiltonian from WFSX.",
        "",
        "## Producer and reproduction",
        "",
        f"- SIESTA version: `{first['siesta_version']}`; executable SHA-256: `{first['executable_sha256']}`.",
        f"- SIESTA executable observed at: `{first['executable']}`.",
        f"- Reported build: {first['architecture']}; {first['compiler']}; `{first['compiler_flags']}`; {first['parallelizations']}, one MPI rank.",
        f"- Source checkout observed separately from the binary version: `{first['source_checkout']['head']}` at `{first['source_checkout']['root']}`.",
        "- The binary's version string and the source checkout HEAD are recorded separately; the checkout HEAD is not claimed as the binary's exact build commit.",
        f"- sisl ingestion API version: `{first['sisl_version']}`.",
        "- All runs used `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no MPI launcher was used (one rank).",
        "- Reproduce in an empty destination with the exact executable and source checkout:",
        "  `rtk env PYTHONPATH=/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j:/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio /home/hexu/projects/myenvs/mydev/bin/python /home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/fixture_driver.py generate --siesta /home/hexu/projects/siesta_dev/siesta/_build/build/Src/siesta --source-root /home/hexu/projects/siesta_dev/siesta --output /path/to/empty-output`.",
        "  The generator refuses to overwrite a non-empty output path, runs scalar, collinear (one run for both exports), and noncollinear inputs, then writes metadata and SHA256SUMS.",
        "",
        "## Physics, convergence, and export semantics",
        "",
        "All cases use the same O2 chain geometry (7 × 7 × 4 Å cell; O positions (2.895, 3.5, 2.0) and (4.105, 3.5, 2.0) Å), PBE/SZP, 150 Ry mesh cutoff, 100 K Fermi-Dirac occupations, and a 1 × 1 × 5 z mesh. The retained `.ORB_INDX` files are unmodified same-run SIESTA outputs; they carry the supercell range needed by sisl/HamiltonIO geometry loading.",
        "",
        "Every SCF converged in nine iterations by the FreeE criterion only (`SCF.FreeE.Tolerance = 1e-8 eV`). No stricter H/DM stationarity or torque claim is made. Magnetic moments below are the final SIESTA moments, not merely initialization values.",
        "",
        "The magnetic nspin=2 FDF has `TimeReversalSymmetryForKpoints true`, `COOP.Write T`, and an explicit `WaveFuncKPoints` list at 0, ±0.2, ±0.4. In the single converged run, `o2.fullBZ.WFSX` is the three-point SCF folded export (`scf-folded`, weights 0.2/0.4/0.4) and `o2.selected.WFSX` is the five-point explicit list (`selected-list`, raw WFSX weights 1.0 each, not integration weights). The two fixture metadata records share one `producer_run_id` and the exact same `.HSX`, `.EIG`, `.KP`, `.ORB_INDX`, `.out`, and FDF ordinary references. The fold map is explicit in the full-list metadata.",
        "",
        "The scalar nspin=1 fixture declares real-Hamiltonian, no-SOC, no-orbital-field, and no-spin-spiral symmetry. The magnetic nspin=2 fixture declares the same scalar symmetry conditions while retaining its two spin channels. The nspin=4 fixture is a full noncollinear spinor with no scalar reduction and no SOC; it is not relabeled as SOC nspin=8.",
        "",
        "## Independent checks observed",
        "",
        "sisl 0.16.2 reads orbital order from same-run `HSX.read_basis()` and `WFSX.read_basis()`, then reads states through `sisl.get_sile(path, lattice=geom.lattice).read_sizes()` and `yield_eigenstate()`. WFSX record indices are zero-based, coefficient arrays are complex64, energies are float64 and absolute eV, and `info['k']` is fractional reciprocal-lattice coordinates after one lattice conversion. `o2.EIG` values read by sisl are relative to E_F; subtracting the exact EIG Fermi record from WFSX energies matches all EIG states. Metadata retains every state-ID vector, spin record, basis/site order, raw/folded k weights, and source dtype.",
        "",
        "Direct sisl H/S generalized-eigenpair checks use same-run HSX `H(k)`, `S(k)` and WFSX vectors in orbital gauge. The verifier requires max residual ≤1e-5 eV and max S-orthogonality defect ≤1e-5. The observed maxima are stored per fixture in `fixture.json`; no WFSX-based H/S reconstruction is used.",
        "WFSX metadata coordinates and canonical full-to-reduced mapping use wrapped componentwise absolute tolerance 1e-8 and reject ambiguous representatives. Same-run `.KP` coordinates are printed to six significant digits, so only WFSX-to-`.KP` lookup uses `KP_PRINTING_TOL = 5e-7`; the maximum measured fractional discrepancy in retained fixtures is 1.832899072e-7. This print-quantization tolerance does not apply to WFSX metadata or canonical fold-map comparisons.",
        "",
        "The ordinary-parser values below are dated 2026-10-30 observations for this retained reference bundle and the cited TW/HW source commits. `fixture_driver.py generate` does not recompute ordinary TB2J parser comparisons; it runs SIESTA and independent direct-sisl checks.",
        "The ordinary parser was exercised with `TB2J.__file__` `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/TB2J/__init__.py` (TW baseline 33c5624) and `HamiltonIO.__file__` `/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio/HamiltonIO/__init__.py`. With retained same-run `.ORB_INDX`, scalar and nspin=2 ordinary spectra on approved HW baseline 829fc0f agree with WFSX to 2.339746174584434e-9 eV and 3.371603440882609e-9 eV, respectively.",
        "Historical nspin=4 behavior on HW baseline 829fc0f was `IndexError: index 5 is out of bounds for axis 3 with size 5` in `convert_sisl_to_spinorham` for HSX CSR shape (18, 54, 5). After separate user authorization, the four-component noncollinear H/S parser fix was committed as e806c29 based on review base 0e1551c; independent review and successful qualified checked finish passed, and the test-hook cleanup review was approved on both axes with the seven-line numerical branch unchanged.",
        "On that unchanged numerical branch, actual ordinary operator smoke for nspin=1, nspin=2, and nspin=4 at fractional k_z = 0, ±0.2, ±0.4 matched native parser H/S with maximum absolute H difference 3.469446951953614e-18 eV and S difference 4.235164736271502e-22. The nspin=4 WFSX-versus-ordinary full-spectrum comparison covered 180 states with maximum difference 3.7032847899354238e-9 eV. These are retained-bundle observations, not freshly computed by `generate`. The repair is limited to four-component noncollinear H/S parsing; no SOC or HSX-offset change was made.",
        "",
        "## Pseudopotential source and redistribution note",
        "",
        "`O.psf` is byte-identical to the already-existing TB2J O2 test fixture and to `siesta/Tests/Pseudos/O.psf` (SHA-256 `d3ecdc324e91f272a7d1b9d7747cd6f08c81899c37d0500256a0f2fc61291c71`). The upstream `Tests/Pseudos/README` says only that the directory contains pseudopotential files for SIESTA tests. No separate per-pseudopotential license was found, so this note makes no license/redistribution claim based solely on SIESTA's source-code GPL. The fixture retains the exact pre-existing test asset without modification.",
        "",
        "`SHA256SUMS` pins the retained inputs, producer outputs, metadata, and this provenance record. Oracle H/S files are validation-only; they are not production-mode metadata inputs.",
        "",
    ]
    for fixture_id, metadata in metadata_by_case.items():
        electrons = metadata["electrons"]
        scf = metadata["scf"]
        sizes = metadata["states"]
        checks = metadata["independent_checks"]
        points = metadata["kpoints"]["points"]
        lines.append(
            f"- `{fixture_id}`: run `{metadata['producer_run_id']}`, nspin={metadata['spin']['nspin']}, "
            f"EF={electrons['fermi_energy_ev']:.8f} eV, N={electrons['total_electrons']:.6f}, "
            f"moment=({scf['final_spin_moment_mu_b'][0]:.5f}, {scf['final_spin_moment_mu_b'][1]:.5f}, "
            f"{scf['final_spin_moment_mu_b'][2]:.5f}) μB, {scf['iteration_count']} FreeE iterations; "
            f"{len(points)} WFSX k points, {len(sizes['groups'])} k/spin groups, {sizes['no_u']} unit-cell orbitals, "
            f"coefficient dtype {sizes['groups'][0]['coefficient_dtype']}, "
            f"max residual {checks['max_generalized_eigenpair_residual_ev']:.9g} eV, "
            f"max S defect {checks['max_s_orthogonality_error']:.9g}."
        )
    lines.append("")
    return "\n".join(lines)


def _source_producer(
    siesta_path: Path,
    source_root: Path,
    run_dir: Path,
    output: dict[str, Any],
    sisl_version: str,
    env: dict[str, str],
    fdf_name: str,
) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _error("PRODUCER_PROVENANCE", f"cannot identify SIESTA source checkout: {exc}")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            _error("PRODUCER_PROVENANCE", f"required audited source is absent: {path}")
        source_hashes[relative] = _sha256(path)
    version_match = re.search(r"-g([0-9a-f]+)", output["version"])
    return {
        "siesta_version": output["version"],
        "binary_version_revision_abbreviation": version_match.group(1)
        if version_match
        else None,
        "executable": str(siesta_path),
        "executable_sha256": _sha256(siesta_path),
        "architecture": output["architecture"],
        "compiler": output["compiler"],
        "compiler_flags": output["compiler_flags"],
        "parallelizations": output["parallelizations"],
        "mpi_ranks": 1,
        "python_version": sys.version.split()[0],
        "working_directory": str(run_dir),
        "environment": {
            key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "argv": [str(siesta_path)],
        "stdin_fdf": fdf_name,
        "stdout_file": "o2.out",
        "source_checkout": {
            "root": str(source_root),
            "head": head,
            "source_sha256": source_hashes,
            "head_is_not_claimed_as_binary_build_commit": True,
        },
        "sisl_version": sisl_version,
    }


def _copy_run_assets(
    run_dir: Path, package: Path, fixture: str, files: tuple[str, ...]
) -> None:
    target = package / fixture
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        source = run_dir / name
        if not source.is_file():
            _error(
                "PRODUCER_OUTPUT_MISSING",
                f"SIESTA did not create required output {source}",
            )
        shutil.copy2(source, target / name)


def _run_siesta(
    executable: Path,
    source_dir: Path,
    scratch: Path,
    fdf_name: str,
    env: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    run_dir = scratch / fdf_name.removesuffix(".fdf")
    run_dir.mkdir(parents=True)
    shutil.copy2(source_dir / fdf_name, run_dir / fdf_name)
    shutil.copy2(source_dir.parent / "O.psf", run_dir / "O.psf")
    with (run_dir / fdf_name).open("r", encoding="utf-8") as stdin, (
        run_dir / "o2.out"
    ).open("w", encoding="utf-8") as stdout:
        result = subprocess.run(
            [str(executable)],
            cwd=run_dir,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        _error(
            "PRODUCER_FAILED",
            f"SIESTA exited {result.returncode}; inspect {run_dir / 'o2.out'}",
        )
    required = ("o2.HSX", "o2.EIG", "o2.KP", "o2.ORB_INDX", "o2.out")
    for name in required:
        if not (run_dir / name).is_file():
            _error(
                "PRODUCER_OUTPUT_MISSING",
                f"successful SIESTA run omitted {run_dir / name}",
            )
    output = _parse_output(run_dir / "o2.out")
    return run_dir, output


def _generate(args: argparse.Namespace) -> None:
    executable = Path(args.siesta).resolve()
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        _error("INVALID_PRODUCER", f"SIESTA executable is not executable: {executable}")
    if not source_root.is_dir():
        _error(
            "INVALID_PRODUCER", f"SIESTA source root is not a directory: {source_root}"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        _error(
            "OUTPUT_NOT_EMPTY", f"refusing to overwrite non-empty output path: {output}"
        )
    source_inputs = {
        "scalar_nspin1": ROOT / "scalar_nspin1" / "o2_scalar.fdf",
        "collinear_nspin2_shared": ROOT
        / "collinear_nspin2_shared"
        / "o2_collinear.fdf",
        "noncollinear_nspin4_full": ROOT
        / "noncollinear_nspin4_full"
        / "o2_noncollinear.fdf",
    }
    for path in (*source_inputs.values(), ROOT / "O.psf"):
        if not path.is_file():
            _error("MISSING_FILE", f"generation input is missing: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import sisl
    except ImportError as exc:
        _error("DEPENDENCY_MISSING", f"fixture generation requires sisl: {exc}")
    with tempfile.TemporaryDirectory(
        prefix="siesta-wfsx-generate-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        shutil.copy2(ROOT / "O.psf", staging / "O.psf")
        for name in (
            "scalar_nspin1",
            "collinear_nspin2_full",
            "collinear_nspin2_reduced",
            "collinear_nspin2_shared",
            "noncollinear_nspin4_full",
        ):
            (staging / name).mkdir()
        shutil.copy2(
            source_inputs["scalar_nspin1"], staging / "scalar_nspin1" / "o2_scalar.fdf"
        )
        shutil.copy2(
            source_inputs["collinear_nspin2_shared"],
            staging / "collinear_nspin2_shared" / "o2_collinear.fdf",
        )
        shutil.copy2(
            source_inputs["noncollinear_nspin4_full"],
            staging / "noncollinear_nspin4_full" / "o2_noncollinear.fdf",
        )
        scratch = Path(temporary) / "runs"
        scratch.mkdir()
        producer_data: dict[str, tuple[dict[str, Any], Path, str]] = {}
        run_specs = (
            ("scalar_nspin1", "scalar_nspin1", "o2_scalar.fdf"),
            ("collinear_nspin2", "collinear_nspin2_shared", "o2_collinear.fdf"),
            (
                "noncollinear_nspin4_full",
                "noncollinear_nspin4_full",
                "o2_noncollinear.fdf",
            ),
        )
        for run_label, input_dir, fdf_name in run_specs:
            run_id = f"{stamp}-{run_label}"
            run_dir, output_info = _run_siesta(
                executable,
                staging / input_dir,
                scratch,
                fdf_name,
                env,
            )
            producer = _source_producer(
                executable,
                source_root,
                run_dir,
                output_info,
                str(sisl.__version__),
                env,
                fdf_name,
            )
            producer_data[run_label] = (producer, run_dir, run_id)
        scalar_run = producer_data["scalar_nspin1"][1]
        _copy_run_assets(
            scalar_run,
            staging,
            "scalar_nspin1",
            ("o2.HSX", "o2.EIG", "o2.KP", "o2.ORB_INDX", "o2.out", "o2.fullBZ.WFSX"),
        )
        collinear_run = producer_data["collinear_nspin2"][1]
        _copy_run_assets(
            collinear_run,
            staging,
            "collinear_nspin2_shared",
            ("o2.HSX", "o2.EIG", "o2.KP", "o2.ORB_INDX", "o2.out"),
        )
        shutil.copy2(
            collinear_run / "o2.selected.WFSX",
            staging / "collinear_nspin2_full" / "o2.selected.WFSX",
        )
        shutil.copy2(
            collinear_run / "o2.fullBZ.WFSX",
            staging / "collinear_nspin2_reduced" / "o2.fullBZ.WFSX",
        )
        noncollinear_run = producer_data["noncollinear_nspin4_full"][1]
        _copy_run_assets(
            noncollinear_run,
            staging,
            "noncollinear_nspin4_full",
            ("o2.HSX", "o2.EIG", "o2.KP", "o2.ORB_INDX", "o2.out", "o2.fullBZ.WFSX"),
        )
        maps = {
            "scalar_nspin1": {
                "fdf": "o2_scalar.fdf",
                "pseudo": "../O.psf",
                "wfsx": "o2.fullBZ.WFSX",
                "hsx": "o2.HSX",
                "eig": "o2.EIG",
                "kp": "o2.KP",
                "orb_indx": "o2.ORB_INDX",
                "stdout": "o2.out",
            },
            "collinear_nspin2_full": {
                "fdf": "../collinear_nspin2_shared/o2_collinear.fdf",
                "pseudo": "../O.psf",
                "wfsx": "o2.selected.WFSX",
                "hsx": "../collinear_nspin2_shared/o2.HSX",
                "eig": "../collinear_nspin2_shared/o2.EIG",
                "kp": "../collinear_nspin2_shared/o2.KP",
                "orb_indx": "../collinear_nspin2_shared/o2.ORB_INDX",
                "stdout": "../collinear_nspin2_shared/o2.out",
            },
            "collinear_nspin2_reduced": {
                "fdf": "../collinear_nspin2_shared/o2_collinear.fdf",
                "pseudo": "../O.psf",
                "wfsx": "o2.fullBZ.WFSX",
                "hsx": "../collinear_nspin2_shared/o2.HSX",
                "eig": "../collinear_nspin2_shared/o2.EIG",
                "kp": "../collinear_nspin2_shared/o2.KP",
                "orb_indx": "../collinear_nspin2_shared/o2.ORB_INDX",
                "stdout": "../collinear_nspin2_shared/o2.out",
            },
            "noncollinear_nspin4_full": {
                "fdf": "o2_noncollinear.fdf",
                "pseudo": "../O.psf",
                "wfsx": "o2.fullBZ.WFSX",
                "hsx": "o2.HSX",
                "eig": "o2.EIG",
                "kp": "o2.KP",
                "orb_indx": "o2.ORB_INDX",
                "stdout": "o2.out",
            },
        }
        metadata_by_case = {}
        for fixture_id, files in maps.items():
            if fixture_id == "scalar_nspin1":
                producer, _, run_id = producer_data["scalar_nspin1"]
            elif fixture_id.startswith("collinear"):
                producer, _, run_id = producer_data["collinear_nspin2"]
            else:
                producer, _, run_id = producer_data["noncollinear_nspin4_full"]
            metadata = _metadata_for_case(
                fixture_id,
                staging / fixture_id,
                run_id,
                producer,
                files,
            )
            metadata_by_case[fixture_id] = metadata
            _write_json(staging / fixture_id / "fixture.json", metadata)
        full = metadata_by_case["collinear_nspin2_full"]
        reduced = metadata_by_case["collinear_nspin2_reduced"]
        full["kpoints"]["fold_map"] = _fold_map(
            full["kpoints"]["points"],
            reduced["kpoints"]["points"],
        )
        reduced["kpoints"]["fold_map"] = list(full["kpoints"]["fold_map"])
        for fixture_id, metadata in metadata_by_case.items():
            _write_json(staging / fixture_id / "fixture.json", metadata)
        provenance = _build_provenance(metadata_by_case)
        (staging / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
        shutil.copy2(ROOT / Path(__file__).name, staging / Path(__file__).name)
        _write_manifest(staging)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    print(f"Generated and hash-pinned fixtures at {output}")


def _expect(condition: bool, code: str, message: str) -> None:
    if not condition:
        _error(code, message)


def _verify_k_coverage(data: dict[str, Any], selected: bool, fixture_id: str) -> None:
    """Check each spin against the independent SCF mesh, not the WFSX header."""
    expected = list(data["kp_points"])
    full_count = math.prod(data["mesh_dimensions"])
    if selected:
        expected = []
        for point, weight in zip(data["kp_points"], data["kp_weights"]):
            multiplicity = int(round(float(weight) * full_count))
            if not math.isclose(
                float(weight) * full_count, multiplicity, abs_tol=1e-8, rel_tol=0
            ) or multiplicity not in (1, 2):
                _error("WEIGHT_MISMATCH", f"invalid SCF multiplicity in {fixture_id}")
            expected.append(point)
            if multiplicity == 2:
                if (
                    int(data["sizes"].nspin) != 2
                    or not data["time_reversal"]
                    or _fractional_close(point, -point, KP_PRINTING_TOL)
                ):
                    _error("WEIGHT_MISMATCH", f"invalid inversion pair in {fixture_id}")
                expected.append(-point)
        if len(expected) != full_count:
            _error(
                "INCOMPLETE_KPOINTS",
                f"SCF mesh does not cover FDF grid in {fixture_id}",
            )
    elif not data["time_reversal"] and len(expected) != full_count:
        _error("INCOMPLETE_KPOINTS", f"SCF full mesh is incomplete in {fixture_id}")
    for index, point in enumerate(expected):
        if any(
            _fractional_close(point, previous, KP_PRINTING_TOL)
            for previous in expected[:index]
        ):
            _error(
                "DUPLICATE_KPOINT",
                f"SCF mesh contains duplicate coordinates in {fixture_id}",
            )
    channels = (0, 1) if int(data["sizes"].nspin) == 2 else (0,)
    seen = {spin: set() for spin in channels}
    for record in data["records"]:
        spin = int(record.info.get("spin", 0))
        if spin not in seen:
            _error("SPIN_MISMATCH", f"unexpected WFSX spin in {fixture_id}")
        index, _ = _match_kpoint(
            record.info["k"], np.asarray(expected), tolerance=KP_PRINTING_TOL
        )
        if index is None:
            _error(
                "KPOINT_MISMATCH", f"WFSX coordinate outside SCF grid in {fixture_id}"
            )
        if index in seen[spin]:
            _error("DUPLICATE_KPOINT", f"duplicate k/spin record in {fixture_id}")
        seen[spin].add(index)
    if any(len(indices) != len(expected) for indices in seen.values()):
        _error("INCOMPLETE_KPOINTS", f"missing k/spin records in {fixture_id}")


def _verify_case(
    root: Path, fixture_id: str, manifest_paths: set[str]
) -> dict[str, Any]:
    case_dir = root / fixture_id
    metadata_path = case_dir / "fixture.json"
    if not metadata_path.is_file():
        _error(
            "MISSING_FILE",
            f"required fixture metadata is absent: {fixture_id}/fixture.json",
        )
    metadata = _json(metadata_path)
    spec = CASE_INFO[fixture_id]
    _expect(
        metadata.get("schema") == SCHEMA,
        "INVALID_METADATA",
        f"unsupported schema in {metadata_path}",
    )
    _expect(
        metadata.get("fixture_id") == fixture_id,
        "INVALID_METADATA",
        f"fixture_id mismatch in {metadata_path}",
    )
    _expect(
        isinstance(metadata.get("producer_run_id"), str)
        and metadata["producer_run_id"],
        "INVALID_METADATA",
        f"missing producer_run_id in {metadata_path}",
    )
    spin = metadata.get("spin", {})
    if (
        spin.get("nspin") != spec["nspin"]
        or spin.get("kind") != spec["kind"]
        or spin.get("magnetic") is not spec["magnetic"]
    ):
        _error(
            "SPIN_MISMATCH", f"declared spin metadata is incompatible with {fixture_id}"
        )
    _expect(
        spin.get("scalar_reduction") is (spec["nspin"] != 4),
        "SPIN_MISMATCH",
        f"scalar/full-spinor reduction metadata is wrong in {fixture_id}",
    )
    _expect(
        spin.get("scalar_symmetry") == spec["symmetry"],
        "SPIN_MISMATCH",
        f"scalar symmetry declarations are wrong in {fixture_id}",
    )
    files = metadata.get("files")
    _expect(
        isinstance(files, dict),
        "INVALID_METADATA",
        f"missing files map in {metadata_path}",
    )
    for relative in files.values():
        asset = _relative_file(root, case_dir, relative)
        if asset.relative_to(root).as_posix() not in manifest_paths:
            _error(
                "UNPINNED_FILE",
                f"fixture asset is not covered by {MANIFEST}: {asset.relative_to(root)}",
            )
    if metadata_path.relative_to(root).as_posix() not in manifest_paths:
        _error(
            "UNPINNED_FILE",
            f"fixture metadata is not covered by {MANIFEST}: {metadata_path.relative_to(root)}",
        )
    data = _read_oracle(root, case_dir, files)
    sizes = data["sizes"]
    nspin, no_u, nk, gamma = sizes
    if int(nspin) != spec["nspin"]:
        _error(
            "SPIN_MISMATCH",
            f"WFSX header nspin={nspin}; expected {spec['nspin']} in {fixture_id}",
        )
    states = metadata.get("states", {})
    _expect(
        int(states.get("no_u", -1)) == int(no_u),
        "INCOMPLETE_STATES",
        f"no_u metadata mismatch in {fixture_id}",
    )
    _expect(
        int(states.get("nk", -1)) == int(nk),
        "INCOMPLETE_STATES",
        f"nk metadata mismatch in {fixture_id}",
    )
    _expect(
        bool(states.get("gamma_only")) is bool(gamma),
        "INCOMPLETE_STATES",
        f"Gamma metadata mismatch in {fixture_id}",
    )
    _expect(
        states.get("all_states_retained") is True
        and states.get("state_ids_zero_based") is True,
        "INCOMPLETE_STATES",
        f"state completeness is not declared in {fixture_id}",
    )
    expected_groups = int(nk) * (2 if int(nspin) == 2 else 1)
    actual_groups = states.get("groups")
    if (
        not isinstance(actual_groups, list)
        or len(actual_groups) != expected_groups
        or len(data["records"]) != expected_groups
    ):
        _error(
            "INCOMPLETE_STATES",
            f"expected {expected_groups} WFSX k/spin groups in {fixture_id}",
        )
    expected_per_group = int(no_u) * (2 if int(nspin) == 4 else 1)
    for group_index, (group, record) in enumerate(zip(actual_groups, data["records"])):
        actual_ids = [int(value) for value in np.asarray(record.info["index"]).tolist()]
        wanted_ids = list(range(expected_per_group))
        if (
            group.get("state_ids") != actual_ids
            or actual_ids != wanted_ids
            or int(group.get("state_count", -1)) != expected_per_group
        ):
            _error(
                "INCOMPLETE_STATES",
                f"state IDs are not complete for group {group_index} in {fixture_id}",
            )
        expected_spin = int(record.info["spin"]) if "spin" in record.info else None
        if group.get("spin_id") != expected_spin:
            _error(
                "SPIN_MISMATCH",
                f"spin channel record mismatch in group {group_index} of {fixture_id}",
            )
        if group.get("energy_dtype") != str(np.asarray(record.c).dtype) or group.get(
            "coefficient_dtype"
        ) != str(np.asarray(record.state).dtype):
            _error(
                "PRECISION_MISMATCH",
                f"source precision metadata mismatch in group {group_index} of {fixture_id}",
            )
        if group.get("coefficient_shape") != [
            int(value) for value in np.asarray(record.state).shape
        ]:
            _error(
                "BASIS_MISMATCH",
                f"coefficient shape metadata mismatch in group {group_index} of {fixture_id}",
            )
    kpoints = metadata.get("kpoints", {})
    if kpoints.get("weight_semantics") != spec["weight_semantics"]:
        _error("WEIGHT_MISMATCH", f"weight semantics mismatch in {fixture_id}")
    metadata_points = kpoints.get("points")
    unique_records: list[dict[str, Any]] = []
    unique_coords: list[np.ndarray] = []
    for record in data["records"]:
        kpoint = np.asarray(record.info["k"], dtype=float)
        index, _ = (
            _match_kpoint(kpoint, np.asarray(unique_coords), allow_inversion=False)
            if unique_coords
            else (None, False)
        )
        if index is None:
            unique_coords.append(kpoint)
            unique_records.append(record.info)
    if not isinstance(metadata_points, list) or len(metadata_points) != len(
        unique_records
    ):
        _error("KPOINT_MISMATCH", f"WFSX k-point count mismatch in {fixture_id}")
    for index, (actual, point) in enumerate(zip(unique_records, metadata_points)):
        declared = np.asarray(point.get("fractional", []), dtype=float)
        if declared.shape != (3,) or not _fractional_close(
            declared, actual["k"], KPOINT_COORD_TOL
        ):
            _error(
                "KPOINT_MISMATCH",
                f"WFSX coordinate metadata mismatch at point {index} in {fixture_id}",
            )
        if not math.isclose(
            float(point.get("source_weight", float("nan"))),
            float(actual["weight"]),
            abs_tol=1e-12,
        ):
            _error(
                "WEIGHT_MISMATCH",
                f"WFSX source weight mismatch at point {index} in {fixture_id}",
            )
    for ordinal, (group, record) in enumerate(zip(actual_groups, data["records"])):
        group_index = group.get("group_index")
        if (
            not isinstance(group_index, int)
            or isinstance(group_index, bool)
            or group_index != ordinal
        ):
            _error("GROUP_MISMATCH", f"record ordinal mismatch in {fixture_id}")
        index = group.get("kpoint_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(metadata_points)
            or not _fractional_close(
                metadata_points[index]["fractional"], record.info["k"], KPOINT_COORD_TOL
            )
        ):
            _error(
                "KPOINT_MISMATCH", f"group {ordinal} links to wrong k in {fixture_id}"
            )
    _verify_k_coverage(data, spec["weight_semantics"] == "selected-list", fixture_id)
    integration = kpoints.get("integration_mesh", {})
    declared_mesh = integration.get("points")
    if (
        integration.get("normalized") is not True
        or not isinstance(declared_mesh, list)
        or len(declared_mesh) != len(data["kp_points"])
    ):
        _error(
            "KPOINT_MISMATCH",
            f"integration mesh metadata is incomplete in {fixture_id}",
        )
    for index, (declared, actual, weight) in enumerate(
        zip(declared_mesh, data["kp_points"], data["kp_weights"])
    ):
        coords = np.asarray(declared.get("fractional", []), dtype=float)
        if coords.shape != (3,) or not _fractional_close(
            coords, actual, KPOINT_COORD_TOL
        ):
            _error(
                "KPOINT_MISMATCH",
                f"KP coordinate metadata mismatch at integration point {index} in {fixture_id}",
            )
        if not math.isclose(
            float(declared.get("weight", float("nan"))), float(weight), abs_tol=1e-8
        ):
            _error(
                "WEIGHT_MISMATCH",
                f"KP source weight mismatch at integration point {index} in {fixture_id}",
            )
    if not math.isclose(float(np.sum(data["kp_weights"])), 1.0, abs_tol=1e-8):
        _error(
            "WEIGHT_MISMATCH",
            f"SCF integration weights do not sum to one in {fixture_id}",
        )
    if spec["weight_semantics"] == "selected-list":
        if any(
            not math.isclose(float(record["weight"]), 1.0, abs_tol=1e-12)
            for record in unique_records
        ):
            _error(
                "WEIGHT_MISMATCH",
                f"selected-list source weights are not raw unit weights in {fixture_id}",
            )
    else:
        for record in unique_records:
            index, inverted = _match_kpoint(
                np.asarray(record["k"], dtype=float),
                data["kp_points"],
                allow_inversion=False,
                tolerance=KP_PRINTING_TOL,
            )
            if index is None or not math.isclose(
                float(record["weight"]), float(data["kp_weights"][index]), abs_tol=1e-8
            ):
                _error(
                    "WEIGHT_MISMATCH",
                    f"SCF-folded WFSX weight does not match KP in {fixture_id}",
                )
    producer = metadata.get("producer", {})
    if (
        not isinstance(producer.get("sisl_version"), str)
        or not producer["sisl_version"]
    ):
        _error(
            "PRODUCER_MISMATCH",
            f"historical ingestion sisl version is missing in {fixture_id}",
        )
    executable_hash = producer.get("executable_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", executable_hash):
        _error(
            "PRODUCER_MISMATCH",
            f"producer executable SHA256 is missing or malformed in {fixture_id}",
        )
    source = producer.get("source_checkout", {})
    if not isinstance(source.get("head"), str) or not source.get("head"):
        _error(
            "PRODUCER_MISMATCH",
            f"SIESTA source checkout revision is missing in {fixture_id}",
        )
    if not isinstance(source.get("source_sha256"), dict) or set(
        source["source_sha256"]
    ) != set(SOURCE_FILES):
        _error(
            "PRODUCER_MISMATCH",
            f"audited SIESTA source hashes are incomplete in {fixture_id}",
        )
    for digest in source["source_sha256"].values():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            _error(
                "PRODUCER_MISMATCH", f"malformed audited source hash in {fixture_id}"
            )
    energy = metadata.get("energy_conventions", {})
    _expect(
        energy.get("wfsx_eigenvalues") == "absolute eV",
        "ENERGY_MISMATCH",
        f"WFSX energy convention is missing in {fixture_id}",
    )
    _expect(
        energy.get("eig_eigenvalues")
        == "relative to E_F (sisl EIG read_data convention)",
        "ENERGY_MISMATCH",
        f"EIG energy convention is missing in {fixture_id}",
    )
    _expect(
        energy.get("wfsx_minus_fermi_matches_eig") is True,
        "ENERGY_MISMATCH",
        f"energy provenance declaration is missing in {fixture_id}",
    )
    scf = metadata.get("scf", {})
    output = data["output"]
    if (
        scf.get("converged") is not True
        or scf.get("convergence_criterion") != output["convergence_criterion"]
        or int(scf.get("iteration_count", -1)) != output["iteration_count"]
    ):
        _error(
            "SCF_MISMATCH",
            f"converged-state provenance differs from same-run output in {fixture_id}",
        )
    if not math.isclose(
        float(scf.get("total_energy_ev", float("nan"))),
        output["total_energy_ev"],
        abs_tol=1e-6,
    ):
        _error(
            "SCF_MISMATCH",
            f"total-energy provenance differs from same-run output in {fixture_id}",
        )
    if not np.allclose(
        np.asarray(scf.get("final_spin_moment_mu_b", []), dtype=float),
        np.asarray(output["final_spin_moment_mu_b"]),
        atol=1e-5,
    ):
        _error(
            "SCF_MISMATCH",
            f"final magnetic-moment provenance differs from same-run output in {fixture_id}",
        )
    fermi = metadata.get("electrons", {}).get("fermi_energy_ev")
    if not isinstance(fermi, (float, int)) or not math.isclose(
        float(fermi), data["fermi_ev"], abs_tol=1e-8
    ):
        _error(
            "FERMI_MISMATCH",
            f"Fermi provenance differs from same-run EIG in {fixture_id}",
        )
    if abs(data["output"]["printed_fermi_ev"] - data["fermi_ev"]) > 1e-5:
        _error(
            "FERMI_MISMATCH", f"printed and EIG Fermi energies disagree in {fixture_id}"
        )
    electrons = metadata.get("electrons", {})
    if not math.isclose(
        float(electrons.get("total_electrons", float("nan"))),
        data["output"]["electron_count"],
        abs_tol=1e-6,
    ):
        _error(
            "ELECTRON_COUNT_MISMATCH",
            f"electron-count provenance differs from SIESTA output in {fixture_id}",
        )
    if not math.isclose(
        float(electrons.get("electronic_temperature_k", float("nan"))),
        data["temperature_k"],
        abs_tol=1e-8,
    ):
        _error(
            "OCCUPATION_MISMATCH",
            f"occupation temperature differs from FDF in {fixture_id}",
        )
    expected_occupations = _occupation_count(
        data["eig_values"], data["kp_weights"], spec["nspin"], data["temperature_k"]
    )
    if not math.isclose(
        float(electrons.get("occupation_count_from_eig_and_kp", float("nan"))),
        expected_occupations,
        abs_tol=1e-7,
    ):
        _error(
            "OCCUPATION_MISMATCH", f"EIG/KP occupation count mismatch in {fixture_id}"
        )
    structure = metadata.get("structure", {})
    geom = data["geom"]
    symbols = [str(atom.symbol) for atom in geom.atoms]
    hsx_basis = _basis_order(data["hsx_basis"])
    wfsx_basis = _basis_order(data["wfsx_basis"])
    if hsx_basis["names_by_atom"] != structure.get("basis_orbitals_by_atom"):
        _error("BASIS_MISMATCH", f"HSX basis-order metadata mismatch in {fixture_id}")
    if wfsx_basis["names_by_atom"] != structure.get("wfsx_basis_orbitals_by_atom"):
        _error("BASIS_MISMATCH", f"WFSX basis-order metadata mismatch in {fixture_id}")
    if hsx_basis["quantum_numbers_by_atom"] != wfsx_basis["quantum_numbers_by_atom"]:
        _error(
            "BASIS_MISMATCH",
            f"HSX/WFSX orbital quantum-number order differs in {fixture_id}",
        )
    if hsx_basis["quantum_numbers_by_atom"] != structure.get(
        "basis_quantum_numbers_by_atom"
    ) or wfsx_basis["quantum_numbers_by_atom"] != structure.get(
        "wfsx_basis_quantum_numbers_by_atom"
    ):
        _error(
            "BASIS_MISMATCH", f"basis quantum-number metadata mismatch in {fixture_id}"
        )
    if sum(map(len, hsx_basis["names_by_atom"])) != int(geom.no):
        _error("BASIS_MISMATCH", f"basis size/order mismatch in {fixture_id}")
    if not np.allclose(
        np.asarray(structure.get("cell_angstrom", []), dtype=float),
        np.asarray(geom.cell),
        atol=1e-8,
    ):
        _error("BASIS_MISMATCH", f"cell metadata mismatch in {fixture_id}")
    if not np.allclose(
        np.asarray(structure.get("positions_angstrom", []), dtype=float),
        np.asarray(geom.xyz),
        atol=1e-8,
    ):
        _error("BASIS_MISMATCH", f"atomic-coordinate metadata mismatch in {fixture_id}")
    if structure.get("atom_symbols") != symbols or int(
        structure.get("basis_size", -1)
    ) != int(geom.no):
        _error(
            "BASIS_MISMATCH", f"basis/site ordering metadata mismatch in {fixture_id}"
        )
    mesh = metadata.get("mesh", {})
    if (
        mesh.get("dimensions") != data["mesh_dimensions"]
        or mesh.get("time_reversal_symmetry_for_kpoints") is not data["time_reversal"]
    ):
        _error(
            "KPOINT_MISMATCH", f"FDF mesh/symmetry metadata mismatch in {fixture_id}"
        )
    checks = _oracle_metadata(data, spec["nspin"])
    recorded_checks = metadata.get("independent_checks", {})
    for key, value in checks.items():
        recorded = recorded_checks.get(key)
        if not isinstance(recorded, (int, float)) or not math.isclose(
            float(recorded), value, rel_tol=1e-5, abs_tol=1e-8
        ):
            _error(
                "ORACLE_MISMATCH",
                f"independent check metadata mismatch for {key} in {fixture_id}",
            )
    return metadata


def _verify_collinear_pair(
    root: Path, full: dict[str, Any], reduced: dict[str, Any]
) -> None:
    if full.get("producer_run_id") != reduced.get("producer_run_id"):
        _error(
            "RUN_MISMATCH",
            "nspin=2 full and reduced WFSX exports do not identify the same producer run",
        )
    full_files = full["files"]
    reduced_files = reduced["files"]
    for name in ("fdf", "hsx", "eig", "kp", "orb_indx", "stdout"):
        if full_files.get(name) != reduced_files.get(name):
            _error(
                "RUN_MISMATCH",
                f"nspin=2 full/reduced ordinary reference {name} differs",
            )
    full_mesh = full["kpoints"]["integration_mesh"]["points"]
    reduced_mesh = reduced["kpoints"]["integration_mesh"]["points"]
    if len(full_mesh) != len(reduced_mesh):
        _error(
            "KPOINT_MISMATCH",
            "nspin=2 full/reduced integration meshes have different lengths",
        )
    for first, second in zip(full_mesh, reduced_mesh):
        if not _fractional_close(
            first["fractional"], second["fractional"], KPOINT_COORD_TOL
        ) or not math.isclose(
            float(first["weight"]), float(second["weight"]), abs_tol=1e-8
        ):
            _error("KPOINT_MISMATCH", "nspin=2 full/reduced integration meshes differ")
    expected_map = _fold_map(full["kpoints"]["points"], reduced["kpoints"]["points"])
    if (
        full["kpoints"].get("fold_map") != expected_map
        or reduced["kpoints"].get("fold_map") != expected_map
    ):
        _error(
            "WEIGHT_MISMATCH",
            "nspin=2 full/reduced explicit-to-folded point map is incomplete or inconsistent",
        )
    if any(
        float(group["state_count"]) != 18
        for group in full["states"]["groups"] + reduced["states"]["groups"]
    ):
        _error(
            "INCOMPLETE_STATES",
            "nspin=2 exports must retain 18 states in every spin channel",
        )


def _verify(root: Path) -> None:
    if not root.is_dir():
        _error("MISSING_FILE", f"fixture root is not a directory: {root}")
    manifest_paths = _verify_manifest(root)
    required = {"O.psf", "PROVENANCE.md"}
    for name in required:
        if not (root / name).is_file():
            _error("MISSING_FILE", f"required root asset is absent: {name}")
        if name not in manifest_paths:
            _error("UNPINNED_FILE", f"root asset is not covered by {MANIFEST}: {name}")
    _expect((root / "O.psf").stat().st_size > 0, "MISSING_FILE", "O.psf is empty")
    metadata = {
        fixture_id: _verify_case(root, fixture_id, manifest_paths)
        for fixture_id in CASE_INFO
    }
    _verify_collinear_pair(
        root,
        metadata["collinear_nspin2_full"],
        metadata["collinear_nspin2_reduced"],
    )
    print(f"PASS: verified {len(metadata)} complete SIESTA WFSX fixtures under {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate", help="run SIESTA and create a new fixture package"
    )
    generate.add_argument(
        "--siesta", required=True, help="path to the pinned SIESTA executable"
    )
    generate.add_argument(
        "--source-root",
        required=True,
        help="SIESTA source checkout used for build provenance",
    )
    generate.add_argument(
        "--output",
        required=True,
        help="new or empty output directory; never overwritten",
    )
    verify = subparsers.add_parser(
        "verify", help="verify hashes, metadata, and independent H/S consistency"
    )
    verify.add_argument(
        "--root", type=Path, default=ROOT, help="fixture root (default: this directory)"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _generate(args)
        else:
            _verify(args.root.resolve())
    except FixtureError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"FIXTURE_INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
