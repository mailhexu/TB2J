#!/usr/bin/env python3
"""Scratch measurement variants using the retained Story002 producer workflow.

fixture_driver.py is itself hash-pinned fixture evidence and remains immutable.
This adapter changes only FDF mesh/supercell fields, certifies printed KP
precision, then reuses its native runner, metadata/oracle and manifest writer.
It neither generates replacement regression fixtures nor introduces a format.
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import fixture_driver as producer
import numpy as np


def _wrap_fractional(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 0.5) % 1.0 - 0.5


def _trs_merge_partners(points: np.ndarray, weights: np.ndarray):
    """Merge exact TRS partner rows SIESTA double-keeps on even meshes.

    The irreducible even-mesh SCF set keeps both time-reversal partners of a
    +/-half-boundary orbit as separate half-weight rows. The retained oracle's
    nspin=2 matcher then finds two canonical classes per record. Merging each
    exact partner pair into one row with summed weight is exact for
    TRS-symmetric data and changes no scientific threshold; the original rows
    stay in SHA256SUMS.
    """
    tol = producer.KPOINT_COORD_TOL
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    frac = _wrap_fractional(points)
    dropped = np.zeros(len(points), dtype=bool)
    merged_weights = weights.copy()
    pairs: list[tuple[int, int]] = []
    for index in range(len(points)):
        if dropped[index] or np.all(np.abs(_wrap_fractional(2.0 * frac[index])) <= tol):
            continue
        partner_delta = np.abs(_wrap_fractional(frac[index] + frac))
        hits = [
            other
            for other in np.flatnonzero(np.all(partner_delta <= tol, axis=1)).tolist()
            if other != index and not dropped[other]
        ]
        if len(hits) > 1:
            producer._error(
                "KP_TRS_AMBIGUOUS",
                f"SCF row {index} has multiple TRS partner rows {hits}",
            )
        if hits:
            partner = hits[0]
            dropped[partner] = True
            merged_weights[index] += weights[partner]
            pairs.append((index, partner))
    kept = np.flatnonzero(~dropped)
    stats = {
        "partner_pairs": len(pairs),
        "original_kpoint_count": int(len(points)),
        "merged_kpoint_count": int(len(kept)),
        "kept_rule": (
            "lowest original SCF KP row index of each partner pair; the partner "
            "weight is summed into the kept row"
        ),
        "max_partner_weight_asymmetry": float(
            max(
                (abs(weights[i] - weights[j]) for i, j in pairs),
                default=0.0,
            )
        ),
        "weight_sum_absolute_error": float(
            abs(merged_weights[kept].sum() - weights.sum())
        ),
        "note": (
            "SIESTA even-mesh irreducible sets keep both time-reversal partners "
            "of +/-half-boundary orbits as separate half-weight SCF rows; the "
            "certified native KP/EIG supplements merge them so the retained "
            "unique-canonical oracle matcher stays exact"
        ),
    }
    return kept, points[kept], merged_weights[kept], pairs, stats


def _require_unique_union_matches(mesh: np.ndarray, points: np.ndarray) -> None:
    """Every SCF row class must keep exactly one canonical match in the mesh."""
    mesh = np.asarray(mesh, dtype=float)
    points = np.asarray(points, dtype=float)
    tol = producer.KPOINT_COORD_TOL
    direct = np.abs(_wrap_fractional(points[:, None, :] - mesh[None, :, :])) <= tol
    inverted = np.abs(_wrap_fractional(-points[:, None, :] - mesh[None, :, :])) <= tol
    counts = np.all(direct, axis=2) | np.all(inverted, axis=2)
    counts = counts.sum(axis=1)
    bad = np.flatnonzero(counts != 1)
    if len(bad):
        producer._error(
            "KP_TRS_AMBIGUOUS",
            f"{len(bad)} SCF rows lack a unique canonical match after TRS merge; "
            f"first offending row {points[bad[0]].tolist()} matches "
            f"{int(counts[bad[0]])} mesh rows",
        )


def _certify_scf_precision(wfsx, geometry, kp_path, printed_points):
    from sisl.unit.siesta import unit_convert

    native_points, native_weights, _ = wfsx.read_info()
    native_points = np.asarray(native_points, dtype=float)
    native_weights = np.asarray(native_weights, dtype=float)
    cartesian_bohr = (
        native_points @ geometry.lattice.rcell * unit_convert("Bohr", "Ang")
    )
    rows = [line.split() for line in kp_path.read_text().splitlines()[1:]]
    if len(rows) != len(native_points) or native_points.shape != np.shape(
        printed_points
    ):
        producer._error(
            "KP_PRECISION_CERTIFICATE", "native and printed SCF cardinalities differ"
        )
    for index, (row, point, weight) in enumerate(
        zip(rows, cartesian_bohr, native_weights)
    ):
        if len(row) != 5 or int(row[0]) != index + 1:
            producer._error(
                "KP_PRECISION_CERTIFICATE", "unexpected printed KP row layout/order"
            )
        printed = np.array([float(value) for value in row[1:]])
        rounded = np.array([float(f"{value:.6g}") for value in (*point, weight)])
        if not np.array_equal(printed, rounded):
            producer._error(
                "KP_PRECISION_CERTIFICATE",
                f"native SCF row {index} does not reproduce six-significant-digit KP output",
            )
    delta = native_points - np.asarray(printed_points)
    kept, merged_points, merged_weights, pairs, merge_stats = _trs_merge_partners(
        native_points, native_weights
    )
    _require_unique_union_matches(merged_points, native_points)
    view = {
        "printed": {
            "source": "same-run fullBZ WFSX SCF header; ordered coordinates and weights reproduce every original KP Cartesian/weight token at six significant digits",
            "printed_kp_sha256": producer._sha256(kp_path),
            "rows_certified": len(rows),
            "max_printed_fractional_rounding": float(
                np.max(np.abs(delta - np.rint(delta)))
            ),
            "native_coordinate_matching_tolerance": producer.KPOINT_COORD_TOL,
            "scientific_thresholds_changed": False,
        },
        "trs_merge": merge_stats,
        "kept_rows": kept.tolist(),
        "partner_pairs": [[int(i), int(j)] for i, j in pairs],
    }
    return view, cartesian_bohr[kept], merged_weights


def _write_merged_eig_supplement(sisl, output, files, view):
    """Write the EIG view aligned with merged KP rows; verify TRS partner energies."""
    original_eig = output / files["eig"]
    eig_sile = sisl.get_sile(str(original_eig))
    shifted = np.asarray(eig_sile.read_data(), dtype=float)  # absolute minus Ef
    fermi_ev = float(eig_sile.read_fermi_level())
    pairs = [tuple(pair) for pair in view["partner_pairs"]]
    if (
        shifted.ndim != 3
        or shifted.shape[1] != view["trs_merge"]["original_kpoint_count"]
    ):
        producer._error(
            "KP_TRS_AMBIGUOUS",
            f"EIG shape {shifted.shape} does not match the certified SCF row count",
        )
    asymmetry = max(
        float(np.max(np.abs(shifted[:, i, :] - shifted[:, j, :]))) for i, j in pairs
    )
    if asymmetry > producer.ENERGY_TOL_EV:
        producer._error(
            "KP_TRS_AMBIGUOUS",
            f"TRS partner EIG columns disagree by {asymmetry} eV, beyond "
            f"{producer.ENERGY_TOL_EV}",
        )
    merged_shifted = shifted[:, view["kept_rows"], :]
    ns, nk, nb = merged_shifted.shape
    lines = [repr(fermi_ev), f"{nb} {ns} {nk}"]
    for index in range(nk):
        values = (merged_shifted[:, index, :] + fermi_ev).reshape(-1)
        lines.append(
            " ".join([str(index + 1)] + [f"{value:.17g}" for value in values[:6]])
        )
        for start in range(6, values.size, 6):
            lines.append(
                " ".join(f"{value:.17g}" for value in values[start : start + 6])
            )
    merged_eig = original_eig.with_name("o2.native_scf.EIG")
    merged_eig.write_text("\n".join(lines) + "\n", encoding="utf-8")
    check = sisl.get_sile(str(merged_eig))
    roundtrip = np.asarray(check.read_data(), dtype=float)
    if (
        float(check.read_fermi_level()) != fermi_ev
        or roundtrip.shape != merged_shifted.shape
        or float(np.max(np.abs(roundtrip - merged_shifted), initial=0.0)) > 1e-9
    ):
        producer._error(
            "KP_TRS_AMBIGUOUS", "merged EIG supplement failed its read-back check"
        )
    view["trs_merge"]["max_partner_energy_asymmetry_ev"] = asymmetry
    view["trs_merge"]["original_eig_sha256"] = producer._sha256(original_eig)
    view["trs_merge"]["merged_eig"] = str(merged_eig.relative_to(output))
    files["printed_eig"] = files["eig"]
    files["eig"] = str(merged_eig.relative_to(output))


def _measurement_metadata(output, run_id, provenance, files):
    """Feed the existing oracle a certified, unique-canonical KP/EIG SCF view."""
    sisl = producer._sisl()

    geometry = sisl.get_sile(str(output / files["hsx"])).read_geometry()
    original_kp = output / files["kp"]
    printed, _ = sisl.get_sile(str(original_kp)).read_data(geometry.lattice)
    wfsx = sisl.get_sile(str(output / files["wfsx"]), lattice=geometry.lattice)
    view, cartesian, weights = _certify_scf_precision(
        wfsx, geometry, original_kp, printed
    )
    native_kp = original_kp.with_name("o2.native_scf.KP")
    sisl.get_sile(str(native_kp), mode="w").write_data(cartesian, weights, fmt=".17e")
    files = {
        **files,
        "printed_kp": files["kp"],
        "kp": str(native_kp.relative_to(output)),
    }
    if view["trs_merge"]["partner_pairs"]:
        _write_merged_eig_supplement(sisl, output, files, view)
    metadata = producer._metadata_for_case(
        "collinear_nspin2_reduced", output, run_id, provenance, files
    )
    metadata["scf_precision"] = view
    merge = view["trs_merge"]
    integration_source = view["printed"]["source"]
    method = (
        "same-run EIG with native SCF weights certified against original printed "
        "KP row order; 100 K Fermi-Dirac"
    )
    if merge["partner_pairs"]:
        integration_source += (
            "; exact +/-half-boundary TRS partner rows merged in the certified view"
        )
        metadata["mesh"]["scf_kpoint_count"] = merge["original_kpoint_count"]
        metadata["mesh"]["trs_merged_scf_kpoint_count"] = merge["merged_kpoint_count"]
        metadata["mesh"]["scf_kpoint_count_note"] = (
            "scf_kpoint_count is SIESTA's own SCF KP row count; the certified "
            "native KP/EIG supplements carry the merged unique-canonical view"
        )
        method += (
            "; TRS partner rows merged with summed weights and a verified "
            "invariant occupation sum"
        )
    metadata["kpoints"]["integration_mesh"]["source"] = integration_source
    metadata["electrons"]["occupation_count_method"] = method
    return metadata


def generate_measurement_case(
    *,
    siesta: Path,
    source_root: Path,
    output: Path,
    mesh: tuple[int, int, int],
    repeat_z: int = 1,
) -> Path:
    """Generate a new scratch case with the existing native producer/oracle."""
    siesta, source_root, output = (
        Path(siesta).resolve(),
        Path(source_root).resolve(),
        Path(output).resolve(),
    )
    mesh = tuple(int(n) for n in mesh)
    if len(mesh) != 3 or min(mesh) < 1 or repeat_z < 1:
        producer._error(
            "INVALID_MEASUREMENT", "mesh and repeat_z must be positive integers"
        )
    if not siesta.is_file() or not os.access(siesta, os.X_OK):
        producer._error(
            "INVALID_PRODUCER", f"SIESTA executable is not executable: {siesta}"
        )
    if not source_root.is_dir():
        producer._error(
            "INVALID_PRODUCER", f"SIESTA source checkout is absent: {source_root}"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        producer._error(
            "OUTPUT_NOT_EMPTY", f"refusing to overwrite measurement data: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir()
    template = producer.ROOT / "collinear_nspin2_shared" / "o2_collinear.fdf"
    text = template.read_text(encoding="utf-8")

    def replace_block(name, rows):
        nonlocal text
        pattern = rf"%block\s+{re.escape(name)}\s*\n.*?%endblock\s+{re.escape(name)}"
        replacement = f"%block {name}\n" + "\n".join(rows) + f"\n%endblock {name}"
        text, count = re.subn(
            pattern,
            lambda _: replacement,
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if count != 1:
            producer._error("INVALID_TEMPLATE", f"missing unique template block {name}")

    text, count = re.subn(
        r"^NumberOfAtoms\s+\d+\s*$",
        f"NumberOfAtoms {2 * repeat_z}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        producer._error("INVALID_TEMPLATE", "missing NumberOfAtoms")
    replace_block(
        "LatticeVectors",
        [" 7.0 0.0 0.0", " 0.0 7.0 0.0", f" 0.0 0.0 {4.0 * repeat_z:.17g}"],
    )
    # Explicit real tokens matter: fdf's typed integer lookup identifies species.
    replace_block(
        "AtomicCoordinatesAndAtomicSpecies",
        [
            f" {x:.17e} 3.5 {2.0 + 4.0 * iz:.17e} 1"
            for iz in range(repeat_z)
            for x in (2.895, 4.105)
        ],
    )
    replace_block("DM.InitSpin", [f" {i + 1} 1.0" for i in range(2 * repeat_z)])
    replace_block(
        "kgrid_Monkhorst_Pack",
        [f" {mesh[0]} 0 0 0.0", f" 0 {mesh[1]} 0 0.0", f" 0 0 {mesh[2]} 0.0"],
    )
    axes = [((np.arange(n) + n // 2) % n - n // 2) / n for n in mesh]
    full_points = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    replace_block(
        "WaveFuncKPoints",
        [" " + " ".join(f"{value:.17e}" for value in point) for point in full_points],
    )
    fdf_name = "o2_measurement.fdf"
    (inputs / fdf_name).write_text(text, encoding="utf-8")
    shutil.copy2(producer.ROOT / "O.psf", output / "O.psf")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    run_dir, stdout = producer._run_siesta(siesta, inputs, output, fdf_name, env)
    provenance = producer._source_producer(
        siesta,
        source_root,
        run_dir,
        stdout,
        str(producer._sisl().__version__),
        env,
        fdf_name,
    )
    prefix = run_dir.relative_to(output)
    files = {
        "fdf": str(prefix / fdf_name),
        "pseudo": "O.psf",
        "wfsx": str(prefix / "o2.fullBZ.WFSX"),
        "hsx": str(prefix / "o2.HSX"),
        "eig": str(prefix / "o2.EIG"),
        "kp": str(prefix / "o2.KP"),
        "orb_indx": str(prefix / "o2.ORB_INDX"),
        "stdout": str(prefix / "o2.out"),
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-measurement")
    metadata = _measurement_metadata(output, run_id, provenance, files)
    # SIESTA also double-keeps both TRS partner records of a boundary orbit,
    # so the retained inversion matcher needs the merged representative view.
    record_points = np.asarray(
        [point["fractional"] for point in metadata["kpoints"]["points"]], dtype=float
    )
    record_weights = np.asarray(
        [point["source_weight"] for point in metadata["kpoints"]["points"]], dtype=float
    )
    kept, merged_reps, merged_weights, record_pairs, record_stats = _trs_merge_partners(
        record_points, record_weights
    )
    fold_reps = [
        {"fractional": point.tolist(), "source_weight": float(weight)}
        for point, weight in zip(merged_reps, merged_weights)
    ]
    metadata["kpoints"]["fold_map"] = producer._fold_map(
        [{"fractional": point.tolist(), "source_weight": 1.0} for point in full_points],
        fold_reps,
    )
    if record_pairs:
        metadata["kpoints"]["trs_merged_record_reps"] = {
            "partner_pairs": len(record_pairs),
            "original_record_count": int(len(record_points)),
            "merged_representative_count": int(len(merged_reps)),
            "note": (
                "fold_map representatives merge exact +/-half-boundary TRS "
                "partner records with summed source weights; per-record groups "
                "in states keep both SIESTA records"
            ),
        }
    metadata["measurement"] = {
        "name": f"o2_repeat{repeat_z}_{'x'.join(map(str, mesh))}",
        "mesh": list(mesh),
        "repeat_z": repeat_z,
        "template_sha256": producer._sha256(template),
        "pseudopotential_sha256": producer._sha256(producer.ROOT / "O.psf"),
        "producer_adapter_sha256": producer._sha256(Path(__file__)),
        "retained_producer_sha256": producer._sha256(Path(producer.__file__)),
        "changed_template_fields": [
            "NumberOfAtoms",
            "LatticeVectors",
            "AtomicCoordinatesAndAtomicSpecies",
            "DM.InitSpin",
            "kgrid_Monkhorst_Pack",
            "WaveFuncKPoints",
        ],
        "physical_settings": "Unchanged SZP/PBE/100 K template and SCF tolerances; exact periodic z repeat.",
        "full_requested_fractional": full_points.tolist(),
    }
    producer._write_json(output / "fixture.json", metadata)
    producer._write_manifest(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siesta", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=int, nargs=3, required=True)
    parser.add_argument("--repeat-z", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        case = generate_measurement_case(
            siesta=args.siesta,
            source_root=args.source_root,
            output=args.output,
            mesh=tuple(args.mesh),
            repeat_z=args.repeat_z,
        )
    except producer.FixtureError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    print(f"Generated hash-pinned measurement case at {case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
