"""Run synthetic nontrivial/trivial open-strip examples through Python and CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import (
    BulkRequest,
    StripRequest,
    calculate_bulk,
    calculate_strip,
    compare_bulk_edge,
    prepare_topology_model,
)

J1 = 0.01
J2 = 0.002
K_SIA = 0.001


def honeycomb_spinio(dmi_z: float, sia_z=(K_SIA, K_SIA), delta: float = 0.0) -> SpinIO:
    """Two-sublattice honeycomb FM; DMI=0 gives the symmetry-trivial control."""
    cell = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    atoms = Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, -1 / 3, 0.0]],
        cell=cell,
        pbc=True,
    )
    moments = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    exchange = {}
    dmi = {}
    jani = {}

    def reverse(R):
        return tuple(-np.asarray(R))

    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        exchange[(R, 0, 1)] = J1
        exchange[(reverse(R), 1, 0)] = J1
        if delta:
            tensor = delta * np.diag([1.0, -1.0, 0.0])
            jani[(R, 0, 1)] = tensor
            jani[(reverse(R), 1, 0)] = tensor.T
    for R in [(1, 0, 0), (-1, 1, 0), (0, -1, 0)]:
        for spin in (0, 1):
            exchange[(R, spin, spin)] = J2
            exchange[(reverse(R), spin, spin)] = J2
        dmi[(R, 0, 0)] = np.array([0.0, 0.0, dmi_z])
        dmi[(reverse(R), 0, 0)] = np.array([0.0, 0.0, -dmi_z])
        dmi[(R, 1, 1)] = np.array([0.0, 0.0, -dmi_z])
        dmi[(reverse(R), 1, 1)] = np.array([0.0, 0.0, dmi_z])
    exc = SpinIO(
        atoms=atoms,
        spinat=moments,
        charges=np.zeros(2),
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exchange,
        dmi_ddict=dmi if dmi_z else None,
        Jani_dict=jani if delta else None,
        sia_tensor={
            0: np.diag([0.0, 0.0, sia_z[0]]),
            1: np.diag([0.0, 0.0, sia_z[1]]),
        },
    )
    exc._build_Rlist()
    return exc


def _resolved_edge_minima(result, edge: str):
    weights = (
        result.left_edge_fraction if edge == "left" else result.right_edge_fraction
    )
    other = result.right_edge_fraction if edge == "left" else result.left_edge_fraction
    minima = []
    for k_index in range(len(result.edge_kpoints)):
        valid = np.isfinite(weights[k_index]) & np.isfinite(other[k_index])
        valid &= weights[k_index] >= 0.5
        valid &= weights[k_index] > other[k_index]
        minima.append(
            float(np.min(result.energies[k_index, valid])) if np.any(valid) else None
        )
    return minima


def _print_width_evidence(coarse, wide):
    """Report comparable localized branches without claiming convergence."""
    differences = []
    for edge in ("left", "right"):
        e0 = _resolved_edge_minima(coarse, edge)
        e1 = _resolved_edge_minima(wide, edge)
        differences.extend(
            abs(a - b) for a, b in zip(e0, e1) if a is not None and b is not None
        )
    if differences:
        print(
            "largest change in the lowest resolved edge-dominant mode between "
            f"widths {coarse.request.width} and {wide.request.width}: "
            f"{max(differences):.6e} eV (matched by edge and sampled k; "
            "evidence only, not a convergence certificate)"
        )
    else:
        print(
            "no common resolved edge-dominant branches were found for the "
            "requested width comparison; no convergence claim is made"
        )


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    python_results = {}
    for label, dmi_z, sia_z, cases in (
        (
            "nontrivial",
            0.0005,
            (K_SIA, K_SIA),
            [(4, (0.0, 0.0)), (8, (0.0, 0.0)), (8, (0.75, 0.0))],
        ),
        (
            "trivial",
            0.0,
            (K_SIA, 1.5 * K_SIA),
            [(4, (0.0, 0.0))],
        ),
    ):
        exc = honeycomb_spinio(dmi_z, sia_z=sia_z)
        model = prepare_topology_model(exc, source_path=f"synthetic-{label}")
        for case_index, (width, termination) in enumerate(cases):
            result = calculate_strip(
                model,
                StripRequest(
                    width=width,
                    kpoints=41,
                    termination=termination,
                ),
            )
            if not result.valid:
                raise RuntimeError(
                    f"{label} width {width} strip is invalid: "
                    f"{result.diagnostics['failures']}"
                )
            key = f"{label}-{case_index}"
            python_results[key] = result
            path = output_dir / f"{key}.json"
            path.write_text(
                json.dumps(result.to_json_dict(), indent=1, allow_nan=False)
            )
            resolved = np.isfinite(result.left_edge_fraction) & np.isfinite(
                result.right_edge_fraction
            )
            both_edge = (
                float(
                    np.max(
                        np.minimum(
                            result.left_edge_fraction[resolved],
                            result.right_edge_fraction[resolved],
                        )
                    )
                )
                if np.any(resolved)
                else None
            )
            print(
                f"Python {label}: width={width}, termination={termination}, "
                f"sites={len(result.site_map)}, max resolved both-edge fraction="
                f"{both_edge}, JSON={path}"
            )

    _print_width_evidence(
        python_results["nontrivial-0"], python_results["nontrivial-1"]
    )
    anomalous_model = prepare_topology_model(
        honeycomb_spinio(0.0005, delta=0.0003),
        source_path="synthetic-stable-anomalous",
    )
    anomalous_bulk = calculate_bulk(
        anomalous_model, BulkRequest(mesh=(12, 12), bands=[0])
    )
    if not anomalous_bulk.valid:
        raise RuntimeError(
            f"stable anomalous benchmark is invalid: "
            f"{anomalous_bulk.diagnostics['failures']}"
        )
    anomalous_h = anomalous_model.magnon().Hq(np.array([[0.11, 0.23, 0.0]]))[0]
    anomalous_pairing = float(np.max(np.abs(anomalous_h[:2, 2:])))
    if anomalous_pairing <= 0.0:
        raise RuntimeError("the declared anomalous benchmark has zero pairing.")
    anomalous_record = anomalous_bulk.to_json_dict()
    anomalous_record["anomalous_pairing_max_eV"] = anomalous_pairing
    anomalous_record["anomalous_kpoint_fractional"] = [0.11, 0.23, 0.0]
    anomalous_path = output_dir / "stable-anomalous-bulk.json"
    anomalous_path.write_text(json.dumps(anomalous_record, indent=1, allow_nan=False))
    print(
        f"stable anomalous bulk: max|B|={anomalous_pairing:.6e} eV, "
        f"C={anomalous_bulk.chern}, JSON={anomalous_path}"
    )
    trivial_model = prepare_topology_model(
        honeycomb_spinio(0.0, sia_z=(K_SIA, 1.5 * K_SIA)),
        source_path="synthetic-trivial-control",
    )
    trivial_bulk = calculate_bulk(trivial_model, BulkRequest(mesh=(12, 12), bands=[0]))
    trivial_strip = calculate_strip(
        trivial_model, StripRequest(width=8, kpoints=43, edge_layers=2)
    )
    trivial_comparison = compare_bulk_edge(
        trivial_bulk, trivial_strip, gap=1, edge="left"
    )
    if not trivial_bulk.valid or not trivial_strip.valid:
        raise RuntimeError("the declared trivial control has an invalid source result.")
    if trivial_bulk.chern != 0 or trivial_comparison.valid:
        raise RuntimeError("the trivial control failed its fail-closed expectation.")
    for filename, result in (
        ("trivial-bulk.json", trivial_bulk),
        ("trivial-comparison-strip.json", trivial_strip),
        ("trivial-comparison.json", trivial_comparison),
    ):
        (output_dir / filename).write_text(
            json.dumps(result.to_json_dict(), indent=1, allow_nan=False)
        )
    print(
        f"Python trivial control: C={trivial_bulk.chern}, "
        f"comparison valid={trivial_comparison.valid}, "
        f"unmatched crossings={trivial_comparison.unmatched_crossings}"
    )

    print(
        "nontrivial termination variation: compare nontrivial-1 and "
        "nontrivial-2 JSON spectra/localization; a single width or cut is not "
        "treated as converged"
    )

    comparison_model = prepare_topology_model(
        honeycomb_spinio(0.0005), source_path="synthetic-nontrivial"
    )
    bulk = calculate_bulk(comparison_model, BulkRequest(mesh=(12, 12), bands=[0]))
    if not bulk.valid:
        raise RuntimeError(f"synthetic bulk result is invalid: {bulk.diagnostics}")
    edge_strip = calculate_strip(
        comparison_model,
        StripRequest(width=12, kpoints=61, edge_layers=2),
    )
    if not edge_strip.valid:
        raise RuntimeError(
            f"synthetic comparison strip is invalid: {edge_strip.diagnostics}"
        )
    comparison = compare_bulk_edge(bulk, edge_strip, gap=1, edge="left")
    if not comparison.valid:
        raise RuntimeError(
            f"synthetic bulk-edge comparison is invalid: {comparison.diagnostics}"
        )
    python_comparison_paths = {
        "bulk": output_dir / "nontrivial-bulk.json",
        "strip": output_dir / "nontrivial-comparison-strip.json",
        "comparison": output_dir / "nontrivial-comparison.json",
    }
    for key, result in (
        ("bulk", bulk),
        ("strip", edge_strip),
        ("comparison", comparison),
    ):
        python_comparison_paths[key].write_text(
            json.dumps(result.to_json_dict(), indent=1, allow_nan=False)
        )
    print(
        f"Python signed comparison: C={comparison.chern_below_gap}, "
        f"left flow={comparison.signed_crossings}; "
        f"records={python_comparison_paths}"
    )

    paired = output_dir / "nontrivial_TB2J_results"
    honeycomb_spinio(0.0005).write_pickle(str(paired))
    trivial_source = output_dir / "trivial_TB2J_results"
    honeycomb_spinio(0.0, sia_z=(K_SIA, 1.5 * K_SIA)).write_pickle(str(trivial_source))

    bulk_path = output_dir / "nontrivial-paired-bulk.json"
    strip_path = output_dir / "nontrivial-paired-strip.json"
    comparison_path = output_dir / "nontrivial-paired-comparison.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "TB2J.magnon.magnon_cli",
            "-p",
            str(paired),
            "--topology-compare",
            "--topology-gap",
            "1",
            "--topology-mesh",
            "12",
            "12",
            "--topology-strip-width",
            "12",
            "--topology-strip-kpoints",
            "61",
            "--topology-strip-edge-layers",
            "2",
            "--topology-edge",
            "left",
            "--topology-out",
            str(bulk_path),
            "--topology-strip-out",
            str(strip_path),
            "--topology-comparison-out",
            str(comparison_path),
        ],
        check=True,
    )
    paired_comparison = json.loads(comparison_path.read_text())
    if not paired_comparison["valid"]:
        raise RuntimeError(
            f"paired CLI comparison is invalid: {paired_comparison['diagnostics']}"
        )
    print(
        "CLI signed comparison: "
        f"C={paired_comparison['bulk_chern_below_gap']}, "
        "left flow="
        f"{paired_comparison['spectral_flow']['signed_crossings']}; "
        f"records={bulk_path}, {strip_path}, {comparison_path}"
    )
    trivial_bulk_path = output_dir / "trivial-paired-bulk.json"
    trivial_strip_path = output_dir / "trivial-paired-strip.json"
    trivial_comparison_path = output_dir / "trivial-paired-comparison.json"
    invalid_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "TB2J.magnon.magnon_cli",
            "-p",
            str(trivial_source),
            "--topology-compare",
            "--topology-gap",
            "1",
            "--topology-mesh",
            "12",
            "12",
            "--topology-strip-width",
            "8",
            "--topology-strip-kpoints",
            "43",
            "--topology-strip-edge-layers",
            "2",
            "--topology-edge",
            "left",
            "--topology-out",
            str(trivial_bulk_path),
            "--topology-strip-out",
            str(trivial_strip_path),
            "--topology-comparison-out",
            str(trivial_comparison_path),
        ],
        check=False,
    )
    if invalid_cli.returncode != 1:
        raise RuntimeError(
            f"trivial paired CLI expected exit 1, got {invalid_cli.returncode}"
        )
    trivial_cli_bulk = json.loads(trivial_bulk_path.read_text())
    trivial_cli_strip = json.loads(trivial_strip_path.read_text())
    trivial_cli_comparison = json.loads(trivial_comparison_path.read_text())
    if not trivial_cli_bulk["valid"] or not trivial_cli_strip["valid"]:
        raise RuntimeError(
            "the trivial paired CLI did not retain valid source records."
        )
    if trivial_cli_comparison["valid"]:
        raise RuntimeError("the trivial paired CLI falsely claimed correspondence.")
    print(
        "CLI trivial control: exit=1; standalone bulk/strip retained; "
        "unmatched crossings="
        f"{trivial_cli_comparison['spectral_flow']['unmatched_crossings']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/magnon-topology-strip"),
        help="Scratch directory for JSON and synthetic SpinIO inputs.",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
