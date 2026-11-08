"""Generate retained evidence for the stable Q=1/2 magnetic-cell folding figure.

The model is the same one-dimensional antiferromagnetic chain used by
``tests/tests/test_magnon_topology_strip.py::
test_explicit_magnetic_cell_strip_restores_folded_spectrum``.  The source cell
has one spin and ``J<0``; ``expand_magnetic_cell`` doubles the cell along x and
rotates the moments by 180 degrees, producing the explicit two-site magnetic
cell.  The periodic-width strip is assembled through the same private seam as
that regression because ``StripRequest`` has no public periodic-width option.

The record compares the actual folded magnetic-cell bulk with the periodic
strip at the width-sampled folded momenta.  It also keeps an open finite-strip
scan for finite-size context.  The two folded bulk branches are chemical-cell
images of one source-cell mode, not an independently retained primitive
magnon band; the source-cell spiral cross-check is labelled accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import (
    LatticeEmbedding,
    StripRequest,
    calculate_strip,
    expand_magnetic_cell,
    prepare_topology_model,
    solve_bosonic_modes,
)
from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian
from TB2J.magnon.topology.strip import _assemble_strip_bonds

SCHEMA_VERSION = "tb2j-topology.magnetic-folding/1"
ENERGY_TOLERANCE_EV = 1e-10
REGRESSION_WIDTH = 3
PROBE_EDGE_K = 0.19
Q = (0.5, 0.0, 0.0)
AXIS = (0.0, 1.0, 0.0)
EMBEDDING = LatticeEmbedding(basis=[[0, 1, 0], [1, 0, 0], [0, 0, -1]])


def chain_spinio(exchange_j: float = -0.01) -> SpinIO:
    """One-site chain matching the stable Q=1/2 regression fixture."""
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.diag([1.0, 5.0, 5.0]), pbc=True)
    exc = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=np.zeros(1),
        index_spin=[0],
        colinear=False,
        exchange_Jdict={
            ((1, 0, 0), 0, 0): exchange_j,
            ((-1, 0, 0), 0, 0): exchange_j,
        },
        sia_tensor={0: np.diag([0.0, 0.0, 0.001])},
    )
    exc._build_Rlist()
    return exc


def _git_revision() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _version() -> str | None:
    try:
        return metadata.version("TB2J")
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_snapshot(model) -> dict:
    return {
        "cell_angstrom": model.cell.tolist(),
        "positions_angstrom": model.positions.tolist(),
        "positions_fractional": model.positions_fractional.tolist(),
        "index_spin": model.index_spin.tolist(),
        "spin_atoms": model.spin_atoms.tolist(),
        "magmoms_muB": model.magmoms.tolist(),
        "Rlist": model.Rlist.tolist(),
        "JR_eV": model.JR.tolist(),
    }


def _compare_at_edge_k(
    magnetic_model,
    JR_strip,
    R_strip,
    moments_strip,
    width: int,
    edge_k: float,
) -> dict:
    strip_k = EMBEDDING.to_input_k([[edge_k, 0.0, EMBEDDING.normal_k]])
    (H,) = local_frame_bosonic_hamiltonian(JR_strip, R_strip, moments_strip, strip_k)
    strip_energies = np.sort(solve_bosonic_modes(H).energies)
    folded_k = np.concatenate(
        [
            EMBEDDING.to_input_k([[edge_k, width_index / width, EMBEDDING.normal_k]])
            for width_index in range(width)
        ],
        axis=0,
    )
    folded_by_momentum = magnetic_model.magnon().get_magnon_bands(kpoints=folded_k)[1]
    folded_energies = np.sort(folded_by_momentum.reshape(-1))
    if strip_energies.size != folded_energies.size:
        raise RuntimeError(
            f"width {width}: strip has {strip_energies.size} modes but the "
            f"folded bulk has {folded_energies.size}."
        )
    modes_by_momentum = [
        {
            "folded_k_fractional": k.tolist(),
            "folded_bulk_energies_eV": np.sort(energies).tolist(),
        }
        for k, energies in zip(folded_k, folded_by_momentum)
    ]
    return {
        "edge_k_fractional": float(edge_k),
        "strip_k_fractional": strip_k[0].tolist(),
        "folded_k_fractional": folded_k.tolist(),
        "strip_energies_eV": strip_energies.tolist(),
        "folded_bulk_energies_eV": folded_energies.tolist(),
        "folded_bulk_modes_by_momentum": modes_by_momentum,
        "max_abs_difference_eV": float(
            np.max(np.abs(strip_energies - folded_energies))
        ),
    }


def _periodic_restoration(
    magnetic_model,
    width: int,
    edge_kpoints: np.ndarray,
    probe_edge_k: float,
) -> dict:
    JR_strip, R_strip, sites, retained, cut = _assemble_strip_bonds(
        magnetic_model,
        width,
        EMBEDDING,
        (0.0, 0.0),
        periodic_width=True,
    )
    moments_strip = np.asarray(
        [magnetic_model.magmoms[site["source_spin"]] for site in sites]
    )
    strip_energies = []
    folded_energies = []
    folded_kpoints = []
    differences = []
    magnon = magnetic_model.magnon()
    for edge_k in edge_kpoints:
        strip_k = EMBEDDING.to_input_k([[edge_k, 0.0, EMBEDDING.normal_k]])
        (H,) = local_frame_bosonic_hamiltonian(
            JR_strip, R_strip, moments_strip, strip_k
        )
        strip = np.sort(solve_bosonic_modes(H).energies)
        folded_k = np.concatenate(
            [
                EMBEDDING.to_input_k(
                    [[edge_k, width_index / width, EMBEDDING.normal_k]]
                )
                for width_index in range(width)
            ],
            axis=0,
        )
        folded = magnon.get_magnon_bands(kpoints=folded_k)[1]
        folded = np.sort(folded.reshape(-1))
        if strip.size != folded.size:
            raise RuntimeError(
                f"width {width}: periodic strip and folded bulk mode counts "
                f"differ ({strip.size} vs {folded.size})."
            )
        difference = float(np.max(np.abs(strip - folded)))
        strip_energies.append(strip.tolist())
        folded_energies.append(folded.tolist())
        folded_kpoints.append(folded_k.tolist())
        differences.append(difference)
    probe = _compare_at_edge_k(
        magnetic_model,
        JR_strip,
        R_strip,
        moments_strip,
        width,
        probe_edge_k,
    )
    return {
        "width": int(width),
        "site_count": int(len(sites)),
        "retained_directed_bond_records": int(retained),
        "cut_directed_bond_records": int(cut),
        "edge_k_fractional": np.asarray(edge_kpoints).tolist(),
        "strip_energies_eV": strip_energies,
        "folded_bulk_k_fractional": folded_kpoints,
        "folded_bulk_energies_eV": folded_energies,
        "max_abs_difference_per_k_eV": differences,
        "max_abs_difference_eV": float(max(differences)),
        "max_edge_momentum_variation_eV": float(
            np.max(np.ptp(np.asarray(strip_energies), axis=0))
        ),
        "tolerance_eV": ENERGY_TOLERANCE_EV,
        "matches": bool(max(differences) <= ENERGY_TOLERANCE_EV),
        "probe": probe,
    }


def _open_strip(magnetic_model, width: int, kpoints: int) -> dict:
    result = calculate_strip(
        magnetic_model,
        StripRequest(width=width, kpoints=kpoints, embedding=EMBEDDING),
    )
    if not result.valid or result.energies is None:
        raise RuntimeError(
            f"open strip width {width} is invalid: "
            f"{result.diagnostics.get('failures')}"
        )
    energies = np.asarray(result.energies)
    localization_available = bool(
        np.all(np.isfinite(result.left_edge_fraction))
        and np.all(np.isfinite(result.right_edge_fraction))
    )
    return {
        "width": int(width),
        "valid": bool(result.valid),
        "edge_k_fractional": np.asarray(result.edge_kpoints).tolist(),
        "energies_eV": energies.tolist(),
        "max_edge_momentum_variation_eV": float(np.max(np.ptp(energies, axis=0))),
        "site_count": int(len(result.site_map)),
        "retained_directed_bond_records": int(
            result.diagnostics["retained_directed_bond_records"]
        ),
        "cut_directed_bond_records": int(
            result.diagnostics["cut_directed_bond_records"]
        ),
        "reference_source": result.diagnostics["reference_source"],
        "stationarity": result.diagnostics["stationarity"],
        "solver_classification_counts": result.diagnostics.get("solver", {}).get(
            "classification_counts", {}
        ),
        "interior_reference_mismatch": result.diagnostics[
            "interior_reference_mismatch"
        ],
        "degenerate_subspace_count": int(len(result.degenerate_subspaces)),
        "localization_top_level_available": localization_available,
        "localization_note": (
            "Top-level edge fractions are NaN when every positive mode sits in "
            "a degenerate pair; the projected spectra live under "
            "degenerate_subspaces and are not duplicated here."
        ),
    }


def run(output_dir: Path, kpoints: int = 21) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = chain_spinio()
    source.write_pickle(str(output_dir / "source_TB2J_results"))
    primitive = prepare_topology_model(source, source_path="synthetic-q1-2-afm-chain")
    magnetic = expand_magnetic_cell(primitive, Q=Q, axis=AXIS)
    transform = dict(magnetic.magnetic_cell_transform or {})
    if transform.get("folded_mode_count") != 2:
        raise RuntimeError("the Q=1/2 expansion did not produce two folded modes.")
    if transform.get("ordering_vector_rational") != ["1/2", "0/1", "0/1"]:
        raise RuntimeError("the Q=1/2 expansion is not commensurate as requested.")

    edge_kpoints = np.arange(kpoints, dtype=float) / kpoints
    widths = (2, 3, 4, 5)
    periodic = [
        _periodic_restoration(magnetic, width, edge_kpoints, PROBE_EDGE_K)
        for width in widths
    ]
    open_strips = [_open_strip(magnetic, width, kpoints) for width in widths]

    bulk_k = np.column_stack(
        [
            np.linspace(-0.5, 0.5, 201),
            np.full(201, PROBE_EDGE_K),
            np.zeros(201),
        ]
    )
    bulk_energies = magnetic.magnon().get_magnon_bands(kpoints=bulk_k)[1]
    supercell_matrix = np.asarray(transform["supercell_matrix"], dtype=float)
    source_k = bulk_k @ np.linalg.inv(supercell_matrix).T
    source_magnon = primitive.magnon(Q=Q, n=AXIS)
    source_energies = source_magnon.get_magnon_bands(kpoints=source_k)[1]
    source_shifted = source_magnon.get_magnon_bands(kpoints=source_k + np.asarray(Q))[1]
    source_difference = float(np.max(np.abs(bulk_energies - source_energies)))
    source_shift_difference = float(np.max(np.abs(bulk_energies - source_shifted)))

    width_3 = next(item for item in periodic if item["width"] == REGRESSION_WIDTH)
    all_periodic_match = all(item["matches"] for item in periodic)
    all_open_valid = all(item["valid"] for item in open_strips)
    if not all_periodic_match:
        raise RuntimeError("periodic restoration failed for at least one width.")
    if not all_open_valid:
        raise RuntimeError("an open finite strip was invalid.")

    record = {
        "kind": "magnetic-cell-folding",
        "schema_version": SCHEMA_VERSION,
        "units": {
            "energy": "eV",
            "moment": "muB",
            "length": "angstrom",
            "kpoints": "fractional reciprocal of the stated cell",
        },
        "provenance": {
            "generator": "TB2J/examples/magnon/magnetic_cell_folding_evidence.py",
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "figure_script": (
                "mypapers/magnon_topology/scripts/"
                "magnetic_cell_folding_q1_2/plot_magnetic_cell_folding.py"
            ),
            "figure_output": (
                "mypapers/magnon_topology/figures/"
                "magnetic_cell_folding_q1_2/magnetic_cell_folding.pdf"
            ),
            "tb2j_version": _version(),
            "source_revision": _git_revision(),
            "source_pickle": "source_TB2J_results/TB2J.pickle",
            "source_model": (
                "synthetic one-site AFM chain constructed in memory; not a DFT "
                "or experimental input"
            ),
            "regression": {
                "path": "tests/tests/test_magnon_topology_strip.py",
                "test": ("test_explicit_magnetic_cell_strip_restores_folded_spectrum"),
                "reproduced": True,
                "notes": (
                    "Same exchange_j=-0.01, Q=(0.5,0,0), axis=(0,1,0), "
                    "embedding [[0,1,0],[1,0,0],[0,0,-1]], width=3 and "
                    "edge_k=0.19 probe."
                ),
            },
        },
        "conventions": {
            "expansion": (
                "expand_magnetic_cell(model, Q=(0.5,0,0), axis=(0,1,0)); "
                "the returned model is solved with Q=0"
            ),
            "folded_mode_count": int(transform["folded_mode_count"]),
            "folding_relation": (
                "magnetic-cell branches are chemical-cell images at k and k+Q; "
                "they are not an independently retained primitive band"
            ),
            "source_k_mapping": ("k_source = k_magnetic @ supercell_matrix^{-T}"),
            "embedding_reciprocal_map": "k_magnetic = k_embed @ basis^{-T}",
            "periodic_strip_assembly": (
                "TB2J.magnon.topology.strip._assemble_strip_bonds("
                "..., periodic_width=True) + "
                "local_frame_bosonic_hamiltonian + solve_bosonic_modes; "
                "the same seam used by the regression"
            ),
            "bulk_energies": ("TopologyModel.magnon().get_magnon_bands with Q=0"),
            "edge_kpoints": "endpoint-exclusive n/N",
        },
        "models": {
            "primitive": {
                "digest": primitive.digest,
                "nspin": int(primitive.nspin),
                "snapshot": _model_snapshot(primitive),
            },
            "magnetic": {
                "digest": magnetic.digest,
                "nspin": int(magnetic.nspin),
                "snapshot": _model_snapshot(magnetic),
                "magnetic_cell_transform": transform,
            },
        },
        "request": {
            "widths": list(widths),
            "regression_width": REGRESSION_WIDTH,
            "kpoints": int(kpoints),
            "probe_edge_k_fractional": PROBE_EDGE_K,
            "embedding_basis": EMBEDDING.basis.tolist(),
            "normal_k_fractional": float(EMBEDDING.normal_k),
            "termination_offsets_width_cells": [0.0, 0.0],
            "reference": "parent_model_reference",
        },
        "bulk": {
            "k_fractional": bulk_k.tolist(),
            "energies_eV": np.asarray(bulk_energies).tolist(),
            "branch_splitting_max_eV": float(np.max(np.ptp(bulk_energies, axis=1))),
            "source_cell_spiral_reference": {
                "model_digest": primitive.digest,
                "ordering_vector_input_fractional": list(Q),
                "rotation_axis_cartesian": list(AXIS),
                "k_source_fractional": source_k.tolist(),
                "energies_eV": np.asarray(source_energies).tolist(),
                "max_abs_difference_vs_magnetic_eV": source_difference,
                "max_abs_difference_vs_q_shifted_eV": source_shift_difference,
                "note": (
                    "The source-cell Q=1/2 spiral is a folding cross-check, "
                    "not an independent primitive magnon representation: it is "
                    "the same input cell before expansion.  Its single branch "
                    "coincides with both magnetic-cell branches."
                ),
            },
        },
        "periodic_restoration": periodic,
        "open_strip": open_strips,
        "validation": {
            "expanded_folded_mode_count": int(transform["folded_mode_count"]),
            "supercell_matrix": transform["supercell_matrix"],
            "periodic_restoration_all_match": bool(all_periodic_match),
            "max_abs_difference_over_widths_eV": float(
                max(item["max_abs_difference_eV"] for item in periodic)
            ),
            "regression_width_3_max_abs_difference_eV": float(
                width_3["max_abs_difference_eV"]
            ),
            "regression_width_3_probe_max_abs_difference_eV": float(
                width_3["probe"]["max_abs_difference_eV"]
            ),
            "open_strip_all_valid": bool(all_open_valid),
            "independent_primitive_band_retained": False,
            "primitive_band_statement": (
                "No independent primitive magnon band is claimed.  The "
                "expanded magnetic cell is the minimal cell of the explicit "
                "antiferromagnetic order; its two branches are folding images "
                "of one chemical-cell mode."
            ),
            "finite_size_statement": (
                "Widths 2, 3, 4 and 5 are a finite-size check of the exact "
                "width-space restoration, not a convergence study.  Width 3 is "
                "the regression case.  The open-strip scan is contextual only; "
                "its top-level edge localization is unresolved for this "
                "degenerate chain."
            ),
            "claim": (
                "A periodic strip reproduces the actual folded magnetic-cell "
                "bulk at the width-sampled folded momenta to machine precision; "
                "this certifies the magnetic-cell remapping and the folding "
                "bookkeeping, not a continuum limit."
            ),
        },
    }
    path = output_dir / "magnetic_cell_folding.json"
    path.write_text(json.dumps(record, indent=1, allow_nan=False))
    probe_residual = record["validation"][
        "regression_width_3_probe_max_abs_difference_eV"
    ]
    print(
        f"wrote {path}; max periodic restoration residual "
        f"{record['validation']['max_abs_difference_over_widths_eV']:.3e} eV; "
        f"width-3 probe {probe_residual:.3e} eV"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/magnetic-cell-folding"),
        help="Directory for the versioned JSON record and source pickle.",
    )
    parser.add_argument("--kpoints", type=int, default=21)
    args = parser.parse_args()
    run(args.output_dir, kpoints=args.kpoints)


if __name__ == "__main__":
    main()
