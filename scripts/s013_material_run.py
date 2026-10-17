"""Story-013 material run: bare + coupled CrI3 spectra with provenance.

Consumes only qualified/hash-verified stores (see story-013 Technical
Notes); no DFT is executed.  The coupled spectra are additionally repeated
over the declared SPEC-013-002 convergence sweep (three per-q phonon-loop
mesh densities x temperatures 15/30/45 K); per-block spectral-max/Sigma
changes and the verdict are recorded in the convergence section of the
summary JSON and manifest, and the observed convergence (or its absence) is
stated with the numbers in the thermal/IR prescription text carried by both.
Bulk artifacts are written to
``$S013_OUT`` (default ``/STORAGE1/hexu/projects_bak/magnon_phonon_spectra/s013``);
small summaries, plots and the manifest are copied into the vault evidence
directory ``$S013_EVIDENCE``.

Usage::

    python scripts/s013_material_run.py [--outdir DIR] [--evidence-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

#: unfold-epc Story-007 worktree providing the qualified phonon ingestion
UNFOLD_EPC_S007 = "/home/hexu/projects/TB2J_dev/unfold-epc-s007"
REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(REPO_ROOT), UNFOLD_EPC_S007):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from TB2J.magnon_phonon.material import (  # noqa: E402
    DEFAULT_BROADENING_EV,
    DEFAULT_TEMPERATURE_K,
    MIXING_UNAVAILABLE_STATEMENT,
    THERMAL_PRESCRIPTION,
    MaterialSettings,
    bare_magnon_dispersion,
    block_to_fixture,
    cri3_magnon_model,
    load_cri3_blocks,
    provenance_manifest,
    run_coupled_spectra,
    sha256_file,
    validate_manifest,
)

VAULT = Path(
    "/home/hexu/projects/memnotes/memnotes/Projects/non-orthogonal-spin-lattice"
)
S002_STORE = VAULT / "evidence/s002/matched-finite-q-rows.json"
S002_EVIDENCE_DIR = VAULT / "evidence"
MODE_TABLES = {
    # Gamma (det2) tables
    "b07dd0323c1f7daa609abef631b3ca33dbf5244c929683157a6e33f685a04b94": VAULT
    / "evidence/quick-task-cri3g-modes.json",
    # M (det2) table
    "8e544140601964a010b4367d5b7f3e3b88fb6a1dfd75b61215f1133b70d56298": VAULT
    / "evidence/quick-task-cri3m-modes.json",
    # K (det3) table
    "ef1181f2ce00fccfce883b2101885ee5aa3f140227e8fa03db936627b59751d6": Path(
        "/home/hexu/projects/TB2J_dev/unfold-epc/.tmp/kosmic/runs/"
        "20261011T030709Z-unfold-epc-implementatio-70f13fa83e95e2a0752b771d05d32453"
        "/scratch/cri3_modes.json"
    ),
}
S012_CAMPAIGN = Path(
    "/STORAGE1/hexu/projects_bak/magnon_phonon_spectra/s012/campaigns/s012-h005-3x"
)
S012_QUALIFICATION = (
    VAULT
    / "subprojects/validated-magnon-phonon-spectra-and-manuscript-evidence"
    / "evidence/s012/qualification-report.json"
)
MAGNON_LOOP_MESH = (
    np.zeros(3),
    np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    np.array([0.5, 0.0, 0.0]),
)
#: declared per-q phonon-loop magnon meshes; M+M folds to Gamma (goldstone
#: occupation diverges), so the M external phonon uses the K point only
K_POINT = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
M_POINT = np.array([0.5, 0.0, 0.0])
PHONON_LOOP_MESHES = {
    (1.0 / 3.0, 1.0 / 3.0, 0.0): ([K_POINT, M_POINT], [0.5, 0.5]),
    (0.5, 0.0, 0.0): ([K_POINT], [1.0]),
    (0.0, 0.0, 0.0): ([K_POINT, M_POINT], [0.5, 0.5]),
}
FIXTURE_LABEL = "K:acoustic-like"


def _complete_mode_tables():
    """Resolve the Gamma ligand-strong table path by its content hash."""
    import hashlib

    target_prefix = "910b8a14"
    for candidate in sorted((VAULT / "evidence").glob("quick-task-cri3g-*.json")):
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest.startswith(target_prefix):
            MODE_TABLES[digest] = candidate
            return
    raise FileNotFoundError("Gamma ligand-strong modes table not found in vault")


def git_revision():
    """Writer-repo HEAD resolved from REPO_ROOT; empty or failing runs error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"git revision unavailable: {error}") from error
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        raise RuntimeError(
            f"git revision unavailable in {REPO_ROOT}: {result.stderr.strip()}"
        )
    return revision


def bare_phonons_qualified():
    """Qualified s012-h005-3x phonons at the covered commensurate points."""
    from unfold_epc.phonon import campaign_phonon_modes

    out = {}
    for name, q in (("gamma", np.zeros(3)), ("K", np.array([1 / 3, 1 / 3, 0.0]))):
        modes = campaign_phonon_modes(S012_CAMPAIGN, q, symmetrize=True)
        out[name] = {
            "q": modes.q.tolist(),
            "frequencies_thz": np.asarray(modes.frequencies_thz, float).tolist(),
            "eigenvalues": np.asarray(
                modes.eigenvalues_eV_per_Ang2_per_amu, float
            ).tolist(),
            "instabilities": modes.instabilities,
            "diagnostics": {
                k: modes.diagnostics[k]
                for k in (
                    "raw_reciprocity_residual_eV_per_Ang2",
                    "translation_residual_eV_per_Ang2",
                )
            },
            "symmetrization": modes.diagnostics["symmetrization"],
            "provenance": {
                k: modes.provenance[k] for k in ("producer", "campaign_path")
            }
            if "producer" in modes.provenance
            else {"campaign_path": modes.provenance.get("campaign_path", "")},
        }
    out["declaration"] = (
        "qualified dataset s012-h005-3x; M=(1/2,0,0) is incommensurate with "
        "the 3x hexagonal supercell and is NOT reported or interpolated"
    )
    return out


def make_plots(outdir: Path, path_data, coupled, phonon_qualified, blocks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # bare magnon path
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    energies = np.asarray(path_data["energies_eV"]) * 1e3
    for branch in range(energies.shape[1]):
        ax.plot(path_data["x"], energies[:, branch], "b-", lw=1.2)
    for vx in path_data["vertex_x"]:
        ax.axvline(vx, color="k", lw=0.6)
    tick_labels = ["Gamma", "K", "M", "Gamma"]
    ax.set_xticks(path_data["vertex_x"], tick_labels)
    ax.set_ylabel("magnon energy (meV)")
    ax.set_title("CrI3 bare magnons (first-neighbour FM, declared model)")
    fig.tight_layout()
    fig.savefig(outdir / "s013_bare_magnon_path.png", dpi=160)
    plt.close(fig)

    # qualified bare phonons at covered points
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for i, name in enumerate(("gamma", "K")):
        freqs = np.asarray(phonon_qualified[name]["frequencies_thz"], float)
        jitter = np.linspace(-0.05, 0.05, freqs.size)
        ax.scatter(np.full_like(freqs, i) + jitter, freqs, s=10, color="C0")
    ax.set_xticks([0, 1], ["Gamma", "K"])
    ax.set_ylabel("frequency (THz)")
    ax.set_title("s012-h005-3x (symmetrized),\ncovered commensurate points only")
    fig.tight_layout()
    fig.savefig(outdir / "s013_bare_phonon_qualified.png", dpi=160)
    plt.close(fig)

    # coupled spectra: three magnon panels (Gamma, K, M) first, then one
    # panel per covered phonon momentum (STD-013-001)
    omega = np.asarray(coupled["omega_grid_eV"]) * 1e3
    n_magnon_panels = coupled["magnon"]["spectral"].shape[0]
    n_panels = n_magnon_panels + len(coupled["phonon"])
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.2))
    mesh_names = ["Gamma", "K", "M"]
    assert n_magnon_panels == len(mesh_names)
    for i_k in range(n_magnon_panels):
        ax = axes[i_k]
        spectral = coupled["magnon"]["spectral"][i_k]
        ax.pcolormesh(
            omega,
            np.arange(0.5, 2.5),
            spectral[:, 0, :].T,
            shading="auto",
            cmap="viridis",
        )
        bare = coupled["magnon"]["bare_energies_eV"][i_k] * 1e3
        ax.plot(omega, np.full_like(omega, 0.5), alpha=0)
        for b in bare:
            ax.axvline(b, color="r", lw=0.8, ls="--")
        ax.set_title(f"A(k, w), k={mesh_names[i_k]}")
        ax.set_xlabel("omega (meV)")
        ax.set_ylim(0.5, 2.5)
    for i_p, entry in enumerate(coupled["phonon"]):
        ax = axes[n_magnon_panels + i_p]
        modes = entry["modes"]
        # bare_energies_eV is already in eV (material.run_coupled_spectra
        # applies thz_to_ev once); plot meV directly (STD-013-002)
        bare_meV = np.atleast_1d(np.asarray(entry["bare_energies_eV"], float)) * 1e3
        ax.plot(omega, -2.0 * np.imag(entry["delta_d"][:, 0]), lw=0.9)
        for i_m in range(1, len(modes)):
            ax.plot(omega, -2.0 * np.imag(entry["delta_d"][:, i_m]), lw=0.9)
        for b in bare_meV:
            ax.axvline(b, color="r", lw=0.8, ls="--")
        q_label = (
            "Gamma"
            if np.allclose(entry["q"], 0.0)
            else ("K" if np.allclose(entry["q"], [1 / 3, 1 / 3, 0]) else "M")
        )
        ax.set_title(f"phonon Im deltaD, q={q_label}")
        ax.set_xlabel("omega (meV)")
    fig.tight_layout()
    fig.savefig(outdir / "s013_coupled_spectra.png", dpi=160)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# SPEC-013-002: internal-mesh / temperature convergence sweep
# ---------------------------------------------------------------------------

#: declared fractional direction of the split-mesh refinement (a*)
SWEEP_DIRECTION = np.array([1.0, 0.0, 0.0])
#: density levels: (name, satellite offsets in fractional units); the declared
#: production mesh is level "declared" (no satellites).  Every declared
#: internal magnon point p of the per-q phonon-loop meshes is replaced by
#: {p} U {p +/- offset * SWEEP_DIRECTION}, all subpoints sharing the declared
#: weight in equal shares.  The magnon-loop mesh
#: itself is COVERAGE-LOCKED (the internal phonon sum runs only over the
#: hash-joined covered q points) and is identical in every level; the
#: declared M+M->Gamma exclusion is kept in every level (the M external
#: phonon never gains points near M).
SWEEP_LEVELS = (
    ("declared", ()),
    ("split3", (0.01,)),
    ("split5", (0.01, 0.005)),
)
#: temperatures bracketing the declared production value 30 K
SWEEP_TEMPERATURES_K = (15.0, 30.0, 45.0)
#: declared relative-change tolerance for the convergence verdict
CONVERGENCE_TOL_REL = 0.05


def sweep_phonon_loop_meshes(phonon_loop_meshes, offsets):
    """Split every declared internal magnon point into the level's mesh."""
    if not offsets:
        return {
            q: ([p.copy() for p in mesh], np.asarray(w, float))
            for q, (mesh, w) in phonon_loop_meshes.items()
        }
    n_sat = 2 * len(offsets)
    share = 1.0 / (1.0 + n_sat)
    refined = {}
    for q, (mesh, weights) in phonon_loop_meshes.items():
        points = []
        wlist = []
        for p, w in zip(mesh, weights):
            points.append(np.asarray(p, float).copy())
            wlist.append(float(w) * share)
            for offset in offsets:
                for sign in (+1.0, -1.0):
                    points.append(
                        np.asarray(p, float) + sign * offset * SWEEP_DIRECTION
                    )
                    wlist.append(float(w) * share)
        refined[q] = (points, np.asarray(wlist, float))
    return refined


def _q_label(q):
    if np.allclose(q, 0.0):
        return "Gamma"
    if np.allclose(q, [1 / 3, 1 / 3, 0]):
        return "K"
    if np.allclose(q, [0.5, 0.0, 0.0]):
        return "M"
    return json.dumps(np.round(np.asarray(q, float), 6).tolist())


def _sweep_record(level_name, offsets, temperature, coupled, mesh_sizes):
    magnon = coupled["magnon"]
    return {
        "level": level_name,
        "split_offsets_fractional": [float(o) for o in offsets],
        "temperature_K": float(temperature),
        "phonon_mesh_sizes": dict(mesh_sizes),
        "magnon": {
            "spectral_max": float(np.max(magnon["spectral"])),
            "sigma_max_abs": float(np.max(np.abs(magnon["sigma"]))),
            "bare_meV": (np.asarray(magnon["bare_energies_eV"], float) * 1e3).tolist(),
        },
        "blocks": {
            f"{_q_label(entry['q'])}:+{'+'.join(entry['modes'])}": {
                "max_abs_sigma": float(np.max(np.abs(entry["sigma"]))),
                "max_abs_delta_d": float(np.max(np.abs(entry["delta_d"]))),
                "bare_meV": (
                    np.asarray(entry["bare_energies_eV"], float) * 1e3
                ).tolist(),
            }
            for entry in coupled["phonon"]
        },
    }


def run_convergence_sweep(
    magnon,
    blocks,
    settings,
    magnon_loop_mesh,
    phonon_loop_meshes,
    levels=SWEEP_LEVELS,
    temperatures=SWEEP_TEMPERATURES_K,
):
    """Repeat the coupled spectra over the declared density/T ladder."""
    records = []
    for level_name, offsets in levels:
        meshes = sweep_phonon_loop_meshes(phonon_loop_meshes, offsets)
        mesh_sizes = {_q_label(q): len(points) for q, (points, _w) in meshes.items()}
        for temperature in temperatures:
            level_settings = MaterialSettings(
                temperature_K=float(temperature),
                broadening_eV=settings.broadening_eV,
                omega_min_meV=settings.omega_min_meV,
                omega_max_meV=settings.omega_max_meV,
                omega_points=settings.omega_points,
            )
            coupled = run_coupled_spectra(
                magnon, blocks, level_settings, magnon_loop_mesh, meshes
            )
            records.append(
                _sweep_record(level_name, offsets, temperature, coupled, mesh_sizes)
            )
            print(
                f"sweep: level={level_name} T={temperature:g} K "
                f"magnon_spectral_max={records[-1]['magnon']['spectral_max']:.6e}"
            )
    return records


def _per_block_series(records, field, level, temperature):
    for record in records:
        if record["level"] == level and record["temperature_K"] == temperature:
            return {k: v[field] for k, v in record["blocks"].items()}
    raise KeyError(f"no sweep record for level={level} T={temperature}")


def _relative_change(coarser, finer):
    denom = max(abs(coarser), 1e-300)
    return abs(finer - coarser) / denom


def convergence_analysis(records, tolerance=CONVERGENCE_TOL_REL):
    """Density/T deltas per block + verdict against the declared tolerance."""
    level_names = [level for level, _ in SWEEP_LEVELS]
    production_temperature = 30.0
    coldest_temperature = min(float(r["temperature_K"]) for r in records)
    hottest_temperature = max(float(r["temperature_K"]) for r in records)
    density_deltas = {}
    for before, after in zip(level_names, level_names[1:]):
        step = {}
        coarse = _per_block_series(
            records, "max_abs_sigma", before, production_temperature
        )
        fine = _per_block_series(
            records, "max_abs_sigma", after, production_temperature
        )
        for label in coarse:
            step[label] = {
                "delta_max_abs_sigma_eV": abs(fine[label] - coarse[label]),
                "relative": _relative_change(coarse[label], fine[label]),
                "coarser_max_abs_sigma_eV": coarse[label],
                "finer_max_abs_sigma_eV": fine[label],
            }
        density_deltas[f"{before}->{after}"] = step
    finest = level_names[-1]
    cold = _per_block_series(records, "max_abs_sigma", finest, coldest_temperature)
    hot = _per_block_series(records, "max_abs_sigma", finest, hottest_temperature)
    temperature_deltas = {
        label: {
            "delta_max_abs_sigma_eV": abs(hot[label] - cold[label]),
            "relative": _relative_change(cold[label], hot[label]),
        }
        for label in cold
    }
    final_step = density_deltas[f"{level_names[-2]}->{level_names[-1]}"]
    worst_relative = max(step["relative"] for step in final_step.values())
    worst_label = max(final_step, key=lambda label: final_step[label]["relative"])
    magnon_cold = next(
        r["magnon"]["spectral_max"]
        for r in records
        if r["level"] == finest and float(r["temperature_K"]) == coldest_temperature
    )
    magnon_hot = next(
        r["magnon"]["spectral_max"]
        for r in records
        if r["level"] == finest and float(r["temperature_K"]) == hottest_temperature
    )
    converged = worst_relative < tolerance
    return {
        "declaration": (
            "SPEC-013-002 sweep: per-q phonon-loop magnon meshes refined "
            "declared -> split3 (exact point + 2 satellites at 0.01) -> split5 "
            "(+/-0.01, +/-0.005; exact point keeps an equal share of each "
            "declared weight); the magnon-loop mesh is coverage-locked to the "
            "hash-joined covered q points and identical in every level; the "
            "declared M+M->Gamma exclusion is kept in every level"
        ),
        "levels": [level for level, _ in SWEEP_LEVELS],
        "temperatures_K": [float(t) for t in SWEEP_TEMPERATURES_K],
        "tolerance_relative": float(tolerance),
        "density_deltas_at_30K": density_deltas,
        "temperature_deltas_finest_level": temperature_deltas,
        "magnon_spectral_max_finest_level": {
            "T_min_K": coldest_temperature,
            "T_max_K": hottest_temperature,
            "spectral_max_min_T": magnon_cold,
            "spectral_max_max_T": magnon_hot,
        },
        "verdict": {
            "converged": bool(converged),
            "worst_relative_change_split3_to_split5": float(worst_relative),
            "worst_block": worst_label,
        },
        "records": records,
    }


def convergence_statement(analysis):
    """Human-readable convergence (or its absence) for the prescription text."""
    verdict = analysis["verdict"]
    level_names = analysis["levels"]
    deltas = analysis["density_deltas_at_30K"]
    first_step = deltas[f"{level_names[0]}->{level_names[1]}"]
    first_worst = max(first_step, key=lambda label: first_step[label]["relative"])
    cold = analysis["magnon_spectral_max_finest_level"]
    if verdict["converged"]:
        finding = (
            f"convergence OBSERVED: the largest per-block max|Sigma| change "
            f"between the two densest meshes is "
            f"{verdict['worst_relative_change_split3_to_split5'] * 100:.2f}% "
            f"(block {verdict['worst_block']}), below the declared "
            f"{analysis['tolerance_relative'] * 100:g}% tolerance"
        )
    else:
        finding = (
            f"convergence NOT observed: the largest per-block max|Sigma| change "
            f"between the two densest meshes is "
            f"{verdict['worst_relative_change_split3_to_split5'] * 100:.2f}% "
            f"(block {verdict['worst_block']}), at or above the declared "
            f"{analysis['tolerance_relative'] * 100:g}% tolerance"
        )
    return (
        f"Internal-mesh/temperature convergence (SPEC-013-002 declared sweep, "
        f"production T = 30 K): refining the per-q phonon-loop magnon meshes "
        f"{level_names[0]} -> {level_names[1]} -> {level_names[2]} at 30 K "
        f"changes the largest per-block max|Sigma| by "
        f"{first_step[first_worst]['relative'] * 100:.2f}% "
        f"({level_names[0]} -> {level_names[1]}, block {first_worst}) and by "
        f"{verdict['worst_relative_change_split3_to_split5'] * 100:.2f}% "
        f"({level_names[1]} -> {level_names[2]}, block {verdict['worst_block']}); "
        f"{finding}. The magnon-loop mesh is coverage-locked (identical in all "
        f"levels), so the magnon channel carries the temperature dependence "
        f"only: over the bracket {cold['T_min_K']:g} K -> {cold['T_max_K']:g} K "
        f"at the densest mesh the magnon spectral maximum moves by "
        f"{abs(cold['spectral_max_max_T'] - cold['spectral_max_min_T']):.3e} "
        f"(from {cold['spectral_max_min_T']:.6e} to "
        f"{cold['spectral_max_max_T']:.6e}) and the largest per-block phonon "
        f"max|Sigma| change is "
        f"{max(s['relative'] for s in analysis['temperature_deltas_finest_level'].values()) * 100:.2f}%. "
        f"The declared M+M->Gamma exclusion is kept in every level."
    )


def write_spectra_npz(outdir: Path, path_data, coupled):
    """Numeric spectra archive (byte-reproducible: numpy 2.x writes fixed
    zip timestamps, verified by TEST-001)."""
    np.savez_compressed(
        outdir / "s013_spectra.npz",
        path_kpoints=path_data["kpoints"],
        path_x=path_data["x"],
        path_vertex_x=path_data["vertex_x"],
        path_energies_eV=path_data["energies_eV"],
        omega_grid_eV=coupled["omega_grid_eV"],
        magnon_kpoints=coupled["magnon"]["kpoints"],
        magnon_bare_eV=coupled["magnon"]["bare_energies_eV"],
        magnon_sigma=coupled["magnon"]["sigma"],
        magnon_spectral=coupled["magnon"]["spectral"],
        **{
            f"phonon_{i}_sigma": entry["sigma"]
            for i, entry in enumerate(coupled["phonon"])
        },
        **{
            f"phonon_{i}_delta_d": entry["delta_d"]
            for i, entry in enumerate(coupled["phonon"])
        },
        **{
            f"phonon_{i}_bare_eV": entry["bare_energies_eV"]
            for i, entry in enumerate(coupled["phonon"])
        },
    )


def build_summary(
    settings,
    blocks,
    skipped,
    path_data,
    phonon_qualified,
    coupled,
    convergence,
    thermal_prescription_text,
):
    summary = {
        "schema": "s013-material-run-summary-v2",
        "settings": settings.as_dict(),
        "covered_blocks": [
            {
                "label": b.label,
                "q": b.q.tolist(),
                "mode_index": b.mode_index,
                "frequency_thz": b.frequency_thz,
                "max_fd_error_meV_per_Ang": b.max_fd_error_meV_per_Ang,
            }
            for b in blocks
        ],
        "skipped_blocks": skipped,
        "bare_magnon_meV": {
            "gamma": (path_data["energies_eV"][0] * 1e3).tolist(),
            "K": (path_data["energies_eV"][60] * 1e3).tolist(),
            "M": (path_data["energies_eV"][120] * 1e3).tolist(),
            "max_meV": float(path_data["energies_eV"].max() * 1e3),
        },
        "bare_phonon": {
            name: {
                "frequencies_thz": phonon_qualified[name]["frequencies_thz"],
                "instabilities": phonon_qualified[name]["instabilities"],
            }
            for name in ("gamma", "K")
        },
        "coupled": {
            "magnon_bare_meV": (coupled["magnon"]["bare_energies_eV"] * 1e3).tolist(),
            "magnon_spectral_max": float(np.max(coupled["magnon"]["spectral"])),
            "phonon_max_abs_delta_d": [
                float(np.max(np.abs(e["delta_d"]))) for e in coupled["phonon"]
            ],
            "phonon_max_abs_sigma": [
                float(np.max(np.abs(e["sigma"]))) for e in coupled["phonon"]
            ],
            # spot-check numbers for the plot markers (eV * 1e3, as plotted)
            "phonon_bare_meV": [
                (np.atleast_1d(np.asarray(e["bare_energies_eV"], float)) * 1e3).tolist()
                for e in coupled["phonon"]
            ],
            "phonon_q": [_q_label(e["q"]) for e in coupled["phonon"]],
        },
        "convergence": convergence,
        "mixing_availability": MIXING_UNAVAILABLE_STATEMENT,
        "thermal_prescription": thermal_prescription_text,
    }
    return summary


def write_numeric_artifacts(
    outdir: Path,
    settings,
    blocks,
    skipped,
    path_data,
    phonon_qualified,
    coupled,
    magnon_loop_mesh,
    phonon_loop_meshes,
    fixture_path,
    manifest_extra,
    thermal_prescription_text,
    convergence,
):
    """Write s013_spectra.npz + s013_summary.json + s013_manifest.json.

    TEST-001 target: this writer is exercised twice into separate directories
    and the three files are compared byte-wise (plots excluded, see the test).
    The manifest is validated (and thereby gains its reference_validation
    statuses) before it is written.  The convergence section carries the full
    sweep records into the summary; the manifest carries everything except
    the raw per-(level, T) records (its completeness gate rejects empty lists,
    which the declared-level offset lists legitimately are).
    """
    write_spectra_npz(outdir, path_data, coupled)
    summary = build_summary(
        settings,
        blocks,
        skipped,
        path_data,
        phonon_qualified,
        coupled,
        convergence,
        thermal_prescription_text,
    )
    (outdir / "s013_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    manifest_convergence = (
        {k: v for k, v in convergence.items() if k != "records"}
        if convergence
        else convergence
    )
    manifest = provenance_manifest(
        settings,
        fixture_path,
        blocks,
        magnon_loop_mesh,
        phonon_loop_meshes,
        extra={**manifest_extra, "convergence": manifest_convergence},
        thermal_prescription=thermal_prescription_text,
    )
    validate_manifest(manifest, fixture_path)
    (outdir / "s013_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    return {"summary": summary, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="/STORAGE1/hexu/projects_bak/magnon_phonon_spectra/s013"
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(
            VAULT
            / "subprojects/validated-magnon-phonon-spectra-and-manuscript-evidence"
            / "evidence/s013"
        ),
    )
    parser.add_argument(
        "--fixture-out",
        default=None,
        help="write the TEST fixture (default: <writer>/tests/tests/data)",
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    evidence_dir = Path(args.evidence_dir)

    _complete_mode_tables()
    settings = MaterialSettings(
        temperature_K=DEFAULT_TEMPERATURE_K,
        broadening_eV=DEFAULT_BROADENING_EV,
    )
    blocks, skipped = load_cri3_blocks(S002_STORE, S002_EVIDENCE_DIR, MODE_TABLES)
    magnon = cri3_magnon_model()

    print(f"covered blocks: {len(blocks)}; skipped by IR cutoff: {len(skipped)}")
    path_data = bare_magnon_dispersion(magnon, n_points=61)
    phonon_qualified = bare_phonons_qualified()
    magnon_loop_mesh = [p.copy() for p in MAGNON_LOOP_MESH]
    phonon_loop_meshes = {
        q: ([p.copy() for p in mesh], np.asarray(w, float))
        for q, (mesh, w) in PHONON_LOOP_MESHES.items()
    }
    coupled = run_coupled_spectra(
        magnon, blocks, settings, magnon_loop_mesh, phonon_loop_meshes
    )

    # ---- SPEC-013-002 convergence sweep -----------------------------------
    sweep_records = run_convergence_sweep(
        magnon, blocks, settings, magnon_loop_mesh, phonon_loop_meshes
    )
    convergence = convergence_analysis(sweep_records)
    thermal_prescription_text = (
        THERMAL_PRESCRIPTION + " " + convergence_statement(convergence)
    )
    print("convergence verdict:", convergence["verdict"])

    # fixture (hermetic TEST-001/002 input) from the K acoustic-like block;
    # written first because the manifest binds its hash
    fixture_block = next(b for b in blocks if b.label == FIXTURE_LABEL)
    fixture_record = block_to_fixture(fixture_block)
    if args.fixture_out:
        fixture_path = Path(args.fixture_out)
    else:
        fixture_path = Path(__file__).resolve().parent.parent / (
            "tests/tests/data/s013_fixture_k_acoustic_like.json"
        )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(fixture_record, indent=1, sort_keys=True) + "\n")

    manifest_extra = {
        "code_revision": git_revision(),
        "s002_store": {
            "path": str(S002_STORE),
            "sha256": sha256_file(S002_STORE),
        },
        "s012_qualification_report": {
            "path": str(S012_QUALIFICATION),
            "sha256": sha256_file(S012_QUALIFICATION),
        },
        "s012_campaign": {
            "path": str(S012_CAMPAIGN),
            "campaign_id": "s012-h005-3x",
            "qualification": (
                "Story-012 qualified dataset (forces); bare phonons only; "
                "ingested via unfold_epc.phonon.campaign_phonon_modes "
                "(symmetrize=True) as in the Story-008 qualification"
            ),
        },
        "magnon_model": {
            "j0_meV": 1.529241586,
            "moment_mu_b": 3.0,
            "bonds": "Story-001 six directed Cr1-Cr2 nearest-neighbour bonds",
            "declaration": "first-neighbour only; longer-range exchange absent from qualified stores",
        },
        "phonon_qualified": {
            name: {
                "q": phonon_qualified[name]["q"],
                "symmetrization": phonon_qualified[name]["symmetrization"],
            }
            for name in ("gamma", "K")
        },
        "declaration": (
            "bare qualified phonon provenance (s012-h005-3x) and coupled "
            "vertex provenance (hash-locked det2/det3 mode evidence) are "
            "kept strictly separate; the det phonon side is NOT the "
            "qualified dataset (Story-012 noise-floor characterization); "
            "the named obtainable campaign that removes this gap is an "
            "S002 re-projection on the relaxed s012-h005-3x reference"
        ),
    }
    write_numeric_artifacts(
        outdir,
        settings,
        blocks,
        skipped,
        path_data,
        phonon_qualified,
        coupled,
        magnon_loop_mesh,
        phonon_loop_meshes,
        fixture_path,
        manifest_extra,
        thermal_prescription_text,
        convergence,
    )

    make_plots(outdir, path_data, coupled, phonon_qualified, blocks)

    # ---- vault evidence copies -------------------------------------------
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "s013_summary.json",
        "s013_manifest.json",
        "s013_bare_magnon_path.png",
        "s013_bare_phonon_qualified.png",
        "s013_coupled_spectra.png",
    ):
        shutil.copy2(outdir / name, evidence_dir / name)
    shutil.copy2(fixture_path, evidence_dir / "s013_fixture_k_acoustic_like.json")
    (evidence_dir / "s013_spectra_sha256.txt").write_text(
        f"{sha256_file(outdir / 's013_spectra.npz')}  s013_spectra.npz\n"
    )
    print("run complete:", outdir)


if __name__ == "__main__":
    main()
