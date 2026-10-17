"""Story-013 material run: bare + coupled CrI3 spectra with provenance.

Consumes only qualified/hash-verified stores (see story-013 Technical
Notes); no DFT is executed.  Bulk artifacts are written to
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
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


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

    # coupled spectra
    omega = np.asarray(coupled["omega_grid_eV"]) * 1e3
    fig, axes = plt.subplots(
        1, 2 + len(coupled["phonon"]), figsize=(4.0 * (2 + len(coupled["phonon"])), 3.2)
    )
    mesh_names = ["Gamma", "K", "M"]
    for i_k in range(coupled["magnon"]["spectral"].shape[0]):
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
        ax = axes[2 + i_p]
        modes = entry["modes"]
        spectral = np.zeros((len(modes), omega.size))
        from TB2J.magnon_phonon.scattering import thz_to_ev

        bare = thz_to_ev(np.asarray(entry["bare_energies_eV"], float))
        ax.plot(omega, -2.0 * np.imag(entry["delta_d"][:, 0]), lw=0.9)
        for i_m in range(1, len(modes)):
            ax.plot(omega, -2.0 * np.imag(entry["delta_d"][:, i_m]), lw=0.9)
        for b in np.atleast_1d(bare) * 1e3:
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
    coupled = run_coupled_spectra(
        magnon,
        blocks,
        settings,
        [p.copy() for p in MAGNON_LOOP_MESH],
        {
            q: ([p.copy() for p in mesh], np.asarray(w, float))
            for q, (mesh, w) in PHONON_LOOP_MESHES.items()
        },
    )

    # ---- numeric artifacts ------------------------------------------------
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

    make_plots(outdir, path_data, coupled, phonon_qualified, blocks)

    # fixture (hermetic TEST-001/002 input) from the K acoustic-like block
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

    summary = {
        "schema": "s013-material-run-summary-v1",
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
        },
        "mixing_availability": MIXING_UNAVAILABLE_STATEMENT,
        "thermal_prescription": THERMAL_PRESCRIPTION,
    }
    (outdir / "s013_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )

    manifest = provenance_manifest(
        settings,
        fixture_path,
        blocks,
        [p.copy() for p in MAGNON_LOOP_MESH],
        {
            q: ([p.copy() for p in mesh], np.asarray(w, float))
            for q, (mesh, w) in PHONON_LOOP_MESHES.items()
        },
        extra={
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
        },
    )
    (outdir / "s013_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )

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
