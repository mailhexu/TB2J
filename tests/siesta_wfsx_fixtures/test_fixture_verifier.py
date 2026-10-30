"""Black-box checks for the retained SIESTA WFSX reference fixtures."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent
DRIVER = FIXTURE_ROOT / "fixture_driver.py"


def _verify(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DRIVER), "verify", "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _copy_root(tmp_path: Path) -> Path:
    copied = tmp_path / "fixtures"
    shutil.copytree(
        FIXTURE_ROOT, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return copied


def _rewrite_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != manifest):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _edit_metadata(root: Path, fixture: str, edit) -> None:
    path = root / fixture / "fixture.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    edit(metadata)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_manifest(root)


def _assert_rejected(root: Path, code: str) -> None:
    result = _verify(root)
    assert result.returncode != 0
    assert f"{code}:" in result.stderr + result.stdout


def test_all_retained_fixtures_pass_the_independent_verifier() -> None:
    result = _verify(FIXTURE_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_verifier_rejects_hash_tampering(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    asset = root / "scalar_nspin1" / "o2.fullBZ.WFSX"
    asset.write_bytes(asset.read_bytes() + b"tampered")
    _assert_rejected(root, "HASH_MISMATCH")


def test_verifier_rejects_a_missing_required_wfsx(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    (root / "noncollinear_nspin4_full" / "o2.fullBZ.WFSX").unlink()
    _assert_rejected(root, "MISSING_FILE")


@pytest.mark.parametrize(
    ("fixture", "mutate", "error_code"),
    [
        (
            "scalar_nspin1",
            lambda metadata: metadata["spin"].update(nspin=2),
            "SPIN_MISMATCH",
        ),
        (
            "collinear_nspin2_full",
            lambda metadata: metadata["states"]["groups"][0]["state_ids"].pop(),
            "INCOMPLETE_STATES",
        ),
        (
            "collinear_nspin2_reduced",
            lambda metadata: metadata["kpoints"]["points"][0]["fractional"].__setitem__(
                2, 0.125
            ),
            "KPOINT_MISMATCH",
        ),
        (
            "collinear_nspin2_reduced",
            lambda metadata: metadata["kpoints"]["points"][0]["fractional"].__setitem__(
                2, metadata["kpoints"]["points"][0]["fractional"][2] + 5e-7
            ),
            "KPOINT_MISMATCH",
        ),
        (
            "scalar_nspin1",
            lambda metadata: metadata["states"]["groups"][0].update(kpoint_index=4),
            "KPOINT_MISMATCH",
        ),
        (
            "scalar_nspin1",
            lambda metadata: metadata["states"]["groups"][0].update(kpoint_index=999),
            "KPOINT_MISMATCH",
        ),
        (
            "scalar_nspin1",
            lambda metadata: metadata["states"]["groups"][0].update(group_index=1),
            "GROUP_MISMATCH",
        ),
        (
            "noncollinear_nspin4_full",
            lambda metadata: metadata["electrons"].update(
                fermi_energy_ev=metadata["electrons"]["fermi_energy_ev"] + 0.02
            ),
            "FERMI_MISMATCH",
        ),
    ],
    ids=[
        "spin",
        "state-completeness",
        "k-coordinate",
        "k-coordinate-near-tolerance",
        "wrong-group-k-link",
        "out-of-range-group-k-link",
        "wrong-group-ordinal",
        "fermi-energy",
    ],
)
def test_verifier_rejects_incompatible_fixture_metadata(
    tmp_path: Path, fixture: str, mutate, error_code: str
) -> None:
    root = _copy_root(tmp_path)
    _edit_metadata(root, fixture, mutate)
    _assert_rejected(root, error_code)


def test_verification_keeps_ingestion_version_separate_from_runtime(
    tmp_path: Path,
) -> None:
    root = _copy_root(tmp_path)
    _edit_metadata(
        root,
        "scalar_nspin1",
        lambda metadata: metadata["producer"].update(sisl_version="0.16.2.post0"),
    )
    result = _verify(root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_verifier_rejects_nonfinite_coefficient_after_rehash(tmp_path: Path) -> None:
    import numpy as np
    import sisl
    from scipy.io import FortranEOFError, FortranFile

    root = _copy_root(tmp_path)
    case = root / "scalar_nspin1"
    source = case / "o2.fullBZ.WFSX"
    ham = sisl.get_sile(str(case / "o2.HSX")).read_hamiltonian()
    metadata = json.loads((case / "fixture.json").read_text())
    fermi = metadata["electrons"]["fermi_energy_ev"]
    states = list(sisl.get_sile(str(source), lattice=ham.lattice).yield_eigenstate())
    residuals, norm_errors = [], []
    for state in states:
        k = state.info["k"]
        h = ham.Hk(k=k, gauge="r", format="array")
        overlap = ham.Sk(k=k, gauge="r", format="array")
        c = state.state.T
        residuals.append(np.max(np.abs(h @ c - (overlap @ c) * (state.eig - fermi))))
        norm_errors.append(
            np.max(np.abs(c.conj().T @ overlap @ c - np.eye(len(state.eig))))
        )
    maxima = {int(np.argmax(residuals)), int(np.argmax(norm_errors))}
    target = next(index for index in range(len(states)) if index not in maxima)
    records = []
    with FortranFile(source, "r") as stream:
        while True:
            try:
                records.append(stream.read_record(np.uint8))
            except FortranEOFError:
                break
    no_u = metadata["states"]["no_u"]
    coefficient_records = [
        index for index, record in enumerate(records) if record.nbytes == no_u * 8
    ]
    if len(coefficient_records) != len(states) * no_u:
        raise RuntimeError("unexpected scalar Fortran coefficient-record layout")
    records[coefficient_records[target * no_u]].view(np.float32)[0] = np.nan
    with FortranFile(source, "w") as stream:
        for record in records:
            stream.write_record(record)
    _rewrite_manifest(root)
    _assert_rejected(root, "NONFINITE")


@pytest.mark.parametrize(
    ("asset", "code"),
    [
        ("negative.selected.WFSX", "INCOMPLETE_KPOINTS"),
        ("duplicate.selected.WFSX", "DUPLICATE_KPOINT"),
    ],
)
def test_verifier_requires_complete_unique_k_coverage(
    tmp_path: Path, asset: str, code: str
) -> None:
    import copy

    import numpy as np
    import sisl

    root = _copy_root(tmp_path)
    case = root / "collinear_nspin2_full"
    source = root / "invalid_selected_subset" / asset
    shutil.copyfile(source, case / "o2.selected.WFSX")
    path = case / "fixture.json"
    metadata = json.loads(path.read_text())
    original_points = metadata["kpoints"]["points"]
    original_groups = metadata["states"]["groups"]
    original_map = metadata["kpoints"]["fold_map"]
    lattice = sisl.Lattice(metadata["structure"]["cell_angstrom"])
    reader = sisl.get_sile(str(source), lattice=lattice)
    records = list(reader.yield_eigenstate())
    metadata["states"]["nk"] = int(reader.read_sizes().nk)
    points, groups, mapping = [], [], []
    ham = sisl.get_sile(
        str((case / metadata["files"]["hsx"]).resolve())
    ).read_hamiltonian()
    residual, norm_error = 0.0, 0.0
    for ordinal, state in enumerate(records):
        coordinate = np.asarray(state.info["k"])
        original_index = next(
            i
            for i, point in enumerate(original_points)
            if np.allclose(coordinate, point["fractional"], atol=1e-8, rtol=0)
        )
        point = original_points[original_index]
        if point not in points:
            points.append(copy.deepcopy(point))
            mapping.append(copy.deepcopy(original_map[original_index]))
        group = copy.deepcopy(
            next(
                group
                for group in original_groups
                if group["kpoint_index"] == original_index
                and group["spin_id"] == state.info["spin"]
            )
        )
        group.update(group_index=ordinal, kpoint_index=points.index(point))
        groups.append(group)
        c = state.state.T
        h = ham.Hk(k=coordinate, spin=state.info["spin"], gauge="r", format="array")
        overlap = ham.Sk(k=coordinate, gauge="r", format="array")
        energies = state.eig - metadata["electrons"]["fermi_energy_ev"]
        residual = max(
            residual, float(np.max(np.abs(h @ c - (overlap @ c) * energies)))
        )
        norm_error = max(
            norm_error,
            float(np.max(np.abs(c.conj().T @ overlap @ c - np.eye(len(energies))))),
        )
    metadata["states"]["groups"] = groups
    metadata["kpoints"].update(points=points, fold_map=mapping)
    metadata["independent_checks"].update(
        max_generalized_eigenpair_residual_ev=residual,
        max_s_orthogonality_error=norm_error,
    )
    path.write_text(json.dumps(metadata))
    reduced_path = root / "collinear_nspin2_reduced" / "fixture.json"
    reduced = json.loads(reduced_path.read_text())
    reduced["kpoints"]["fold_map"] = mapping
    reduced_path.write_text(json.dumps(reduced))
    _rewrite_manifest(root)
    _assert_rejected(root, code)
