"""Real second-variational GPAW split-SOC driver gates (opt-in fixture)."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from TB2J.interfaces.gpaw_spinor_split_soc import c_gpaw, o_from_c
from TB2J.interfaces.gpaw_split_soc import (
    _contour_second_order_shift,
    gen_exchange_gpaw_split_soc,
)
from TB2J.io_merge import read_pickle


def _artifact_provenance(path):
    text = (
        path.read_text()
        if path.suffix == ".out"
        else "".join(ElementTree.parse(path).getroot().itertext())
    )
    return json.loads(
        next(
            line.split(": ", 1)[1]
            for line in text.splitlines()
            if line.startswith("split_soc_provenance: ")
        )
    )


@pytest.mark.parametrize("nz", [12, 30])
@pytest.mark.parametrize("fermi", [0.0, 6.3])
def test_two_level_contour_mae_matches_exact_curvature(nz, fermi):
    # Lowest eigenvalue of [[-1, lambda*w], [lambda*w, 1]] is
    # -sqrt(1 + lambda**2*w**2); its lambda**2 coefficient is -w**2/2.
    w = 0.1
    leg = SimpleNamespace(
        eigenvalues_strength0=np.array([[-1.0, 1.0]]) + fermi,
        w_soc=np.array([[[0.0, w], [w, 0.0]]]),
        weights=np.array([1.0]),
        efermi=fermi,
    )
    assert _contour_second_order_shift(leg, nz=nz, smearing_eV=0.05) == pytest.approx(
        -w * w / 2, abs=1e-7
    )


def test_three_leg_merge_recovers_known_anisotropic_tensor(tmp_path):
    from TB2J.interfaces.gpaw_split_soc import _write_merged
    from TB2J.Jtensor import decompose_J_tensor
    from TB2J.split_soc_kernel import merge_transverse_legs, rotate_transverse_leg

    cell = np.eye(3) * 5.0
    data = SimpleNamespace(
        atomic_numbers=np.array([26, 26]),
        positions=np.array([[0, 0, 0], [1, 0, 0]]),
        cell=cell,
    )
    tensor = np.array([[1.55, 0.15, -0.16], [0.09, 1.15, 0.01], [-0.02, 0.15, 1.05]])
    key = ((0, 0, 0), 0, 1)
    reverse = ((0, 0, 0), 1, 0)
    legs = {}
    for name, theta, phi in (("x", 90, 0), ("y", 90, 90), ("z", 0, 0)):
        rotation = o_from_c(c_gpaw(*np.deg2rad([theta, phi])))
        measured = rotation.T @ tensor @ rotation
        measured[2, :] = 0.0
        measured[:, 2] = 0.0
        entries = {
            key: {"J_leg": measured, "mask_residual": 0.0, "frame": {"n": 2}},
            reverse: {"J_leg": measured.T, "mask_residual": 0.0, "frame": {"n": 2}},
        }
        legs[name] = {
            "exchange": rotate_transverse_leg(entries, rotation, "xyz".index(name))
        }
    combined = merge_transverse_legs(legs)
    np.testing.assert_allclose(combined["exchange"][key]["tensor"], tensor, atol=1e-12)
    np.testing.assert_allclose(
        combined["exchange"][reverse]["tensor"], tensor.T, atol=1e-12
    )
    _write_merged(
        tmp_path / "merged",
        data,
        combined["exchange"],
        [0, 1],
        np.array([2.0, 2.0]),
        4.0,
        {"merge_mode": "raw_rank_nine"},
    )
    output = read_pickle(str(tmp_path / "merged"))
    jiso, dmi, jani = decompose_J_tensor(tensor)
    assert output.exchange_Jdict[key] == pytest.approx(jiso, abs=1e-12)
    np.testing.assert_allclose(output.dmi_ddict[key], dmi, atol=1e-12)
    np.testing.assert_allclose(output.Jani_dict[key], jani, atol=1e-12)


@pytest.mark.skipif(
    not os.environ.get("TB2J_GPAW_SPLIT_SOC_GPW"),
    reason="requires a converged legacy GPAW collinear no-SOC checkpoint",
)
def test_real_three_leg_merge_and_mae(tmp_path):
    pytest.importorskip("gpaw")
    from gpaw import GPAW
    from gpaw.spinorbit import soc_eigenstates

    checkpoint = Path(os.environ["TB2J_GPAW_SPLIT_SOC_GPW"])
    calc = GPAW(str(checkpoint), legacy_gpaw=True)
    out = gen_exchange_gpaw_split_soc(
        str(checkpoint),
        output_path=tmp_path / "split",
        Rpts=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        Rcut=4.0,
        nz=12,
        smearing_eV=0.1,
        magnetic_sites=[0, 1],
    )
    assert set(out["leg_paths"]) == {"x", "y", "z"}
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for direction, path in out["leg_paths"].items():
        with np.load(path / "split_soc_leg.npz") as leg:
            axis = "xyz".index(direction)
            assert leg["J_leg"].shape[1:] == (3, 3)
            np.testing.assert_allclose(leg["J_leg"][:, axis, :], 0, atol=1e-12)
            np.testing.assert_allclose(leg["J_leg"][:, :, axis], 0, atol=1e-12)
        provenance = out["metadata"][direction]
        assert provenance["strength0_reference"]["sha256"] == digest
        study = provenance["band_window"]["convergence_study"]
        assert study["windows"][0]["nband"] < study["windows"][1]["nband"]
    merged = read_pickle(str(out["merged_path"]))
    assert merged.exchange_Jdict
    assert merged.split_soc_provenance["merge_mode"] == "raw_rank_nine"
    assert merged.split_soc_provenance["legs"] == out["metadata"]
    assert merged.split_soc_provenance["diagnostics"]["min_rank"] == 9
    for artifact in (
        out["merged_path"] / "exchange.out",
        out["merged_path"] / "Multibinit" / "exchange.xml",
    ):
        assert _artifact_provenance(artifact) == merged.split_soc_provenance
    report = json.loads((tmp_path / "split" / "split_soc_report.json").read_text())
    assert report["merge_diagnostics"]["min_rank"] == 9
    for direction, angles in {"x": (90, 0), "y": (90, 90), "z": (0, 0)}.items():
        direct = soc_eigenstates(
            calc, theta=angles[0], phi=angles[1], projected=False
        ).calculate_band_energy()
        assert out["mae"][direction]["band_energy_eV"] == direct
    assert out["mae"]["z"]["relative_to_z_eV"] == 0.0
    assert all(np.isfinite(row["relative_to_z_eV"]) for row in out["mae"].values())
    assert all(row["contour_within_tolerance"] for row in out["mae"].values())
    assert all(np.isfinite(row["contour_residual_eV"]) for row in out["mae"].values())


@pytest.mark.skipif(
    not os.environ.get("TB2J_GPAW_SPLIT_SOC_GPW"),
    reason="requires a converged legacy GPAW collinear no-SOC checkpoint",
)
def test_scale_zero_matches_current_collinear_exporter_shell_by_shell(tmp_path):
    pytest.importorskip("gpaw")
    from gpaw import GPAW

    from TB2J.interfaces.gpaw_projector import gen_exchange_gpaw

    calc = GPAW(os.environ["TB2J_GPAW_SPLIT_SOC_GPW"], legacy_gpaw=True)
    _, reference = gen_exchange_gpaw(
        calc,
        atoms=calc.get_atoms(),
        output_path=tmp_path / "collinear",
        index_magnetic_atoms=[0, 1],
        Rcut=3.0,
        nz=12,
    )
    ref_output = read_pickle(str(tmp_path / "collinear"))
    expected = {
        key
        for key, (_vec, dist) in ref_output.distance_dict.items()
        if 1e-6 < dist < 3.0
    }
    out = gen_exchange_gpaw_split_soc(
        calc,
        output_path=tmp_path / "split",
        Rcut=3.0,
        nz=12,
        smearing_eV=0.05,
        magnetic_sites=[0, 1],
        scale=0.0,
        vertex_component="delta_xc",
    )
    merged = read_pickle(str(out["merged_path"])).exchange_Jdict
    assert set(merged) == expected
    assert max(abs(merged[key] - reference[key]) for key in expected) < 1e-8
