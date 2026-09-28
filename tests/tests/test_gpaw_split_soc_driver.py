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
    _write_leg,
    gen_exchange_gpaw_split_soc,
)
from TB2J.io_merge import Merger, merge, read_pickle


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
    # Rotation and the rank-six SpinIO merge must recover a physical tensor,
    # not just produce three pickles. Includes nonzero off-diagonal Jani/DMI.
    cell = np.eye(3) * 5.0
    data = SimpleNamespace(
        atomic_numbers=np.array([26, 26]),
        positions=np.array([[0, 0, 0], [1, 0, 0]]),
        cell=cell,
    )
    jani = np.array([[0.3, 0.12, -0.09], [0.12, -0.1, 0.08], [-0.09, 0.08, -0.2]])
    dmi = np.array([0.05, -0.07, 0.03])
    paths = []
    for name, theta, phi in (("x", 90, 0), ("y", 90, 90), ("z", 0, 0)):
        t, p = np.deg2rad([theta, phi])
        rotation = o_from_c(c_gpaw(t, p))
        axis = rotation @ np.array([0, 0, 1])
        exchange = {
            ((0, 0, 0), 0, 1): {
                "Jiso": 1.25,
                "jani": rotation.T @ jani @ rotation,
                "dmi": rotation.T @ dmi,
            },
            ((0, 0, 0), 1, 0): {
                "Jiso": 1.25,
                "jani": rotation.T @ jani @ rotation,
                "dmi": -(rotation.T @ dmi),
            },
        }
        path = tmp_path / name
        _write_leg(
            path, data, exchange, [0, 1], np.array([2.0, 2.0]), axis, rotation, 4.0
        )
        paths.append(str(path))
    merger = Merger(*paths)
    assert all(
        np.linalg.matrix_rank(matrix, tol=1e-2) == 6
        for matrix in merger.coeff_matrix.values()
    )
    merger.merge_Jiso()
    merger.merge_DMI()
    merger.merge_Jani()
    merger.standardize()
    key = ((0, 0, 0), 0, 1)
    assert merger.main_dat.exchange_Jdict[key] == pytest.approx(1.25, abs=1e-10)
    np.testing.assert_allclose(merger.main_dat.Jani_dict[key], jani, atol=1e-10)
    np.testing.assert_allclose(merger.main_dat.dmi_ddict[key], dmi, atol=1e-10)


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
        obj = read_pickle(str(path))
        axis = np.eye(3)[{"x": 0, "y": 1, "z": 2}[direction]]
        np.testing.assert_allclose(
            obj.spinat[:2] / np.linalg.norm(obj.spinat[:2], axis=1)[:, None],
            np.tile(axis, (2, 1)),
            atol=1e-14,
        )
        assert obj.exchange_Jdict
        provenance = obj.split_soc_provenance
        assert provenance == out["metadata"][direction]
        assert provenance["strength0_reference"]["sha256"] == digest
        study = provenance["band_window"]["convergence_study"]
        assert len(study["windows"]) == 2
        assert study["windows"][0]["nband"] < study["windows"][1]["nband"]
        assert study["windows"][1]["Jiso"]["[0,0,0]"] == pytest.approx(
            obj.exchange_Jdict[((0, 0, 0), 0, 1)], abs=1e-12
        )
        for artifact in (path / "exchange.out", path / "Multibinit" / "exchange.xml"):
            assert _artifact_provenance(artifact) == provenance
    merger = Merger(*(str(p) for p in out["leg_paths"].values()))
    assert all(
        np.linalg.matrix_rank(a, tol=1e-2) == 6 for a in merger.coeff_matrix.values()
    )
    merged = read_pickle(str(out["merged_path"]))
    assert merged.exchange_Jdict
    assert merged.split_soc_provenance["merge_mode"] == "three_legs"
    assert merged.split_soc_provenance["legs"] == out["metadata"]
    for artifact in (
        out["merged_path"] / "exchange.out",
        out["merged_path"] / "Multibinit" / "exchange.xml",
    ):
        assert _artifact_provenance(artifact) == merged.split_soc_provenance
    report = json.loads((tmp_path / "split" / "split_soc_report.json").read_text())
    assert report["metadata"] == out["metadata"]
    # Public generic merge must not silently inherit only the last z leg.
    generic_path = tmp_path / "generic_merge"
    merge(
        *(str(out["leg_paths"][name]) for name in ("x", "y", "z")),
        write_path=str(generic_path),
    )
    generic = read_pickle(str(generic_path))
    records = generic.split_soc_provenance["inputs"]
    assert [record["provenance"] for record in records] == [
        out["metadata"][name] for name in ("x", "y", "z")
    ]
    for artifact in (
        generic_path / "exchange.out",
        generic_path / "Multibinit" / "exchange.xml",
    ):
        assert _artifact_provenance(artifact) == generic.split_soc_provenance
    for key in merged.exchange_Jdict:
        assert generic.exchange_Jdict[key] == pytest.approx(
            merged.exchange_Jdict[key], abs=1e-12
        )
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
    for path in out["leg_paths"].values():
        jiso = read_pickle(str(path)).exchange_Jdict
        assert set(jiso) == expected
        assert max(abs(jiso[key] - reference[key]) for key in expected) < 1e-8
