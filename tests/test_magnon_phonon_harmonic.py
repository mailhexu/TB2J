"""Story 011: harmonic mixed spin/lattice Hessian solver (ADR-005).

TEST-001 controlled two-boson benchmark against independently derived
energies across a detuning sweep.
TEST-002 stability threshold crossing flagged, never regularized.
TEST-003 forbidden scalar-collinear control: rejected input, and a typed
zero-coupling Hessian returns bare modes with zero mixing.
TEST-004 magnetic/displacement weight transfer at resonance matches the
derived expressions.
"""

import numpy as np
import pytest

from TB2J.magnon_phonon.harmonic import (
    ControlledTwoBosonModel,
    HarmonicMixedModel,
    MixedSpinDisplacementHessian,
    hybrid_modes,
)

POINTS = np.array([[0.0, 0.0, 0.0]])


def _derived_energies(a, b, g):
    disc = np.sqrt(((a**2 - b**2) / 2) ** 2 + 4 * a * b * g**2)
    return np.sort(np.sqrt([(a**2 + b**2) / 2 + disc, (a**2 + b**2) / 2 - disc]))


def _derived_share(a, b, g, E):
    r = (E**2 - a**2) / (2 * g * b)
    Wm = 1 + (a / E) ** 2
    Wp = r**2 * (1 + (b / E) ** 2)
    return Wm / (Wm + Wp)


def test_controlled_model_detuning_sweep():
    """TEST-001: solver energies equal the derived closed form across detuning."""
    beta = 2.0
    for ratio in [0.4, 0.7, 1.0, 1.6, 2.5, 4.0]:
        alpha = ratio * beta
        for gscale in [0.05, 0.2, 0.45]:
            g = gscale * np.sqrt(alpha * beta) * 0.9
            model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
            res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
            derived = _derived_energies(alpha, beta, g)
            assert np.allclose(np.sort(res.energies[0]), derived, atol=1e-10)
            assert res.is_stable[0]


def test_stability_threshold_flagged_not_regularized():
    """TEST-002: 4g^2 >= alpha beta is flagged with NaN energies and zero shift."""
    alpha = beta = 2.0
    g_above = np.sqrt(alpha * beta / 4) * 1.05
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g_above)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert not res.is_stable[0]
    assert res.diagnostics.stability_flag[0]
    assert np.all(np.isnan(res.energies[0]))
    assert res.diagnostics.shift[0] == 0.0
    g_below = np.sqrt(alpha * beta / 4) * 0.95
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g_below)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert res.is_stable[0]
    assert not res.diagnostics.stability_flag[0]


def test_forbidden_scalar_collinear_input_rejected():
    """TEST-003a: a scalar-collinear-derived Hessian is rejected at construction."""
    with pytest.raises(ValueError, match="scalar"):
        MixedSpinDisplacementHessian(
            n_mag_sites=1,
            displacement_labels=((0, 2),),
            producer="scalar_collinear_exchange_striction",
            magnon_bare=lambda pts: np.zeros((len(pts), 2, 2)),
            phonon_bare=lambda pts: np.ones((len(pts), 1, 1)),
            magnon_phonon_coupling=lambda pts: np.ones((len(pts), 1, 1)),
            provenance={"source": "single collinear transverse measurement"},
        )


def test_scalar_collinear_forbidden_control_returns_bare_modes():
    """TEST-003b: scalar-collinear control produces bare modes, never mixing."""
    alpha, beta = 2.0, 1.0

    def magnon_bare(points):
        result = np.zeros((len(points), 2, 2))
        result[:, 0, 0] = alpha
        result[:, 1, 1] = alpha
        return result

    mixed = MixedSpinDisplacementHessian.scalar_collinear_forbidden_control(
        n_mag_sites=1,
        displacement_labels=((0, 2),),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.full((len(pts), 1, 1), beta**2),
        provenance={"source": "scalar-collinear exchange input"},
    )
    res = hybrid_modes(POINTS, mixed)
    energies = np.sort(res.energies[0])
    assert np.allclose(energies, np.sort([alpha, beta]), atol=1e-12)
    # each bare mode carries unit weight on its own species
    order = np.argsort(res.energies[0])
    assert np.allclose(res.displacement_weight[0][order], [1.0, 0.0], atol=1e-12)
    assert np.allclose(res.magnetic_weight[0][order], [0.0, 1.0], atol=1e-12)
    assert np.allclose(
        res.magnetic_weight[0] + res.displacement_weight[0], 1.0, atol=1e-12
    )


def test_weight_transfer_at_resonance():
    """TEST-004: weight shares match the derived expressions on and off resonance."""
    beta = 1.5
    for ratio in [0.6, 1.0, 2.2]:
        alpha = ratio * beta
        for gscale in [0.1, 0.35]:
            g = gscale * np.sqrt(alpha * beta)
            model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
            res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
            derived = _derived_energies(alpha, beta, g)
            for mode in range(2):
                share = _derived_share(alpha, beta, g, derived[mode])
                t = res.paraunitary[0][:, mode]
                wm = abs(t[0]) ** 2 + abs(t[2]) ** 2
                wp = abs(t[1]) ** 2 + abs(t[3]) ** 2
                assert abs(res.magnetic_weight[0][mode] - share) < 1e-9
                assert abs(wm / (wm + wp) - share) < 1e-9
                assert abs(res.displacement_weight[0][mode] - (1 - share)) < 1e-9
    # exact resonance: both branches share weights equally
    alpha = beta = 2.0
    g = 0.4
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert np.allclose(res.magnetic_weight[0], 0.5, atol=1e-12)
    assert np.allclose(res.displacement_weight[0], 0.5, atol=1e-12)


def test_hybrid_cli_manifest_result():
    """The hybrid CLI calculation accepts the controlled JSON-manifest shape."""
    from TB2J.magnon_phonon.cli import _run_hybrid

    result = _run_hybrid(
        {
            "calculation": "hybrid",
            "model": {
                "type": "controlled_two_boson",
                "alpha_eV": 2.0,
                "beta_eV": 1.0,
                "g_eV": 0.1,
            },
            "points": [[0.0, 0.0, 0.0]],
        }
    )
    assert result["calculation"] == "hybrid"
    assert result["is_stable"] == [True]
    assert len(result["energies_eV"][0]) == 2


def test_hybrid_cli_accepts_typed_npz_hessian(tmp_path):
    """The CLI loads pre-supplied typed Hessian blocks outside the kernel."""
    from TB2J.magnon_phonon.cli import _run_hybrid

    path = tmp_path / "mixed.npz"
    np.savez(
        path,
        points=POINTS,
        magnon_bare=np.array([[[2.0, 0.0], [0.0, 2.0]]]),
        phonon_bare=np.array([[[1.0]]]),
        coupling=np.array([[[0.1]]]),
    )
    result = _run_hybrid(
        {
            "calculation": "hybrid",
            "model": {
                "type": "mixed_hessian_npz",
                "npz_path": str(path),
                "n_mag_sites": 1,
                "spin_coordinate_labels": [[0, "local-transverse-complex"]],
                "displacement_labels": [[0, 2]],
                "producer": "controlled_mixed_torque",
                "provenance": {"source": "test controlled Hessian"},
            },
            "points": POINTS.tolist(),
        }
    )
    assert result["is_stable"] == [True]
    assert np.allclose(np.sort(result["energies_eV"][0]), _derived_energies(2, 1, 0.1))


def test_model_hybrid_modes_facade():
    """The model facade exposes the requested hybrid_modes(points, Hessian) API."""
    model = ControlledTwoBosonModel(alpha_eV=2.0, beta_eV=1.0, g_eV=0.15)
    mixed = model.to_mixed_hessian()
    result = HarmonicMixedModel(mixed).hybrid_modes(POINTS, mixed)
    assert np.allclose(result.energies[0], _derived_energies(2.0, 1.0, 0.15))


def test_hybrid_cli_writes_strict_json_for_unstable_modes(monkeypatch, capsys):
    """Flagged NaN numerical outputs are emitted as standards-compliant JSON null."""
    import io
    import json
    import sys

    from TB2J.magnon_phonon.cli import main

    manifest = {
        "calculation": "hybrid",
        "model": {
            "type": "controlled_two_boson",
            "alpha_eV": 2.0,
            "beta_eV": 2.0,
            "g_eV": 1.05,
        },
        "points": [[0.0, 0.0, 0.0]],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(manifest)))
    main(["hybrid", "-"])
    result = json.loads(capsys.readouterr().out)
    assert result["is_stable"] == [False]
    assert result["energies_eV"] == [[None, None]]
    assert result["diagnostics"]["shift"] == [0.0]
