"""Story 007 additive frozen-spiral response API: persisted v2 bundle -> CLI JSON.

The orchestration layer (``TB2J.spiral_response``) integrates the real
TBUpy v2 sidecar loader with the TB2J frozen-response primitives:

* ``TB2J.spiral_first_order.frozen_band_gradient``  (local-angle gradient),
* ``TB2J.spiral_pitch.frozen_pitch_slope``          (frozen-q response),
* ``TB2J.spiral_nonstationary.nonstationary_curvature``,

and emits a standalone JSON report (no SpinIO object anywhere) via
``python -m TB2J.scripts.tb2j_spiral_response``.

The bundle is built END-TO-END through the real v2 machinery: the real
:class:`tbupy.planar_spiral.PlanarSpiralProvider` at a deliberately
nonstationary pitch, the real planar spinor SCF
(:func:`tbupy.planar_spiral_scf.run_planar_spiral_scf`), the real v2
factory :func:`tbupy.spiral_state.planar_state_from_scf` (explicit
constraint / field_symmetry / field_role / field_rotation_policy), the
real saver and loader (``*.spiral.nc``, schema tag 2) -- and the CLI runs
in a fresh interpreter via ``python -m``.

The report is verified against independent finite-difference oracles on
the same reference:

* same-reference local-angle FD for ``frozen_band_gradient`` (per-atom
  beta/delta channels, eV/rad, field-only rotation -- basis fixed)

* frozen-q FD (fixed variational occupations) for
  ``frozen_band_pitch_slope`` (eV per fractional q component, including
  the moving-basis dS/dq of the nonorthogonal overlap)

* second-difference along the same vertices versus the co-rotating
  contraction of the nonstationary lab-angle Hessian (cell-major sites).

``self_consistent_pitch_slope`` must appear only when a matched-q TBUpy
record is supplied
without it the key is absent (never fabricated).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# model: two-atom planar spiral ring, deliberately nonstationary
# ---------------------------------------------------------------------------

NCELL = 6
Q_FRAC = np.array([1.0 / 6.0, 0.0, 0.0])  # m=1: NOT tuned to a stationary pitch
TAUS = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])  # per atom
PHIS = np.array([0.0, 0.4])  # noncollinear planar offsets
B_LOCAL = np.array([3.0, 3.0])  # full up/dn splitting, eV
NEL = 3.0  # partially filled dn channel -> genuinely q-sensitive metal
WIDTH = 0.05  # frozen-occupation smearing width, eV
T1, T2, T3 = 0.35, 0.25, 0.15
EPS = (0.05, -0.05)
OVERLAP_S = 0.06  # nearest-neighbour overlap -> moving basis (dS/dq != 0)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _reference_arrays():
    """Spin-scalar ``(Rlist, HR, SR)`` of the nonorthogonal dimer chain."""
    classes = [
        (0, 0, 1, T1),
        (0, 0, -1, T1),
        (1, 1, 1, T1),
        (1, 1, -1, T1),
        (0, 1, 0, T2),
        (1, 0, 0, T2),
        (0, 1, 1, T3),
        (0, 1, -1, T3),
        (1, 0, 1, T3),
        (1, 0, -1, T3),
    ]
    rs = sorted({R for (_, _, R, _) in classes} | {0})
    norb = 2
    HR = np.zeros((len(rs), norb, norb))
    SR = np.zeros((len(rs), norb, norb))
    SR[rs.index(0)] = np.eye(norb)
    for mu, eps in enumerate(EPS):
        HR[rs.index(0), mu, mu] += eps
    for mu, nu, R, amp in classes:
        HR[rs.index(R), mu, nu] += amp
        SR[rs.index(R), mu, nu] += OVERLAP_S
    rlist = np.zeros((len(rs), 3), dtype=np.int64)
    rlist[:, 0] = rs
    return rlist, HR, SR


def _kmesh():
    """Half-shifted commensurate mesh (m = q*NCELL is odd)."""
    k1 = (np.arange(NCELL) + 0.5) / NCELL
    kpts = np.column_stack((k1, np.zeros(NCELL), np.zeros(NCELL)))
    return kpts, np.full(NCELL, 1.0 / NCELL)


def _base_provider(q_frac):
    from tbupy.planar_spiral import PlanarSpiralConfig, PlanarSpiralProvider

    rlist, HR, SR = _reference_arrays()
    config = PlanarSpiralConfig(
        np.asarray(q_frac, dtype=float), TAUS, PHIS, np.arange(2, dtype=int)
    )
    return PlanarSpiralProvider(HR, SR, rlist, config, B_LOCAL, is_orthogonal=False)


def _persist_bundle(path):
    """Real v2 pipeline: planar SCF -> v2 factory -> sidecar on disk."""
    from tbupy.planar_spiral_scf import run_planar_spiral_scf
    from tbupy.spiral_state import planar_state_from_scf, save_spiral_state

    provider = _base_provider(Q_FRAC)
    kpts, kweights = _kmesh()
    scf = run_planar_spiral_scf(provider, kpts, kweights, nel=NEL, width=WIDTH)
    state = planar_state_from_scf(
        provider,
        scf,
        kpts,
        kweights,
        WIDTH,
        constraint={"kind": "none"},
        field_symmetry={
            "local_field_axis": "y",
            "field_axis_policy": "co_rotating_local",
            "spectral_projector_frame": "rotating",
            "rho_role": "same_frame_density",
        },
        field_role="external",
        field_rotation_policy="co_rotating_local",
        density_tolerance=1e-8,
    )
    save_spiral_state(path, state)
    return path


# ---------------------------------------------------------------------------
# oracles: same-reference FD on the rebuilt folded pencil
# ---------------------------------------------------------------------------


def _frozen_energy(bundle, perturb=None, q_frac=None):
    """Frozen-band energy sum_k w_k f_nk eps_nk with occupations FIXED to
    the reference.  ``perturb`` adds a Hermitian operator to every folded
    Hq (local-angle FD along a declared vertex)
    ``q_frac`` overrides the
    spiral wavevector (frozen-q FD -- occupations unchanged)."""
    from TB2J.spiral_response import reference_eigenpairs

    provider = bundle.response_provider(q_frac=q_frac)
    evals, _ = reference_eigenpairs(provider, bundle.kpts, extra=perturb)
    return float(np.sum(bundle.kweights[:, None] * bundle.occupations * evals))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def persisted_bundle(tmp_path_factory):
    """Deliberately-nonstationary v2 sidecar from the real v2 factory."""
    return _persist_bundle(
        tmp_path_factory.mktemp("bundle") / "nonstationary.spiral.nc"
    )


def _run_cli(bundle_path, out_path, extra=()):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "TB2J.scripts.tb2j_spiral_response",
            "--spiral-state",
            str(bundle_path),
            "--output",
            str(out_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_cli_roundtrip_json_keys(persisted_bundle, tmp_path):
    """Actual ``python -m`` CLI invocation on the persisted v2 bundle."""
    out = tmp_path / "response.json"
    proc = _run_cli(persisted_bundle, out)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())

    grad = report["frozen_band_gradient"]
    assert len(grad["g_beta"]) == 2
    assert len(grad["g_delta"]) == 2
    assert "eV" in grad["units"] and "rad" in grad["units"]

    pitch = report["frozen_band_pitch_slope"]
    assert len(pitch["frozen_band_pitch_slope"]) == 3
    assert "eV" in pitch["units"] and "fractional q" in pitch["units"]

    # deliberately nonstationary: the frozen gradient must NOT vanish
    assert np.linalg.norm(grad["g_beta"]) + np.linalg.norm(grad["g_delta"]) > 1e-6
    assert np.linalg.norm(pitch["frozen_band_pitch_slope"]) > 1e-6

    # no matched-q record supplied -> the certified slope is absent
    assert "self_consistent_pitch_slope" not in report
    # nonstationary curvature summary (full blocks stay on the result)
    curv = report["nonstationary_curvature"]
    assert isinstance(curv, dict)
    assert "local_curvature" in curv and "units" in curv
    # full metadata: provenance, units, sign conventions, gates
    meta = report["metadata"]
    assert meta["provenance"]["gauge"] == "planar_y"
    assert meta["provenance"]["electron_count"] == pytest.approx(NEL)
    assert meta["provenance"]["field_role"] == "external"
    assert meta["provenance"]["field_rotation_policy"] == "co_rotating_local"
    assert "eV/rad" in json.dumps(report)


def test_gradient_matches_same_reference_local_angle_fd(persisted_bundle):
    """g_beta/g_delta == central FD of the frozen band energy along the
    declared folded vertex directions (same reference, same occupations)."""
    from TB2J.spiral_response import (
        compute_frozen_response,
        load_response_bundle,
        planar_field_vertices,
    )

    bundle = load_response_bundle(persisted_bundle)
    v1_beta, v1_delta = planar_field_vertices(bundle)
    h = 1e-4
    result = compute_frozen_response(persisted_bundle)
    for iatom in range(2):
        for channel, v1 in (("beta", v1_beta), ("delta", v1_delta)):
            e_p = _frozen_energy(bundle, perturb=+h * v1[iatom])
            e_m = _frozen_energy(bundle, perturb=-h * v1[iatom])
            fd = (e_p - e_m) / (2 * h)
            got = (
                result.frozen_band_gradient.g_beta[iatom]
                if channel == "beta"
                else result.frozen_band_gradient.g_delta[iatom]
            )
            assert got == pytest.approx(fd, abs=1e-6), (iatom, channel, got, fd)


def test_pitch_slope_matches_frozen_q_fd(persisted_bundle):
    """dE/dq at frozen occupations, including the moving-basis dS/dq."""
    from TB2J.spiral_response import compute_frozen_response, load_response_bundle

    bundle = load_response_bundle(persisted_bundle)
    result = compute_frozen_response(persisted_bundle)
    h = 1e-4
    for a in range(3):
        q_p = bundle.q_frac.copy()
        q_m = bundle.q_frac.copy()
        q_p[a] += h
        q_m[a] -= h
        e_p = _frozen_energy(bundle, q_frac=q_p)
        e_m = _frozen_energy(bundle, q_frac=q_m)
        fd = (e_p - e_m) / (2 * h)
        assert result.frozen_band_pitch_slope.vector[a] == pytest.approx(
            fd, abs=1e-6
        ), a


def _field_rotation_operator(bundle, iatom, channel, angle):
    """Exact folded on-site field for atom ``iatom`` rotated by ``angle``.

    beta rotates the local field inside the spiral plane
    (``Bf (sz cos(angle) + sx sin(angle))``), delta tilts it out of the
    plane (``Bf (sz cos(angle) + sy sin(angle))``).  Returned as the
    ADDITIVE operator relative to the reference pencil (the reference
    ``Bf sz`` block is subtracted), for the exact-angle-path finite
    difference whose second derivative includes the quadratic field
    term that ``v2_diagonal`` carries.
    """
    from TB2J.spiral_response import planar_field_vertices  # noqa: F401

    state = bundle.state
    atom_of_orbital = bundle.orbital_to_atom
    b_local = np.asarray(state.B_local, dtype=float)
    norb = len(b_local)
    n = 2 * norb
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    perpendicular = sx if channel == "beta" else sy
    op = np.zeros((n, n), dtype=complex)
    for mu in range(norb):
        if int(atom_of_orbital[mu]) != iatom:
            continue
        bf = 0.5 * b_local[mu]
        sl = slice(2 * mu, 2 * mu + 2)
        op[sl, sl] += bf * ((np.cos(angle) - 1.0) * sz + np.sin(angle) * perpendicular)
    return op


def test_curvature_matches_second_difference(persisted_bundle):
    """Folded second difference of the frozen band energy along the EXACT
    rotated-field path (including the quadratic field term) == the
    periodic contraction of the nonstationary Hessian blocks and the
    aggregated ``local_curvature``."""
    from TB2J.spiral_response import compute_frozen_response, load_response_bundle

    bundle = load_response_bundle(persisted_bundle)
    result = compute_frozen_response(persisted_bundle, translation_cutoff=NCELL)
    blocks = result.nonstationary_curvature.blocks
    local = np.asarray(result.nonstationary_curvature.local_curvature)
    e0 = _frozen_energy(bundle)
    h = 1e-3
    for iatom in range(2):
        for ichannel, (channel, key) in enumerate(
            (("beta", ("b", "b")), ("delta", ("d", "d")))
        ):
            e_p = _frozen_energy(
                bundle,
                perturb=_field_rotation_operator(bundle, iatom, channel, +h),
            )
            e_m = _frozen_energy(
                bundle,
                perturb=_field_rotation_operator(bundle, iatom, channel, -h),
            )
            fd2 = (e_p + e_m - 2 * e0) / h**2
            co_rotating = np.zeros(NCELL * 2)
            co_rotating[np.arange(NCELL) * 2 + iatom] = 1.0
            contracted = float(co_rotating @ np.asarray(blocks[key]) @ co_rotating)
            # _frozen_energy is the per-primitive-cell energy (kweights 1/N),
            # so the per-cell FD equals the total finite-ring contraction
            # divided by the ring size (Main-adjudicated normalization)
            assert fd2 == pytest.approx(contracted / NCELL, rel=2e-3, abs=1e-4), (
                iatom,
                channel,
                fd2,
                contracted / NCELL,
            )
            assert fd2 == pytest.approx(local[iatom, ichannel], rel=2e-3, abs=1e-4), (
                iatom,
                channel,
                fd2,
                local[iatom, ichannel],
            )
    bb = np.asarray(blocks[("b", "b")])
    bd = np.asarray(blocks[("b", "d")])
    dd = np.asarray(blocks[("d", "d")])
    assert np.allclose(bb, bb.T, atol=1e-8)
    assert np.allclose(dd, dd.T, atol=1e-8)
    assert np.allclose(bd, np.asarray(blocks[("d", "b")]).T, atol=1e-8)


def test_density_gate_fails_closed(persisted_bundle, tmp_path):
    """Corrupting the stored density must fail the CLI (no silent report).

    The v2 saver refuses to write a non-spectral density, so the corrupt
    bundle is produced by patching the ``rho`` variables of the persisted
    NetCDF directly."""
    from scipy.io import netcdf_file

    corrupt_path = tmp_path / "corrupt.spiral.nc"
    with netcdf_file(str(persisted_bundle), "r", mmap=False) as src:
        with netcdf_file(str(corrupt_path), "w") as dst:
            for name, dim in src.dimensions.items():
                dst.createDimension(name, dim)
            for name in src._attributes.keys():
                setattr(dst, name, getattr(src, name))
            for name, var in src.variables.items():
                data = np.array(var[:])
                if name in ("rho_real", "rho_imag"):
                    data = data + 0.1
                out = dst.createVariable(name, var.typecode(), var.dimensions)
                out[:] = data
    out = tmp_path / "bad.json"
    proc = _run_cli(corrupt_path, out)
    assert proc.returncode != 0
    assert not out.exists()
    assert "density" in (proc.stderr + proc.stdout).lower()


def test_v1_bundle_rejected(persisted_bundle, tmp_path):
    """A legacy v1 sidecar must be rejected closed by the CLI."""

    from tbupy.spiral_state import SpiralState, load_spiral_state, save_spiral_state

    state = load_spiral_state(persisted_bundle)
    v1 = SpiralState(
        HR_up=state.HR_up,
        HR_dn=state.HR_dn,
        SR=state.SR,
        Rlist=state.Rlist,
        q_frac=state.q_frac,
        taus=state.taus,
        phis=state.phis,
        B_local=state.B_local,
        rho=state.rho,
        V_U=state.V_U,
        efermi=state.efermi,
        metadata_json=state.metadata_json,
    )
    v1_path = tmp_path / "legacy.spiral.nc"
    save_spiral_state(v1_path, v1)
    out = tmp_path / "v1.json"
    proc = _run_cli(v1_path, out)
    assert proc.returncode != 0
    assert not out.exists()
    assert "schema_version" in (proc.stderr + proc.stdout)


def test_orthogonal_v2_bundle_retains_provider_overlap_class(tmp_path, monkeypatch):
    """A signed-R list starts at -R, not necessarily the onsite R=0 class."""
    monkeypatch.setattr(sys.modules[__name__], "OVERLAP_S", 0.0)
    path = _persist_bundle(tmp_path / "orthogonal.spiral.nc")
    from TB2J.spiral_response import load_response_bundle

    bundle = load_response_bundle(path)
    assert bundle.provider.metadata.is_orthogonal is True
    from TB2J.spiral_response import compute_frozen_response, response_report

    response = compute_frozen_response(path, density_tol=1e-7, field_tol=1e-7)
    meta = response_report(response)["metadata"]
    assert meta["model"]["nonorthogonal_overlap"] is False
    assert meta["gates"]["density_tol"] == 1e-7
    assert meta["gates"]["field_tol"] == 1e-7


def test_real_matched_q_certificate_survives_sidecar_and_cli(
    persisted_bundle, tmp_path
):
    """The physical slope comes from real matched TBUpy SCF legs, not JSON literals."""
    from tbupy.spiral_pitch import run_matched_q_scf_pitch_slope

    kpts, kweights = _kmesh()
    matched = run_matched_q_scf_pitch_slope(
        _base_provider,
        q_center=Q_FRAC,
        kpts=kpts,
        kweights=kweights,
        nel=NEL,
        width=WIDTH,
        steps=(0.02, 0.01),
        scf_tol=1e-7,
        slope_step_tol=0.02,
        branch_moment_tol=0.05,
        tolerance_stability_tol=0.001,
    )
    assert matched.supported, matched.unsupported_reason
    assert len(matched.legs) == 24
    assert all(leg.energy_certified for leg in matched.legs)
    assert matched.step_residual < 0.02
    record = tmp_path / "certified-matched-q.json"
    record.write_text(json.dumps(matched.as_dict()))
    report_path = tmp_path / "dual-response.json"
    proc = _run_cli(persisted_bundle, report_path, ("--matched-q-record", str(record)))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text())
    np.testing.assert_allclose(
        report["self_consistent_pitch_slope"]["slope"],
        matched.self_consistent_pitch_slope,
        atol=1e-12,
    )
    assert "frozen_band_gradient" in report
    assert "frozen_band_pitch_slope" in report
    assert (
        "certified free-energy slope" in report["self_consistent_pitch_slope"]["source"]
    )
    assert report["frozen_band_pitch_slope"]["units"].startswith("eV")

    tampered = matched.as_dict()
    tampered["legs"][0]["free_energy"] += 0.1
    bad_record = tmp_path / "tampered-leg.json"
    bad_record.write_text(json.dumps(tampered))
    bad_report = tmp_path / "tampered-response.json"
    bad = _run_cli(
        persisted_bundle, bad_report, ("--matched-q-record", str(bad_record))
    )
    assert bad.returncode != 0
    assert not bad_report.exists()

    wrong_mesh = matched.as_dict()
    wrong_mesh["protocol"]["kpts"][0][0] += 0.01
    wrong_path = tmp_path / "wrong-mesh.json"
    wrong_path.write_text(json.dumps(wrong_mesh))
    from TB2J.spiral_response import (
        SpiralResponseMatchedQError,
        compute_frozen_response,
    )

    with pytest.raises(SpiralResponseMatchedQError, match="leg|certif"):
        compute_frozen_response(persisted_bundle, matched_q_record=wrong_path)

    wrong_functional = matched.as_dict()
    wrong_functional["protocol"]["hubbard_dict"] = {"Fe": {"U": 4.0, "J": 0.0, "L": 2}}
    with pytest.raises(SpiralResponseMatchedQError, match="Hubbard|functional"):
        compute_frozen_response(persisted_bundle, matched_q_record=wrong_functional)


def test_uncertified_matched_q_record_cannot_claim_physical_slope(persisted_bundle):
    """A user-supplied JSON label is not proof of matched converged SCF legs."""
    from TB2J.spiral_response import (
        SpiralResponseMatchedQError,
        compute_frozen_response,
    )

    unproven = {
        "supported": True,
        "q_center": Q_FRAC.tolist(),
        "self_consistent_pitch_slope": [123.0, 0.0, 0.0],
        "protocol": {
            "nel": NEL,
            "width": WIDTH,
            "field_policy": "co_rotating",
            "hubbard_dict": {},
            "hubbard_type": "dudarev",
            "dc_type": "FLL-ns",
        },
    }
    with pytest.raises(SpiralResponseMatchedQError, match="leg|certif"):
        compute_frozen_response(persisted_bundle, matched_q_record=unproven)


def test_response_reports_measured_fd_and_conditional_ward_residuals(persisted_bundle):
    """The public report must expose checks, not merely compute them privately."""
    from TB2J.spiral_response import compute_frozen_response, response_report

    report = response_report(
        compute_frozen_response(persisted_bundle, translation_cutoff=NCELL)
    )
    fd = report["metadata"]["finite_difference"]
    assert fd["local_gradient_max_abs_eV_per_rad"] < 1e-8
    assert fd["pitch_max_abs_eV_per_fractional_q"] < 1e-8
    assert report["frozen_band_pitch_slope"]["metadata"]["fd_residual"] is not None
    curvature = report["nonstationary_curvature"]
    assert curvature["ward"]["applicable"] is True
    assert np.isfinite(curvature["ward"]["max_residual"])
    assert np.isfinite(curvature["pair_model"]["max_abs_mismatch"])
    assert "pairwise_isotropic_assumption" in curvature["pair_model"]


def test_fixed_lab_constraint_preserves_gradient_but_skips_ward(
    persisted_bundle, tmp_path
):
    """A held laboratory constraint breaks global Ward symmetry, not the frozen reference."""
    from scipy.linalg import eigh
    from tbupy.occupations import SpinorOccupationAdapter
    from tbupy.spiral_state import load_spiral_state, save_spiral_state

    from TB2J.spiral_response import (
        compute_frozen_response,
        load_response_bundle,
        response_report,
    )

    source = load_response_bundle(persisted_bundle)
    state = load_spiral_state(persisted_bundle)
    constraint_potential = np.diag([0.03, -0.03, 0.0, 0.0]).astype(complex)
    pencil = source.provider.with_periodic_operators(
        V_U=state.V_U, V_con=constraint_potential
    )
    eigenpairs = [eigh(*pencil.gen_ham(k)) for k in source.kpts]
    state.evals = np.array([eps for eps, _ in eigenpairs])
    state.evecs = np.array([vec for _, vec in eigenpairs])
    state.occupations, state.efermi = SpinorOccupationAdapter(NEL, WIDTH).occupy(
        state.evals, source.kweights
    )
    state.rho = np.einsum(
        "k,kn,kbn,kcn->bc",
        source.kweights,
        state.occupations,
        state.evecs,
        state.evecs.conj(),
        optimize=True,
    )
    state.constraint_potential = constraint_potential
    meta = json.loads(state.metadata_json)
    meta["constraint"] = {
        "kind": "spinor_moment_component",
        "sites": [
            {
                "operator": "sigma_z(atom0)",
                "type": "spinor_moment_component",
                "target": 0.0,
                "multiplier": 0.03,
                "rotation": "lab_fixed",
                "hold_fixed": True,
            }
        ],
    }
    meta["field_role"] = "intrinsic_exchange"
    state.metadata_json = json.dumps(meta)
    path = tmp_path / "lab-field.spiral.nc"
    save_spiral_state(path, state)
    report = response_report(compute_frozen_response(path))
    assert np.linalg.norm(report["frozen_band_gradient"]["g_beta"]) > 1e-6
    assert report["nonstationary_curvature"]["ward"]["applicable"] is False
    assert "laboratory" in report["nonstationary_curvature"]["ward"]["reason"]
    assert report["nonstationary_curvature"]["pair_model"]["applicable"] is False
