"""Story-015 tests: +U Hubbard rotation contract.

Contract (derived first in ``docs/sympy/siesta_spiral_hubbard_rotation.py``
+ paired report): a finite local spin rotation ``U_s`` at one correlated
site co-rotates that site's spinor occupation matrix; the FLL Hubbard
part ``W`` is reassembled from the co-rotated occupations and equals
``U_s W U_s^dag`` to 1e-12 RELATIVE (approved tolerance); the double
counting term is FROZEN bit-exactly at the converged reference value;
the spin-trace (charge-channel) Hubbard part and every other site are
untouched.  Gates follow the story-006 rotation-legitimacy discipline
(operator changes, scalar untouched, others frozen, trace preservation,
zero angle is a no-op).

Fixtures mirror ``test_siesta_spiral_hubbard.py``: the producer-derived
``fe4_dftu_story014.npz`` plus the committed ground-truth occupation
dump ``fe4_occ.dftu_occ``.  Tests skip cleanly when the npz is absent
(seeding ``TB2J_test_data`` is a user action).
"""

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral_hubbard import (
    DftuProvenance,
    DftuProvenanceError,
    assemble_hubbard_operator,
    rotate_hubbard_site_operator,
    rotate_site_occupations,
    site_spin_rotation,
)

DATA = Path(__file__).parent / "data"
NPZ = DATA / "fe4_dftu_story014.npz"
GT_OCC = DATA / "fe4_occ.dftu_occ"
TOL = 1e-12  # approved story-015 numeric tolerance (relative)

SOURCE_REFS = {
    "u_j_stdout": "fe4_so_dftu.stdout:DFTUprojs lines 278-279",
    "slater_integrals": "dftu_specs.f:1749-1758",
    "dc_potential": "m_pot_dftu.F90:974-1035",
    "hubbard_potential": "m_pot_dftu.F90:1057-1141",
    "occupation_loop": "m_occ_proj_dftu.F90:681-741",
    "projector_dump": "Fe_nc.dftu_proj header",
}


def parse_gt_occ(path):
    """fe4_occ.dftu_occ: header 'nua nproj nspin', rows
    'site spin m m' re im' (1-based site/spin)."""
    lines = path.read_text().split("\n")
    nua, _nproj, nspin = map(int, lines[0].split())
    gt = np.zeros((nua, nspin, 5, 5), dtype=complex)
    for line in lines[1:]:
        p = line.split()
        if len(p) == 6:
            s, sig, m, mp = (int(v) for v in p[:4])
            gt[s - 1, sig - 1, m - 1, mp - 1] = complex(float(p[4]), float(p[5]))
    return gt


def make_prov(data):
    u_ry, j_ry = float(data["u_ry"]), float(data["j_ry"])
    slater_f2 = 14.0 * j_ry / 1.625
    return DftuProvenance(
        u_ev=float(data["u_ev"]),
        j_ev=float(data["j_ev"]),
        u_ry=u_ry,
        j_ry=j_ry,
        projector_rc_bohr=float(data["proj_rc"]),
        projector_npts=int(data["proj_npts"]),
        slater_f0_ry=u_ry,
        slater_f2_ry=slater_f2,
        slater_f4_ry=0.625 * slater_f2,
        source_refs=dict(SOURCE_REFS),
        correlated_species="Fe",
        principal_n=3,
        site_indices=(0, 1, 2, 3),
        projector_identity="Fe_nc.dftu_proj:rc=2.975687,npts=500",
    )


@pytest.fixture(scope="module")
def story():
    if not NPZ.exists():
        pytest.skip("story-014 fixture npz absent (producer-derived fixture)")
    if not GT_OCC.exists():
        pytest.skip("ground-truth occupation dump absent")
    data = np.load(NPZ)
    return {"occ": parse_gt_occ(GT_OCC), "prov": make_prov(data)}


def _rodrigues(axis, angle):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kmat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0.0]])
    return (
        np.eye(3) * np.cos(angle)
        + np.sin(angle) * kmat
        + (1 - np.cos(angle)) * np.outer(k, k)
    )


def _spinor(channels):
    """(4, 5, 5) -> (2, 2, 5, 5) spinor blocks (uu, dd, ud, du order)."""
    n = np.zeros((2, 2, 5, 5), dtype=complex)
    n[0, 0] = channels[0]
    n[1, 1] = channels[1]
    n[0, 1] = channels[2]
    n[1, 0] = channels[3]
    return n


def _channels(spinor):
    occ = np.zeros((4, 5, 5), dtype=complex)
    occ[0] = spinor[0, 0]
    occ[1] = spinor[1, 1]
    occ[2] = spinor[0, 1]
    occ[3] = spinor[1, 0]
    return occ


# ---------------------------------------------------------------------------
# rotation geometry helpers
# ---------------------------------------------------------------------------


def test_site_spin_rotation_is_su2():
    rng = np.random.default_rng(15)
    for _ in range(4):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(0.1, 3.0))
        u = site_spin_rotation(axis, angle)
        assert u.shape == (2, 2)
        assert np.max(np.abs(u.conj().T @ u - np.eye(2))) < 1e-14
    assert np.array_equal(site_spin_rotation((0.0, 0.0, 1.0), 0.0), np.eye(2))


def test_site_spin_rotation_refuses_bad_axis():
    with pytest.raises(Exception, match="axis"):
        site_spin_rotation((0.0, 0.0, 0.0), 0.3)
    with pytest.raises(Exception, match="axis"):
        site_spin_rotation((1.0, 0.0, np.nan), 0.3)


# ---------------------------------------------------------------------------
# occupation co-rotation
# ---------------------------------------------------------------------------


def test_rotate_site_occupations_touches_only_declared_site(story):
    occ = story["occ"]
    rotated = rotate_site_occupations(occ, 1, axis=(0.3, -0.5, 0.81), angle=0.7)
    assert rotated.shape == occ.shape
    for site in (0, 2, 3):
        assert np.array_equal(rotated[site], occ[site])
    assert not np.array_equal(rotated[1], occ[1])
    # full spinor trace preserved at the rotated site
    tr0 = np.trace(occ[1, 0]) + np.trace(occ[1, 1])
    tr1 = np.trace(rotated[1, 0]) + np.trace(rotated[1, 1])
    assert abs(tr1 - tr0) <= TOL * max(abs(tr0), 1.0)


def test_rotate_site_occupations_matches_explicit_conjugation(story):
    occ = story["occ"]
    axis, angle = (0.2, 0.9, -0.4), 1.1
    rotated = rotate_site_occupations(occ, 2, axis=axis, angle=angle)
    u = site_spin_rotation(axis, angle)
    expected = _channels(np.einsum("ia,abpq,jb->ijpq", u, _spinor(occ[2]), u.conj()))
    assert np.max(np.abs(rotated[2] - expected)) == 0.0


def test_zero_angle_is_a_no_operation(story):
    occ = story["occ"]
    rotated = rotate_site_occupations(occ, 0, axis=(0.0, 1.0, 0.0), angle=0.0)
    assert np.array_equal(rotated, occ)


def test_rotate_site_occupations_refuses_bad_payload(story):
    with pytest.raises(Exception, match="site"):
        rotate_site_occupations(story["occ"], 9, angle=0.2)
    with pytest.raises(Exception, match=r"\(nsites, 4, 5, 5\)"):
        rotate_site_occupations(story["occ"][0], 0, angle=0.2)


# ---------------------------------------------------------------------------
# AC-2: the assembled +U vertex under a finite local rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis,angle",
    [
        ((0.0, 0.0, 1.0), 0.35),
        ((1.0, 0.0, 0.0), np.pi / 2),
        ((0.2, -0.9, 0.4), 2.2),
    ],
)
def test_hubbard_contribution_transforms_per_contract(story, axis, angle):
    occ, prov = story["occ"], story["prov"]
    result = rotate_hubbard_site_operator(occ, prov, 1, axis=axis, angle=angle)
    report = result["report"]
    assert report["passed"], report
    assert report["hubbard_covariance_relative"] <= TOL
    assert report["magnetic_change"] > 1e-8  # the Hubbard operator varies
    # the returned occupations match the helper-level co-rotation
    assert np.array_equal(
        result["occupations_rotated"],
        rotate_site_occupations(occ, 1, axis=axis, angle=angle),
    )


def test_dc_term_frozen_bit_exactly(story):
    occ, prov = story["occ"], story["prov"]
    v_hub_ref, v_dc_ref, _ = assemble_hubbard_operator(occ, prov)
    result = rotate_hubbard_site_operator(
        occ, prov, 2, axis=(0.5, 0.5, 0.7071067811865476), angle=1.3
    )
    assert np.array_equal(result["v_dc_frozen"], v_dc_ref)
    # the contract is non-vacuous: rebuilding dc from the co-rotated
    # occupations would move it by a finite amount
    v_hub_rot, v_dc_rebuilt, _ = assemble_hubbard_operator(
        result["occupations_rotated"], prov
    )
    drift = float(np.max(np.abs(v_dc_rebuilt[2] - v_dc_ref[2])))
    assert drift > 1e-6
    assert result["report"]["would_be_dc_drift_ry"] == pytest.approx(drift, rel=1e-12)
    # the Hubbard part equals the reassembly from the rotated occupations
    assert np.max(np.abs(result["v_hubbard_rotated"][2] - v_hub_rot[2])) == 0.0


def test_scalar_spin_trace_untouched(story):
    occ, prov = story["occ"], story["prov"]
    v_hub_ref, _, _ = assemble_hubbard_operator(occ, prov)
    result = rotate_hubbard_site_operator(occ, prov, 0, axis=(0.1, 0.3, 0.9), angle=0.8)
    v_ref = v_hub_ref[0]
    v_rot = result["v_hubbard_rotated"][0]
    charge_ref = 0.5 * (v_ref[0] + v_ref[1])
    charge_rot = 0.5 * (v_rot[0] + v_rot[1])
    assert np.max(np.abs(charge_rot - charge_ref)) <= TOL
    assert result["report"]["spin_trace_change_max"] <= TOL


def test_rotation_legitimacy_gates_story006_discipline(story):
    occ, prov = story["occ"], story["prov"]
    result = rotate_hubbard_site_operator(occ, prov, 3, axis=(0.0, 1.0, 0.0), angle=0.9)
    report = result["report"]
    assert report["rotated_site"] == 3
    assert report["untouched_sites_unchanged"] is True
    assert report["max_trace_deviation"] <= TOL
    assert report["dc_change_max"] == 0.0
    assert report["convention"]["adr"] == "ADR-007"
    assert report["convention"]["dc"] == "frozen_at_reference_traces"
    # zero angle: exact no-op, gate passes with zero magnetic change
    noop = rotate_hubbard_site_operator(occ, prov, 3, angle=0.0)
    assert np.array_equal(noop["occupations_rotated"], occ)
    assert noop["report"]["passed"]
    assert noop["report"]["magnetic_change"] == 0.0
    assert np.array_equal(
        noop["v_hubbard_rotated"], assemble_hubbard_operator(occ, prov)[0]
    )


def test_contract_requires_provenance(story):
    with pytest.raises(DftuProvenanceError):
        rotate_hubbard_site_operator(story["occ"], None, 1, angle=0.2)


def test_contract_records_ev_and_ry_units(story):
    result = rotate_hubbard_site_operator(story["occ"], story["prov"], 1, angle=0.4)
    assert result["report"]["convention"]["units"] == "Ry"


def test_mathdoc_marker_on_contract_function():
    """The implemented contract carries the MathDoc marker linking it to
    the registered derivation (mathdocs/siesta-spiral-hubbard-rotation)."""
    import TB2J.interfaces.siesta_spiral_hubbard as module

    source = Path(module.__file__).read_text()
    assert "# MathDoc: tb2j/siesta-spiral-hubbard-rotation" in source


def test_derivation_authority_runs_green():
    """The derivation-first authority (docs/sympy) passes end to end."""
    pytest.importorskip("sympy")
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    script = repo / "docs" / "sympy" / "siesta_spiral_hubbard_rotation.py"
    if not script.exists():  # pragma: no cover
        pytest.skip("derivation script not present")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL ASSERTIONS GREEN" in proc.stdout
