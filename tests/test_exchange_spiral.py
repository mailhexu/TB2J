"""Story 005: ExchangeSpiral tensor mapping, SpinIO pipeline, diagnostics, CLI.

Covers the acceptance criteria on toy-ring bundles: known-J recovery within
the story-004 probe class (contour vs eigenbasis kernels, diagonal
consistency, ring inversion symmetry, ``C^db = 0``), the verbatim
``A -> J`` conversion through the existing ExchangeNCL hierarchy, AFM
normalization on a phi = pi two-sublattice fixture, the standard output
tree round-trip through ``Magnon.load_from_io``, the diagnostic report,
and the end-to-end CLI on a ``*.spiral.nc`` bundle written/read by tbupy.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest
from spiral_fixtures import make_ring_state, ring_hopping_classes

from TB2J.exchange_spiral import (
    ExchangeSpiral,
    SpiralParameters,
    a_tensors_to_jtensors,
    derive_ncell,
)
from TB2J.spiral_kernels import (
    SpiralGateError,
    eigenbasis_kernels_dense,
    lab_supercell,
    spiral_angles,
)

Q = 1.0 / 6.0
WIDTH = 5e-3


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _sharp(state, ncell, width=WIDTH):
    """Centre ``efermi`` in the largest gap, sharp frozen filling."""
    H, S = lab_supercell(state, ncell)
    if S is None:
        ev = np.linalg.eigvalsh(H)
    else:
        L = np.linalg.cholesky(S)
        Li = np.linalg.inv(L)
        A = Li @ H @ Li.conj().T
        ev = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
    ev = np.sort(ev)
    gaps = ev[1:] - ev[:-1]
    cand = [i for i, g in enumerate(gaps) if g > 0.2]
    i0 = max(cand or range(len(gaps)), key=lambda i: gaps[i])
    mu = float(0.5 * (ev[i0] + ev[i0 + 1]))
    meta = json.loads(state.metadata_json)
    meta["width"] = width
    return dataclasses.replace(state, efermi=mu, metadata_json=json.dumps(meta))


def _ring(N=6, seed=71, q=Q):
    classes, eps = ring_hopping_classes(N, seed)
    return _sharp(make_ring_state(N, classes, eps, q=q, b_local=(1.5,), width=0.2), N)


def _ladder_phis_pi(N=6, t1=0.4, t2=0.3):
    """Two-leg ladder with sublattice phases (0, pi): a phi = pi fixture.

    Same-copy nearest-neighbour hopping ``t1`` plus a rung hopping ``t2``,
    so cross-sublattice pairs (antiparallel reference moments) carry
    nonzero exchange.
    """
    classes = [
        (0, 0, 1, t1),
        (0, 0, -1, t1),
        (1, 1, 1, t1),
        (1, 1, -1, t1),
        (0, 1, 0, t2),
        (1, 0, 0, t2),
    ]
    return _sharp(
        make_ring_state(
            N,
            classes,
            [0.1, 0.1],
            q=Q,
            b_local=(1.5, 1.5),
            phis=[0.0, np.pi],
            width=0.2,
        ),
        N,
    )


def _pair_sign(calc, key):
    ai, mi, aj, mj = calc._pair_sites[key]
    return float(np.sign(np.dot(calc.moms[ai, mi], calc.moms[aj, mj])) or 1.0)


# ---------------------------------------------------------------------------
# parameters and helpers
# ---------------------------------------------------------------------------
def test_derive_ncell():
    state = _ring()
    assert derive_ncell(state) == 6
    calc = ExchangeSpiral.from_spiral_state(state)
    assert calc.ncell == 6
    with pytest.raises(ValueError, match="commensurate"):
        derive_ncell(dataclasses.replace(state, q_frac=np.array([0.13, 0.0, 0.0])))
    with pytest.raises(ValueError, match="ncell"):
        derive_ncell(dataclasses.replace(state, q_frac=np.zeros(3)))


def test_spiral_parameters_validation():
    with pytest.raises(ValueError, match="kernel"):
        SpiralParameters(kernel="stiffness")
    with pytest.raises(ValueError, match="frozen-bundle"):
        SpiralParameters(reference_protocol="per-q-scf")
    with pytest.raises(NotImplementedError, match="axis"):
        SpiralParameters(spiral_axis=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="kmesh"):
        SpiralParameters(kmesh=[4, 4, 4])
    with pytest.raises(ValueError, match="cell"):
        SpiralParameters(cell=np.zeros((3, 3)).tolist())
    assert SpiralParameters(kmesh=[8, 1, 1]).kmesh == [8, 1, 1]


def test_from_spiral_state_mirrors_tbupy_object():
    """A tbupy SpiralState bundle object converts field-for-field."""
    tbupy_state = pytest.importorskip("tbupy.spiral_state").SpiralState
    from TB2J.spiral_green import SpiralState as TB2JState

    state = _ring()
    bundle = tbupy_state(
        **{f.name: getattr(state, f.name) for f in dataclasses.fields(TB2JState)}
    )
    calc = ExchangeSpiral.from_spiral_state(bundle)
    assert isinstance(calc.state, TB2JState)
    assert np.allclose(calc.state.HR_up, state.HR_up)
    assert calc.state.norb == state.norb


# ---------------------------------------------------------------------------
# verbatim A -> J conversion through the ExchangeNCL hierarchy
# ---------------------------------------------------------------------------
def test_a_tensors_to_jtensors_matches_formulas():
    """The adapter wires A_ijR/spinat/ispin so the existing formulas apply."""
    rng = np.random.default_rng(5)
    norb = 3
    Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    A_ijR = {}
    for R in Rs:
        for i in range(norb):
            for j in range(norb):
                if R == (0, 0, 0) and i == j:
                    continue
                val = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
                A_ijR[(R, i, j)] = val
    spinat = rng.normal(size=(norb, 3))

    out = a_tensors_to_jtensors(A_ijR, spinat)

    for key, val in A_ijR.items():
        R, i, j = key
        valm = A_ijR[((-R[0], -R[1], -R[2]), j, i)]
        is_nonself = not (R == (0, 0, 0) and i == j)
        if not is_nonself:
            continue
        jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])
        assert out.exchange_Jdict[key] == jiso
        dmi = [np.real(val[0, u + 1] - val[u + 1, 0]) for u in range(3)]
        assert np.allclose(out.dmi_ddict[key], dmi)
        jani = np.imag(val[1:, 1:] + valm[1:, 1:])
        assert np.allclose(out.Jani_dict[key], jani)
        jprime = np.imag(val[0, 0] - val[3, 3]) - 2 * np.sign(
            np.dot(spinat[i], spinat[j])
        ) * np.imag(val[3, 3])
        assert out.biquadratic_Jdict[key] == (jprime, np.imag(val[3, 3]))


def test_mapping_recovers_known_j_from_curvature():
    """A^{00} = -i C^dd makes J_iso = -C^dd verbatim; other slots vanish."""
    state = _ring()
    curv = eigenbasis_kernels_dense(state, 6)
    calc = ExchangeSpiral.from_spiral_state(state, kernel="eigenbasis")
    report = calc.run()
    raw = calc._raw_jtensors
    Cdd = curv[("d", "d")]
    norb = state.norb
    for key, val in calc.exchange_Jdict.items():
        sgn = _pair_sign(calc, key)
        ai, mi, aj, mj = calc._pair_sites[key]
        expected_raw = -Cdd[ai * norb + mi, aj * norb + mj]
        assert raw.exchange_Jdict[key] == pytest.approx(expected_raw)
        assert val == pytest.approx(expected_raw / sgn)
    assert report["gates"]["diag_consistency"]["passed"]
    for key in raw.dmi_ddict:
        assert np.allclose(raw.dmi_ddict[key], 0.0)
        assert np.allclose(raw.Jani_dict[key], 0.0)
        jprime, b = raw.biquadratic_Jdict[key]
        assert b == 0.0
        assert jprime == raw.exchange_Jdict[key]


# ---------------------------------------------------------------------------
# known-J ring recovery (probe class) on the real pipeline
# ---------------------------------------------------------------------------
def test_known_j_ring_recovery():
    state = _ring()
    calc = ExchangeSpiral.from_spiral_state(state)  # contour, ncell derived
    report = calc.run()

    # gates green: goldstone, torque, diagonal consistency <= 1e-6
    assert all(rep["passed"] for rep in report["gates"].values())
    assert report["gates"]["diag_consistency"]["max_residual"] <= 1e-6
    # C^db = 0 (exact class)
    assert report["cdb_max"] <= 1e-9

    # probe class: contour J == eigenbasis reference J (<= 1e-5 x scale)
    ref = ExchangeSpiral.from_spiral_state(state, kernel="eigenbasis")
    ref.run()
    scale = max(
        max(abs(v) for v in calc.exchange_Jdict.values()),
        max(abs(v) for v in ref.exchange_Jdict.values()),
        1e-12,
    )
    assert set(calc.exchange_Jdict) == set(ref.exchange_Jdict)
    for key, val in calc.exchange_Jdict.items():
        assert abs(val - ref.exchange_Jdict[key]) <= 1e-5 * scale

    # full-matrix consistency: stored J == -C^dd / sgn on translation-
    # equivalent cell pairs (ring translation invariance of the curvature)
    eig = eigenbasis_kernels_dense(state, 6)
    Cdd = eig[("d", "d")]
    norb = state.norb
    for key, val in calc.exchange_Jdict.items():
        (d, _, _), mi, mj = key
        if d >= 0:
            ai, aj = 0, d
        else:
            ai, aj = -d, 0
        sgn = float(np.sign(np.dot(calc.moms[ai, mi], calc.moms[aj, mj])) or 1.0)
        expected = -Cdd[ai * norb + mi, aj * norb + mj] / sgn
        assert val == pytest.approx(expected, abs=1e-5 * max(1.0, abs(expected)))

    # ring inversion symmetry: J(R, i, j) == J(-R, j, i) and J(R, i, i) == J(-R, i, i)
    for (R, i, j), val in calc.exchange_Jdict.items():
        Rm = (-R[0], -R[1], -R[2])
        assert val == pytest.approx(calc.exchange_Jdict[(Rm, j, i)], abs=1e-12)


def test_ring_translation_consistency_of_pairs():
    """Every cell pair with the same (R, i, j) maps to the same J key value."""
    state = _ring(seed=72)
    calc = ExchangeSpiral.from_spiral_state(state)
    calc.run()
    norb = state.norb
    # rebuild the per-cell values from the curvature and compare with the dict
    Cdd = calc.curvature[("d", "d")]
    for key, val in calc.exchange_Jdict.items():
        R, mi, mj = key
        d = R[0]
        for ai in range(6):
            aj = ai + d
            if not 0 <= aj < 6:
                continue
            si, sj = ai * norb + mi, aj * norb + mj
            sgn = float(np.sign(np.dot(calc.moms[ai, mi], calc.moms[aj, mj])) or 1.0)
            assert val == pytest.approx(-Cdd[si, sj] / sgn, abs=1e-9)


# ---------------------------------------------------------------------------
# AFM normalization on the phi = pi fixture
# ---------------------------------------------------------------------------
def test_afm_normalization_phi_pi():
    state = _ladder_phis_pi()
    calc = ExchangeSpiral.from_spiral_state(
        state, kernel="eigenbasis", gate_override=True
    )
    report = calc.run()
    raw = calc._raw_jtensors.exchange_Jdict

    # gates other than torque are green; torque is overridden (reported)
    assert report["gates"]["goldstone"]["passed"]
    assert report["gates"]["diag_consistency"]["passed"]
    assert report["gates"]["torque"]["passed"] is False

    # every stored J equals the raw one divided by sgn(S_i . S_j)
    for key, val in calc.exchange_Jdict.items():
        sgn = _pair_sign(calc, key)
        assert val == pytest.approx(raw[key] / sgn, abs=1e-14)

    # rung pair (R = 0, sublattices 0 and 1): reference moments antiparallel
    key = ((0, 0, 0), 0, 1)
    sgn = _pair_sign(calc, key)
    assert sgn == -1.0
    assert np.sign(raw[key]) == -np.sign(calc.exchange_Jdict[key])

    # moment directions follow the spiral angles (up to one global orientation)
    thetas = spiral_angles(state, 6).ravel()
    mvec = calc.moms.reshape(-1, 3)
    ang_m = np.arctan2(mvec[:, 0], mvec[:, 2])
    align = np.cos(ang_m - thetas)
    assert np.all(np.abs(align) > 0.99)  # collinear with the local field
    assert np.all(np.sign(align) == np.sign(align[0]))  # single global orientation


# ---------------------------------------------------------------------------
# SpinIO tree, Magnon round-trip
# ---------------------------------------------------------------------------
def test_spinio_tree_roundtrip_magnon(tmp_path):
    from TB2J.io_exchange import SpinIO
    from TB2J.magnon.magnon3 import Magnon

    state = _ring(seed=73)
    calc = ExchangeSpiral.from_spiral_state(state)
    calc.run()
    out = str(tmp_path / "TB2J_results")
    calc.write_output(path=out)

    for fname in (
        "TB2J.pickle",
        "exchange.out",
        "structure.vasp",
        "spiral_diagnostics.json",
    ):
        assert os.path.exists(os.path.join(out, fname))

    exc = SpinIO.load_pickle(path=out)
    assert exc.nspin == 1
    assert exc.index_spin == [0]
    # a nonzero pair read back through the standard accessors
    R, i, j = max(exc.exchange_Jdict, key=lambda k: abs(exc.exchange_Jdict[k]))
    assert exc.get_Jiso(i, j, R) == pytest.approx(exc.exchange_Jdict[(R, i, j)])
    assert np.allclose(exc.get_DMI(i, j, R), 0.0)
    assert np.allclose(exc.get_Jani(i, j, R), 0.0)

    magnon = Magnon.load_from_io(exc)
    JR = exc.get_full_Jtensor_for_Rlist(order="ij33")
    assert JR.shape == (len(exc.Rlist), exc.nspin, exc.nspin, 3, 3)
    # the isotropic J is the only content: full tensor == J * I
    iR = exc.Rlist.index(tuple(R))
    assert np.allclose(JR[iR, i, j], exc.exchange_Jdict[(R, i, j)] * np.eye(3))

    # second load from the in-memory SpinIO object (no pickle round-trip)
    magnon2 = Magnon.load_from_io(calc.get_spinio())
    assert magnon2.nspin == magnon.nspin


# ---------------------------------------------------------------------------
# diagnostic report
# ---------------------------------------------------------------------------
def test_diagnostic_report_contents():
    state = _ring()
    calc = ExchangeSpiral.from_spiral_state(state)
    report = calc.run()
    # torque norms of the bundle are carried
    assert report["torque_norms"] == [0.0]
    # zero-mode residuals: goldstone and torque cos/sin
    zm = report["zero_modes"]
    assert len(zm["goldstone_residuals"]) == 6
    assert len(zm["torque_residuals_cos"]) == 6
    assert len(zm["torque_residuals_sin"]) == 6
    assert max(abs(v) for v in zm["goldstone_residuals"]) <= 1e-7
    # non-Heisenberg diagnostic present with the scale
    nh = report["nonheisenberg_Cbb"]
    assert nh["scale"] > 0
    assert nh["max_abs_offdiag"] >= 0.0
    # report is json-serializable (round-trip through the writer)
    assert json.loads(json.dumps(report))["gates"].keys() == report["gates"].keys()


def test_diagnostic_q0_nonheisenberg_zero_and_anchor():
    state = _ring(q=0.0)
    calc = ExchangeSpiral.from_spiral_state(state, ncell=6)
    report = calc.run()
    assert report["gates"]["q0_anchor"]["passed"]
    # strictly bilinear (q=0): the in-plane block collapses onto the mapping
    nh = report["nonheisenberg_Cbb"]
    assert nh["max_abs_offdiag"] <= 1e-12
    assert nh["max_abs_full"] <= 1e-8


def test_gate_hard_fail_and_override():
    """A grid-minimizing (non-torque-free) pitch violates the torque gate."""
    ncell = 6
    classes, eps = ring_hopping_classes(ncell, 71)
    base = make_ring_state(ncell, classes, eps, q=Q, b_local=(1.5,), width=0.2)

    def e_of_q(qv):
        st = dataclasses.replace(base, q_frac=np.array([qv, 0.0, 0.0]))
        H, _ = lab_supercell(st, ncell)
        ev = np.linalg.eigvalsh(H)
        mu = float(np.quantile(ev, 0.5))
        f = 1.0 / (1.0 + np.exp((ev - mu) / 0.05))
        return float(np.sum(ev * f))

    grid = np.linspace(0.0, 1.0, 241, endpoint=False)
    q_star = float(grid[int(np.argmin([e_of_q(qv) for qv in grid]))])
    state = _sharp(
        make_ring_state(ncell, classes, eps, q=q_star, b_local=(1.5,), width=0.2),
        ncell,
    )
    # incommensurate pitch: ncell must be explicit
    with pytest.raises(ValueError, match="commensurate"):
        ExchangeSpiral.from_spiral_state(state).run()
    # a non-torque-free reference violates the torque zero modes: hard fail
    calc = ExchangeSpiral.from_spiral_state(state, ncell=ncell)
    with pytest.raises(SpiralGateError, match="torque"):
        calc.run()
    # override: reported instead of raised; the bundle torque_norms flag it
    torqued = dataclasses.replace(state, torque_norms=np.full(state.norb, 0.1))
    calc2 = ExchangeSpiral.from_spiral_state(torqued, ncell=ncell, gate_override=True)
    report = calc2.run()
    assert report["gates"]["torque"]["passed"] is False
    assert report["gates"]["torque"]["flagged"] is True
    # without recorded bundle torques the violation is passed-but-not-flagged
    calc3 = ExchangeSpiral.from_spiral_state(state, ncell=ncell, gate_override=True)
    report3 = calc3.run()
    assert report3["gates"]["torque"]["passed"] is False
    assert report3["gates"]["torque"]["flagged"] is False


# ---------------------------------------------------------------------------
# E(q) diagnostic table
# ---------------------------------------------------------------------------
def test_eq_table():
    classes, eps = ring_hopping_classes(6, 74)
    state = make_ring_state(6, classes, eps, q=Q, b_local=(1.5,), width=0.2)
    calc = ExchangeSpiral.from_spiral_state(state, ncell=6, eq_table=True)
    report = calc.run()
    table = report["eq_table"]
    qs = [row["q_frac"] for row in table["points"]]
    es = [row["E_frozen_eV"] for row in table["points"]]
    assert qs == [[0.0, 0.0, 0.0], [1 / 6, 0.0, 0.0], [-1 / 6, 0.0, 0.0]]
    assert all(np.isfinite(es))
    # inversion symmetry of the frozen band energy
    assert es[1] == pytest.approx(es[2], abs=1e-9)
    # explicit q_set is honoured
    calc2 = ExchangeSpiral.from_spiral_state(
        state, ncell=6, eq_table=True, q_set=[[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0]]
    )
    table2 = calc2.run()["eq_table"]
    assert len(table2["points"]) == 2
    assert table2["points"][1]["q_frac"] == [1 / 3, 0.0, 0.0]
    with pytest.raises(ValueError, match="E\\(q\\) table point"):
        ExchangeSpiral.from_spiral_state(
            state, ncell=6, eq_table=True, q_set=[[0.13, 0.0, 0.0]]
        ).run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_main_end_to_end(tmp_path):
    tbupy_ss = pytest.importorskip("tbupy.spiral_state")
    from TB2J.scripts.tb2j_spiral import main
    from TB2J.spiral_green import SpiralState as TB2JState

    state = _ring(seed=75)
    bundle_path = str(tmp_path / "toy.spiral.nc")
    tbupy_ss.save_spiral_state(
        bundle_path,
        tbupy_ss.SpiralState(
            **{f.name: getattr(state, f.name) for f in dataclasses.fields(TB2JState)}
        ),
    )

    out = str(tmp_path / "TB2J_results")
    argv = [
        "--spiral-state",
        bundle_path,
        "--output",
        out,
        "--eq-table",
        "--symbols",
        "Cr",
    ]
    calc = main(argv)
    assert calc.report["gates"]["diag_consistency"]["passed"]

    from TB2J.io_exchange import SpinIO

    exc = SpinIO.load_pickle(path=out)
    assert exc.nspin == 1
    assert exc.atoms.get_chemical_symbols() == ["Cr"]
    with open(os.path.join(out, "spiral_diagnostics.json")) as myfile:
        saved = json.load(myfile)
    assert "nonheisenberg_Cbb" in saved
    assert saved["eq_table"]["points"]
