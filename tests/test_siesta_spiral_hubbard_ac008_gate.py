"""Story-016 tests: the AC-008 gate on the genuine +U reference.

TEST-001 runs the story-010 matched frozen-rotation machinery
(``validate_finite_rotations``) on a frozen problem whose rotating
operator is the certified +U Hubbard vertex of the fe4 producer state,
and binds the story-015 rotation contract to the SAME frozen problem at
the energy level (contract-rotated operator vs SU(2) conjugation, no
new SCF).  TEST-002 audits the XC+Hubbard provenance of the +U artifact
and refuses stripped/substituted outputs with distinct actionable
diagnostics (FR-014).  IMPL-001 assembles the verdict report through
``run_ac008_gate``.

Frozen problem (declared per ADR-005): the Gamma-point pencil of the
correlated Hubbard subspace -- per-site Hubbard operators W_s (eV) in
the pinned 4-component spin packing, frozen genuine projector-overlap
channel S, and no other rotating contribution.  The Hubbard operator is
strictly onsite (ADR-007 local self-energy), so the frozen channel that
makes the spectrum respond is the intersite overlap; this declaration
is recorded in the verdict.  Local frames follow the DM.InitSpin 90
deg/atom texture (moments +z, +x, -z, -x), so all four sites carry
inequivalent orientations.

Tolerances are FIXED AT STORY-016 APPROVAL (2026-10-16) and mirrored in
the story record budgets; see AC008_* constants in
TB2J/interfaces/siesta_spiral_hubbard.py.

Fixtures mirror test_siesta_spiral_hubbard_rotation.py: the producer
npz plus the committed ground-truth occupation dump; tests skip cleanly
when the npz is absent (seeding ``TB2J_test_data`` is a user action).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from TB2J.interfaces.siesta_spiral import BUNDLE_VERSION, SpiralBundle, SpiralImporter
from TB2J.interfaces.siesta_spiral_hubbard import (
    AC008_ANGLE_LADDER_RAD,
    AC008_CONTRACT_ENERGY_BUDGET_EV,
    AC008_CURVATURE_BUDGET_EV_PER_RAD2,
    AC008_PACKING_FIDELITY_BUDGET_EV,
    AC008_STABLE_WINDOW_RUNGS,
    AC008_TORQUE_BUDGET_EV_PER_RAD,
    RY_TO_EV,
    DftuProvenance,
    HubbardBundle,
    HubbardBundleError,
    ProvenanceAuditError,
    assemble_hubbard_operator,
    audit_xc_hubbard_provenance,
    build_xc_hubbard_audit_artifact,
    edftu_oracle,
    hubbard_site_block_components,
    hubbard_site_block_unpack,
    ingest_dftu_run,
    rotated_hubbard_site_components,
    run_ac008_gate,
    xc_source_payload_sha256,
)
from TB2J.interfaces.siesta_spiral_response import (
    CURVATURE_BUDGET_EV_PER_RAD2,
    ENSEMBLE_GAPPED_T0,
    RESPONSE_IDENTITY_TOL_EV,
    TORQUE_BUDGET_EV_PER_RAD,
    EnergyContour,
    Ensemble,
    FrozenResponseProblem,
    evaluate_response_set,
    occupied_band_energy,
)
from TB2J.interfaces.siesta_spiral_validation import validate_finite_rotations
from TB2J.interfaces.siesta_spiral_vertex import (
    OPERATOR_MODE_ONSITE_DELTA,
    extract_magnetic_vertex,
)

DATA = Path(__file__).parent / "data"
NPZ = DATA / "fe4_dftu_story014.npz"
GT_OCC = DATA / "fe4_occ.dftu_occ"
PRODUCER_OCC_DUMP = (
    DATA / "../../../.tmp/siesta-xc-mft/fe4_dftu_probe/occ_dump"
).resolve()

NORBD = 5  # correlated d shell per site
NSITES = 4
NORB = NSITES * NORBD
NOCC = 20  # half of the 40 correlated spinor states; open 0.788 eV gap
SITE_LABELS = tuple(f"Fe{i + 1}" for i in range(NSITES))
ORBITALS_BY_SITE = {
    label: [s * NORBD + p for p in range(NORBD)] for s, label in enumerate(SITE_LABELS)
}

# Angles and budgets FIXED AT STORY-016 APPROVAL (mirrored in the module
# constants and the story-016 budgets block; measured worst values are
# recorded in the story body).
ANGLE_LADDER = AC008_ANGLE_LADDER_RAD
STABLE_WINDOW_ANGLES = tuple(ANGLE_LADDER[i] for i in AC008_STABLE_WINDOW_RUNGS)


def parse_gt_occ(path):
    """fe4_occ.dftu_occ: header 'nua nproj nspin', rows
    'site spin m m' re im' (1-based site/spin)."""
    lines = path.read_text().split("\n")
    nua, _nproj, nspin = map(int, lines[0].split())
    gt = np.zeros((nua, nspin, NORBD, NORBD), dtype=complex)
    for line in lines[1:]:
        p = line.split()
        if len(p) == 6:
            s, sig, m, mp = (int(v) for v in p[:4])
            gt[s - 1, sig - 1, m - 1, mp - 1] = complex(float(p[4]), float(p[5]))
    return gt


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _transverse_frame(normal):
    """Right-handed {x, y} frame transverse to ``normal``."""
    normal = _unit(normal)
    seed = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    x = _unit(np.cross(seed, normal))
    return {"x": tuple(x), "y": tuple(np.cross(normal, x))}


@pytest.fixture(scope="module")
def producer_hubbard():
    """HubbardBundle from ingest_dftu_run on the genuine producer export
    (skip-gated): its ``stored_h`` is the producer HSX's stored total H
    over the certified correlated subspace -- the XC-bearing carrier the
    audit must digest (XC presence is a producer fact, never a
    downstream-assembled Hubbard-only pencil)."""
    if not (PRODUCER_OCC_DUMP / "fe4_occ.HSX").exists():
        pytest.skip(
            "producer +U run '.tmp/siesta-xc-mft/fe4_dftu_probe/occ_dump' "
            "is unavailable; the AC-008 gate's XC carrier must come from "
            "ingest_dftu_run on the genuine export (fixture seeding to "
            "TB2J_test_data is a user action, see story-014)"
        )
    dump = PRODUCER_OCC_DUMP
    _spiral, hb = ingest_dftu_run(
        dump / "fe4_occ.HSX",
        dump / "fe4_occ.DM",
        dump / "fe4_occ.out",
        dump / "Fe_nc.dftu_proj",
        occ_dump=dump,
        species_label="Fe",
        principal_n=3,
        site_indices=(1, 2, 3, 4),
    )
    assert hb.stored_h is not None
    assert hb.stored_h_provenance["kind"] == "producer_hsx_stored_total_H"
    return hb


@pytest.fixture(scope="module")
def dftu(producer_hubbard):
    """Genuine +U state: GT occupations, certified assembler, provenance."""
    if not NPZ.exists():
        pytest.skip("fe4_dftu_story014.npz not seeded (user action)")
    data = np.load(NPZ)
    occ = parse_gt_occ(GT_OCC)
    u_ry, j_ry = float(data["u_ry"]), float(data["j_ry"])
    refs = json.loads(str(data["refs_json"]))
    prov = DftuProvenance(
        u_ev=float(data["u_ev"]),
        j_ev=float(data["j_ev"]),
        u_ry=u_ry,
        j_ry=j_ry,
        projector_rc_bohr=float(data["proj_rc"]),
        projector_npts=int(data["proj_npts"]),
        slater_f0_ry=u_ry,
        slater_f2_ry=14.0 * j_ry / 1.625,
        slater_f4_ry=0.625 * 14.0 * j_ry / 1.625,
        source_refs=refs,
        correlated_species="Fe",
        principal_n=3,
        site_indices=tuple(range(NSITES)),
        projector_identity=(
            f"Fe_nc.dftu_proj:rc={float(data['proj_rc']):.6f},"
            f"npts={int(data['proj_npts'])}"
        ),
    )
    v_hubbard, v_dc, conventions = assemble_hubbard_operator(occ, prov)
    # Stored total H over the certified correlated subspace
    # (H_stored = H_xc + H_U): taken from ingest_dftu_run's output on
    # the genuine producer export -- the certified producer HSX's
    # stored total H is the XC-bearing carrier digested into the
    # certificate (FR-014).  The Hubbard-only Gamma pencil below is
    # response machinery only, never the XC carrier.
    h_stored = producer_hubbard.stored_h
    bundle = HubbardBundle(
        prov=prov,
        occupations=occ,
        v_hubbard=v_hubbard,
        v_dc=v_dc,
        edftu_ev=edftu_oracle(occ, v_hubbard, v_dc, prov=prov),
        edftu_reference_ev=float(data["edftu_ref_ev"]),
        conventions=conventions,
        stored_h=h_stored,
        stored_h_provenance=producer_hubbard.stored_h_provenance,
    )
    # Response machinery (ADR-005 Gamma pencil): the genuine folded
    # projector overlap for the frozen channel -- not the XC carrier.
    no_l = 4 * 9  # 9 orbitals per atom in the producer cell
    S36 = np.zeros((no_l, no_l))
    np.add.at(S36, (data["s_rows"], data["s_cols"] % no_l), data["s_vals"])
    didx = [a * 9 + 1 + p for a in range(NSITES) for p in range(NORBD)]
    S20_block = S36[np.ix_(didx, didx)]
    fold_asymmetry = float(np.max(np.abs(S20_block - S20_block.T)))
    S20 = 0.5 * (S20_block + S20_block.T)
    return {
        "data": data,
        "refs": refs,
        "occ": occ,
        "prov": prov,
        "v_hubbard": v_hubbard,
        "v_dc": v_dc,
        "bundle": bundle,
        "S20": S20,
        "S20_block": S20_block,
        "fold_asymmetry": fold_asymmetry,
        "stored_h": h_stored,
    }


@pytest.fixture(scope="module")
def moments(dftu):
    """GT moment directions per site (MathDoc moment_vector convention)."""
    occ = dftu["occ"]
    out = []
    for s in range(NSITES):
        t_ud = np.trace(occ[s, 2])
        t_du = np.trace(occ[s, 3])
        vector = np.array(
            [
                np.real(t_ud + t_du),
                np.real(1j * (t_ud - t_du)),
                np.real(np.trace(occ[s, 0]) - np.trace(occ[s, 1])),
            ]
        )
        out.append(_unit(vector))
    return tuple(out)


def gamma_hubbard_bundle(v_hubbard, S20):
    """Gamma-point frozen problem of the correlated subspace.

    Sparse H: per-site Hubbard operators (bundle file packing, eV);
    intersite H blocks are exactly zero (the +U operator is strictly
    onsite).  Overlap: the genuine Gamma-folded projector overlap.
    """
    sparse, overlap = {}, {}
    for mu in range(NORB):
        for nu in range(NORB):
            sval = float(S20[mu, nu])
            if mu // NORBD == nu // NORBD:
                site = mu // NORBD
                p, q = mu % NORBD, nu % NORBD
                w = v_hubbard[site] * RY_TO_EV
                comps = (
                    float(w[0, p, q].real),
                    float(w[1, p, q].real),
                    float(w[2, p, q].real),
                    float(-w[2, p, q].imag),
                )
            else:
                if abs(sval) < 1e-12:
                    continue
                comps = (0.0, 0.0, 0.0, 0.0)
            sparse[(mu, nu, 0)] = comps
            overlap[(mu, nu, 0)] = sval
    return SpiralBundle(
        version=BUNDLE_VERSION,
        norb=NORB,
        ncell=1,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse=sparse,
        overlap=overlap,
        metadata={
            "origin": "story-016 AC-008 gate",
            "frozen_problem": (
                "Gamma-point correlated-subspace pencil: Hubbard operators "
                "rotate (story-015 contract), genuine projector overlap and "
                "zero intersite Hubbard blocks stay frozen"
            ),
        },
    )


@pytest.fixture(scope="module")
def gate(dftu, moments):
    """Bundle, declared ensemble, analytic response (story-009 machinery)."""
    fold_asymmetry = dftu["fold_asymmetry"]
    S20 = dftu["S20"]

    bundle = gamma_hubbard_bundle(dftu["v_hubbard"], S20)
    bundle.validate()
    vertex = extract_magnetic_vertex(
        bundle, orbitals_by_site=ORBITALS_BY_SITE, operator=OPERATOR_MODE_ONSITE_DELTA
    )
    axes_by_site = {
        label: _transverse_frame(m) for label, m in zip(SITE_LABELS, moments)
    }
    problem = FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={s: tuple(f.items()) for s, f in axes_by_site.items()},
    )
    values = eigh(*problem.pencil(0.0), eigvals_only=True)
    gap = float(values[NOCC] - values[NOCC - 1])
    fermi = float(0.5 * (values[NOCC] + values[NOCC - 1]))
    ensemble = Ensemble(ENSEMBLE_GAPPED_T0, NOCC, float(NOCC), fermi, 0.0, gap)
    contour = EnergyContour.declared(
        z0=fermi,
        occupied_min=float(values[0]),
        gap_ev=gap,
        margin_ev=0.3,
        n_nodes=512,
    )
    coords = tuple((s, a) for s in problem.site_order for a in ("x", "y"))
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=[0.0],
        k_weights=[1.0],
        contour=contour,
        finite_steps=(),
    )
    return {
        "bundle": bundle,
        "problem": problem,
        "ensemble": ensemble,
        "response": response,
        "axes_by_site": axes_by_site,
        "reference_directions_by_site": {
            label: tuple(float(v) for v in m) for label, m in zip(SITE_LABELS, moments)
        },
        "gap_ev": gap,
        "fermi_ev": fermi,
        "fold_asymmetry": fold_asymmetry,
        "occupied_energy_ev": occupied_band_energy(*problem.pencil(0.0), NOCC),
    }


@pytest.fixture(scope="module")
def cross_check(gate):
    """Story-010 matched frozen-rotation summary on the +U frozen problem."""
    result = validate_finite_rotations(
        gate["problem"],
        gate["response"],
        axes_by_site=gate["axes_by_site"],
        reference_directions_by_site=gate["reference_directions_by_site"],
        ensemble=gate["ensemble"],
        kpoints=[0.0],
        k_weights=[1.0],
        angle_ladder=ANGLE_LADDER,
    )
    return result.finite_rotation_cross_check


@pytest.fixture(scope="module")
def contract_table(dftu, gate):
    """Story-015 contract vs SU(2) conjugation on the SAME frozen problem.

    For every coordinate and every ladder angle the rotated pencil is
    built twice through the certified importer: once from the
    contract-rotated operator (rotate_hubbard_site_operator: occupation
    co-rotation, W reassembly, dc frozen), once by the plain per-site
    conjugation the story-009/010 machinery differentiates.
    """
    problem = gate["problem"]
    occ, prov = dftu["occ"], dftu["prov"]
    sparse = gate["bundle"].sparse
    rows = []
    for site, label_site in enumerate(SITE_LABELS):
        for axis_label, axis in gate["axes_by_site"][label_site].items():
            for angle in ANGLE_LADDER:
                comps, report = rotated_hubbard_site_components(
                    occ, prov, site, axis=axis, angle=angle
                )
                site_keys = {
                    (mu, nu, 0): comps[(mu % NORBD, nu % NORBD)]
                    for mu, nu, _r in sparse
                    if mu // NORBD == site == nu // NORBD
                }
                contract_bundle = SpiralBundle(
                    version=BUNDLE_VERSION,
                    norb=NORB,
                    ncell=1,
                    q=(0.0,),
                    spin_packing="H4=-ImH12",
                    phase_convention="q_half_per_spin_sector",
                    hermitization="none_hamiltonian_path",
                    fermi_shift="none",
                    pitch_sign="qphs_up_exp_minus",
                    sparse={**sparse, **site_keys},
                    overlap=gate["bundle"].overlap,
                )
                h_contract, s = SpiralImporter().assemble_pencil(contract_bundle, 0.0)
                h_conj, s_conj = problem.rotated_pencil(
                    0.0, {label_site: {axis_label: angle}}
                )
                assert np.array_equal(s, s_conj)
                e_contract = occupied_band_energy(h_contract, s, NOCC)
                e_conj = occupied_band_energy(h_conj, s, NOCC)
                rows.append(
                    {
                        "site": label_site,
                        "axis": axis_label,
                        "angle_rad": float(angle),
                        "energy_conjugation_ev": e_conj,
                        "energy_contract_ev": e_contract,
                        "abs_delta_ev": abs(e_contract - e_conj),
                        "hubbard_covariance_relative": report[
                            "hubbard_covariance_relative"
                        ],
                        "dc_change_max": report["dc_change_max"],
                        "would_be_dc_drift_ry": report["would_be_dc_drift_ry"],
                    }
                )
    return rows


@pytest.fixture(scope="module")
def audit_artifact(dftu):
    """Audit envelope with the XC source pinned intrinsically: the
    digest is derived from the certificate's own stored-H payload (the
    stored total H, H_stored = H_xc + H_U), never from caller bytes."""
    return build_xc_hubbard_audit_artifact(
        dftu["bundle"],
        vertex_kind="xc+hubbard",
        producer_u_ry=float(dftu["data"]["u_ry"]),
        producer_j_ry=float(dftu["data"]["j_ry"]),
        producer_u_j_source=dftu["refs"]["u_j_stdout"],
        xc_source={"kind": "stored_total_H"},
    )


def texture_provenance(dftu, moments):
    """Producer texture state recorded verbatim (fixture-recorded)."""
    data = dftu["data"]
    return {
        "producer_run_id": (
            "fe4_so_dftu (stdout fe4_so_dftu.stdout; fixture "
            "tests/data/fe4_dftu_story014.npz)"
        ),
        "convergence": (
            "converged +U SO-texture reference; fixture-recorded Edftu "
            f"reference {float(data['edftu_ref_ev'])} eV"
        ),
        "edftu_reference_ev": float(data["edftu_ref_ev"]),
        "edftu_oracle_ev": float(dftu["bundle"].edftu_ev),
        "texture_description": (
            "DM.InitSpin 90 deg/atom texture on the 4-atom Fe chain "
            "(commensurate q=1/4 pitch, 3.644 bohr spacing); moments "
            "+z, +x, -z, -x by site order; the texture is set and held "
            "by DM.InitSpin on the pinned build (story-004), q-control "
            "of the texture is not validated"
        ),
        "declared_vertex_kind": "xc+hubbard",
        "u_ev": float(data["u_ev"]),
        "j_ev": float(data["j_ev"]),
        "pitch": "commensurate q=1/4 (pi/2 per site), InitSpin-defined",
        "moments_unit": [[float(v) for v in m] for m in moments],
    }


# ---------------------------------------------------------------------------
# TEST-001: matched frozen-rotation agreement on the +U reference
# ---------------------------------------------------------------------------


def test_matched_frozen_rotation_agreement(gate, cross_check):
    """Analytic local response vs finite rotations of the SAME frozen
    problem, over the approved angle ladder (story-010 machinery)."""
    assert cross_check["passed"] is True, cross_check["failure_reasons"]
    assert cross_check["stable_error_regime"] == {"gradient": True, "curvature": True}
    windows = cross_check["common_stable_windows"]
    assert len(windows) == 1, "unexpected extra stable windows"
    assert windows[0]["angles_rad"] == list(STABLE_WINDOW_ANGLES)

    # Tolerances FIXED AT STORY-016 APPROVAL: torque/curvature budgets
    # inherit the story-009/010 approved values; every ladder rung records
    # the mismatch, the roundoff floor and the minimum rotated gap.
    assert cross_check["budgets"]["torque_ev_per_rad"] == TORQUE_BUDGET_EV_PER_RAD
    assert (
        cross_check["budgets"]["curvature_ev_per_rad2"] == CURVATURE_BUDGET_EV_PER_RAD2
    )
    rungs = cross_check["rungs"]
    assert [rung["angle_rad"] for rung in rungs] == list(ANGLE_LADDER)
    for rung in rungs:
        assert rung["gradient_max_abs_mismatch"] <= TORQUE_BUDGET_EV_PER_RAD
        assert rung["minimum_rotated_gap_ev"] > 0.0
        assert np.isfinite(rung["curvature_roundoff_floor"])
    for index in AC008_STABLE_WINDOW_RUNGS:
        assert (
            rungs[index]["curvature_max_abs_mismatch"] <= CURVATURE_BUDGET_EV_PER_RAD2
        )
        assert rungs[index]["curvature_roundoff_floor"] <= CURVATURE_BUDGET_EV_PER_RAD2

    # Stable error regime consistent with the story-010 pattern: the
    # truncation-dominated errors decay as theta**2 across the halving
    # rungs (theta halves -> error ~ quarter) before the roundoff plateau.
    for field in ("gradient_max_abs_mismatch", "curvature_max_abs_mismatch"):
        for coarse, fine in zip(rungs, rungs[1:]):
            if np.isclose(2.0 * fine["angle_rad"], coarse["angle_rad"]):
                assert 2.5 * fine[field] <= coarse[field] <= 5.5 * fine[field], (
                    field,
                    coarse["angle_rad"],
                    coarse[field],
                    fine[field],
                )

    # Nonzero reference torque preserved and reported verbatim (FR-008);
    # rotation probes changed the operator (story-006 legitimacy).
    reference_torque = cross_check["reference_torque"]
    assert len(reference_torque) == 2 * len(SITE_LABELS)
    assert max(abs(v) for v in reference_torque.values()) > 0.0
    assert cross_check["torque_state"] == gate["response"].torque_state
    assert all(v > 1e-12 for v in cross_check["physical_rotation_change_norm"].values())


def test_contract_rotation_matches_conjugation_energy(dftu, gate, contract_table):
    """The story-015 contract reproduces the conjugated frozen problem at
    the ENERGY level over the whole ladder (no new SCF anywhere)."""
    assert len(contract_table) == len(SITE_LABELS) * 2 * len(ANGLE_LADDER)
    worst_delta = max(row["abs_delta_ev"] for row in contract_table)
    worst_covariance = max(row["hubbard_covariance_relative"] for row in contract_table)
    # FIXED AT STORY-016 APPROVAL: contract-vs-conjugation energy budget
    # 1e-11 eV (measured 1.7e-13), covariance at the story-015 tolerance.
    assert worst_delta <= AC008_CONTRACT_ENERGY_BUDGET_EV
    assert worst_covariance <= 1e-12
    assert all(row["dc_change_max"] == 0.0 for row in contract_table)
    assert all(row["would_be_dc_drift_ry"] > 0.0 for row in contract_table)

    # the rotations are physical: they move the frozen-problem energy
    energies = [row["energy_conjugation_ev"] for row in contract_table]
    e0 = gate["occupied_energy_ev"]
    assert max(abs(e - e0) for e in energies) > 1e-6


def test_declared_ensemble_and_identity_gate(gate):
    """ADR-005 declaration: gapped step ensemble on the Gamma frozen
    problem; the story-009 contour/eigenbasis identity gate holds."""
    ensemble = gate["ensemble"]
    ensemble.validate()
    assert ensemble.kind == ENSEMBLE_GAPPED_T0
    assert ensemble.n_occupied == NOCC
    assert gate["gap_ev"] > 0.1  # open 0.788 eV gap on the declared reference
    assert gate["response"].contour_vs_band_max_abs <= RESPONSE_IDENTITY_TOL_EV
    # packing fidelity: the Gamma-folded overlap is symmetric, and the
    # assembled pencil reproduces the certified operator through the
    # importer (the complex residue of W's spin-diagonal channels is
    # bounded by the same gate).
    assert gate["fold_asymmetry"] <= 1e-12


def test_packing_fidelity_reproduces_certified_operator(dftu):
    """Pinned packing-fidelity gate: rebuild the certified +U operator
    from the 4-component file packing (undoing the documented packing
    for this texture class: real spin-diagonal channels, du = conj-swap
    of ud) and assert reproduction within the pinned budget.  The
    reproduction error is COMPUTED here on the genuine reference
    (measured worst 2.2978e-13 eV, dominated by the discarded Im
    spin-diagonal residual), not asserted from the story record."""
    v_certified = np.asarray(dftu["v_hubbard"], dtype=complex) * RY_TO_EV
    worst = 0.0
    for site in range(NSITES):
        comps = hubbard_site_block_components(dftu["v_hubbard"][site])
        rebuilt = hubbard_site_block_unpack(comps, energy_scale=1.0)
        assert rebuilt.shape == v_certified[site].shape
        worst = max(worst, float(np.max(np.abs(rebuilt - v_certified[site]))))
    assert worst <= AC008_PACKING_FIDELITY_BUDGET_EV, (
        f"packing fidelity {worst:.3e} eV exceeds the pinned budget "
        f"{AC008_PACKING_FIDELITY_BUDGET_EV:.1e} eV"
    )


def test_packing_refuses_out_of_class_texture(dftu):
    """The packing restriction is enforced: a texture with complex
    spin-diagonal channels (rotated out of the xz plane) carries O(eV)
    imaginary diagonal parts that the 4-component packing would discard
    silently -- the packer must refuse."""
    v = np.asarray(dftu["v_hubbard"], dtype=complex).copy()
    v[0, 0] = v[0, 0] + 0.37j * np.max(np.abs(v[0, 0]))  # complex uu channel
    with pytest.raises(HubbardBundleError) as excinfo:
        hubbard_site_block_components(v[0])
    assert "restriction" in str(excinfo.value)
    assert "xz-plane" in str(excinfo.value)

    # and a texture whose du channel is not the conj-swap of ud
    w = np.asarray(dftu["v_hubbard"], dtype=complex).copy()
    w[0, 3, 0, 0] = w[0, 3, 0, 0] + 0.19
    with pytest.raises(HubbardBundleError) as excinfo:
        hubbard_site_block_components(w[0])
    assert "conj(ud)" in str(excinfo.value)


# ---------------------------------------------------------------------------
# TEST-002: provenance audit (genuine passes, stripped/substituted refuse)
# ---------------------------------------------------------------------------


def test_audit_passes_on_genuine_outputs(dftu, audit_artifact):
    verdict = audit_xc_hubbard_provenance(audit_artifact)
    assert verdict["passed"] is True
    checks = verdict["checks"]
    assert checks["vertex_kind_xc_plus_hubbard"] is True
    assert checks["u_ry_matches_producer"] is True
    assert checks["j_ry_matches_producer"] is True
    assert checks["projector_identity"] == dftu["prov"].projector_identity
    assert len(checks["rotation_contract_mathdoc_sha256"]) == 64
    # XC source identity pinned and digested at export time (FR-014);
    # the digest is intrinsic to the certificate's stored-H payload
    assert checks["xc_source_kind"] == "stored_total_H"
    assert len(checks["xc_source_digest"]) == 64
    assert checks["xc_digest_intrinsic"] is True
    checked_bundle = HubbardBundle.from_certificate(audit_artifact["certificate"])
    assert audit_artifact["xc_source"]["digest"] == xc_source_payload_sha256(
        checked_bundle.stored_h
    )
    assert audit_artifact["version"] == "1.3"
    # Producer-anchored XC carrier: XC presence is a producer fact,
    # delivered by ingest_dftu_run from the producer HSX (FR-014).
    assert checks["xc_carrier_producer_anchored"] is True
    carrier = audit_artifact["xc_carrier_provenance"]
    assert carrier["carrier"] == "producer HSX via ingest_dftu_run"
    assert carrier["kind"] == "producer_hsx_stored_total_H"
    assert carrier["hsx"] == dftu["bundle"].stored_h_provenance["hsx"]
    # story-014 open item: the Edftu oracle is recorded, never gated
    info = verdict["informational"]
    assert info["edftu_reference_ev"] == float(dftu["data"]["edftu_ref_ev"])
    assert np.isfinite(info["edftu_delta_ev"])


def test_audit_refuses_stripped_xc_source(audit_artifact):
    """A +U export without an XC source identity is indistinguishable
    from an H0+V^U output with the XC contribution stripped (FR-014)."""
    for stripped in (
        {k: v for k, v in audit_artifact.items() if k != "xc_source"},
        {**audit_artifact, "xc_source": None},
        {**audit_artifact, "xc_source": {"kind": "stored_total_H"}},
        {**audit_artifact, "xc_source": {"kind": "hub_only", "digest": "0" * 64}},
    ):
        with pytest.raises(ProvenanceAuditError) as excinfo:
            audit_xc_hubbard_provenance(stripped)
        assert excinfo.value.reason == "stripped_xc_contribution"
        assert "xc_source" in str(excinfo.value) or "XC" in str(excinfo.value)


def test_audit_refuses_substituted_xc_digest(audit_artifact):
    """Substituting the XC digest after export is refused twice over:
    the recomputed stored-H digest no longer matches, and the
    export-time binding to the certificate breaks; the audit refuses
    fail-closed."""
    tampered = {
        **audit_artifact,
        "xc_source": {
            "kind": "stored_total_H",
            "digest": "ab" * 32,
        },
    }
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(tampered)
    assert excinfo.value.reason == "stripped_xc_contribution"
    assert "substituted" in str(excinfo.value)


def test_build_refuses_caller_payload(dftu):
    """The caller payload parameter is gone: presenting XC content as
    caller bytes is refused -- the digest provenance is the
    certificate's stored-H payload, not the caller's."""
    with pytest.raises(HubbardBundleError) as excinfo:
        build_xc_hubbard_audit_artifact(
            dftu["bundle"],
            vertex_kind="xc+hubbard",
            producer_u_ry=float(dftu["data"]["u_ry"]),
            producer_j_ry=float(dftu["data"]["j_ry"]),
            producer_u_j_source=dftu["refs"]["u_j_stdout"],
            xc_source={
                "kind": "stored_total_H",
                "payload": b"unrelated caller bytes",
            },
        )
    assert "payload" in str(excinfo.value)


def test_build_refuses_missing_stored_h(dftu):
    """A bundle whose certified state carries no stored-H payload has
    no XC carrier: the artifact cannot be built (empty_xc_source)."""
    bare = HubbardBundle(
        prov=dftu["prov"],
        occupations=dftu["occ"],
        v_hubbard=dftu["v_hubbard"],
        v_dc=dftu["v_dc"],
        edftu_ev=dftu["bundle"].edftu_ev,
        edftu_reference_ev=dftu["bundle"].edftu_reference_ev,
        conventions=dftu["bundle"].conventions,
    )
    with pytest.raises(ProvenanceAuditError) as excinfo:
        build_xc_hubbard_audit_artifact(
            bare,
            vertex_kind="xc+hubbard",
            producer_u_ry=float(dftu["data"]["u_ry"]),
            producer_j_ry=float(dftu["data"]["j_ry"]),
            producer_u_j_source=dftu["refs"]["u_j_stdout"],
        )
    assert excinfo.value.reason == "empty_xc_source"
    assert "no XC carrier" in str(excinfo.value)


def test_build_refuses_all_zero_stored_h(dftu):
    """An all-zero stored-H payload certifies a state with no XC
    contribution in the total H: refused as empty_xc_source.  The
    synthetic bundle uses a PRODUCER-REPRESENTATIVE carrier (the
    stored_h_provenance ingest_dftu_run sets) so the refusal is the
    all-zero gate, not the missing-producer-anchor gate."""
    zeroed = HubbardBundle(
        prov=dftu["prov"],
        occupations=dftu["occ"],
        v_hubbard=dftu["v_hubbard"],
        v_dc=dftu["v_dc"],
        edftu_ev=dftu["bundle"].edftu_ev,
        edftu_reference_ev=dftu["bundle"].edftu_reference_ev,
        conventions=dftu["bundle"].conventions,
        stored_h=np.zeros_like(dftu["stored_h"]),
        stored_h_provenance=dict(dftu["bundle"].stored_h_provenance),
    )
    with pytest.raises(ProvenanceAuditError) as excinfo:
        build_xc_hubbard_audit_artifact(
            zeroed,
            vertex_kind="xc+hubbard",
            producer_u_ry=float(dftu["data"]["u_ry"]),
            producer_j_ry=float(dftu["data"]["j_ry"]),
            producer_u_j_source=dftu["refs"]["u_j_stdout"],
        )
    assert excinfo.value.reason == "empty_xc_source"
    assert "all-zero" in str(excinfo.value)


def test_build_refuses_non_producer_stored_h(dftu):
    """A stored-H payload NOT set by the certified ingestion path
    (no producer anchor -- e.g. a Hubbard-only pencil) is not an XC
    carrier: refused as empty_xc_source (FR-014)."""
    unanchored_provenance = {
        "kind": "downstream_pencil",
        "note": "Hubbard-only Gamma pencil assembled below the producer",
    }
    with pytest.raises(HubbardBundleError) as excinfo:
        HubbardBundle(
            prov=dftu["prov"],
            occupations=dftu["occ"],
            v_hubbard=dftu["v_hubbard"],
            v_dc=dftu["v_dc"],
            edftu_ev=dftu["bundle"].edftu_ev,
            edftu_reference_ev=dftu["bundle"].edftu_reference_ev,
            conventions=dftu["bundle"].conventions,
            stored_h=dftu["stored_h"],
            stored_h_provenance=unanchored_provenance,
        )
    assert "producer_hsx_stored_total_H" in str(excinfo.value)
    # The producer anchor is also enforced on the import path: a
    # re-digested certificate carrying an unanchored payload cannot be
    # reimported (and the audit maps the defect to empty_xc_source --
    # see test_audit_refuses_audit_side_unanchored_stored_h).
    certificate = dftu["bundle"].to_certificate()
    certificate["payload"]["stored_h_provenance"] = unanchored_provenance
    arrays = {
        key: np.asarray(certificate["payload"][key]["re"], dtype=float)
        + 1j * np.asarray(certificate["payload"][key]["im"], dtype=float)
        for key in ("occupations", "v_hubbard", "v_dc", "stored_h")
    }
    certificate["payload_sha256"] = HubbardBundle._digest_with(
        certificate["payload"], arrays
    )
    with pytest.raises(HubbardBundleError) as excinfo:
        HubbardBundle.from_certificate(certificate)
    assert "producer_hsx_stored_total_H" in str(excinfo.value)


def test_audit_refuses_audit_side_unanchored_stored_h(dftu, audit_artifact):
    """Even re-digested consistently, a certificate whose stored-H
    payload lost the producer anchor has no XC carrier: the audit
    refuses (empty_xc_source)."""
    certificate = json.loads(json.dumps(audit_artifact["certificate"]))
    payload = certificate["payload"]
    payload["stored_h_provenance"] = {"kind": "downstream_pencil"}
    arrays = {
        key: np.asarray(payload[key]["re"], dtype=float)
        + 1j * np.asarray(payload[key]["im"], dtype=float)
        for key in ("occupations", "v_hubbard", "v_dc", "stored_h")
    }
    certificate["payload_sha256"] = HubbardBundle._digest_with(payload, arrays)
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance({**audit_artifact, "certificate": certificate})
    assert excinfo.value.reason == "empty_xc_source"
    assert "producer-anchored" in str(excinfo.value)


def test_audit_refuses_tampered_stored_h(audit_artifact):
    """Tampering the certificate's stored-H arrays (with a re-digested
    certificate so the payload digest stays consistent) is caught by
    the intrinsic recompute: the recorded XC digest no longer matches."""
    certificate = json.loads(json.dumps(audit_artifact["certificate"]))
    payload = certificate["payload"]
    re_arr = np.asarray(payload["stored_h"]["re"], dtype=float)
    payload["stored_h"]["re"] = (re_arr * 1.01).tolist()
    arrays = {
        key: np.asarray(payload[key]["re"], dtype=float)
        + 1j * np.asarray(payload[key]["im"], dtype=float)
        for key in ("occupations", "v_hubbard", "v_dc", "stored_h")
    }
    certificate["payload_sha256"] = HubbardBundle._digest_with(payload, arrays)
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance({**audit_artifact, "certificate": certificate})
    assert excinfo.value.reason == "stripped_xc_contribution"
    assert "recomputed" in str(excinfo.value)


def test_audit_refuses_audit_side_zeroed_stored_h(audit_artifact):
    """Even with consistent re-digestion, zeroing the stored-H payload
    removes the XC carrier entirely: refused as empty_xc_source."""
    certificate = json.loads(json.dumps(audit_artifact["certificate"]))
    payload = certificate["payload"]
    payload["stored_h"] = {
        "re": np.zeros_like(np.asarray(payload["stored_h"]["re"])).tolist(),
        "im": np.zeros_like(np.asarray(payload["stored_h"]["im"])).tolist(),
    }
    arrays = {
        key: np.asarray(payload[key]["re"], dtype=float)
        + 1j * np.asarray(payload[key]["im"], dtype=float)
        for key in ("occupations", "v_hubbard", "v_dc", "stored_h")
    }
    certificate["payload_sha256"] = HubbardBundle._digest_with(payload, arrays)
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance({**audit_artifact, "certificate": certificate})
    assert excinfo.value.reason == "empty_xc_source"


def test_audit_refuses_stripped_hubbard_declaration(audit_artifact):
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance({**audit_artifact, "vertex_kind": "xc"})
    assert excinfo.value.reason == "stripped_hubbard_contribution"
    assert "xc+hubbard" in str(excinfo.value)


def test_audit_refuses_stripped_hubbard_operator(dftu, audit_artifact):
    zeroed = HubbardBundle(
        prov=dftu["prov"],
        occupations=dftu["occ"],
        v_hubbard=np.zeros_like(dftu["v_hubbard"]),
        v_dc=dftu["v_dc"],
        edftu_ev=dftu["bundle"].edftu_ev,
        edftu_reference_ev=dftu["bundle"].edftu_reference_ev,
        conventions=dftu["bundle"].conventions,
        # Producer-representative carrier (stored_h_provenance as set by
        # ingest_dftu_run): the refusal under test is the stripped
        # Hubbard operator, not a missing XC anchor.
        stored_h=dftu["stored_h"],
        stored_h_provenance=dict(dftu["bundle"].stored_h_provenance),
    )
    stripped = build_xc_hubbard_audit_artifact(
        zeroed,
        vertex_kind="xc+hubbard",
        producer_u_ry=float(dftu["data"]["u_ry"]),
        producer_j_ry=float(dftu["data"]["j_ry"]),
        producer_u_j_source=dftu["refs"]["u_j_stdout"],
    )
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(stripped)
    assert excinfo.value.reason == "stripped_hubbard_contribution"
    assert "all-zero Hubbard operator" in str(excinfo.value)


def test_audit_refuses_substituted_u_j(audit_artifact, dftu):
    # the certificate (and its digest) stay untouched: substitution is
    # caught against the independent producer stdout reference
    tampered = {**audit_artifact, "producer_u_ry": float(dftu["data"]["u_ry"]) * 1.01}
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(tampered)
    assert excinfo.value.reason == "substituted_u_j"
    assert dftu["refs"]["u_j_stdout"] in str(excinfo.value)

    tampered_j = {**audit_artifact, "producer_j_ry": float(dftu["data"]["j_ry"]) - 0.01}
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(tampered_j)
    assert excinfo.value.reason == "substituted_u_j"


def _repackaged_certificate(dftu, prov_payload_mutator):
    """Re-digest the certificate after mutating the provenance payload."""
    certificate = dftu["bundle"].to_certificate()
    payload = json.loads(json.dumps(certificate["payload"]))
    prov_payload_mutator(payload["prov"])
    payload["prov"] = {k: v for k, v in payload["prov"].items()}
    tampered_certificate = {
        "kind": certificate["kind"],
        "version": certificate["version"],
        "payload": payload,
        "payload_sha256": HubbardBundle._digest_with(
            payload,
            {
                "occupations": dftu["bundle"].occupations.astype(complex),
                "v_hubbard": dftu["bundle"].v_hubbard.astype(complex),
                "v_dc": dftu["bundle"].v_dc.astype(complex),
                "stored_h": dftu["bundle"].stored_h.astype(complex),
            },
        ),
    }
    return tampered_certificate


def test_audit_refuses_stripped_projector_identity(dftu):
    def blank_identity(prov_payload):
        prov_payload["projector_identity"] = ""

    artifact = {
        "kind": "xc_hubbard_provenance_audit",
        "version": "1.0",
        "vertex_kind": "xc+hubbard",
        "certificate": _repackaged_certificate(dftu, blank_identity),
        "producer_u_ry": float(dftu["data"]["u_ry"]),
        "producer_j_ry": float(dftu["data"]["j_ry"]),
        "producer_u_j_source": dftu["refs"]["u_j_stdout"],
        "rotation_contract_mathdoc_sha256": "0" * 64,
    }
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(artifact)
    assert excinfo.value.reason == "stripped_projector_identity"
    assert "projector" in str(excinfo.value)


@pytest.mark.parametrize(
    "sha", [None, "0" * 64], ids=["missing_mathdoc_sha", "stale_mathdoc_sha"]
)
def test_audit_refuses_uncertified_rotation_contract(audit_artifact, sha):
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance(
            {**audit_artifact, "rotation_contract_mathdoc_sha256": sha}
        )
    assert excinfo.value.reason == "uncertified_rotation_contract"


def test_audit_refuses_tampered_certificate_payload(dftu, audit_artifact):
    certificate = json.loads(json.dumps(audit_artifact["certificate"]))
    certificate["payload"]["edftu_ev"] = float(certificate["payload"]["edftu_ev"]) + 1.0
    with pytest.raises(ProvenanceAuditError) as excinfo:
        audit_xc_hubbard_provenance({**audit_artifact, "certificate": certificate})
    assert excinfo.value.reason == "tampered_certificate_payload"


# ---------------------------------------------------------------------------
# IMPL-001: gate driver verdict
# ---------------------------------------------------------------------------


def test_gate_driver_verdict(
    dftu, gate, cross_check, contract_table, audit_artifact, moments, tmp_path
):
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["gate"] == "AC-008"
    assert verdict["passed"] is True, verdict["failure_reasons"]
    assert verdict["failure_reasons"] == []
    for key in ("vertex_sha256", "source_bundle_sha256"):
        assert verdict[key] == gate["response"].provenance[key]
        assert verdict[key] == gate["problem"].vertex.certificate[key]
        assert verdict[key] == cross_check[key]
    assert (
        verdict["raw_response_sha256"] == gate["response"].to_dict()["content_sha256"]
    )
    assert (
        verdict["hubbard_payload_sha256"]
        == dftu["bundle"].to_certificate()["payload_sha256"]
    )

    # declared ensemble (ADR-005 string + record)
    assert "ADR-005" in verdict["declared_ensemble_string"]
    assert "no cross-SCF comparison" in verdict["declared_ensemble_string"]
    assert verdict["declared_ensemble"]["kind"] == ENSEMBLE_GAPPED_T0
    assert verdict["declared_ensemble"]["n_occupied"] == NOCC

    # agreement table + tolerances
    assert [row["angle_rad"] for row in verdict["agreement_table"]] == list(
        ANGLE_LADDER
    )
    tolerances = verdict["tolerances"]
    assert tolerances["torque_ev_per_rad"] == AC008_TORQUE_BUDGET_EV_PER_RAD
    assert tolerances["curvature_ev_per_rad2"] == AC008_CURVATURE_BUDGET_EV_PER_RAD2
    assert tolerances["contract_energy_abs_ev"] == AC008_CONTRACT_ENERGY_BUDGET_EV
    assert tolerances["pinned_at"] == "story-016 approval"

    # provenance verdicts (audit wired into the driver)
    assert verdict["provenance_verdicts"]["audit"]["passed"] is True
    assert verdict["provenance_verdicts"]["rotation_contract"]["passed"] is True
    assert verdict["contract_wire_in"]["double_counting_frozen"] is True
    assert (
        verdict["contract_wire_in"]["worst_abs_delta_ev"]
        <= AC008_CONTRACT_ENERGY_BUDGET_EV
    )

    # texture provenance verbatim
    texture = verdict["texture_provenance"]
    assert "fe4_so_dftu" in texture["producer_run_id"]
    assert "DM.InitSpin 90 deg/atom" in texture["texture_description"]
    assert texture["edftu_reference_ev"] == float(dftu["data"]["edftu_ref_ev"])

    # the verdict report is JSON-serializable for the evidence pack
    # (tests write only inside tmp_path; the evidence pack write happens
    # at review time)
    report_path = tmp_path / "ac008_verdict.json"
    report_path.write_text(json.dumps(verdict, indent=1))
    reparsed = json.loads(report_path.read_text())
    assert reparsed["passed"] is True


def test_gate_driver_records_provenance_refusal(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    stripped = {**audit_artifact, "vertex_kind": "xc"}
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=stripped,
    )
    assert verdict["passed"] is False
    assert (
        "provenance_audit_stripped_hubbard_contribution" in verdict["failure_reasons"]
    )
    assert verdict["provenance_verdicts"]["audit"]["passed"] is False


def test_gate_driver_refuses_tampered_contract_energies(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    tampered = [
        dict(row, energy_contract_ev=row["energy_contract_ev"] + 1e-6)
        for row in contract_table
    ]
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=tampered,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["passed"] is False
    assert "contract_energy_budget_exceeded" in verdict["failure_reasons"]


def test_gate_driver_refuses_unpinned_cross_check_ladder(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    """The gate must refuse a cross-check computed over a different
    angle ladder: such a check can pass every budget while the verdict
    still reports the approved AC-008 ladder.  Refusal names the pinned
    values."""
    shrunk = {
        **cross_check,
        "angle_ladder_rad": tuple(ANGLE_LADDER[:5]),
        "rungs": cross_check["rungs"][:5],
    }
    with pytest.raises(HubbardBundleError) as excinfo:
        run_ac008_gate(
            dftu["bundle"],
            cross_check=shrunk,
            contract_table=contract_table,
            ensemble=gate["ensemble"],
            texture_provenance=texture_provenance(dftu, moments),
            audit_artifact=audit_artifact,
        )
    message = str(excinfo.value)
    for angle in AC008_ANGLE_LADDER_RAD:
        assert f"{round(angle, 6)}" in message
    assert "stable window" in message


def test_gate_driver_accepts_pinned_stable_window_ladder(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    """The pinned stable window (rungs 4-6) is an approved alternative
    to the full ladder."""
    window_angles = tuple(ANGLE_LADDER[i] for i in AC008_STABLE_WINDOW_RUNGS)
    window_rungs = [
        rung for rung in cross_check["rungs"] if rung["angle_rad"] in window_angles
    ]
    window_table = [row for row in contract_table if row["angle_rad"] in window_angles]
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check={
            **cross_check,
            "angle_ladder_rad": window_angles,
            "rungs": window_rungs,
        },
        contract_table=window_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["passed"] is True, verdict["failure_reasons"]


def test_gate_driver_refuses_missing_dc_freeze_evidence(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    """Absent dc evidence must fail closed: a contract row without
    'dc_change_max' proves nothing about double counting being frozen."""
    stripped = [
        {k: v for k, v in row.items() if k != "dc_change_max"} for row in contract_table
    ]
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=stripped,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["passed"] is False
    assert "missing_dc_freeze_evidence" in verdict["failure_reasons"]
    assert verdict["contract_wire_in"]["double_counting_frozen"] is False


def _identity_producer_case(source_ref):
    """Full d-shell shapes through the real extraction/response/FD producers.

    This deterministic model is not a genuine SIESTA export and deliberately
    lacks an XC audit carrier; it must not obtain a passing AC-008 verdict.
    The genuine-data gate above separately covers a passing certification.
    """
    from dataclasses import replace

    from test_siesta_spiral_hubbard import make_prov

    prov = make_prov()
    prov = replace(prov, source_refs={**prov.source_refs, "u_j_stdout": source_ref})
    occ = np.zeros((NSITES, 4, NORBD, NORBD), dtype=complex)
    occ[:, 0] = 0.8 * np.eye(NORBD)
    occ[:, 1] = 0.2 * np.eye(NORBD)
    vh, vd, conventions = assemble_hubbard_operator(occ, prov)
    energy = edftu_oracle(occ, vh, vd, prov=prov)
    hb = HubbardBundle(prov, occ, vh, vd, energy, energy, conventions=conventions)
    bundle = gamma_hubbard_bundle(vh, np.eye(NORB))
    # Add a known model XC field to the Hubbard Hamiltonian before declaring
    # the total-H +U provenance; a Hubbard-only pencil is not XC+Hubbard.
    sparse = {}
    for key, (up, down, real, imag) in bundle.sparse.items():
        delta = 0.1 if key[0] == key[1] else 0.0
        sparse[key] = (up + delta, down - delta, real, imag)
    bundle = replace(bundle, sparse=sparse, hubbard=prov)
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=ORBITALS_BY_SITE)
    frames = {site: _transverse_frame((0.0, 0.0, 1.0)) for site in SITE_LABELS}
    problem = FrozenResponseProblem(
        bundle, vertex, axes_by_site={s: tuple(f.items()) for s, f in frames.items()}
    )
    eig = eigh(*problem.pencil(0.0), eigvals_only=True)
    gap = float(eig[NOCC] - eig[NOCC - 1])
    fermi = float((eig[NOCC] + eig[NOCC - 1]) / 2)
    ensemble = Ensemble(ENSEMBLE_GAPPED_T0, NOCC, float(NOCC), fermi, 0.0, gap)
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=tuple((s, a) for s in SITE_LABELS for a in ("x", "y")),
        kpoints=[0.0],
        k_weights=[1.0],
        contour=EnergyContour.declared(
            z0=fermi,
            occupied_min=float(eig[0]),
            gap_ev=gap,
            margin_ev=0.3,
            n_nodes=512,
        ),
        finite_steps=(),
    )
    cross_check = validate_finite_rotations(
        problem,
        response,
        axes_by_site=frames,
        reference_directions_by_site={s: (0.0, 0.0, 1.0) for s in SITE_LABELS},
        ensemble=ensemble,
        kpoints=[0.0],
        k_weights=[1.0],
        angle_ladder=STABLE_WINDOW_ANGLES,
    ).finite_rotation_cross_check
    return hb, bundle, vertex, response, ensemble, cross_check


def test_gate_producer_identities_match_response_and_distinguish_source_refs():
    cases = [_identity_producer_case(f"reference-{i}.out:DFTUprojs") for i in (1, 2)]
    verdicts = []
    for hb, bundle, vertex, response, ensemble, cross_check in cases:
        before = response.to_dict()
        verdict = run_ac008_gate(
            hb,
            cross_check=cross_check,
            contract_table=[],
            ensemble=ensemble,
            texture_provenance={},
            audit_artifact={"certificate": hb.to_certificate()},
        )
        assert not verdict["passed"]  # no genuine XC carrier / contract evidence
        assert verdict["provenance_verdicts"]["input_certificate"]["matches_audit"]
        for key in ("vertex_sha256", "source_bundle_sha256"):
            assert verdict[key] == response.provenance[key] == vertex.certificate[key]
        assert verdict["raw_response_sha256"] == before["content_sha256"]
        assert response.to_dict() == before
        assert response.provenance["vertex_kind"] == "xc+hubbard"
        assert response.provenance["hubbard"] == bundle.hubbard.to_dict()
        from TB2J.interfaces.siesta_spiral_metallic import (
            evaluate_metallic_response_set,
        )

        metallic = evaluate_metallic_response_set(
            FrozenResponseProblem(
                bundle,
                vertex,
                axes_by_site={
                    s: tuple(_transverse_frame((0.0, 0.0, 1.0)).items())
                    for s in SITE_LABELS
                },
            ),
            ensemble=Ensemble(
                "metallic_fermi_dirac_fixed_n",
                NOCC,
                float(NOCC),
                ensemble.fermi_energy_ev,
                0.2,
                ensemble.gap_ev,
            ),
            coords=tuple((s, a) for s in SITE_LABELS for a in ("x", "y")),
            kpoints=[0.0],
            k_weights=[1.0],
            finite_steps=(),
        )
        assert metallic.provenance["vertex_kind"] == "xc+hubbard"
        assert metallic.provenance["hubbard"] == bundle.hubbard.to_dict()
        for key in ("vertex_sha256", "source_bundle_sha256"):
            assert metallic.provenance[key] == response.provenance[key]
        verdicts.append(verdict)
    for key in ("vertex_sha256", "source_bundle_sha256"):
        assert verdicts[0][key] != verdicts[1][key]
    assert cases[0][3].curvature == cases[1][3].curvature


@pytest.mark.parametrize(
    "key", ["vertex_sha256", "source_bundle_sha256", "raw_response_sha256"]
)
def test_gate_driver_refuses_absent_certified_identity(
    dftu, gate, cross_check, contract_table, audit_artifact, moments, key
):
    stripped = {k: v for k, v in cross_check.items() if k != key}
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=stripped,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert not verdict["passed"]
    assert "missing_certified_response_identity" in verdict["failure_reasons"]


def test_gate_driver_refuses_foreign_hubbard_audit_certificate(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    from dataclasses import replace

    other = replace(dftu["bundle"], fixture_digest="different-reference")
    verdict = run_ac008_gate(
        other,
        cross_check=cross_check,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert not verdict["passed"]
    assert (
        "audit_certificate_does_not_match_hubbard_input" in verdict["failure_reasons"]
    )
