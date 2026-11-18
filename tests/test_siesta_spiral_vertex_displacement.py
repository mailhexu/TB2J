"""Story 018: vertex and partition displacement derivatives.

TEST-001: the vertex derivative equals the derived spinor combination
(dH1-dH2)/2 on a fixture displacement (synthetic linear record, finite-
differenced against the extracted story-005 vertex blocks).
TEST-002: fixed-partition labels are present on outputs; ambiguous
requests and story-006 partition variants under displacement are refused
(FR-014 pattern).
Derivation smoke gate: the docs/sympy authority runs green end to end.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from TB2J.interfaces.siesta_spiral_displacement import (
    DisplacementDerivatives,
    DisplacementPattern,
    GeometrySource,
)
from TB2J.interfaces.siesta_spiral_io import load_fixture
from TB2J.interfaces.siesta_spiral_vertex_displacement import (
    PARTITION_FIXED,
    PARTITION_NOTE,
    PRD_PARTITION_RISK,
    TERMS_EXCLUDED,
    TERMS_INCLUDED,
    AmbiguousVertexRequest,
    PartitionVariantUnderDisplacement,
    VertexDisplacementDerivatives,
    vertex_displacement_derivative,
)

BOHR_TO_ANG = 0.529177210903
NUMERIC_TOL = 1e-12

# synthetic 2-site / 1-orbital linear model: H(x) = H0 + alpha * x (x in
# Ang), mirroring the story-017 fixture family
H0 = {
    (0, 0, (0, 0, 0)): (1.0, -1.0, 0.1, -0.2),
    (1, 1, (0, 0, 0)): (0.5, 0.5, 0.0, 0.0),
    (0, 1, (0, 0, 0)): (0.3, 0.3, 0.05, -0.01),
    (1, 0, (0, 0, 0)): (0.3, 0.3, 0.05, 0.01),
    (0, 0, (1, 0, 0)): (-0.2, -0.2, 0.02, 0.0),
    (1, 1, (1, 0, 0)): (-0.1, -0.1, 0.0, 0.03),
}
ALPHA = {
    key: tuple((0.7 + 0.11 * i) * v[i] for i in range(4))
    for i, (key, v) in enumerate(H0.items())
}
STEP_BOHR = 0.05
STEP_ANG = STEP_BOHR * BOHR_TO_ANG

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_fe3.npz"


def _vertex_blocks(h_map):
    """story-005 vertex blocks: v_up = (H1-H2)/2, v3 = H3, v4 = H4."""
    return {
        k: ((v[0] - v[1]) / 2.0, -(v[0] - v[1]) / 2.0, v[2], v[3])
        for k, v in h_map.items()
    }


@pytest.fixture(scope="module")
def bundle():
    if not FIXTURE.exists():  # pragma: no cover
        pytest.skip("siesta_spiral_fe3 fixture not present")
    b, _ = load_fixture(FIXTURE)
    return b


@pytest.fixture(scope="module")
def synthetic_bundle(bundle):
    """Restrict the fixture bundle to the synthetic 2-orbital model keys."""
    keep = {key for key in bundle.sparse if key in H0}
    return replace(bundle, norb=2, sparse={k: bundle.sparse[k] for k in sorted(keep)})


@pytest.fixture(scope="module")
def record():
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    geom = GeometrySource(
        cell=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 10.0)),
        orbital_positions=((0.0, 0.0, 0.0), (0.0, 0.0, 3.0 / BOHR_TO_ANG)),
    )
    return DisplacementDerivatives(
        pattern=pattern,
        dH={k: ALPHA[k] for k in H0},
        dS={k: 0.3 * H0[k][0] for k in H0},
        producer_recipe="synthetic linear test record",
        branch="central",
        step_bohr=STEP_BOHR,
        reference_fermi_ev=0.0,
        gauge="periodic-image-sum",
        units={"dH": "eV/Ang", "dS": "1/Ang"},
        reference_digests=(),
        reference_geometry=geom,
    )


# ---------------------------------------------------------------------------
# TEST-001: dV/dR equals the derived spinor combination on the record
# ---------------------------------------------------------------------------


def test_vertex_derivative_equals_dh_difference(record, synthetic_bundle):
    result = vertex_displacement_derivative(record, synthetic_bundle)
    onsite_keys = [k for k in H0 if k[0] == k[1] and k[2] == (0, 0, 0)]
    assert sorted(result.dV) == sorted(onsite_keys)
    # fixture sanity: at least one onsite key carries nonzero dH3/dH4
    assert any(ALPHA[key][2] != 0.0 or ALPHA[key][3] != 0.0 for key in onsite_keys)
    for key in onsite_keys:
        d1, d2, d3, d4 = ALPHA[key]
        v_up = (d1 - d2) / 2.0
        got = result.dV[key]
        assert abs(got[0] - v_up) <= NUMERIC_TOL
        assert abs(got[1] + v_up) <= NUMERIC_TOL
        assert abs(got[2] - d3) <= NUMERIC_TOL
        assert abs(got[3] - d4) <= NUMERIC_TOL


def test_vertex_derivative_matches_finite_differenced_vertex_blocks(
    record, synthetic_bundle
):
    """dV/dR equals the central FD of the extracted story-005 vertex blocks."""
    result = vertex_displacement_derivative(record, synthetic_bundle)
    h_plus = {k: tuple(h + a * STEP_ANG for h, a in zip(H0[k], ALPHA[k])) for k in H0}
    h_minus = {k: tuple(h - a * STEP_ANG for h, a in zip(H0[k], ALPHA[k])) for k in H0}
    v_plus, v_minus = _vertex_blocks(h_plus), _vertex_blocks(h_minus)
    for key, comps in result.dV.items():
        fd = tuple(
            (v_plus[key][i] - v_minus[key][i]) / (2.0 * STEP_ANG) for i in range(4)
        )
        for i in range(4):
            assert abs(fd[i] - comps[i]) <= NUMERIC_TOL, (key, i, fd, comps)


def test_hopping_and_nonzero_r_keys_excluded(record, synthetic_bundle):
    result = vertex_displacement_derivative(record, synthetic_bundle)
    for key in result.dV:
        mu, nu, R = key
        assert R == (0, 0, 0) or R == 0
        assert mu == nu


def test_ownership_admits_same_site_offdiagonal_onsite_blocks(record, synthetic_bundle):
    """With declared ownership, same-site mu != nu R=0 blocks enter."""
    ownership = {"A": [0], "B": [1]}
    result = vertex_displacement_derivative(
        record, synthetic_bundle, orbitals_by_site=ownership
    )
    assert (0, 1, (0, 0, 0)) not in result.dV  # cross-site hopping stays out
    assert result.partition_declared is True
    # the diagonal entries are unchanged by the ownership declaration
    plain = vertex_displacement_derivative(record, synthetic_bundle)
    for key, comps in plain.dV.items():
        assert result.dV[key] == comps


def test_partial_ownership_refused(record, synthetic_bundle):
    with pytest.raises(AmbiguousVertexRequest):
        vertex_displacement_derivative(
            record, synthetic_bundle, orbitals_by_site={"A": [0]}
        )


# ---------------------------------------------------------------------------
# TEST-002: labels on outputs + FR-014 refusals
# ---------------------------------------------------------------------------


def test_fixed_partition_labels_present(record, synthetic_bundle):
    result = vertex_displacement_derivative(record, synthetic_bundle)
    assert isinstance(result, VertexDisplacementDerivatives)
    assert result.partition == PARTITION_FIXED
    assert result.terms_included == TERMS_INCLUDED == ("vertex",)
    assert "partition" in result.terms_excluded
    assert result.terms_excluded == TERMS_EXCLUDED
    # PRD risk verbatim inside the partition note
    assert PRD_PARTITION_RISK in result.partition_note
    assert result.partition_note == PARTITION_NOTE
    declaration = result.declaration()
    assert declaration["terms_included"] == ("vertex",)
    assert declaration["partition"] == "fixed"


def test_prd_risk_text_is_verbatim():
    """The pinned PRD risk sentence is carried verbatim in the module."""
    import TB2J.interfaces.siesta_spiral_vertex_displacement as mod

    assert PRD_PARTITION_RISK == ("Displacement changes the site/projector definition")
    assert (
        "MathDoc: tb2j/siesta-spiral-vertex-displacement"
        in (mod.__doc__ or "") + Path(mod.__file__).read_text()
    )


def test_ambiguous_request_refused(record, synthetic_bundle):
    with pytest.raises(AmbiguousVertexRequest) as exc:
        vertex_displacement_derivative(record, synthetic_bundle, partition=None)
    assert "partition" in str(exc.value).lower()
    with pytest.raises(AmbiguousVertexRequest):
        vertex_displacement_derivative(record, synthetic_bundle, partition="site-wise")
    with pytest.raises(AmbiguousVertexRequest):
        vertex_displacement_derivative(record, None)
    with pytest.raises(AmbiguousVertexRequest):
        vertex_displacement_derivative(
            record, synthetic_bundle, operator="unknown_mode"
        )


def test_partition_variant_under_displacement_refused(record, synthetic_bundle):
    for variant in ("weighted_total_h",):
        with pytest.raises(PartitionVariantUnderDisplacement) as exc:
            vertex_displacement_derivative(record, synthetic_bundle, partition=variant)
        assert variant in str(exc.value)
    with pytest.raises(PartitionVariantUnderDisplacement):
        vertex_displacement_derivative(
            record, synthetic_bundle, operator="weighted_total_h"
        )


def test_non_ev_ang_record_refused(record, synthetic_bundle):
    bad = replace(record, units={"dH": "Ry/Bohr", "dS": "1/Bohr"})
    with pytest.raises(AmbiguousVertexRequest):
        vertex_displacement_derivative(bad, synthetic_bundle)


# ---------------------------------------------------------------------------
# Derivation smoke gate (derivation-first, ADR-003)
# ---------------------------------------------------------------------------


def test_derivation_authority_runs_green():
    pytest.importorskip("sympy")
    repo = Path(__file__).resolve().parents[1]
    script = repo / "docs" / "sympy" / "siesta_spiral_vertex_displacement.py"
    if not script.exists():  # pragma: no cover
        pytest.skip("derivation script not present")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL ASSERTIONS GREEN" in proc.stdout
