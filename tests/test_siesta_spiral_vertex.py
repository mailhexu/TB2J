"""Story-005 production magnetic vertex tests on real Fe3 and O2 fixtures."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral import (
    BUNDLE_VERSION,
    SpiralBundle,
    SpiralImportError,
)
from TB2J.interfaces.siesta_spiral_io import load_fixture
from TB2J.interfaces.siesta_spiral_vertex import (
    DEFAULT_OPERATOR_MODE,
    OPERATOR_MODE_WEIGHTED_TOTAL_H,
    PARTITION_DECLARATION,
    PARTITION_DECLARATION_ONSITE,
    MagneticVertex,
    extract_magnetic_vertex,
    rotate_site_blocks,
)

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_fe3.npz"


@pytest.fixture(scope="module")
def fe3():
    return load_fixture(FIXTURE)


@pytest.fixture(params=["fe3", "o2"])
def producer(request):
    return load_fixture(FIXTURE.with_name(f"siesta_spiral_{request.param}.npz"))


def test_vertex_is_traceless_spin_split_of_saved_h(producer):
    bundle, _ = producer
    vertex = extract_magnetic_vertex(bundle)
    assert vertex.certificate["vertex_convention"] == "oroszlany_total_h_spin_split"
    for key, (v_up, v_dn, v3, v4) in vertex.blocks.items():
        h1, h2, h3, h4 = bundle.sparse[key]
        assert v_up == pytest.approx(0.5 * (h1 - h2))
        assert v_dn == pytest.approx(-v_up)
        assert (v3, v4) == (h3, h4)


def test_onsite_vs_intersite_inventory(fe3):
    bundle, _ = fe3
    vertex = extract_magnetic_vertex(bundle)
    onsite = [k for k in vertex.blocks if k[0] == k[1] and all(r == 0 for r in k[2])]
    intersite = [k for k in vertex.blocks if k not in onsite]

    def mix(keys):
        return np.sqrt(
            np.mean([vertex.blocks[k][2] ** 2 + vertex.blocks[k][3] ** 2 for k in keys])
        )

    assert mix(onsite) > mix(intersite) > 0.0  # no onsite-only truncation


def test_certificate_roundtrip_and_tamper_detection(producer):
    bundle, _ = producer
    for operator in (DEFAULT_OPERATOR_MODE, OPERATOR_MODE_WEIGHTED_TOTAL_H):
        vertex = extract_magnetic_vertex(bundle, operator=operator)
        assert vertex.certificate["operator_mode"] == operator
        clone = MagneticVertex.from_dict(vertex.to_dict())
        assert clone.blocks.keys() == vertex.blocks.keys()
        for key in vertex.blocks:
            assert clone.blocks[key] == pytest.approx(vertex.blocks[key])
        assert clone.certificate == vertex.certificate
    vertex = extract_magnetic_vertex(bundle)
    bad = vertex.to_dict()
    first = next(iter(bad["blocks"]))
    bad["blocks"][first] = [v + 1.0 for v in bad["blocks"][first]]
    with pytest.raises(SpiralImportError, match="digest mismatch"):
        MagneticVertex.from_dict(bad)
    bad = vertex.to_dict()
    bad["certificate"]["vertex_convention"] = "xc_grid_vertex"
    with pytest.raises(SpiralImportError, match="vertex_convention mismatch"):
        MagneticVertex.from_dict(bad)
    bad = vertex.to_dict()
    bad["certificate"]["q"] = [0.0, 0.0, 0.0]
    with pytest.raises(SpiralImportError, match="digest mismatch"):
        MagneticVertex.from_dict(bad)
    # the operator mode is hash-pinned into the digest. On an undeclared
    # vertex the declaration text is mode-independent, so swapping to the
    # other known mode without re-digesting surfaces as a digest mismatch
    undeclared = extract_magnetic_vertex(bundle)
    bad = undeclared.to_dict()
    bad["certificate"]["operator_mode"] = OPERATOR_MODE_WEIGHTED_TOTAL_H
    with pytest.raises(SpiralImportError, match="digest mismatch"):
        MagneticVertex.from_dict(bad)
    # a certificate missing the mode or carrying an unknown mode is
    # refused outright, before the digest gate
    bad = vertex.to_dict()
    del bad["certificate"]["operator_mode"]
    with pytest.raises(SpiralImportError, match="operator_mode mismatch"):
        MagneticVertex.from_dict(bad)
    bad = vertex.to_dict()
    bad["certificate"]["operator_mode"] = "onsite_delta_total_h"
    with pytest.raises(SpiralImportError, match="operator_mode mismatch"):
        MagneticVertex.from_dict(bad)
    # a mode swap on a DECLARED vertex is caught by the per-mode
    # declaration gate (onsite text recorded, weighted text expected)
    # before the digest is consulted
    per_site = bundle.norb // 3 if bundle.norb % 3 == 0 else bundle.norb // 2
    sites = (
        {f"S{i + 1}": list(range(i * per_site, (i + 1) * per_site)) for i in range(3)}
        if bundle.norb % 3 == 0
        else {
            "S1": list(range(per_site)),
            "S2": list(range(per_site, bundle.norb)),
        }
    )
    declared = extract_magnetic_vertex(bundle, orbitals_by_site=sites)
    bad = declared.to_dict()
    bad["certificate"]["operator_mode"] = OPERATOR_MODE_WEIGHTED_TOTAL_H
    with pytest.raises(SpiralImportError, match="partition_declaration mismatch"):
        MagneticVertex.from_dict(bad)


def test_pitch_overlap_bundle_identity_is_bound(producer):
    bundle, _ = producer
    vertex = extract_magnetic_vertex(bundle)
    q_other = tuple(q + (0.01 if i == 2 else 0.0) for i, q in enumerate(bundle.q))
    shifted = replace(bundle, q=q_other)
    changed = extract_magnetic_vertex(shifted)
    assert (
        changed.certificate["source_bundle_sha256"]
        != vertex.certificate["source_bundle_sha256"]
    )
    assert changed.certificate["vertex_sha256"] != vertex.certificate["vertex_sha256"]
    overlap = dict(bundle.overlap)
    key = next(iter(overlap))
    pair = (key[1], key[0], tuple(-r for r in key[2]))
    overlap[key] += 0.01
    overlap[pair] += 0.01
    changed = extract_magnetic_vertex(replace(bundle, overlap=overlap))
    assert changed.certificate["vertex_sha256"] != vertex.certificate["vertex_sha256"]


def test_partition_declaration_and_locality(fe3):
    bundle, _ = fe3
    per = bundle.norb // 3
    sites = {f"Fe{i+1}": list(range(i * per, (i + 1) * per)) for i in range(3)}
    # default: the onsite-delta operator declares only the R=0 same-site
    # blocks of each site; intersite magnetic blocks are excluded
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=sites)
    assert vertex.certificate["operator_mode"] == "onsite_delta"
    assert vertex.certificate["partition_declaration"] == PARTITION_DECLARATION_ONSITE
    for site, members in vertex.partition.items():
        orbs = set(sites[site])
        for mu, nu, R in members:
            assert mu == nu and mu in orbs
            assert all(r == 0 for r in (R if isinstance(R, tuple) else (R,)))
    onsite_keys = {
        k for k in vertex.blocks if k[0] == k[1] and all(r == 0 for r in k[2])
    }
    assert set().union(*vertex.partition.values()) == onsite_keys
    assert onsite_keys < set(vertex.blocks)  # intersite blocks exist, excluded
    clone = MagneticVertex.from_dict(vertex.to_dict())
    assert clone.partition == vertex.partition
    # explicit weighted variant keeps the version-2 incidence declaration
    weighted = extract_magnetic_vertex(
        bundle, orbitals_by_site=sites, operator=OPERATOR_MODE_WEIGHTED_TOTAL_H
    )
    assert weighted.certificate["operator_mode"] == OPERATOR_MODE_WEIGHTED_TOTAL_H
    assert weighted.certificate["partition_declaration"] == PARTITION_DECLARATION
    for site, members in weighted.partition.items():
        orbs = set(sites[site])
        for mu, nu, _R in members:
            assert mu in orbs or nu in orbs
    assert set().union(*weighted.partition.values()) == set(weighted.blocks)
    bad = weighted.to_dict()
    other_key = next(
        k for k in weighted.partition["Fe2"] if k not in weighted.partition["Fe1"]
    )
    mu, nu, R = other_key
    import json

    bad["partition"]["Fe1"].append(json.dumps([mu, nu, list(R)]))
    with pytest.raises(SpiralImportError, match="digest mismatch"):
        MagneticVertex.from_dict(bad)
    # a mixed certificate (onsite_delta mode recorded over the weighted
    # declaration text) describes two different operators: refused on the
    # per-mode declaration gate before the digest is even consulted
    mixed = weighted.to_dict()
    mixed["certificate"]["operator_mode"] = "onsite_delta"
    with pytest.raises(SpiralImportError, match="partition_declaration mismatch"):
        MagneticVertex.from_dict(mixed)
    # a stale pre-2.1 vertex (version 2.0, no operator_mode field) fails
    # closed on the version gate before the mode check
    stale_v2_0 = vertex.to_dict()
    stale_v2_0["vertex_version"] = "2.0"
    stale_v2_0["certificate"]["vertex_version"] = "2.0"
    with pytest.raises(SpiralImportError, match="unsupported magnetic-vertex version"):
        MagneticVertex.from_dict(stale_v2_0)
    # the pre-2.0 unweighted declaration remains refused under the current
    # version tag as well
    old_declaration = (
        "two-sided same-U conjugation on entries with mu or nu in the " "declared site"
    )
    stale_declaration = vertex.to_dict()
    stale_declaration["certificate"]["partition_declaration"] = old_declaration
    with pytest.raises(SpiralImportError, match="partition_declaration mismatch"):
        MagneticVertex.from_dict(stale_declaration)


def test_fail_closed_on_bad_ownership(fe3):
    bundle, _ = fe3
    with pytest.raises(SpiralImportError, match="every orbital"):
        extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0]})
    with pytest.raises(SpiralImportError, match="disjoint"):
        extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0, 1], "B": [1, 2]})
    vertex = extract_magnetic_vertex(bundle)
    assert vertex.partition == {}
    assert vertex.certificate["partition_declaration"].startswith("undeclared")
    assert vertex.certificate["operator_mode"] == DEFAULT_OPERATOR_MODE
    with pytest.raises(SpiralImportError, match="unknown magnetic operator"):
        extract_magnetic_vertex(bundle, operator="xc_reconstructed")


def test_default_onsite_delta_leaves_intersite_blocks_frozen(fe3):
    bundle, _ = fe3
    per = bundle.norb // 3
    sites = {f"Fe{i+1}": list(range(i * per, (i + 1) * per)) for i in range(3)}
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=sites)
    for site in sites:
        rotated = rotate_site_blocks(vertex, site, axis=(1.0, 1.0, 1.0), angle=0.9)
        changed = False
        for key in vertex.blocks:
            if key in vertex.partition[site]:
                # members are exactly the R=0 same-site blocks
                assert key[0] == key[1] and all(r == 0 for r in key[2])
                changed = changed or rotated[key] != vertex.blocks[key]
            else:
                # intersite magnetic blocks stay frozen under any site
                # rotation, like the scalar part
                assert rotated[key] == vertex.blocks[key]
        assert changed  # the rotation does move the site's onsite blocks


def test_scalar_cell_vertex_certificate_roundtrip():
    bundle = SpiralBundle(
        version=BUNDLE_VERSION,
        norb=1,
        ncell=3,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse={(0, 0, 0): (2.0, -2.0, 0.5, 0.0)},
        overlap={(0, 0, 0): 1.0},
    )
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"site": [0]})
    assert vertex.certificate["operator_mode"] == DEFAULT_OPERATOR_MODE
    assert vertex.certificate["partition_declaration"] == PARTITION_DECLARATION_ONSITE
    assert MagneticVertex.from_dict(vertex.to_dict()) == vertex
