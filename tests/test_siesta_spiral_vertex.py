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
from TB2J.interfaces.siesta_spiral_vertex import MagneticVertex, extract_magnetic_vertex

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
    vertex = extract_magnetic_vertex(bundle)
    clone = MagneticVertex.from_dict(vertex.to_dict())
    assert clone.blocks.keys() == vertex.blocks.keys()
    for key in vertex.blocks:
        assert clone.blocks[key] == pytest.approx(vertex.blocks[key])
    assert clone.certificate == vertex.certificate
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
    assert (
        changed.certificate["source_bundle_sha256"]
        != vertex.certificate["source_bundle_sha256"]
    )


def test_partition_declaration_and_locality(fe3):
    bundle, _ = fe3
    per = bundle.norb // 3
    sites = {f"Fe{i+1}": list(range(i * per, (i + 1) * per)) for i in range(3)}
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=sites)
    assert vertex.certificate["partition_declaration"].startswith("two-sided")
    for site, members in vertex.partition.items():
        orbs = set(sites[site])
        for mu, nu, _R in members:
            assert mu in orbs or nu in orbs
    assert set().union(*vertex.partition.values()) == set(vertex.blocks)
    clone = MagneticVertex.from_dict(vertex.to_dict())
    assert clone.partition == vertex.partition
    bad = vertex.to_dict()
    other_key = next(
        k for k in vertex.partition["Fe2"] if k not in vertex.partition["Fe1"]
    )
    mu, nu, R = other_key
    import json

    bad["partition"]["Fe1"].append(json.dumps([mu, nu, list(R)]))
    with pytest.raises(SpiralImportError, match="digest mismatch"):
        MagneticVertex.from_dict(bad)


def test_fail_closed_on_bad_ownership(fe3):
    bundle, _ = fe3
    with pytest.raises(SpiralImportError, match="every orbital"):
        extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0]})
    with pytest.raises(SpiralImportError, match="disjoint"):
        extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0, 1], "B": [1, 2]})
    vertex = extract_magnetic_vertex(bundle)
    assert vertex.partition == {}
    assert vertex.certificate["partition_declaration"].startswith("undeclared")


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
    assert MagneticVertex.from_dict(vertex.to_dict()) == vertex
