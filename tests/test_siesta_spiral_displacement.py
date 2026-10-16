"""Story 017: displacement seam integration (dH/dR, dS/dR onto the spiral bundle).

The displacement data reuse the nonorth-spin-lattice conventions (ADR-008):
sparse derivatives over the SAME pattern as the bundle's H/S, provenance
carried verbatim, and one refusal class per mismatch kind.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral_displacement import (
    ConventionDrift,
    DhsdrNcProvider,
    DhsdrUnavailable,
    DisplacedReferenceFDProvider,
    DisplacementPattern,
    DisplacementSeamError,
    GaugeMismatch,
    GeometryMismatch,
    GeometrySource,
    MissingDisplacedReference,
    SparsePatternMismatch,
    UnsupportedBranch,
    assemble_onto_bundle,
    expected_units,
)
from TB2J.interfaces.siesta_spiral_io import load_fixture

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_fe3.npz"
BOHR_TO_ANG = 0.529177210903


# ---------------------------------------------------------------------------
# Synthetic displaced HSX siles (fake sisl backend, certified loader path)
# ---------------------------------------------------------------------------


class _FakeCSR:
    def __init__(self, entries, no_l, ncomp):
        # entries: {(io, o, nsc): tuple of components}
        rows = {}
        for (io, o, nsc), comp in entries.items():
            rows.setdefault(io, []).append((o, nsc, comp))
        ptr, col, D = [0], [], []
        for io in range(no_l):
            for o, nsc, comp in rows.get(io, []):
                col.append(o + nsc[0] * no_l)  # 1D cell encoding is enough here
                D.append(tuple(comp) + (0.0,) * (ncomp - len(comp)))
            ptr.append(len(col))
        self.ptr = np.asarray(ptr, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self._D = np.asarray(D, dtype=float)

    @property
    def shape(self):
        return (len(self.ptr) - 1, None)


class _FakeGeometry:
    def __init__(self, cell_bohr, positions_bohr, sc_off):
        self.cell = np.asarray(cell_bohr) * BOHR_TO_ANG  # sisl geometry is Ang
        self.xyz = np.asarray(positions_bohr) * BOHR_TO_ANG
        self.sc_off = np.asarray(sc_off, dtype=int)
        self.na = len(positions_bohr)
        self.firsto = np.arange(self.na)
        self.lasto = np.arange(self.na)


class _FakeHamiltonian:
    def __init__(self, entries, no_l):
        self._csr = _FakeCSR(entries, no_l, 4)

    def shift(self, ef):  # sisl shifts in place; the loader undoes it
        self._csr._D[:, :4] += ef  # identity on overlap-like scalar is fine for fakes


class _FakeOverlap:
    def __init__(self, entries, no_l):
        self._csr = _FakeCSR(entries, no_l, 4)


class _FakeHsxFourComponent:
    """Minimal stand-in for a sisl HSX sile with the certified read API."""

    def __init__(self, cell_bohr, positions_bohr, sc_off, h_entries, s_entries, ef):
        self._geom = _FakeGeometry(cell_bohr, positions_bohr, sc_off)
        no_l = len(positions_bohr)
        self._ham = _FakeHamiltonian(h_entries, no_l)
        s_csr = _FakeCSR(s_entries, no_l, 4)
        self._s = _FakeOverlap(s_entries, no_l)
        self._ef = ef
        # real SIESTA HSX files repeat the overlap as a 5th CSR column
        hcsr = self._ham._csr
        hcsr._D = np.concatenate([hcsr._D, s_csr._D[:, :1]], axis=1)

    def read_geometry(self):
        return self._geom

    def read_hamiltonian(self):
        return self._ham

    def read_overlap(self):
        return self._s

    def read_fermi_level(self):
        return self._ef


CELL = ((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 10.0))
SC_OFF = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
# 2 atoms, 1 orbital each; keys (io, o, nsc) with 4 file components H1..H4.
_BASE_H = {
    (0, 0, (0, 0, 0)): (1.0, -1.0, 0.1, -0.2),
    (1, 1, (0, 0, 0)): (0.5, 0.5, 0.0, 0.0),
    (0, 1, (0, 0, 0)): (0.3, 0.3, 0.05, -0.01),
    (1, 0, (0, 0, 0)): (0.3, 0.3, 0.05, 0.01),
    (0, 0, (1, 0, 0)): (-0.2, -0.2, 0.02, 0.0),
    (1, 1, (1, 0, 0)): (-0.1, -0.1, 0.0, 0.03),
}
_BASE_S = {
    (io, o, nsc): (1.0 if io == o else 0.4, 0.0, 0.0, 0.0) for (io, o, nsc) in _BASE_H
}
# Analytic linear model: H(x) = H0 + alpha * x (x in Ang), S(x) = S0 + beta * x.
ALPHA = {
    key: tuple((0.7 + 0.11 * i) * v[i] for i in range(4))
    for i, (key, v) in enumerate(_BASE_H.items())
}
BETA = {key: 0.3 * v[0] for key, v in _BASE_S.items()}


def _positions(z_atom1_bohr):
    return ((0.0, 0.0, z_atom1_bohr), (0.0, 0.0, 3.0))


def _fake_sile(z_atom1_bohr, ef=0.0):
    # file order: D4 = -(H4), the loader's diag3k fix restores H4
    h = {}
    for key, base in _BASE_H.items():
        vals = []
        for i in range(4):
            v = base[i] + ALPHA[key][i] * z_atom1_bohr * BOHR_TO_ANG
            vals.append(-v if i == 3 else v)
        h[key] = tuple(vals)
    s = {
        key: (v[0] + BETA[key] * z_atom1_bohr * BOHR_TO_ANG, 0.0, 0.0, 0.0)
        for key, v in _BASE_S.items()
    }
    return _FakeHsxFourComponent(CELL, _positions(z_atom1_bohr), SC_OFF, h, s, ef)


@pytest.fixture(scope="module")
def bundle():
    b, _ = load_fixture(FIXTURE)
    return b


@pytest.fixture(scope="module")
def synthetic_bundle():
    """A bundle built from the synthetic reference (z offset 0) via the
    certified loader, so its sparse pattern matches the FD records."""
    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    b = _bundle_from_hs_geometry(_fake_sile(0.0), "synthetic", "")
    return replace(
        b, metadata=dict(b.metadata, displacement_gauge="periodic-image-sum")
    )


@pytest.fixture
def synthetic_provider():
    pattern = DisplacementPattern(
        atom_index=1,
        axis="z",
        step_bohr=0.05,
        reference_digests=("sha256:plus", "sha256:minus"),
    )
    provider = DisplacedReferenceFDProvider(
        plus_sile=_fake_sile(+0.05),
        minus_sile=_fake_sile(-0.05),
        pattern=pattern,
        reference_fermi_ev=-4.5,
    )
    return pattern, provider.provide()


# ---------------------------------------------------------------------------
# TEST-001: ingestion round-trips with provenance
# ---------------------------------------------------------------------------


def test_fd_provider_matches_analytic_derivative(synthetic_provider):
    pattern, derivatives = synthetic_provider
    assert derivatives.branch == "central"
    assert derivatives.gauge == "periodic-image-sum"
    assert derivatives.units == expected_units()
    for key, analytic in ALPHA.items():
        assert derivatives.dH[key] == pytest.approx(analytic, abs=1e-10)
        assert derivatives.dS[key] == pytest.approx(BETA[key], abs=1e-10)


def test_assemble_roundtrips_provenance_onto_bundle(
    synthetic_provider, synthetic_bundle
):
    pattern, derivatives = synthetic_provider
    assembled = assemble_onto_bundle(synthetic_bundle, derivatives, pattern)
    assert assembled.bundle is synthetic_bundle
    assert assembled.derivatives == derivatives
    assert assembled.pattern == pattern
    prov = assembled.provenance["displacement"]
    assert prov["producer_recipe"] == derivatives.producer_recipe
    assert prov["branch"] == "central"
    assert prov["step_bohr"] == pattern.step_bohr
    assert prov["gauge"] == "periodic-image-sum"
    assert prov["reference_digests"] == pattern.reference_digests
    assert prov["reference_fermi_ev"] == derivatives.reference_fermi_ev
    assert prov["units"] == expected_units()
    assert prov["atom_index"] == 1
    assert prov["axis"] == "z"


def test_gauge_label_propagates_end_to_end(synthetic_provider, synthetic_bundle):
    pattern, derivatives = synthetic_provider
    declared = replace(
        synthetic_bundle,
        metadata=dict(synthetic_bundle.metadata, displacement_gauge="localized"),
    )
    derivatives = replace(derivatives, gauge="localized")
    assembled = assemble_onto_bundle(declared, derivatives, pattern)
    assert assembled.provenance["displacement"]["gauge"] == "localized"


# ---------------------------------------------------------------------------
# TEST-002: one refusal class per mismatch
# ---------------------------------------------------------------------------


def test_geometry_mismatch_refused(synthetic_provider, synthetic_bundle):
    pattern, derivatives = synthetic_provider
    wrong_cell = replace(
        synthetic_bundle,
        cell=tuple(tuple(v + 0.5 for v in row) for row in synthetic_bundle.cell),
    )
    with pytest.raises(GeometryMismatch):
        assemble_onto_bundle(wrong_cell, derivatives, pattern)
    shifted_all = replace(
        synthetic_bundle,
        orbital_positions=tuple(
            tuple(np.asarray(p) + np.array([0.0, 0.0, 0.01]))
            for p in synthetic_bundle.orbital_positions
        ),
    )
    with pytest.raises(GeometryMismatch):
        assemble_onto_bundle(shifted_all, derivatives, pattern)


def test_gauge_mismatch_missing_unknown_inconsistent(
    synthetic_provider, synthetic_bundle
):
    pattern, derivatives = synthetic_provider
    with pytest.raises(GaugeMismatch):  # missing
        assemble_onto_bundle(
            replace(
                synthetic_bundle,
                metadata={
                    k: v
                    for k, v in synthetic_bundle.metadata.items()
                    if k != "displacement_gauge"
                },
            ),
            derivatives,
            pattern,
        )
    unknown_bundle = replace(
        synthetic_bundle,
        metadata=dict(synthetic_bundle.metadata, displacement_gauge="bloch-summed"),
    )
    with pytest.raises(GaugeMismatch):  # unknown label on the bundle
        assemble_onto_bundle(unknown_bundle, derivatives, pattern)
    bad_record = replace(derivatives, gauge="bloch-summed")
    with pytest.raises(GaugeMismatch):  # unknown label on the record
        assemble_onto_bundle(synthetic_bundle, bad_record, pattern)
    other = replace(derivatives, gauge="localized")
    with pytest.raises(GaugeMismatch):  # inconsistent with the bundle
        assemble_onto_bundle(synthetic_bundle, other, pattern)


def test_unsupported_branch_refused():
    with pytest.raises(UnsupportedBranch):
        DisplacementPattern(atom_index=1, axis="z", step_bohr=0.05, branch="forward")
    with pytest.raises(UnsupportedBranch):
        DisplacementPattern(atom_index=1, axis="z", step_bohr=-0.05)


def test_sparse_pattern_mismatch_refused(synthetic_provider, synthetic_bundle):
    pattern, derivatives = synthetic_provider
    trim = replace(
        derivatives,
        dH={k: v for i, (k, v) in enumerate(derivatives.dH.items()) if i},
        dS={k: v for i, (k, v) in enumerate(derivatives.dS.items()) if i},
    )
    with pytest.raises(SparsePatternMismatch):
        assemble_onto_bundle(synthetic_bundle, trim, pattern)


def test_convention_drift_refused(synthetic_provider, synthetic_bundle):
    pattern, derivatives = synthetic_provider
    shifted_bundle = replace(synthetic_bundle, fermi_shift="sisl_-Ef")
    with pytest.raises(ConventionDrift):
        assemble_onto_bundle(shifted_bundle, derivatives, pattern)
    bad_units = replace(derivatives, units={"dH": "Ry/Bohr", "dS": "1/Bohr"})
    with pytest.raises(ConventionDrift):
        assemble_onto_bundle(synthetic_bundle, bad_units, pattern)
    mismatched_pattern = replace(pattern, step_bohr=0.04)
    with pytest.raises(ConventionDrift):
        assemble_onto_bundle(synthetic_bundle, derivatives, mismatched_pattern)


def test_missing_displaced_reference_refused(tmp_path):
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=0.05)
    provider = DisplacedReferenceFDProvider(
        plus_path=tmp_path / "plus.HSX",
        minus_path=tmp_path / "minus.HSX",
        pattern=pattern,
    )
    with pytest.raises(MissingDisplacedReference):
        provider.provide()


def test_fd_provider_refuses_divergent_displaced_patterns():
    plus = _fake_sile(+0.05)
    trimmed_entries = dict(list(_BASE_H.items())[:-1])
    trimmed_s = {k: v for k, v in _BASE_S.items() if k in trimmed_entries}
    broken = _FakeHsxFourComponent(
        CELL, _positions(-0.05), SC_OFF, trimmed_entries, trimmed_s, 0.0
    )
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=0.05)
    provider = DisplacedReferenceFDProvider(
        plus_sile=plus, minus_sile=broken, pattern=pattern
    )
    with pytest.raises(SparsePatternMismatch):
        provider.provide()


# ---------------------------------------------------------------------------
# DhsdrNcProvider: import guard + synthetic decode
# ---------------------------------------------------------------------------


def test_dhsdr_provider_import_guard():
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=0.05)
    provider = DhsdrNcProvider(nc_path="/nonexistent/dHSdR.nc", pattern=pattern)
    with pytest.raises(DhsdrUnavailable) as exc:
        provider.provide()
    message = str(exc.value)
    assert "HamiltonIO.siesta.dhsdr" in message
    assert "DisplacedReferenceFDProvider" in message


def test_dhsdr_provider_decode_with_fake_reader(bundle, monkeypatch):
    norb = bundle.norb
    Ry_eV = 13.605693122994
    Bohr_Ang = 0.529177210903
    rlist = sorted({R for _, _, R in bundle.sparse})
    ncell = len(rlist)
    dhr = np.zeros((ncell, norb, norb, 2))
    dsr = np.zeros((ncell, norb, norb))
    rng = np.random.default_rng(7)
    dense0 = rng.standard_normal((norb, norb))
    dense0 = dense0 + dense0.T
    dsr[0] = rng.standard_normal((norb, norb))
    dsr[0] = dsr[0] + dsr[0].T
    for i in range(ncell):
        dhr[i, :, :, 0] = (i + 1) * dense0
        dhr[i, :, :, 1] = -(i + 1) * dense0
        if i:
            dsr[i] = (i + 1) * dsr[0]

    class _Block:
        atom_index = 1
        axis = 3
        Rlist = rlist
        dHR = dhr
        dSR = dsr

    def fake_read_dHSdR(path, displacements=None):
        assert displacements == [(1, 3)]
        return [_Block()]

    fake_mod = types.ModuleType("HamiltonIO.siesta.dhsdr")
    fake_mod.read_dHSdR = fake_read_dHSdR
    monkeypatch.setitem(sys.modules, "HamiltonIO.siesta.dhsdr", fake_mod)

    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=0.05)
    provider = DhsdrNcProvider(
        nc_path="dHSdR.nc",
        pattern=pattern,
        reference_geometry=GeometrySource(
            cell=bundle.cell,
            orbital_positions=bundle.orbital_positions,
            displaced_axis_offset_bohr=0.0,
        ),
        reference_fermi_ev=-4.5,
        sparse_keys=set(bundle.sparse),
    )
    derivatives = provider.provide()
    assert derivatives.gauge == "periodic-image-sum"
    factor = Ry_eV / Bohr_Ang
    dense_by_R = {R: (i + 1) * dense0 for i, R in enumerate(rlist)}
    for (mu, nu, R), value in derivatives.dH.items():
        dense = dense_by_R[R]
        expected = (
            factor * dense[mu, nu],
            -factor * dense[mu, nu],
            0.0,
            0.0,
        )
        assert value == pytest.approx(expected, abs=1e-10)
    for (mu, nu, R), value in derivatives.dS.items():
        i = rlist.index(R)
        assert value == pytest.approx(dsr[i][mu, nu] / Bohr_Ang, abs=1e-10)
    declared = replace(
        bundle, metadata=dict(bundle.metadata, displacement_gauge="periodic-image-sum")
    )
    assembled = assemble_onto_bundle(declared, derivatives, pattern)
    assert assembled.provenance["displacement"]["producer_recipe"].startswith(
        "siesta dHSdR.nc"
    )


# ---------------------------------------------------------------------------
# Real displaced campaign (skip-gated)
# ---------------------------------------------------------------------------


CAMPAIGN = Path(
    "/home/hexu/projects/TB2J_dev/.tmp/siesta-xc-mft/fe4_displaced_campaign"
)
VAULT_FE4 = Path(
    "/home/hexu/projects/memnotes/memnotes/Projects/TB2J/evidence/"
    "2026-10-15-siesta-finite-pitch-mapping/fe4"
)


@pytest.fixture(scope="module")
def real_campaign():
    if not CAMPAIGN.is_dir() or not (VAULT_FE4 / "fe4.HSX").is_file():
        pytest.skip("fe4 displaced campaign or vault reference absent")
    import sisl

    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    def open_hsx(path):
        return sisl.get_sile(str(path))

    reference = _bundle_from_hs_geometry(
        open_hsx(VAULT_FE4 / "fe4.HSX"), "fe4.out", "Fe_nc.ion.nc"
    )
    reference = replace(
        reference,
        metadata=dict(reference.metadata, displacement_gauge="periodic-image-sum"),
    )
    return reference


@pytest.mark.skipif(
    not CAMPAIGN.is_dir() or not (VAULT_FE4 / "fe4.HSX").is_file(),
    reason="fe4 displaced campaign or vault reference absent",
)
def test_real_fe4_z_displacement_der(real_campaign):
    bundle = real_campaign
    pattern = DisplacementPattern(
        atom_index=1,
        axis="z",
        step_bohr=0.05,
    )
    provider = DisplacedReferenceFDProvider(
        plus_path=CAMPAIGN / "a1zp" / "fe4d_a1zp.HSX",
        minus_path=CAMPAIGN / "a1zm" / "fe4d_a1zm.HSX",
        pattern=pattern,
        reference_fermi_ev=None,
    )
    derivatives = provider.provide()
    # pattern match against the bundle's own H/S pattern
    assert set(derivatives.dH) == set(bundle.sparse)
    assembled = assemble_onto_bundle(bundle, derivatives, pattern)
    prov = assembled.provenance["displacement"]
    for fieldname in (
        "producer_recipe",
        "branch",
        "step_bohr",
        "gauge",
        "reference_digests",
        "reference_fermi_ev",
        "units",
    ):
        assert fieldname in prov and prov[fieldname] is not None
    max_dH = max(abs(c) for v in derivatives.dH.values() for c in v)
    max_dS = max(abs(v) for v in derivatives.dS.values())
    assert 0.0 < max_dH < 1e4  # sane eV/Ang magnitude for an SZP Fe chain
    print(
        f"\n[real fe4 a1z] max |dH/dR| = {max_dH:.6f} eV/Ang; max |dS/dR| = {max_dS:.6f} 1/Ang"
    )


def test_refusal_classes_are_distinct():
    classes = (
        GeometryMismatch,
        GaugeMismatch,
        MissingDisplacedReference,
        ConventionDrift,
        SparsePatternMismatch,
        UnsupportedBranch,
        DhsdrUnavailable,
    )
    assert all(issubclass(c, DisplacementSeamError) for c in classes)
    assert len({c.__name__ for c in classes}) == len(classes)
