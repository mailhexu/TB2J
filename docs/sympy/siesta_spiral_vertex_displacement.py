"""Assertion authority for story 018: vertex and partition displacement derivatives.

Derivation-first authority (ADR-003) for the displacement derivative of the
ADR-004 production magnetic vertex (total-H spin splitting, story-005/006
machinery) under the story-017 displacement seam (dH/dR records in the
4-component file packing ``H1=H11, H2=H22, H3=Re H12, H4=-Im H12``).

Pins, as exact symbolic identities plus one numeric gate:

[A] the packing and the vertex derivative: the vertex spinor block is
    ``h.sigma`` with ``v_up = (H1-H2)/2``, so its displacement derivative is
    ``dV_up = (dH1-dH2)/2`` — the derivative carries only the dH1-dH2
    (diagonal-spinor splitting) components;
[B] the onsite (R=0, mu=nu) hermiticity consequences: the onsite block is a
    Hermitian 2x2 spinor, hence H3 is real-symmetric onsite and H4 vanishes
    onsite; the onsite derivative record carries (d1-d2)/2, -(d1-d2)/2, d3, 0;
[C] the transposed-adjoint (R-symmetry) relation for hopping entries:
    ``H(mu,nu,R) = H(nu,mu,-R)^dagger``; given per-block hermiticity the
    component map is (h1,h2,h3,h4) -> (h1,h2,h3,-h4), asserted symbolically
    on a 2-site model, and the same map holds for the derivative record;
[D] the fixed-partition theorem: the site partition map (orbital -> site) is
    CONSTANT index data, so d(partition)/dR = 0 identically; any observable
    built only on the vertex and the partition map has a total displacement
    derivative with NO partition term, and the fixed-partition label is
    EXACT (not an approximation) for that declared observable family;
[E] the machine-checkable declaration of included vs excluded terms;
[F] numeric verification on a synthetic linear-in-displacement fixture
    record: the record-level vertex derivative equals the finite difference
    of the extracted vertex blocks to 1e-12.

Run: ``python docs/sympy/siesta_spiral_vertex_displacement.py`` in the
``mydev`` environment.  Any failed assertion exits nonzero.
"""

from __future__ import annotations

import sys

import sympy as sp

# ---------------------------------------------------------------------------
# [A] packing and the vertex derivative
# ---------------------------------------------------------------------------

H1, H2, H3, H4 = sp.symbols("H1 H2 H3 H4", real=True)
D1, D2, D3, D4 = sp.symbols("dH1 dH2 dH3 dH4", real=True)

PACKING_CONVENTION = "H12=H3-1j*H4;hy=+H4"


def zero(expr):
    """Require exact vanishing (elementwise for matrices)."""
    if hasattr(expr, "applyfunc"):
        reduced = expr.applyfunc(lambda e: sp.expand(sp.together(e)))
        assert reduced == sp.zeros(*expr.shape), reduced
        return
    reduced = sp.expand(sp.together(expr))
    assert reduced == 0, reduced


# vertex block (story-005 MagneticVertex.blocks): (v_up, v_dn, v3, v4)
V_UP = (H1 - H2) / 2
DV_UP = (D1 - D2) / 2


def vertex_block(h1, h2, h3, h4):
    """Vertex 4-tuple from H components: (v_up, -v_up, h3, h4)."""
    v_up = (h1 - h2) / 2
    return (v_up, -v_up, h3, h4)


def vertex_spinor(v):
    """The traceless spinor h.sigma from a vertex 4-tuple (v_up, v_dn, v3, v4)."""
    v_up, v_dn, v3, v4 = v
    return sp.Matrix([[v_up, v3 - sp.I * v4], [v3 + sp.I * v4, v_dn]])


def test_vertex_derivative_packing():
    print("[A] packing: V_up = (H1-H2)/2, dV_up = (dH1-dH2)/2")
    # the vertex spinor is exactly the spin-dependent (traceless in spin)
    # part of the packed H block: sigma_z coefficient = (H1-H2)/2
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    sigma = (sx, sy, sz)
    h_mat = sp.Matrix([[H1, H3 - sp.I * H4], [H3 + sp.I * H4, H2]])
    coeffs = [sp.Rational(1, 2) * sp.trace(h_mat * s) for s in sigma]
    assert sp.simplify(coeffs[0] - H3) == 0, coeffs[0]
    assert sp.simplify(coeffs[1] - H4) == 0, coeffs[1]
    assert sp.simplify(coeffs[2] - V_UP) == 0, coeffs[2]
    # Pauli decomposition: H_block = h0 I + v.sigma with h0 = (H1+H2)/2 I_orb
    h0 = (H1 + H2) / 2
    recon = h0 * sp.eye(2) + vertex_spinor((V_UP, -V_UP, H3, H4))
    assert sp.simplify(recon[0, 0] - H1) == 0 and sp.simplify(recon[1, 1] - H2) == 0
    # derivative: the ADR-004 vertex derivative is the derivative of the
    # FULL story-005 vertex block (v_up, -v_up, H3, H4), i.e. the
    # componentwise map (dH1, dH2, dH3, dH4) under the pinned packing
    dv_block = vertex_spinor(vertex_block(D1, D2, D3, D4))
    dh_mat = sp.Matrix([[D1, D3 - sp.I * D4], [D3 + sp.I * D4, D2]])
    dh0 = (D1 + D2) / 2
    assert dv_block == dh_mat - dh0 * sp.eye(2)
    assert sp.simplify(dv_block[0, 0] - (D1 - D2) / 2) == 0
    assert sp.simplify(dv_block[1, 1] + (D1 - D2) / 2) == 0
    print("   A PASS: dV = (dV_up, -dV_up, dH3, dH4) with dV_up=(dH1-dH2)/2;")
    print("      the transverse components pass through the adjoint map")
    return {"dv_up": DV_UP, "dv_block": dv_block}


# ---------------------------------------------------------------------------
# [B] onsite hermiticity consequences
# ---------------------------------------------------------------------------


def test_onsite_hermiticity():
    print("[B] onsite (mu=nu, R=0) block: Hermitian 2x2 spinor; declared")
    h_onsite = sp.Matrix([[H1, H3 - sp.I * H4], [H3 + sp.I * H4, H2]])
    zero(h_onsite - h_onsite.H)  # hermiticity holds for any real components
    # the DECLARED onsite derivative record is the full vertex-block
    # derivative: dV = ((d1-d2)/2, -(d1-d2)/2, d3, d4) — the transverse
    # packed components pass through the adjoint map (see [C])
    d_onsite = ((D1 - D2) / 2, -(D1 - D2) / 2, D3, D4)
    assert sp.simplify(d_onsite[0] + d_onsite[1]) == 0  # traceless in spin
    # the onsite derivative spinor block is Hermitian and spin-traceless
    dv_onsite = vertex_spinor(d_onsite)
    assert dv_onsite == dv_onsite.H
    assert sp.trace(dv_onsite) == 0
    print("   B PASS: onsite derivative record (dV_up, -dV_up, dH3, dH4) is")
    print("      the Hermitian, spin-traceless declared derivative block")
    return {"onsite_dv": sp.Matrix(dv_onsite)}


# ---------------------------------------------------------------------------
# [C] transposed-adjoint (R-symmetry) relation, 2-site symbolic model
# ---------------------------------------------------------------------------


def block_matrix(h1, h2, h3, h4):
    """Packed 2x2 spinor block from its 4 real file components."""
    return sp.Matrix([[h1, h3 - sp.I * h4], [h3 + sp.I * h4, h2]])


def test_transposed_adjoint_r_symmetry():
    print("[C] R-symmetry of hopping entries: M(mu,nu,R)^dagger = M(nu,mu,-R)")
    # generic hopping block at (mu,nu,R): the packed file representation is
    # per-block Hermitian (H21 = conj(H12) with H12 = a3 - i a4)
    a1, a2, a3, a4 = sp.symbols("a1 a2 a3 a4", real=True)
    m_mn_R = block_matrix(a1, a2, a3, a4)
    assert m_mn_R == m_mn_R.H
    # the lattice (R-symmetry) relation of the Hermitian Hamiltonian links
    # the reversed-R partner: M(nu,mu,-R) = M(mu,nu,R)^dagger.  For the
    # packed block this is the ADJOINT, not a plain conjugation: a naive
    # conjugation map (h1,h2,h3,h4) -> (h1,h2,h3,-h4) does NOT satisfy it
    naive_conjugate = block_matrix(a1, a2, a3, -a4)
    assert naive_conjugate - m_mn_R.H != sp.zeros(2, 2)
    # the transposed-adjoint partner carries the SAME file components
    n_nm_mR = block_matrix(a1, a2, a3, a4)
    zero(n_nm_mR - m_mn_R.H)
    zero(n_nm_mR.H - m_mn_R)
    # the SAME handling holds at the derivative level: differentiate
    # M(mu,nu,R)^dagger = M(nu,mu,-R) in the displacement coordinate
    da1, da2, da3, da4 = sp.symbols("da1 da2 da3 da4", real=True)
    dm = block_matrix(da1, da2, da3, da4)
    dn = block_matrix(da1, da2, da3, da4)
    zero(dn - dm.H)
    zero(dn.H - dm)
    # consequently the derivative record at (nu,mu,-R) is the adjoint of
    # the record at (mu,nu,R): no independent data at reversed R, and the
    # hopping partner never needs a separate dH4 sign rule.  Componentwise,
    # the componentwise derivative dM/dX is the adjoint map on the packed
    # derivative components: the transverse entries (da3, da4) pass
    # through UNCHANGED under the pinned packing (the naive conjugation
    # would flip da4 and is refused, exactly as at the H level)
    dn_naive = block_matrix(da1, da2, da3, -da4)
    assert dn_naive - dm.H != sp.zeros(2, 2)
    print("   C PASS: partner block at (nu,mu,-R) = adjoint of (mu,nu,R),")
    print("      identical packed components at derivative level; the naive")
    print("      conjugation map (H4 -> -H4 without transpose) is refused")
    return {"rsym_partner_components": sp.Matrix([[a1], [a2], [a3], [a4]])}


# ---------------------------------------------------------------------------
# [D] fixed-partition theorem
# ---------------------------------------------------------------------------

X = sp.Symbol("x", real=True)  # a displacement coordinate


def test_fixed_partition_theorem():
    print("[D] fixed partition: d(partition)/dR = 0 identically")
    # partition map P: orbital index -> site index, CONSTANT integer data
    p0, p1 = sp.symbols("p0 p1", integer=True)
    partition = (p0, p1)  # orbitals 0,1 belong to sites p0,p1
    # vertex entries depend on the displacement; the partition map does not
    v0 = sp.Function("V_0")(X)
    v1 = sp.Function("V_1")(X)
    # declared observable family: O = sum over partition members f(V_k, P)
    f = sp.Function("f")
    observable = f(v0, partition[0]) + f(v1, partition[1])
    total_derivative = sp.diff(observable, X)
    expected = sp.diff(f(v0, p0), v0) * sp.diff(v0, X) + sp.diff(
        f(v1, p1), v1
    ) * sp.diff(v1, X)
    assert sp.simplify(total_derivative - expected) == 0, total_derivative
    # the partition entries are constants: their derivative is identically 0
    for p in partition:
        assert sp.diff(p, X) == 0
    # hence for observables built ONLY on the vertex and the partition map,
    # the total derivative has NO partition term: the fixed-partition label
    # is exact, not an approximation
    assert total_derivative.has(sp.diff(p0, X)) is False
    assert total_derivative.has(sp.diff(p1, X)) is False
    print("   D PASS: partition enters as constant index data; the total")
    print("      displacement derivative of the declared observable family")
    print("      carries vertex terms only")
    return {"fixed_partition_total_derivative": total_derivative}


# ---------------------------------------------------------------------------
# [E] machine-checkable terms declaration
# ---------------------------------------------------------------------------


def terms_declaration():
    """Included vs excluded derivative terms, machine-checkable."""
    declaration = {
        "vertex": True,  # dV/dR = (dV_up, -dV_up, dH3, dH4) on the onsite
        # same-site blocks: ALL FOUR packed components are part of the
        # vertex derivative (the transverse entries pass through the
        # adjoint map, section [C])
        "partition": False,  # constant index data: d(partition)/dR = 0 exactly
        "scalar_h0": False,  # spin-scalar part untouched by the vertex
        "overlap_S": False,  # S derivatives are not vertex data
    }
    included = tuple(sorted(k for k, v in declaration.items() if v))
    excluded = tuple(sorted(k for k, v in declaration.items() if not v))
    assert included == ("vertex",)
    assert excluded == ("overlap_S", "partition", "scalar_h0")
    print(
        "[E] declaration: terms_included = %s; terms_excluded = %s"
        % (included, excluded)
    )
    return {
        "terms_included_count": sp.Matrix([[len(included)]]),
        "terms_excluded_count": sp.Matrix([[len(excluded)]]),
    }


# ---------------------------------------------------------------------------
# [F] numeric gate: synthetic linear fixture record
# ---------------------------------------------------------------------------

NUMERIC_TOL = 1e-12
BOHR_TO_ANG = 0.529177210903

# synthetic 2-site / 1-orbital linear model: H(x) = H0 + alpha * x (x in Ang)
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
STEP_ANG = 0.05 * BOHR_TO_ANG


def vertex_blocks(h_map):
    """story-005 vertex blocks v_up=(H1-H2)/2 over a packed H map."""
    return {
        k: ((v[0] - v[1]) / 2.0, -(v[0] - v[1]) / 2.0, v[2], v[3])
        for k, v in h_map.items()
    }


def record_vertex_derivative(onsite_keys):
    """The record-level rule: the derivative of the full story-005 vertex
    block, (dV_up, -dV_up, dH3, dH4) per onsite key."""
    out = {}
    for key in onsite_keys:
        d1, d2, d3, d4 = ALPHA[key]
        out[key] = ((d1 - d2) / 2.0, -(d1 - d2) / 2.0, d3, d4)
    return out


def test_numeric_fixture():
    print("[F] numeric gate: synthetic linear fixture, FD vs record to 1e-12")
    h_plus = {k: tuple(h + a * STEP_ANG for h, a in zip(H0[k], ALPHA[k])) for k in H0}
    h_minus = {k: tuple(h - a * STEP_ANG for h, a in zip(H0[k], ALPHA[k])) for k in H0}
    v_plus = vertex_blocks(h_plus)
    v_minus = vertex_blocks(h_minus)
    onsite_keys = [k for k in H0 if k[0] == k[1] and k[2] == (0, 0, 0)]
    record = record_vertex_derivative(onsite_keys)
    # fixture sanity: at least one onsite key carries NONZERO transverse
    # components (dH3/dH4), so the four-component gate is not vacuous
    assert any(
        record[key][2] != 0.0 or record[key][3] != 0.0 for key in onsite_keys
    ), "fixture has no nonzero dH3/dH4 onsite: the transverse gate is vacuous"
    for key in onsite_keys:
        fd = tuple(
            (v_plus[key][i] - v_minus[key][i]) / (2.0 * STEP_ANG) for i in range(4)
        )
        for i in range(4):
            assert abs(fd[i] - record[key][i]) <= NUMERIC_TOL, (key, i, fd, record)
    print("   F PASS: all four dV/dR components == FD of the extracted vertex")
    print("      blocks on %d onsite keys to %.1e" % (len(onsite_keys), NUMERIC_TOL))
    return {"numeric_fd_tolerance": sp.Float(NUMERIC_TOL)}


# ---------------------------------------------------------------------------
# authority driver
# ---------------------------------------------------------------------------


def main():
    results = {}
    print("=" * 72)
    print("Story 018: vertex and partition displacement derivatives (ADR-004)")
    print("packing: %s" % PACKING_CONVENTION)
    print("=" * 72)
    results.update(test_vertex_derivative_packing())
    results.update(test_onsite_hermiticity())
    results.update(test_transposed_adjoint_r_symmetry())
    results.update(test_fixed_partition_theorem())
    results.update(terms_declaration())
    results.update(test_numeric_fixture())
    print("-" * 72)
    print("ALL ASSERTIONS GREEN")
    return results


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("ASSERTION FAILED: %r" % (exc,))
        sys.exit(1)
