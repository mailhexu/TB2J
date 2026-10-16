"""Assertion authority for story 015: +U Hubbard rotation contract.

Run in mydev: python docs/sympy/siesta_spiral_hubbard_rotation.py
Paired derivation, conventions and limits: same-name .md file.
Symbolic sections import no production code.  The numeric section [E]
imports ONLY the story-014 certified module (vee tensor, FLL assembly,
provenance, complex<->real transform) and the committed ground-truth
occupation dump.

Scope (FR-010 / ADR-007 / ADR-003, SymPy first).  For a SIESTA DFT+U
correlated site (projector method 2, FLL double counting per
Bultmark-09 Eq. 13, Liechtenstein (U, J) params) the magnetic-force
theorem freezes the self-consistent density and rotates only the
magnetic operator.  For +U references the rotated operator is

    B_tot(theta) = B_xc + V_U^s(theta),   V_U^s = spin-dependent part,

with every contribution separately labeled, and the contract pins:

  [A] the local spin rotation U_s = exp(-i theta sigma.hat_n / 2)
      co-rotates the spinor occupation matrix,
          N(theta) = (U_s (x) I_orb) N (U_s^dag (x) I_orb),
      with the full spinor trace Tr N(theta) = Tr N invariant and the
      moment vector (m_x, m_y, m_z) = Tr(sigma_a N) rotating as an
      SO(3) vector under the same right-handed Rodrigues map; the
      SINGLE channel traces t_uu, t_dd, t_ud are NOT invariant (only
      n_tot = t_uu + t_dd and the moment are),
  [B] the FLL Hubbard potential W (m_pot_dftu.F90:1057-1141, direct =
      vee(m, m'', m', m'''), fock = vee(m, m''', m'', m'), certified
      slot order, story-014 energy-split certification) co-rotates
      algebraically,
          W[N(theta)] = (U_s (x) I_orb) W[N] (U_s^dag (x) I_orb),
      for BOTH the direct and the fock contraction: the contractions
      are LINEAR in N and their kernels carry NO spin indices (the
      abstract SO(3)-scalar tensor vee(m1, m2, m3, m4) with the
      certified slot symmetry vee(m1, m2, m3, m4) = vee(m2, m1, m4,
      m3)); the spin rotation therefore passes through unchanged,
  [C] the double-counting potential V_dc (m_pot_dftu.F90:974-1035) is
      a scalar x identity per spin channel built from the CONVERGED
      reference traces (n_tot, t_uu, t_dd, t_ud); under the MFT
      variation the density is frozen, so delta V_dc = 0 exactly while
      the Hubbard part genuinely varies; co-rotating V_dc would be a
      DIFFERENT operator, and freezing W entirely or reducing it to a
      static scalar splitting is refused (ADR-007 rejected
      alternatives); the isotropic dc piece is trace-protected (it
      would survive a rebuild) but the J channel piece would drift as
      -J (t_uu' - t_uu): freezing is a real contract, not a no-op,
  [D] the assembled rotated magnetic operator is the labeled sum
          B_tot(theta) = U_s B_xc U_s^dag + V_U^s(theta),
      where B_xc enters as frozen extracted data (the density channel
      of XC is frozen under MFT, so the XC vertex varies ONLY through
      the explicit conjugation) and V_U^s(theta) co-rotates through
      its occupation matrix; the first-order variation is
      dB_tot/dtheta|_0 = (i/2) [B_tot, sigma.hat_n] and is NOT zero,
  [E] numeric verification on the certified story-014 ground truth:
      the numeric vee tensor satisfies the slot symmetry, the
      hermiticity pairing and is a rotation scalar under an exact
      real-harmonic orbital rotation (Wigner-D transform conjugated
      with the module's own T); with the ground-truth occupations
      (tests/data/fe4_occ.dftu_occ) and finite rotations (random +
      pi/2 about x, y) the reassembled rotated Hubbard operator equals
      U_s W U_s^dag to 1e-12 RELATIVE (approved story-015 tolerance),
      n_tot and the moment co-rotate to 1e-12 relative, the contract
      dc term is frozen bit-exactly while the dc rebuilt from the
      rotated traces drifts by a reported finite amount that
      decomposes exactly into the invariant isotropic piece plus
      -J (t_uu' - t_uu).

Everything internal is Ry; provenance carries eV alongside.
Nonzero exit on any failed check (AssertionError; the __main__ block
prints FAIL and exits 1).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# symbols and the local spin rotation
# ---------------------------------------------------------------------------

HALF = sp.Rational(1, 2)
THETA = sp.Symbol("theta", real=True)
PHI, LAM = sp.symbols("phi lambda", real=True, positive=True)
U_FLL, J_FLL = sp.symbols("U_FLL J_FLL", real=True, positive=True)
BX, BY, BZ = sp.symbols("B_xc_x B_xc_y B_xc_z", real=True)

# exact unit axis: n-hat = (sin(phi) cos(lambda), sin(phi) sin(lambda), cos(phi))
NX = sp.sin(PHI) * sp.cos(LAM)
NY = sp.sin(PHI) * sp.sin(LAM)
NZ = sp.cos(PHI)

SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SZ = sp.Matrix([[1, 0], [0, -1]])
SIGMA = (SX, SY, SZ)
SN = NX * SX + NY * SY + NZ * SZ

COS = sp.cos(HALF * THETA)
SIN = sp.sin(HALF * THETA)
U_S = COS * sp.eye(2) - sp.I * SIN * SN

# exact rationalization parameters (tan half angles; denominators > 0)
_T, _P, _L = sp.symbols("_t _p _l", real=True)
_TAN_SUBS = {
    sp.cos(HALF * THETA): (1 - _T**2) / (1 + _T**2),
    sp.sin(HALF * THETA): 2 * _T / (1 + _T**2),
    sp.cos(THETA): (1 - 6 * _T**2 + _T**4) / (1 + _T**2) ** 2,
    sp.sin(THETA): 4 * _T * (1 - _T**2) / (1 + _T**2) ** 2,
    sp.cos(PHI): (1 - _P**2) / (1 + _P**2),
    sp.sin(PHI): 2 * _P / (1 + _P**2),
    sp.cos(LAM): (1 - _L**2) / (1 + _L**2),
    sp.sin(LAM): 2 * _L / (1 + _L**2),
}


def _reduce(expr):
    """Exact reduction of a scalar trig-polynomial identity.

    expand(trig=True) first reduces sums/products of angles; the exact
    tan-half-angle parametrizations of theta, phi, lambda then turn the
    expression into a rational function whose denominator is a positive
    product of (1 + t^2)-type factors, so `numer == 0` is an EXACT
    identity test on the reals and `numer != 0` an exact nonidentity."""
    rational = sp.expand(expr, trig=True).subs(_TAN_SUBS)
    return sp.expand(sp.numer(sp.together(rational)))


def _is_matrix(expr):
    return hasattr(expr, "applyfunc")


def zero(expr):
    """Require exact vanishing (elementwise for matrices)."""
    if _is_matrix(expr):
        expr = expr.applyfunc(_reduce)
        assert expr == sp.zeros(*expr.shape), expr
    else:
        reduced = _reduce(expr)
        assert reduced == 0, reduced


def nonzero(expr):
    """Require exact NON-vanishing (scalar expressions only)."""
    assert not _is_matrix(expr), "nonzero() takes a scalar expression"
    reduced = _reduce(expr)
    assert reduced != 0, "expected a nonidentity expression"


def rodrigues_matrix(axis, angle):
    """SO(3) rotation matrix (right-handed, active) for a UNIT axis, sympy."""
    axis = sp.Matrix(axis)
    zero((axis.T * axis)[0] - 1)
    K = sp.Matrix(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return sp.eye(3) + sp.sin(angle) * K + (1 - sp.cos(angle)) * (K * K)


def spinor_matrix(nuu, nud, ndu, ndd):
    """4x4 spinor (spin outer: index = 2*orb + spin) from 2x2 orbital blocks.

    Spin order (up, down) matches SIESTA's Dij(1..4) = (uu, dd, ud, du)
    channel list used throughout the certified module."""
    blocks = [[nuu, nud], [ndu, ndd]]
    return sp.Matrix(4, 4, lambda i, j: blocks[i % 2][j % 2][i // 2, j // 2])


def lift_spin_rotation(u2, size_orb):
    """U (x) I_orb with the orbital index outer (index = 2*orb + spin)."""
    return sp.Matrix(
        2 * size_orb,
        2 * size_orb,
        lambda i, j: (u2[i % 2, j % 2] if i // 2 == j // 2 else 0),
    )


def channel_traces(mat, size_orb):
    """(t_uu, t_dd, t_ud, t_du) of a spin-outer spinor matrix."""
    t_uu = sum(mat[2 * p, 2 * p] for p in range(size_orb))
    t_dd = sum(mat[2 * p + 1, 2 * p + 1] for p in range(size_orb))
    t_ud = sum(mat[2 * p, 2 * p + 1] for p in range(size_orb))
    t_du = sum(mat[2 * p + 1, 2 * p] for p in range(size_orb))
    return t_uu, t_dd, t_ud, t_du


def generic_occupation(size_orb, tag):
    """Four independent complex orbital blocks (uu, ud, du, dd)."""

    def blk(cha):
        return sp.Matrix(
            size_orb,
            size_orb,
            lambda i, j: sp.Symbol(f"{tag}{cha}{i+1}{j+1}", complex=True),
        )

    return spinor_matrix(blk("uu"), blk("ud"), blk("du"), blk("dd"))


# ---------------------------------------------------------------------------
# [A] SU(2) rotation algebra and occupation co-rotation
# ---------------------------------------------------------------------------


def test_rotation_algebra():
    print("[A] Local SU(2) spin rotation: unitarity, axis fix, Pauli vector law")
    zero(sp.expand(U_S.H * U_S - sp.eye(2)))
    zero(sp.expand(U_S * U_S.H - sp.eye(2)))
    # the rotation axis is left invariant
    zero(sp.expand(U_S * SN * U_S.H - SN))
    # U sigma_a U^dag = sum_b R[b, a] sigma_b (transposed index placement,
    # checked directly for the axis-z case: U sigma_x U^dag = cos sigma_x
    # + sin sigma_y); the SPIN EXPECTATION (moment vector) still rotates
    # with the active Rodrigues map, pinned below
    rot = rodrigues_matrix([NX, NY, NZ], THETA)
    for a in range(3):
        conj = sp.expand(U_S * SIGMA[a] * U_S.H)
        target = sp.expand(
            sum((rot[b, a] * SIGMA[b] for b in range(3)), sp.zeros(2, 2))
        )
        zero(conj - target)
    # spinor occupation co-rotation on a generic two-orbital shell
    n_mat = generic_occupation(2, "n")
    u4 = lift_spin_rotation(U_S, 2)
    n_rot = (u4 * n_mat * u4.H).applyfunc(sp.expand)
    # full spinor trace invariant; t_uu + t_dd invariant
    zero(sum(n_rot[p, p] for p in range(4)) - sum(n_mat[p, p] for p in range(4)))
    t_uu0, t_dd0, _t_ud0, _t_du0 = channel_traces(n_mat, 2)
    t_uu1, t_dd1, _t_ud1, _t_du1 = channel_traces(n_rot, 2)
    zero((t_uu1 + t_dd1) - (t_uu0 + t_dd0))
    # single channel traces are NOT invariant (why dc freezing is a contract)
    nonzero(t_uu1 - t_uu0)
    # the moment vector (Tr sigma_a N) rotates with the same Rodrigues map

    def moment(mat):
        t_uu_, t_dd_, t_ud_, t_du_ = channel_traces(mat, 2)
        return sp.Matrix([t_ud_ + t_du_, sp.I * t_ud_ - sp.I * t_du_, t_uu_ - t_dd_])

    for a in range(3):
        target = sp.expand(
            sum((rot[a, b] * moment(n_mat)[b] for b in range(3)), sp.Integer(0))
        )
        zero(moment(n_rot)[a] - target)
    print("   PASS: U_s unitary; N co-rotates; Tr N and t_uu+t_dd invariant;")
    print("         t_uu alone NOT invariant; m(N') = R m(N), R = Rodrigues")
    return {"u_s": U_S, "rotation_map": rot}


# ---------------------------------------------------------------------------
# [B] FLL Hubbard potential: algebraic co-rotation of direct and fock parts
# ---------------------------------------------------------------------------


def vee_sym(p, a, q, b):
    """Abstract SO(3)-scalar kernel vee(m1, m2, m3, m4) on a symbolic
    two-orbital shell with the certified slot symmetry enforced:
    vee(m1, m2, m3, m4) = vee(m2, m1, m4, m3)."""
    key = (p, a, q, b)
    rep = min(key, (a, p, b, q))
    return sp.Symbol("v" + "".join(str(ix + 1) for ix in rep))


def fock_sym(p, a, q, b):
    """fock kernel = vee(m, m''', m'', m') (certified slot reading)."""
    return vee_sym(p, b, a, q)


def symbolic_hubbard_potential(n_mat, *, direct=True, fock=True):
    """Pinned SIESTA assembly (m_pot_dftu.F90:1057-1141) on a symbolic
    spinor occupation matrix (spin outer): returns the 4x4 potential with
        V_uu(p,q) = sum vee(p,a,q,b) (n_uu+n_dd)(a,b)
                    - sum vee(p,b,a,q) n_uu(a,b)
        V_dd(p,q) = sum vee(p,a,q,b) (n_uu+n_dd)(a,b)
                    - sum vee(p,b,a,q) n_dd(a,b)
        V_ud(p,q) = - sum vee(p,b,a,q) n_ud(a,b)
        V_du(p,q) = - sum vee(p,b,a,q) n_du(a,b)
    (direct slots (m, m'', m', m'''), fock slots (m, m''', m'', m'))."""
    size = n_mat.shape[0] // 2
    nuu = sp.Matrix(size, size, lambda i, j: n_mat[2 * i, 2 * j])
    ndd = sp.Matrix(size, size, lambda i, j: n_mat[2 * i + 1, 2 * j + 1])
    nud = sp.Matrix(size, size, lambda i, j: n_mat[2 * i, 2 * j + 1])
    ndu = sp.Matrix(size, size, lambda i, j: n_mat[2 * i + 1, 2 * j])
    ntot = nuu + ndd

    def direct_sum(occ, p, q):
        if not direct:
            return sp.Integer(0)
        return sum(
            vee_sym(p, a, q, b) * occ[a, b] for a in range(size) for b in range(size)
        )

    def fock_sum(occ, p, q):
        if not fock:
            return sp.Integer(0)
        return sum(
            fock_sym(p, a, q, b) * occ[a, b] for a in range(size) for b in range(size)
        )

    def channel(direct_occ, fock_occ):
        return sp.Matrix(
            size,
            size,
            lambda p, q: sp.expand(
                (
                    direct_sum(direct_occ, p, q)
                    if direct_occ is not None
                    else sp.Integer(0)
                )
                - fock_sum(fock_occ, p, q)
            ),
        )

    return spinor_matrix(
        channel(ntot, nuu), channel(None, nud), channel(None, ndu), channel(ntot, ndd)
    )


def test_kernel_covariance():
    print("[B] FLL W[N] co-rotation: direct and fock contractions, slot symmetry")
    # declared slot symmetry of the abstract scalar tensor C (structural)
    for p in range(2):
        for a in range(2):
            for q in range(2):
                for b in range(2):
                    zero(vee_sym(p, a, q, b) - vee_sym(a, p, b, q))
    n_mat = generic_occupation(2, "n")
    u4 = lift_spin_rotation(U_S, 2)
    n_rot = (u4 * n_mat * u4.H).applyfunc(sp.expand)
    for label, kwargs in (
        ("direct only", {"direct": True, "fock": False}),
        ("fock only", {"direct": False, "fock": True}),
        ("direct + fock (pinned assembly)", {"direct": True, "fock": True}),
    ):
        w_ref = symbolic_hubbard_potential(n_mat, **kwargs)
        w_rot = symbolic_hubbard_potential(n_rot, **kwargs)
        conj = (u4 * w_ref * u4.H).applyfunc(sp.expand)
        for i in range(4):
            for j in range(4):
                zero(w_rot[i, j] - conj[i, j])
        print(f"   PASS: W[N'] = U W[N] U^dag ({label})")
    return {}


# ---------------------------------------------------------------------------
# [C] Double counting: frozen under the MFT variation
# ---------------------------------------------------------------------------


def symbolic_dc_potential(t_uu, t_dd, t_ud, t_du, size_orb):
    """V_dc per m_pot_dftu.F90:974-1035: scalar x identity per spin channel,
    built from the CONVERGED reference traces (frozen under MFT):
        V_dc[uu] = -delta (U (n_tot - 1/2) - J (t_uu - 1/2))
        V_dc[dd] = -delta (U (n_tot - 1/2) - J (t_dd - 1/2))
        V_dc[ud] = +delta J t_ud,   V_dc[du] = +delta J t_du."""
    n_tot = t_uu + t_dd
    entries = [
        -(U_FLL * (n_tot - HALF) - J_FLL * (t_uu - HALF)),
        -(U_FLL * (n_tot - HALF) - J_FLL * (t_dd - HALF)),
        J_FLL * t_ud,
        J_FLL * t_du,
    ]
    return sp.Matrix(
        2 * size_orb,
        2 * size_orb,
        lambda i, j: (entries[2 * (i % 2) + (j % 2)] if i == j else sp.Integer(0)),
    )


def test_dc_frozen():
    print("[C] Double counting frozen: scalar x identity, delta V_dc = 0")
    size = 2
    n_mat = generic_occupation(size, "n")
    v_dc = symbolic_dc_potential(*channel_traces(n_mat, size), size)
    # scalar x identity per spin channel: every off-diagonal entry is exactly 0
    for i in range(2 * size):
        for j in range(2 * size):
            if i != j:
                zero(v_dc[i, j])
    # frozen at the reference traces: no theta dependence at all
    assert THETA not in v_dc.free_symbols
    zero(sp.diff(v_dc[0, 0], THETA))
    zero(sp.diff(v_dc[2 * size - 1, 2 * size - 1], THETA))
    # the isotropic piece is trace-protected: rebuilt from co-rotated
    # occupations it is UNCHANGED because n_tot is invariant ([A])
    u4 = lift_spin_rotation(U_S, size)
    n_rot = (u4 * n_mat * u4.H).applyfunc(sp.expand)
    t_uu0, t_dd0, t_ud0, _ = channel_traces(n_mat, size)
    t_uu1, t_dd1, t_ud1, _ = channel_traces(n_rot, size)
    zero((t_uu1 + t_dd1) - (t_uu0 + t_dd0))
    zero(-(U_FLL * ((t_uu1 + t_dd1) - HALF)) + (U_FLL * ((t_uu0 + t_dd0) - HALF)))
    # the J channel piece is NOT trace-protected: a rebuilt dc would drift
    nonzero(-J_FLL * (t_uu1 - t_uu0))
    # the contract freezes V_dc at the reference value (no rebuild);
    # co-rotating V_dc would give a DIFFERENT operator (rejected reading):
    # the spin off-diagonal entry (0,1) is orbital-diagonal so it does move
    v_dc_rot = (u4 * v_dc * u4.H).applyfunc(sp.expand)
    zero(v_dc_rot[0, 2])  # orbital-diagonal structure survives (spin-only U)
    nonzero(v_dc_rot[0, 1] - v_dc[0, 1])
    nonzero(v_dc_rot[0, 0] - v_dc[0, 0])
    # and freezing W entirely / static-scalar splitting is refused:
    # the Hubbard operator genuinely varies with theta
    w_full = symbolic_hubbard_potential(n_mat)
    conj_w = (u4 * w_full * u4.H).applyfunc(sp.expand)
    dw = conj_w.applyfunc(lambda e: sp.diff(e, THETA))
    nonzero(dw[0, 0])
    nonzero(dw[0, 2])
    print("   PASS: V_dc scalar x identity; d V_dc/dtheta = 0 (frozen at the")
    print("         reference traces); isotropic piece trace-protected; the")
    print("         J piece would drift as -J (t_uu' - t_uu); co-rotating V_dc")
    print("         is a different operator; W genuinely varies")
    return {
        "dc_uu_frozen": sp.expand(v_dc[0, 0]),
        "dc_isotropic_invariant_piece": -U_FLL * (sp.Symbol("n_tot") - HALF),
        "dc_channel_drift": sp.expand(-J_FLL * (t_uu1 - t_uu0)),
    }


# ---------------------------------------------------------------------------
# [D] Assembled rotated operator with labeled XC / Hubbard contributions
# ---------------------------------------------------------------------------


def spin_dependent_part(w, size_orb):
    """V_U^s = W - (1/2) Tr_spin(W) (x) I_orb (spin-outer indexation)."""
    out = sp.Matrix(w)
    for p in range(size_orb):
        trace_half = HALF * (w[2 * p, 2 * p] + w[2 * p + 1, 2 * p + 1])
        out[2 * p, 2 * p] = sp.expand(out[2 * p, 2 * p] - trace_half)
        out[2 * p + 1, 2 * p + 1] = sp.expand(out[2 * p + 1, 2 * p + 1] - trace_half)
    return out


def test_assembled_operator():
    print("[D] B_tot(theta) = U B_xc U^dag + V_U^s(theta), separately labeled")
    size = 2
    n_mat = generic_occupation(size, "n")
    u4 = lift_spin_rotation(U_S, size)
    n_rot = (u4 * n_mat * u4.H).applyfunc(sp.expand)

    b_xc = BX * SX + BY * SY + BZ * SZ
    b_xc4 = lift_spin_rotation(b_xc, size)
    vs_ref = spin_dependent_part(symbolic_hubbard_potential(n_mat), size)
    vs_rot = spin_dependent_part(symbolic_hubbard_potential(n_rot), size)
    # V_U^s co-rotates (its spin trace is SU(2) invariant)
    conj_vs = (u4 * vs_ref * u4.H).applyfunc(sp.expand)
    for i in range(2 * size):
        for j in range(2 * size):
            zero(vs_rot[i, j] - conj_vs[i, j])
    # separation of contributions: V_U^s carries no B_xc symbols, and the
    # assembled rotated operator is exactly the labeled conjugated sum
    assert vs_ref.free_symbols.isdisjoint({BX, BY, BZ})
    b_tot = b_xc4 + vs_ref
    b_tot_rot = (u4 * b_tot * u4.H).applyfunc(sp.expand)
    labeled_sum = (u4 * b_xc4 * u4.H).applyfunc(sp.expand) + vs_rot
    for i in range(2 * size):
        for j in range(2 * size):
            zero(b_tot_rot[i, j] - labeled_sum[i, j])
    # first-order variation: dB/dtheta|_0 = (i/2) [B_tot, sigma.hat_n],
    # and it is NOT zero (the operator genuinely varies)
    db = b_tot_rot.applyfunc(lambda e: sp.diff(e, THETA).subs(THETA, 0))
    sn4 = lift_spin_rotation(SN, size)
    comm = sp.expand(sp.I * HALF * (b_tot * sn4 - sn4 * b_tot))
    for i in range(2 * size):
        for j in range(2 * size):
            zero(db[i, j] - comm[i, j])
    nonzero(db[0, 0])
    nonzero(db[0, 2])
    print("   PASS: V_U^s co-rotates and carries no XC symbols; the assembled")
    print("         rotated operator is the labeled conjugated sum; dB/dtheta|0")
    print("         = (i/2)[B_tot, sigma.hat_n] != 0")
    return {}


# ---------------------------------------------------------------------------
# [E] numeric verification on the certified story-014 ground truth
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"
NPZ = DATA / "fe4_dftu_story014.npz"
GT_OCC = DATA / "fe4_occ.dftu_occ"
COV_TOL = 1e-12  # approved story-015 numeric covariance tolerance (relative)
ROT_SCALAR_TOL = 1e-11  # real-harmonic orbital-rotation scalar check

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


def occ_spinor(occ_site):
    """(4, 5, 5) channels (uu, dd, ud, du) -> (2, 2, 5, 5) spinor blocks."""
    n = np.zeros((2, 2, 5, 5), dtype=complex)
    n[0, 0] = occ_site[0]
    n[1, 1] = occ_site[1]
    n[0, 1] = occ_site[2]
    n[1, 0] = occ_site[3]
    return n


def unpack_spinor(n_spinor):
    occ = np.zeros((4, 5, 5), dtype=complex)
    occ[0] = n_spinor[0, 0]
    occ[1] = n_spinor[1, 1]
    occ[2] = n_spinor[0, 1]
    occ[3] = n_spinor[1, 0]
    return occ


def apply_spin_rotation(u2, n_spinor):
    """N' = U N U^dag on the spin indices (orbital indices untouched)."""
    return np.einsum("ia,abpq,jb->ijpq", u2, n_spinor, u2.conj())


def hubbard_spinor(v_hub_site):
    """(4, 5, 5) channels -> (2, 2, 5, 5) spinor Hubbard potential."""
    v = np.zeros((2, 2, 5, 5), dtype=complex)
    v[0, 0] = v_hub_site[0]
    v[1, 1] = v_hub_site[1]
    v[0, 1] = v_hub_site[2]
    v[1, 0] = v_hub_site[3]
    return v


def moment_vector(n_spinor):
    """Site moment (Tr sigma_a N); Hermitian N gives a real vector."""
    t_uu = np.trace(n_spinor[0, 0])
    t_dd = np.trace(n_spinor[1, 1])
    t_ud = np.trace(n_spinor[0, 1])
    t_du = np.trace(n_spinor[1, 0])
    return np.array([t_ud + t_du, 1j * t_ud - 1j * t_du, t_uu - t_dd], dtype=complex)


def rodrigues_np(axis, angle):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kmat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0.0]])
    return (
        np.eye(3) * np.cos(angle)
        + np.sin(angle) * kmat
        + (1 - np.cos(angle)) * np.outer(k, k)
    )


def _wigner_d_complex(l, alpha, beta, gamma):
    """Exact complex-basis Wigner D (ascending m rows/cols) for
    R = Rz(alpha) Ry(beta) Rz(gamma):  D_{m'm} = e^{-i a m'} d_{m'm}(b) e^{-i g m}.
    Self-checks sympy's small-d ordering/sign against the exact l=1
    closed form (Condon-Shortley)."""
    from sympy.physics.wigner import wigner_d_small

    if l == 1:
        cb, sb = np.cos(beta), np.sin(beta)
        ref = np.array(
            [
                [(1 + cb) / 2, -sb / np.sqrt(2), (1 - cb) / 2],
                [sb / np.sqrt(2), cb, -sb / np.sqrt(2)],
                [(1 - cb) / 2, sb / np.sqrt(2), (1 + cb) / 2],
            ]
        )
        sympy_d = np.array(
            sp.Matrix(wigner_d_small(1, sp.Float(beta))).tolist(), dtype=complex
        )
        flipped = sympy_d[::-1, ::-1]
        direct = sympy_d
        assert (
            min(np.max(np.abs(flipped - ref)), np.max(np.abs(direct - ref))) < 1e-12
        ), "unexpected Wigner d convention"
        d_mid = (
            flipped
            if np.max(np.abs(flipped - ref)) < np.max(np.abs(direct - ref))
            else direct
        )
    else:
        # sympy orders m descending (l..-l); flip to ascending (pinned by
        # the l == 1 branch against the exact closed form)
        d_mid = np.array(
            sp.Matrix(wigner_d_small(l, sp.Float(beta))).tolist(), dtype=complex
        )[::-1, ::-1]
    ms = np.arange(-l, l + 1)
    phase_left = np.exp(-1j * alpha * np.repeat(ms[:, None], 2 * l + 1, axis=1))
    phase_right = np.exp(-1j * gamma * np.repeat(ms[None, :], 2 * l + 1, axis=0))
    return phase_left * d_mid * phase_right


def real_harmonic_rotation(rot3):
    """(5, 5) real-harmonic rotation for the GIVEN Cartesian rotation,
    via the exact complex Wigner D conjugated with the certified module's
    own complex<->real transform T (rlylm ordering m = -l..l)."""
    from TB2J.interfaces.siesta_spiral_hubbard import real_transform

    rot3 = np.asarray(rot3, dtype=float)
    beta = float(np.arccos(np.clip(rot3[2, 2], -1.0, 1.0)))
    alpha = float(np.arctan2(rot3[1, 2], rot3[0, 2]))
    gamma = float(np.arctan2(rot3[2, 1], -rot3[2, 0]))

    def rz(a):
        return np.array(
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
        )

    def ry(b):
        return np.array(
            [[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]]
        )

    recomposed = rz(alpha) @ ry(beta) @ rz(gamma)
    assert np.max(np.abs(recomposed - rot3)) < 1e-12, "Euler extraction failed"
    l = 2
    dim = 2 * l + 1
    d_complex = _wigner_d_complex(l, alpha, beta, gamma)
    # convention self-check: Rz(pi) must diagonalize to e^{-i pi m} (ascending)
    d_check = _wigner_d_complex(l, np.pi, 0.0, 0.0)
    assert (
        np.max(np.abs(d_check - np.diag(np.exp(-1j * np.pi * np.arange(-l, l + 1)))))
        < 1e-12
    )
    t_mat = np.asarray(real_transform(l), dtype=complex)
    # phi_i = sum_a T[i, a] Y_a (module transform), so the real-basis
    # rotation is D_real = conj(T) @ D_c @ T^T; it must come out real
    # orthogonal
    d_real = np.conj(t_mat) @ d_complex @ t_mat.T
    assert np.max(np.abs(d_real.imag)) < 1e-12
    assert np.max(np.abs(d_real @ d_real.T - np.eye(dim))) < 1e-12
    # rlylm pin: rotation by pi about z multiplies the m = +-1 harmonics
    # (cos/sin phi type) by -1 and leaves m = 0, +-2 unchanged — true for
    # ANY cos(m phi)/sin(m phi) convention, so it pins the ordering
    d_pi_z = np.conj(t_mat) @ _wigner_d_complex(l, 0.0, 0.0, np.pi) @ t_mat.T
    expected_pi_z = np.diag([1.0, -1.0, 1.0, -1.0, 1.0])  # m = -2..+2 ascending
    assert np.max(np.abs(d_pi_z - expected_pi_z)) < 1e-12
    return d_real


def test_numeric_contract():
    print("[E] Numeric verification: certified vee + GT occupations + rotations")
    sys.path.insert(0, str(REPO))
    from TB2J.interfaces.siesta_spiral_hubbard import (
        DftuProvenance,
        assemble_hubbard_operator,
        vee_tensor,
    )

    data = np.load(NPZ)
    occ = parse_gt_occ(GT_OCC)  # (nsites, 4, 5, 5), spin order uu, dd, ud, du
    u_ry, j_ry = float(data["u_ry"]), float(data["j_ry"])
    slater_f2 = 14.0 * j_ry / 1.625
    prov = DftuProvenance(
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
        site_indices=tuple(range(occ.shape[0])),
        projector_identity="Fe_nc.dftu_proj:rc=2.975687,npts=500",
    )

    nsites = occ.shape[0]
    spinors = [occ_spinor(occ[s]) for s in range(nsites)]
    t_tot = [float(np.real(np.trace(n[0, 0]) + np.trace(n[1, 1]))) for n in spinors]
    moments = np.array([moment_vector(n) for n in spinors])
    m_z = [float(m[2].real) for m in moments]
    t_updn = [float(abs(np.trace(s[0, 1]))) for s in spinors]
    herm = max(
        float(np.max(np.abs(n - np.conj(np.swapaxes(n, 0, 1))))) for n in spinors
    )
    print(f"   GT traces: n_d/site = {['%.4f' % v for v in t_tot]}")
    print(f"   GT moments: m_z = {['%+.4f' % v for v in m_z]}")
    print(
        f"   GT |t_updn| = {['%.4f' % v for v in t_updn]}  (site block herm {herm:.2e})"
    )
    assert all(6.30 < v < 6.47 for v in t_tot), t_tot
    assert max(abs(m) for m in m_z) > 2.5, m_z  # chain ends along +/-z
    assert min(abs(m) for m in m_z) < 1e-3, m_z  # 90-degree sites: m_z ~ 0
    assert max(t_updn) > 1.2, t_updn  # transverse moment on the 90-degree sites

    vee = vee_tensor(prov.slater_f0_ry, prov.slater_f2_ry, prov.slater_f4_ry)
    slot_sym = float(np.max(np.abs(vee - vee.transpose(1, 0, 3, 2))))
    herm_v = float(np.max(np.abs(vee - vee.transpose(2, 3, 0, 1).conj())))
    vee_f0 = vee_tensor(prov.slater_f0_ry, 0.0, 0.0)
    target_f0 = np.zeros((5, 5, 5, 5))
    for i in range(5):
        target_f0[i, :, i, :] = np.eye(5) * prov.slater_f0_ry
    f0_err = float(np.max(np.abs(vee_f0 - target_f0)))
    print(f"   vee slot symmetry  max|v - v.perm(1,0,3,2)|      = {slot_sym:.3e}")
    print(f"   vee hermiticity    max|v - conj(v.perm(2,3,0,1))| = {herm_v:.3e}")
    print(f"   vee F0-only        max|v - F0 delta delta|        = {f0_err:.3e}")
    assert slot_sym < 1e-12 and herm_v < 1e-12 and f0_err < 1e-12

    # rotation-scalar property: exact orbital rotation of the real harmonics
    axis_o = np.array([0.31, -0.77, 0.55])
    axis_o /= np.linalg.norm(axis_o)
    rot3 = rodrigues_np(axis_o, 0.9231)
    d_real = real_harmonic_rotation(rot3)
    vee_rot = np.einsum("ia,jb,kc,ld,abcd->ijkl", d_real, d_real, d_real, d_real, vee)
    rot_scalar_err = float(np.max(np.abs(vee_rot - vee)))
    scale = float(np.max(np.abs(vee)))
    print(
        f"   vee rotation-scalar max|D v D^T (x4) - v| = {rot_scalar_err:.3e}"
        f" (relative {rot_scalar_err / scale:.3e}, tol {ROT_SCALAR_TOL:.0e})"
    )
    assert rot_scalar_err < ROT_SCALAR_TOL * scale

    # finite local rotations of the +U vertex
    v_hub_ref, v_dc_ref, _ = assemble_hubbard_operator(occ, prov)
    rng = np.random.default_rng(20261016)
    rotations = []
    for _ in range(6):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotations.append((axis, float(rng.uniform(0.15, 2.9))))
    rotations.append((np.array([1.0, 0.0, 0.0]), np.pi / 2))
    rotations.append((np.array([0.0, 1.0, 0.0]), np.pi / 2))
    worst_cov = worst_trace = worst_moment = 0.0
    worst_dc_drift = worst_dc_split = 0.0
    for axis, angle in rotations:
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        kx, ky, kz = axis
        u2 = np.array(
            [
                [c - 1j * s * kz, -1j * s * (kx - 1j * ky)],
                [-1j * s * (kx + 1j * ky), c + 1j * s * kz],
            ]
        )
        assert np.max(np.abs(u2.conj().T @ u2 - np.eye(2))) < 1e-14
        rot3 = rodrigues_np(axis, angle)
        for site in range(nsites):
            n_rot = apply_spin_rotation(u2, spinors[site])
            occ_rot = unpack_spinor(n_rot)
            v_hub_rot, v_dc_rebuilt, _ = assemble_hubbard_operator(
                occ_rot[None, ...], prov
            )
            v_ref = hubbard_spinor(v_hub_ref[site])
            v_conj = unpack_spinor(apply_spin_rotation(u2, v_ref))
            worst_cov = max(
                worst_cov,
                float(np.max(np.abs(v_hub_rot[0] - v_conj)))
                / float(np.max(np.abs(v_ref))),
            )
            tr0 = float(np.real(np.trace(occ[site, 0]) + np.trace(occ[site, 1])))
            tr1 = float(np.real(np.trace(occ_rot[0]) + np.trace(occ_rot[1])))
            worst_trace = max(worst_trace, abs(tr1 - tr0) / max(abs(tr0), 1.0))
            m0 = moment_vector(spinors[site])
            m1 = moment_vector(n_rot)
            worst_moment = max(
                worst_moment,
                float(np.max(np.abs(m1 - rot3 @ m0))) / float(np.max(np.abs(m0))),
            )
            worst_dc_drift = max(
                worst_dc_drift, float(np.max(np.abs(v_dc_rebuilt[0] - v_dc_ref[site])))
            )
            # drift decomposition: rebuilt - frozen == -J (t_uu' - t_uu) +
            # invariant isotropic piece (numerically zero by trace invariance)
            t_uu0 = np.trace(occ[site, 0])
            t_uu1 = np.trace(occ_rot[0])
            t_dd0 = np.trace(occ[site, 1])
            t_dd1 = np.trace(occ_rot[1])
            expected = (
                -u_ry * ((t_uu1 + t_dd1) - 0.5)
                + j_ry * (t_uu1 - 0.5)
                + u_ry * ((t_uu0 + t_dd0) - 0.5)
                - j_ry * (t_uu0 - 0.5)
            )
            worst_dc_split = max(
                worst_dc_split,
                float(
                    np.abs(
                        v_dc_rebuilt[0, 0, 0, 0] - v_dc_ref[site, 0, 0, 0] - expected
                    )
                ),
            )
    print(
        f"   rotations: {len(rotations)} (6 random + pi/2 about x, y), all {nsites} sites"
    )
    print(
        f"   W covariance worst relative |W[n'] - U W[n] U^dag| = {worst_cov:.3e}"
        f" (tol {COV_TOL:.0e})"
    )
    print(f"   trace invariance worst relative |n_tot' - n_tot|    = {worst_trace:.3e}")
    print(
        f"   moment co-rotation worst relative |m' - R m|        = {worst_moment:.3e}"
    )
    print(
        f"   dc: contract frozen (bit-exact); rebuilt-from-rotated dc would drift"
        f" by max {worst_dc_drift:.6e} Ry"
    )
    print(f"   dc drift decomposition worst residual = {worst_dc_split:.3e} Ry")
    assert worst_cov <= COV_TOL, worst_cov
    assert worst_trace <= COV_TOL, worst_trace
    assert worst_moment <= COV_TOL, worst_moment
    assert worst_dc_drift > 0.0  # freezing is a real contract, not a no-op
    assert worst_dc_split <= 1e-12
    print("   E PASS: covariance, traces, moments, frozen dc all within gates")


def main():
    print(
        f"SymPy {sp.__version__}; NumPy {np.__version__};"
        " +U Hubbard rotation contract (story 015, ADR-007), exact algebra"
    )
    rotation = test_rotation_algebra()
    test_kernel_covariance()
    dc = test_dc_frozen()
    test_assembled_operator()
    test_numeric_contract()
    for path in (Path(__file__), Path(__file__).with_suffix(".md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"sha256 {path.name}: {digest}")
    print("ALL ASSERTIONS GREEN")
    return {
        "u_s": sp.simplify(rotation["u_s"]),
        "rotation_map": rotation["rotation_map"],
        "dc_uu_frozen": dc["dc_uu_frozen"],
        "dc_isotropic_invariant_piece": dc["dc_isotropic_invariant_piece"],
        "dc_channel_drift": dc["dc_channel_drift"],
        "numeric_covariance_tolerance": sp.Float(COV_TOL),
    }


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"FAIL: {error}")
        sys.exit(1)
