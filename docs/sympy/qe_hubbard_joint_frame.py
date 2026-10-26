"""Exact QE XC+U joint-frame and atomic-only normalization contracts.

Run with mydev: python docs/sympy/qe_hubbard_joint_frame.py.
The examples use exact complex numbers. They are algebraic contracts, not
native QE action certification or a completeness proof for the XC subspace.
"""

import sympy as sp


# MathDoc: tb2j/qe-hubbard-joint-frame#joint_trace
# Rectangular cyclicity does not require independent projector columns.
def check_joint_trace():
    def matrix(prefix, rows, cols):
        return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f"{prefix}_{i}{j}"))

    # Independent complex symbols: no Hermiticity or real-energy assumption
    # is imposed on either resolvent or on the two different site frames.
    wi = matrix("wi", 2, 3)
    wj = matrix("wj", 2, 3)
    gp = matrix("gp", 2, 2)
    gm = matrix("gm", 2, 2)
    xi = matrix("xi", 2, 2)
    xj = matrix("xj", 2, 2)
    ui, uj = sp.symbols("ui uj")
    di = sp.diag(xi, ui)
    dj = sp.diag(xj, uj)
    direct = sp.trace(wi * di * wi.H * gp * wj * dj * wj.H * gm)
    joint = sp.trace(di * (wi.H * gp * wj) * dj * (wj.H * gm * wi))
    assert sp.expand(direct - joint) == 0
    # Three channels in two physical dimensions are necessarily redundant.
    assert (wi.H * wi).det().expand() == 0

    pi = sp.MatrixSymbol("P_i", 2, 3)
    pj = sp.MatrixSymbol("P_j", 2, 3)
    vi = sp.MatrixSymbol("D_i", 3, 3)
    vj = sp.MatrixSymbol("D_j", 3, 3)
    gu = sp.MatrixSymbol("G_up", 2, 2)
    gd = sp.MatrixSymbol("G_down", 2, 2)
    return sp.Eq(
        sp.Trace(pi * vi * sp.Adjoint(pi) * gu * pj * vj * sp.Adjoint(pj) * gd),
        sp.Trace(vi * (sp.Adjoint(pi) * gu * pj) * vj * (sp.Adjoint(pj) * gd * pi)),
        evaluate=False,
    )


# MathDoc: tb2j/qe-hubbard-joint-frame#mixed_terms
# Site-local coefficients are block diagonal by source, not the Green matrix.
def check_mixed_sources():
    wx = sp.Matrix([[1, sp.I], [2 + sp.I, 1], [0, 2]])
    wu = sp.Matrix([[1 - sp.I, 2], [0, 1 + sp.I], [3, -1]])
    # A full intersite atomic filter, not separate onsite inverses.
    filt = (wx.H * wx).inv()
    frame = (wx * filt).row_join(wu)
    assert (frame.H * frame).det() == 0
    gp = sp.Matrix([[1 + sp.I, 2, 0], [3 * sp.I, -2, 1], [2, 1 - sp.I, 3]])
    gm = sp.Matrix([[2, 1 + sp.I, 4], [1, -sp.I, 2], [3, 0, 1 + 2 * sp.I]])
    xi, ui, xj, uj = sp.symbols("x_i u_i x_j u_j", real=True)
    di = sp.diag(xi, 0, ui, 0)
    dj = sp.diag(0, xj, 0, uj)
    gij = frame.H * gp * frame
    gji = frame.H * gm * frame
    complete = sp.expand(sp.trace(di * gij * dj * gji))
    direct = sp.trace(frame * di * frame.H * gp * frame * dj * frame.H * gm)
    assert sp.expand(complete - direct) == 0
    contributions = {
        (a, b): sp.expand(complete).coeff(a).coeff(b)
        for a in (xi, ui)
        for b in (xj, uj)
    }
    assert all(value != 0 for value in contributions.values())
    isolated = xi * xj * contributions[xi, xj] + ui * uj * contributions[ui, uj]
    mixed = sp.expand(complete - isolated)
    assert mixed != 0
    assert sp.expand(
        mixed - xi * uj * contributions[xi, uj] - ui * xj * contributions[ui, xj]
    ) == 0
    return mixed


# MathDoc: tb2j/qe-hubbard-joint-frame#row_filter_transpose
# Stored rows C[n,p]=<p|n>; get_Gk contracts C.T, not C.H.
def check_row_filter():
    cx = sp.Matrix([[1, 2 + sp.I], [sp.I, 3], [2 - sp.I, -1]])
    cu = sp.Matrix([[2, 1 + sp.I], [1 - sp.I, 4], [sp.I, -2]])
    a, b, r, s = sp.symbols("a b r s", real=True)
    filt = sp.Matrix([[a, r + sp.I * s], [r - sp.I * s, b]])
    assert filt.H == filt
    denominator = sp.diag(1 + sp.I, sp.Rational(2, 3) - sp.I, -2 + sp.I / 2)
    original = cx.row_join(cu)
    normalized = (cx * filt.T).row_join(cu)
    transform = sp.diag(filt, sp.eye(cu.cols))
    spectral = original.T * denominator * original.conjugate()
    actual = normalized.T * denominator * normalized.conjugate()
    expected = transform * spectral * transform.H
    assert (actual - expected).applyfunc(sp.expand) == sp.zeros(4)
    assert actual[2:, 2:] == spectral[2:, 2:]
    assert (actual[:2, 2:] - filt * spectral[:2, 2:]).applyfunc(sp.expand) == sp.zeros(2)
    assert (actual[2:, :2] - spectral[2:, :2] * filt).applyfunc(sp.expand) == sp.zeros(2)
    # Complex off-diagonal entries discriminate transpose from adjoint/no transpose.
    wrong = (cx * filt).row_join(cu)
    wrong_green = wrong.T * denominator * wrong.conjugate()
    assert (actual - wrong_green).applyfunc(sp.expand) != sp.zeros(4)
    concrete = {a: 2, b: 3, r: 1, s: 1}
    twice = (cx * (filt.T ** 2)).row_join(cu)
    twice_green = twice.T * denominator * twice.conjugate()
    assert (twice_green - actual).subs(concrete).applyfunc(sp.expand) != sp.zeros(4)
    # Neither the native overlaps nor their UU block acquire an atomic filter.
    wrong_u = (cx * filt.T).row_join(cu * filt.T)
    wrong_u_green = wrong_u.T * denominator * wrong_u.conjugate()
    assert (wrong_u_green[2:, 2:] - actual[2:, 2:]).subs(concrete).applyfunc(sp.expand) != sp.zeros(2)
    c = sp.MatrixSymbol("C_X", 3, 2)
    f = sp.MatrixSymbol("F", 2, 2)
    filtered = sp.MatrixSymbol("C_X_filtered", 3, 2)
    return sp.Eq(filtered, c * f.T, evaluate=False)


# MathDoc: tb2j/qe-hubbard-joint-frame#fourier_filter
# Normalize each full per-k XC frame before source concatenation and Fourier sum.
def check_fourier_filter():
    cx = (
        sp.Matrix([[1, sp.I], [2, 1 + sp.I], [1 - sp.I, 0]]),
        sp.Matrix([[sp.I, 2], [1, -sp.I], [2 + sp.I, 3]]),
    )
    cu = (
        sp.Matrix([[1], [sp.I], [2]]),
        sp.Matrix([[2 - sp.I], [1], [-sp.I]]),
    )
    filters = (
        sp.Matrix([[2, sp.I], [-sp.I, 3]]),
        # A positive semidefinite, rank-truncated filter is allowed too.
        sp.Matrix([[1, sp.I], [-sp.I, 1]]),
    )
    assert filters[1].det() == 0
    denominators = (sp.diag(1, 2 + sp.I, 3), sp.diag(2, -1, 1 - sp.I))
    phases = (sp.Rational(1, 3), 2 * sp.I / 3)
    normalized_sum = sp.zeros(3)
    expected_sum = sp.zeros(3)
    raw_sum = sp.zeros(3)
    for atomic, native, filt, denom, phase in zip(cx, cu, filters, denominators, phases):
        rows = atomic.row_join(native)
        normalized = (atomic * filt.T).row_join(native)
        raw = rows.T * denom * rows.conjugate()
        transform = sp.diag(filt, sp.eye(1))
        normalized_sum += phase * normalized.T * denom * normalized.conjugate()
        expected_sum += phase * transform * raw * transform.H
        raw_sum += phase * raw
    assert (normalized_sum - expected_sum).applyfunc(sp.expand) == sp.zeros(3)
    first_filter = sp.diag(filters[0], sp.eye(1))
    assert (normalized_sum - first_filter * raw_sum * first_filter.H).applyfunc(sp.expand) != sp.zeros(3)
    return sp.ImmutableMatrix(normalized_sum)


def main():
    results = {}
    for name, check in (
        ("joint_trace", check_joint_trace),
        ("mixed_terms", check_mixed_sources),
        ("row_filter_transpose", check_row_filter),
        ("fourier_filter", check_fourier_filter),
    ):
        results[name] = check()
        print(f"PASS: QE Hubbard {name}")
    return results


if __name__ == "__main__":
    main()
