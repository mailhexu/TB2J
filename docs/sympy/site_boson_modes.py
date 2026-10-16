"""Assertion-checked derivation of TB2J canonical site-boson modes.

Run with the project environment: python docs/sympy/site_boson_modes.py
"""

import sympy as sp


def _colpa_two_component(A, B, omega=None):
    """Derive the normalized paraunitary for a real 2x2 BdG Hamiltonian."""
    w = sp.sqrt(A**2 - B**2) if omega is None else omega
    u = sp.sqrt((A + w) / (2 * w))
    v = sp.sqrt((A - w) / (2 * w))
    H = sp.Matrix([[A, B], [B, A]])
    g = sp.diag(1, -1)
    T = sp.Matrix([[u, v], [-v, -u]])
    assert sp.simplify(T.T * g * T - g) == sp.zeros(2)
    assert sp.simplify(T.T * H * T - sp.diag(w, w)) == sp.zeros(2)
    assert sp.simplify(u**2 - v**2 - 1) == 0
    return w, u, v, H, g, T


def main():
    # NumPy's lower Cholesky obeys H=K K^dagger.  TB2J diagonalizes
    # M=K^dagger g K (not K g K^dagger); M is similar to gH via K^dagger.
    a, c = sp.symbols("a c", positive=True, real=True)
    z = sp.symbols("z", complex=True)
    K = sp.Matrix([[a, 0], [z, c]])
    g = sp.diag(1, -1)
    H = K * K.H
    M = K.H * g * K
    BdG = g * H
    assert sp.simplify(K.H * BdG * (K.H).inv() - M) == sp.zeros(2)

    # For a Euclidean-unit eigenvector x of the code's M eigensystem,
    # psi=K^{-dagger}x is the physical generalized eigenvector of gH.
    omega = sp.symbols("omega", positive=True, real=True)
    x1, x2 = sp.symbols("x1 x2", complex=True)
    x = sp.Matrix([x1, x2])
    psi = (K.H).inv() * x
    physical = sp.sqrt(omega) * psi
    residual = BdG * psi - omega * psi
    assert sp.simplify(residual - (K.H).inv() * (M * x - omega * x)) == sp.zeros(2, 1)
    assert sp.simplify((psi.H * H * psi)[0] - (x.H * x)[0]) == 0
    assert sp.simplify((psi.H * g * psi)[0] - (x.H * M.inv() * x)[0]) == 0
    # The difference from 1 is exactly proportional to the eigensystem
    # residual; it vanishes when Mx=omega*x and x^dagger*x=1.
    metric_difference = (physical.H * g * physical)[0] - (x.H * x)[0]
    residual_identity = sp.sqrt(omega) ** 2 * (
        (x.H * M.inv() * x)[0] - (x.H * x)[0] / omega
    )
    assert sp.simplify(metric_difference - residual_identity) == 0
    assert (
        sp.simplify(
            (x.H * M.inv() * x)[0]
            - (x.H * x)[0] / omega
            + (x.H * M.inv() * (M * x - omega * x))[0] / omega
        )
        == 0
    )
    assert sp.simplify((physical.H * H * physical)[0] - omega * (x.H * x)[0]) == 0

    # The full transformation is made from both M branches, scaled by
    # sqrt(abs(omega)); positive columns precede negative-norm partners.
    gap, B = sp.symbols("gap B", positive=True, real=True)
    A = sp.sqrt(gap**2 + B**2)
    w, u, v, H2, g2, T = _colpa_two_component(A, B, gap)
    assert sp.simplify(w - gap) == 0
    assert sp.simplify(T.T * g2 * T - g2) == sp.zeros(2)
    assert sp.simplify(T.T * H2 * T - sp.diag(w, w)) == sp.zeros(2)

    # Small-model checks: FM chain (B=0), two-sublattice AFM, and an
    # engineered anomalous model with nonzero pair-creation block.
    q, J, S = sp.symbols("q J S", positive=True, real=True)
    fm_energy = 4 * J * S * (1 - sp.cos(q))
    Afm = sp.symbols("A_fm", positive=True, real=True)
    fm_w, fm_u, fm_v, _, _, _ = _colpa_two_component(Afm, sp.Integer(0), Afm)
    assert sp.simplify(fm_w - Afm) == 0
    assert sp.simplify(fm_u - 1) == 0
    assert sp.simplify(fm_v) == 0
    assert sp.simplify(sp.limit(fm_energy, q, 0, dir="+") - 0) == 0

    # Symmetric two-sublattice AFM: A=2JS and B=2JS cos(q), with
    # omega=2JS|sin(q)| away from the Goldstone point.
    afm_A = 2 * J * S
    afm_B = 2 * J * S * sp.cos(q)
    afm_w = sp.sqrt(afm_A**2 - afm_B**2)
    assert sp.simplify(afm_w**2 - (2 * J * S * sp.sin(q)) ** 2) == 0
    assert sp.simplify((afm_A / afm_w + 1) / 2 - ((afm_A + afm_w) / (2 * afm_w))) == 0
    assert sp.simplify((afm_A / afm_w - 1) / 2 - ((afm_A - afm_w) / (2 * afm_w))) == 0

    # Engineered anomalous case: B != 0, with strict stability A>|B|.
    b0 = sp.symbols("b0", positive=True, real=True)
    gap0 = sp.symbols("gap0", positive=True, real=True)
    A0 = sp.sqrt(gap0**2 + b0**2)
    w0, u0, v0, H0, g0, T0 = _colpa_two_component(A0, b0, gap0)
    assert sp.simplify(w0 - gap0) == 0
    assert sp.simplify((u0**2 - v0**2) - 1) == 0
    assert sp.simplify((u0 * v0) ** 2 - b0**2 / (4 * gap0**2)) == 0

    # The BdG particle-hole identity relates opposite k, not the positive
    # branches at the same k. A nonreciprocal normal term has A(k)!=A(-k).
    Ak, Am, Bk = sp.symbols("A_k A_minus_k B", real=True)
    Hk = sp.Matrix([[Ak, Bk], [Bk, Am]])
    Hminus = sp.Matrix([[Am, Bk], [Bk, Ak]])
    tau_x = sp.Matrix([[0, 1], [1, 0]])
    assert sp.simplify(Hminus - tau_x * sp.conjugate(Hk) * tau_x) == sp.zeros(2)
    gk = sp.diag(1, -1)
    Dk, Dminus = gk * Hk, gk * Hminus
    assert sp.simplify(tau_x * sp.conjugate(Dk) * tau_x + Dminus) == sp.zeros(2)
    root = sp.sqrt((Ak + Am) ** 2 - 4 * Bk**2)
    negative_k = (Ak - Am - root) / 2
    positive_minus_k = (Am - Ak + root) / 2
    assert sp.simplify(positive_minus_k + negative_k) == 0

    # TB2J puts sqrt(S_i S_j) into the exchange blocks; the modes then
    # multiply unit-commutator HP bosons, whose transverse spin carries sqrt(2S_i).
    si, sj, Jij = sp.symbols("S_i S_j J_ij", positive=True, real=True)
    scaled_block = sp.sqrt(si * sj) * Jij
    assert sp.simplify(scaled_block**2 - si * sj * Jij**2) == 0
    b = sp.symbols("b", commutative=False)
    delta_spin_plus = sp.sqrt(2 * si) * b
    bi, bdi, bj, bdj = sp.symbols("b_i b_i_dag b_j b_j_dag")
    Jx, Jy = sp.symbols("J_x J_y", real=True)
    pref_i, pref_j = sp.sqrt(si / 2), sp.sqrt(sj / 2)
    Sxi, Syi = pref_i * (bi + bdi), -sp.I * pref_i * (bi - bdi)
    Sxj, Syj = pref_j * (bj + bdj), -sp.I * pref_j * (bj - bdj)
    transverse = sp.expand(Jx * Sxi * Sxj + Jy * Syi * Syj)
    assert (
        sp.simplify(transverse.coeff(bdi * bj) - sp.sqrt(si * sj) * (Jx + Jy) / 2) == 0
    )
    assert (
        sp.simplify(transverse.coeff(bi * bj) - sp.sqrt(si * sj) * (Jx - Jy) / 2) == 0
    )
    assert sp.simplify(delta_spin_plus / sp.sqrt(2 * si) - b) == 0

    # At an exact zero mode the unregularized Cholesky is singular. In the
    # FM limit B=0, u=1,v=0 continuously; for anomalous branches the
    # canonical amplitudes may diverge while their metric norm remains one.
    assert sp.simplify(sp.limit(fm_u, q, 0, dir="+") - 1) == 0
    assert sp.simplify(sp.limit(fm_v, q, 0, dir="+") - 0) == 0
    eps = sp.symbols("eps", positive=True, real=True)
    Aeps = sp.sqrt(eps**2 + b0**2)
    weps, _, veps, _, _, _ = _colpa_two_component(Aeps, b0, eps)
    assert sp.simplify(weps - eps) == 0
    assert sp.limit(veps**2, eps, 0, dir="+") is sp.oo
    print(
        "lower-Cholesky reconstruction, paraunitarity, FM/AFM/anomalous models, "
        "spin factors, reciprocal partners, and zero-mode limits: OK"
    )


if __name__ == "__main__":
    main()
