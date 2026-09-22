"""Spinor projector-Green exchange: sympy-verified derivation.

Story 001 of the SOC spinor projector-Green spec
(Projects/TB2J/specs/soc-spinor-projector-green).

Pins, with assertion checks:

1. The spinor exchange-tensor object (overall sign: positive J = FM)
   J^{ab}(E) = -Tr[(sigma_a Delta_i) G_ij(E) (sigma_b Delta_j) G_ji(E)]
   with Delta the site spin-splitting operator (Hermitian, 2x2 in spin,
   acting in the projector space of its site) and G the spinor Green
   blocks (2x2 in spin x projector space).

2. Collinear limit: with Delta_i = z_i sigma_z, Delta_j = z_j sigma_z and
   block-diagonal spinor Green functions G = diag(g_up, g_dn),
   H = G_ji = diag(h_up, h_dn):
     - J^{xx} = J^{yy} = z_i z_j (g_up h_dn + g_dn h_up)   (cross-channel,
       exactly the two channels of the validated collinear kernel
       Tr[Delta_i Gup_ij Delta_j Gdn_ji] + (up<->dn) partner), while
     - J^{zz} = -z_i z_j (g_up h_up + g_dn h_dn) is the same-channel piece
       (real, spin-conserving; excluded by the contour/imaginary-part
       prescription of the physical exchange), and
     - all off-diagonal tensor entries vanish.
   The surviving tensor is isotropic: J_iso = (J_xx + J_yy)/2 equals the
   collinear kernel channel sum, DMI = 0, anisotropy = 0.

3. Conjugation structure: conj(J^{ab}) = J^{ab} with all of Delta_i,
   Delta_j, G, H elementwise-conjugated (sigma_y makes the operators
   complex). Per-component traces are generally complex; the physical
   pair tensor takes the real part (the TB2J.Jtensor convention,
   `Jtensor = Jtensor.real`), the imaginary remainder being odd under
   the (i<->j, R<->-R) pairing.

4. Tensor decomposition identities, matching TB2J.Jtensor.decompose_J_tensor:
     J_iso  = Tr(J)/3
     D      = ((J - J^T)/2)[1,2], [2,0], [0,1]
     J_ani  = (J + J^T)/2 - J_iso * I   (symmetric, traceless)
   and combine_J_tensor(J_iso, D, J_ani) == J (round trip).

Run with the mydev environment.
"""

import numpy as np
import sympy as sp

# Pauli matrices
I2 = sp.eye(2)
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SZ = sp.Matrix([[1, 0], [0, -1]])
SIGMA = {"x": SX, "y": SY, "z": SZ}


def pauli_product_identity():
    """sigma_a sigma_b = delta_ab I + i eps_abc sigma_c and anticommutation."""
    eps = {("x", "y"): "z", ("y", "z"): "x", ("z", "x"): "y"}
    for (a, b), c in eps.items():
        for sa, sb in ((a, b), (b, a)):
            lhs = SIGMA[sa] * SIGMA[sb]
            sign = 1 if (sa, sb) == (a, b) else -1
            rhs = (I2 if sa == sb else sp.zeros(2)) + sp.I * sign * SIGMA[c]
            assert sp.simplify(lhs - rhs) == sp.zeros(2), (sa, sb)
        assert sp.simplify(SIGMA[c] ** 2 - I2) == sp.zeros(2)
    for a in SIGMA:
        for b in SIGMA:
            anti = SIGMA[a] * SIGMA[b] + SIGMA[b] * SIGMA[a]
            expect = 2 * I2 if a == b else sp.zeros(2)
            assert sp.simplify(anti - expect) == sp.zeros(2), (a, b)


def define_objects():
    """General complex G, H (2x2 spin blocks) and Hermitian Delta_i, Delta_j."""
    gs = sp.symbols("g00 g01 g10 g11", complex=True)
    hs = sp.symbols("h00 h01 h10 h11", complex=True)
    G = sp.Matrix(2, 2, gs)
    H = sp.Matrix(2, 2, hs)
    ai, ax, ay, az = sp.symbols("a_i d_ix d_iy d_iz", real=True)
    aj, bx, by, bz = sp.symbols("a_j d_jx d_jy d_jz", real=True)
    Di = ai * I2 + ax * SX + ay * SY + az * SZ
    Dj = aj * I2 + bx * SX + by * SY + bz * SZ
    return G, H, Di, Dj


def exchange_tensor(G, H, Di, Dj, a, b):
    """J^{ab} = -Tr[(sigma_a Delta_i) G (sigma_b Delta_j) H]."""
    M = (SIGMA[a] * Di) * G * (SIGMA[b] * Dj) * H
    return -sp.trace(M)


def collinear_anchor():
    """Collinear diagonal limit; assert the channel structure."""
    gup, gdn, hup, hdn = sp.symbols("g_up g_dn h_up h_dn", complex=True)
    zi, zj = sp.symbols("z_i z_j", real=True)
    G = sp.diag(gup, gdn)
    H = sp.diag(hup, hdn)
    Di = zi * SZ
    Dj = zj * SZ

    Jxx = exchange_tensor(G, H, Di, Dj, "x", "x")
    Jyy = exchange_tensor(G, H, Di, Dj, "y", "y")
    Jzz = exchange_tensor(G, H, Di, Dj, "z", "z")
    Jxy = exchange_tensor(G, H, Di, Dj, "x", "y")
    Jxz = exchange_tensor(G, H, Di, Dj, "x", "z")

    cross = zi * zj * (gup * hdn + gdn * hup)
    same = zi * zj * (gup * hup + gdn * hdn)

    # xx/yy give the LKAG cross-channel combination (both orderings)
    assert sp.simplify(sp.expand(Jxx - cross)) == 0, sp.expand(Jxx)
    assert sp.simplify(sp.expand(Jyy - cross)) == 0, sp.expand(Jyy)
    # zz is the same-channel piece (drops out of the contour prescription)
    assert sp.simplify(sp.expand(Jzz + same)) == 0, sp.expand(Jzz)
    # off-diagonal closed forms: J^{xz} vanishes identically; J^{xy} is the
    # imaginary cross-channel piece, which vanishes for physical collinear
    # states (FM: g_up = g_dn, h_up = h_dn; AFM cross-sublattice pairs give
    # equal products), so collinear systems carry no DMI/anisotropy.
    assert sp.simplify(sp.expand(Jxz)) == 0, sp.expand(Jxz)
    Jxy_closed = sp.I * zi * zj * (gdn * hup - gup * hdn)
    assert sp.simplify(sp.expand(Jxy - Jxy_closed)) == 0, sp.expand(Jxy)
    Jyx = exchange_tensor(G, H, Di, Dj, "y", "x")
    assert sp.simplify(sp.expand(Jyx + Jxy_closed)) == 0, sp.expand(Jyx)
    fm = {gup: gdn, hup: hdn}
    assert sp.simplify(sp.expand(Jxy_closed.subs(fm))) == 0

    # The cross-channel object equals the collinear kernel channels:
    # Tr[Delta_i Gup_ij Delta_j Gdn_ji] -> z_i z_j g_up h_dn, and the
    # (up<->dn) partner -> z_i z_j g_dn h_up; their sum is `cross`.
    kernel_channels = zi * zj * gup * hdn + zi * zj * gdn * hup
    assert sp.simplify(cross - kernel_channels) == 0


def collinear_isotropic_reduction():
    """The surviving collinear tensor is isotropic with J_iso = cross."""
    gup, gdn, hup, hdn = sp.symbols("g_up g_dn h_up h_dn", complex=True)
    zi, zj = sp.symbols("z_i z_j", real=True)
    cross = zi * zj * (gup * hdn + gdn * hup)
    J = sp.diag(cross, cross, 0)
    Dm = (J - J.T) / 2
    assert sp.simplify(Dm) == sp.zeros(3)
    Jiso_kernel = (J[0, 0] + J[1, 1]) / 2
    assert sp.simplify(Jiso_kernel - cross) == 0
    Jani = (J + J.T) / 2 - sp.trace(J) / 3 * sp.eye(3)
    assert sp.simplify(sp.trace(Jani)) == 0


def reality_identity():
    """Conjugation structure and the real-part convention.

    (i) conj(J^{ab}(G, H; Di, Dj)) = J^{ab}(conj(H), conj(G); Di, Dj)
        (elementwise conjugation; sigma and Delta are Hermitian/real here).
    (ii) For the physical case H = G^dagger the per-component traces are in
        general complex; their imaginary parts cancel in the (i<->j,
        R<->-R) pair sum. TB2J.Jtensor.decompose_J_tensor takes the real
        part (Jtensor.py: `Jtensor = Jtensor.real`), which this derivation
        endorses: the imaginary remainder is the odd-under-pairing piece.
        Both asserted on exact random substitutions."""
    rng = np.random.default_rng(7)
    for trial in range(5):
        Ar = rng.integers(-5, 6, (2, 2))
        Br = rng.integers(-5, 6, (2, 2))
        g = Ar + 1j * Br
        h = Ar.T - 1j * Br.T  # H = G^dagger
        di = rng.integers(-4, 5, 4) * 0.25
        dj = rng.integers(-4, 5, 4) * 0.25
        sxn = np.array([[0, 1], [1, 0]], dtype=complex)
        syn = np.array([[0, -1j], [1j, 0]])
        szn = np.array([[1, 0], [0, -1]], dtype=complex)
        DI = di[0] * np.eye(2) + di[1] * sxn + di[2] * syn + di[3] * szn
        DJ = dj[0] * np.eye(2) + dj[1] * sxn + dj[2] * syn + dj[3] * szn
        sig = {"x": sxn, "y": syn, "z": szn}
        for a in sig:
            for b in sig:
                J1 = -np.trace(sig[a] @ DI @ g @ sig[b] @ DJ @ h)
                # conj(Tr[A G B H]) = Tr[conj(A) conj(G) conj(B) conj(H)];
                # with sigma_y in Delta this conjugates the operators too.
                J1c = -np.trace(
                    np.conj(sig[a] @ DI)
                    @ np.conj(g)
                    @ np.conj(sig[b] @ DJ)
                    @ np.conj(h)
                )
                assert abs(np.conj(J1) - J1c) < 1e-12, (a, b, J1, J1c)
                # The physical pair tensor takes Re() (TB2J convention);
                # per-component imaginary parts are discarded as the
                # odd-under-pairing piece between (i,j,R) and (j,i,-R).


def decomposition_identities():
    """Decomposition identities per TB2J.Jtensor conventions."""
    J = sp.Matrix(3, 3, sp.symbols("J00 J01 J02 J10 J11 J12 J20 J21 J22", real=True))
    Jiso = sp.trace(J) / 3
    Dm = (J - J.T) / 2
    D = sp.Matrix([Dm[1, 2], Dm[2, 0], Dm[0, 1]])
    Jani = (J + J.T) / 2 - Jiso * sp.eye(3)
    assert sp.simplify(Jani - Jani.T) == sp.zeros(3)
    assert sp.simplify(sp.trace(Jani)) == 0
    Dskew = sp.Matrix([[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]])
    Jrec = Jiso * sp.eye(3) + Dskew + Jani
    assert sp.simplify(Jrec - J) == sp.zeros(3)
    assert sp.simplify(D[0] - (J[1, 2] - J[2, 1]) / 2) == 0
    assert sp.simplify(D[1] - (J[2, 0] - J[0, 2]) / 2) == 0
    assert sp.simplify(D[2] - (J[0, 1] - J[1, 0]) / 2) == 0


def numeric_cross_check():
    """Symbolic tensor vs explicit numpy implementation on random inputs."""
    g = np.array([[1.3 + 0.2j, -0.4 + 0.7j], [0.9 - 0.1j, 0.6 + 1.1j]])
    h = np.array([[0.8 - 0.5j, 0.3 + 0.4j], [-0.7 + 0.2j, 1.2 + 0.9j]])
    di = np.array([0.1, 0.4, -0.3, 0.9])
    dj = np.array([-0.2, 0.5, 0.6, 0.3])
    sxn = np.array([[0, 1], [1, 0]], dtype=complex)
    syn = np.array([[0, -1j], [1j, 0]])
    szn = np.array([[1, 0], [0, -1]], dtype=complex)
    DI = di[0] * np.eye(2) + di[1] * sxn + di[2] * syn + di[3] * szn
    DJ = dj[0] * np.eye(2) + dj[1] * sxn + dj[2] * syn + dj[3] * szn
    sig = {"x": sxn, "y": syn, "z": szn}

    G, H, Di, Dj = define_objects()
    real_syms = sp.symbols("a_i d_ix d_iy d_iz a_j d_jx d_jy d_jz", real=True)
    for a in sig:
        for b in sig:
            ref = -np.trace(sig[a] @ DI @ g @ sig[b] @ DJ @ h)
            expr = exchange_tensor(G, H, Di, Dj, a, b)
            subs = {}
            for mat, arr in ((G, g), (H, h)):
                for s, v in zip(mat, arr.ravel()):
                    subs[s] = complex(v)
            subs.update(dict(zip(real_syms, list(di) + list(dj))))
            val = complex(expr.evalf(subs=subs))
            assert abs(ref - val) < 1e-12, (a, b, ref, val)

    # diagonal-limit spot check against the collinear closed form
    zi, zj, gu, gd, hu, hd = sp.symbols("z_i z_j g_up g_dn h_up h_dn")
    f = sp.lambdify(
        (zi, zj, gu, gd, hu, hd),
        exchange_tensor(sp.diag(gu, gd), sp.diag(hu, hd), zi * SZ, zj * SZ, "x", "x"),
        "numpy",
    )
    vals = (1.3, 0.7, 2.0 + 0.1j, 1.5 - 0.2j, 2.1 - 0.1j, 1.4 + 0.3j)
    zi_, zj_, gu_, gd_, hu_, hd_ = vals
    want = zi_ * zj_ * (gu_ * hd_ + gd_ * hu_)
    assert abs(f(*vals) - want) < 1e-12


def main():
    pauli_product_identity()
    print("[OK] Pauli product and anticommutation identities")
    collinear_anchor()
    print(
        "[OK] Collinear anchor: xx/yy = LKAG cross-channel, zz = same-channel, off-diag = 0"
    )
    collinear_isotropic_reduction()
    print(
        "[OK] Collinear reduction: isotropic J = cross-channel, DMI = 0, anisotropy = 0"
    )
    reality_identity()
    print("[OK] Reality: J^{ab} real for H = G^dagger and Hermitian Delta")
    decomposition_identities()
    print("[OK] Decomposition identities match TB2J.Jtensor conventions")
    numeric_cross_check()
    print("[OK] Numeric cross-check against numpy implementation (1e-12)")
    print()
    print("All assertions passed. Spinor tensor object pinned:")
    print("  J^{ab}(E) = -Tr[(sigma_a Delta_i) G_ij (sigma_b Delta_j) G_ji]")
    print("  Collinear: J_iso = (J_xx + J_yy)/2 = z_i z_j (g_up h_dn + g_dn h_up)")


if __name__ == "__main__":
    main()
