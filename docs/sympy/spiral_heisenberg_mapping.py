"""Assertion authority for story 012: A-tensor -> Heisenberg mapping layer.

Run in mydev: python docs/sympy/spiral_heisenberg_mapping.py
Paired derivation, conventions and limits: same-name .md file.
No production code is imported. Exact assertions never fall back to probes.

Scope (FR-009 / ADR-006, SymPy first). From ONE spiral reference with
per-site local frames the ResponseSet (TB2J/interfaces/siesta_spiral_response.py,
version 2.1) reports the transverse Hessian K[(a,alpha),(b,beta)] over the two
declared transverse axes of each site (a = b allowed), split as
onsite + response, and the gradients (torque = -gradient). Longitudinal
response is absent by construction: the unit-moment chart has exactly two
tilt coordinates per site. This script derives and pins, with exact SymPy
assertions (nonzero exit on any failure):

  [A] chart identities: unit norm, transversality (no longitudinal
      direction), Clairaut/seagull symmetry of the same-site second vertex;
  [B] the A-tensor -> pair-parameter mapping for the pair-once energy
          E_pair = -J (e_a.e_b) + D.(e_a x e_b) + e_a.Gamma.e_b
      (Gamma symmetric traceless, J > 0 ferromagnetic, unit directions,
      no |m|^2 division):
          K_ab[alpha,beta] = -J (u_a.alpha . u_b.beta)
                             + D . (u_a.alpha x u_b.beta)
                             + u_a.alpha . Gamma . u_b.beta
          K_aa[alpha,beta] = -delta[alpha,beta] (n_a . Gamma . n_b)
          g_a.alpha        = -J (u_a.alpha . n_b) + D . (u_a.alpha x n_b)
                             + u_a.alpha . Gamma . n_b
          g_b.beta         = -J (n_a . u_b.beta) + D . (n_a x u_b.beta)
                             + u_b.beta . Gamma . n_a
      with tilt directions u_a.alpha = t_a.alpha x n_a; D enters the mixed
      cross products and the onsite DIAGONAL blocks through
      -D.(n_a x n_b) (zero at collinear references where n_a x n_b = 0);
      J never enters onsite blocks beyond the +delta J (n_a.n_b) trace;
  [C] the zero-pitch (q = 0 collinear, common transverse direction)
      reduction to the story-011 identity
          J_stored = -K_ab / (2 transverse_dot),  J_pair = 2 J_stored,
      with transverse_dot the SITE-LOCAL tilt dot (u_a.u_b), so the AFM
      alignment sign is already encoded in the charts (u_b = -u_a);
      equivalently common-axis projection with explicit alignment;
      including the in-plane-anisotropy contamination of the extracted J
      and its onsite closure;
  [D] the single-reference identifiability matrices for two sites:
      exact ranks and null spaces over the parameters
      (J, D_x, D_y, D_z local-projected, Gamma in-plane pair plus the
      longitudinal-transverse components), the 3x3 Szilva pair block
      minus gauge, and torque/multi-reference completion;
  [E] residual and compatibility-flag definitions (per-component residual,
      weighted aggregate over the gauge-free data, dimensionless statistic
      with explicit denominator floor);
  [F] synthetic constructions: exact-Heisenberg (zero residual), pure
      biquadratic (invisible at a collinear anchor, flagged at a
      noncollinear reference), pure three-site (flagged through the
      onsite-block violation).

Conventions are those of docs/sympy/collinear_anchor_projection.py
(story 011): pair-once energy, TB2J stores both ordered directions at
J_stored = J_pair / 2, positive J ferromagnetic, DMI sign fixed by
E_dmi = +D.(e_a x e_b). Literature anchor: A. Szilva et al., Rev. Mod.
Phys. 95, 035004 (2023) (arXiv:2206.02415).
"""

import hashlib
from pathlib import Path

import sympy as sp

HALF = sp.Rational(1, 2)
X = sp.Matrix([1, 0, 0])
Y = sp.Matrix([0, 1, 0])
Z = sp.Matrix([0, 0, 1])

# Pair parameters, one ordered pair (a, b); Gamma is symmetric traceless so
# Gamma_zz = -Gamma_xx - Gamma_yy. D is parametrized in site a's local chart
# (the anchors below place site a's frame on the Cartesian axes, so the
# local projections D_x, D_y, D_z ARE the Cartesian components there).
J, D_X, D_Y, D_Z, G_XX, G_XY, G_XZ, G_YZ, G_YY = sp.symbols(
    "J D_x D_y D_z Gamma_xx Gamma_xy Gamma_xz Gamma_yz Gamma_yy", real=True
)
PARAMS = [J, D_X, D_Y, D_Z, G_XX, G_XY, G_XZ, G_YZ, G_YY]
PARAM_NAMES = [str(p) for p in PARAMS]
GAMMA = sp.Matrix([[G_XX, G_XY, G_XZ], [G_XY, G_YY, G_YZ], [G_XZ, G_YZ, -G_XX - G_YY]])
DVEC = sp.Matrix([D_X, D_Y, D_Z])

# Compatibility statistic conventions (story-012 policy parameters).
CHI_MAX = sp.Rational(1, 20)
CHI2_MAX = CHI_MAX**2
K_FLOOR = sp.Rational(1, 10**6)

# Synthetic strengths.
J_SYN = sp.Rational(1, 10)
B_SYN = sp.Rational(1, 100)
K3_SYN = sp.Rational(3, 100)


def zero(value):
    """Require exact symbolic equality to zero, entrywise for a matrix."""
    reduced = sp.simplify(value)
    if isinstance(reduced, sp.MatrixBase):
        for entry in reduced:
            assert sp.simplify(entry) == 0, entry
    else:
        assert reduced == 0, reduced


def rodrigues_matrix(axis, angle):
    axis = sp.Matrix(axis)
    n = axis.norm()
    K = sp.Matrix(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return sp.eye(3) + sp.sin(angle) * K / n + (1 - sp.cos(angle)) * (K / n) ** 2


def triad(t1, n):
    """Right-handed orthonormal frame (t1, t2, n) with t1 x t2 = n.

    Unit norms are asserted when numeric; symbolic directions (s, c) with
    s^2 + c^2 = 1 are carried through exactly.
    """
    t1 = sp.Matrix(t1)
    n = sp.Matrix(n)
    zero(t1.dot(n))
    t2 = sp.simplify(n.cross(t1))
    for vec in (t1, n):
        norm2 = sp.simplify(vec.dot(vec))
        if norm2.is_number:
            zero(norm2 - 1)
    zero(sp.simplify(t2.dot(t1)))
    zero(sp.simplify(t2.dot(n)))
    zero(sp.simplify(t2.dot(t2) - n.dot(n)))
    return [t1, t2, n]


def tilts(frame):
    """Tilt directions u_alpha = t_alpha x n of the exponential chart."""
    t1, t2, n = frame
    return [sp.simplify(t1.cross(n)), sp.simplify(t2.cross(n))]


def chart(frame, angles):
    """Two-axis composed exponential chart e(theta1, theta2) = R2 R1 n."""
    t1, t2, n = frame
    return sp.simplify(
        rodrigues_matrix(t2, angles[1]) * rodrigues_matrix(t1, angles[0]) * n
    )


def dmi_term(dvec, e_a, e_b):
    return sp.simplify((dvec.T * (e_a.cross(e_b)))[0, 0])


def coef(expression, parameter):
    return sp.simplify(sp.diff(sp.expand(expression), parameter))


def mapping_rows(ua, ub, na, nb):
    """Closed-form rows of the pair mapping over PARAMS.

    Mixed block K_ab[alpha,beta], then both onsite 2x2 blocks in the order
    (11, 12, 21, 22) for site a and site b. Gradient rows are separate.
    """
    mixed = []
    for u_a in ua:
        for u_b in ub:
            dot = sp.simplify(u_a.dot(u_b))
            cross = sp.simplify(u_a.cross(u_b))
            gam = (u_a.T * GAMMA * u_b)[0, 0]
            mixed.append(
                [sp.simplify(-dot)]
                + [cross[i] for i in range(3)]
                + [coef(gam, p) for p in PARAMS[4:]]
            )
    ell = (na.T * GAMMA * nb)[0, 0]
    nab = sp.simplify(na.dot(nb))
    nab_cross = sp.simplify(na.cross(nb))
    # Onsite entry rows (11, 12, 21, 22): Heisenberg +J (n_a.n_b) delta and
    # anisotropy -delta (n_a.Gamma.n_b); the DMI contributes -D.(n_a x n_b)
    # on the diagonal entries only (d^2 e/dtheta^2 = -n at theta = 0, so
    # d^2 E_D/dtheta_alpha^2 = -D.(n_a x n_b)); off-diagonals stay zero
    # (the composed chart's mixed second vertex vanishes, asserted in [A3]).
    onsite_row = (
        [nab] + [-nab_cross[i] for i in range(3)] + [coef(-ell, p) for p in PARAMS[4:]]
    )
    onsite = []
    for _ in range(2):
        onsite.append(list(onsite_row))
        onsite.append([sp.Integer(0)] * 9)
        onsite.append([sp.Integer(0)] * 9)
        onsite.append(list(onsite_row))
    return sp.Matrix(mixed + onsite)


def gradient_rows(ua, ub, na, nb):
    """Gradient rows of the pair model over PARAMS.

    From E_D = +D.(e_a x e_b), direct differentiation gives
    dE/dtheta_a.alpha = D.(u_a.alpha x n_b) and
    dE/dtheta_b.beta  = D.(n_a x u_b.beta)  (NOT u_b x n_a; the cross
    product is ordered as (e_a x de_b/dtheta)). The Heisenberg and Gamma
    gradients are symmetric under the a/b swap.
    """
    rows = []
    for u, other, dmi_cross in (
        (ua, nb, lambda u_, n_: u_.cross(n_)),
        (ub, na, lambda u_, n_: n_.cross(u_)),
    ):
        for u_a in u:
            dot = sp.simplify(u_a.dot(other))
            cross = sp.simplify(dmi_cross(u_a, other))
            gam = (u_a.T * GAMMA * other)[0, 0]
            rows.append(
                [sp.simplify(-dot)]
                + [cross[i] for i in range(3)]
                + [coef(gam, p) for p in PARAMS[4:]]
            )
    return sp.Matrix(rows)


def isotropic_layer(ua, ub, na, nb):
    """Rows of the isotropic Heisenberg-only representation (12 entries)."""
    mixed = [[sp.simplify(-(u_a.dot(u_b)))] for u_a in ua for u_b in ub]
    nab = sp.simplify(na.dot(nb))
    onsite = []
    for _ in range(2):
        onsite += [[nab], [sp.Integer(0)], [sp.Integer(0)], [nab]]
    return sp.Matrix(mixed + onsite)


def raw_data(energy, angles_a, angles_b):
    """Differentiate the actual energy on the composed charts.

    Layout: mixed (a-angles x b-angles), onsite blocks (11, 12, 21, 22)
    for a then b. All evaluated at the reference theta = 0.
    """
    zero_ = {t: 0 for t in list(angles_a) + list(angles_b)}
    data = []
    for i in range(2):
        for j in range(2):
            data.append(
                sp.simplify(sp.diff(energy, angles_a[i], angles_b[j]).subs(zero_))
            )
    for angles in (angles_a, angles_b):
        block = [
            [
                sp.simplify(sp.diff(energy, angles[i], angles[j]).subs(zero_))
                for j in range(2)
            ]
            for i in range(2)
        ]
        data += [block[0][0], block[0][1], block[1][0], block[1][1]]
    # Same-site Clairaut: the composed chart is identical under swap.
    for angles in (angles_a, angles_b):
        zero(
            sp.simplify(
                sp.diff(energy, angles[0], angles[1])
                - sp.diff(energy, angles[1], angles[0])
            )
        )
    return sp.Matrix(data)


def chi_squared(data, layer):
    """(r2, d2) for the exact least-squares representation residual."""
    proj = layer * layer.pinv()
    resid = sp.simplify(data - proj * data)
    r2 = sp.simplify((resid.T * resid)[0])
    d2 = sp.simplify((data.T * data)[0])
    return sp.simplify(sp.cancel(r2)), sp.simplify(sp.cancel(d2))


# ---------------------------------------------------------------------------
# Anchors. Site a always carries the Cartesian frame (X, Y, Z), so its local
# projections (D_x, D_y, D_z) are the Cartesian components (documented).
# ---------------------------------------------------------------------------
def anchor_collinear():
    fa = triad(X, Z)
    return fa, fa


def anchor_planar(c, s):
    """n_b = c X + s Z: the reference tilted along t1_a inside the x-z plane."""
    fa = triad(X, Z)
    nb = sp.Matrix([c, 0, s])
    fb = triad(Y, nb)
    return fa, fb


def anchor_flipped(c, s):
    """n_b = s Y + c Z: the reference tilted along t2_a."""
    fa = triad(X, Z)
    nb = sp.Matrix([0, s, c])
    fb = triad(X, nb)
    return fa, fb


def anchor_nonplanar(components, t_along=Y):
    fa = triad(X, Z)
    nb = sp.Matrix(components)
    t1 = sp.simplify(t_along - nb * (t_along.dot(nb)))
    t1 = sp.simplify(t1 / t1.norm())
    fb = triad(t1, nb)
    return fa, fb


def test_chart_and_data_model():
    """[A] Story-012 data model: transverse chart, gauge, symmetry."""
    print("[A] Chart and data model (transverse Hessian, no longitudinal axis)")
    frame = triad(X, Z)
    t1, t2, n = frame
    th1, th2 = sp.symbols("theta_a1 theta_a2", real=True)

    # [A1] unit moment for arbitrary two-axis rotations (spin length frozen).
    e = chart(frame, (th1, th2))
    zero(sp.simplify(e.dot(e) - 1))

    # [A2] transversality: orthogonal to the current moment at any angles,
    # tangent to the reference plane at theta = 0 (no longitudinal axis).
    d1 = sp.diff(e, th1)
    d2 = sp.diff(e, th2)
    zero(sp.simplify(e.dot(d1)))
    zero(sp.simplify(e.dot(d2)))
    zero(sp.simplify(n.dot(d1.subs({th1: 0, th2: 0}))))
    zero(sp.simplify(n.dot(d2.subs({th1: 0, th2: 0}))))
    tangent = sp.Matrix.hstack(
        sp.simplify(d1.subs({th1: 0, th2: 0})), sp.simplify(d2.subs({th1: 0, th2: 0}))
    )
    assert tangent.rank() == 2, tangent.rank()

    # [A3] Clairaut symmetry of the same-site second vertex (the symmetric
    # seagull of the declared exponential chart), on the anisotropy channel
    # with site b held at its reference direction.
    ea = chart(triad(X, Z), (th1, th2))
    nb = sp.Matrix([sp.Rational(4, 5), 0, sp.Rational(3, 5)])
    energy = (ea.T * GAMMA * nb)[0, 0]
    zero(sp.simplify(sp.diff(energy, th1, th2) - sp.diff(energy, th2, th1)))
    # Off-diagonal same-site second vertex vanishes in the composed chart;
    # the -n seagull is diagonal only.
    mixed = sp.simplify(sp.diff(energy, th1, th2).subs({th1: 0, th2: 0}))
    zero(mixed)
    diagonal = sp.simplify(sp.diff(energy, th1, 2).subs({th1: 0, th2: 0}))
    zero(diagonal + (Z.T * GAMMA * nb)[0, 0])

    # [A4] torque convention: tau = -gradient (ResponseSet FR-008).
    theta = sp.symbols("theta", real=True)
    e_tilt = sp.simplify(rodrigues_matrix(Y, theta) * Z)
    gradient = sp.diff(-J * e_tilt.dot(X), theta).subs(theta, 0)
    torque = -gradient
    zero(torque - J)
    print(
        "A PASS: e.e = 1; n.de/dtheta = 0 (2 tangent axes, no longitudinal); "
        "seagull symmetric; tau = -g"
    )
    return {"transversality": True, "seagull_clairaut": True, "torque_sign": True}


def test_pair_mapping():
    """[B] A-tensor -> pair parameters, symbolic frames, channel by channel."""
    print("[B] Pair mapping: mixed/onsite/gradient blocks from exact derivatives")
    n = sp.Matrix([sp.Rational(1, 5), -1, sp.Rational(7, 5)])
    n = sp.simplify(n / n.norm())
    ta = sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 3), -sp.Rational(1, 7)])
    ta = sp.simplify(ta - n * (ta.dot(n)))
    ta = sp.simplify(ta / ta.norm())
    tb = sp.Matrix([-sp.Rational(2, 5), sp.Rational(3, 5), sp.Rational(1, 9)])
    tb = sp.simplify(tb - n * (tb.dot(n)))
    tb = sp.simplify(tb / tb.norm())
    ua, ub = ta.cross(n), tb.cross(n)
    th_a, th_b = sp.symbols("theta_a theta_b", real=True)
    e_a = sp.simplify(rodrigues_matrix(ta, th_a) * n)
    e_b = sp.simplify(rodrigues_matrix(tb, th_b) * n)
    at0 = {th_a: 0, th_b: 0}
    dvec = sp.Matrix(sp.symbols("Dx Dy Dz", real=True))

    # Heisenberg channel: K_ab = -J (u_a.u_b), K_aa = +J (n.n) delta,
    # gradient = -J (u.n).
    energy = -J * e_a.dot(e_b)
    kab = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    zero(kab + J * ua.dot(ub))
    onsite_heis = sp.simplify(sp.diff(energy, th_a, 2).subs(at0))
    zero(onsite_heis - J)
    mixed_cross = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    zero(mixed_cross + J * ua.dot(ub))
    zero(sp.simplify(sp.diff(energy, th_a).subs(at0)) + J * ua.dot(n))

    # DMI channel at this (collinear-reference) chart pair: K_ab =
    # D.(u_a x u_b), onsite vanishes because n x n = 0, gradient_a =
    # D.(u_a x n).
    energy = dmi_term(dvec, e_a, e_b)
    kab = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    zero(kab - dvec.dot(ua.cross(ub)))
    zero(sp.simplify(sp.diff(energy, th_a, th_b).subs(at0)) - dvec.dot(ua.cross(ub)))
    zero(sp.simplify(sp.diff(energy, th_a).subs(at0)) - dvec.dot(ua.cross(n)))
    onsite = sp.simplify(sp.diff(energy, th_a, 2).subs(at0))
    zero(onsite)

    # Noncollinear reference (regression for the b-site DMI gradient sign and
    # the DMI onsite term): two DISTINCT site directions n_a != n_b with
    # independent site-local charts, differentiated directly.
    n_a2 = Z
    n_b2 = sp.Matrix([2, 1, 2]) / 3
    ta2 = sp.Matrix([3, 4, 0]) / 5
    tb2 = sp.Matrix([1, -2, 0]) / sp.sqrt(5)
    zero(sp.simplify(ta2.dot(n_a2)))
    zero(sp.simplify(tb2.dot(n_b2)))
    e_a2 = sp.simplify(rodrigues_matrix(ta2, th_a) * n_a2)
    e_b2 = sp.simplify(rodrigues_matrix(tb2, th_b) * n_b2)
    ua2 = sp.simplify(ta2.cross(n_a2))
    ub2 = sp.simplify(tb2.cross(n_b2))
    wrong_minus_right = sp.simplify(
        dvec.dot(ub2.cross(n_a2)) - dvec.dot(n_a2.cross(ub2))
    )
    assert wrong_minus_right != 0, wrong_minus_right
    # -> the wrong-sign row (D.(u_b x n_a)) is distinguishable from the
    #    corrected one (D.(n_a x u_b)) at this reference.

    energy = -J * e_a2.dot(e_b2)
    zero(sp.simplify(sp.diff(energy, th_a, th_b).subs(at0)) + J * ua2.dot(ub2))
    zero(sp.simplify(sp.diff(energy, th_a, 2).subs(at0)) - J * n_a2.dot(n_b2))
    zero(sp.simplify(sp.diff(energy, th_b).subs(at0)) + J * n_a2.dot(ub2))

    energy = dmi_term(dvec, e_a2, e_b2)
    zero(sp.simplify(sp.diff(energy, th_a, th_b).subs(at0)) - dvec.dot(ua2.cross(ub2)))
    zero(sp.simplify(sp.diff(energy, th_a).subs(at0)) - dvec.dot(ua2.cross(n_b2)))
    # The corrected b-site gradient: D.(n_a x u_b), differentiated directly.
    zero(sp.simplify(sp.diff(energy, th_b).subs(at0)) - dvec.dot(n_a2.cross(ub2)))
    # Both onsite diagonals equal -D.(n_a x n_b) at a noncollinear reference.
    onsite_aa2 = sp.simplify(sp.diff(energy, th_a, 2).subs(at0))
    onsite_bb2 = sp.simplify(sp.diff(energy, th_b, 2).subs(at0))
    zero(onsite_aa2 + dvec.dot(n_a2.cross(n_b2)))
    zero(onsite_bb2 + dvec.dot(n_a2.cross(n_b2)))
    assert sp.simplify(onsite_aa2.subs(zip((D_X, D_Y, D_Z), (1, 2, 3)))) != 0
    # Same-site off-diagonal DMI onsite stays zero (vanishing mixed vertex of
    # the composed chart), diagonal is -D.(n_a x n_b).
    th1, th2 = sp.symbols("theta_a1 theta_a2", real=True)
    e_composed = chart(triad(X, Z), (th1, th2))
    energy_m = dmi_term(dvec, e_composed, n_b2)
    zero(sp.simplify(sp.diff(energy_m, th1, th2).subs({th1: 0, th2: 0})))
    zero(
        sp.simplify(sp.diff(energy_m, th1, 2).subs({th1: 0, th2: 0}))
        + dvec.dot(Z.cross(n_b2))
    )

    # Anisotropy channel: K_ab = u_a.Gamma.u_b, K_aa = -delta (n.Gamma.n_b).
    gamma_full = sp.Matrix(
        [
            [
                sp.Symbol("Gxx", real=True),
                sp.Symbol("Gxy", real=True),
                sp.Symbol("Gxz", real=True),
            ],
            [
                sp.Symbol("Gxy", real=True),
                sp.Symbol("Gyy", real=True),
                sp.Symbol("Gyz", real=True),
            ],
            [
                sp.Symbol("Gxz", real=True),
                sp.Symbol("Gyz", real=True),
                sp.Symbol("Gzz", real=True),
            ],
        ]
    )
    energy = (e_a.T * gamma_full * e_b)[0, 0]
    kab = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    expected = (ua.T * gamma_full * ub)[0, 0]
    zero(kab - expected)
    onsite_aa = sp.simplify(sp.diff(energy, th_a, 2).subs(at0))
    zero(onsite_aa + (n.T * gamma_full * n)[0, 0])
    grad = sp.simplify(sp.diff(energy, th_a).subs(at0))
    zero(grad - (ua.T * gamma_full * n)[0, 0])

    # Traceless closure: onsite carries only Gamma_nn = -(Gamma_11 + Gamma_22).
    frame = triad(X, Z)
    e_cart = sp.simplify(rodrigues_matrix(frame[0], th_a) * frame[2])
    energy_t = (e_cart.T * GAMMA * frame[2])[0, 0]
    onsite_traceless = sp.simplify(sp.diff(energy_t, th_a, 2).subs({th_a: 0}))
    zero(onsite_traceless - (G_XX + G_YY))
    # Noncollinear onsite closure for Gamma: both diagonal second derivatives
    # carry -delta (n_a.Gamma.n_b) with n_a != n_b.
    energy = (e_a2.T * gamma_full * e_b2)[0, 0]
    zero(
        sp.simplify(sp.diff(energy, th_a, 2).subs(at0))
        + (n_a2.T * gamma_full * n_b2)[0, 0]
    )
    zero(
        sp.simplify(sp.diff(energy, th_b, 2).subs(at0))
        + (n_a2.T * gamma_full * n_b2)[0, 0]
    )
    print(
        "B PASS: K_ab = -J u.u + D.(u x u) + u.G.u; K_aa = -delta (n.G.n_b); "
        "onsite DMI = -delta D.(n_a x n_b); "
        "g_a = -J u.n_b + D.(u x n_b) + u.G.n_b; g_b = -J n_a.u + D.(n_a x u) + u.G.n_a"
    )
    return {
        "mixed": "K_ab = -J (u_a.u_b) + D.(u_a x u_b) + u_a.Gamma.u_b",
        "onsite": "K_aa = -delta (n_a.Gamma.n_b) - delta D.(n_a x n_b)",
        "gradient": "g_a = -J (u_a.n_b) + D.(u_a x n_b) + u_a.Gamma.n_b; "
        "g_b = -J (n_a.u_b) + D.(n_a x u_b) + u_b.Gamma.n_a",
    }


def test_zero_pitch_reduction():
    """[C] q = 0 collinear reduction to the story-011 identity."""
    print("[C] Zero-pitch reduction: story-011 stored ordered-pair identity")
    n = sp.Matrix([1, 0, 0])
    ta = sp.Matrix([0, 1, 0])
    tb = sp.Matrix([0, 1, 1]) / sp.sqrt(2)
    ua, ub = ta.cross(n), tb.cross(n)
    th_a, th_b = sp.symbols("theta_a theta_b", real=True)
    e_a = rodrigues_matrix(ta, th_a) * n
    e_b = rodrigues_matrix(tb, th_b) * n
    at0 = {th_a: 0, th_b: 0}

    # Common transverse direction: K_ab = -J exactly; DMI channel silent.
    energy = -J * e_a.dot(e_b)
    kab = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    tc = sp.Matrix([0, 1, 0])
    kab_common = sp.simplify(
        sp.diff(
            -J
            * sp.simplify(rodrigues_matrix(tc, th_a) * n).dot(
                sp.simplify(rodrigues_matrix(tc, th_b) * n)
            ),
            th_a,
            th_b,
        ).subs(at0)
    )
    zero(kab_common + J)
    dvec = sp.Matrix(
        [
            sp.Symbol("Dx", real=True),
            sp.Symbol("Dy", real=True),
            sp.Symbol("Dz", real=True),
        ]
    )
    dmi_common = sp.simplify(
        sp.diff(
            dmi_term(
                dvec,
                sp.simplify(rodrigues_matrix(tc, th_a) * n),
                sp.simplify(rodrigues_matrix(tc, th_b) * n),
            ),
            th_a,
            th_b,
        ).subs(at0)
    )
    zero(dmi_common)

    # Distinct transverse axes: J_pair = -K_ab/(u_a.u_b), stored = half.
    kab = sp.simplify(sp.diff(energy, th_a, th_b).subs(at0))
    dot = sp.simplify(ua.dot(ub))
    j_pair = sp.simplify(-kab / dot)
    zero(j_pair - J)
    alignment, transverse_dot, k_symbol = sp.symbols(
        "alignment transverse_dot K_ab", real=True, nonzero=True
    )
    stored_j = -k_symbol / (2 * alignment * transverse_dot)
    zero(sp.simplify(2 * alignment * transverse_dot * stored_j + k_symbol))
    zero(sp.simplify(2 * stored_j - (-k_symbol / (alignment * transverse_dot))))
    # common-axis form; in site-local charts alignment is encoded in the tilt
    # dot and the divisor is the LOCAL (u_a.u_b) only (asserted above).

    # AFM charts (convention pinned): n_b = -n_a with the SAME site-local
    # transverse axis t, so the LOCAL tilts are u_b = t x n_b = -u_a. The
    # transverse dot of the actual charts is -1: the alignment is already
    # encoded in (u_a.u_b), and the extraction divides by the LOCAL tilt dot
    # ONLY: j_stored = -K_ab / (2 u_a.u_b) = +J/2 for exact K_ab = -J(u_a.u_b)
    # = +J. Differentiating the actual two-site AFM charts (not a post-hoc
    # flip of e_b):
    n_afm = sp.Matrix([1, 0, 0])
    t_afm = sp.Matrix([0, 1, 0])
    e_a_afm = sp.simplify(rodrigues_matrix(t_afm, th_a) * n_afm)
    e_b_afm = sp.simplify(rodrigues_matrix(t_afm, th_b) * (-n_afm))
    u_a_afm = sp.simplify(t_afm.cross(n_afm))
    u_b_afm = sp.simplify(t_afm.cross(-n_afm))
    dot_afm = sp.simplify(u_a_afm.dot(u_b_afm))
    zero(dot_afm + 1)  # transverse_dot(u_a, u_b) = -1 exactly
    energy_afm = -J * e_a_afm.dot(e_b_afm)
    kab_afm = sp.simplify(sp.diff(energy_afm, th_a, th_b).subs(at0))
    zero(kab_afm + J * u_a_afm.dot(u_b_afm))  # K_ab = -J (u_a.u_b) = +J
    zero(kab_afm - J)
    j_stored_afm = sp.simplify(-kab_afm / (2 * dot_afm))
    zero(j_stored_afm - J / 2)
    # Equivalently: common-axis projection with EXPLICIT alignment
    # sign(n_a.n_b) = -1 and common transverse_dot u_c.u_c = +1:
    j_common_afm = sp.simplify(-kab_afm / (2 * sp.Integer(-1)))
    zero(j_common_afm - J / 2)

    # In-plane anisotropy contamination of the scalar extraction.
    gamma_full = sp.Matrix(
        [
            [
                sp.Symbol("Gxx", real=True),
                sp.Symbol("Gxy", real=True),
                sp.Symbol("Gxz", real=True),
            ],
            [
                sp.Symbol("Gxy", real=True),
                sp.Symbol("Gyy", real=True),
                sp.Symbol("Gyz", real=True),
            ],
            [
                sp.Symbol("Gxz", real=True),
                sp.Symbol("Gyz", real=True),
                sp.Symbol("Gzz", real=True),
            ],
        ]
    )
    energy_g = (e_a.T * gamma_full * e_b)[0, 0]
    k_ab_g = sp.simplify(sp.diff(energy_g, th_a, th_b).subs(at0))
    zero(k_ab_g - (ua.T * gamma_full * ub)[0, 0])
    # Common direction: extracted J_hat = J - u.Gamma.u (in-plane projection
    # onto the tilt direction u = t x n), from the combined pair energy.
    e_ac = sp.simplify(rodrigues_matrix(tc, th_a) * n)
    e_bc = sp.simplify(rodrigues_matrix(tc, th_b) * n)
    energy_gc = -J * e_ac.dot(e_bc) + (e_ac.T * gamma_full * e_bc)[0, 0]
    k_common_g = sp.simplify(sp.diff(energy_gc, th_a, th_b).subs(at0))
    j_hat = sp.simplify(-k_common_g)
    u_c = sp.simplify(tc.cross(n))  # tilt direction, unit and transverse
    zero(j_hat - (J - (u_c.T * gamma_full * u_c)[0, 0]))
    # Onsite closure: K_aa = +J (n.n) - Gamma_nn = J + Gamma_11 + Gamma_22
    # (traceless) in the chart basis, pair form with site b at its reference.
    e_c = sp.simplify(rodrigues_matrix(X, th_a) * Z)
    energy_ct = -J * e_c.dot(Z) + (e_c.T * GAMMA * Z)[0, 0]
    onsite_ct = sp.simplify(sp.diff(energy_ct, th_a, 2).subs({th_a: 0}))
    zero(onsite_ct - (J + G_XX + G_YY))
    print(
        "C PASS: J_stored = -K_ab/(2 transverse_dot) with the SITE-LOCAL tilt "
        "dot (AFM alignment encoded in the charts, u_b = -u_a; j_stored = +J/2); "
        "J_pair = 2 J_stored; "
        "J_hat = J - u.Gamma.u with onsite closure K_aa = J + Gamma_11 + Gamma_22"
    )
    return {"stored_ordered_pair_j": stored_j, "pair_once_j": sp.simplify(2 * stored_j)}


def named_vector(name):
    """Chart-named parameter-space directions for the null-space asserts."""
    signs = {
        "D_x": {D_X: 1},
        "D_y": {D_Y: 1},
        "D_z": {D_Z: 1},
        "Gamma_xz": {G_XZ: 1},
        "Gamma_yz": {G_YZ: 1},
        "longitudinal_A_zz": {J: 1, G_XX: 1, G_YY: 1},
        "trade_D_x_Gamma_yz": {D_X: -1, G_YZ: 1},
        "trade_D_y_Gamma_xz": {D_Y: 1, G_XZ: 1},
    }
    vec = [sp.Integer(0)] * 9
    for key, value in signs[name].items():
        vec[PARAMS.index(key)] = sp.Integer(value)
    return sp.Matrix(vec)


def assert_null(matrix, names):
    for name in names:
        zero(matrix * named_vector(name))


def full_map(fa, fb):
    """(hessian_rows, gradient_rows) of the pair model at one anchor."""
    ua, ub = tilts(fa), tilts(fb)
    return (
        mapping_rows(ua, ub, fa[2], fb[2]),
        gradient_rows(ua, ub, fa[2], fb[2]),
    )


def test_identifiability():
    """[D] Single-reference identifiability: exact ranks and null spaces."""
    print("[D] Identifiability for two sites: ranks, null spaces, completion")
    results = {}

    # [D0] scalar common-direction extraction (story-011 protocol): 1 number.
    fa, fb = anchor_collinear()
    ua = tilts(fa)
    u1 = ua[0]
    scalar_row = sp.Matrix(
        [
            [sp.simplify(-(u1.dot(u1)))]
            + [sp.Integer(0)] * 3
            + [coef((u1.T * GAMMA * u1)[0, 0], p) for p in PARAMS[4:]]
        ]
    )
    assert scalar_row.rank() == 1
    assert len(scalar_row.nullspace()) == 8
    for column in range(1, 4):
        zero(scalar_row[0, column])  # DMI invisible, reference axis included
    results["scalar_extraction"] = {
        "rank": 1,
        "null_dimension": 8,
        "dmi_invisible": True,
    }

    # [D1] mixed 2x2 block only (rows 0..3), collinear anchor, common frames.
    mixed, gradients = full_map(fa, fb)
    mixed_only = mixed[0:4, :]
    assert mixed_only.shape == (4, 9)
    rank1 = mixed_only.rank()
    print("   collinear mixed block: rank", rank1, "null", len(mixed_only.nullspace()))
    assert rank1 == 4
    assert len(mixed_only.nullspace()) == 5
    assert_null(mixed_only, ["D_x", "D_y", "Gamma_xz", "Gamma_yz", "longitudinal_A_zz"])
    results["collinear_mixed"] = {
        "rank": 4,
        "null_dimension": 5,
        "null": ["D_x", "D_y", "Gamma_xz", "Gamma_yz", "longitudinal A_zz <-> J trade"],
    }

    # [D2] onsite closure: full transverse Hessian (mixed + onsite entries).
    rank2 = mixed.rank()
    print("   collinear + onsite closure: rank", rank2, "null", len(mixed.nullspace()))
    assert rank2 == 5
    assert len(mixed.nullspace()) == 4
    assert_null(mixed, ["D_x", "D_y", "Gamma_xz", "Gamma_yz"])
    results["collinear_onsite_closure"] = {
        "rank": 5,
        "null_dimension": 4,
        "null": ["D_x", "D_y", "Gamma_xz", "Gamma_yz"],
    }

    # [D3] torque (gradient rows) at the collinear anchor. With the CORRECTED
    # b-site gradient (D.(n_a x u_b), the d/dtheta of E_D = +D.(e_a x e_b)),
    # the site-b torque rows carry the OPPOSITE sign of the site-a rows in the
    # DMI channel while the Gamma rows are symmetric: sums AND differences are
    # measured, so the former D <-> Gamma tradeoffs are broken and the
    # collinear anchor with torque determines all 9 parameters exactly.
    # (The defective row u_b x n_a reproduced the site-a sign and left the
    # two trades null — rank 7.)
    full_collinear = sp.Matrix.vstack(mixed, gradients)
    rank3 = full_collinear.rank()
    print("   collinear + torque: rank", rank3, "null", len(full_collinear.nullspace()))
    assert rank3 == 9
    assert len(full_collinear.nullspace()) == 0
    # Regression on the corrected convention: neither former trade direction
    # is annihilated any more.
    assert sp.simplify(full_collinear * named_vector("trade_D_x_Gamma_yz")) != 0
    assert sp.simplify(full_collinear * named_vector("trade_D_y_Gamma_xz")) != 0
    results["collinear_torque"] = {
        "rank": 9,
        "null_dimension": 0,
        "null": [],
        "note": "corrected g_b = D.(n_a x u_b) breaks the former D/Gamma trades",
    }

    # [D4] planar single-q reference (symbolic tilt along t1_a). The
    # corrected onsite DMI row (-D.(n_a x n_b) = -c Y here) makes D_y visible
    # through the Hessian, and the corrected torque breaks the residual
    # in-plane trade: the full map is exactly rank 9, null 0, for ARBITRARY
    # (c, s) — no blind direction survives at a planar reference.
    c, s = sp.symbols("c s", real=True)
    fa_p, fb_p = anchor_planar(c, s)
    rows_p, grads_p = full_map(fa_p, fb_p)
    full_planar = sp.Matrix.vstack(rows_p, grads_p)
    assert rows_p.rank() == 5
    assert full_planar.rank() == 9
    assert len(full_planar.nullspace()) == 0
    # Regression: the former blind trades of the defective rows are NOT in
    # the null space of the corrected map.
    assert sp.simplify(full_planar * named_vector("trade_D_x_Gamma_yz")) != 0
    # Flipped tilt (along t2_a): same exact full determination.
    fa_fl, fb_fl = anchor_flipped(c, s)
    rows_fl, grads_fl = full_map(fa_fl, fb_fl)
    full_flip = sp.Matrix.vstack(rows_fl, grads_fl)
    assert full_flip.rank() == 9
    assert len(full_flip.nullspace()) == 0
    assert sp.simplify(full_flip * named_vector("trade_D_y_Gamma_xz")) != 0
    # Exact 3-4-5 planar anchor: same exact full determination.
    fa_r, fb_r = anchor_planar(sp.Rational(4, 5), sp.Rational(3, 5))
    rows_r, grads_r = full_map(fa_r, fb_r)
    full_r = sp.Matrix.vstack(rows_r, grads_r)
    assert full_r.rank() == 9
    assert len(full_r.nullspace()) == 0
    assert sp.simplify(full_r * named_vector("trade_D_x_Gamma_yz")) != 0
    results["planar_spiral"] = {
        "rank_hessian_only": 5,
        "rank_with_torque": 9,
        "blind_tilt_along_t1": "none (corrected rows; former D_{t1_a} <-> "
        "Gamma_{t2_a n} trade broken)",
        "blind_tilt_along_t2": "none (corrected rows; former D_{t2_a} <-> "
        "Gamma_{t1_a n} trade broken)",
    }

    # [D5] multi-reference consistency sweep. Each full single-reference map
    # (collinear with torque, planar, either nonplanar) is already exactly
    # rank 9 under the corrected rows; the stacked references stay rank 9.
    fa_n1, fb_n1 = anchor_nonplanar(
        [sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)]
    )
    fa_n2, fb_n2 = anchor_nonplanar(
        [sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)]
    )
    np1 = sp.Matrix.vstack(*full_map(fa_n1, fb_n1))
    np2 = sp.Matrix.vstack(*full_map(fa_n2, fb_n2))
    full_col = sp.Matrix.vstack(mixed, gradients)
    assert sp.Matrix.vstack(full_col, full_r).rank() == 9
    assert sp.Matrix.vstack(np1, np2).rank() == 9
    assert sp.Matrix.vstack(full_col, np1, np2).rank() == 9
    results["multireference"] = {
        "collinear_plus_planar": 9,
        "two_nonplanar": 9,
        "collinear_plus_two_nonplanar": 9,
    }

    # [D6] the 3x3 Szilva pair block minus gauge: transverse restriction.
    probe_axes = [X, Y, Z]
    na, nb = fa_r[2], fb_r[2]

    def block_entry(i, j):
        va = sp.simplify(probe_axes[i].cross(na))
        vb = sp.simplify(probe_axes[j].cross(nb))
        return sp.simplify(
            -J * va.dot(vb) + dmi_term(DVEC, va, vb) + (va.T * GAMMA * vb)[0, 0]
        )

    a_block = sp.Matrix(3, 3, block_entry)
    # Chart covariance: an infinitesimal tilt about t_alpha is the linear
    # combination sum_i (t_alpha)_i xi_i of Cartesian-axis tilts, so the
    # transverse restriction is C_a A C_b^T with C[alpha, i] = (t_alpha)_i.
    ua_r, ub_r = tilts(fa_r), tilts(fb_r)
    c_a = sp.Matrix([[t[i] for i in range(3)] for t in fa_r[:2]])
    c_b = sp.Matrix([[t[i] for i in range(3)] for t in fb_r[:2]])
    restricted = sp.simplify(c_a * a_block * c_b.T)
    for alpha in range(2):
        for beta in range(2):
            predicted = (
                -J * ua_r[alpha].dot(ub_r[beta])
                + dmi_term(DVEC, ua_r[alpha], ub_r[beta])
                + (ua_r[alpha].T * GAMMA * ub_r[beta])[0, 0]
            )
            zero(restricted[alpha, beta] - predicted)
    # Gauge: the transverse chart measures exactly the C_a A C_b^T
    # projection (4 of 9 entries); the annihilator {M : C_a M C_b^T = 0}
    # is the 5-dimensional gauge, containing the longitudinal components.
    gauge_map = sp.Matrix(4, 9, lambda r, k: c_a[r // 2, k // 3] * c_b[r % 2, k % 3])
    gauge_basis = gauge_map.nullspace()
    assert len(gauge_basis) == 5, len(gauge_basis)
    gauge_m = sp.Matrix(3, 3, gauge_basis[0])
    zero(sp.simplify(c_a * gauge_m * c_b.T))
    # The onsite blocks supply exactly one gauge direction: n_a.Gamma.n_b
    # (up to the J delta pattern pinned by the Heisenberg part).
    results["szilva_block_gauge"] = {
        "entries": 9,
        "transverse_restrictions": 4,
        "gauge_dimension": 5,
        "onsite_closes": "n_a.Gamma.n_b",
    }
    print("D PASS: identifiability ranks and null spaces pinned (see results)")
    return results


def test_residual_definitions():
    """[E] Residual aggregate and dimensionless compatibility statistic."""
    print("[E] Residual definitions: weighted aggregate, floored statistic")
    # Pinned definitions. Data d collects the gauge-free K entries of one
    # pair (mixed 2x2 block, then both onsite 2x2 blocks); a layer is a
    # linear map M from pair parameters to that data layout; the map
    # representation is the exact least-squares projection P = M M^+.
    d_symbolic = sp.Matrix(sp.symbols("d0:12"))
    layer = sp.Matrix(sp.symbols("m0:12"))
    coefficient = (layer.T * d_symbolic)[0] / (layer.T * layer)[0]
    resid = sp.simplify(d_symbolic - coefficient * layer)
    r2 = (resid.T * resid)[0]
    # Floor-free statistic is exactly scale invariant: chi(2 d) = chi(d).
    scaled_resid2 = sp.expand(
        r2.subs({k: 2 * v for k, v in zip(list(d_symbolic), list(d_symbolic))})
    )
    zero(sp.simplify(sp.expand(scaled_resid2 - 4 * sp.expand(r2))))
    print(
        "E PASS: r_i = K_raw - K_map; R = ||d - P_layer d||_W; "
        f"chi = sqrt(r2 / max(d2_W, {K_FLOOR}^2)); flag chi > {CHI_MAX}; scale invariant"
    )
    return {
        "chi_max": CHI_MAX,
        "chi2_max": CHI2_MAX,
        "k_floor": K_FLOOR,
        "scale_invariant": True,
    }


def test_synthetic():
    """[F] Exact-Heisenberg, biquadratic and three-site synthetics."""
    print("[F] Synthetics: zero-residual Heisenberg; flagged biquadratic/three-site")
    results = {}
    angles_a = sp.symbols("ta1 ta2", real=True)
    angles_b = sp.symbols("tb1 tb2", real=True)
    fa, fb = anchor_planar(sp.Rational(4, 5), sp.Rational(3, 5))
    ea = chart(fa, angles_a)
    eb = chart(fb, angles_b)
    ua, ub = tilts(fa), tilts(fb)
    layer_iso = isotropic_layer(ua, ub, fa[2], fb[2])
    layer_sector = mapping_rows(ua, ub, fa[2], fb[2])

    # Mapping prediction of the exact Heisenberg pair, checked entrywise.
    e_heis = -J_SYN * (ea.T * eb)[0, 0]
    d_heis = raw_data(e_heis, angles_a, angles_b)
    predicted = mapping_rows(ua, ub, fa[2], fb[2]) * sp.Matrix([J_SYN] + [0] * 8)
    for index in (0, 1, 2, 3):
        zero(d_heis[index] - predicted[index])
    r2_iso, d2 = chi_squared(d_heis, layer_iso)
    zero(r2_iso)
    r2_sec, _ = chi_squared(d_heis, layer_sector)
    zero(r2_sec)
    j_hat = sp.simplify(((layer_iso.T * d_heis)[0]) / ((layer_iso.T * layer_iso)[0]))
    zero(j_hat - J_SYN)
    results["heisenberg"] = {
        "chi2_isotropic": 0,
        "chi2_sector": 0,
        "j_hat_recovered": True,
    }

    # Pure biquadratic: collinear anchor (invisible), noncollinear (flagged).
    e_biq = B_SYN * (ea.T * eb)[0, 0] ** 2
    d_biq = raw_data(e_biq, angles_a, angles_b)
    r2_iso_nc, d2_nc = chi_squared(d_biq, layer_iso)
    r2_sec_nc, _ = chi_squared(d_biq, layer_sector)

    # Floored statistic: chi = sqrt(r2 / max(d2, K_FLOOR^2)); scale invariant
    # exactly while the floor is inactive (d2 >> K_FLOOR^2).
    def floored_chi2(data, layer_map):
        r2_, d2_ = chi_squared(data, layer_map)
        return sp.simplify(r2_ / sp.Max(d2_, K_FLOOR**2))

    zero(
        sp.simplify(floored_chi2(2 * d_biq, layer_iso) - floored_chi2(d_biq, layer_iso))
    )
    assert r2_iso_nc > 0
    assert r2_iso_nc / d2_nc > CHI2_MAX, r2_iso_nc / d2_nc
    assert r2_sec_nc / d2_nc > CHI2_MAX
    print(
        f"F PASS: biquadratic at noncollinear reference: chi2_isotropic = "
        f"{r2_iso_nc}/{d2_nc} = {sp.nsimplify(r2_iso_nc / d2_nc)}, flagged"
    )
    results["biquadratic_noncollinear"] = {
        "chi2_isotropic": sp.nsimplify(r2_iso_nc / d2_nc),
        "chi2_sector": sp.nsimplify(r2_sec_nc / d2_nc),
        "flagged": True,
    }

    fa_c, fb_c = anchor_collinear()
    ea_c = chart(fa_c, angles_a)
    eb_c = chart(fb_c, angles_b)
    ua_c, ub_c = tilts(fa_c), tilts(fb_c)
    iso_c = isotropic_layer(ua_c, ub_c, fa_c[2], fb_c[2])
    sector_c = mapping_rows(ua_c, ub_c, fa_c[2], fb_c[2])
    d_biq_c = raw_data(B_SYN * (ea_c.T * eb_c)[0, 0] ** 2, angles_a, angles_b)
    r2_iso_c, d2_c = chi_squared(d_biq_c, iso_c)
    zero(r2_iso_c)  # invisible at the collinear anchor: exactly J-representable
    r2_sec_c, _ = chi_squared(d_biq_c, sector_c)
    zero(r2_sec_c)
    j_hat_c = sp.simplify(((iso_c.T * d_biq_c)[0]) / ((iso_c.T * iso_c)[0]))
    zero(j_hat_c + 2 * B_SYN)  # pure biquadratic masquerades as J = -2B(n.n)
    results["biquadratic_collinear"] = {
        "chi2_isotropic": 0,
        "chi2_sector": 0,
        "masquerades_as_j": sp.simplify(-2 * B_SYN),
        "flagged": False,
    }

    # Pure three-site term on a three-site reference: flagged through onsite.
    nc = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5), 0])
    fc = triad(sp.Matrix([sp.Rational(4, 5), sp.Rational(-3, 5), 0]), nc)
    ec = fc[2]
    e_three = K3_SYN * ((ea.T * eb)[0, 0]) * ((ea.T * ec)[0, 0])
    d_three = raw_data(e_three, angles_a, angles_b)
    r2_iso_3, d2_3 = chi_squared(d_three, layer_iso)
    r2_sec_3, _ = chi_squared(d_three, layer_sector)
    assert r2_iso_3 / d2_3 > CHI2_MAX
    assert r2_sec_3 / d2_3 > CHI2_MAX
    resid_three = sp.simplify(d_three - layer_sector * (layer_sector.pinv() * d_three))
    onsite_part = resid_three.extract(list(range(4, 12)), [0])
    assert any(sp.simplify(v) != 0 for v in onsite_part)
    print(
        f"F PASS: three-site at noncollinear reference: chi2_sector = "
        f"{sp.nsimplify(r2_sec_3 / d2_3)}, flagged (onsite-block violation)"
    )
    results["three_site"] = {
        "chi2_isotropic": sp.nsimplify(r2_iso_3 / d2_3),
        "chi2_sector": sp.nsimplify(r2_sec_3 / d2_3),
        "flagged_via_onsite": True,
    }

    # Reciprocity of every synthetic: K_ba = K_ab^T is built into raw_data.
    print(
        "F PASS: Heisenberg residual zero and J_hat = J; biquadratic invisible "
        "collinear / flagged noncollinear; three-site flagged via onsite blocks"
    )
    return results


def main():
    print(
        f"SymPy {sp.__version__}; spiral A-tensor -> Heisenberg mapping, exact algebra"
    )
    chart_results = test_chart_and_data_model()
    mapping = test_pair_mapping()
    zero_pitch = test_zero_pitch_reduction()
    identifiability = test_identifiability()
    residual = test_residual_definitions()
    synthetic = test_synthetic()
    for path in (Path(__file__), Path(__file__).with_suffix(".md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"sha256 {path.name}: {digest}")
    print("ALL ASSERTIONS GREEN")
    return {
        "chart": chart_results,
        "mixed_block_mapping": mapping["mixed"],
        "onsite_block_mapping": mapping["onsite"],
        "gradient_mapping": mapping["gradient"],
        "stored_ordered_pair_j": zero_pitch["stored_ordered_pair_j"],
        "pair_once_j": zero_pitch["pair_once_j"],
        "identifiability": identifiability,
        "compatibility_statistic": residual,
        "synthetic_flags": synthetic,
    }


if __name__ == "__main__":
    main()
