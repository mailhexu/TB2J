"""Assertion authority for story 013: reference dependence and comparison scope.

Run in mydev: python docs/sympy/spiral_reference_dependence.py
Paired derivation, conventions and limits: same-name .md file.
No production code is imported. Exact assertions never fall back to probes.

Scope (AC-005 / AC-007, ADR-003 SymPy first; normative mapping rows are those
of docs/sympy/spiral_heisenberg_mapping.py, story 012, and are re-derived
here only where consumed). This script pins, with exact SymPy assertions
(nonzero exit on any failure):

  [A] DEFINITION. Reference dependence is the difference
          delta J = J_hat(ref_1) - J_hat(ref_2)
      of the pair-once isotropic Heisenberg coefficient fitted by exact
      least squares on the transverse response of the SAME frozen model at
      TWO reference orientations of the SAME pair/site identity. It is NOT
      an E(q) fit and involves no change of ensemble, SCF, or pair.

  [B] HEISENBERG INVARIANCE. For the exact bilinear pair
          E = -J (e_a . e_b)
      the raw transverse Hessian/torque data at any two noncollinear
      references is represented with ZERO residual by the isotropic layer
      and the fitted coefficient is exactly J at both: delta J = 0
      symbolically and numerically. The exact bilinear Heisenberg J is
      reference invariant.

  [C] BIQUADRATIC DEPARTURE. For
          E = -J (e_a . e_b) - B (e_a . e_b)^2
      (e_a . e_b)^2 is NOT bilinear: at a collinear reference BOTH the
      isotropic projection AND the full nine-parameter joint fit (K +
      torque, the layer whose J component production exports as
      j_pair_once_eV) are exact and ABSORB B (fitted J = J + 2 B, zero
      residual: a single reference cannot distinguish the pair), while at
      noncollinear references the fit leaves an exact nonzero residual and
      the fitted J moves with the reference. The exact counterexample is
      derived by direct differentiation of E at two noncollinear
      orientations; the delta J of BOTH fitted quantities is asserted
      nonzero symbolically (exact rational multiples of B) and numerically.
      The isotropic J_hat is a distinct diagnostic, not the exported
      parameter.

  [D] THRESHOLD BOUNDARIES (approved in story 012): the dimensionless
      statistic chi^2 = r2 / d2 labels a representation
          compatible     when chi2 <= 1e-4   (= 1/10000),
          indeterminate  in the grey band,
          incompatible   when chi2 >= 2.5e-3  (= (1/20)^2, story-012 CHI2_MAX).
      The boundaries are pinned here and the section-[C] synthetics are
      asserted to land on the correct side.

  [E] COMPARISON SCOPE DECLARATION. Permitted: scalar pair-once J
      comparisons across reference orientations of the same frozen model,
      same pair/site identity, same SCF ensemble (provenance recorded).
      Forbidden: chart-local D/Gamma comparisons without the site-local
      frame transforms; cross-pitch or cross-SCF comparisons without
      matched provenance (story-004 ~0.1 eV/cell cross-SCF spread); any
      comparison of chart-local anisotropy projections as if scalar.

  [F] NONPLANAR THREE-SITE IDENTIFIABILITY. For a GENERIC noncoplanar
      three-site reference (exact rational directions with nonzero triple
      product), the existing story-012 pair K/torque rows (mixed + onsite
      closure + gradients) have EXACTLY rank 9 with trivial null space for
      EVERY pair, and the extracted J is identical from the mixed row, the
      onsite closure, the torque rows, and across all three pairs:
      reference-class independent. No planarity of the moment normals is
      required for identifiability: the planar-only gate can be removed.

Conventions are those of docs/sympy/spiral_heisenberg_mapping.py
(story 012): pair-once energy, transverse two-axis exponential chart,
torque = -gradient, DMI sign fixed by E_dmi = +D.(e_a x e_b).
Literature anchor: A. Szilva et al., Rev. Mod. Phys. 95, 035004 (2023)
(reference-state dependence of mapped parameters).
"""

import hashlib
from pathlib import Path

import sympy as sp

HALF = sp.Rational(1, 2)
X = sp.Matrix([1, 0, 0])
Y = sp.Matrix([0, 1, 0])
Z = sp.Matrix([0, 0, 1])

J, D_X, D_Y, D_Z, G_XX, G_XY, G_XZ, G_YZ, G_YY = sp.symbols(
    "J D_x D_y D_z Gamma_xx Gamma_xy Gamma_xz Gamma_yz Gamma_yy", real=True
)
PARAMS = [J, D_X, D_Y, D_Z, G_XX, G_XY, G_XZ, G_YZ, G_YY]
GAMMA = sp.Matrix([[G_XX, G_XY, G_XZ], [G_XY, G_YY, G_YZ], [G_XZ, G_YZ, -G_XX - G_YY]])

B = sp.Symbol("B", real=True)

# Synthetic strengths (story-012 conventions).
J_SYN = sp.Rational(1, 10)
B_SYN = sp.Rational(1, 100)

# Story-012/story-013 approved compatibility boundaries (pinned).
COMPATIBLE_CHI2_MAX = sp.Rational(1, 10**4)  # chi2 <= 1e-4  -> compatible
INCOMPATIBLE_CHI2_MIN = sp.Rational(1, 20) ** 2  # chi2 >= 2.5e-3 -> incompatible
assert COMPATIBLE_CHI2_MAX < INCOMPATIBLE_CHI2_MIN


def zero(value):
    """Require exact symbolic equality to zero, entrywise for a matrix."""
    reduced = sp.simplify(value)
    if isinstance(reduced, sp.MatrixBase):
        for entry in reduced:
            assert sp.simplify(entry) == 0, entry
    else:
        assert reduced == 0, reduced


def nonzero(value):
    """Require exact symbolic NON-equality to zero (scalar)."""
    reduced = sp.simplify(value)
    assert reduced != 0, reduced
    return reduced


def rodrigues_matrix(axis, angle):
    axis = sp.Matrix(axis)
    n = axis.norm()
    K = sp.Matrix(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return sp.eye(3) + sp.sin(angle) * K / n + (1 - sp.cos(angle)) * (K / n) ** 2


def triad(t1, n):
    """Right-handed orthonormal frame (t1, t2, n) with t1 x t2 = n."""
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


def coef(expression, parameter):
    return sp.simplify(sp.diff(sp.expand(expression), parameter))


def mapping_rows(ua, ub, na, nb):
    """Closed-form mixed + onsite rows over PARAMS (story-012 authority rows)."""
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
    """Torque = -gradient rows over PARAMS (story-012 corrected convention)."""
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
    """Differentiate the actual energy on the composed charts (12 entries)."""
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
    for angles in (angles_a, angles_b):
        zero(
            sp.simplify(
                sp.diff(energy, angles[0], angles[1])
                - sp.diff(energy, angles[1], angles[0])
            )
        )
    return sp.Matrix(data)


def chi_squared(data, layer):
    """(r2, d2) of the exact least-squares representation residual."""
    proj = layer * layer.pinv()
    resid = sp.simplify(data - proj * data)
    r2 = sp.simplify((resid.T * resid)[0])
    d2 = sp.simplify((data.T * data)[0])
    return sp.simplify(sp.cancel(r2)), sp.simplify(sp.cancel(d2))


def fit_coefficient(data, layer):
    """Exact least-squares scalar coefficient of a one-parameter layer."""
    return sp.simplify(sp.cancel(((layer.T * data)[0]) / ((layer.T * layer)[0])))


# ---------------------------------------------------------------------------
# Anchors (story-012 conventions). Site a always carries the Cartesian frame.
# ---------------------------------------------------------------------------
def anchor_collinear():
    fa = triad(X, Z)
    return fa, fa


def anchor_planar(c, s):
    """n_b = c X + s Z tilted along t1_a inside the x-z plane."""
    fa = triad(X, Z)
    nb = sp.Matrix([c, 0, s])
    fb = triad(Y, nb)
    return fa, fb


def anchor_nonplanar(components, t_along=Y):
    fa = triad(X, Z)
    nb = sp.Matrix(components)
    t1 = sp.simplify(t_along - nb * (t_along.dot(nb)))
    t1 = sp.simplify(t1 / t1.norm())
    fb = triad(t1, nb)
    return fa, fb


def generic_site_frame(n_components, t_along):
    """Site-local frame for a generic rational direction and tangent seed."""
    n = sp.Matrix(n_components)
    n = sp.simplify(n / n.norm())
    t1 = sp.simplify(sp.Matrix(t_along))
    t1 = sp.simplify(t1 - n * (t1.dot(n)))
    t1 = sp.simplify(t1 / t1.norm())
    return triad(t1, n)


def isotropic_fit_at(anchor, energy, angles_a, angles_b):
    """(J_hat, r2, d2) of the isotropic layer fit at one reference anchor."""
    fa, fb = anchor
    layer = isotropic_layer(tilts(fa), tilts(fb), fa[2], fb[2])
    data = raw_data(energy, angles_a, angles_b)
    return fit_coefficient(data, layer), *chi_squared(data, layer)


def gradient_data(energy, angles_a, angles_b):
    """First-derivative (gradient) entries at the reference, 8 entries."""
    zero_ = {t: 0 for t in list(angles_a) + list(angles_b)}
    rows = [
        sp.simplify(sp.diff(energy, angle).subs(zero_))
        for angle in list(angles_a) + list(angles_b)
    ]
    return sp.Matrix(rows)


def full_layer(fa, fb):
    """Full 9-parameter joint layer: 12 Hessian rows + 8 gradient rows."""
    ua, ub = tilts(fa), tilts(fb)
    return sp.Matrix.vstack(
        mapping_rows(ua, ub, fa[2], fb[2]), gradient_rows(ua, ub, fa[2], fb[2])
    )


def full_fit_at(anchor, energy, angles_a, angles_b):
    """Full 9-parameter joint fit (K + torque): (J_fit, r2, d2, params).

    This mirrors the production j_pair_once_eV provenance: the mapped J is
    the J component of the least-squares solution of the FULL nine-parameter
    layer, not the isotropic K-only projection.
    """
    layer = full_layer(anchor[0], anchor[1])
    data = sp.Matrix.vstack(
        raw_data(energy, angles_a, angles_b), gradient_data(energy, angles_a, angles_b)
    )
    params = layer.pinv() * data
    resid = sp.simplify(data - layer * params)
    r2 = sp.simplify((resid.T * resid)[0])
    d2 = sp.simplify(sp.cancel((data.T * data)[0]))
    return sp.simplify(sp.cancel(params[0])), sp.simplify(sp.cancel(r2)), d2, params


# ---------------------------------------------------------------------------
# [A]/[B]/[C]: two-orientation reference dependence.
# ---------------------------------------------------------------------------
ANGLES_A = sp.symbols("ta1 ta2", real=True)
ANGLES_B = sp.symbols("tb1 tb2", real=True)

REF_COLLINEAR = anchor_collinear()
REF_PLANAR = anchor_planar(sp.Rational(4, 5), sp.Rational(3, 5))
REF_NONPLANAR = anchor_nonplanar(
    [sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)]
)
REFERENCES = {
    "collinear": REF_COLLINEAR,
    "planar_345": REF_PLANAR,
    "nonplanar_rational": REF_NONPLANAR,
}


def energy_heisenberg(anchor):
    fa, fb = anchor
    ea, eb = chart(fa, ANGLES_A), chart(fb, ANGLES_B)
    return -J * (ea.T * eb)[0, 0]


def energy_biquadratic(anchor):
    fa, fb = anchor
    ea, eb = chart(fa, ANGLES_A), chart(fb, ANGLES_B)
    x = (ea.T * eb)[0, 0]
    return -J * x - B * x**2


def test_heisenberg_reference_invariance():
    """[B] Exact bilinear Heisenberg J is two-orientation invariant."""
    print("[B] Two-orientation invariance of the exact bilinear Heisenberg J")
    j_hats = {}
    for name, anchor in REFERENCES.items():
        j_hat, r2, d2 = isotropic_fit_at(
            anchor, energy_heisenberg(anchor), ANGLES_A, ANGLES_B
        )
        zero(j_hat - J)
        zero(r2)
        j_hats[name] = j_hat
    delta_j = sp.simplify(j_hats["collinear"] - j_hats["planar_345"])
    zero(delta_j)
    delta_j_2 = sp.simplify(j_hats["planar_345"] - j_hats["nonplanar_rational"])
    zero(delta_j_2)
    # Numerical form of the same statement (exact rationals evaluated).
    numeric = [float(j_hats[n].subs(J, J_SYN)) for n in REFERENCES]
    for value in numeric:
        assert value == float(J_SYN)
    print(
        "B PASS: J_hat = J at collinear, planar (3-4-5) and noncollinear "
        "references; delta J = 0 exactly (symbolic and numeric); residual 0"
    )
    return {"j_hat": j_hats, "delta_j": 0}


def test_biquadratic_reference_dependence():
    """[C] Biquadratic term: absorbed at one reference, exposed by two."""
    print("[C] Biquadratic departure: exact counterexample by direct differentiation")
    # [C1] Collinear reference: (n_a.n_b) = 1 constant; the Hessian of
    # -B (e_a.e_b)^2 in the transverse angles is EXACTLY isotropic-layer
    # representable, so the fit absorbs B with ZERO residual and
    # J_hat = J + 2 B (the pure-biquadratic masquerade J_hat = -2B of
    # story 012, shifted by the sign of the B term here).
    j_col, r2_col, d2_col = isotropic_fit_at(
        REF_COLLINEAR, energy_biquadratic(REF_COLLINEAR), ANGLES_A, ANGLES_B
    )
    zero(j_col - (J + 2 * B))
    zero(r2_col)

    # [C2] Planar 3-4-5 reference: direct differentiation of
    # E = -J x - B x^2 leaves an exact nonzero isotropic residual; J_hat
    # moves by an exact rational multiple of B.
    j_pla, r2_pla, d2_pla = isotropic_fit_at(
        REF_PLANAR, energy_biquadratic(REF_PLANAR), ANGLES_A, ANGLES_B
    )
    delta_j = sp.simplify(sp.cancel(j_col - j_pla))
    nonzero(delta_j)
    zero(delta_j.subs(B, 0))  # the whole shift is the B channel
    shift = sp.simplify(sp.cancel(delta_j / B))
    nonzero(shift)  # exact rational multiple of B
    nonzero(r2_pla)

    # [C3] Noncollinear-rational reference: same conclusion, distinct shift.
    j_non, r2_non, d2_non = isotropic_fit_at(
        REF_NONPLANAR, energy_biquadratic(REF_NONPLANAR), ANGLES_A, ANGLES_B
    )
    delta_j_2 = sp.simplify(sp.cancel(j_col - j_non))
    nonzero(delta_j_2)
    zero(delta_j_2.subs(B, 0))
    nonzero(sp.simplify(sp.cancel(delta_j - delta_j_2)))  # orientation-specific

    # [C4] FULL nine-parameter joint fit (K + torque rows), the layer whose
    # J component the production map exports as j_pair_once_eV. The same
    # reference dependence must appear there: at the collinear reference the
    # joint fit is still exact and STILL absorbs B (J_fit = J + 2 B); at the
    # noncollinear references J_fit moves by an exact rational multiple of B.
    jf_col, rf_col, _, _ = full_fit_at(
        REF_COLLINEAR, energy_biquadratic(REF_COLLINEAR), ANGLES_A, ANGLES_B
    )
    zero(jf_col - (J + 2 * B))
    zero(rf_col)
    jf_pla, rf_pla, df_pla, _ = full_fit_at(
        REF_PLANAR, energy_biquadratic(REF_PLANAR), ANGLES_A, ANGLES_B
    )
    jf_non, rf_non, df_non, _ = full_fit_at(
        REF_NONPLANAR, energy_biquadratic(REF_NONPLANAR), ANGLES_A, ANGLES_B
    )
    delta_jf = sp.simplify(sp.cancel(jf_col - jf_pla))
    nonzero(delta_jf)
    zero(delta_jf.subs(B, 0))
    shift_f = sp.simplify(sp.cancel(delta_jf / B))
    nonzero(shift_f)
    delta_jf_2 = sp.simplify(sp.cancel(jf_col - jf_non))
    nonzero(delta_jf_2)
    zero(delta_jf_2.subs(B, 0))
    nonzero(sp.simplify(sp.cancel(delta_jf - delta_jf_2)))
    # Sanity: with EXACT Heisenberg data the full joint fit is exact and
    # recovers J at every reference (the layer contains the truth).
    for name, anchor in REFERENCES.items():
        jf_h, rf_h, _, _ = full_fit_at(
            anchor, energy_heisenberg(anchor), ANGLES_A, ANGLES_B
        )
        zero(jf_h - J)
        zero(rf_h)

    # [C5] Numeric verdict at J_SYN, B_SYN: nonzero deltas (both the isotropic
    # diagnostic and the full joint fit), and the residual statistics on the
    # noncollinear references must NOT be labelable compatible
    # (chi2 > COMPATIBLE_CHI2_MAX with the pinned B/J ratio).
    subs_map = {J: J_SYN, B: B_SYN}
    delta_num = float(delta_j.subs(subs_map))
    assert delta_num != 0.0
    deltaf_num = float(delta_jf.subs(subs_map))
    assert deltaf_num != 0.0
    chi2_pla = sp.simplify(sp.cancel(r2_pla / d2_pla)).subs(subs_map)
    chi2_non = sp.simplify(sp.cancel(r2_non / d2_non)).subs(subs_map)
    chi2f_pla = sp.simplify(sp.cancel(rf_pla / df_pla)).subs(subs_map)
    chi2f_non = sp.simplify(sp.cancel(rf_non / df_non)).subs(subs_map)
    assert float(chi2_pla) > float(COMPATIBLE_CHI2_MAX)
    assert float(chi2_non) > float(COMPATIBLE_CHI2_MAX)
    assert float(chi2f_pla) > float(COMPATIBLE_CHI2_MAX)
    assert float(chi2f_non) > float(COMPATIBLE_CHI2_MAX)
    flagged_incompatible = float(chi2_pla) >= float(INCOMPATIBLE_CHI2_MIN) and float(
        chi2_non
    ) >= float(INCOMPATIBLE_CHI2_MIN)
    assert flagged_incompatible

    # The full joint fit spreads part of B into the D/Gamma channels
    # (single-reference absorption across the full hierarchy), so its
    # residual statistic is smaller: planar stays incompatible, the
    # noncollinear value 25/10326 sits in the grey band (indeterminate).
    # NEITHER full-fit statistic is labelable compatible.
    # Compatibility labels round-trip through the pinned boundaries.
    def label(chi2_value):
        if chi2_value <= COMPATIBLE_CHI2_MAX:
            return "compatible"
        if chi2_value >= INCOMPATIBLE_CHI2_MIN:
            return "incompatible"
        return "indeterminate"

    assert label(chi2_pla) == "incompatible"
    assert label(chi2_non) == "incompatible"
    assert label(chi2f_pla) == "incompatible"
    assert label(chi2f_non) == "indeterminate"
    assert float(chi2f_pla) >= float(INCOMPATIBLE_CHI2_MIN)
    assert COMPATIBLE_CHI2_MAX < float(chi2f_non) < float(INCOMPATIBLE_CHI2_MIN)
    print(
        f"C PASS: collinear J_hat = J + 2 B (absorbs B, residual 0); "
        f"delta J(collinear, planar) = {delta_j} = {shift} x B, "
        f"delta J(collinear, noncollinear) = {delta_j_2}; "
        f"numeric |delta J| = {abs(delta_num)}; "
        f"chi2_planar = {chi2_pla}, chi2_nonplanar = {chi2_non} -> both INCOMPATIBLE; "
        f"FULL joint fit: J_fit moves identically "
        f"(delta J_fit(col,pla) = {delta_jf} = {shift_f} x B, "
        f"delta J_fit(col,non) = {delta_jf_2}; numeric |delta J_fit| = "
        f"{abs(deltaf_num)}); chi2_fit planar = {chi2f_pla} (INCOMPATIBLE), "
        f"nonplanar = {chi2f_non} (INDETERMINATE grey band)"
    )
    return {
        "j_hat_collinear": j_col,
        "j_hat_planar": j_pla,
        "j_hat_nonplanar": j_non,
        "delta_j_planar": delta_j,
        "delta_j_nonplanar": delta_j_2,
        "chi2_planar": chi2_pla,
        "chi2_nonplanar": chi2_non,
        "labels": {
            k: label(v) for k, v in (("planar", chi2_pla), ("nonplanar", chi2_non))
        },
        "j_fit_collinear": jf_col,
        "j_fit_planar": jf_pla,
        "j_fit_nonplanar": jf_non,
        "delta_j_fit_planar": delta_jf,
        "delta_j_fit_nonplanar": delta_jf_2,
        "chi2_fit_planar": chi2f_pla,
        "chi2_fit_nonplanar": chi2f_non,
        "labels_full_fit": {
            k: label(v) for k, v in (("planar", chi2f_pla), ("nonplanar", chi2f_non))
        },
    }


def test_threshold_boundaries():
    """[D] Pin the approved chi2 boundaries and their exact rational forms."""
    print("[D] Approved compatibility boundaries (story-012 approval, pinned)")
    zero(COMPATIBLE_CHI2_MAX - sp.Rational(1, 10000))
    zero(INCOMPATIBLE_CHI2_MIN - sp.Rational(1, 400))
    # Exact-Heisenberg data lands compatible (chi2 = 0 <= 1e-4).
    _, r2_b, d2_b = isotropic_fit_at(
        REF_PLANAR, energy_heisenberg(REF_PLANAR), ANGLES_A, ANGLES_B
    )
    zero(r2_b)
    assert sp.Rational(0) <= COMPATIBLE_CHI2_MAX
    # Boundary ordering and non-overlap of the grey band.
    grey_width = INCOMPATIBLE_CHI2_MIN - COMPATIBLE_CHI2_MAX
    nonzero(grey_width)
    print(
        f"D PASS: compatible chi2 <= {COMPATIBLE_CHI2_MAX} (1e-4); "
        f"incompatible chi2 >= {INCOMPATIBLE_CHI2_MIN} (2.5e-3); "
        f"grey band width {grey_width}"
    )
    return {
        "compatible_max": COMPATIBLE_CHI2_MAX,
        "incompatible_min": INCOMPATIBLE_CHI2_MIN,
        "grey_band": (COMPATIBLE_CHI2_MAX, INCOMPATIBLE_CHI2_MIN),
    }


def test_comparison_scope():
    """[E] Comparison-scope declaration (machine-readable, asserted)."""
    print("[E] Comparison scope declaration")
    scope = {
        "reference_dependence_definition": (
            "delta J = J_hat(ref_1) - J_hat(ref_2), pair-once isotropic J, "
            "same frozen model, same pair/site identity, two reference "
            "orientations only; NOT an E(q) fit"
        ),
        "permitted": [
            "scalar pair-once J across reference orientations of the same "
            "frozen model and same SCF ensemble, with recorded provenance",
        ],
        "forbidden": [
            "chart-local D/Gamma comparisons without the site-local frame "
            "transforms (they are projections onto per-site tilt charts)",
            "cross-pitch or cross-SCF comparisons without matched provenance "
            "(story-004 ~0.1 eV/cell cross-SCF spread forbids unmatched "
            "cross-SCF exchange differences as converged results)",
            "presenting an incompatible response as an exact Heisenberg fit",
        ],
        "statistic": "chi2 = r2 / max(d2, k_floor^2) with story-012 floor",
        "boundaries": {
            "compatible_max_chi2": float(COMPATIBLE_CHI2_MAX),
            "incompatible_min_chi2": float(INCOMPATIBLE_CHI2_MIN),
        },
    }
    # The declared boundaries equal the pinned rationals.
    assert scope["boundaries"]["compatible_max_chi2"] == 1e-4
    assert scope["boundaries"]["incompatible_min_chi2"] == 2.5e-3
    print("E PASS: scope declared (see .md for the normative text)")
    return scope


def test_nonplanar_three_site():
    """[F] Generic noncoplanar 3-site: rank-9 pair maps, J class-independent."""
    print("[F] Generic noncoplanar three-site reference: rank 9 per pair")
    # Generic exact rational normals with NONZERO triple product (asserted):
    # the Fe4-style situation where Mulliken moment normals are not exactly
    # coplanar. Frames use distinct generic tangent seeds per site.
    normals = [
        ([2, 1, 2], [1, 0, 3]),
        ([1, 2, 2], [0, 1, 3]),
        ([2, 2, 1], [1, 3, 0]),
    ]
    nvecs = [sp.Matrix(c) for c, _ in normals]
    nvecs = [sp.simplify(n / n.norm()) for n in nvecs]
    triple = sp.simplify(nvecs[0].dot(nvecs[1].cross(nvecs[2])))
    nonzero(triple)
    print(f"   triple product of site normals = {triple} != 0 (noncoplanar)")

    frames = [
        generic_site_frame(c, t) for (c, _), t in zip(normals, [t for _, t in normals])
    ]
    results = {}
    extracted = []
    for i in range(3):
        for j in range(i + 1, 3):
            fa, fb = frames[i], frames[j]
            ua, ub = tilts(fa), tilts(fb)
            na, nb = fa[2], fb[2]
            rows = sp.Matrix.vstack(
                mapping_rows(ua, ub, na, nb), gradient_rows(ua, ub, na, nb)
            )
            rank = rows.rank()
            assert rank == 9, (i, j, rank)
            assert len(rows.nullspace()) == 0, (i, j)
            # J extraction with exact Heisenberg data on this pair's chart:
            ea, eb = chart(fa, ANGLES_A), chart(fb, ANGLES_B)
            energy = -J * (ea.T * eb)[0, 0]
            data = raw_data(energy, ANGLES_A, ANGLES_B)
            # (i) single mixed entry against its tilt dot;
            j_mixed_single = sp.simplify(-data[0] / sp.simplify(ua[0].dot(ub[0])))
            zero(j_mixed_single - J)
            # (ii) full isotropic least-squares layer;
            layer = isotropic_layer(ua, ub, na, nb)
            zero(fit_coefficient(data, layer) - J)
            zero(chi_squared(data, layer)[0])
            # (iii) onsite closure row (K_aa = +J (n_a.n_b) delta);
            j_onsite = sp.simplify(data[4] / sp.simplify(na.dot(nb)))
            zero(j_onsite - J)
            # (iv) torque rows: differentiate E once (torque = -gradient) and
            # match the story-012 gradient row g_a,alpha = -J (u_a.n_b).
            grads = gradient_rows(ua, ub, na, nb)
            gradient_num = sp.simplify(
                sp.diff(energy, ANGLES_A[0]).subs(
                    {t: 0 for t in list(ANGLES_A) + list(ANGLES_B)}
                )
            )
            zero(gradient_num - (grads * sp.Matrix([J] + [0] * 8))[0])
            j_torque = sp.simplify(-gradient_num / sp.simplify(ua[0].dot(nb)))
            zero(j_torque - J)
            extracted.append((i, j, j_mixed_single))
            results[f"pair_{i}{j}"] = {"rank_full": 9, "null_dimension": 0}
    # Reference-class independence: the SAME J is extracted from every pair,
    # and the per-pair two-orientation invariance of [B] holds with both
    # sites' frames nonplanar (no Cartesian/collinear anchor anywhere).
    # NOTE: identifiability is asserted PAIR-LOCAL only; no claim is made
    # about a joint stacked multi-pair map.
    for i, j, j_pair in extracted:
        zero(j_pair - J)
    print(
        "F PASS: every noncoplanar pair map has exact rank 9, null space 0; "
        "J identical from mixed, isotropic fit, onsite closure and torque "
        "rows, and across all three pairs: J extraction is reference-class "
        "independent; the planar-only gate is not required by identifiability"
    )
    return results


def main():
    print(
        f"SymPy {sp.__version__}; spiral reference dependence and comparison "
        "scope, exact algebra"
    )
    invariance = test_heisenberg_reference_invariance()
    biquadratic = test_biquadratic_reference_dependence()
    boundaries = test_threshold_boundaries()
    scope = test_comparison_scope()
    nonplanar = test_nonplanar_three_site()
    for path in (Path(__file__), Path(__file__).with_suffix(".md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"sha256 {path.name}: {digest}")
    print("ALL ASSERTIONS GREEN")
    return {
        "heisenberg_invariance": invariance,
        "biquadratic_departure": biquadratic,
        "boundaries": boundaries,
        "comparison_scope": scope,
        "nonplanar_three_site": nonplanar,
    }


if __name__ == "__main__":
    main()
