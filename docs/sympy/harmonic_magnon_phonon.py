"""Story 011 / ADR-005: harmonic mixed spin/lattice quadratic problem.

Controlled one-magnon/one-phonon model
--------------------------------------

With unit-commutator Holstein--Primakoff magnon ``b`` (bare energy
``alpha``) and a mass-normalized optical phonon coordinate ``a`` (bare
energy ``beta``), both in eV, the controlled magnetoelastic quadratic
Hamiltonian keeps the counter-rotating terms:

    H = alpha b^+ b + beta a^+ a + g (b + b^+)(a + a^+),   g real.

Bosonic BdG/Colpa convention (identical to docs/sympy/site_boson_modes.md):
with the Nambu spinor ``Psi = (b, a, b^+, a^+)^T``,

    H = (1/2) Psi^+ M Psi + const,
    M = [[A, B], [B^T*, A^T*]],  A = [[alpha, g], [g, beta]],
    B = [[0, g], [g, 0]],

so A is Hermitian, B is symmetric, and M is Hermitian.  The metric is
``G = diag(1, 1, -1, -1)``.  Positive physical mode vectors ``t`` are the
positive-norm eigenvectors of the generalized problem ``M t = E G t``
with ``t^+ G t = 1``; the paraunitary ``T`` collects them (and their
negative-norm partners) and satisfies ``T^+ G T = G`` and
``T^+ M T = diag(E_-, E_+, E_+, E_-)`` (ascending positive-energy order).

Closed-form energies and stability
----------------------------------

The characteristic polynomial is quadratic in ``E^2``:

    E_+-^2 = (alpha^2 + beta^2)/2
             +/- sqrt( ((alpha^2 - beta^2)/2)^2 + 4 alpha beta g^2 ).

Both branches are real and positive exactly when the M matrix is positive
definite, which for this model is the stability condition

    4 g^2 < alpha beta.

``E_-^2`` is strictly decreasing in ``g^2`` and vanishes exactly at
``g^2 = alpha beta / 4``; above threshold the harmonic expansion is
unstable.  The solver must flag, never regularize, this crossing.

Observable weights
------------------

Solving ``M t = E G t`` per positive branch and using the normalization
``t^+ G t = 1``, the squared amplitudes of the physical column on the
(b, b^+) and (a, a^+) coordinate pairs are, with
``r = (E^2 - alpha^2)/(2 g beta)`` and the sector amplitudes
``x_- = (1, r)``, ``x_+ = (alpha, beta r)/E`` so that
``u = (x_+ + x_-)/sqrt(2)``, ``v = (x_+ - x_-)/sqrt(2)``:

    W_m(E) = |u_b|^2 + |v_b|^2 = 1 + (alpha/E)^2,
    W_p(E) = |u_a|^2 + |v_a|^2 = r^2 (1 + (beta/E)^2).

Weight shares are normalization independent:

    w_m(E) = W_m / (W_m + W_p),   w_p(E) = 1 - w_m(E).

At resonance (alpha = beta) the exchange alpha <-> beta maps a mode to the
same energy with w_m <-> w_p, so every mode has w_m = w_p = 1/2; for
g -> 0 the magnon-like branch has w_m = 1 and the phonon-like branch
w_m = 0 (complete weight transfer across the detuning sweep).

General assembled problem (implemented in TB2J/magnon_phonon/harmonic.py)
-------------------------------------------------------------------------

For n magnetic sites and n_ph displacement coordinates at fractional point
k, with the magnon Hessian ``Hq(k) = [[A_b, B_b], [B_b^+, A_b^*]]`` (Story
006 path), a Hermitian positive-semidefinite mass-weighted phonon Hessian
``D(k)`` (eigenvalues = phonon energies^2 in eV^2), and a rectangular
mixed block ``Gamma(k)`` in eV coupling transverse-spin and displacement
coordinates, the assembled Nambu Hessian is

    A = [[A_b, Gamma], [Gamma^+, sqrt(D)]],
    B = [[B_b, Gamma], [Gamma^T, 0]],

    M = [[A, B], [B^T*, A^T*]].

B symmetric holds by construction (cross-species pairing is
``Gamma``/``Gamma^T``); the same Colpa/paraunitary machinery then applies.
A zero Gamma leaves the bare magnon and phonon spectra and unit species
weights; that is the symmetry-forbidden control.  This script also runs
the implemented solver over a detuning sweep and checks the derived
energies and weight shares.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sympy as sp

alpha, beta, gam = sp.symbols("alpha beta gamma", positive=True, real=True)
E, E2 = sp.symbols("E E2", positive=True, real=True)


def controlled_bdg_matrix(a, b, g):
    """Symbolic controlled-model Nambu Hessian and metric."""
    A = sp.Matrix([[a, g], [g, b]])
    B = sp.Matrix([[0, g], [g, 0]])
    M = sp.Matrix(sp.BlockMatrix([[A, B], [B.T, A]]))
    G = sp.diag(1, 1, -1, -1)
    return A, B, M, G


def derive_energies():
    """Assert the closed-form positive-mode energies of the controlled model."""
    _, _, M, G = controlled_bdg_matrix(alpha, beta, gam)
    char = sp.expand(sp.det(M - E * G))
    # even powers only: quadratic in E^2
    char_E2 = sp.expand(char.subs(E**2, E2))
    poly = sp.Poly(char_E2, E2)
    assert poly.degree() == 2, char_E2
    assert (
        sp.simplify(
            char_E2
            - (
                E2**2
                - E2 * (alpha**2 + beta**2)
                + alpha**2 * beta**2
                - 4 * alpha * beta * gam**2
            )
        )
        == 0
    )
    root = sp.solve(sp.Eq(char_E2, 0), E2)
    disc = ((alpha**2 - beta**2) / 2) ** 2 + 4 * alpha * beta * gam**2
    disc_alt = alpha**4 - 2 * alpha**2 * beta**2 + 16 * alpha * beta * gam**2 + beta**4
    assert sp.simplify(4 * disc - disc_alt) == 0
    Ep2 = (alpha**2 + beta**2) / 2 + sp.sqrt(disc_alt) / 2
    Em2 = (alpha**2 + beta**2) / 2 - sp.sqrt(disc_alt) / 2
    assert {sp.simplify(r) for r in root} == {sp.simplify(Ep2), sp.simplify(Em2)}, root
    # E_+^2 > E_-^2 for all positive parameters; E_-^2 > 0 is the
    # stability condition asserted in derive_stability.
    assert sp.simplify(Ep2 - Em2 - sp.sqrt(disc_alt)) == 0
    print("[ok] E_+-^2 = (a^2+b^2)/2 +/- sqrt(((a^2-b^2)/2)^2 + 4abg^2)")
    return Ep2, Em2, disc


def derive_stability(Em2, disc):
    """Assert the stability threshold 4 g^2 < alpha beta."""
    # E_-^2 vanishes exactly at the threshold and decreases with g^2.
    at_threshold = sp.simplify(Em2.subs(gam**2, alpha * beta / 4)).subs(
        sp.sqrt(alpha**4 + 2 * alpha**2 * beta**2 + beta**4), alpha**2 + beta**2
    )
    assert at_threshold == 0, at_threshold
    s = sp.symbols("s", positive=True)
    Em2_s = sp.expand(Em2.subs(gam**2, s))
    dEm2_ds = sp.simplify(sp.diff(Em2_s, s))
    disc_s = sp.expand(disc.subs(gam**2, s))
    assert sp.simplify(dEm2_ds + 4 * alpha * beta / (2 * sp.sqrt(disc_s))) == 0
    # det M = (alpha beta - 4 g^2) * alpha beta: positive definiteness gate
    _, _, M, _ = controlled_bdg_matrix(alpha, beta, gam)
    assert sp.simplify(sp.det(M) - (alpha * beta - 4 * gam**2) * alpha * beta) == 0
    print("[ok] stability: E_-^2 > 0 iff 4 g^2 < alpha beta (= det M > 0)")
    return dEm2_ds


def derive_weights():
    """Assert the mode amplitude weights and their shares."""
    _, _, M, G = controlled_bdg_matrix(alpha, beta, gam)
    u1, u2, v1, v2 = sp.symbols("u1 u2 v1 v2", real=True)
    # Physical amplitudes from the sector solution:
    # x_- = (1, r), x_+ = (alpha, beta r)/E, u=(x_+ + x_-)/sqrt2, v=(x_+ - x_-)/sqrt2
    r = (E**2 - alpha**2) / (2 * gam * beta)
    u_b, u_a = (alpha / E + 1) / sp.sqrt(2), (beta * r / E + r) / sp.sqrt(2)
    v_b, v_a = (alpha / E - 1) / sp.sqrt(2), (beta * r / E - r) / sp.sqrt(2)
    Wm = sp.simplify(u_b**2 + v_b**2)
    Wp = sp.simplify(u_a**2 + v_a**2)
    assert sp.simplify(Wm - (1 + alpha**2 / E**2)) == 0, Wm
    assert sp.simplify(Wp - r**2 * (1 + beta**2 / E**2)) == 0, Wp
    # The amplitudes solve the generalized problem with unit metric norm
    # (metric norm is normalization dependent; fix it through G-norm = 1).
    psi = sp.Matrix([u_b, u_a, v_b, v_a])
    resid = sp.simplify(
        (M * psi - E * G * psi).subs(
            E,
            sp.sqrt(
                (alpha**2 + beta**2) / 2
                + sp.sqrt(
                    disc_alt := alpha**4
                    - 2 * alpha**2 * beta**2
                    + 16 * alpha * beta * gam**2
                    + beta**4
                )
                / 2
            ),
        )
    )
    assert resid.is_zero_matrix, resid
    gnorm = sp.simplify((psi.T * G * psi)[0])
    assert gnorm != 0
    disc_alt = alpha**4 - 2 * alpha**2 * beta**2 + 16 * alpha * beta * gam**2 + beta**4
    Epos = sp.sqrt((alpha**2 + beta**2) / 2 + sp.sqrt(disc_alt) / 2)
    psi_pos = psi.subs(E, Epos)
    norm_pos = sp.simplify(gnorm.subs(E, Epos))
    psi_unit = sp.simplify(psi_pos / sp.sqrt(norm_pos))
    assert sp.simplify((psi_unit.T * G * psi_unit)[0] - 1) == 0
    assert sp.simplify((psi_unit.T * M * psi_unit)[0] - Epos) == 0
    share_m = sp.simplify(Wm / (Wm + Wp))
    # Resonance: alpha = beta forces share 1/2 for both branches.
    disc_res = 16 * alpha**2 * gam**2  # disc_alt at alpha = beta
    Ep2_res = alpha**2 + sp.sqrt(disc_res) / 2
    Em2_res = alpha**2 - sp.sqrt(disc_res) / 2
    resonance_share = sp.simplify(share_m.subs(beta, alpha).subs(E**2, Ep2_res))
    assert resonance_share == sp.Rational(1, 2), resonance_share
    resonance_share_minus = sp.simplify(share_m.subs(beta, alpha).subs(E**2, Em2_res))
    assert resonance_share_minus == sp.Rational(1, 2), resonance_share_minus
    # Zero-coupling limit: magnon-like branch pure magnon (share -> 1) as
    # g -> 0 on the upper branch E_+ -> alpha.
    share_plus_limit = sp.simplify(
        sp.limit(share_m.subs(E, sp.sqrt(alpha**2 + gam**2)), gam, 0, "+")
    )
    assert share_plus_limit == 1, share_plus_limit
    print(
        "[ok] weights W_m = 1 + (a/E)^2, W_p = r^2 (1 + (b/E)^2); resonance share 1/2"
    )
    return share_m


def numeric_certificate():
    """Numeric paraunitary/normalization checks at rational model points."""
    for a, b, g in [
        (2, 1, sp.Rational(1, 10)),
        (3, 1, sp.Rational(2, 5)),
        (2, 2, sp.Rational(1, 2)),
        (sp.Rational(7, 3), sp.Rational(5, 2), sp.Rational(3, 10)),
    ]:
        av, bv, gv = float(a), float(b), float(g)
        A = np.array([[av, gv], [gv, bv]])
        B = np.array([[0.0, gv], [gv, 0.0]])
        M = np.block([[A, B], [B.T, A.conj()]])
        G = np.diag([1.0, 1.0, -1.0, -1.0])
        K = np.linalg.cholesky(M)
        w, x = np.linalg.eigh(K.conj().T @ G @ K)
        assert np.all(w[:2] < 0) and np.all(w[2:] > 0), w
        pos, neg = w[2:], w[:2]
        psi = np.linalg.solve(K.conj().T, x) * np.sqrt(np.abs(w))[None, :]
        T = np.column_stack([psi[:, 2:], psi[:, :2][:, ::-1]])  # (E+, E-; partners)
        # paraunitarity and diagonalization
        assert np.allclose(T.conj().T @ G @ T, G, atol=1e-11)
        assert np.allclose(
            T.conj().T @ M @ T,
            np.diag([pos[0], pos[1], abs(neg[1]), abs(neg[0])]),
            atol=1e-10,
        )
        disc = ((av**2 - bv**2) / 2) ** 2 + 4 * av * bv * gv**2
        derived = np.sqrt(
            [(av**2 + bv**2) / 2 + np.sqrt(disc), (av**2 + bv**2) / 2 - np.sqrt(disc)]
        )
        assert np.allclose(np.sort(pos), np.sort(derived), atol=1e-11), (pos, derived)
        # weight shares vs closed form
        for Ei, wmi in zip(np.sort(pos), [0, 1]):
            r = (Ei**2 - av**2) / (2 * gv * bv)
            Wm = 1 + (av / Ei) ** 2
            Wp = r**2 * (1 + (bv / Ei) ** 2)
            t = T[:, wmi]
            wm_num = abs(t[0]) ** 2 + abs(t[2]) ** 2
            wp_num = abs(t[1]) ** 2 + abs(t[3]) ** 2
            assert abs(wm_num / (wm_num + wp_num) - Wm / (Wm + Wp)) < 1e-11
    print(
        "[ok] numeric certificate: paraunitarity, diagonalization, energies, weight shares"
    )


def derive_hessian_to_coupling():
    """Assert the physical mixed-Hessian to bosonic energy conversion.

    The typed input is the second derivative of the energy with respect to
    (i) a local transverse spin component pair, expressed through the
    complex amplitude ``s_i = S_i1 + i S_i2``, and (ii) a displacement
    coordinate.  With unit-commutator HP bosons and the mass-weighted
    normal coordinate of a mode with energy ``omega``,

        s = sqrt(S/2) (b + b^+),        w = e (a + a^+)/sqrt(2 omega),

    so an energy term ``G s w`` becomes a bosonic coupling

        Gamma = G sqrt(S/2) e / sqrt(2 omega),

    with the spin-length and phonon zero-point factors applied exactly
    once.  For a Cartesian displacement basis, ``w = sqrt(m) u`` converts
    first: ``G_massweighted = G_cartesian / sqrt(m)``.
    """
    S, m = sp.symbols("S m", positive=True)
    b, bp, a, ap = sp.symbols("b b^+ a a^+", commutative=False)
    G, omega = sp.symbols("G omega", positive=True)

    s = sp.sqrt(S / 2) * (b + bp)
    w = (a + ap) / sp.sqrt(2 * omega)
    term = sp.expand(G * s * w)
    Gamma = G * sp.sqrt(S / 2) / sp.sqrt(2 * omega)
    collected = sp.simplify(term / (b * a + b * ap + bp * a + bp * ap))
    assert sp.simplify(collected - Gamma) == 0, collected

    # controlled-model round trip: choosing G = g sqrt(2 omega)/sqrt(S/2)
    # returns exactly the bosonic coupling g of the derived Hamiltonian.
    g = sp.symbols("g", positive=True)
    round_trip = sp.simplify(
        (g * sp.sqrt(2 * omega) / sp.sqrt(S / 2)) * sp.sqrt(S / 2) / sp.sqrt(2 * omega)
    )
    assert round_trip == g, round_trip

    # Cartesian variant: u = w / sqrt(m), so the Cartesian-basis derivative
    # converts to the mass-weighted derivative as G_mw = G_cart / sqrt(m).
    Gc = sp.symbols("Gc", positive=True)
    u = sp.Symbol("u", positive=True)
    bc, bpc, ac, apc = sp.symbols("bc bpc ac apc", positive=True, commutative=True)
    s_c = sp.sqrt(S / 2) * (bc + bpc)
    w_c = (ac + apc) / sp.sqrt(2 * omega)
    Gmw = Gc / sp.sqrt(m)
    term_from_cart = sp.expand((Gc * s_c * u).subs(u, w_c / sp.sqrt(m)))
    term_from_mw = sp.expand(Gmw * s_c * w_c)
    assert sp.simplify(term_from_cart - term_from_mw) == 0
    print(
        "[ok] conversion Gamma = G sqrt(S/2) e / sqrt(2 omega); "
        "controlled round trip exact; Cartesian to mass-weighted G -> G/sqrt(m)"
    )
    return Gamma


def derive_observable_strengths():
    """Assert operator-defined magnetic and displacement mode strengths.

    The physical transverse-spin operator is ``zeta_i = S_i1 + i S_i2 =
    sqrt(2 S_i) b_i`` to linear order, and the mass-weighted normal
    displacement is ``Q = (a + a^+)/sqrt(2 beta)``.  For a positive hybrid
    mode with normalized column ``t = (t_b, t_a, t_b^*, t_a^*)`` the
    one-quantum matrix elements are ``<0|zeta|mode> = sqrt(2 S) t_b`` and
    ``<0|Q|mode> = t_a / sqrt(2 beta)``, giving the physical strengths

        M = 2 S |t_b|^2,        D = |t_a|^2 / (2 beta).

    The bare limits pin both normalizations: the magnon-like mode has
    ``M = 2 S`` and ``D = 0``; the phonon-like mode has ``M = 0`` and
    ``D = 1/(2 beta)`` (the zero-point amplitude of Q).  Normalized
    shares built from the strengths equal the amplitude shares.
    """
    S = sp.symbols("S", positive=True)
    r = (E**2 - alpha**2) / (2 * gam * beta)
    u_b = (alpha / E + 1) / sp.sqrt(2)
    u_a = r * (beta / E + 1) / sp.sqrt(2)
    v_a = r * (beta / E - 1) / sp.sqrt(2)
    gnorm = sp.simplify(2 * (alpha + beta * r**2) / E)
    t_b2 = sp.simplify(u_b**2 / gnorm)
    t_a2 = sp.simplify(u_a**2 / gnorm)
    v_a2 = sp.simplify(v_a**2 / gnorm)
    Mag = sp.simplify(2 * S * t_b2)
    # Hermitian displacement field: particle + conjugated hole component
    # <0|Q|mode> = (t_a + t_bar_a)/sqrt(2 beta) for the real model, so
    # D = (t_a + t_bar_a)^2 / (2 beta)
    Dis = sp.simplify(((sp.sqrt(t_a2) + sp.sqrt(v_a2)) ** 2) / (2 * beta))
    # bare limits
    mag_limit = sp.simplify(
        sp.limit(Mag.subs(E, sp.sqrt(alpha**2 + gam**2)), gam, 0, "+")
    )
    assert mag_limit == 2 * S, mag_limit
    dis_limit_phonon = sp.simplify(
        sp.limit(
            Dis.subs({beta: 3 * alpha, E: sp.sqrt(9 * alpha**2 + gam**2)}), gam, 0, "+"
        )
    )
    assert sp.simplify(dis_limit_phonon - 1 / (6 * alpha)) == 0, dis_limit_phonon
    dis_limit_magnon = sp.simplify(
        sp.limit(Dis.subs(E, sp.sqrt(alpha**2 + gam**2)), gam, 0, "+")
    )
    assert dis_limit_magnon == 0, dis_limit_magnon
    # Normalized strength shares are the particle-amplitude shares; the
    # previously recorded W-amp shares include hole amplitudes and are a
    # separate diagnostic.  Resonance: strength shares are 1/2 on both
    # branches.
    disc_res = 16 * alpha**2 * gam**2
    for branch in (+1, -1):
        # alpha = beta with r = 1: t_b = t_a exactly, so the operator
        # strength share is exactly 1/2 on both branches.
        tb_res = sp.simplify(
            t_b2.subs(beta, alpha).subs(E**2, alpha**2 + branch * sp.sqrt(disc_res) / 2)
        )
        ta_res = sp.simplify(
            t_a2.subs(beta, alpha).subs(E**2, alpha**2 + branch * sp.sqrt(disc_res) / 2)
        )
        assert sp.simplify(tb_res - ta_res) == 0, (tb_res, ta_res)
    print(
        "[ok] strengths M = 2 S |t_b|^2, "
        "D = (t_a + t_bar_a)^2/(2 beta) (Hermitian field); bare limits "
        "M=2S / D=1/(2 beta); resonance strengths symmetric"
    )
    return Mag, Dis


def derive_displacement_matrix_element():
    """Assert the full vacuum-to-mode displacement matrix element.

    The Hermitian displacement field is ``w_beta = sum_nu e_{beta nu}
    (a_nu + a_nu^+)/sqrt(2 omega_nu)``.  With the hybrid expansion
    ``a_nu = sum_lam (t_{a nu, lam} gamma_lam + t_{a-bar nu, lam}
    gamma_lam^+)`` the Hermitian conjugate gives
    ``a_nu^+ = sum_lam (t^*_{a nu, lam} gamma_lam^+ + t^*_{a-bar nu, lam}
    gamma_lam)``, so BOTH the particle and the conjugated hole component
    enter the one-quantum matrix element:

        <0|w_beta|lam> = sum_nu e_{beta nu}
                         (t_{a nu, lam} + t^*_{a-bar nu, lam})
                         / sqrt(2 omega_nu),

    and the per-mode zero-point factor multiplies the normal-mode
    amplitudes BEFORE the eigenvector projection (it is indexed by the
    mode nu, not by the coordinate beta).
    """
    beta_sym, omega1, omega2 = sp.symbols("beta omega1 omega2", positive=True)
    x1, x2 = sp.symbols("x1 x2", real=True)
    t_ah1, t_ah2 = sp.symbols("tah1 tah2", real=True)
    theta = sp.symbols("theta", real=True)
    # mixed two-mode eigenvector rotation
    e = sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
    # correct: per-mode zero point applied to mode amplitudes, then project
    x = sp.Matrix([x1 + t_ah1, x2 + t_ah2])  # particle + conjugated hole
    amp_correct_0 = e[0, 0] * x[0] / sp.sqrt(2 * omega1) + e[0, 1] * x[1] / sp.sqrt(
        2 * omega2
    )
    # wrong: project first, then apply a coordinate-indexed zero point
    amp_wrong_0 = (e[0, 0] * x[0] + e[0, 1] * x[1]) / sp.sqrt(2 * omega1)
    diff = sp.simplify(amp_correct_0 - amp_wrong_0)
    assert (
        diff != 0
    ), "zero-point ordering is indistinguishable only for degenerate omega"
    print(
        "[ok] displacement element uses particle + conjugated hole amplitudes "
        "with per-mode 1/sqrt(2 omega) applied before eigenvector projection"
    )
    # controlled-model strength: D = |t_a + t_bar_a*|^2 / (2 beta)
    rr = (E**2 - alpha**2) / (2 * gam * beta)
    t_a = (rr * (beta / E + 1) / sp.sqrt(2)) / sp.sqrt(2 * (alpha + beta * rr**2) / E)
    t_ah = (rr * (beta / E - 1) / sp.sqrt(2)) / sp.sqrt(2 * (alpha + beta * rr**2) / E)
    D = sp.simplify((t_a + t_ah) ** 2 / (2 * beta))
    dis_limit_phonon = sp.simplify(
        sp.limit(D.subs(E, sp.sqrt(beta**2 + gam**2)), gam, 0, "+")
    )
    assert dis_limit_phonon == 1 / (2 * beta), dis_limit_phonon
    dis_limit_magnon = sp.simplify(
        sp.limit(D.subs(E, sp.sqrt(alpha**2 + gam**2)), gam, 0, "+")
    )
    assert dis_limit_magnon == 0, dis_limit_magnon
    print(
        "[ok] controlled-model displacement strength "
        "D = (t_a + t_bar_a)^2/(2 beta); bare limits preserved"
    )
    return D


def solver_detuning_sweep():
    """Check the implemented hybrid solver against the derived expressions."""
    from TB2J.magnon_phonon.harmonic import ControlledTwoBosonModel, hybrid_modes

    points = np.zeros((1, 3))
    worst_energy = 0.0
    worst_weight = 0.0
    worst_strength = 0.0
    for ratio in [0.4, 0.7, 1.0, 1.6, 2.5, 4.0]:
        for gscale in [0.05, 0.2, 0.45]:
            beta_v = 2.0
            alpha_v = ratio * beta_v
            g = gscale * np.sqrt(alpha_v * beta_v) * 0.9  # stay below threshold
            model = ControlledTwoBosonModel(alpha_eV=alpha_v, beta_eV=beta_v, g_eV=g)
            res = hybrid_modes(points, model.to_mixed_hessian())
            disc = np.sqrt(
                ((alpha_v**2 - beta_v**2) / 2) ** 2 + 4 * alpha_v * beta_v * g**2
            )
            derived = np.sort(
                np.sqrt(
                    [
                        (alpha_v**2 + beta_v**2) / 2 + disc,
                        (alpha_v**2 + beta_v**2) / 2 - disc,
                    ]
                )
            )
            got = np.sort(res.energies[0])
            worst_energy = max(worst_energy, np.max(np.abs(got - derived)))
            for mode in range(2):
                t = res.paraunitary[0][:, mode]
                wm = abs(t[0]) ** 2 + abs(t[2]) ** 2
                wp = abs(t[1]) ** 2 + abs(t[3]) ** 2
                share = wm / (wm + wp)
                Ei = derived[mode]
                r = (Ei**2 - alpha_v**2) / (2 * g * beta_v)
                Wm = 1 + (alpha_v / Ei) ** 2
                Wp = r**2 * (1 + (beta_v / Ei) ** 2)
                worst_weight = max(worst_weight, abs(share - Wm / (Wm + Wp)))
                # species weights recorded by the solver must match too
                assert abs(res.magnetic_weight[0, mode] - share) < 1e-9
                assert abs(res.displacement_weight[0, mode] - (1 - share)) < 1e-9
                # operator-defined physical strengths (S = 2 controlled model):
                # M = 2 S |t_b|^2,
                # D = |t_a + t_bar_a*|^2/(2 beta)  (Hermitian field: both
                # phonon components enter)
                t_col = res.paraunitary[0][:, mode]
                mag_expect = 2 * 2.0 * abs(t_col[0]) ** 2
                dis_expect = abs(t_col[1] + np.conj(t_col[3])) ** 2 / (2 * beta_v)
                assert abs(res.magnetic_strength[0, mode] - mag_expect) < 1e-9
                assert abs(res.displacement_strength[0, mode] - dis_expect) < 1e-9
                worst_strength = max(
                    worst_strength,
                    abs(res.magnetic_strength[0, mode] - mag_expect),
                    abs(res.displacement_strength[0, mode] - dis_expect),
                )
    assert worst_energy < 1e-10, worst_energy
    assert worst_weight < 1e-9, worst_weight
    assert worst_strength < 1e-9, worst_strength
    print(
        f"[ok] implemented solver reproduces derived energies/weights/strengths "
        f"across detuning (max dev E={worst_energy:.2e}, w={worst_weight:.2e}, "
        f"strength={worst_strength:.2e})"
    )

    # Stability threshold: flagged, not regularized.
    beta_v, alpha_v = 2.0, 2.0
    g = np.sqrt(alpha_v * beta_v / 4) * 1.05
    model = ControlledTwoBosonModel(alpha_eV=alpha_v, beta_eV=beta_v, g_eV=g)
    res = hybrid_modes(points, model.to_mixed_hessian())
    assert not res.is_stable[0] and np.all(np.isnan(res.energies[0]))
    assert res.diagnostics.stability_flag[0]
    assert res.diagnostics.shift[0] == 0.0
    g = np.sqrt(alpha_v * beta_v / 4) * 0.95
    model = ControlledTwoBosonModel(alpha_eV=alpha_v, beta_eV=beta_v, g_eV=g)
    res = hybrid_modes(points, model.to_mixed_hessian())
    assert res.is_stable[0]
    print("[ok] stability threshold crossing flagged (never shifted)")


if __name__ == "__main__":
    Ep2, Em2, disc = derive_energies()
    derive_stability(Em2, disc)
    derive_weights()
    derive_hessian_to_coupling()
    derive_observable_strengths()
    derive_displacement_matrix_element()
    numeric_certificate()
    solver_detuning_sweep()
    print("harmonic_magnon_phonon: all derivation checks green")
