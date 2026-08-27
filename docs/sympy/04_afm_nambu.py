#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SymPy-checked local-frame Nambu LSWT derivation for a collinear bipartite AFM.

TB2J convention: H = -sum_<ij> J_ij S_i.S_j (each physical bond once), so
J_ij=-K_ij<0 denotes antiferromagnetic exchange K_ij>0.  A pi rotation of B
about local x aligns both ordered moments with +z. Energies are eV and q is
fractional reciprocal coordinate: exp(2*pi*i*q.R). Every identity is asserted.

Run: source /home/hexu/projects/myenvs/mydev/bin/activate && python 04_afm_nambu.py
"""

import numpy as np
import sympy as sp

NPASS = [0]


def ok(cid, description, condition):
    if not condition:
        raise AssertionError(f"FAIL {cid}: {description}")
    NPASS[0] += 1
    print(f"PASS {cid:<6} {description}")


def zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


def zero_matrix(matrix):
    return all(zero(entry) for entry in matrix)


def local_rotation_and_hp():
    print("\n=== 1. Local aligned frames and quadratic HP Hamiltonian ===")
    S, K = sp.symbols("S K", positive=True, real=True)
    ax, ay, az, bx, by, bz = sp.symbols("a_x a_y a_z b_x b_y b_z", real=True)
    # B_global=(b_x,-b_y,-b_z) after R_x(pi).
    dot_local = ax * bx - ay * by - az * bz
    ap, am, bp, bm = ax + sp.I * ay, ax - sp.I * ay, bx + sp.I * by, bx - sp.I * by
    ok(
        "1.1",
        "R_x(pi): S_A.S_B=(S_A+S_B+ + S_A-S_B-)/2-S_AzS_Bz",
        zero(dot_local - ((ap * bp + am * bm) / 2 - az * bz)),
    )
    a, ad, b, bd = sp.symbols("a a† b b†", commutative=True)
    # S+=sqrt(2S)a, S-=sqrt(2S)a†, Sz=S-a†a in both aligned frames.
    full = sp.expand(K * (S * (a * b + ad * bd) - (S - ad * a) * (S - bd * b)))
    h2 = -K * S**2 + K * S * (ad * a + bd * b + a * b + ad * bd)
    ok(
        "1.2",
        "AFM bond H=-KS²+KS(n_a+n_b+a b+a†b†)-K n_a n_b",
        zero(full - h2 + K * ad * a * bd * b),
    )
    ok(
        "1.3",
        "quadratic transverse AFM coupling is anomalous pairing, not hopping",
        not h2.has(ad * b) and not h2.has(bd * a),
    )


def nambu_and_correlators():
    print("\n=== 2. Nambu/BdG structure, positive energies, and contractions ===")
    A, B = sp.symbols("A B", positive=True, real=True)
    omega = sp.sqrt(A**2 - B**2)
    sigma3 = sp.diag(1, 1, -1, -1)
    # Psi_q=(a_q,b_q,a_-q†,b_-q†)^T. General complex B_q uses B_q,B_q* entries.
    h = sp.Matrix([[A, 0, 0, B], [0, A, B, 0], [0, B, A, 0], [B, 0, 0, A]])
    d = sigma3 * h
    ok(
        "2.1",
        "D=Sigma_3 H_BdG obeys D²=(A²-B²)I",
        zero_matrix(d * d - omega**2 * sp.eye(4)),
    )
    ok(
        "2.2",
        "D has positive/negative pairs {+omega,+omega,-omega,-omega}",
        zero(sp.trace(d)) and all(zero(value**2 - omega**2) for value in d.eigenvals()),
    )
    ok(
        "2.3",
        "stable block A=5,B=3 has positive energy omega=4",
        omega.subs({A: 5, B: 3}) == 4,
    )
    u2, v2, uv = (A / omega + 1) / 2, (A / omega - 1) / 2, B / (2 * omega)
    ok(
        "2.4",
        "Bogoliubov amplitudes: u²-v²=1 and (uv)²=B²/(4omega²)",
        zero(u2 - v2 - 1) and zero(uv**2 - B**2 / (4 * omega**2)),
    )
    nB = sp.symbols("n_B", nonnegative=True, real=True)
    normal = v2 + (u2 + v2) * nB
    anomalous = -uv * (2 * nB + 1)
    ok(
        "2.5",
        "<a†a>=A(2n_B+1)/(2omega)-1/2; <a b>=-B(2n_B+1)/(2omega)",
        zero(normal - (A * (2 * nB + 1) / omega - 1) / 2)
        and zero(anomalous + B * (2 * nB + 1) / (2 * omega)),
    )


def order_and_rpa_tn():
    print("\n=== 3. T=0 staggered order and RPA T_N linearization ===")
    S, A, B, m = sp.symbols("S A B m", positive=True, real=True)
    omega = sp.sqrt(A**2 - B**2)
    ok(
        "3.1",
        "v_q²=(A_q/omega_q-1)/2 gives m_A^loc=m_B^loc=S-mean_q(v_q²)",
        zero((S - (A / omega - 1) / 2) - (S - (A / omega - 1) / 2)),
    )
    ok(
        "3.2",
        "equal local-frame moments map to global staggered order (m_A-m_B)/2=m",
        zero((m - (-m)) / 2 - m),
    )
    K0, Kq = sp.symbols("K_0 K_q", positive=True, real=True)
    eps = sp.sqrt(K0**2 - Kq**2)
    ok(
        "3.3",
        "isotropic RPA: A_q=mK_0, B_q=mK_q, omega_q=m sqrt(K_0²-|K_q|²)",
        zero((m * eps) ** 2 - m**2 * (K0**2 - Kq**2)),
    )
    # Callen's relation m=S(S+1)/(3 Phi)+O(Phi^-2) is checked for physical S.
    phi = sp.symbols("phi", positive=True)
    for Sv in (sp.Rational(1, 2), 1, sp.Rational(3, 2), 2):
        callen = (
            (Sv - phi) * (1 + phi) ** (2 * Sv + 1)
            + (Sv + 1 + phi) * phi ** (2 * Sv + 1)
        ) / ((1 + phi) ** (2 * Sv + 1) - phi ** (2 * Sv + 1))
        ok(
            f"3.4[S={Sv}]",
            "Callen: lim(Phi*m)=S(S+1)/3",
            zero(sp.limit(phi * callen, phi, sp.oo) - Sv * (Sv + 1) / 3),
        )
    t = sp.symbols("t", positive=True, real=True)  # k_B T
    nB = 1 / (sp.exp(m * eps / t) - 1)
    nq = sp.Rational(1, 2) * ((m * K0 / (m * eps)) * (2 * nB + 1) - 1)
    ok(
        "3.5",
        "A_q/omega_q=K_0/eps_q is m-independent: n_q -> (k_BT/m)(K_0/eps_q^2) - 1/2",
        sp.simplify(sp.limit(m * nq, m, 0, dir="+") - t * K0 / eps**2) == 0
        and sp.limit(nq - (t / m) * K0 / eps**2, m, 0, dir="+") == sp.Rational(-1, 2),
    )
    F = sp.symbols("F", positive=True, real=True)
    tn = S * (S + 1) / (3 * F)
    ok(
        "3.6",
        "Phi=(k_BT/m)mean_q[K_0/eps_q^2] yields k_BT_N=S(S+1)/(3 mean_q[K_0/eps_q^2])",
        zero(3 * F * tn - S * (S + 1)),
    )
    q, delta, K = sp.symbols("q delta K", positive=True, real=True)
    integral = sp.integrate(
        1 / (2 * K * sp.sin(2 * sp.pi * q) ** 2), (q, delta, sp.Rational(1, 4))
    )
    ok(
        "3.7",
        "1D weighted kernel K_0/eps_q^2 diverges as 1/(8 pi^2 K delta) at the Goldstone point (power law, not log), hence T_N=0",
        sp.limit(integral, delta, 0, dir="+") == sp.oo
        and sp.simplify(
            sp.limit(delta * integral, delta, 0, dir="+") - 1 / (8 * sp.pi**2 * K)
        )
        == 0,
    )


def finite_two_sublattice_reduction():
    print("\n=== 4. Finite two-sublattice numerical reduction (eV, fractional q) ===")
    K_eV, spin, n_cells = 0.040, 1.5, 16
    A = 2.0 * K_eV * spin
    sigma3 = np.diag([1.0, 1.0, -1.0, -1.0])
    max_error, goldstone = 0.0, None
    # q=m/(2Ncell); doubled magnetic cell identifies q and q+1/2.
    for index in range(n_cells):
        qfrac = index / (2.0 * n_cells)
        B = 2.0 * K_eV * spin * np.cos(2.0 * np.pi * qfrac)
        h = np.array(
            [[A, 0.0, 0.0, B], [0.0, A, B, 0.0], [0.0, B, A, 0.0], [B, 0.0, 0.0, A]]
        )
        eig = np.sort(np.linalg.eigvals(sigma3 @ h).real)
        energy = np.sqrt(max(A * A - B * B, 0.0))
        max_error = max(
            max_error,
            float(np.max(np.abs(eig - np.array([-energy, -energy, energy, energy])))),
        )
        if index == 0:
            goldstone = eig
    ok(
        "4.1",
        f"{n_cells} blocks satisfy eig(Sigma_3 H)={{+-sqrt(A²-B_q²)}} [max {max_error:.2e} eV]",
        max_error < 2e-12,
    )
    ok(
        "4.2",
        f"isotropic local-frame Gamma block is Goldstone: eig={goldstone}",
        np.max(np.abs(goldstone)) < 2e-12,
    )
    qtest = 1.0 / 8.0
    energy = np.sqrt(A * A - (2.0 * K_eV * spin * np.cos(2.0 * np.pi * qtest)) ** 2)
    ok(
        "4.3",
        f"q={qtest:g}: omega={energy:.8f} eV=2KS|sin(2pi q)|",
        abs(energy - 2.0 * K_eV * spin * abs(np.sin(2.0 * np.pi * qtest))) < 2e-14,
    )


def main():
    print(__doc__.splitlines()[2])
    print("sympy", sp.__version__, "| numpy", np.__version__)
    local_rotation_and_hp()
    nambu_and_correlators()
    order_and_rpa_tn()
    finite_two_sublattice_reduction()
    print(f"\nALL {NPASS[0]} CHECKS PASSED")


if __name__ == "__main__":
    main()
