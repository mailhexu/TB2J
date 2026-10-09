"""QE separable beta-operator trace: sympy-verified contraction.

Story 1/2 of the QE projector-Green exporter spec
(Projects/TB2J/specs/qe-projector-exporter).

QE's nonlocal Hamiltonian is V_NL = beta D beta^dagger with D = deeq and
projector coefficients P = beta^dagger psi = becp.  The exchange trace of
two spin channels therefore reduces exactly to a D/G_beta contraction:

    K_phys = Tr[(beta Du beta^dagger) Gup (beta Dd beta^dagger) Gdn]
           = Tr[Du Gb_up Dd Gb_dn],      Gb = beta^dagger G beta.

This script asserts, for several independent complex-rational realizations:

1. The direct separable identity above (the TB2J contract: becp enters as
   already-dual coefficients, deeq as the matching covariant separable
   operator, overlap_k = None).
2. The jointly-transformed equivalent: with M = beta^dagger beta,
   Delta_cov = M D M and G_dual = M^{-1} Gb M^{-1} give the same trace
   (algebraically valid, computationally pointless; not used).
3. The falsified alternative: dressing ONLY the Green matrix by M^{-1}
   while leaving D unchanged breaks the identity -- this is why qq_at or
   M must never be used as a channel metric for the undressed becp Green.
4. Empirical corroboration (real QE 7.4 dumps, recorded in the research
   memo): ||M - qq_at||_F spans 1e2..1e3 on US/PAW golden runs, so the
   Gram hypothesis is also numerically excluded, not only algebraically
   when mispaired.

Run with the mydev environment.
"""

import random

import sympy as sp

RNG = random.Random(20261008)


def random_matrix(rows, cols, lo, hi):
    return sp.Matrix(
        rows, cols, lambda i, j: RNG.randint(lo, hi) + sp.I * RNG.randint(lo, hi)
    )


def hermitian(mat):
    return mat + mat.conjugate().T


def check_trial(nb, nch):
    """nb plane-wave dims, nch beta channels; returns True if all pass."""
    beta = random_matrix(nb, nch, -3, 3)
    gram = beta.conjugate().T * beta
    if gram.det() == 0:
        return False  # skip degenerate draws

    # Hermitian spin-dependent operators and Green matrices
    d_up = hermitian(random_matrix(nch, nch, -2, 2))
    d_dn = hermitian(random_matrix(nch, nch, -2, 2))
    g_up = hermitian(random_matrix(nb, nb, -2, 2))
    g_dn = hermitian(random_matrix(nb, nb, -2, 2))

    v_up = beta * d_up * beta.conjugate().T
    v_dn = beta * d_dn * beta.conjugate().T
    gb_up = beta.conjugate().T * g_up * beta
    gb_dn = beta.conjugate().T * g_dn * beta

    k_phys = sp.trace(v_up * g_up * v_dn * g_dn)
    k_beta = sp.trace(d_up * gb_up * d_dn * gb_dn)
    assert sp.simplify(k_phys - k_beta) == 0, "direct separable trace failed"

    # Jointly transformed equivalent (correct but unused)
    gram_inv = gram.inv()
    gd_up = gram_inv * gb_up * gram_inv
    gd_dn = gram_inv * gb_dn * gram_inv
    dc_up = gram * d_up * gram
    dc_dn = gram * d_dn * gram
    k_cov = sp.trace(dc_up * gd_up * dc_dn * gd_dn)
    assert sp.simplify(k_phys - k_cov) == 0, "jointly transformed trace failed"

    # Gram-only dressing of the Green (the falsified pairing)
    k_wrong = sp.trace(d_up * gd_up * d_dn * gd_dn)
    assert sp.simplify(k_phys - k_wrong) != 0, "naive Gram dressing unexpectedly equal"
    return True


def main():
    trials = 0
    attempted = 0
    while trials < 3 and attempted < 10:
        attempted += 1
        if check_trial(3, 2):
            trials += 1
            print(
                f"trial {trials - 1}: direct D/Gb == physical trace; "
                "joint transform == physical; naive G-only dressing fails"
            )
    assert trials == 3, "could not draw 3 nonsingular realizations"
    print("PASS: QE separable-operator identity, 3 complex-rational trials")


if __name__ == "__main__":
    main()
