"""Assertion-checked derivation of Story-008 gauge-invariant subspace comparison.

Run with the project environment: python docs/sympy/mode_qualification_projectors.py

Claim (used by TB2J.magnon_phonon.qualification): for a cluster of exactly
degenerate eigenvalues with orthonormal eigenvector block X (columns), any
re-diagonalization returns X' = X W with W unitary *within* the cluster, and
the orthogonal projector P = X X^dagger is invariant:

    P' = X' X'^dagger = X W W^dagger X^dagger = X X^dagger = P.

Hence comparing P between repeated diagonalizations is a gauge-invariant
determinism witness, whereas comparing individual eigenvectors is not.
"""

import sympy as sp


def main():
    # Symbolic: 2-fold degenerate cluster, generic 2x2 unitary W (Euler form).
    th, ph1, ph2 = sp.symbols("theta phi1 phi2", real=True)
    W = sp.Matrix(
        [
            [sp.exp(sp.I * ph1) * sp.cos(th), sp.exp(sp.I * ph2) * sp.sin(th)],
            [-sp.exp(-sp.I * ph2) * sp.sin(th), sp.exp(-sp.I * ph1) * sp.cos(th)],
        ]
    )
    assert sp.simplify(W * W.H - sp.eye(2)) == sp.zeros(2)

    # Use a full-rank 2x2 block B with B B^dagger = P.
    b11, b12, b21, b22 = sp.symbols("b11 b12 b21 b22", complex=True)
    B = sp.Matrix([[b11, b12], [b21, b22]])
    P = B * B.H
    Pp = (B * W) * (B * W).H
    assert sp.simplify(Pp - P) == sp.zeros(2)

    # Converse: mixing with a vector outside the cluster changes P.
    eps = sp.symbols("epsilon", real=True, nonzero=True)
    out = sp.Matrix([sp.exp(sp.I * ph1), 0, 0])
    # 3-dimensional illustration: first column of B perturbed out of subspace.
    B3 = sp.diag(sp.Matrix([[b11, b12], [b21, b22]]), 1)
    col_out = B3[:, 0] + eps * out
    B3_mod = B3.copy()
    B3_mod[:, 0] = col_out
    diff = B3_mod * B3_mod.H - B3 * B3.H
    assert sp.simplify(diff) != sp.zeros(3)

    # Numeric cross-check with the actual kernels used in production
    # (numpy eigh on a Hermitian matrix with a degenerate pair).
    import numpy as np

    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    A = Q @ np.diag([1.0, 1.0, 2.0, 3.0, 4.0]) @ Q.conj().T
    A = A + A.conj().T
    vals, vecs = np.linalg.eigh(A)
    assert np.allclose(vals[:2], vals[0], atol=1e-12)
    cluster = vecs[:, :2]
    # Random unitary gauge rotation inside the cluster.
    Z, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    rotated = cluster @ Z
    P1 = cluster @ cluster.conj().T
    P2 = rotated @ rotated.conj().T
    assert np.max(np.abs(P1 - P2)) < 1e-12
    # Numerical residual of the projector construction itself.
    assert np.max(np.abs(P1 @ P1 - P1)) < 1e-12

    print("mode_qualification_projectors: all assertions passed")


if __name__ == "__main__":
    main()
