"""SymPy certificate for EMC ferromagnetic angle/boson conventions.

Run with the mydev environment: python docs/sympy/emc_magnon_normalization.py
The electron spinor vertex and q-phase certificate lives in the EMC package.
"""

import sympy as sp


def main():
    S, D = sp.symbols("S D", positive=True, real=True)
    u, b, bd = sp.symbols("u b bdag", complex=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sm = (sx - sp.I * sy) / 2
    sp_ = (sx + sp.I * sy) / 2
    # n_+ = sqrt(2/S) u b, n_- = sqrt(2/S) u* bdag;
    # n_x=(n_++n_-)/2 and n_y=(n_+-n_-)/(2i).
    plus = sp.sqrt(2 / S) * u * b
    minus = sp.sqrt(2 / S) * sp.conjugate(u) * bd
    perturbation = D * ((plus + minus) * sx / 2 + (plus - minus) * sy / (2 * sp.I))
    assert sp.simplify(perturbation.diff(b) - D * sp.sqrt(2 / S) * u * sm) == sp.zeros(
        2
    )
    assert sp.simplify(
        perturbation.diff(bd) - D * sp.sqrt(2 / S) * sp.conjugate(u) * sp_
    ) == sp.zeros(2)
    theta = sp.symbols("theta", real=True)
    assert sp.limit((sp.sin(theta) - theta) / theta**3, theta, 0) == -sp.Rational(1, 6)

    # TB2J returns the Euclidean-unit eigenvector v of A=K^dag g K.
    # This generic invertible lower triangular K proves the inverse identity
    # used to recover psi=sqrt(omega) K^{-dag}v and its metric norm.
    a, c = sp.symbols("a c", positive=True, real=True)
    t = sp.symbols("t", real=True)
    K = sp.Matrix([[a, 0], [t, c]])
    g = sp.diag(1, -1)
    A = K.H * g * K
    assert sp.simplify(A.inv() - K.inv() * g * K.H.inv()) == sp.zeros(2)
    omega = sp.symbols("omega", positive=True, real=True)
    v1, v2 = sp.symbols("v1 v2", complex=True)
    v = sp.Matrix([v1, v2])
    psi = sp.sqrt(omega) * K.H.inv() * v
    # The first equality reduces using A v=omega v; the second follows
    # from omega v^dag A^-1 v=v^dag v for a unit Euclidean eigenvector.
    residual_identity = (
        K * K.H * psi
        - omega * g * psi
        - sp.sqrt(omega) * K * A.inv() * (A * v - omega * v)
    )
    assert residual_identity.applyfunc(sp.simplify) == sp.zeros(2, 1)
    assert sp.simplify((psi.H * g * psi)[0] - omega * (v.H * A.inv() * v)[0]) == 0
    assert sp.simplify((sp.sqrt(omega) * K.H.inv() * v).subs(omega, 0)) == sp.zeros(
        2, 1
    )
    print("HP spin lowering/raising, sqrt(2/S), and Cholesky metric identities: OK")


if __name__ == "__main__":
    main()
