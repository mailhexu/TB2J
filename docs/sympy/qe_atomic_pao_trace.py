"""Exact atomic-PAO/Kleinman--Bylander channel-pairing checks for QE.

Run with mydev: python docs/sympy/qe_atomic_pao_trace.py.
This is an algebraic convention check, not a completeness proof for a truncated
atomic-orbital subspace. All entries are exact Gaussian integers/rationals.
"""

import random

import sympy as sp


def random_matrix(rng, rows, cols):
    return sp.Matrix(
        rows,
        cols,
        lambda i, j: rng.randint(-2, 2) + sp.I * rng.randint(-2, 2),
    )


def hermitian(matrix):
    return matrix + matrix.conjugate().T


def verify(rng, dimension, channels):
    phi = random_matrix(rng, dimension, channels)
    metric = phi.conjugate().T * phi
    if metric.det() == 0:
        return False
    beta = random_matrix(rng, dimension, channels)
    local_xc = hermitian(random_matrix(rng, dimension, dimension))
    augmentation = hermitian(random_matrix(rng, channels, channels))
    green_up = hermitian(random_matrix(rng, dimension, dimension))
    green_down = hermitian(random_matrix(rng, dimension, dimension))

    # V is the genuine full-space spin vertex: multiplicative local XC
    # plus separable augmentation. B = <atomic|KB-beta> is not the Gram.
    cross = phi.conjugate().T * beta
    vertex = local_xc + beta * augmentation * beta.conjugate().T
    covariant = phi.conjugate().T * vertex * phi
    assert (
        covariant
        - (
            phi.conjugate().T * local_xc * phi
            + cross * augmentation * cross.conjugate().T
        )
    ).applyfunc(sp.expand) == sp.zeros(channels)

    # C=<atomic|psi> is primal, so Gcov=<atomic|G|atomic>. M^-1
    # dresses BOTH sides to give the dual Green used with covariant vertex.
    g_up_cov = phi.conjugate().T * green_up * phi
    g_down_cov = phi.conjugate().T * green_down * phi
    inverse = metric.inv()
    g_up_dual = inverse * g_up_cov * inverse
    g_down_dual = inverse * g_down_cov * inverse
    k_pao = sp.trace(covariant * g_up_dual * covariant * g_down_dual)
    k_equivalent = sp.trace(
        (inverse * covariant * inverse)
        * g_up_cov
        * (inverse * covariant * inverse)
        * g_down_cov
    )
    assert sp.expand(k_pao - k_equivalent) == 0
    if channels == dimension:
        # Complete invertible basis: the channel trace is the full trace.
        k_physical = sp.trace(vertex * green_up * vertex * green_down)
        assert sp.expand(k_pao - k_physical) == 0
    else:
        # Truncation is a physical approximation; the identity does NOT
        # certify a complete exchange vertex for arbitrary omitted states.
        k_physical = sp.trace(vertex * green_up * vertex * green_down)
        assert sp.expand(k_pao - k_physical) != 0
    return True


def main():
    rng = random.Random(20261009)
    assert sum(verify(rng, 2, 2) for _ in range(3)) == 3
    assert sum(verify(rng, 3, 2) for _ in range(3)) == 3
    # Bloch sums contain image-overlap terms. The onsite covariant vertex
    # is the R=0 Fourier coefficient, not its Gamma-only Bloch value.
    onsite = sp.Matrix([[2, 1], [1, 3]])
    neighbor = sp.Matrix([[1, sp.I], [-sp.I, 0]])
    gamma = onsite + neighbor + neighbor.conjugate().T
    zone_edge = onsite - neighbor - neighbor.conjugate().T
    assert (gamma + zone_edge) / 2 == onsite
    assert gamma != onsite
    print("PASS: QE atomic-PAO covariant vertex, dual Green, and truncation boundary")


if __name__ == "__main__":
    main()
