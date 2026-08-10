"""Regression test for the collinear GPU exchange tensor.

``ExchangeCL2GPU._compute_collinear_A_batch`` built the orbital-resolved A
tensor with ``einsum("raj,rji->rai", X, Y)``, which *contracts* over the
orbital index j (a matrix product). The CPU reference — both the
non-vectorized ``einsum("ij,ji->ij", ...)`` and the vectorized
``einsum("ab,rbc,cd,rda->rac", ...)`` — is an *element-wise* product with no
contraction. The GPU therefore returned wrong J values (and the wrong shape
``(nR, ni, ni)`` instead of ``(nR, ni, nj)``); on bccFe the nearest-neighbour J
was ~52 meV via GPU vs ~17.6 meV via CPU.

The numpy test below runs in the default suite and pins the einsum formula; the
JAX test (marked ``gpu``) exercises the actual JIT kernel.
"""

from __future__ import annotations

import numpy as np
import pytest


def _cpu_reference(Delta_i, Delta_j, Gij, Gji):
    """CPU vectorized collinear A tensor: element-wise, no contraction."""
    return np.einsum("ab,rbc,cd,rda->rac", Delta_i, Gij, Delta_j, Gji)


def test_collinear_gpu_einsum_is_elementwise():
    """The corrected GPU einsum matches the CPU reference; the old one does not."""
    rng = np.random.default_rng(0)
    nR, ni, nj = 3, 5, 4
    Delta_i = rng.standard_normal((ni, ni))
    Delta_j = rng.standard_normal((nj, nj))
    Gij = rng.standard_normal((nR, ni, nj)) + 1j * rng.standard_normal((nR, ni, nj))
    Gji = rng.standard_normal((nR, nj, ni)) + 1j * rng.standard_normal((nR, nj, ni))

    ref = _cpu_reference(Delta_i, Delta_j, Gij, Gji)
    assert ref.shape == (nR, ni, nj)

    X = np.einsum("ab,rbj->raj", Delta_i, Gij)
    Y = np.einsum("jk,rki->rji", Delta_j, Gji)

    fixed = np.einsum("raj,rja->raj", X, Y)
    np.testing.assert_allclose(fixed, ref, atol=1e-12)

    # The old buggy contraction summed over j -> wrong shape and wrong values.
    buggy = np.einsum("raj,rji->rai", X, Y)
    assert buggy.shape == (nR, ni, ni)
    assert not np.allclose(buggy.sum(axis=(1, 2)), ref.sum(axis=(1, 2)), atol=1e-6)


@pytest.mark.gpu
def test_collinear_gpu_kernel_matches_cpu():
    """The JIT kernel _compute_collinear_A_batch agrees with the CPU reference."""
    pytest.importorskip("jax")
    from TB2J.gpu.exchangeCL_gpu import _compute_collinear_A_batch

    rng = np.random.default_rng(1)
    nR, ni, nj = 3, 5, 4
    Delta_i = rng.standard_normal((ni, ni))
    Delta_j = rng.standard_normal((nj, nj))
    Gij = rng.standard_normal((nR, ni, nj)) + 1j * rng.standard_normal((nR, ni, nj))
    Gji = rng.standard_normal((nR, nj, ni)) + 1j * rng.standard_normal((nR, nj, ni))

    import jax.numpy as jnp

    t, a_total = _compute_collinear_A_batch(
        jnp.asarray(Gij), jnp.asarray(Gji), jnp.asarray(Delta_i), jnp.asarray(Delta_j)
    )
    t = np.asarray(t)
    ref = _cpu_reference(Delta_i, Delta_j, Gij, Gji)
    np.testing.assert_allclose(t, ref, atol=1e-8)
    np.testing.assert_allclose(np.asarray(a_total), ref.sum(axis=(1, 2)), atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
