"""Consumer-visible CFR2 normalization, complex coherence and depth boundaries."""

import numpy as np
import pytest
from ase.units import kB
from scipy.linalg import eigh
from scipy.special import expit

from TB2J.mycfr import CFR2


@pytest.mark.parametrize("depth", [0, -2, 1, 3, 31, 81, 2.0, 2.5, True, "80"])
def test_unsupported_depth_is_rejected(depth):
    with pytest.raises(ValueError):
        CFR2(nz=depth, T=600)


def test_low_order_and_convergence():
    errors = []
    for depth in (2, 8, 20, np.int64(80)):
        contour = CFR2(nz=depth, T=600)
        epsilon = 0.8
        integral = contour.integrate_values(1 / (contour.path - epsilon))
        errors.append(abs(-integral.imag / np.pi - expit(-epsilon / (kB * 600))))
        assert np.isfinite(integral)
    assert errors[0] > 1e-3
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < 1e-10


@pytest.mark.parametrize("entrypoint", ["values", "callback"])
def test_scalar_occupations_and_energy_reference(entrypoint):
    contour = CFR2(nz=80, T=600)
    for epsilon in (-0.2, 0.2):
        if entrypoint == "values":
            result = contour.integrate_values(1 / (contour.path - epsilon))
        else:

            def resolvent(z, ef):
                return 1 / (z + ef - (epsilon + 6.3))

            result = contour.integrate_func(resolvent, ef=6.3)
        expected = -1j * np.pi * expit(-epsilon / (kB * 600))
        assert abs(result - expected) < 1e-10


@pytest.mark.parametrize("entrypoint", ["values", "callback"])
def test_complex_nonorthogonal_density_and_batched_operator_trace(entrypoint):
    contour = CFR2(nz=80, T=600)
    frame = np.array([[1, 0.2 + 0.1j], [0, 1.3]], complex)
    overlap = frame.conj().T @ frame
    hamiltonian = frame.conj().T @ np.diag([-0.2, 0.3]) @ frame
    eigenvalues, eigenvectors = eigh(hamiltonian, overlap)
    occupations = expit(-eigenvalues / (kB * 600))
    reference = (eigenvectors * occupations) @ eigenvectors.conj().T
    operator = np.array([[0.4, 0.3j], [-0.3j, -0.2]])

    def resolvent(z, ef=0):
        g = np.linalg.inv((z + ef)[:, None, None] * overlap - hamiltonian)
        return np.stack([g, operator @ g], axis=1)

    if entrypoint == "values":
        integrated = contour.integrate_values(resolvent(contour.path))
    else:
        integrated = contour.integrate_func(resolvent, ef=0)
    density = (integrated[0].conj().T - integrated[0]) / (2j * np.pi)
    np.testing.assert_allclose(density, reference, atol=1e-10, rtol=0)
    assert abs(np.trace(overlap @ density) - occupations.sum()) < 1e-10
    expectation = -np.trace(integrated[1]).imag / np.pi
    assert abs(expectation - np.trace(operator @ reference).real) < 1e-10
    moment = 1j * contour.Rinf * resolvent(np.array([1j * contour.Rinf]))[0, 0]
    np.testing.assert_allclose(moment, np.linalg.inv(overlap), atol=1e-9, rtol=0)


@pytest.mark.parametrize("entrypoint", ["values", "callback"])
def test_grand_potential_response_has_no_density_half_shift(entrypoint):
    contour = CFR2(nz=80, T=600)
    d, v = 0.2, 0.1

    def kernel(z, ef=0):
        return 2 * v**2 / ((z + ef) ** 2 - d**2)

    if entrypoint == "values":
        integrated = contour.integrate_values(kernel(contour.path))
    else:
        integrated = contour.integrate_func(kernel)
    response = -integrated.imag / (2 * np.pi)
    expected = -(v**2) * np.tanh(d / (2 * kB * 600)) / (2 * d)
    assert abs(response - expected) < 1e-10
