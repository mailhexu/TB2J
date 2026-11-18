"""Independent CFR2 normalization, metric-moment and finite-T response proofs.

MathDoc: tb2j/cfr2-integration-contract
Energies are in eV; beta is inverse eV. The response is a grand-potential
coefficient at fixed chemical potential, not a general fixed-N band energy.
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp
from ase.units import kB
from scipy.special import expit


# MathDoc: tb2j/cfr2-integration-contract
def derive():
    beta, d = sp.symbols("beta d", positive=True)
    strength, lam, energy = sp.symbols("v lambda epsilon", real=True)
    z = sp.symbols("z")
    residue, radius = sp.symbols("r R", positive=True)
    sample = sp.symbols("g")

    def f(e):
        return 1 / (1 + sp.exp(beta * e))

    legacy_weight = 2 * sp.I * residue / beta
    canonical_weight = -sp.pi * sp.I * residue / beta
    assert sp.simplify(canonical_weight / legacy_weight + sp.pi / 2) == 0
    moment_contribution = sp.simplify(-sp.pi * sp.I * (sp.I * radius * sample) / 2)
    assert moment_contribution == sp.pi * radius * sample / 2

    # One spinless two-level Hamiltonian; the trace counts both transitions.
    splitting = sp.sqrt(d**2 + lam**2 * strength**2)
    omega = (
        -(sp.log(1 + sp.exp(beta * splitting)) + sp.log(1 + sp.exp(-beta * splitting)))
        / beta
    )
    coefficient = sp.simplify(sp.diff(omega, lam, 2).subs(lam, 0) / 2)
    spectral_coefficient = strength**2 * (f(-d) - f(d)) / (-2 * d)
    assert sp.cancel((coefficient - spectral_coefficient).rewrite(sp.exp)) == 0
    assert sp.simplify(f(-splitting) + f(splitting) - 1) == 0
    h = sp.diag(-d, d)
    vertex = sp.Matrix([[0, strength], [strength, 0]])
    green = (z * sp.eye(2) - h).inv()
    kernel = sp.simplify(sp.trace(vertex * green * vertex * green))
    assert sp.simplify(kernel - 2 * strength**2 / ((z - d) * (z + d))) == 0
    # Integral of each simple resolvent is -i*pi*f(e). Partial fractions
    # therefore establish the response sign and the physical -Im/(2*pi).
    e1, e2 = sp.symbols("epsilon_1 epsilon_2", real=True)
    partial = (1 / (z - e1) - 1 / (z - e2)) / (e1 - e2)
    assert sp.factor(partial - 1 / ((z - e1) * (z - e2))) == 0
    integrated_kernel = -sp.pi * sp.I * 2 * spectral_coefficient
    assert (
        sp.cancel(
            (-sp.im(integrated_kernel) / (2 * sp.pi) - coefficient).rewrite(sp.exp)
        )
        == 0
    )
    equal_energy_kernel = sp.diff(f(energy), energy)
    assert sp.simplify(equal_energy_kernel + beta * f(energy) * (1 - f(energy))) == 0

    # A finite positive-definite metric is assumed. H is finite and Hermitian.
    a, b, s1, s2 = sp.symbols("a b s_1 s_2", real=True)
    metric = sp.Matrix([[s1, a + sp.I * b], [a - sp.I * b, s2]])
    inverse_metric = metric.inv()
    assert sp.simplify(metric * inverse_metric - sp.eye(2)) == sp.zeros(2)
    # z*G(z) = (S-H/z)^-1: use y=1/z to evaluate the asymptotic limit.
    y = sp.symbols("y")
    h1, h2, c, q = sp.symbols("h_1 h_2 c q", real=True)
    hermitian_h = sp.Matrix([[h1, c + sp.I * q], [c - sp.I * q, h2]])
    scaled_green = (metric - y * hermitian_h).inv()
    assert sp.simplify(scaled_green.subs(y, 0) - inverse_metric) == sp.zeros(2)
    assert sp.simplify(metric * scaled_green.subs(y, 0) - sp.eye(2)) == sp.zeros(2)
    # Two resolvents give O(z^-2), hence no density-like leading moment.
    assert sp.simplify(
        (y * scaled_green * vertex * scaled_green * vertex).subs(y, 0)
    ) == sp.zeros(2)

    # Derive the spectral representation in a complex non-orthogonal frame.
    scale = sp.symbols("c", positive=True)
    coefficients = sp.Matrix([[1, a + sp.I * b], [0, scale]])
    inverse_coefficients = coefficients.inv()
    spectral_metric = inverse_coefficients.adjoint() * inverse_coefficients
    spectral_h = inverse_coefficients.adjoint() * sp.diag(e1, e2) * inverse_coefficients
    spectral_green = (
        coefficients * sp.diag(1 / (z - e1), 1 / (z - e2)) * coefficients.adjoint()
    )
    assert sp.simplify(
        coefficients.adjoint() * spectral_metric * coefficients - sp.eye(2)
    ) == sp.zeros(2)
    assert sp.simplify(
        (z * spectral_metric - spectral_h) * spectral_green - sp.eye(2)
    ) == sp.zeros(2)
    spectral_density = coefficients * sp.diag(f(e1), f(e2)) * coefficients.adjoint()
    assert sp.simplify(
        coefficients * coefficients.adjoint() - spectral_metric.inv()
    ) == sp.zeros(2)
    assert (
        sp.simplify(sp.trace(spectral_metric * spectral_density) - f(e1) - f(e2)) == 0
    )

    n1, n2, x, t = sp.symbols("n_1 n_2 x t", real=True)
    density = sp.Matrix([[n1, x + sp.I * t], [x - sp.I * t, n2]])
    integral = -sp.pi * sp.I * density
    extracted = sp.simplify((integral.adjoint() - integral) / (2 * sp.pi * sp.I))
    assert extracted == density
    assert -sp.im(integral[0, 1]) / sp.pi != density[0, 1]
    o1, o2, u, w = sp.symbols("o_1 o_2 u w", real=True)
    observable = sp.Matrix([[o1, u + sp.I * w], [u - sp.I * w, o2]])
    traced_density = sp.simplify(sp.trace(observable * density))
    assert (
        sp.simplify(-sp.im(sp.trace(observable * integral)) / sp.pi - traced_density)
        == 0
    )

    return {
        "density_matrix_extraction": (
            sp.Adjoint(sp.MatrixSymbol("K", 2, 2)) - sp.MatrixSymbol("K", 2, 2)
        )
        / (2 * sp.pi * sp.I),
        "equal_energy_response_kernel": equal_energy_kernel,
        "finite_pole_normalization_ratio": sp.simplify(
            canonical_weight / legacy_weight
        ),
        "mae_grand_potential_coefficient": coefficient,
        "moment_sample_contribution": moment_contribution,
        "nonorthogonal_resolvent_moment": inverse_metric,
        "response_resolvent_kernel": kernel,
        "spectral_density_complex_nonorthogonal": spectral_density,
        "traced_density_extraction": traced_density,
    }


def numerical_checks():
    # Import the actual worktree, not an editable install from another branch.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from TB2J import mycfr

    assert Path(mycfr.__file__).resolve() == root / "TB2J" / "mycfr.py"
    contour = mycfr.CFR2(nz=80, T=600)
    beta = 1 / (kB * 600)
    scalar_error = 0.0
    for epsilon in (-0.2, 0.2):
        result = contour.integrate_values(1 / (contour.path - epsilon))
        expected = expit(-beta * epsilon)
        scalar_error = max(scalar_error, abs(-result.imag / np.pi - expected))
        shifted = contour.integrate_func(
            lambda z, ef: 1 / (z + ef - (epsilon + 6.3)), ef=6.3
        )
        assert abs(-shifted.imag / np.pi - expected) < 1e-10
    assert scalar_error < 1e-10

    transform = np.array([[1, 0.2 + 0.1j], [0, 1.3]], complex)
    metric = transform.conj().T @ transform
    angle = 0.37
    unitary = np.array(
        [[np.cos(angle), 1j * np.sin(angle)], [1j * np.sin(angle), np.cos(angle)]]
    )
    energies = np.array([-0.2, 0.3])
    h = transform.conj().T @ unitary @ np.diag(energies) @ unitary.conj().T @ transform
    green = np.linalg.inv(contour.path[:, None, None] * metric - h)
    result = contour.integrate_values(green)
    extracted = (result.conj().T - result) / (2j * np.pi)
    coefficients = np.linalg.solve(transform, unitary)
    spectral_density = (coefficients * expit(-beta * energies)) @ coefficients.conj().T
    density_error = np.max(np.abs(extracted - spectral_density))
    moment_error = np.max(np.abs(1j * contour.Rinf * green[-1] - np.linalg.inv(metric)))
    assert density_error < 1e-10
    assert moment_error < 1e-9
    assert (
        abs(np.trace(metric @ extracted).real - np.sum(expit(-beta * energies))) < 1e-10
    )
    spin_operator = np.array([[0, -1j], [1j, 0]])
    assert (
        abs(
            -np.trace(spin_operator @ result).imag / np.pi
            - np.trace(spin_operator @ spectral_density).real
        )
        < 1e-10
    )

    d, strength = 0.2, 0.1
    values = 2 * strength**2 / ((contour.path - d) * (contour.path + d))
    response = -contour.integrate_values(values).imag / (2 * np.pi)
    expected_response = strength**2 * (expit(beta * d) - expit(-beta * d)) / (-2 * d)
    response_error = abs(response - expected_response)
    assert response_error < 1e-10
    for nz, tolerance in ((12, 1e-7), (30, 1e-9)):
        auxiliary_contour = mycfr.CFR2(nz=nz, T=0.05 / kB)
        for fermi in (0.0, 6.3):
            levels = np.array([-1.0, 1.0]) + fermi
            g = 1 / (auxiliary_contour.path[:, None] + fermi - levels)
            value = -auxiliary_contour.integrate_values(
                2 * 0.1**2 * g[:, 0] * g[:, 1]
            ).imag / (2 * np.pi)
            assert abs(value + 0.005) < tolerance
    print(f"source={mycfr.__file__}")
    print(
        f"scalar_error={scalar_error:.3e} density_error={density_error:.3e} "
        f"moment_error={moment_error:.3e} response_error={response_error:.3e}"
    )


def main():
    results = derive()
    numerical_checks()
    return results


if __name__ == "__main__":
    verified = main()
    print(
        f"Verified {len(verified)} symbolic results and independent CFR2 numerical oracles"
    )
