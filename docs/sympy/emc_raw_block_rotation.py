#!/usr/bin/env python3
"""Assertion-checked EMC authority for raw full-block frozen response.

The symbolic equal-endpoint ansatz is a named fixed-lab-basis approximation,
not a derivative uniquely determined by a saved Hamiltonian.  The numerical
oracle independently builds a commensurate real-space supercell by finite
SU(2) rotations, then projects k to k+q.  It imports no TB2J or HamiltonIO
implementation.

Run from the TB2J root:
    python docs/sympy/emc_raw_block_rotation.py

The registry protocol is:
    python docs/sympy/emc_raw_block_rotation.py --kosmic-json

Both entry points execute the complete assertion suite.  The JSON entry point
prints only the Kosmic version-1 named SymPy-results payload.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass

import numpy as np
import sympy as sp

# Numerical oracle protocol.  q and k lie on the N=8 supercell mesh, while
# q != -q modulo a primitive reciprocal vector, so cosine and sine projections
# select an unaliased complex +q channel.
SUPER_CELL = 8
Q_X = 1.0 / 4.0
K_X = 1.0 / 8.0
ANGLE_STEP = 2.0e-6
ZERO_REFERENCE_THRESHOLD = 1.0e-12
EXACT_ZERO_ABS_TOL = 5.0e-9
NONZERO_REL_TOL = 5.0e-8
GATE_CURVATURE_STEP = 2.0e-3
GATE_CURVATURE_TOL = 2.0e-6


# ---------------------------------------------------------------------------
# Exact symbolic authority
# ---------------------------------------------------------------------------

I = sp.I
I2 = sp.eye(2)
SY = sp.Matrix([[0, -I], [I, 0]])
HALF = sp.Rational(1, 2)


def symbolic_rotation(theta: sp.Symbol) -> sp.Matrix:
    """U(theta)=exp(-i theta sigma_y/2), with real theta."""
    return sp.cos(theta / 2) * I2 - I * sp.sin(theta / 2) * SY


def symbolic_generator(magnetic: sp.Matrix) -> sp.Matrix:
    """Derivative at zero of U(theta) B U(theta)^dagger."""
    return -I * (SY * magnetic - magnetic * SY) / 2


def assert_symbolic_zero(value: sp.Expr | sp.MatrixBase) -> None:
    entries = list(value) if isinstance(value, sp.MatrixBase) else [value]
    for entry in entries:
        reduced = sp.trigsimp(sp.simplify(sp.expand(entry)))
        assert reduced == 0, reduced


def derive_expressions() -> dict[str, sp.Expr | sp.Matrix]:
    """Return named exact results; constructing them also asserts the algebra."""
    w00, w01, w10, w11 = sp.symbols("w00 w01 w10 w11", complex=True)
    W = sp.Matrix([[w00, w01], [w10, w11]])
    h0 = sp.trace(W) / 2
    B = W - h0 * I2
    theta_a, theta_b = sp.symbols("theta_a theta_b", real=True)
    Ua, Ub = symbolic_rotation(theta_a), symbolic_rotation(theta_b)
    equal_endpoint_block = h0 * I2 + HALF * Ua * B * Ua.H + HALF * Ub * B * Ub.H
    partner_block = W.H
    partner_h0 = sp.conjugate(h0)
    partner_B = B.H
    partner_rotated = (
        partner_h0 * I2 + HALF * Ua * partner_B * Ua.H + HALF * Ub * partner_B * Ub.H
    )
    common_u_block = h0 * I2 + Ua * B * Ua.H
    one_endpoint_block = h0 * I2 + Ua * B * Ua.H
    first_response = symbolic_generator(B)

    q, k, R, T, C_i, C_j = sp.symbols("q k R T C_i C_j", real=True)
    tau_i, tau_j = sp.symbols("tau_i tau_j", real=True)
    image_displacement = R + T
    phase_left = sp.exp(2 * sp.pi * I * q * C_i)
    phase_right = sp.exp(2 * sp.pi * I * q * (image_displacement + C_j))
    phase_site_left = sp.exp(2 * sp.pi * I * q * (C_i + tau_i))
    phase_site_right = sp.exp(2 * sp.pi * I * q * (image_displacement + C_j + tau_j))
    fourier_electronic_phase = sp.exp(2 * sp.pi * I * k * image_displacement)
    complex_q_block_response = (
        fourier_electronic_phase * HALF * (phase_left + phase_right) * first_response
    )
    same_endpoint_q_response = phase_left * first_response
    half_cell_cancellation = HALF * (1 + sp.exp(2 * sp.pi * I * HALF)) * first_response
    ndegen, n_images = sp.symbols("ndegen n_images", positive=True, integer=True)
    image_weight_once = 1 / (ndegen * n_images)

    assert_symbolic_zero(sp.trace(B))
    assert_symbolic_zero(W - h0 * I2 - B)
    assert_symbolic_zero(sp.trace(equal_endpoint_block) - sp.trace(W))
    assert_symbolic_zero(
        equal_endpoint_block.diff(theta_a).subs({theta_a: 0, theta_b: 0})
        - HALF * first_response
    )
    assert_symbolic_zero(
        equal_endpoint_block.diff(theta_b).subs({theta_a: 0, theta_b: 0})
        - HALF * first_response
    )
    assert_symbolic_zero(equal_endpoint_block.subs(theta_b, theta_a) - common_u_block)
    assert_symbolic_zero(partner_rotated - equal_endpoint_block.H)
    assert_symbolic_zero(
        one_endpoint_block.diff(theta_a).subs(theta_a, 0) - first_response
    )
    assert_symbolic_zero(half_cell_cancellation)
    assert sp.simplify(HALF + HALF - 1) == 0

    return {
        "block_decomposition": W,
        "cell_endpoint_phase_left": phase_left,
        "cell_endpoint_phase_right": phase_right,
        "common_u_full_block": common_u_block,
        "complex_q_k_to_k_plus_q_response": complex_q_block_response,
        "equal_endpoint_frozen_block": equal_endpoint_block,
        "equal_endpoint_first_order_generator": first_response,
        "equal_endpoint_weight_sum": HALF + HALF,
        "effective_image_displacement": image_displacement,
        "hamiltonio_convention2_electronic_phase": fourier_electronic_phase,
        "hermitian_partner_block": partner_block,
        "hermitian_partner_equal_endpoint_block": partner_rotated,
        "image_weight_applied_once": image_weight_once,
        "same_endpoint_first_order_response": same_endpoint_q_response,
        "same_sublattice_half_cell_cancellation": half_cell_cancellation,
        "site_referenced_phase_left": phase_site_left,
        "site_referenced_phase_right": phase_site_right,
        "spin_dependent_traceless_part": B,
        "spin_scalar_part_fixed": h0,
    }


def derive() -> list[dict[str, str]]:
    """Kosmic MathDoc API: named exact SymPy expressions in lexical order."""
    expressions = derive_expressions()
    return [
        {
            "name": name,
            "srepr": sp.srepr(expressions[name]),
            "latex": sp.latex(expressions[name]),
        }
        for name in sorted(expressions)
    ]


# ---------------------------------------------------------------------------
# Independent finite-angle supercell oracle (NumPy only)
# ---------------------------------------------------------------------------

I2_NP = np.eye(2, dtype=complex)
SY_NP = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def numeric_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def numeric_generator(magnetic: np.ndarray) -> np.ndarray:
    return -0.5j * (SY_NP @ magnetic - magnetic @ SY_NP)


def add_cells(*vectors: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sum(vector[axis] for vector in vectors) for axis in range(3))


def negate_cell(vector: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(-component for component in vector)


def dot3(left: tuple[float, float, float], right: tuple[int, int, int]) -> float:
    return float(sum(left[axis] * right[axis] for axis in range(3)))


@dataclass(frozen=True)
class EffectiveImage:
    """One HamiltonIO effective image of a shell/pair, with its source weight."""

    row: int
    col: int
    R: tuple[int, int, int]
    T: tuple[int, int, int]
    W: np.ndarray
    ndegen: int
    n_images: int

    @property
    def displacement(self) -> tuple[int, int, int]:
        return add_cells(self.R, self.T)

    @property
    def weight(self) -> float:
        return 1.0 / (self.ndegen * self.n_images)


def oriented_images(
    images: list[EffectiveImage],
):
    """Yield a block and its explicit Hermitian partner, without symmetrizing W."""
    for image in images:
        d = image.displacement
        yield image.row, image.col, d, image.W, image.weight
        yield (
            image.col,
            image.row,
            negate_cell(d),
            image.W.conj().T,
            image.weight,
        )


def endpoint_records(
    row: int,
    col: int,
    displacement: tuple[int, int, int],
    owners: list[tuple[str, tuple[int, int, int]]],
    base_cell: tuple[int, int, int] = (0, 0, 0),
    *,
    collapse_same_owner_cells: bool = False,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Deduplicate by (owner atom, actual owner lattice cell), never site only."""
    owner_i, offset_i = owners[row]
    owner_j, offset_j = owners[col]
    cell_i = add_cells(base_cell, offset_i)
    cell_j = add_cells(base_cell, displacement, offset_j)
    if collapse_same_owner_cells and owner_i == owner_j:
        # Deliberately wrong negative control: throw away the intercell identity.
        cell_j = cell_i
    candidates = ((owner_i, cell_i), (owner_j, cell_j))
    unique: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[tuple[str, tuple[int, int, int]]] = set()
    for endpoint in candidates:
        if endpoint not in seen:
            seen.add(endpoint)
            unique.append(endpoint)
    assert unique
    return unique


def rotate_finite_block(
    W: np.ndarray,
    endpoints: list[tuple[str, tuple[int, int, int]]],
    angle_for_endpoint,
) -> np.ndarray:
    """Finite physical response: scalar fixed, equal weighted full conjugations."""
    scalar = np.trace(W) / 2.0
    magnetic = W - scalar * I2_NP
    out = scalar * I2_NP
    share = 1.0 / len(endpoints)
    for owner, cell in endpoints:
        U = numeric_rotation(float(angle_for_endpoint(owner, cell)))
        out = out + share * (U @ magnetic @ U.conj().T)
    return out


def build_real_space_supercell(
    images: list[EffectiveImage],
    owners: list[tuple[str, tuple[int, int, int]]],
    *,
    q: tuple[float, float, float],
    amplitude: float,
    pattern: str,
) -> np.ndarray:
    """Build H(alpha) directly from finite-angle real-space spinor blocks."""
    if pattern not in {"cos", "sin"}:
        raise ValueError("pattern must be 'cos' or 'sin'")
    norb = len(owners)
    H = np.zeros((2 * SUPER_CELL * norb, 2 * SUPER_CELL * norb), dtype=complex)

    def angle_for_endpoint(_owner: str, cell: tuple[int, int, int]) -> float:
        phase = 2.0 * np.pi * dot3(q, cell)
        modulation = np.cos(phase) if pattern == "cos" else np.sin(phase)
        return amplitude * float(modulation)

    for n in range(SUPER_CELL):
        base_cell = (n, 0, 0)
        for row, col, displacement, W, shell_weight in oriented_images(images):
            endpoints = endpoint_records(
                row,
                col,
                displacement,
                owners,
                base_cell,
            )
            block = shell_weight * rotate_finite_block(W, endpoints, angle_for_endpoint)
            row_cell = n
            col_cell = (n + displacement[0]) % SUPER_CELL
            row_start = 2 * (row_cell * norb + row)
            col_start = 2 * (col_cell * norb + col)
            H[row_start : row_start + 2, col_start : col_start + 2] += block
    return H


def project_k_to_k_plus_q(
    H: np.ndarray,
    norb: int,
    *,
    k_x: float,
    q_x: float,
) -> np.ndarray:
    """Project with |k>_n=exp(+2pi i k n)|n>, yielding <k+q|H|k>."""
    projected = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for n in range(SUPER_CELL):
        row_phase = np.exp(-2j * np.pi * (k_x + q_x) * n)
        for m in range(SUPER_CELL):
            col_phase = np.exp(2j * np.pi * k_x * m)
            factor = row_phase * col_phase / SUPER_CELL
            for row in range(norb):
                row_start = 2 * (n * norb + row)
                out_row = 2 * row
                for col in range(norb):
                    col_start = 2 * (m * norb + col)
                    out_col = 2 * col
                    projected[out_row : out_row + 2, out_col : out_col + 2] += (
                        factor
                        * H[
                            row_start : row_start + 2,
                            col_start : col_start + 2,
                        ]
                    )
    return projected


def hamiltonio_convention2_block_sum(
    images: list[EffectiveImage],
    norb: int,
    *,
    k_x: float,
) -> np.ndarray:
    """Explicit HamiltonIO convention-2 sum with each image weight once."""
    Hk = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for row, col, displacement, W, shell_weight in oriented_images(images):
        phase = np.exp(2j * np.pi * k_x * displacement[0])
        row_start, col_start = 2 * row, 2 * col
        Hk[row_start : row_start + 2, col_start : col_start + 2] += (
            shell_weight * phase * W
        )
    return Hk


def finite_difference_projection(
    images: list[EffectiveImage],
    owners: list[tuple[str, tuple[int, int, int]]],
    *,
    k_x: float,
    q_x: float,
    pattern: str,
) -> np.ndarray:
    """Central difference of the independently finite-rotated supercell."""
    plus = build_real_space_supercell(
        images,
        owners,
        q=(q_x, 0.0, 0.0),
        amplitude=ANGLE_STEP,
        pattern=pattern,
    )
    minus = build_real_space_supercell(
        images,
        owners,
        q=(q_x, 0.0, 0.0),
        amplitude=-ANGLE_STEP,
        pattern=pattern,
    )
    for matrix in (plus, minus):
        assert np.max(np.abs(matrix - matrix.conj().T)) < 2.0e-12
    projected_plus = project_k_to_k_plus_q(plus, len(owners), k_x=k_x, q_x=q_x)
    projected_minus = project_k_to_k_plus_q(minus, len(owners), k_x=k_x, q_x=q_x)
    return (projected_plus - projected_minus) / (2.0 * ANGLE_STEP)


def analytic_k_to_k_plus_q_response(
    images: list[EffectiveImage],
    owners: list[tuple[str, tuple[int, int, int]]],
    *,
    k_x: float,
    q_x: float,
    collapse_same_owner_cells: bool = False,
) -> np.ndarray:
    """Candidate Fourier response used only as the comparison target."""
    norb = len(owners)
    response = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for row, col, displacement, W, shell_weight in oriented_images(images):
        endpoints = endpoint_records(
            row,
            col,
            displacement,
            owners,
            collapse_same_owner_cells=collapse_same_owner_cells,
        )
        magnetic = W - (np.trace(W) / 2.0) * I2_NP
        derivative = numeric_generator(magnetic)
        endpoint_response = np.zeros((2, 2), dtype=complex)
        share = 1.0 / len(endpoints)
        for _owner, cell in endpoints:
            phase_q = np.exp(2j * np.pi * dot3((q_x, 0.0, 0.0), cell))
            endpoint_response += share * phase_q * derivative
        phase_k = np.exp(2j * np.pi * k_x * displacement[0])
        row_start, col_start = 2 * row, 2 * col
        response[row_start : row_start + 2, col_start : col_start + 2] += (
            shell_weight * phase_k * endpoint_response
        )
    return response


def comparison_metrics(
    observed: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float | str]:
    """Use absolute error near zero, relative error only for nonzero references."""
    difference = observed - expected
    max_abs = float(np.max(np.abs(difference)))
    reference_norm = float(np.linalg.norm(expected))
    error_norm = float(np.linalg.norm(difference))
    if reference_norm <= ZERO_REFERENCE_THRESHOLD:
        assert max_abs <= EXACT_ZERO_ABS_TOL, (max_abs, EXACT_ZERO_ABS_TOL)
        return {
            "comparison_rule": "exact/near-zero reference: absolute max norm",
            "max_abs_error": max_abs,
            "reference_norm": reference_norm,
        }
    relative_error = error_norm / reference_norm
    assert relative_error <= NONZERO_REL_TOL, (
        relative_error,
        NONZERO_REL_TOL,
    )
    return {
        "comparison_rule": "nonzero reference: Frobenius relative norm",
        "max_abs_error": max_abs,
        "reference_norm": reference_norm,
        "relative_error": relative_error,
    }


# ---------------------------------------------------------------------------
# Independent uniform-response and pure-gauge gates
# ---------------------------------------------------------------------------


def response_hamiltonian(
    H: np.ndarray,
    owners: list[tuple[str, tuple[int, int, int]]],
    angle_by_owner: dict[str, float],
) -> np.ndarray:
    """Actual declared equal-endpoint map at q=0 for a finite orbital matrix."""
    norb = len(owners)
    out = np.zeros_like(H, dtype=complex)

    def angle_for_endpoint(owner: str, _cell: tuple[int, int, int]) -> float:
        return float(angle_by_owner.get(owner, 0.0))

    for row in range(norb):
        for col in range(norb):
            row_start, col_start = 2 * row, 2 * col
            block = H[row_start : row_start + 2, col_start : col_start + 2]
            endpoints = endpoint_records(row, col, (0, 0, 0), owners, (0, 0, 0))
            out[row_start : row_start + 2, col_start : col_start + 2] = (
                rotate_finite_block(block, endpoints, angle_for_endpoint)
            )
    return out


def block_diagonal_spin_rotation(angles: list[float]) -> np.ndarray:
    G = np.zeros((2 * len(angles), 2 * len(angles)), dtype=complex)
    for orbital, angle in enumerate(angles):
        start = 2 * orbital
        G[start : start + 2, start : start + 2] = numeric_rotation(angle)
    return G


def occupied_energy(H: np.ndarray, n_occupied: int = 2) -> float:
    return float(np.linalg.eigvalsh(H)[:n_occupied].sum())


def energy_curvature(builder, step: float = GATE_CURVATURE_STEP) -> float:
    e0 = occupied_energy(builder(0.0))
    ep = occupied_energy(builder(step))
    em = occupied_energy(builder(-step))
    return float((ep + em - 2.0 * e0) / (step * step))


def run_checks() -> dict[str, float | int | bool | str]:
    """Run all exact assertions, independent finite-angle oracles and controls."""
    derive_expressions()

    # Physical endpoint identity: same owner/same cell coalesces; different
    # owner or different actual cell remains distinct, regardless of orbital ID.
    shared_owner = [("A", (0, 0, 0)), ("A", (0, 0, 0))]
    assert len(endpoint_records(0, 1, (0, 0, 0), shared_owner)) == 1
    assert len(endpoint_records(0, 1, (2, 0, 0), shared_owner)) == 2
    distinct_owners = [("A", (0, 0, 0)), ("L", (0, 0, 0))]
    assert len(endpoint_records(0, 1, (0, 0, 0), distinct_owners)) == 2

    # Arbitrary complex, nonsymmetric directed W; its partner is built as W^dagger.
    W = np.array(
        [
            [0.37 + 0.11j, 0.23 - 0.19j],
            [-0.08 + 0.17j, -0.21 - 0.06j],
        ],
        dtype=complex,
    )
    assert np.linalg.norm(W - W.conj().T) > 0.1
    images = [
        EffectiveImage(0, 1, (1, 0, 0), (0, 0, 0), W, ndegen=2, n_images=2),
        EffectiveImage(0, 1, (1, 0, 0), (1, 0, 0), W, ndegen=2, n_images=2),
    ]
    owners = [
        ("magnetic-A", (1, 0, 0)),
        ("ligand-L", (-1, 0, 0)),
    ]
    assert all(image.displacement == add_cells(image.R, image.T) for image in images)
    assert all(image.weight == 1.0 / (2 * 2) for image in images)
    assert all(
        np.allclose(partner, W.conj().T)
        for row, col, d, partner, _weight in oriented_images(images)
        if (row, col, d) == (1, 0, (-1, 0, 0))
    )

    # An undeformed direct-supercell projection must reproduce the explicit
    # convention-2 sum; this pins +k and the single ndegen*N_T weight.
    unrotated_supercell = build_real_space_supercell(
        images,
        owners,
        q=(0.0, 0.0, 0.0),
        amplitude=0.0,
        pattern="cos",
    )
    projected_hk = project_k_to_k_plus_q(
        unrotated_supercell, len(owners), k_x=K_X, q_x=0.0
    )
    explicit_hk = hamiltonio_convention2_block_sum(images, len(owners), k_x=K_X)
    fourier_weight_error = float(np.max(np.abs(projected_hk - explicit_hk)))
    assert fourier_weight_error < 2.0e-12

    cosine_fd = finite_difference_projection(
        images, owners, k_x=K_X, q_x=Q_X, pattern="cos"
    )
    sine_fd = finite_difference_projection(
        images, owners, k_x=K_X, q_x=Q_X, pattern="sin"
    )
    # The real cosine and sine patterns each supply half the +q channel;
    # this convention combines them as V_cos + i V_sin, without another 2.
    finite_angle_channel = cosine_fd + 1j * sine_fd
    analytic_channel = analytic_k_to_k_plus_q_response(images, owners, k_x=K_X, q_x=Q_X)
    oracle_comparison = comparison_metrics(finite_angle_channel, analytic_channel)

    # Negative control: the opposite sine sign selects the wrong q channel and
    # therefore cannot reproduce the k -> k+q projection.
    wrong_sign_channel = cosine_fd - 1j * sine_fd
    analytic_norm = float(np.linalg.norm(analytic_channel))
    assert analytic_norm > 1.0e-6
    wrong_sign_relative = float(
        np.linalg.norm(wrong_sign_channel - analytic_channel) / analytic_norm
    )
    assert wrong_sign_relative > 0.9, wrong_sign_relative

    # Same-owner intercell bond with q.d=1/2: physical cells remain distinct,
    # equal endpoint generators cancel.  The collapsed-site oracle is wrong.
    cancellation_image = EffectiveImage(
        0,
        0,
        (2, 0, 0),
        (0, 0, 0),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        ndegen=1,
        n_images=1,
    )
    same_cell_owner = [("same-atom", (0, 0, 0))]
    cancellation_analytic = analytic_k_to_k_plus_q_response(
        [cancellation_image], same_cell_owner, k_x=0.0, q_x=Q_X
    )
    assert np.max(np.abs(cancellation_analytic)) < 1.0e-14
    cancellation_cos = finite_difference_projection(
        [cancellation_image],
        same_cell_owner,
        k_x=0.0,
        q_x=Q_X,
        pattern="cos",
    )
    cancellation_sin = finite_difference_projection(
        [cancellation_image],
        same_cell_owner,
        k_x=0.0,
        q_x=Q_X,
        pattern="sin",
    )
    cancellation_fd = cancellation_cos + 1j * cancellation_sin
    cancellation_comparison = comparison_metrics(cancellation_fd, cancellation_analytic)
    collapsed_candidate = analytic_k_to_k_plus_q_response(
        [cancellation_image],
        same_cell_owner,
        k_x=0.0,
        q_x=Q_X,
        collapse_same_owner_cells=True,
    )
    collapsed_control_norm = float(np.linalg.norm(collapsed_candidate))
    collapsed_control_gap = float(np.linalg.norm(collapsed_candidate - cancellation_fd))
    assert collapsed_control_norm > 0.1
    assert collapsed_control_gap > 0.1

    # Structural common-U identity is constructed independently as a full
    # basis similarity and includes a ligand owner.  A transverse derivative
    # is deliberately nonzero and is not a gate failure.
    onsite_a = 1.3 * np.array([[1, 0], [0, -1]], dtype=complex) + 0.2 * np.array(
        [[0, 1], [1, 0]], dtype=complex
    )
    onsite_l = -0.8 * np.array([[1, 0], [0, -1]], dtype=complex) + 0.3 * SY_NP
    H_structural = np.block([[onsite_a, W], [W.conj().T, onsite_l]])
    structural_owners = [
        ("magnetic-A", (0, 0, 0)),
        ("ligand-L", (0, 0, 0)),
    ]
    common_angle = 0.17
    global_U = np.kron(np.eye(2), numeric_rotation(common_angle))
    structural_rotated = global_U @ H_structural @ global_U.conj().T
    structural_spectrum_drift = float(
        np.max(
            np.abs(
                np.linalg.eigvalsh(structural_rotated)
                - np.linalg.eigvalsh(H_structural)
            )
        )
    )
    structural_from_response = response_hamiltonian(
        H_structural,
        structural_owners,
        {"magnetic-A": common_angle, "ligand-L": common_angle},
    )
    assert np.max(np.abs(structural_from_response - structural_rotated)) < 2.0e-13
    assert structural_spectrum_drift < 2.0e-12
    structural_curvature = energy_curvature(
        lambda theta: (
            block_diagonal_spin_rotation([theta, theta])
            @ H_structural
            @ block_diagonal_spin_rotation([theta, theta]).conj().T
        )
    )
    assert abs(structural_curvature) < GATE_CURVATURE_TOL
    generator4 = np.kron(np.eye(2), SY_NP / 2.0)
    transverse_derivative = -1j * (
        generator4 @ H_structural - H_structural @ generator4
    )
    transverse_norm = float(np.linalg.norm(transverse_derivative))
    assert transverse_norm > 1.0e-6

    # Pure-gauge endpoint dressing G^dagger H G is separately isospectral and
    # is not the fixed-basis response: it rotates scalar inter-owner hopping.
    gauge = block_diagonal_spin_rotation([0.31, -0.17])
    pure_gauge = gauge.conj().T @ H_structural @ gauge
    gauge_spectrum_drift = float(
        np.max(
            np.abs(np.linalg.eigvalsh(pure_gauge) - np.linalg.eigvalsh(H_structural))
        )
    )
    fixed_basis_physical = response_hamiltonian(
        H_structural,
        structural_owners,
        {"magnetic-A": 0.31, "ligand-L": -0.17},
    )
    gauge_physical_matrix_gap = float(np.max(np.abs(pure_gauge - fixed_basis_physical)))
    assert gauge_spectrum_drift < 2.0e-12
    assert gauge_physical_matrix_gap > 1.0e-2

    # Actual response-map uniform gate: all owners (including the ligand) pass;
    # a restricted map that freezes the polarized ligand is explicitly uncertified.
    b_field, scalar_hopping = 2.0, 1.0
    sz_np = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    H_gate = np.block(
        [
            [b_field * sz_np, scalar_hopping * I2_NP],
            [scalar_hopping * I2_NP, b_field * sz_np],
        ]
    )
    gate_owners = [("A", (0, 0, 0)), ("ligand", (0, 0, 0))]
    all_owner_curvature = energy_curvature(
        lambda theta: response_hamiltonian(
            H_gate, gate_owners, {"A": theta, "ligand": theta}
        )
    )
    assert abs(all_owner_curvature) < GATE_CURVATURE_TOL
    restricted_curvature = energy_curvature(
        lambda theta: response_hamiltonian(H_gate, gate_owners, {"A": theta})
    )
    exact_restricted_curvature = (
        -b_field * scalar_hopping**2 / (2.0 * (b_field**2 - scalar_hopping**2))
    )
    assert abs(restricted_curvature - exact_restricted_curvature) < 2.0e-5
    restricted_physical_gate_passes = abs(restricted_curvature) < GATE_CURVATURE_TOL
    assert not restricted_physical_gate_passes

    return {
        "symbolic_assertions": "passed",
        "physical_endpoint_identity": "owner atom plus actual owner lattice cell",
        "shared_orbital_endpoint_count": 1,
        "same_owner_different_cell_endpoint_count": 2,
        "distinct_owner_same_cell_endpoint_count": 2,
        "directed_hopping_is_nonhermitian": True,
        "hermitian_partner": "explicit W-dagger, endpoints exchanged, same image weight",
        "image_effective_displacements": ",".join(
            str(image.displacement[0]) for image in images
        ),
        "weight_per_effective_image": images[0].weight,
        "convention2_fourier_weight_error": fourier_weight_error,
        "supercell_cells": SUPER_CELL,
        "commensurate_q_x": Q_X,
        "commensurate_k_x": K_X,
        "finite_difference_angle_step_rad": ANGLE_STEP,
        "complex_q_channel_normalization": "V_cos + i V_sin, no extra factor of 2",
        "nonzero_oracle_relative_error": float(
            oracle_comparison.get("relative_error", 0.0)
        ),
        "oracle_error_rule": str(oracle_comparison["comparison_rule"]),
        "wrong_q_sign_relative_mismatch": wrong_sign_relative,
        "equal_generator_q_dot_d": 0.5,
        "cancellation_fd_max_abs": float(np.max(np.abs(cancellation_fd))),
        "cancellation_error_rule": str(cancellation_comparison["comparison_rule"]),
        "collapsed_owner_control_norm": collapsed_control_norm,
        "collapsed_owner_control_gap_vs_oracle": collapsed_control_gap,
        "structural_common_u_spectrum_drift": structural_spectrum_drift,
        "structural_common_u_energy_curvature": structural_curvature,
        "structural_common_u_transverse_derivative_norm": transverse_norm,
        "actual_all_owner_map_energy_curvature": all_owner_curvature,
        "restricted_map_energy_curvature": restricted_curvature,
        "restricted_map_expected_curvature": exact_restricted_curvature,
        "restricted_map_physical_gate_pass": restricted_physical_gate_passes,
        "restricted_map_certification": "withheld: physical uniform-response gate fails",
        "pure_gauge_spectrum_drift": gauge_spectrum_drift,
        "pure_gauge_vs_fixed_basis_response_max_gap": gauge_physical_matrix_gap,
        "zero_reference_abs_tolerance": EXACT_ZERO_ABS_TOL,
        "nonzero_reference_relative_tolerance": NONZERO_REL_TOL,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kosmic-json",
        action="store_true",
        help="emit version-1 named SymPy results after running every assertion",
    )
    args = parser.parse_args()
    observations = run_checks()
    if args.kosmic_json:
        payload = {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "sympy_version": sp.__version__,
            "results": derive(),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"SymPy {sp.__version__}; EMC raw-block authority assertions: OK")
        for name, value in sorted(observations.items()):
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
