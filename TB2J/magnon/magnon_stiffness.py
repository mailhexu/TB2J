"""Spin-wave stiffness from the magnon dispersion.

The spin-wave stiffness is the leading coefficient of the long-wavelength
expansion of the acoustic magnon branch around its Goldstone (or gap) point,

    omega(k) = omega0 + k^T A k      (quadratic branch, e.g. a ferromagnet)
    omega(k) = omega0 + v . k        (linear branch, e.g. an antiferromagnet)

`A` is a cartesian tensor in meV A^2 and `v` a cartesian vector in meV A.  The
magnon convention is the one of `TB2J.magnon.magnon3` (H = -sum_<ij> J S_i.S_j
with unordered bonds once, J in meV, |S| = |magmom|/2); the sign of `A` and
`v` follows the sign of the band energies that the module reports.

The tensor is extracted from the exact `eig(gH)` spectrum, not from the
shifted-Cholesky route of `Magnon._magnon_energies` (which is inaccurate and
returns negative frequencies at a ferromagnetic Goldstone point).

The real-space sum rule `A = -(S/2) sum_bonds J v v` is the textbook route
(Liechtenstein-Katsnelson-Gubanov; Pajda et al., PRB 64, 174402 (2001)) and is
available as a cross-check.  It is valid only for a single-sublattice (Bravais)
ferromagnet: for a multi-sublattice lattice the exact LSWT band is not the
bond sum.  It also needs the *true* bond vectors (lattice offset plus
intra-cell displacement), which the magnon calculator does not carry, so the
caller supplies them.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sl

__all__ = ["SpinStiffness", "get_spin_stiffness", "stiffness_sum_rule"]

EV_TO_MEV = 1e3


@dataclass
class SpinStiffness:
    """Long-wavelength stiffness of the acoustic magnon branch.

    Attributes
    ----------
    A : numpy.ndarray or None
        (3, 3) cartesian stiffness tensor in meV A^2.  ``None`` when the
        branch is linear (an antiferromagnet), where no quadratic coefficient
        exists.
    order : int
        2 for a quadratic branch, 1 for a linear one.
    velocity : numpy.ndarray or None
        (3,) directional spin-wave speeds d omega/d|k| along the cartesian
        axes, in meV A (linear branch only).
    k0 : numpy.ndarray
        Fractional expansion point (in units of 2 pi).
    step : float
        Finite-difference step in cartesian 1/A.
    omega0 : float
        Acoustic branch energy at ``k0`` in meV (0 at a Goldstone point).
    sum_rule : numpy.ndarray or None
        (3, 3) stiffness tensor in meV A^2 from the real-space sum rule, when
        the caller supplied the bond vectors.
    """

    A: np.ndarray | None
    order: int
    velocity: np.ndarray | None
    k0: np.ndarray
    step: float
    omega0: float
    sum_rule: np.ndarray | None = None

    def __str__(self) -> str:
        lines = [f"expansion point k0 (fractional): {np.round(self.k0, 6).tolist()}"]
        lines.append(f"acoustic branch energy at k0: {self.omega0:.6g} meV")
        if self.order == 2:
            lines.append("branch order: 2 (quadratic)")
            lines.append("stiffness tensor A (meV A^2):")
            for row in self.A:
                lines.append("  [" + "  ".join(f"{v: .8g}" for v in row) + "]")
            lines.append(
                "directional stiffness A(n,n) (meV A^2): "
                + ", ".join(f"{d}={self.A[i, i]: .6g}" for i, d in enumerate("xyz"))
            )
        else:
            lines.append("branch order: 1 (linear)")
            lines.append("directional spin-wave speeds d omega/d|k| (meV A):")
            lines.append("  [" + "  ".join(f"{v: .8g}" for v in self.velocity) + "]")
            lines.append(
                "speed |v| = {:.6g} meV A".format(np.linalg.norm(self.velocity))
            )
        if self.sum_rule is not None:
            lines.append("sum-rule tensor A (meV A^2), Bravais cross-check:")
            for row in self.sum_rule:
                lines.append("  [" + "  ".join(f"{v: .8g}" for v in row) + "]")
        return "\n".join(lines)


def _k_cart_to_frac(k_cart, cell):
    """Cartesian k (1/A) -> fractional k (units of 2 pi).

    The magnon code's Fourier phase is exp(-2 pi i R.q) with R in lattice
    coordinates, so the physical wavevector satisfies k.a_i = 2 pi q_i and
    q = cell @ k / (2 pi) (cell rows are the lattice vectors).
    """
    return np.asarray(cell, dtype=float) @ np.asarray(k_cart, dtype=float) / (
        2.0 * np.pi
    )


def _branch_energy(M, k_frac):
    """Energy (meV) of the acoustic magnon branch at fractional k.

    Uses the exact route: the eigenvalues of g H come in +/- pairs, so the
    acoustic magnon energy is the smallest absolute eigenvalue (zero at a
    Goldstone point, the gap for a gapped reference state).
    """
    H = M.Hq(np.atleast_2d(k_frac))
    n = H.shape[-1] // 2
    g = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    w = sl.eigvals(g @ H[0]).real
    return np.abs(w).min() * EV_TO_MEV


def stiffness_sum_rule(J_iso, S, bond_vectors):
    """Real-space stiffness sum rule (Bravais ferromagnets only).

    Parameters
    ----------
    J_iso : array_like
        Isotropic exchange of each bond in meV (the same sign convention as
        the magnon module).
    S : float
        Spin length (|S| = |magmom|/2).
    bond_vectors : array_like
        (nbonds, 3) true bond vectors in Angstrom (lattice offset plus
        intra-cell displacement).

    Returns
    -------
    numpy.ndarray
        (3, 3) stiffness tensor in meV A^2, with the sign convention of the
        magnon module.
    """
    J_iso = np.asarray(J_iso, dtype=float).ravel()
    vectors = np.asarray(bond_vectors, dtype=float).reshape(-1, 3)
    if J_iso.size != vectors.shape[0]:
        raise ValueError(
            "J_iso and bond_vectors must have the same number of bonds "
            f"(got {J_iso.size} and {vectors.shape[0]})."
        )
    A = np.zeros((3, 3))
    for J, v in zip(J_iso, vectors):
        A += J * np.outer(v, v)
    return -0.5 * float(S) * A


def get_spin_stiffness(
    M,
    k0=None,
    step=0.01,
    J_iso=None,
    bond_vectors=None,
):
    """Spin-wave stiffness of the acoustic magnon branch.

    Parameters
    ----------
    M : Magnon
        The magnon calculator (its ``cell`` must be in Angstrom).
    k0 : array_like, optional
        Fractional expansion point in units of 2 pi, default Gamma.  For an
        antiferromagnet the Goldstone point sits at the ordering vector, so
        pass it explicitly.
    step : float, optional
        Finite-difference step in cartesian 1/A, default 0.01.
    J_iso : array_like, optional
        Isotropic exchange per bond in meV; with ``bond_vectors`` it enables
        the real-space sum-rule cross-check.
    bond_vectors : array_like, optional
        (nbonds, 3) true bond vectors in Angstrom.

    Returns
    -------
    SpinStiffness
    """
    cell = np.asarray(M.cell, dtype=float)
    if step <= 0:
        raise ValueError("The finite-difference step must be positive.")
    k0 = np.zeros(3) if k0 is None else np.asarray(k0, dtype=float).ravel()
    if k0.size != 3:
        raise ValueError("k0 must have three components.")

    omega0 = _branch_energy(M, k0)

    # Symmetric stencil in cartesian k: the three axes and the six mixed
    # directions.  The stencil is symmetric under k -> -k, so the odd terms of
    # the expansion cancel and the fit returns the quadratic coefficient.
    e = np.eye(3)
    samples = []
    for i in range(3):
        for sign in (-1.0, 1.0):
            samples.append(sign * step * e[i])
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    samples.append(step * (si * e[i] + sj * e[j]))

    rows, values = [], []
    for k_cart in samples:
        k_frac = k0 + _k_cart_to_frac(k_cart, cell)
        rows.append(
            [
                k_cart[0] ** 2,
                2.0 * k_cart[0] * k_cart[1],
                2.0 * k_cart[0] * k_cart[2],
                k_cart[1] ** 2,
                2.0 * k_cart[1] * k_cart[2],
                k_cart[2] ** 2,
            ]
        )
        values.append(_branch_energy(M, k_frac) - omega0)
    rows = np.array(rows)
    values = np.array(values)

    # Branch order: for a quadratic branch the deviation grows like k^2 (ratio
    # 4 between step and step/2), for a linear branch like k (ratio 2).
    axis = np.zeros(3)
    axis[0] = 1.0
    dev_step = _branch_energy(M, k0 + _k_cart_to_frac(step * axis, cell)) - omega0
    dev_half = (
        _branch_energy(M, k0 + _k_cart_to_frac(0.5 * step * axis, cell)) - omega0
    )
    if abs(dev_half) < 1e-14:
        ratio = 4.0 if abs(dev_step) < 1e-14 else float("inf")
    else:
        ratio = dev_step / dev_half
    order = 2 if ratio > 3.0 else 1

    if order == 2:
        coef, *_ = np.linalg.lstsq(rows, values, rcond=None)
        A = np.zeros((3, 3))
        A[0, 0] = coef[0]
        A[0, 1] = A[1, 0] = coef[1]
        A[0, 2] = A[2, 0] = coef[2]
        A[1, 1] = coef[3]
        A[1, 2] = A[2, 1] = coef[4]
        A[2, 2] = coef[5]
        velocity = None
    else:
        A = None
        # Linear branch: the directional speeds d omega/d|k| along the three
        # cartesian axes.  A forward difference is used because the acoustic
        # branch of an antiferromagnet is even in k (omega ~ v|q|), so a
        # central difference would cancel.
        velocity = np.zeros(3)
        for i in range(3):
            direction = np.zeros(3)
            direction[i] = 1.0
            plus = _branch_energy(
                M, k0 + _k_cart_to_frac(step * direction, cell)
            ) - omega0
            velocity[i] = plus / step

    sum_rule = None
    if J_iso is not None and bond_vectors is not None:
        S = float(np.mean(M.Snorm))
        sum_rule = stiffness_sum_rule(J_iso, S, bond_vectors)

    return SpinStiffness(
        A=A,
        order=order,
        velocity=velocity,
        k0=k0,
        step=step,
        omega0=omega0,
        sum_rule=sum_rule,
    )
