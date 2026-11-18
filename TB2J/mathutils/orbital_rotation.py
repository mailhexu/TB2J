"""Orbital-space rotation utilities for real spherical harmonics.

These functions build the real-to-complex SH transformation and Wigner-d
rotation matrices needed to rotate the *orbital* part of a density matrix
when the spin-quantization axis changes direction.

For a correlated shell with angular momentum *l*, the orbital rotation
matrix in the real-SH basis is

.. math::

    R^{(l)}(\theta,\phi) = C^\dagger\, R_z(\phi)\, d^l(\theta)\, R_z(-\phi)\, C,

where *C* is the unitary complex-to-real SH conversion matrix and
:math:`d^l(\theta)` is the Wigner small-d matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Shell-block descriptor parsed from ORB_INDX
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellBlock:
    """One complete (l, atom, zeta) shell with its global orbital indices."""

    l: int
    indices: tuple[int, ...]


def parse_orb_indx(path: str | Path) -> tuple[ShellBlock, ...]:
    """Parse a SIESTA ``ORB_INDX`` file into a tuple of ShellBlocks.

    Each consecutive group of ``2l+1`` orbitals with the same atom/species/
    n/zeta and ordered ``m = -l .. +l`` forms one block.
    """
    path = Path(path)
    records: list[tuple[int, tuple[str, str, str, int, int, int], int]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit():
            continue
        try:
            io = int(fields[0]) - 1
            l = int(fields[6])
            m = int(fields[7])
            key = (fields[1], fields[2], fields[3], int(fields[5]), l, int(fields[8]))
        except ValueError:
            continue
        records.append((io, key, m))

    blocks: list[ShellBlock] = []
    index = 0
    while index < len(records):
        start, key, m = records[index]
        l = key[4]
        size = 2 * l + 1
        group = records[index : index + size]
        expected_m = list(range(-l, l + 1))
        if m != -l or len(group) != size:
            raise ValueError(
                f"ORB_INDX has an incomplete shell starting at orbital {start + 1}"
            )
        if [entry[1] for entry in group] != [key] * size or [
            entry[2] for entry in group
        ] != expected_m:
            raise ValueError(
                f"ORB_INDX shell starting at orbital {start + 1} is not ordered m=-l..l"
            )
        indices = tuple(entry[0] for entry in group)
        if indices != tuple(range(start, start + size)):
            raise ValueError(
                f"ORB_INDX shell starting at orbital {start + 1} is not contiguous"
            )
        blocks.append(ShellBlock(l=l, indices=indices))
        index += size
    return tuple(blocks)


# ---------------------------------------------------------------------------
# Complex → real spherical-harmonic conversion
# ---------------------------------------------------------------------------


def complex_to_real_sh(l: int) -> np.ndarray:
    """Unitary matrix C[m_complex, m_real] transforming complex Y_l^m to real SH.

    The convention follows the standard SIESTA/Quantum ESPRESSO real-SH
    ordering: ``m = -l, ..., +l`` with
    ``Y_l^R(m>0) = (Y_l^{|m|} + (-1)^m Y_l^{-|m|})/sqrt(2)``.
    """
    size = 2 * l + 1
    C = np.zeros((size, size), dtype=complex)
    s2 = np.sqrt(2.0)
    for m in range(-l, l + 1):
        if m == 0:
            C[l, l] = 1.0
        elif m > 0:
            C[m + l, m + l] = (-1) ** m / s2
            C[-m + l, m + l] = 1.0 / s2
        else:
            C[m + l, m + l] = 1j / s2
            C[-m + l, m + l] = -1j * (-1) ** (-m) / s2
    return C


# ---------------------------------------------------------------------------
# Real-SH rotation for a 90° z→x or z→y rotation (hard-coded, exact)
# ---------------------------------------------------------------------------


def real_sh_rotation_90(l: int) -> np.ndarray:
    """Real-SH rotation matrix for a 90° rotation from *z* to *x*.

    Used when only the three canonical directions (x, y, z) are needed,
    avoiding the sympy dependency.
    """
    root3 = np.sqrt(3.0)
    root6 = np.sqrt(6.0)
    root10 = np.sqrt(10.0)
    root15 = np.sqrt(15.0)
    rotations = {
        0: ((1.0,),),
        1: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        2: (
            (0.0, -1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -0.5, 0.0, root3 / 2.0),
            (0.0, 0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, root3 / 2.0, 0.0, 0.5),
        ),
        3: (
            (0.25, 0.0, root15 / 4.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (root15 / 4.0, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, -root6 / 4.0, 0.0, root10 / 4.0),
            (0.0, 0.0, 0.0, root6 / 4.0, 0.0, -root10 / 4.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, root10 / 4.0, 0.0, root6 / 4.0),
            (0.0, 0.0, 0.0, -root10 / 4.0, 0.0, -root6 / 4.0, 0.0),
        ),
    }
    try:
        return np.array(rotations[l], dtype=float)
    except KeyError as error:
        raise ValueError(f"No documented real-SH 90° rotation for l={l}") from error


# ---------------------------------------------------------------------------
# General-angle real-SH rotation (needs sympy for Wigner-d)
# ---------------------------------------------------------------------------


def real_sh_rotation(l: int, theta: float, phi: float) -> np.ndarray:
    """Real-SH rotation matrix for arbitrary (theta, phi) in radians.

    .. math:: R = C^\dagger R_z(\phi) d^l(\theta) R_z(-\phi) C

    Falls back to :func:`real_sh_rotation_90` when ``theta == pi/2`` and
    ``phi == 0`` to avoid the sympy dependency for the common case.
    """
    if abs(theta - np.pi / 2.0) < 1e-12 and abs(phi) < 1e-12:
        return real_sh_rotation_90(l)

    from sympy.physics.wigner import wigner_d_small

    C = complex_to_real_sh(l)

    d_sym = wigner_d_small(l, theta)
    d_complex = np.array(d_sym, dtype=complex)
    rz = np.diag([np.exp(-1j * m * phi) for m in range(-l, l + 1)])

    R = C.conj().T @ rz @ d_complex.T @ rz.conj().T @ C
    return np.real(R)


# ---------------------------------------------------------------------------
# Assemble the full orbital rotation matrix over all shells
# ---------------------------------------------------------------------------


def orbital_rotation_matrix(
    n_orbitals: int,
    blocks: Sequence[ShellBlock] = (),
    theta: float | None = None,
    phi: float | None = None,
) -> np.ndarray:
    """Build the full (n_orbitals × n_orbitals) orbital rotation matrix.

    Parameters
    ----------
    n_orbitals
        Total number of orbitals in the basis.
    blocks
        Shell blocks from :func:`parse_orb_indx`.  If empty, returns the
        identity (spin-only rotation).
    theta, phi
        Target direction in radians.  If both *None*, uses the 90° z→x
        rotation (:func:`real_sh_rotation_90`).
    """
    rotation = np.eye(n_orbitals)
    assigned: set[int] = set()
    for block in blocks:
        if theta is not None:
            shell_rotation = real_sh_rotation(block.l, theta, phi or 0.0)
        else:
            shell_rotation = real_sh_rotation_90(block.l)
        if len(block.indices) != shell_rotation.shape[0]:
            raise ValueError(f"Wrong block length for l={block.l}")
        if any(idx < 0 or idx >= n_orbitals for idx in block.indices):
            raise ValueError("ORB_INDX orbital is outside the density-matrix basis")
        if assigned.intersection(block.indices):
            raise ValueError("ORB_INDX shell blocks overlap")
        assigned.update(block.indices)
        rotation[np.ix_(block.indices, block.indices)] = shell_rotation
    return rotation
