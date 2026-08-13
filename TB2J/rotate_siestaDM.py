"""Rotate a SIESTA density matrix in both spin and orbital space.

This module enhances the original spin-only rotation (``sisl``'s built-in
``DM.spin_rotate``) with a full **orbital** rotation that applies a real
spherical-harmonic Wigner-d rotation to each correlated shell.

The rotation operates on the SIESTA SOC density matrix (a sparse sisl
object with 8 spin components) by:

1. Converting it to a dense ``(n_orb, n_orb, 2, 2)`` spinor array.
2. Applying the orbital rotation *R* and spin rotation *U*:

   .. math::

       \\rho'_{ij}^{st} = R_{ia}\\, U_{su}\\, \\rho_{ab}^{uv}\\,
                          R^*_{jb}\\, U^*_{tv}

3. Writing the result back into the same sparse pattern.

For the spin-only case (``orbital=False`` or no ``ORB_INDX`` provided),
*R* is the identity and the result is equivalent to ``sisl``'s
``spin_rotate``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sisl

from TB2J.mathutils.orbital_rotation import (
    ShellBlock,
    orbital_rotation_matrix,
    parse_orb_indx,
)

TOLERANCE = 1.0e-10


# ---------------------------------------------------------------------------
# Spin rotation helpers (SU(2))
# ---------------------------------------------------------------------------


def spin_rotation_90_y() -> np.ndarray:
    """SU(2) for a 90° rotation about y (z → x)."""
    return np.array(((1.0, -1.0), (1.0, 1.0))) / np.sqrt(2.0)


def spin_rotation(theta: float, phi: float) -> np.ndarray:
    """SU(2) rotation that maps the z-axis spinor to direction (theta, phi).

    Parameters in radians.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array(
        ((c, -s * np.exp(-1j * phi)), (s * np.exp(1j * phi), c)),
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# sisl sparse DM  <->  dense spinor array
# ---------------------------------------------------------------------------


def density_matrix_to_spinor(dm) -> np.ndarray:
    """Convert a sisl SOC density matrix to a dense ``(n, n, 2, 2)`` array."""
    if not dm.spin.is_spinorbit:
        raise ValueError("The input density matrix is not spin-orbit/non-collinear")
    n_orbitals = int(dm.shape[0])
    spinor = np.zeros((n_orbitals, n_orbitals, 2, 2), dtype=complex)
    for row, column in dm.iter_nnz():
        i, j = int(row), int(column)
        spinor[i, j] = (
            (
                dm[i, j, dm.M11r] + 1j * dm[i, j, dm.M11i],
                dm[i, j, dm.M12r] + 1j * dm[i, j, dm.M12i],
            ),
            (
                dm[i, j, dm.M21r] + 1j * dm[i, j, dm.M21i],
                dm[i, j, dm.M22r] + 1j * dm[i, j, dm.M22i],
            ),
        )
    return spinor


def sparse_pattern(dm) -> np.ndarray:
    """Boolean mask of the sisl DM's non-zero (i, j) pairs."""
    n_orbitals = int(dm.shape[0])
    pattern = np.zeros((n_orbitals, n_orbitals), dtype=bool)
    for row, column in dm.iter_nnz():
        pattern[int(row), int(column)] = True
    if not np.array_equal(pattern, pattern.T):
        raise ValueError("The input density matrix has a non-Hermitian sparse pattern")
    return pattern


def spinor_to_density_matrix(dm, spinor: np.ndarray, pattern: np.ndarray):
    """Write a dense spinor back into a sisl DM with the original sparse pattern."""
    if np.any(np.abs(spinor[~pattern]) > TOLERANCE):
        raise ValueError(
            "Rotation would create entries outside the input sparse pattern"
        )
    output = dm.copy()
    for row, column in np.argwhere(pattern):
        i, j = int(row), int(column)
        block = spinor[i, j]
        output[i, j, output.M11r] = block[0, 0].real
        output[i, j, output.M11i] = block[0, 0].imag
        output[i, j, output.M22r] = block[1, 1].real
        output[i, j, output.M22i] = block[1, 1].imag
        output[i, j, output.M12r] = block[0, 1].real
        output[i, j, output.M12i] = block[0, 1].imag
        output[i, j, output.M21r] = block[1, 0].real
        output[i, j, output.M21i] = block[1, 0].imag
    if not dm.spsame(output):
        raise ValueError("Rotation changed the sparse density-matrix pattern")
    return output


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _check_hermitian(spinor: np.ndarray, tolerance: float = TOLERANCE) -> None:
    conj_T = spinor.transpose(1, 0, 3, 2).conj()
    if not np.allclose(spinor, conj_T, atol=tolerance, rtol=0.0):
        raise ValueError("Density matrix is not globally Hermitian")


def _verify_rotation(
    before: np.ndarray,
    after: np.ndarray,
    orbital_r: np.ndarray,
    spin_w: np.ndarray,
) -> None:
    _check_hermitian(before)
    _check_hermitian(after)
    tr_before = np.trace(before[:, :, 0, 0]) + np.trace(before[:, :, 1, 1])
    tr_after = np.trace(after[:, :, 0, 0]) + np.trace(after[:, :, 1, 1])
    if not np.isclose(tr_before, tr_after, atol=TOLERANCE, rtol=0.0):
        raise ValueError("Rotation changed the density-matrix trace")
    recovered = _apply_rotation(after, orbital_r.T, spin_w.conj().T)
    if not np.allclose(recovered, before, atol=TOLERANCE, rtol=0.0):
        raise ValueError("Inverse rotation did not recover the input density matrix")


# ---------------------------------------------------------------------------
# Core: apply orbital + spin rotation to a spinor DM
# ---------------------------------------------------------------------------


def _apply_rotation(
    spinor: np.ndarray, orbital_left: np.ndarray, spin_left: np.ndarray
) -> np.ndarray:
    """ρ'_{ij}^{st} = R_{ia} U_{su} ρ_{ab}^{uv} R*_{jb} U*_{tv}."""
    n = spinor.shape[0]
    if spinor.shape != (n, n, 2, 2):
        raise ValueError("Expected an (orbital, orbital, spin, spin) matrix")
    return np.einsum(
        "ia,su,abuv,jb,tv->ijst",
        orbital_left,
        spin_left,
        spinor,
        orbital_left.conj(),
        spin_left.conj(),
        optimize=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rotate_density_matrix(
    dm,
    blocks: Sequence[ShellBlock] = (),
    theta: float | None = None,
    phi: float | None = None,
):
    """Rotate a sisl SOC density matrix in both orbital and spin space.

    Parameters
    ----------
    dm
        sisl density matrix (must have ``spin.is_spinorbit``).
    blocks
        Shell blocks from :func:`~TB2J.mathutils.orbital_rotation.parse_orb_indx`.
        If empty, only the spin part is rotated.
    theta, phi
        Target direction in radians.  If both *None*, a 90° z→x rotation
        is applied (compatible with the original TB2J x/y/z convention).
    """
    pattern = sparse_pattern(dm)
    before = density_matrix_to_spinor(dm)

    if theta is not None:
        orbital_r = orbital_rotation_matrix(
            int(dm.shape[0]), blocks, theta=theta, phi=phi
        )
        spin_w = spin_rotation(theta, phi or 0.0)
    else:
        orbital_r = orbital_rotation_matrix(int(dm.shape[0]), blocks)
        spin_w = spin_rotation_90_y()

    after = _apply_rotation(before, orbital_r, spin_w)
    output = spinor_to_density_matrix(dm, after, pattern)

    written = density_matrix_to_spinor(output)
    _verify_rotation(before, written, orbital_r, spin_w)
    return output


def read_density_matrix(dm_path: str | Path, fdf_path: str | Path):
    """Read a sisl density matrix using the geometry from the FDF."""
    dm_path = Path(dm_path)
    fdf_path = Path(fdf_path)
    if not dm_path.is_file() or not fdf_path.is_file():
        raise FileNotFoundError("Both the DM and matching FDF input must exist")
    with sisl.get_sile(str(fdf_path)) as fdf_sile:
        geometry = fdf_sile.read_geometry()
    with sisl.get_sile(str(dm_path)) as dm_sile:
        return dm_sile.read_density_matrix(geometry=geometry)


def rotate_file(
    dm_path: str | Path,
    fdf_path: str | Path,
    output_path: str | Path,
    *,
    orbital: bool = True,
    orb_indx_path: Optional[str | Path] = None,
    theta: float | None = None,
    phi: float | None = None,
):
    """Read, rotate, and write a SIESTA .DM file.

    Parameters
    ----------
    dm_path, fdf_path
        Input .DM and matching .fdf files.
    output_path
        Output .DM path (must not already exist).
    orbital
        If *True*, rotate the orbital part (requires ``orb_indx_path``).
    orb_indx_path
        Path to the SIESTA ``ORB_INDX`` file.
    theta, phi
        Target direction in radians.  If both *None*, a 90° z→x rotation
        is applied.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".dm":
        raise ValueError("Output must have a .DM suffix")
    # Refuse to overwrite
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(output_path), flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        ) from exc
    os.close(fd)

    input_dm = read_density_matrix(dm_path, fdf_path)

    # For spin-only rotation (no orbital blocks), use sisl's built-in
    # spin_rotate which correctly handles the full sparse pattern.
    blocks = parse_orb_indx(orb_indx_path) if orbital and orb_indx_path else ()
    if orbital and orb_indx_path and not blocks:
        raise ValueError("ORB_INDX contains no complete shell/zeta blocks")

    if not blocks:
        # Spin-only: use sisl's built-in, which handles supercell sparsity
        if theta is not None:
            angles = [0.0, float(np.degrees(theta)), float(np.degrees(phi or 0.0))]
        else:
            angles = [0.0, 90.0, 0.0]  # default z→x
        output_dm = input_dm.spin_rotate(angles)
    else:
        # Full orbital + spin rotation
        output_dm = rotate_density_matrix(input_dm, blocks, theta=theta, phi=phi)

    output_dm.write(str(output_path))

    # Verify round-trip
    reloaded = read_density_matrix(output_path, fdf_path)
    if not input_dm.spsame(reloaded):
        raise ValueError("Written density matrix changed the sparse pattern")
    return reloaded


# ---------------------------------------------------------------------------
# Legacy compatibility: the original sisl-only spin rotation
# ---------------------------------------------------------------------------


def rotate_siesta_DM(DM, noncollinear=False):
    """Yield spin-rotated DMs using sisl's built-in (spin-only) rotation.

    Preserved for backward compatibility with the original TB2J interface.
    New code should use :func:`rotate_density_matrix` instead.
    """
    angles_list = [[0.0, 90.0, 0.0], [0.0, 90.0, 90.0]]
    if noncollinear:
        angles_list += [
            [0.0, 45.0, 0.0],
            [0.0, 90.0, 45.0],
            [0.0, 45.0, 90.0],
        ]
    for angles in angles_list:
        yield DM.spin_rotate(angles)


def read_label(fdf_fname: str) -> str:
    """Extract the system label from an FDF file."""
    label = "siesta"
    with open(fdf_fname, "r") as f:
        for line in f:
            corrected = line.lower().replace(".", "").replace("-", "")
            if "systemlabel" in corrected:
                label = line.split()[1]
                break
    return label


def rotate_DM(fdf_fname, noncollinear=False):
    """Original TB2J interface: generate spin-rotated DMs from an FDF.

    Writes ``{label}_0.DM`` (reference) through ``{label}_N.DM``.
    """
    fdf = sisl.get_sile(fdf_fname)
    DM = fdf.read_density_matrix()
    label = read_label(fdf_fname)

    rotated = rotate_siesta_DM(DM, noncollinear=noncollinear)
    for i, rotated_DM in enumerate(rotated):
        rotated_DM.write(f"{label}_{i+1}.DM")
    DM.write(f"{label}_0.DM")
    print(
        f"The output has been written to the {label}_i.DM files. "
        f"{label}_0.DM contains the reference density matrix."
    )
