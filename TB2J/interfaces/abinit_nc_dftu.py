"""Reader for ABINIT NC DFT+U occupation and potential NetCDF files.

Reads ``*_DFTU.nc`` files produced by the ABINIT input variable
``prtdftu_nc=1`` with ``usedftu_nc=1``.  The file contains the Hubbard
occupation matrix ``occ_matrix`` and Hubbard potential matrix ``vmatrix``
in the compact ``(nlmnmax, nlmnmax, natom, nsppol)`` angular-momentum
basis for the correlated ``dftu_l`` channel.

The :func:`embed_dftu_potential` helper expands the compact potential
into the full NC-PAO projector basis, producing a ``delta_U`` operator
component suitable for :class:`~TB2J.projector_green.ProjectorGreenData`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    nc = None


@dataclasses.dataclass(frozen=True)
class NcDftuData:
    """Container for NC DFT+U occupation and potential matrices.

    Attributes
    ----------
    cell : ndarray (3, 3)
        Unit-cell vectors in Angstrom.
    positions : ndarray (3, natom)
        Cartesian positions in Angstrom.
    atomic_numbers : ndarray (natom,)
        Atomic numbers.
    typat : ndarray (natom,)
        Atom-type indices (1-based, ABINIT convention).
    dftu_l : ndarray (ntypat,)
        Angular momentum for the correlated channel per type (-1 = none).
    dftu_u : ndarray (ntypat,)
        Hubbard U parameter per type in eV.
    dftu_j : ndarray (ntypat,)
        Hund's J parameter per type in eV.
    dftu_dc : int
        Double-counting mode (0 = Dudarev, nonzero = Liechtenstein).
    occ_matrix : ndarray (nlmnmax, nlmnmax, natom, nsppol)
        Hubbard occupation matrix (dimensionless).
    vmatrix : ndarray (nlmnmax, nlmnmax, natom, nsppol)
        Hubbard potential matrix in eV.
    """

    cell: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray
    typat: np.ndarray
    dftu_l: np.ndarray
    dftu_u: np.ndarray
    dftu_j: np.ndarray
    dftu_dc: int
    occ_matrix: np.ndarray
    vmatrix: np.ndarray


def read_abinit_dftu_nc(path: str | Path) -> NcDftuData:
    """Read an ABINIT NC DFT+U NetCDF file.

    Parameters
    ----------
    path : str or Path
        Path to the ``*_DFTU.nc`` file.

    Returns
    -------
    NcDftuData
        Parsed occupation and potential data.
    """
    if nc is None:
        raise ImportError("netCDF4 is required to read DFTU NetCDF files.")

    path = str(path)
    with nc.Dataset(path, "r") as root:
        # Validate schema
        schema = getattr(root, "schema_name", "")
        if schema != "abinit.nc_dftu":
            raise ValueError(f"Not an ABINIT NC DFTU file: schema_name='{schema}'")

        struct_grp = root.groups["structure"]
        cell = struct_grp.variables["cell"][:].astype(float)
        positions = struct_grp.variables["positions"][:].astype(float)
        atomic_numbers = struct_grp.variables["atomic_numbers"][:].astype(int)
        typat = struct_grp.variables["typat"][:].astype(int)
        param_grp = root.groups["parameters"]
        dftu_l = param_grp.variables["dftu_l"][:].astype(int)
        dftu_u = param_grp.variables["dftu_u"][:].astype(float)
        dftu_j = param_grp.variables["dftu_j"][:].astype(float)
        dftu_dc = int(getattr(root, "dftu_dc", 0))

        occ_grp = root.groups["occupation"]
        occ_matrix = occ_grp.variables["occ_matrix"][:].astype(float)

        pot_grp = root.groups["potential"]
        vmatrix = pot_grp.variables["vmatrix"][:].astype(float)

    # ABINIT nctk reverses Fortran dimension order in the NetCDF file.
    # Variables appear as (nsppol, natom, nlmnmax, nlmnmax) in the file;
    # transpose to (nlmnmax, nlmnmax, natom, nsppol) for Python use.
    if vmatrix.ndim == 4 and vmatrix.shape[-1] >= vmatrix.shape[0]:
        vmatrix = vmatrix.transpose(3, 2, 1, 0)
        occ_matrix = occ_matrix.transpose(3, 2, 1, 0)
    vmatrix = np.ascontiguousarray(vmatrix)
    occ_matrix = np.ascontiguousarray(occ_matrix)
    cell = (
        np.ascontiguousarray(cell.T)
        if cell.shape[0] == 3
        else np.ascontiguousarray(cell)
    )
    positions = (
        np.ascontiguousarray(positions.T)
        if positions.shape[0] == 3
        else np.ascontiguousarray(positions)
    )

    return NcDftuData(
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        typat=typat,
        dftu_l=dftu_l,
        dftu_u=dftu_u,
        dftu_j=dftu_j,
        dftu_dc=dftu_dc,
        occ_matrix=occ_matrix,
        vmatrix=vmatrix,
    )


def embed_dftu_potential(
    dftu: NcDftuData,
    projector_l: Sequence[int],
    site_nproj: Sequence[int],
) -> np.ndarray:
    """Embed the DFT+U potential into the full NC-PAO projector basis.

    For each atom, the compact ``vmatrix`` block (shape
    ``(2*l+1, 2*l+1)``) is placed at the projector indices where
    ``projector_l == dftu_l[typat]``.  All other entries are zero.

    The result is a spin-difference operator suitable as a ``delta_U``
    component: ``V_up - V_down`` in the full projector basis.

    Parameters
    ----------
    dftu : NcDftuData
        Parsed DFTU data from :func:`read_abinit_dftu_nc`.
    projector_l : array-like (nproj,)
        Angular momentum ``l`` of each projector in the NC-PAO basis.
        This comes from the H/S NetCDF ``projector_l`` variable.
    site_nproj : array-like (nsite,)
        Number of projectors per site.

    Returns
    -------
    ndarray (nproj, nproj), complex
        ``V_up - V_down`` embedded in the full projector basis, in eV.
    """
    projector_l = np.asarray(projector_l, dtype=int)
    site_nproj = np.asarray(site_nproj, dtype=int)
    nproj = int(site_nproj.sum())
    nsite = len(site_nproj)
    nsppol = dftu.vmatrix.shape[-1]

    if nsppol != 2:
        raise ValueError(f"delta_U requires nsppol=2, got {nsppol}")

    # Build per-site projector index offsets
    site_offsets = np.concatenate([[0], np.cumsum(site_nproj)])

    delta_u = np.zeros((nproj, nproj), dtype=complex)

    for isite in range(nsite):
        if isite >= dftu.vmatrix.shape[2]:
            break

        itypat = dftu.typat[isite] - 1  # typat is 1-based from ABINIT
        l_target = dftu.dftu_l[itypat]
        if l_target < 0:
            continue

        norb = 2 * l_target + 1
        v_up = dftu.vmatrix[:norb, :norb, isite, 0]
        v_dn = dftu.vmatrix[:norb, :norb, isite, 1]
        v_diff = v_up - v_dn

        # Find projector indices for this site with l == l_target
        lo = int(site_offsets[isite])
        hi = int(site_offsets[isite + 1])
        site_l = projector_l[lo:hi]

        mask = site_l == l_target
        indices = np.where(mask)[0]
        if len(indices) != norb:
            raise ValueError(
                f"Site {isite}: expected {norb} projectors with l={l_target}, "
                f"found {len(indices)}"
            )

        global_idx = lo + indices
        delta_u[np.ix_(global_idx, global_idx)] = v_diff.astype(complex)

    return delta_u
