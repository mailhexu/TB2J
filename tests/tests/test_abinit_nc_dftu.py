"""Tests for the ABINIT NC DFT+U reader and projector-basis embedding."""

import numpy as np
import pytest

netcdf4 = pytest.importorskip("netCDF4")

from TB2J.interfaces.abinit_nc_dftu import (
    NcDftuData,
    embed_dftu_potential,
    read_abinit_dftu_nc,
)


def _write_synthetic_dftu_nc(
    path, natom=1, ntypat=1, l_dftu=2, nsppol=2, U_eV=3.0, J_eV=0.0, dftu_dc=0
):
    """Write a minimal synthetic DFTU NetCDF file matching the ABINIT schema."""
    import netCDF4 as nc

    nlmnmax = 2 * l_dftu + 1

    root = nc.Dataset(str(path), "w", format="NETCDF4")
    root.schema_name = "abinit.nc_dftu"
    root.schema_version = "1.0"
    root.source_code = "test"
    root.spin_mode = "collinear"

    root.createDimension("three", 3)
    root.createDimension("natom", natom)
    root.createDimension("ntypat", ntypat)
    root.createDimension("nsppol", nsppol)
    root.createDimension("nlmnmax", nlmnmax)

    struct = root.createGroup("structure")
    cell_v = struct.createVariable("cell", "f8", ("three", "three"))
    cell_v[:] = np.eye(3) * 4.0
    pos_v = struct.createVariable("positions", "f8", ("three", "natom"))
    pos_v[:, 0] = [0, 0, 0]
    zn_v = struct.createVariable("atomic_numbers", "i4", ("natom",))
    zn_v[:] = [26] * natom
    typ_v = struct.createVariable("typat", "i4", ("natom",))
    typ_v[:] = [1] * natom

    param = root.createGroup("parameters")
    dl = param.createVariable("dftu_l", "i4", ("ntypat",))
    dl[:] = [l_dftu]
    du = param.createVariable("dftu_u", "f8", ("ntypat",))
    du[:] = [U_eV]
    dj = param.createVariable("dftu_j", "f8", ("ntypat",))
    dj[:] = [J_eV]
    dc = param.createVariable("dftu_dc", "i4")
    dc[...] = dftu_dc

    occ_grp = root.createGroup("occupation")
    occ_v = occ_grp.createVariable(
        "occ_matrix", "f8", ("nlmnmax", "nlmnmax", "natom", "nsppol")
    )
    occ_v[:] = np.zeros((nlmnmax, nlmnmax, natom, nsppol))
    for s in range(nsppol):
        for m in range(nlmnmax):
            occ_v[m, m, 0, s] = 0.4 if s == 0 else 0.2

    pot_grp = root.createGroup("potential")
    pot_v = pot_grp.createVariable(
        "vmatrix", "f8", ("nlmnmax", "nlmnmax", "natom", "nsppol")
    )
    U_eff = U_eV - J_eV
    pot_v[:] = np.zeros((nlmnmax, nlmnmax, natom, nsppol))
    for s in range(nsppol):
        for m in range(nlmnmax):
            pot_v[m, m, 0, s] = U_eff * (0.5 - (0.4 if s == 0 else 0.2))

    root.close()


def test_read_dftu_nc_roundtrip(tmp_path):
    """Read a synthetic DFTU file and verify all fields."""
    path = tmp_path / "test_DFTU.nc"
    _write_synthetic_dftu_nc(path, l_dftu=2, U_eV=5.0)

    data = read_abinit_dftu_nc(path)
    assert data.dftu_l.tolist() == [2]
    assert data.dftu_u.tolist() == [5.0]
    assert data.dftu_j.tolist() == [0.0]
    assert data.dftu_dc == 0
    assert data.vmatrix.shape == (5, 5, 1, 2)
    assert data.occ_matrix.shape == (5, 5, 1, 2)
    # Check diagonal values
    np.testing.assert_allclose(data.vmatrix[0, 0, 0, 0], 5.0 * (0.5 - 0.4), atol=1e-10)
    np.testing.assert_allclose(data.vmatrix[0, 0, 0, 1], 5.0 * (0.5 - 0.2), atol=1e-10)


def test_read_dftu_nc_wrong_schema(tmp_path):
    """Reject a file with the wrong schema_name."""
    import netCDF4 as nc

    path = tmp_path / "bad.nc"
    root = nc.Dataset(str(path), "w")
    root.schema_name = "something_else"
    root.createDimension("x", 1)
    root.close()

    with pytest.raises(ValueError, match="Not an ABINIT NC DFTU file"):
        read_abinit_dftu_nc(path)


def test_embed_dftu_potential_single_d_channel():
    """Embed a d-channel potential into a projector basis with s+d orbitals."""
    nlmnmax = 5  # 2*2+1
    dftu = NcDftuData(
        cell=np.eye(3) * 4.0,
        positions=np.zeros((3, 1)),
        atomic_numbers=np.array([26]),
        typat=np.array([1]),
        dftu_l=np.array([2]),
        dftu_u=np.array([5.0]),
        dftu_j=np.array([0.0]),
        dftu_dc=0,
        occ_matrix=np.zeros((nlmnmax, nlmnmax, 1, 2)),
        vmatrix=np.zeros((nlmnmax, nlmnmax, 1, 2)),
    )
    # Set up a Dudarev potential: V = U_eff * (0.5*I - n)
    for s in range(2):
        for m in range(5):
            dftu.vmatrix[m, m, 0, s] = 5.0 * (0.5 - (0.4 if s == 0 else 0.2))

    # Projector layout: [s(1), d(5)] = 6 total
    projector_l = np.array([0, 2, 2, 2, 2, 2])
    site_nproj = np.array([6])

    delta_u = embed_dftu_potential(dftu, projector_l, site_nproj)

    assert delta_u.shape == (6, 6)
    # The s-s block should be zero
    assert delta_u[0, 0] == 0.0
    # The d-d diagonal should be V_up - V_down
    expected_diff = 5.0 * (0.5 - 0.4) - 5.0 * (0.5 - 0.2)
    for m in range(5):
        np.testing.assert_allclose(delta_u[1 + m, 1 + m], expected_diff, atol=1e-10)


def test_embed_dftu_potential_multi_site():
    """Embed potentials for two atoms with different l channels."""
    nlmnmax = 5  # max(2*2+1)
    natom = 2
    # Two atoms, both d-channel
    vmatrix = np.zeros((nlmnmax, nlmnmax, natom, 2))
    for m in range(5):
        vmatrix[m, m, 0, 0] = 1.0  # atom 0, spin up
        vmatrix[m, m, 0, 1] = -1.0  # atom 0, spin down
        vmatrix[m, m, 1, 0] = 2.0  # atom 1, spin up
        vmatrix[m, m, 1, 1] = -2.0  # atom 1, spin down

    dftu = NcDftuData(
        cell=np.eye(3) * 4.0,
        positions=np.zeros((3, natom)),
        atomic_numbers=np.array([26, 26]),
        typat=np.array([1, 1]),
        dftu_l=np.array([2]),
        dftu_u=np.array([5.0]),
        dftu_j=np.array([0.0]),
        dftu_dc=0,
        occ_matrix=np.zeros((nlmnmax, nlmnmax, natom, 2)),
        vmatrix=vmatrix,
    )
    # Two sites: [s, d×5] each = 6 per site, 12 total
    projector_l = np.array([0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2])
    site_nproj = np.array([6, 6])

    delta_u = embed_dftu_potential(dftu, projector_l, site_nproj)
    assert delta_u.shape == (12, 12)
    # Atom 0 d-block: V_up - V_down = 1 - (-1) = 2
    np.testing.assert_allclose(delta_u[1, 1], 2.0, atol=1e-10)
    # Atom 1 d-block: V_up - V_down = 2 - (-2) = 4
    np.testing.assert_allclose(delta_u[7, 7], 4.0, atol=1e-10)
    # Cross-site blocks zero
    assert delta_u[1, 7] == 0.0


def test_embed_dftu_potential_wrong_projector_count():
    """Raise ValueError when projector count for l-channel is wrong."""
    dftu = NcDftuData(
        cell=np.eye(3) * 4.0,
        positions=np.zeros((3, 1)),
        atomic_numbers=np.array([26]),
        typat=np.array([1]),
        dftu_l=np.array([2]),
        dftu_u=np.array([5.0]),
        dftu_j=np.array([0.0]),
        dftu_dc=0,
        occ_matrix=np.zeros((5, 5, 1, 2)),
        vmatrix=np.zeros((5, 5, 1, 2)),
    )
    # Only 3 d-projectors instead of 5
    projector_l = np.array([0, 2, 2, 2])
    site_nproj = np.array([4])

    with pytest.raises(ValueError, match="expected 5 projectors"):
        embed_dftu_potential(dftu, projector_l, site_nproj)
