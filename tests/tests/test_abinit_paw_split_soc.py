"""ABINIT PAW split-SOC loader + leg builder tests (story 006, split-soc-ks spec).

Covers:

- schema-1.1 ``soc_pauli`` component acceptance (FR-021): whitelisted
  ``spin_treatment="pauli_2x2"``, Hartree->eV conversion, provenance
  validation (unit strength, lattice-frame quantization, all-atom
  coverage, soc-only term class);
- rejection of wrong units, wrong spinor order, wrong shape, broken
  Hermiticity, and soc_pauli in schema-1.0 files;
- the old schema-1.0 collinear path loading unchanged;
- SOC-off anchor: the spinor leg with ``W_SO = 0`` reproduces the
  existing collinear savetb2j exchange shell by shell;
- projector-space j-splitting structure of the exported operator
  (synthetic) and the I-atom p-shell oracle (real fixture, gated);
- all-atom ligand SOC entering the propagator while vertices stay
  magnetic-only (delta_total/delta_xc, no SOC, no ligand vertex);
- full-BZ phase bookkeeping through the leg builder (non-Gamma k
  points, inter-site R shells) against the independent collinear kernel;
- three-leg x/y/z rotate/merge driver with ``T_lattice = O T_leg O^T``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from TB2J.interfaces.abinit_savetb2j import load_abinit_savetb2j

HARTREE_TO_EV = 27.211386245988

REAL_FIXTURE_DIR = os.environ.get("TB2J_ABINIT_SPLIT_SOC_FIXTURES", "")


# ---------------------------------------------------------------------------
# synthetic schema-1.1 fixture
# ---------------------------------------------------------------------------


def _hermitian(rng, n):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / 2


def _antisymmetric(rng, n):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a - a.T) / 2


def build_soc_pauli_blocks(rng, nsite, nmax, xi_hartree):
    """Packed ABINIT L·S structure: uu=A (Hermitian), dd=-uu, du=-conj(ud),
    ud=B antisymmetric (=> dense block Hermitian)."""
    blocks = np.zeros((nsite, nmax, nmax, 2, 2), dtype=complex)
    for site in range(nsite):
        a = _hermitian(rng, nmax) * xi_hartree[site]
        b = _antisymmetric(rng, nmax) * xi_hartree[site]
        blocks[site, :, :, 0, 0] = a
        blocks[site, :, :, 1, 1] = -a
        blocks[site, :, :, 0, 1] = b
        blocks[site, :, :, 1, 0] = -b.conj()
    return blocks


def write_soc_pauli_savetb2j_fixture(
    path,
    schema_version="1.1",
    with_soc=True,
    soc_units="Hartree",
    soc_blocks_hartree=None,
    swap_spin_order=False,
    transpose_blocks=False,
    soc_wrong_shape=False,
    nkpt=3,
    nband=3,
    nsite=2,
    nproj_per_site=2,
    seed=17,
):
    """Write a small collinear ABINIT savetb2j fixture (optionally 1.1+soc)."""
    netcdf4 = pytest.importorskip("netCDF4")

    rng = np.random.default_rng(seed)
    nproj = nsite * nproj_per_site
    nmax = nproj_per_site

    with netcdf4.Dataset(path, "w") as nc:
        nc.createDimension("nspin", 2)
        nc.createDimension("nkpt", nkpt)
        nc.createDimension("nband", nband)
        nc.createDimension("nproj", nproj)
        nc.createDimension("nsite", nsite)
        nc.createDimension("nproj_site_max", nmax)
        nc.createDimension("natom", nsite)
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        if with_soc:
            nc.createDimension("nspinor", 2)
        nc.schema_name = "abinit.savetb2j.projector"
        nc.schema_version = schema_version
        nc.source_code = "abinit"
        nc.abinit_version = "synthetic"
        nc.spin_mode = "collinear"
        nc.spin_channel_order = "up,down"
        nc.full_bz = 1
        nc.kpoint_convention = "fractional_reciprocal"
        nc.phase_convention = "exp(-2*pi*i*k.R)"
        nc.coefficient_source = "abinit.cprj"
        nc.operator_basis = "abinit_native_paw_projector"
        units = {
            "length": "Angstrom",
            "cell": "Angstrom",
            "positions": "Angstrom",
            "energy": "eV",
            "operators": "eV",
        }
        if with_soc:
            units["soc_pauli"] = soc_units
        nc.units_json = json.dumps(units)

        structure = nc.createGroup("structure")
        structure.createVariable("cell", "f8", ("three", "three"))[:] = 2.0 * np.eye(3)
        positions = np.zeros((nsite, 3))
        if nsite > 1:
            positions[1] = [2.0, 0.0, 0.0]
        structure.createVariable("positions", "f8", ("natom", "three"))[:] = positions
        structure.createVariable("atomic_numbers", "i4", ("natom",))[:] = [26, 8][
            :nsite
        ]

        kpoints = nc.createGroup("kpoints")
        kpoints.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.5, 0.25],
        ][:nkpt]
        kpoints.createVariable("weights", "f8", ("nkpt",))[:] = np.full(
            nkpt, 1.0 / nkpt
        )

        bands = nc.createGroup("bands")
        bands.efermi = -0.7
        rng_eig = np.random.default_rng(seed + 1)
        eig = rng_eig.uniform(-6.0, -0.5, size=(2, nkpt, nband))
        bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = eig
        bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = 1.0

        projectors = nc.createGroup("projectors")
        projectors.coefficient_source = "abinit.cprj"
        projectors.coefficient_projector = "paw_nonlocal_projector"
        projectors.channel_interpretation = "abinit_paw_lmn_channel"
        projectors.operator_basis = "abinit_native_paw_projector"
        projectors.index_base = 0
        coeff = rng.normal(size=(2, nkpt, nband, nproj)) + 1j * rng.normal(
            size=(2, nkpt, nband, nproj)
        )
        coefficients = projectors.createVariable(
            "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
        )
        coefficients[..., 0] = coeff.real
        coefficients[..., 1] = coeff.imag
        projector_atom = np.repeat(np.arange(nsite), nproj_per_site)
        projector_site = np.repeat(np.arange(nsite), nproj_per_site)
        projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = (
            projector_atom
        )
        projectors.createVariable("projector_site", "i4", ("nproj",))[:] = (
            projector_site
        )
        projectors.createVariable("projector_l", "i4", ("nproj",))[:] = np.tile(
            [1, 1], nsite
        )
        projectors.createVariable("projector_m", "i4", ("nproj",))[:] = np.tile(
            [-1, 1], nsite
        )
        projectors.createVariable("projector_radial", "i4", ("nproj",))[:] = np.zeros(
            nproj, dtype=int
        )
        projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = np.full(
            nsite, nproj_per_site
        )
        site_indices = np.arange(nproj).reshape(nsite, nproj_per_site)
        projectors.createVariable(
            "site_projector_indices", "i4", ("nsite", "nproj_site_max")
        )[:] = site_indices
        overlap = np.zeros((nproj, nproj, 2))
        overlap[..., 0] = np.eye(nproj)
        projectors.createVariable(
            "overlap_metric", "f8", ("nproj", "nproj", "complex")
        )[:] = overlap

        operators = nc.createGroup("operators")
        hij = operators.createVariable(
            "hij",
            "f8",
            ("nspin", "nsite", "nproj_site_max", "nproj_site_max", "complex"),
        )
        hij[:] = 0.0
        delta_blocks = np.zeros((nsite, nmax, nmax))
        for site in range(nsite):
            orbital = np.eye(nmax) + 0.1 * np.eye(nmax)[::-1]
            delta_blocks[site] = orbital * (0.7 + 0.3 * site)
        hij[0, ..., 0] = delta_blocks / 2
        hij[1, ..., 0] = -delta_blocks / 2
        hij.definition = "abinit_total_paw_dij"
        hij.units = "eV"
        hij.source = "paw_ij%dij"
        hij.projection = "native ABINIT PAW projector basis"
        hij.operator_basis = "abinit_native_paw_projector"

        components = operators.createGroup("operator_components")
        delta = components.createVariable(
            "delta_total",
            "f8",
            ("nsite", "nproj_site_max", "nproj_site_max", "complex"),
        )
        delta[:] = 0.0
        delta[..., 0] = delta_blocks
        delta.source = "paw_ij%dij(up)-paw_ij%dij(down)"
        delta.units = "eV"
        delta.operator_basis = "abinit_native_paw_projector"
        delta.spin_treatment = "spin_difference"
        delta.completeness = "complete"

        if with_soc:
            if soc_blocks_hartree is None:
                soc_blocks_hartree = build_soc_pauli_blocks(
                    rng, nsite, nmax, [0.004, 0.001][:nsite]
                )
            values = soc_blocks_hartree
            if swap_spin_order:
                values = values.transpose(0, 1, 2, 4, 3)
            if transpose_blocks:
                values = values.transpose(0, 2, 1, 3, 4)
            dims = (
                (
                    "nsite",
                    "nproj_site_max",
                    "nproj_site_max",
                    "nspinor",
                    "nspinor",
                    "complex",
                )
                if not soc_wrong_shape
                else ("nsite", "nproj_site_max", "nproj_site_max", "complex")
            )
            var = components.createVariable("soc_pauli", "f8", dims)
            if soc_wrong_shape:
                var[...] = 0.0
            else:
                var[..., 0] = values.real
                var[..., 1] = values.imag
            var.source = "pawdijso(frozen_density)"
            var.units = soc_units
            var.operator_basis = "abinit_native_paw_projector"
            var.spin_treatment = "pauli_2x2"
            var.completeness = "complete"
            var.soc_strength = "1.0"
            var.spinaxis = "0 0 1"
            var.zora_term_class = "soc_only"
            var.quantization = "lattice_frame"
            var.covers = "all_atoms"
            var.reference = "strength_zero_frozen_density"
            var.pauli_component_order = (
                "(spin_row,spin_col) = (up-up, down-down, up-down, down-up) blocks"
            )
    return soc_blocks_hartree


# ---------------------------------------------------------------------------
# loader acceptance (TEST-001 roundtrip)
# ---------------------------------------------------------------------------


def test_soc_pauli_roundtrip_hartree_to_ev(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "fe_ligand_soc1.nc"
    written = write_soc_pauli_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)

    assert data.metadata["abinit_schema_version"] == "1.1"
    assert "soc_pauli" in data.operator_components
    soc = data.operator_components["soc_pauli"]
    assert soc.shape == (2, 2, 2, 2, 2)
    # loader normalization: the on-disk block is the composite transpose
    # of the operator, D = O^T over the (projector, spin) index; the
    # component is stored as O[i,j,s,t] = D[j,i,t,s] = <p_i s|W_SO|p_j t>
    np.testing.assert_allclose(
        soc,
        written.transpose(0, 2, 1, 4, 3) * HARTREE_TO_EV,
        rtol=0.0,
        atol=1e-12 * HARTREE_TO_EV,
    )
    meta = data.operator_component_metadata["soc_pauli"]
    assert meta["spin_treatment"] == "pauli_2x2"
    assert meta["units"] == "eV"
    assert meta["source_units"] == "Hartree"
    assert meta["orientation"] == "operator"
    assert "cprj <p|psi>" in meta["band_space_contraction"]
    assert meta["covers"] == "all_atoms"
    assert meta["quantization"] == "lattice_frame"
    # negative control: the normalization must swap BOTH index pairs
    # (full composite transpose, c^dag D.T c); an orbital-only transpose
    # would leave the spin slots unswapped and is rejected by the
    # iodine element-level oracle below
    orbital_only = written.transpose(0, 2, 1, 3, 4) * HARTREE_TO_EV
    assert not np.allclose(
        soc, orbital_only, atol=1e-12 * HARTREE_TO_EV
    ), "test fixture lost its complex antisymmetric ud block"
    # the collinear component is untouched
    delta_meta = data.operator_component_metadata["delta_total"]
    assert delta_meta["units"] == "eV"
    assert "source_units" not in delta_meta


def test_normalized_soc_component_survives_generic_netcdf_roundtrip(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.projector_green import ProjectorGreenData

    source = tmp_path / "raw_soc.nc"
    write_soc_pauli_savetb2j_fixture(source)
    original = load_abinit_savetb2j(source)
    destination = tmp_path / "normalized_soc.nc"
    original.save_netcdf(destination)
    reread = ProjectorGreenData.load_netcdf(destination)
    np.testing.assert_array_equal(
        reread.operator_components["soc_pauli"],
        original.operator_components["soc_pauli"],
    )
    assert reread.operator_component_metadata["soc_pauli"]["orientation"] == "operator"


def test_soc_off_file_schema_10_unchanged(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "soc0.nc"
    write_soc_pauli_savetb2j_fixture(filename, schema_version="1.0", with_soc=False)
    data = load_abinit_savetb2j(filename)
    assert data.metadata["abinit_schema_version"] == "1.0"
    assert "soc_pauli" not in data.operator_components
    assert data.operator_components["delta_total"].shape == (2, 2, 2)


def test_soc_pauli_committed_schema10_fixture_still_loads():
    pytest.importorskip("netCDF4")
    fixture = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "inputs"
        / "abinit_savetb2j"
        / "fe_savetb2j.nc"
    )
    if not fixture.exists():
        pytest.skip("tests/data submodule not initialized")
    data = load_abinit_savetb2j(fixture)
    assert data.metadata["abinit_schema_version"] == "1.0"
    assert "soc_pauli" not in data.operator_components


# ---------------------------------------------------------------------------
# loader rejections
# ---------------------------------------------------------------------------


def _expect_load_error(path, match):
    with pytest.raises(ValueError, match=match):
        load_abinit_savetb2j(path)


def test_reject_wrong_soc_units(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "wrong_units.nc"
    write_soc_pauli_savetb2j_fixture(filename, soc_units="eV")
    _expect_load_error(filename, "soc_pauli units must be Hartree")


def test_reject_units_json_mismatch(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "units_json_mismatch.nc"
    write_soc_pauli_savetb2j_fixture(filename)

    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(filename, "a") as nc:
        units = json.loads(nc.units_json)
        units["soc_pauli"] = "eV"
        nc.units_json = json.dumps(units)
    _expect_load_error(filename, "units_json")


def test_reject_packing_du_sign(tmp_path):
    """The generic Hermitian unpacker bug (du = +conj(ud)) must be rejected."""
    pytest.importorskip("netCDF4")
    filename = tmp_path / "du_sign.nc"
    blocks = build_soc_pauli_blocks(np.random.default_rng(3), 2, 2, [0.004, 0.001])
    blocks[..., 1, 0] = np.conj(blocks[..., 0, 1])
    write_soc_pauli_savetb2j_fixture(filename, soc_blocks_hartree=blocks)
    _expect_load_error(filename, "[Hh]ermitian")


def test_reject_packing_dd_sign(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "dd_sign.nc"
    blocks = build_soc_pauli_blocks(np.random.default_rng(3), 2, 2, [0.004, 0.001])
    blocks[..., 1, 1] = blocks[..., 0, 0]
    write_soc_pauli_savetb2j_fixture(filename, soc_blocks_hartree=blocks)
    _expect_load_error(filename, "packing convention")


def test_reject_broken_hermiticity(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "broken_herm.nc"
    blocks = build_soc_pauli_blocks(np.random.default_rng(3), 2, 2, [0.004, 0.001])
    blocks[0, 0, 1, 0, 0] += 1e-3  # break uu Hermiticity
    write_soc_pauli_savetb2j_fixture(filename, soc_blocks_hartree=blocks)
    _expect_load_error(filename, "[Hh]ermitian")


def test_reject_soc_pauli_in_schema_10(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "soc_in_10.nc"
    write_soc_pauli_savetb2j_fixture(filename, schema_version="1.0")
    _expect_load_error(filename, "schema_version 1.1")


def test_reject_wrong_soc_shape(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "wrong_shape.nc"
    write_soc_pauli_savetb2j_fixture(filename, soc_wrong_shape=True)
    _expect_load_error(filename, "shape")


def test_reject_non_unit_soc_strength(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "wrong_strength.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(filename, "a") as nc:
        nc.groups["operators"].groups["operator_components"].variables[
            "soc_pauli"
        ].soc_strength = "0.5"
    _expect_load_error(filename, "soc_strength")


# ---------------------------------------------------------------------------
# SOC-off anchor vs the existing collinear savetb2j exchange
# ---------------------------------------------------------------------------


def _collinear_reference_jdict(data, rpts, sites, nz=24, smearing=0.05):
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict

    return compute_projector_exchange_jdict(
        data,
        Rpts=rpts,
        nz=nz,
        smearing_eV=smearing,
        sites=sites,
        operator_component="delta_total",
    )


def _r_grid():
    from TB2J.interfaces.gpaw_projector import _R_grid

    return _R_grid(nmax=1)


def test_soc_off_leg_matches_collinear_kernel_shell_by_shell(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import build_paw_split_soc_leg
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    filename = tmp_path / "anchor.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    data = load_abinit_savetb2j(filename)

    sites = [0, 1]
    rpts = _r_grid()
    reference = _collinear_reference_jdict(data, rpts, sites)

    leg_data = build_paw_split_soc_leg(data, leg="z", sites_magnetic=sites)
    assert leg_data.nspinor == 2
    assert leg_data.coefficients.shape == (1, data.nkpt, 2 * data.nband, 2, data.nproj)
    # A prefix ending at 2*m retains m states of EACH collinear channel.
    for spin in (0, 1):
        np.testing.assert_array_equal(
            leg_data.eigenvalues[0, :, spin::2], data.eigenvalues[spin]
        )
        np.testing.assert_array_equal(
            leg_data.occupations[0, :, spin::2], data.occupations[spin]
        )
        np.testing.assert_array_equal(
            leg_data.coefficients[0, :, spin::2, spin, :], data.coefficients[spin]
        )
    result = compute_ks_split_soc_exchange(
        leg_data,
        np.zeros((data.nkpt, 2 * data.nband, 2 * data.nband)),
        lam=1.0,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
        sites=sites,
    )
    assert set(result["exchange"]) == {
        (tuple(r), i, j) for r in map(tuple, rpts) for i in sites for j in sites
    }
    for (r, i, j), entry in result["exchange"].items():
        assert entry["Jiso"] == pytest.approx(
            reference[(r, i, j)], rel=1e-8, abs=1e-10
        ), (r, i, j)
        assert np.linalg.norm(entry["dmi"]) < 1e-9


# ---------------------------------------------------------------------------
# synthetic j-splitting structure of the assembled operator
# ---------------------------------------------------------------------------


def test_band_w_soc_matches_independent_assembly(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import paw_split_soc_band_w_soc

    filename = tmp_path / "wassembly.nc"
    written = write_soc_pauli_savetb2j_fixture(filename)
    data = load_abinit_savetb2j(filename)

    w = paw_split_soc_band_w_soc(data, leg="z")
    nkpt, nband = data.nkpt, data.nband
    assert w.shape == (nkpt, 2 * nband, 2 * nband)
    scale = float(np.abs(w).max())
    assert scale > 1e-6
    assert np.allclose(w, np.conj(np.swapaxes(w, 1, 2)), atol=1e-12 * scale)

    # independent loop assembly in the interleaved spinor basis,
    # consuming the operator-orientation blocks (the on-disk orientation
    # is the full composite transpose, normalized by the loader)
    soc_ev = written.transpose(0, 2, 1, 4, 3) * HARTREE_TO_EV
    coeff = data.coefficients  # (nspin, nkpt, nband, nproj)
    expected = np.zeros_like(w)
    for ik in range(nkpt):
        c = np.zeros((2 * nband, 2, data.nproj), dtype=complex)
        for s in range(2):
            c[s::2, s, :] = coeff[s, ik]
        ref = np.zeros((2 * nband, 2 * nband), dtype=complex)
        for site in range(len(data.site_nproj)):
            proj = data.site_projector_indices[site]
            nproj_site = data.site_nproj[site]
            block = soc_ev[site]
            for n in range(2 * nband):
                for m in range(2 * nband):
                    acc = 0j
                    for a in range(nproj_site):
                        for b in range(nproj_site):
                            acc += (
                                np.conj(c[n, :, proj[a]])
                                @ block[a, b]
                                @ c[m, :, proj[b]]
                            )
                    ref[n, m] += acc
        expected[ik] = ref
    np.testing.assert_allclose(w, expected, rtol=1e-10, atol=1e-12 * scale)


def test_assembled_soc_eigenvalue_pairs_synthetic(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import paw_split_soc_band_w_soc

    filename = tmp_path / "pairs.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    data = load_abinit_savetb2j(filename)
    w = paw_split_soc_band_w_soc(data, leg="z")
    assert np.all(np.abs(w) > 0.0)
    # L·S structure: each site's projector-space block has a spectrum
    # symmetric about zero (uu + dd = 0 identically).  The band-window
    # operator inherits Hermiticity but not tracelessness (truncated
    # window), so only the projector-space symmetry is pinned here.
    soc = data.operator_components["soc_pauli"]
    for site in range(len(data.site_nproj)):
        n = data.site_nproj[site]
        block = soc[site, :n, :n]
        dense = np.empty((2 * n, 2 * n), dtype=complex)
        dense[0::2, 0::2] = block[:, :, 0, 0]
        dense[1::2, 1::2] = block[:, :, 1, 1]
        dense[0::2, 1::2] = block[:, :, 0, 1]
        dense[1::2, 0::2] = block[:, :, 1, 0]
        eig = np.linalg.eigvalsh(dense).real
        np.testing.assert_allclose(np.sum(eig), 0.0, atol=1e-9)
        pos = np.sort(eig[eig > 0])
        neg = np.sort(eig[eig < 0])
        assert pos.size == neg.size
        np.testing.assert_allclose(neg, -pos[::-1], atol=1e-9)


# ---------------------------------------------------------------------------
# magnetic vertices: source, sites, no SOC contamination
# ---------------------------------------------------------------------------


def test_vertices_magnetic_only_and_soc_free(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import build_paw_split_soc_leg

    filename = tmp_path / "vertices.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    data = load_abinit_savetb2j(filename)

    leg = build_paw_split_soc_leg(data, leg="z", sites_magnetic=[0])
    ops = leg.spinor_operator
    # ligand vertex exactly zero even though delta_total[1] is nonzero
    np.testing.assert_allclose(ops[1], 0.0, atol=0.0)
    # magnetic vertex is delta_total (x sigma_z), never the SOC operator
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    delta0 = data.get_operator_component("delta_total", site=0)
    np.testing.assert_allclose(
        ops[0, : delta0.shape[0], : delta0.shape[1]],
        np.einsum("pq,st->pqst", delta0, sz),
        atol=1e-14,
    )
    # the vertex is the collinear splitting only: the (nonzero) SOC blocks
    # of the same site must not appear anywhere in the spinor operator
    soc0 = data.operator_components["soc_pauli"][0]
    assert np.abs(soc0).max() > 0.0
    assert (
        np.abs(ops[0]).max()
        < np.abs(np.einsum("pq,st->pqst", delta0, sz)).max() + 1e-12
    )
    np.testing.assert_allclose(ops[0][:, :, 0, 1], delta0 * sz[0, 1], atol=1e-14)


def test_ligand_soc_enters_propagator_not_vertex(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import (
        build_paw_split_soc_leg,
        paw_split_soc_band_w_soc,
    )
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    filename = tmp_path / "ligand.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    data = load_abinit_savetb2j(filename)

    w_with = paw_split_soc_band_w_soc(data, leg="z")
    # zero the ligand SOC block and reassemble
    data_soc = load_abinit_savetb2j(filename)
    data_soc.operator_components["soc_pauli"][1] = 0.0
    w_without = paw_split_soc_band_w_soc(data_soc, leg="z")
    assert not np.allclose(w_with, w_without, atol=1e-12)

    rpts = _r_grid()
    leg_with = build_paw_split_soc_leg(data, leg="z", sites_magnetic=[0])
    res = compute_ks_split_soc_exchange(
        leg_with,
        w_with,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
        sites=[0],
    )
    assert set(res["exchange"]) == {(tuple(r), 0, 0) for r in map(tuple, rpts)}
    assert res["metadata"]["vertices"]["sites"] == [0]
    assert res["metadata"]["vertices"]["magnetic_only"] is True
    assert res["metadata"]["vertices"]["soc_free"] is True
    assert res["metadata"]["soc_operator"]["coverage"] == "all_atoms"

    leg_without = build_paw_split_soc_leg(data_soc, leg="z", sites_magnetic=[0])
    res_off = compute_ks_split_soc_exchange(
        leg_without,
        w_without,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
        sites=[0],
    )
    pairs = [k for k in res["exchange"] if k[0] != (0, 0, 0)]
    assert any(
        not np.isclose(
            res["exchange"][k]["Jiso"], res_off["exchange"][k]["Jiso"], atol=1e-9
        )
        for k in pairs
    ), "ligand SOC must affect the propagator"


# ---------------------------------------------------------------------------
# leg rotations and three-leg driver
# ---------------------------------------------------------------------------


def test_leg_rotation_matrices():
    from TB2J.interfaces.abinit_paw_split_soc import (
        rotation_matrix_from_su2,
        su2_leg_rotation,
    )

    e_z = np.array([0.0, 0.0, 1.0])
    for leg, axis in {"x": e_z[0], "y": e_z[1], "z": e_z[2]}.items():
        u = su2_leg_rotation(leg)
        o = rotation_matrix_from_su2(u)
        np.testing.assert_allclose(o @ o.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(o) - 1.0) < 1e-12
        target = np.zeros(3)
        target["xyz".index(leg)] = 1.0
        np.testing.assert_allclose(o @ e_z, target, atol=1e-12)


def test_driver_three_leg_rotate_merge(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import (
        gen_exchange_abinit_paw_split_soc,
    )

    filename = tmp_path / "driver.nc"
    write_soc_pauli_savetb2j_fixture(filename)
    rpts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=int,
    )
    out = gen_exchange_abinit_paw_split_soc(
        filename,
        output_path=tmp_path / "TB2J_split_soc",
        index_magnetic_atoms=[0],
        Rpts=rpts,
        nz=16,
        smearing_eV=0.05,
    )
    merged_dir = Path(out["output_path"])
    assert (merged_dir / "TB2J.pickle").exists()
    assert (merged_dir / "exchange.out").exists()
    assert (merged_dir / "split_soc_provenance.json").exists()
    for leg in ("x", "y", "z"):
        leg_dir = tmp_path / "TB2J_split_soc" / f"leg_{leg}"
        assert (leg_dir / "TB2J.pickle").exists()
        meta = json.loads((leg_dir / "split_soc_provenance.json").read_text())
        assert meta["merge_mode"] == "three_leg_rotate_merge"
        assert meta["frame"]["leg"] == leg
        study = meta["band_window"]["convergence_study"]
        assert [row["nband"] for row in study["windows"]] == [4, 6]
        assert (
            meta["strength0_reference"]["sha256"]
            == hashlib.sha256(filename.read_bytes()).hexdigest()
        )
        from TB2J.io_merge import read_pickle

        sio = read_pickle(str(leg_dir))
        assert study["windows"][-1]["Jiso"]["[1,0,0]"] == pytest.approx(
            sio.exchange_Jdict[((1, 0, 0), 0, 0)], abs=1e-12
        )
        assert sio.split_soc_provenance == meta
        for text in (
            (leg_dir / "exchange.out").read_text(),
            "".join(
                ElementTree.parse(leg_dir / "Multibinit/exchange.xml")
                .getroot()
                .itertext()
            ),
        ):
            assert (
                json.loads(
                    next(
                        line.split(": ", 1)[1]
                        for line in text.splitlines()
                        if line.startswith("split_soc_provenance: ")
                    )
                )
                == meta
            )

    # The merge solves a rank-six anisotropy system and then moves its
    # isotropic trace into Jiso. The final traceless Jani cannot be used
    # to reconstruct that pre-standardization trace from leg scalars.
    from TB2J.io_merge import Merger, read_pickle

    merger = Merger(*(str(out["leg_paths"][leg]) for leg in ("x", "y", "z")))
    assert merger.coeff_matrix
    assert all(
        np.linalg.matrix_rank(coeff, tol=1e-2) == 6
        for coeff in merger.coeff_matrix.values()
    )
    merged = read_pickle(str(merged_dir))
    assert merged.exchange_Jdict
    assert merged.split_soc_provenance == out["metadata"]
    assert merged.split_soc_provenance["legs"] == out["leg_metadata"]
    for key, value in merged.exchange_Jdict.items():
        assert np.isfinite(value)
        jani = merged.Jani_dict[key]
        np.testing.assert_allclose(jani, jani.T, atol=1e-12)
        assert abs(np.trace(jani)) < 1e-11
    for leg in ("x", "y", "z"):
        sio = read_pickle(str(out["leg_paths"][leg]))
        np.testing.assert_allclose(
            sio.spinat[0] / np.linalg.norm(sio.spinat[0]),
            -np.eye(3)["xyz".index(leg)],
            atol=1e-12,
        )


def test_driver_refuses_to_assume_ligand_is_magnetic(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import gen_exchange_abinit_paw_split_soc

    source = tmp_path / "fe_ligand.nc"
    write_soc_pauli_savetb2j_fixture(source)  # second site has nonzero delta_total
    with pytest.raises(
        ValueError, match="magnetic_sites|index_magnetic_atoms|magnetic_elements"
    ):
        gen_exchange_abinit_paw_split_soc(source, output_path=tmp_path / "invalid")
    with pytest.raises(ValueError, match="absolute second_variation"):
        gen_exchange_abinit_paw_split_soc(
            source,
            output_path=tmp_path / "derivative_as_J",
            index_magnetic_atoms=[0],
            mode="first_order_insertion",
        )


def test_afm_leg_writes_opposite_site_axes(tmp_path):
    nc4 = pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import gen_exchange_abinit_paw_split_soc
    from TB2J.io_merge import read_pickle

    source = tmp_path / "afm.nc"
    write_soc_pauli_savetb2j_fixture(source)
    with nc4.Dataset(source, "a") as nc:
        ops = nc.groups["operators"]
        for variable in (
            ops.variables["hij"],
            ops.groups["operator_components"].variables["delta_total"],
        ):
            values = variable[:]
            site_axis = 1 if variable.name.endswith("hij") else 0
            sl = [slice(None)] * values.ndim
            sl[site_axis] = 1
            values[tuple(sl)] *= -1
            variable[:] = values
    rpts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    )
    out = gen_exchange_abinit_paw_split_soc(
        source,
        output_path=tmp_path / "afm_exchange",
        index_magnetic_atoms=[0, 1],
        Rpts=rpts,
        nz=12,
    )
    for direction, path in out["leg_paths"].items():
        sio = read_pickle(str(path))
        np.testing.assert_allclose(
            sio.spinat[0], -np.eye(3)["xyz".index(direction)], atol=1e-12
        )
        np.testing.assert_allclose(sio.spinat[1], -sio.spinat[0])


def test_driver_output_rotation_formula(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_paw_split_soc import (
        _rotate_entry_tensors,
        rotation_matrix_from_su2,
        su2_leg_rotation,
    )

    rng = np.random.default_rng(5)
    entry = {
        "Jiso": 0.3,
        "dmi": rng.normal(size=3),
        "jani": rng.normal(size=(3, 3)),
    }
    entry["jani"] = (entry["jani"] + entry["jani"].T) / 2
    entry["tensor"] = (
        entry["Jiso"] * np.eye(3)
        + 0.5 * (entry["dmi"][:, None] - entry["dmi"][None, :])
        + 0.5 * (entry["jani"] + entry["jani"].T)
    )
    for leg in ("x", "y", "z"):
        o = rotation_matrix_from_su2(su2_leg_rotation(leg))
        out = _rotate_entry_tensors(dict(entry), o)
        np.testing.assert_allclose(out["Jiso"], entry["Jiso"])
        np.testing.assert_allclose(out["dmi"], o @ entry["dmi"], atol=1e-12)
        np.testing.assert_allclose(out["jani"], o @ entry["jani"] @ o.T, atol=1e-12)
        expected_tensor = (
            entry["Jiso"] * np.eye(3)
            + 0.5 * (out["dmi"][:, None] - out["dmi"][None, :])
            + 0.5 * (out["jani"] + out["jani"].T)
        )
        np.testing.assert_allclose(out["tensor"], expected_tensor, atol=1e-12)


# ---------------------------------------------------------------------------
# real-fixture phase-sensitive matrix-element oracle (I 5p, gated)
# ---------------------------------------------------------------------------


def _analytic_ls_angular_l1():
    """Analytic <y_a|L.S|y_b> angular-spin matrices for l=1.

    Real tesseral harmonics with Condon-Shortley phase in ABINIT order
    m = (-1, 0, +1) (pypao real_sph_harm convention: m>0 -> cos, m<0 ->
    sin, directions (y, z, x)).  Built from the complex-CS ladder
    (Lz|lm> = m|lm>, L+- ladder) transformed to the real basis; L.S =
    Lz Sz + (L+ S- + L- S+)/2 with S = sigma/2.
    """
    # Gauss-Legendre grid on the sphere
    nodes, _ = np.polynomial.legendre.leggauss(48)
    phi = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    cos_t, phi_g = np.meshgrid(nodes, phi, indexing="ij")
    cos_t = cos_t.ravel()
    phi_g = phi_g.ravel()
    sin_t = np.sqrt(1.0 - cos_t**2)

    # complex Condon-Shortley Y_1m
    c = np.sqrt(3.0 / (8.0 * np.pi))
    y_cplx = {
        -1: c * sin_t * np.exp(-1j * phi_g),
        0: np.sqrt(3.0 / (4.0 * np.pi)) * cos_t,
        1: -c * sin_t * np.exp(1j * phi_g),
    }
    # real tesseral, exactly the pypao real_sph_harm convention:
    # y = sqrt2 * sqrt((2l+1)/4pi) * sqrt((l-|m|)!/(l+|m|)!) * (-1)^|m|
    #     * (cos(|m| phi) if m>0 else sin(|m| phi)) * P_|m|(cos theta)
    # scipy lpmv includes the Condon-Shortley (-1)^m. pypao applies
    # a second (-1)^|m|, hence p_y and p_x are POSITIVE coordinates.
    # SymPy: -assoc_legendre(1,1,x) == +sqrt(1-x**2).
    # m = (-1, 0, +1) corresponds to (y, z, x).
    y_real = {
        -1: np.sqrt(3.0 / (4.0 * np.pi)) * sin_t * np.sin(phi_g),
        0: np.sqrt(3.0 / (4.0 * np.pi)) * cos_t,
        1: np.sqrt(3.0 / (4.0 * np.pi)) * sin_t * np.cos(phi_g),
    }
    # U[a, m]: y_a = sum_m U[a, m] Y_1m^c  (least squares on the grid)
    grid = np.stack([y_cplx[m] for m in (-1, 0, 1)], axis=1)
    u = np.empty((3, 3), dtype=complex)
    for a, m in enumerate((-1, 0, 1)):
        u[a], *_ = np.linalg.lstsq(grid, y_real[m], rcond=None)

    # L operators in the complex basis (l=1)
    lz_c = np.diag([-1.0, 0.0, 1.0]).astype(complex)
    lp_c = np.zeros((3, 3), dtype=complex)  # |m+1><m|
    for m in (-1, 0):
        lp_c[m + 2, m + 1] = np.sqrt(2.0 - m * (m + 1))
    lm_c = lp_c.conj().T
    lx_c = (lp_c + lm_c) / 2.0
    ly_c = (lp_c - lm_c) / (2.0j)

    def to_real(m_cplx):
        # |R_a> = sum_m U[a,m]|Y_m>; the BRA carries conj(U).
        # SymPy verifies this gives Lx_yz=-i, Ly_zx=-i, Lz_yx=+i.
        return u.conj() @ m_cplx @ u.T

    lz = to_real(lz_c)
    lminus = to_real(lx_c - 1j * ly_c)
    lplus = to_real(lx_c + 1j * ly_c)

    ls = np.zeros((3, 3, 2, 2), dtype=complex)
    ls[:, :, 0, 0] = lz / 2.0
    ls[:, :, 1, 1] = -lz / 2.0
    ls[:, :, 0, 1] = lminus / 2.0
    ls[:, :, 1, 0] = lplus / 2.0
    return ls


def _fit_xi(block, ls):
    """Least-squares real xi for block ?= xi * ls; returns (xi, rel residual)."""
    ref = ls.ravel()
    data = np.asarray(block, dtype=complex).ravel()
    xi = float(np.real(np.vdot(ref, data)) / np.vdot(ref, ref))
    residual = float(np.linalg.norm(data - xi * ref) / (abs(xi) * np.linalg.norm(ref)))
    return xi, residual


@pytest.mark.skipif(
    not REAL_FIXTURE_DIR or not Path(REAL_FIXTURE_DIR).is_dir(),
    reason="set TB2J_ABINIT_SPLIT_SOC_FIXTURES to the abinit split-soc-ks fixtures",
)
def test_iodine_valence_p_complex_matrix_element_oracle():
    """Nontrivial complex oracle on the I 5p block (S5 orientation audit).

    The loader-normalized valence-5p block must equal +xi * (L.S) with
    xi = +0.022902 Ha (S5: machine-exact fit, zero residual) in the
    operator orientation; the RAW file orientation (packed-pair
    transpose) must fail the same fit.  This pins the lmn transpose AND
    the complex conjugation chain phase-sensitively (spectra are
    transpose-blind).
    """
    pytest.importorskip("netCDF4")
    data = load_abinit_savetb2j(
        Path(REAL_FIXTURE_DIR) / "i_atom/run/i_refo_SAVETB2J.nc"
    )
    soc = data.operator_components["soc_pauli"]  # operator orientation, eV
    assert data.projector_l[2:5].tolist() == [1, 1, 1]
    block = soc[0, 2:5, 2:5]  # valence 5p, m = -1, 0, +1

    ls = _analytic_ls_angular_l1()
    xi, residual = _fit_xi(block, ls)
    assert residual < 1e-10, f"operator-orientation fit residual {residual}"
    xi_expected = 0.022902 * HARTREE_TO_EV
    np.testing.assert_allclose(xi, xi_expected, rtol=1e-4)

    # negative control: the RAW on-disk orientation (the naive c^dag D c
    # contraction, equivalently the elementwise conjugate of the
    # normalized block, since the stored block is Hermitian) must fail
    raw = block.transpose(1, 0, 3, 2)  # undo the loader normalization
    _, residual_raw = _fit_xi(raw, ls)
    assert residual_raw > 0.1, residual_raw

    # negative control: an orbital-only normalization (spin slots left
    # unswapped -- the rejected (0,2,1,3,4) hypothesis) differs from the
    # correct full composite transpose exactly in the spin-flip slots
    # and must fail the same fit
    _, residual_spin_swap = _fit_xi(block.transpose(0, 1, 3, 2), ls)
    assert residual_spin_swap > 0.1, residual_spin_swap

    # the complex spin-flip (ud) elements alone must carry the fit:
    # sub-block fit on the ud slot pins xi independently of the spin
    # diagonal (Lz) terms
    xi_ud, residual_ud = _fit_xi(block[:, :, 0, 1], ls[:, :, 0, 1])
    assert residual_ud < 1e-10, residual_ud
    np.testing.assert_allclose(xi_ud, xi_expected, rtol=1e-4)

    # j-manifold of the fitted operator: {-xi (2x), +xi/2 (4x)}
    dense = np.empty((6, 6), dtype=complex)
    dense[0::2, 0::2] = block[:, :, 0, 0]
    dense[1::2, 1::2] = block[:, :, 1, 1]
    dense[0::2, 1::2] = block[:, :, 0, 1]
    dense[1::2, 0::2] = block[:, :, 1, 0]
    eig = np.linalg.eigvalsh(dense).real
    np.testing.assert_allclose(np.sort(eig), [-xi] * 2 + [xi / 2.0] * 4, rtol=1e-8)


@pytest.mark.skipif(
    not REAL_FIXTURE_DIR or not Path(REAL_FIXTURE_DIR).is_dir(),
    reason="set TB2J_ABINIT_SPLIT_SOC_FIXTURES to the abinit split-soc-ks fixtures",
)
class TestRealFixtures:
    def test_iron_soc_roundtrip(self):
        """Both radial p/d shells carry the 2+4 and 4+6 j manifolds."""
        pytest.importorskip("netCDF4")
        data = load_abinit_savetb2j(
            Path(REAL_FIXTURE_DIR) / "fe/soc1/fe_soc1o_SAVETB2J.nc"
        )
        soc = data.operator_components["soc_pauli"]
        assert soc.shape == (1, 18, 18, 2, 2)
        assert data.metadata["abinit_schema_version"] == "1.1"
        for l in (1, 2):
            for radial in (1, 2):
                indices = np.flatnonzero(
                    (data.projector_l == l) & (data.projector_radial == radial)
                )
                assert indices.size == 2 * l + 1
                block = soc[0][np.ix_(indices, indices)]
                dense = block.transpose(0, 2, 1, 3).reshape(
                    2 * indices.size, 2 * indices.size
                )
                eig = np.linalg.eigvalsh(dense)
                xi = 2.0 * eig[-1] / l
                assert xi > 0.0
                np.testing.assert_allclose(
                    eig,
                    [-xi * (l + 1) / 2] * (2 * l) + [xi * l / 2] * (2 * l + 2),
                    atol=1e-12,
                    rtol=1e-11,
                )

    def test_iron_smoke_three_leg(self, tmp_path):
        pytest.importorskip("netCDF4")
        from TB2J.interfaces.abinit_paw_split_soc import (
            gen_exchange_abinit_paw_split_soc,
        )

        out = gen_exchange_abinit_paw_split_soc(
            Path(REAL_FIXTURE_DIR) / "fe/soc1/fe_soc1o_SAVETB2J.nc",
            output_path=tmp_path / "fe_split_soc",
            index_magnetic_atoms=[0],
            Rcut=8.0,
            nz=20,
            smearing_eV=0.05,
        )
        merged_dir = Path(out["output_path"])
        assert (merged_dir / "TB2J.pickle").exists()
        assert merged_dir.joinpath("exchange.out").exists()

    def test_iron_soc_off_matches_schema10_shells(self, tmp_path):
        pytest.importorskip("netCDF4")
        from TB2J.interfaces.abinit_paw_split_soc import (
            gen_exchange_abinit_paw_split_soc,
        )
        from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
        from TB2J.io_merge import read_pickle

        root = Path(REAL_FIXTURE_DIR) / "fe-k8"
        reference_data = load_abinit_savetb2j(root / "soc0/fe_k8_soc0o_SAVETB2J.nc")
        rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        expected = compute_projector_exchange_jdict(
            reference_data,
            Rpts=rpts,
            nz=12,
            smearing_eV=0.05,
            sites=[0],
            operator_component="delta_total",
        )
        assert (
            abs(expected[((1, 0, 0), 0, 0)]) > 0.01
        )  # real 8-k Fe shell, not Gamma-only zero
        out = gen_exchange_abinit_paw_split_soc(
            root / "soc1/fe_k8_soc1o_SAVETB2J.nc",
            output_path=tmp_path / "soc_off_fe",
            index_magnetic_atoms=[0],
            Rpts=rpts,
            Rcut=4.0,
            nz=12,
            smearing_eV=0.05,
            lam=0.0,
        )
        for direction, path in out["leg_paths"].items():
            actual = read_pickle(str(path)).exchange_Jdict
            np.testing.assert_allclose(
                read_pickle(str(path)).spinat[0],
                np.eye(3)["xyz".index(direction)],
                atol=1e-12,
            )
            assert set(actual) == {((1, 0, 0), 0, 0), ((-1, 0, 0), 0, 0)}
            assert max(abs(actual[key] - expected[key]) for key in actual) < 1e-8

    def test_iron_native_small_soc_response_from_consumer_band_matrix(self):
        """Fe-class j-splitting response: actual loaded cprj/soc_pauli KS path.

        Both native spinor legs solve wavefunctions at the SAME fixed nspden4
        density (iscf=-2, nstep=50); subtracting lambda=0 cancels the
        nspden2->4 baseline change. This is a small-lambda check, not a
        full-strength window-convergence certificate.
        """
        nc4 = pytest.importorskip("netCDF4")
        from TB2J.interfaces.abinit_paw_split_soc import (
            build_paw_split_soc_leg,
            paw_split_soc_band_w_soc,
        )
        from TB2J.split_soc_kernel import second_variation_spectrum

        root = Path(REAL_FIXTURE_DIR) / "fe-k8"
        data = load_abinit_savetb2j(root / "soc1/fe_k8_soc1o_SAVETB2J.nc")
        leg = build_paw_split_soc_leg(data, leg="z", sites_magnetic=[0])
        w_soc = paw_split_soc_band_w_soc(data, leg="z")
        predicted, _ = second_variation_spectrum(leg.eigenvalues, w_soc, lam=0.005)
        with nc4.Dataset(
            root / "native_0conv/fe_k8_native_0convo_DEN.nc"
        ) as d0, nc4.Dataset(
            root / "native_p005conv/fe_k8_native_p005convo_DEN.nc"
        ) as d1:
            np.testing.assert_array_equal(d0["density"][:], d1["density"][:])
        with nc4.Dataset(
            root / "native_0conv/fe_k8_native_0convo_EIG.nc"
        ) as z, nc4.Dataset(
            root / "native_p005conv/fe_k8_native_p005convo_EIG.nc"
        ) as small:
            native_shift = (
                27.211386245988
                * (
                    np.sort(np.asarray(small["Eigenvalues"][:]), axis=-1)
                    - np.sort(np.asarray(z["Eigenvalues"][:]), axis=-1)
                )[0]
            )
        predicted_shift = predicted - np.sort(leg.eigenvalues[0], axis=-1)
        assert np.max(np.abs(native_shift)) > 0.002  # nonzero 2.5-meV response
        residual = predicted_shift - native_shift
        assert np.max(np.abs(residual)) < 0.0012
        assert np.sqrt(np.mean(np.abs(residual) ** 2)) < 0.0001
