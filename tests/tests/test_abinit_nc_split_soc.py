"""ABINIT NC split-SOC consumer tests (story-009).

All tests are synthetic and deterministic.  The gate semantics encode the
story adjudication: one collinear reference determines only the raw 2x2
transverse block (J_xx, J_xy, J_yx, J_yy) and D_z — never a full 3x3 J —
so the dimer FR-032-style equivalence is enforced projection-only, and the
full tensor is meaningful only after the rank-9 three-leg merge.
"""

import json

import numpy as np
import pytest

from TB2J.interfaces.abinit_nc_split_soc import (
    NC_SOC_KS_SCHEMA_NAME,
    SOC_OFF_LEG_DIRECTIONS,
    SocKsSidecar,
    _tangent_projection_check,
    dualize_pao_coefficients,
    gen_exchange_abinit_nc_split_soc,
    load_soc_ks_sidecar,
    sha256_of_file,
    validate_sidecar_pairing,
)

HARTREE_TO_EV = 27.211386245988

RNG = np.random.default_rng(20260928)


# ---------------------------------------------------------------------------
# synthetic fixture builders
# ---------------------------------------------------------------------------


def _hermitian(rng, *shape):
    mat = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return (mat + np.conj(np.swapaxes(mat, -1, -2))) / 2


def _synthetic_pao_hs(path, rng, nkpt=2, nband=3, nproj_site=4, nsites=2):
    """Write an ``abinit.nc_pao_hs`` v2 dimer file (full-BZ, eV in memory)."""
    from netCDF4 import Dataset

    nproj = nproj_site * nsites
    cell = np.eye(3) * 6.0
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    kpts = np.array([[0.0, 0.0, 0.0], [0.37, -0.21, 0.11]])
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = np.sort(rng.normal(loc=1.0, scale=2.0, size=(2, nkpt, nband)), axis=2)
    coefficients = rng.normal(size=(2, nkpt, nband, nproj)) + 1j * rng.normal(
        size=(2, nkpt, nband, nproj)
    )
    overlap = _hermitian(rng, nkpt, nproj, nproj)
    overlap += np.eye(nproj)[None] * (2.0 + nproj)
    delta = _hermitian(rng, nsites, nproj_site, nproj_site)
    for site in range(nsites):
        delta[site, 0, 0] += 0.4 * (site + 1)

    delta_global = np.zeros((nproj, nproj), dtype=complex)
    for site in range(nsites):
        sl = slice(site * nproj_site, (site + 1) * nproj_site)
        delta_global[sl, sl] = delta[site]

    with Dataset(path, "w", format="NETCDF4") as nc:
        for name, size in (
            ("nproj", nproj),
            ("nkpt_ibz", nkpt),
            ("nsppol", 2),
            ("nband", nband),
            ("natom", nsites),
            ("ntypat", 1),
            ("three", 3),
            ("nkpt_bz", nkpt),
        ):
            nc.createDimension(name, size)
        nc.schema_name = "abinit.nc_pao_hs"
        nc.schema_version = "2"
        nc.basis_type = "pseudo_atomic_orbital"
        nc.complex_storage = "split_real_imag_variables"
        nc.overlap_exchange_ready = np.int32(1)
        nc.fermi_energy = 0.2

        nc.createVariable("atom_index", "i4", ("nproj",))[:] = np.tile(
            np.arange(nsites) + 1, (nproj_site, 1)
        ).T.ravel()
        nc.createVariable("atom_types", "i4", ("natom",))[:] = [1] * nsites
        nc.createVariable("atomic_numbers", "i4", ("ntypat",))[:] = [26]

        cell_var = nc.createVariable("primitive_vectors", "f8", ("three", "three"))
        cell_var[:] = cell
        cell_var.units = "angstrom"
        cell_var.mnemonics = "cell vectors"
        pos_var = nc.createVariable("atom_positions", "f8", ("natom", "three"))
        pos_var[:] = positions
        pos_var.units = "angstrom"
        pos_var.mnemonics = "cartesian"

        nc.createVariable("kpoints_ibz", "f8", ("nkpt_ibz", "three"))[:] = kpts
        nc.createVariable("kweights_ibz", "f8", ("nkpt_ibz",))[:] = weights
        nc.createVariable("eigenvalues", "f8", ("nsppol", "nkpt_ibz", "nband"))[:] = (
            eigenvalues / HARTREE_TO_EV
        )

        def write_split(name, dims, array):
            nc.createVariable(f"{name}_real", "f8", dims)[:] = np.real(array)
            nc.createVariable(f"{name}_imag", "f8", dims)[:] = np.imag(array)

        write_split(
            "coefficients_ibz",
            ("nsppol", "nkpt_ibz", "nband", "nproj"),
            np.conj(coefficients),
        )
        write_split("overlap_ibz", ("nkpt_ibz", "nproj", "nproj"), overlap)
        write_split("delta_total", ("nproj", "nproj"), delta_global / HARTREE_TO_EV)

        nc.createVariable("kpoints_bz", "f8", ("nkpt_bz", "three"))[:] = kpts
        nc.createVariable("eigenvalues_bz", "f8", ("nsppol", "nkpt_bz", "nband"))[:] = (
            eigenvalues / HARTREE_TO_EV
        )
        write_split(
            "coefficients_bz",
            ("nsppol", "nkpt_bz", "nband", "nproj"),
            np.conj(coefficients),
        )
        write_split("overlap_bz", ("nkpt_bz", "nproj", "nproj"), overlap)
        nc.createVariable("bz_to_ibz", "i4", ("nkpt_bz",))[:] = [1, 2]
    return path


def _sidecar_from_pao(pao_path, sidecar_path, rng, pao_hs_sha=None, **overrides):
    """Write an ``abinao.nc_soc_ks`` v1 sidecar matched to the PAO fixture."""
    from netCDF4 import Dataset

    with Dataset(pao_path) as nc:
        kpts = np.asarray(nc.variables["kpoints_bz"][:], dtype=float)
        eigen_ha = np.asarray(nc.variables["eigenvalues_bz"][:], dtype=float)
    nkpt = kpts.shape[0]
    nband = eigen_ha.shape[2]
    nc_band = 2 * nband
    stacked = np.empty((nkpt, nc_band))
    stacked[:, 0::2] = eigen_ha[0] * HARTREE_TO_EV
    stacked[:, 1::2] = eigen_ha[1] * HARTREE_TO_EV
    w_soc = _hermitian(rng, 3, nkpt, nc_band, nc_band)
    w_soc *= 1.0e-3

    attrs = {
        "schema_name": NC_SOC_KS_SCHEMA_NAME,
        "schema_version": "1",
        "source_wfk": "synthetic_WFK.nc",
        "source_wfk_sha256": "0" * 64,
        "energy_unit": "eV",
        "hartree_to_ev": HARTREE_TO_EV,
        "spinor_band_index": "2*n + sigma (sigma: 0=up, 1=dn)",
        "band_window_lo": np.int64(-1),
        "band_window_hi": np.int64(-1),
        "spnorbscl": 1.0,
        "all_atoms_covered": np.int64(1),
        "created": "2026-09-28T00:00:00Z",
        "fr050_metadata": json.dumps(
            {
                "legs": list(SOC_OFF_LEG_DIRECTIONS),
                "strength0_provenance": {"nsppol": 2, "nspinor": 1},
                "operator": {"source": "synthetic test kernel", "units": "hartree"},
                "energy_unit": "eV",
            }
        ),
    }
    attrs.update(overrides.pop("attrs", {}))
    with Dataset(sidecar_path, "w", format="NETCDF4") as nc:
        nc.createDimension("nleg", 3)
        nc.createDimension("n3", 3)
        nc.createDimension("nkpt", nkpt)
        nc.createDimension("nspinor_band", nc_band)
        leg_var = nc.createVariable("leg", str, ("nleg",))
        leg_var[:] = np.array(list(SOC_OFF_LEG_DIRECTIONS), dtype=object)
        nc.createVariable("kpts", "f8", ("nkpt", "n3"))[:] = kpts
        nc.createVariable("kweights", "f8", ("nkpt",))[:] = 1.0 / nkpt
        nc.createVariable("spinaxis", "f8", ("nleg", "n3"))[:] = np.eye(3)
        nc.createVariable(
            "w_so_real", "f8", ("nleg", "nkpt", "nspinor_band", "nspinor_band")
        )[:] = np.real(w_soc)
        nc.createVariable(
            "w_so_imag", "f8", ("nleg", "nkpt", "nspinor_band", "nspinor_band")
        )[:] = np.imag(w_soc)
        nc.createVariable("eigenvalues_ev", "f8", ("nleg", "nkpt", "nspinor_band"))[
            :
        ] = stacked[None].repeat(3, axis=0)
        for name, value in attrs.items():
            setattr(nc, name, value)
        if pao_hs_sha is not None:
            nc.pao_hs = "PAO_HS.nc"
            nc.pao_hs_sha256 = pao_hs_sha
    return sidecar_path


@pytest.fixture()
def synthetic_dimer(tmp_path):
    rng = np.random.default_rng(4242)
    pao = _synthetic_pao_hs(tmp_path / "PAO_HS.nc", rng)
    sidecar = _sidecar_from_pao(
        pao, tmp_path / "nc_soc_ks.nc", rng, pao_hs_sha=sha256_of_file(pao)
    )
    return pao, sidecar


RPTS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


# ---------------------------------------------------------------------------
# sidecar reader and refusals
# ---------------------------------------------------------------------------


def test_sidecar_reader_roundtrip(synthetic_dimer):
    pytest.importorskip("netCDF4")
    _, sidecar_path = synthetic_dimer
    sidecar = load_soc_ks_sidecar(sidecar_path)
    assert sidecar.legs == ("x", "y", "z")
    assert sidecar.nband_composite % 2 == 0
    assert sidecar.spnorbscl == 1.0
    assert sidecar.all_atoms_covered
    assert sidecar.band_window is None
    assert sidecar.pao_hs_sha256 is not None
    assert sidecar.fr050["strength0_provenance"]["nspinor"] == 1


def test_sidecar_refusals(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    _, sidecar_path = synthetic_dimer
    import shutil

    def mutated(mutate, expect):
        target = tmp_path / f"bad_{expect[:12]}.nc"
        shutil.copy(sidecar_path, target)
        mutate(target)
        with pytest.raises(ValueError, match=expect):
            load_soc_ks_sidecar(target)

    from netCDF4 import Dataset

    def set_attr(name, value):
        def _m(path):
            with Dataset(path, "a") as nc:
                setattr(nc, name, value)

        return _m

    mutated(set_attr("schema_version", "2"), "schema_version")
    mutated(set_attr("spnorbscl", 0.5), "non-unit SOC scaling")
    mutated(set_attr("all_atoms_covered", np.int64(0)), "all atoms")
    mutated(set_attr("energy_unit", "hartree"), "eV")

    def break_hermitian(path):
        with Dataset(path, "a") as nc:
            real = np.asarray(nc.variables["w_so_real"][:])
            real[0, 0, 1, 0] += 0.5
            nc.variables["w_so_real"][:] = real

    mutated(break_hermitian, "Hermitian")

    def break_weights(path):
        with Dataset(path, "a") as nc:
            nc.variables["kweights"][:] = [0.3, 0.3]

    mutated(break_weights, "ungauged")

    def break_axis(path):
        with Dataset(path, "a") as nc:
            axis = np.asarray(nc.variables["spinaxis"][:])
            axis[0] = [0.0, 1.0, 0.0]
            nc.variables["spinaxis"][:] = axis

    mutated(break_axis, "does not align")


def test_sidecar_spinor_flavor_refused(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    pao, sidecar_path = synthetic_dimer
    data = load_abinit_nc_pao_savetb2j(pao)
    import json as json_mod

    from netCDF4 import Dataset

    target = tmp_path / "spinor_flavor.nc"
    shutil = __import__("shutil")
    shutil.copy(sidecar_path, target)
    with Dataset(target, "a") as nc:
        fr050 = json_mod.loads(str(nc.fr050_metadata))
        fr050["strength0_provenance"] = {"nsppol": 1, "nspinor": 2}
        nc.fr050_metadata = json_mod.dumps(fr050)
    spinor_sidecar = load_soc_ks_sidecar(target)
    with pytest.raises(ValueError, match="spinor-flavor"):
        validate_sidecar_pairing(spinor_sidecar, data, pao)


def test_pairing_refusals(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    pao, sidecar_path = synthetic_dimer
    data = load_abinit_nc_pao_savetb2j(pao)
    sidecar = load_soc_ks_sidecar(sidecar_path)

    # hash mismatch
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_sidecar_pairing(sidecar, data, pao, wfk_path=pao)

    # missing sidecar hash at all
    import shutil

    from netCDF4 import Dataset

    no_hash = tmp_path / "no_hash.nc"
    shutil.copy(sidecar_path, no_hash)
    with Dataset(no_hash, "a") as nc:
        del nc.pao_hs_sha256
        del nc.pao_hs
    bad_hash = load_soc_ks_sidecar(no_hash)
    assert bad_hash.pao_hs_sha256 is None

    # k-point order mismatch
    swapped = SocKsSidecar(
        legs=sidecar.legs,
        kpoints=sidecar.kpoints[::-1].copy(),
        kweights=sidecar.kweights[::-1].copy(),
        spinaxis=sidecar.spinaxis,
        w_so=sidecar.w_so,
        w_so_site=None,
        eigenvalues_ev=sidecar.eigenvalues_ev,
        band_window=None,
        source_wfk=sidecar.source_wfk,
        source_wfk_sha256=sidecar.source_wfk_sha256,
        pao_hs=sidecar.pao_hs,
        pao_hs_sha256=sidecar.pao_hs_sha256,
        spnorbscl=sidecar.spnorbscl,
        all_atoms_covered=True,
        fr050=sidecar.fr050,
    )
    with pytest.raises(ValueError, match="k-point"):
        validate_sidecar_pairing(swapped, data, pao)


# ---------------------------------------------------------------------------
# dualization and leg construction
# ---------------------------------------------------------------------------


def test_dualize_matches_inverse_overlap_mode(synthetic_dimer):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j
    from TB2J.projector_green import ProjectorGreen

    pao, sidecar_path = synthetic_dimer
    data = load_abinit_nc_pao_savetb2j(pao)
    sidecar = load_soc_ks_sidecar(sidecar_path)
    validate_sidecar_pairing(sidecar, data, pao)
    dual = dualize_pao_coefficients(data)
    assert dual.overlap_k is None
    green_dual = ProjectorGreen(dual)
    green_ref = ProjectorGreen(data, overlap_mode="inverse")
    energy = -3.1 + 0.4j
    for ik in range(data.nkpt):
        for spin in range(2):
            np.testing.assert_allclose(
                green_dual.get_Gk(ik, energy, ispin=spin),
                green_ref.get_Gk(ik, energy, ispin=spin),
                atol=1e-11,
            )


def test_build_leg_composite_mapping_and_vertices(synthetic_dimer):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_nc_split_soc import build_nc_split_soc_leg
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    pao, sidecar_path = synthetic_dimer
    data = load_abinit_nc_pao_savetb2j(pao)
    sidecar = load_soc_ks_sidecar(sidecar_path)
    dual = dualize_pao_coefficients(data)
    leg = build_nc_split_soc_leg(sidecar, dual, leg="z", sites_magnetic=[0, 1])
    assert leg.nspinor == 2
    assert leg.coefficients.shape == (1, data.nkpt, 2 * data.nband, 2, data.nproj)
    for spin in range(2):
        np.testing.assert_array_equal(
            leg.eigenvalues[0, :, spin::2], data.eigenvalues[spin]
        )
        np.testing.assert_array_equal(
            leg.coefficients[0, :, spin::2, spin, :], dual.coefficients[spin]
        )
        other = 1 - spin
        np.testing.assert_array_equal(leg.coefficients[0, :, spin::2, other, :], 0.0)
    nproj_site = data.site_nproj[0]
    vertex = leg.spinor_operator[0, :nproj_site, :nproj_site]
    assert np.abs(vertex[:, :, 0, 1]).max() == 0.0
    delta = data.get_operator_component("delta_total", site=0)
    np.testing.assert_allclose(vertex[:, :, 0, 0], delta)
    # V = Delta (x) sigma_z: the sigma_down block carries -Delta
    np.testing.assert_allclose(vertex[:, :, 1, 1], -delta)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


class _MergedStub:
    def __init__(self, exchange_Jdict, dmi_ddict, Jani_dict):
        self.exchange_Jdict = exchange_Jdict
        self.dmi_ddict = dmi_ddict
        self.Jani_dict = Jani_dict


def _gate_fixture():
    cell = np.eye(3) * 6.0
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    key = ((1, 0, 0), 0, 1)
    jiso, dmi, jani = -0.02, np.array([0.0, 0.0, 3e-4]), np.diag([1e-3, -5e-4, -5e-4])
    entry = {"Jiso": jiso, "dmi": dmi, "jani": jani}
    merged = _MergedStub({key: jiso}, {key: dmi}, {key: jani})
    return key, entry, merged, cell, positions


def test_tangent_projection_gate_passes_and_ignores_undetermined():
    key, entry, merged, cell, positions = _gate_fixture()
    report = _tangent_projection_check(
        merged, {key: entry}, [0, 1], cell, positions, 1e-2
    )
    assert report["passed"]
    assert report["pairs_compared"] == 1
    assert report["max_transverse_dev_eV"] == pytest.approx(0.0, abs=1e-14)

    # a perturbation confined to components NOT determined by one z
    # reference (DMI_x, Jani_zz, ...) must NOT fail the projection gate
    merged_x = _MergedStub(
        {key: entry["Jiso"]},
        {key: entry["dmi"] + np.array([5e-2, 0.0, 0.0])},
        {key: entry["jani"] + np.diag([0.0, 0.0, 9e-2])},
    )
    report = _tangent_projection_check(
        merged_x, {key: entry}, [0, 1], cell, positions, 1e-2
    )
    assert report["passed"]

    # perturbing the determined transverse block must fail the gate
    merged_bad = _MergedStub(
        {key: entry["Jiso"]},
        {key: entry["dmi"]},
        {key: entry["jani"] + np.diag([9e-2, 0.0, 0.0])},
    )
    with pytest.raises(ValueError, match="tangent projection gate FAILED"):
        _tangent_projection_check(
            merged_bad, {key: entry}, [0, 1], cell, positions, 1e-2
        )


def test_tangent_projection_gate_needs_nononsite_pairs():
    _, entry, merged, cell, positions = _gate_fixture()
    colocated = np.zeros((2, 3))
    with pytest.raises(ValueError, match="no pairs"):
        _tangent_projection_check(
            merged, {((0, 0, 0), 0, 1): entry}, [0, 1], cell, colocated, 1e-2
        )


# ---------------------------------------------------------------------------
# end-to-end driver (story-009 TEST-001/TEST-002 mechanics)
# ---------------------------------------------------------------------------


def test_gen_exchange_end_to_end_synthetic_dimer(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    pao, sidecar = synthetic_dimer
    result = gen_exchange_abinit_nc_split_soc(
        pao,
        sidecar,
        output_path=tmp_path / "TB2J_results_nc_split_soc",
        index_magnetic_atoms=[0, 1],
        Rpts=RPTS,
        nz=12,
        wfk=None,
        verify_tangent_projection=False,
    )
    assert set(result["leg_paths"]) == {"x", "y", "z"}
    for leg, provenance in result["leg_metadata"].items():
        assert provenance["leg"] == leg
        assert provenance["soc_off_anchor"]["passed"] is True
        study = provenance["band_window"]["convergence_study"]
        assert [w["nband"] for w in study["windows"]] == [4, 6]
        assert (
            tmp_path / "TB2J_results_nc_split_soc" / f"leg_{leg}" / "exchange.out"
        ).exists()
        assert (
            tmp_path
            / "TB2J_results_nc_split_soc"
            / f"leg_{leg}"
            / "split_soc_provenance.json"
        ).exists()
    assert result["tangent_projection_check"] is None
    merged = json.loads(
        (
            tmp_path / "TB2J_results_nc_split_soc" / "split_soc_provenance.json"
        ).read_text()
    )
    assert merged["backend"] == "abinit_nc"
    assert "rank-9" in merged["full_tensor"]
    assert (tmp_path / "TB2J_results_nc_split_soc" / "TB2J.pickle").exists()


def test_gen_refuses_mismatched_pao_hash(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    import shutil

    from netCDF4 import Dataset

    pao, sidecar = synthetic_dimer
    tampered = tmp_path / "tampered_pao.nc"
    shutil.copy(pao, tampered)
    with Dataset(tampered, "a") as nc:
        eigen = np.asarray(nc.variables["eigenvalues_bz"][:])
        eigen[0, 0, 0] += 0.5 / HARTREE_TO_EV
        nc.variables["eigenvalues_bz"][:] = eigen
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        gen_exchange_abinit_nc_split_soc(
            tampered,
            sidecar,
            output_path=tmp_path / "never_written",
            index_magnetic_atoms=[0, 1],
            Rpts=RPTS,
        )


def test_gen_requires_explicit_magnetic_sites(synthetic_dimer, tmp_path):
    pytest.importorskip("netCDF4")
    pao, sidecar = synthetic_dimer
    with pytest.raises(ValueError, match="index_magnetic_atoms"):
        gen_exchange_abinit_nc_split_soc(
            pao,
            sidecar,
            output_path=tmp_path / "never_written",
            Rpts=RPTS,
        )
