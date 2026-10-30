"""Real TBGreen consumers driven by complete SIESTA states and same-run H/S."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sisl")

from ase.units import kB  # noqa: E402
from HamiltonIO.siesta.wfsx import SiestaWFSXParser  # noqa: E402
from HamiltonIO.siesta.sisl_wrapper import SislParser  # noqa: E402
from TB2J.green import TBGreen  # noqa: E402

FIXTURES = Path(__file__).parent / "siesta_wfsx_fixtures"


def fixture_case(name):
    case = FIXTURES / name
    manifest = json.loads((case / "fixture.json").read_text())
    points = np.array([row["fractional"] for row in manifest["kpoints"]["points"]])
    return case, manifest, points


def test_opt_in_complete_spectrum_matches_full_hs_resolvent():
    case, manifest, points = fixture_case("scalar_nspin1")
    model = SislParser(str(case / manifest["files"]["fdf"])).get_model()
    green = TBGreen(
        model,
        kpts=points,
        kweights=np.full(len(points), 1 / len(points)),
        efermi=0,
        initial_emin=-100,
        complete_spectrum=True,
    )
    for ik, point in enumerate(points):
        h, overlap, _, _ = model.HSE_k(point)
        for energy in (-20 + 0.7j, -5 + 1j, 0.3 + 2j):
            np.testing.assert_allclose(
                green.get_Gk(ik, energy),
                np.linalg.inv(energy * overlap - h),
                atol=1e-10,
                rtol=1e-10,
            )


def source_model(tmp_path, name="scalar_nspin1", spin=0, reduce_scalar=False):
    case, manifest, points = fixture_case(name)
    structure, electrons, spin_info = (
        manifest["structure"],
        manifest["electrons"],
        manifest["spin"],
    )
    metadata = {
        "schema_version": 1,
        "cell_angstrom": structure["cell_angstrom"],
        "positions_angstrom": structure["positions_angstrom"],
        "atom_symbols": structure["atom_symbols"],
        "orbital_names_by_atom": structure["wfsx_basis_orbitals_by_atom"],
        "nspin": spin_info["nspin"],
        "spinor_layout": "interleaved" if spin_info["nspin"] in (4, 8) else None,
        "gauge": "orbital",
        "eigenvalue_reference": "absolute-eV",
        "fermi_energy_ev": electrons["fermi_energy_ev"],
        "fermi_source": electrons["fermi_energy_source"],
        "electron_count": electrons["total_electrons"],
        "electron_source": electrons["total_electrons_source"],
        "occupation_convention": electrons["occupation_convention"],
        "electronic_temperature_k": electrons["electronic_temperature_k"],
        "source_provenance": manifest["producer_run_id"],
        "weight_semantics": manifest["kpoints"]["weight_semantics"],
        "scalar_symmetry": spin_info["scalar_symmetry"] or {},
    }
    original_points = (
        [row["selected_fractional"] for row in manifest["kpoints"]["fold_map"]]
        if manifest["kpoints"]["fold_map"]
        else points.tolist()
    )
    metadata["scf_mesh"] = {
        "kpoints": original_points,
        "weights": [1 / len(original_points)] * len(original_points),
    }
    source = tmp_path / "states.WFSX"
    shutil.copyfile(case / manifest["files"]["wfsx"], source)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    parser = SiestaWFSXParser(source, json.loads(metadata_path.read_text()))
    model = SimpleNamespace(
        eigenstate_source=parser, spin_channel=spin, reduce_scalar=reduce_scalar
    )
    return model, metadata, manifest, np.array(original_points)


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_imported_scalar_serves_full_consumers_without_hs_or_eigensolve(
    tmp_path, monkeypatch, mode
):
    import scipy.linalg
    import sisl

    if mode == "default":
        monkeypatch.delenv("TB2J_SIESTA_GREEN_MODE", raising=False)
    else:
        monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    case, manifest, points = fixture_case("scalar_nspin1")
    model, metadata, _, _ = source_model(tmp_path)
    width = metadata["electronic_temperature_k"] * kB
    ordinary_model = SislParser(str(case / manifest["files"]["fdf"])).get_model()
    ordinary = TBGreen(
        ordinary_model,
        kpts=points,
        kweights=np.full(5, 0.2),
        efermi=0,
        initial_emin=-100,
        smearing_width=width,
        complete_spectrum=True,
    )
    operators = [ordinary_model.HSE_k(point)[:2] for point in points]
    assert {path.suffix for path in tmp_path.iterdir()} == {".WFSX", ".json"}
    calls = []
    real_lu = scipy.linalg.lu_factor

    def lu(*args, **kwargs):
        calls.append(1)
        return real_lu(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", lu)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "imported Green invoked electronic diagonalization or ordinary parsing"
        )

    for module in (np.linalg, scipy.linalg):
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(SislParser, "__init__", forbidden)
    real_get_sile = sisl.get_sile

    def only_states(path, *args, **kwargs):
        assert Path(path).resolve() == model.eigenstate_source.wfsx_path
        return real_get_sile(path, *args, **kwargs)

    monkeypatch.setattr(sisl, "get_sile", only_states)
    green = TBGreen(
        model,
        kpts=points,
        kweights=np.full(5, 0.2),
        efermi=0,
        initial_emin=-100,
        smearing_width=width,
        cache_path=tmp_path / "cache",
    )
    try:
        assert len(calls) == 5
        for ik, (h, overlap) in enumerate(operators):
            np.testing.assert_allclose(green.get_Hk(ik), h, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(green.get_Sk(ik), overlap, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(green.H0, ordinary.H0, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(
            green.get_density_matrix(),
            ordinary.get_density_matrix(),
            atol=1e-4,
            rtol=1e-5,
        )
        assert (
            abs(
                2 * np.trace(green.get_density_matrix()).real
                - metadata["electron_count"]
            )
            < 1e-4
        )
        rpoints = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])
        np.testing.assert_allclose(
            green.get_rho_R(rpoints), ordinary.get_rho_R(rpoints), atol=1e-4, rtol=1e-5
        )
        for energy in (-20 + 0.7j, -5 + 1j, 0.3 + 2j):
            np.testing.assert_allclose(
                green.get_Gk_all(energy),
                ordinary.get_Gk_all(energy),
                atol=2e-5,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                green.get_GR(rpoints, energy),
                ordinary.get_GR(rpoints, energy),
                atol=2e-5,
                rtol=1e-4,
            )
        assert len(calls) == 5
    finally:
        cache = Path(green.cache_path)
        green.clean_cache()
        assert not cache.exists()


def rewrite_test_records(path, update):
    from scipy.io import FortranFile, FortranEOFError

    records = []
    with FortranFile(path, "r") as stream:
        while True:
            try:
                records.append(stream.read_record(np.uint8))
            except FortranEOFError:
                break
    update(records)
    with FortranFile(path, "w") as stream:
        for record in records:
            stream.write_record(record)


@pytest.mark.parametrize("scale", [0, 1e-12])
def test_bad_coefficient_condition_refuses_and_cleans_only_owned_cache(tmp_path, scale):
    from HamiltonIO.siesta.wfsx import SingularEigenbasis

    model, _, manifest, points = source_model(tmp_path)
    path = model.eigenstate_source.wfsx_path

    def scale_one_ao_row(records):
        for band in range(manifest["states"]["no_u"]):
            records[9 + 3 * band].view(np.complex64)[0] *= scale

    rewrite_test_records(path, scale_one_ao_row)
    original = path.read_bytes()
    cache_parent = tmp_path / "cache"
    cache_parent.mkdir()
    marker = cache_parent / "user-file"
    marker.write_text("keep")
    with pytest.raises(SingularEigenbasis):
        TBGreen(model, kpts=points, kweights=np.full(5, 0.2), cache_path=cache_parent)
    assert path.read_bytes() == original
    assert list(cache_parent.iterdir()) == [marker]
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("mode", ["default", "contravariant"])
@pytest.mark.parametrize("spin", [0, 1])
def test_full_and_folded_collinear_consumers_match_complete_hs(
    tmp_path, monkeypatch, mode, spin
):
    if mode == "default":
        monkeypatch.delenv("TB2J_SIESTA_GREEN_MODE", raising=False)
    else:
        monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    models = []
    for name in ("collinear_nspin2_full", "collinear_nspin2_reduced"):
        directory = tmp_path / name
        directory.mkdir()
        models.append(source_model(directory, name, spin=spin, reduce_scalar=True))
    points = models[0][3][[2, 4, 0, 3, 1]]
    case, manifest, _ = fixture_case("collinear_nspin2_full")
    # Match gen_exchange_siesta's ordinary two-model construction.
    ordinary_model = SislParser(str(case / manifest["files"]["fdf"])).get_model()[spin]
    width = models[0][1]["electronic_temperature_k"] * kB
    ordinary = TBGreen(
        ordinary_model,
        kpts=points,
        kweights=np.full(5, 0.2),
        efermi=0,
        complete_spectrum=True,
        initial_emin=-100,
        smearing_width=width,
    )
    rpoints = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]])
    imported = []
    try:
        for model, _, _, _ in models:
            green = TBGreen(
                model,
                kpts=points,
                kweights=np.full(5, 0.2),
                efermi=0,
                initial_emin=-100,
                smearing_width=width,
                cache_path=tmp_path / "cache",
            )
            imported.append(green)
            assert green.spectral_stats["factorizations"] == 3
            np.testing.assert_allclose(green.H0, ordinary.H0, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(
                green.get_density_matrix(),
                ordinary.get_density_matrix(),
                atol=1e-4,
                rtol=1e-5,
            )
            for energy in (-20 + 0.7j, -5 + 1j, 0.3 + 2j):
                np.testing.assert_allclose(
                    green.get_Gk_all(energy),
                    ordinary.get_Gk_all(energy),
                    atol=2e-5,
                    rtol=1e-4,
                )
                np.testing.assert_allclose(
                    green.get_GR(rpoints, energy),
                    ordinary.get_GR(rpoints, energy),
                    atol=2e-5,
                    rtol=1e-4,
                )
                np.testing.assert_allclose(
                    green.get_GR(rpoints, energy, Gk_all=green.get_Gk_all(energy)),
                    ordinary.get_GR(
                        rpoints, energy, Gk_all=ordinary.get_Gk_all(energy)
                    ),
                    atol=2e-5,
                    rtol=1e-4,
                )
            assert green.spectral_stats["factorizations"] == 3
    finally:
        for green in imported:
            green.clean_cache()


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_full_spinor_consumers_match_complete_hs(tmp_path, monkeypatch, mode):
    if mode == "default":
        monkeypatch.delenv("TB2J_SIESTA_GREEN_MODE", raising=False)
    else:
        monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    model, metadata, manifest, points = source_model(
        tmp_path, "noncollinear_nspin4_full"
    )
    case, _, _ = fixture_case("noncollinear_nspin4_full")
    ordinary_model = SislParser(str(case / manifest["files"]["fdf"])).get_model()
    width = metadata["electronic_temperature_k"] * kB
    ordinary = TBGreen(
        ordinary_model,
        kpts=points,
        kweights=np.full(5, 0.2),
        efermi=0,
        complete_spectrum=True,
        initial_emin=-100,
        smearing_width=width,
    )
    green = TBGreen(
        model,
        kpts=points,
        kweights=np.full(5, 0.2),
        efermi=0,
        initial_emin=-100,
        smearing_width=width,
        cache_path=tmp_path / "cache",
    )
    try:
        assert green.spectral_stats["factorizations"] == 5
        np.testing.assert_allclose(green.H0, ordinary.H0, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(
            green.get_density_matrix(),
            ordinary.get_density_matrix(),
            atol=1e-4,
            rtol=1e-5,
        )
        assert (
            abs(np.trace(green.get_density_matrix()).real - metadata["electron_count"])
            < 1e-4
        )
        for energy in (-20 + 0.7j, -5 + 1j, 0.3 + 2j):
            np.testing.assert_allclose(
                green.get_Gk_all(energy),
                ordinary.get_Gk_all(energy),
                atol=2e-5,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                green.get_GR([[0, 0, 1]], energy),
                ordinary.get_GR([[0, 0, 1]], energy),
                atol=2e-5,
                rtol=1e-4,
            )
    finally:
        green.clean_cache()


def test_exported_partner_operator_disagreement_refuses_before_contour(tmp_path):
    from HamiltonIO.siesta.wfsx import InconsistentMetadata

    model, _, manifest, points = source_model(
        tmp_path, "collinear_nspin2_full", reduce_scalar=True
    )
    stride = 3 + 3 * manifest["states"]["no_u"]

    def change_partner(records):
        # Complete, finite test-only coefficient mutation at -0.2, spin up.
        for band in range(manifest["states"]["no_u"]):
            records[9 + 6 * stride + 3 * band].view(np.complex64)[0] *= 1.1

    rewrite_test_records(model.eigenstate_source.wfsx_path, change_partner)
    parent = tmp_path / "cache"
    with pytest.raises(InconsistentMetadata):
        TBGreen(model, kpts=points, kweights=np.full(5, 0.2), cache_path=parent)
    assert not list(parent.iterdir())


def permute_k_records(path, manifest, order):
    nch = 2 if manifest["spin"]["nspin"] == 2 else 1
    nbasis = manifest["states"]["no_u"] * (
        2 if manifest["spin"]["nspin"] in (4, 8) else 1
    )
    stride = 3 + 3 * nbasis

    def reorder(records):
        groups = [
            records[4 + ik * nch * stride : 4 + (ik + 1) * nch * stride]
            for ik in range(len(order))
        ]
        output = records[:4]
        for ik, old in enumerate(order):
            group = groups[old]
            for spin in range(nch):
                group[spin * stride][:4].view(np.int32)[0] = ik + 1
            output.extend(group)
        records[:] = output

    rewrite_test_records(path, reorder)


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_file_and_requested_permutations_preserve_full_green_arrays(
    tmp_path, monkeypatch, mode
):
    if mode == "default":
        monkeypatch.delenv("TB2J_SIESTA_GREEN_MODE", raising=False)
    else:
        monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first, _, _, points = source_model(
        first_dir, "collinear_nspin2_full", reduce_scalar=True
    )
    second, _, manifest, _ = source_model(
        second_dir, "collinear_nspin2_full", reduce_scalar=True
    )
    permute_k_records(second.eigenstate_source.wfsx_path, manifest, [3, 0, 4, 2, 1])
    order = np.array([2, 4, 0, 3, 1])
    green1 = TBGreen(
        first, kpts=points, kweights=np.full(5, 0.2), cache_path=tmp_path / "cache"
    )
    try:
        green2 = TBGreen(
            second,
            kpts=points[order] + [1, -2, 3],
            kweights=np.full(5, 0.2),
            cache_path=tmp_path / "cache",
        )
        try:
            np.testing.assert_allclose(green2.H0, green1.H0, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(
                green2.get_density_matrix(),
                green1.get_density_matrix(),
                atol=1e-4,
                rtol=1e-5,
            )
            for energy in (-5 + 1j, 0.3 + 2j):
                np.testing.assert_allclose(
                    green2.get_Gk_all(energy),
                    green1.get_Gk_all(energy)[order],
                    atol=2e-5,
                    rtol=1e-4,
                )
                np.testing.assert_allclose(
                    green2.get_GR([[0, 0, 1], [0, 0, -1]], energy),
                    green1.get_GR([[0, 0, 1], [0, 0, -1]], energy),
                    atol=2e-5,
                    rtol=1e-4,
                )
            assert (
                green1.spectral_stats["factorizations"]
                == green2.spectral_stats["factorizations"]
                == 3
            )
        finally:
            green2.clean_cache()
    finally:
        green1.clean_cache()


@pytest.mark.parametrize("indices", [[2], [3, 1, 4]])
def test_scalar_gamma_and_nonclosed_subset_keep_requested_quadrature(tmp_path, indices):
    model, metadata, manifest, original = source_model(tmp_path, reduce_scalar=True)
    points = original[indices]
    weights = np.full(len(indices), 1 / len(indices))
    case, _, _ = fixture_case("scalar_nspin1")
    ordinary_model = SislParser(str(case / manifest["files"]["fdf"])).get_model()
    width = metadata["electronic_temperature_k"] * kB
    ordinary = TBGreen(
        ordinary_model,
        kpts=points,
        kweights=weights,
        efermi=0,
        complete_spectrum=True,
        smearing_width=width,
    )
    green = TBGreen(
        model,
        kpts=points,
        kweights=weights,
        efermi=0,
        smearing_width=width,
        cache_path=tmp_path / "cache",
    )
    try:
        assert green.spectral_stats["factorizations"] == len(indices)
        np.testing.assert_array_equal(green.kpts, points)
        np.testing.assert_array_equal(green.kweights, weights)
        np.testing.assert_allclose(green.H0, ordinary.H0, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(
            green.get_density_matrix(),
            ordinary.get_density_matrix(),
            atol=1e-4,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            green.get_GR([[0, 0, 1]], -5 + 1j),
            ordinary.get_GR([[0, 0, 1]], -5 + 1j),
            atol=2e-5,
            rtol=1e-4,
        )
    finally:
        green.clean_cache()


@pytest.mark.parametrize(
    "name", ["scalar_nspin1", "collinear_nspin2_full", "noncollinear_nspin4_full"]
)
def test_wrong_declared_electron_count_refuses_before_contour(tmp_path, name):
    from HamiltonIO.siesta.wfsx import InconsistentMetadata

    model, metadata, _, points = source_model(tmp_path, name)
    metadata["electron_count"] = 10
    model.eigenstate_source = SiestaWFSXParser(
        model.eigenstate_source.wfsx_path, metadata
    )
    with pytest.raises(InconsistentMetadata):
        TBGreen(
            model, kpts=points, kweights=np.full(5, 0.2), cache_path=tmp_path / "cache"
        )


def test_declared_temperature_is_used_for_electron_validation(tmp_path):
    from HamiltonIO.siesta.wfsx import InconsistentMetadata

    model, metadata, _, points = source_model(tmp_path)
    metadata["electronic_temperature_k"] = 100000
    model.eigenstate_source = SiestaWFSXParser(
        model.eigenstate_source.wfsx_path, metadata
    )
    with pytest.raises(InconsistentMetadata):
        TBGreen(
            model, kpts=points, kweights=np.full(5, 0.2), cache_path=tmp_path / "cache"
        )


def test_electron_validation_includes_unselected_spin_and_full_scf_domain(tmp_path):
    from HamiltonIO.siesta.wfsx import InconsistentMetadata

    model, _, manifest, points = source_model(tmp_path, "collinear_nspin2_full", spin=0)
    nbasis = manifest["states"]["no_u"]
    stride = 3 + 3 * nbasis

    def change_other_spin_outside_requested_subset(records):
        start = 4 + stride  # First k, down-spin record; request is Gamma/up.
        for band in range(nbasis):
            records[start + 4 + 3 * band].view(np.float64)[:] += 100

    rewrite_test_records(
        model.eigenstate_source.wfsx_path, change_other_spin_outside_requested_subset
    )
    with pytest.raises(InconsistentMetadata):
        TBGreen(
            model, kpts=points[[2]], kweights=np.ones(1), cache_path=tmp_path / "cache"
        )


def test_spectral_preparation_requires_independent_provenance_mesh(tmp_path):
    from HamiltonIO.siesta.wfsx import MissingMetadata

    model, metadata, _, points = source_model(tmp_path)
    metadata.pop("scf_mesh", None)
    model.eigenstate_source = SiestaWFSXParser(
        model.eigenstate_source.wfsx_path, metadata
    )
    with pytest.raises(MissingMetadata):
        TBGreen(
            model, kpts=points, kweights=np.full(5, 0.2), cache_path=tmp_path / "cache"
        )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("factory", [False, True])
def test_ordinary_jax_subclass_retains_density_and_green_consumers(
    tmp_path, monkeypatch, factory, cached
):
    jax = pytest.importorskip("jax")
    from TB2J.GreenGPU import TBGreenGPU
    from TB2J.gpu import jax_utils

    # Exercise the real JAX numerical backend on CPU without requiring hardware.
    monkeypatch.setattr(jax_utils, "_require_jax_runtime", lambda: None)
    h = np.array([[-2, 0.3], [0.3, 1]], dtype=complex)
    model = SimpleNamespace(
        is_orthogonal=False,
        R2kfactor=2j * np.pi,
        norb=2,
        nbasis=2,
        Rlist=np.zeros((1, 3)),
        HR=h[None],
        SR=np.eye(2)[None],
    )
    constructor = TBGreen if factory else TBGreenGPU
    options = {"use_gpu": True} if factory else {}
    with jax.default_device(jax.devices("cpu")[0]):
        green = constructor(
            model,
            kpts=np.zeros((1, 3)),
            kweights=np.ones(1),
            efermi=0,
            initial_emin=-100,
            use_cache=cached,
            cache_path=tmp_path if cached else None,
            **options,
        )
    z = -1 + 0.8j
    np.testing.assert_allclose(
        green.get_Gk(0, z), np.linalg.inv(z * np.eye(2) - h), atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        green.get_GR([[0, 0, 1]], z)[0], green.get_Gk(0, z), atol=1e-10, rtol=1e-10
    )
    e, c = np.linalg.eigh(h)
    expected_density = (c * (1 / (1 + np.exp(e / 0.01)))) @ c.conj().T
    np.testing.assert_allclose(
        green.get_density_matrix(), expected_density, atol=1e-10, rtol=1e-10
    )
    green.clean_cache()
    assert not list(tmp_path.iterdir())


def test_truncated_late_record_preserves_inputs_and_closes_partial_cache(tmp_path):
    from HamiltonIO.siesta.wfsx import IncompleteSpectrum

    model, _, _, points = source_model(tmp_path)
    path = model.eigenstate_source.wfsx_path
    truncated = path.read_bytes()[:-8]
    path.write_bytes(truncated)
    parent = tmp_path / "cache"
    parent.mkdir()
    marker = parent / "user-file"
    marker.write_text("keep")
    with pytest.raises(IncompleteSpectrum):
        TBGreen(model, kpts=points, kweights=np.full(5, 0.2), cache_path=parent)
    assert path.read_bytes() == truncated
    assert list(parent.iterdir()) == [marker]
    assert marker.read_text() == "keep"
