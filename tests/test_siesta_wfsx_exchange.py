"""Existing exchange kernels consume complete imported SIESTA spectra."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sisl")

from ase.units import kB  # noqa: E402
from HamiltonIO.siesta.sisl_wrapper import SislParser  # noqa: E402
from test_siesta_wfsx_green import fixture_case, source_model  # noqa: E402

from TB2J.exchange import ExchangeNCL  # noqa: E402
from TB2J.exchangeCL2 import ExchangeCL2  # noqa: E402


def ordinary_exchange(name, **params):
    case, manifest, _ = fixture_case(name)
    model = SislParser(str(case / manifest["files"]["fdf"])).get_model()
    first = model[0] if isinstance(model, tuple) else model
    basis = dict(zip(first.orbs, range(first.nbasis)))
    cls = ExchangeCL2 if isinstance(model, tuple) else ExchangeNCL
    return cls(
        tbmodels=model,
        atoms=first.atoms,
        basis=basis,
        efermi=0,
        complete_spectrum=True,
        **params,
    )


def exchange_params(tmp_path):
    return dict(
        kmesh=[1, 1, 5],
        magnetic_elements=["O"],
        Rcut=4.3,
        nz=8,
        nproc=1,
        smearing=100 * kB,
        output_path=str(tmp_path),
    )


def assert_jdict_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], atol=1e-6, rtol=1e-4, err_msg=str(key)
        )


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_collinear_full_internal_fold_and_representatives_match_hs_exchange(
    tmp_path, monkeypatch, mode
):
    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    params = exchange_params(tmp_path)
    oracle = ordinary_exchange("collinear_nspin2_full", **params)
    oracle.calculate_all()
    inputs = [
        ("collinear_nspin2_full", False, 5),
        ("collinear_nspin2_full", True, 3),
        ("collinear_nspin2_reduced", False, 3),
    ]
    rpoints = [[0, 0, 0], [0, 0, 1], [0, 0, -1]]
    for index, (name, fold, representatives) in enumerate(inputs):
        folder = tmp_path / str(index)
        folder.mkdir()
        model, _, _, _ = source_model(folder, name)
        exchange = prepare_exchange_siesta_wfsx(
            model.eigenstate_source,
            reduce_scalar=fold,
            cache_path=folder / "cache",
            **params,
        )
        try:
            np.testing.assert_allclose(
                exchange.Delta, oracle.Delta, atol=2e-5, rtol=1e-5
            )
            for actual, expected in [
                (exchange.Gup, oracle.Gup),
                (exchange.Gdn, oracle.Gdn),
            ]:
                assert actual.spectral_stats["factorizations"] == representatives
                np.testing.assert_allclose(actual.H0, expected.H0, atol=2e-5, rtol=1e-5)
                np.testing.assert_allclose(
                    actual.get_density_matrix(),
                    expected.get_density_matrix(),
                    atol=1e-4,
                    rtol=1e-5,
                )
                for z in [-20 + 0.7j, -5 + 1j, 0.3 + 2j]:
                    np.testing.assert_allclose(
                        actual.get_Gk_all(z),
                        expected.get_Gk_all(z),
                        atol=2e-5,
                        rtol=1e-4,
                    )
                    np.testing.assert_allclose(
                        actual.get_GR(rpoints, z),
                        expected.get_GR(rpoints, z),
                        atol=2e-5,
                        rtol=1e-4,
                    )
            exchange.calculate_all()
            assert_jdict_equal(exchange.exchange_Jdict, oracle.exchange_Jdict)
            np.testing.assert_allclose(
                exchange.charges, oracle.charges, atol=1e-4, rtol=1e-5
            )
            np.testing.assert_allclose(
                exchange.spinat, oracle.spinat, atol=1e-4, rtol=1e-5
            )
            assert exchange.Gup.spectral_stats["factorizations"] == representatives
            assert exchange.Gdn.spectral_stats["factorizations"] == representatives
            expected_expansions = (
                0 if representatives == 5 else 2 * (6 + len(exchange.contour.path))
            )
            assert (
                exchange.Gup.spectral_stats["scalar_partner_expansions"]
                == expected_expansions
            )
            assert (
                exchange.Gdn.spectral_stats["scalar_partner_expansions"]
                == expected_expansions
            )
        finally:
            exchange.finalize()


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_full_spinor_tensor_uses_existing_kernel_without_scalar_reconstruction(
    tmp_path, monkeypatch, mode
):
    import scipy.linalg
    import sisl

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx
    from TB2J.io_exchange import SpinIO

    monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    params = exchange_params(tmp_path)
    oracle = ordinary_exchange("noncollinear_nspin4_full", **params)
    oracle.calculate_all()
    oracle.write_output(path=tmp_path / "ordinary")
    expected = SpinIO.load_pickle(path=tmp_path / "ordinary")
    folder = tmp_path / "input"
    folder.mkdir()
    model, _, _, _ = source_model(folder, "noncollinear_nspin4_full")
    assert {path.name for path in folder.iterdir()} == {"states.WFSX", "metadata.json"}

    def forbidden(*args, **kwargs):
        pytest.fail(
            "WFSX exchange invoked ordinary parsing or an electronic eigensolver"
        )

    monkeypatch.setattr(SislParser, "__init__", forbidden)
    for module in (np.linalg, scipy.linalg):
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            monkeypatch.setattr(module, name, forbidden)
    real_sile = sisl.get_sile

    def states_only(path, *args, **kwargs):
        assert str(path) == str(model.eigenstate_source.wfsx_path)
        return real_sile(path, *args, **kwargs)

    monkeypatch.setattr(sisl, "get_sile", states_only)
    exchange = prepare_exchange_siesta_wfsx(
        model.eigenstate_source, cache_path=folder / "cache", **params
    )
    try:
        np.testing.assert_allclose(exchange.HR0, oracle.HR0, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(exchange.rho, oracle.rho, atol=1e-4, rtol=1e-5)
        exchange.calculate_all()
        exchange.write_output(path=tmp_path / "wavefunctions")
        actual = SpinIO.load_pickle(path=tmp_path / "wavefunctions")
        expected_tensors = dict(
            zip(
                map(tuple, expected.Rlist),
                expected.get_full_Jtensor_for_Rlist(order="ij33"),
            )
        )
        actual_tensors = dict(
            zip(
                map(tuple, actual.Rlist),
                actual.get_full_Jtensor_for_Rlist(order="ij33"),
            )
        )
        assert_jdict_equal(actual_tensors, expected_tensors)
        assert_jdict_equal(exchange.exchange_Jdict, oracle.exchange_Jdict)
        assert_jdict_equal(exchange.Jani, oracle.Jani)
        assert_jdict_equal(exchange.DMI, oracle.DMI)
        assert exchange.G.spectral_stats["scalar_partner_expansions"] == 0
        assert exchange.G.spectral_stats["factorizations"] == 5
    finally:
        exchange.finalize()


@pytest.mark.parametrize(
    "options",
    [
        {"operation": "split-soc"},
        {"operation": "mae"},
        {"operation": "operator-rescaling"},
        {"operation": "spin-spiral"},
        {"operation": "gbt"},
        {"read_H_soc": True},
        {"orth": True},
        {"use_gpu": True},
        {"ibz": True},
        {"so_strength": 1},
        {"mae_angles": [0]},
        {"nproc": 2},
        {"dmft": True},
    ],
)
def test_unsupported_operations_refuse_before_factorization(
    tmp_path, monkeypatch, options
):
    import scipy.linalg
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, _, _ = source_model(tmp_path, "collinear_nspin2_full")

    def forbidden(*args, **kwargs):
        pytest.fail(
            "unsupported request reached ordinary parsing or spectral factorization"
        )

    monkeypatch.setattr(SislParser, "__init__", forbidden)
    monkeypatch.setattr(scipy.linalg, "lu_factor", forbidden)
    with pytest.raises(UnsupportedOperatorMode):
        prepare_exchange_siesta_wfsx(
            model.eigenstate_source, **{**exchange_params(tmp_path), **options}
        )


def test_spinor_reduction_refuses_before_factorization(tmp_path, monkeypatch):
    import scipy.linalg
    from HamiltonIO.siesta.wfsx import UnsupportedSpinMode

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, _, _ = source_model(tmp_path, "noncollinear_nspin4_full")

    def forbidden(*args, **kwargs):
        pytest.fail(
            "spinor reduction reached ordinary parsing or spectral factorization"
        )

    monkeypatch.setattr(SislParser, "__init__", forbidden)
    monkeypatch.setattr(scipy.linalg, "lu_factor", forbidden)
    with pytest.raises(UnsupportedSpinMode):
        prepare_exchange_siesta_wfsx(
            model.eigenstate_source, reduce_scalar=True, **exchange_params(tmp_path)
        )


@pytest.mark.parametrize("declaration", [None, False])
def test_scalar_reduction_requires_every_symmetry_declaration(
    tmp_path, monkeypatch, declaration
):
    import scipy.linalg
    from HamiltonIO.siesta.wfsx import MissingMetadata, SiestaWFSXParser

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")
    if declaration is None:
        metadata["scalar_symmetry"].pop("no_soc")
    else:
        metadata["scalar_symmetry"]["no_soc"] = declaration
    parser = SiestaWFSXParser(model.eigenstate_source.wfsx_path, metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("unproven reduction reached coefficient factorization")

    monkeypatch.setattr(scipy.linalg, "lu_factor", forbidden)
    with pytest.raises(MissingMetadata):
        prepare_exchange_siesta_wfsx(
            parser, reduce_scalar=True, **exchange_params(tmp_path)
        )


@pytest.mark.parametrize("kind", ["weights", "missing", "ambiguous"])
def test_requested_mesh_errors_propagate_without_normalization(tmp_path, kind):
    from HamiltonIO.siesta.wfsx import AmbiguousKPoint, InvalidKWeights, MissingKPoint

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, _, points = source_model(tmp_path, "collinear_nspin2_full")
    points = points.copy()
    weights = np.full(5, 0.2)
    expected = InvalidKWeights
    if kind == "weights":
        weights[:] = 0.1
    elif kind == "missing":
        points[0, 2] = 0.13
        expected = MissingKPoint
    else:
        points[1] = points[0]
        expected = AmbiguousKPoint
    with pytest.raises(expected):
        prepare_exchange_siesta_wfsx(
            model.eigenstate_source,
            kpts=points,
            kweights=weights,
            **exchange_params(tmp_path),
        )


def test_failed_second_channel_removes_first_channel_owned_cache(tmp_path):
    from HamiltonIO.siesta.wfsx import SingularEigenbasis
    from test_siesta_wfsx_green import rewrite_test_records

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, manifest, _ = source_model(tmp_path, "collinear_nspin2_full")
    n = manifest["states"]["no_u"]
    stride = 3 + 3 * n

    def singular_down_channel(records):
        for band in range(n):
            records[4 + stride + 5 + 3 * band].view(np.complex64)[0] = 0

    rewrite_test_records(model.eigenstate_source.wfsx_path, singular_down_channel)
    original = model.eigenstate_source.wfsx_path.read_bytes()
    parent = tmp_path / "cache"
    parent.mkdir()
    marker = parent / "user-file"
    marker.write_text("keep")
    with pytest.raises(SingularEigenbasis):
        prepare_exchange_siesta_wfsx(
            model.eigenstate_source, cache_path=parent, **exchange_params(tmp_path)
        )
    assert list(parent.iterdir()) == [marker]
    assert marker.read_text() == "keep"
    assert model.eigenstate_source.wfsx_path.read_bytes() == original


@pytest.mark.parametrize("mode", ["default", "contravariant"])
def test_file_request_permutations_and_wrapping_preserve_exchange(
    tmp_path, monkeypatch, mode
):
    from test_siesta_wfsx_green import permute_k_records

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    monkeypatch.setenv("TB2J_SIESTA_GREEN_MODE", mode)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first, _, _, points = source_model(first_dir, "collinear_nspin2_full")
    second, _, manifest, _ = source_model(second_dir, "collinear_nspin2_full")
    permute_k_records(second.eigenstate_source.wfsx_path, manifest, [3, 0, 4, 2, 1])
    order = np.array([2, 4, 0, 3, 1])
    params = exchange_params(tmp_path)
    a = prepare_exchange_siesta_wfsx(
        first.eigenstate_source,
        reduce_scalar=True,
        kpts=points,
        kweights=np.full(5, 0.2),
        **params,
    )
    b = prepare_exchange_siesta_wfsx(
        second.eigenstate_source,
        reduce_scalar=True,
        kpts=points[order] + [2, -1, 3],
        kweights=np.full(5, 0.2),
        **params,
    )
    try:
        for ga, gb in [(a.Gup, b.Gup), (a.Gdn, b.Gdn)]:
            for z in [-5 + 1j, 0.3 + 2j]:
                np.testing.assert_allclose(
                    gb.get_Gk_all(z), ga.get_Gk_all(z)[order], atol=2e-5, rtol=1e-4
                )
                np.testing.assert_allclose(
                    gb.get_GR([[0, 0, 1], [0, 0, -1]], z),
                    ga.get_GR([[0, 0, 1], [0, 0, -1]], z),
                    atol=2e-5,
                    rtol=1e-4,
                )
            np.testing.assert_allclose(ga.H0, gb.H0, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(
                ga.get_density_matrix(), gb.get_density_matrix(), atol=1e-4, rtol=1e-5
            )
        a.calculate_all()
        b.calculate_all()
        assert_jdict_equal(a.exchange_Jdict, b.exchange_Jdict)
    finally:
        a.finalize()
        b.finalize()


@pytest.mark.parametrize("indices", [[2], [3, 1, 4]])
def test_gamma_and_nonclosed_shifted_exchange_keep_explicit_quadrature(
    tmp_path, indices
):
    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, _, full_points = source_model(tmp_path, "collinear_nspin2_full")
    points = full_points[indices] + [1, -2, 3]
    weights = np.full(len(indices), 1 / len(indices))
    params = {**exchange_params(tmp_path), "kpts": points, "kweights": weights}
    oracle = ordinary_exchange("collinear_nspin2_full", **params)
    exchange = prepare_exchange_siesta_wfsx(
        model.eigenstate_source, reduce_scalar=True, **params
    )
    try:
        for ga, ge in [(exchange.Gup, oracle.Gup), (exchange.Gdn, oracle.Gdn)]:
            np.testing.assert_array_equal(ga.kpts, points)
            np.testing.assert_array_equal(ga.kweights, weights)
            assert ga.spectral_stats["representative_kpoints"] == len(indices)
            np.testing.assert_allclose(ga.H0, ge.H0, atol=2e-5, rtol=1e-5)
            for z in [-5 + 1j, 0.3 + 2j]:
                np.testing.assert_allclose(
                    ga.get_Gk_all(z), ge.get_Gk_all(z), atol=2e-5, rtol=1e-4
                )
                np.testing.assert_allclose(
                    ga.get_GR([[0, 0, 1], [0, 0, -1]], z),
                    ge.get_GR([[0, 0, 1], [0, 0, -1]], z),
                    atol=2e-5,
                    rtol=1e-4,
                )
        oracle.calculate_all()
        exchange.calculate_all()
        assert_jdict_equal(exchange.exchange_Jdict, oracle.exchange_Jdict)
        assert exchange.Gup.spectral_stats["scalar_partner_expansions"] == 0
        assert exchange.Gdn.spectral_stats["scalar_partner_expansions"] == 0
    finally:
        exchange.finalize()


def test_opposite_channel_state_phases_do_not_change_reduction_or_exchange(tmp_path):
    from test_siesta_wfsx_green import rewrite_test_records

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first, _, _, _ = source_model(first_dir, "collinear_nspin2_full")
    second, _, manifest, _ = source_model(second_dir, "collinear_nspin2_full")
    n = manifest["states"]["no_u"]
    stride = 3 + 3 * n

    def change_down_channel_phases(records):
        for ik in range(5):
            start = 4 + (2 * ik + 1) * stride
            for band in range(n):
                records[start + 5 + 3 * band].view(np.complex64)[:] *= -1

    rewrite_test_records(second.eigenstate_source.wfsx_path, change_down_channel_phases)
    params = exchange_params(tmp_path)
    a = prepare_exchange_siesta_wfsx(
        first.eigenstate_source, reduce_scalar=True, **params
    )
    b = prepare_exchange_siesta_wfsx(
        second.eigenstate_source, reduce_scalar=True, **params
    )
    try:
        for ga, gb in [(a.Gup, b.Gup), (a.Gdn, b.Gdn)]:
            assert (
                ga.spectral_stats["factorizations"]
                == gb.spectral_stats["factorizations"]
                == 3
            )
            np.testing.assert_allclose(ga.H0, gb.H0, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(
                ga.get_Gk_all(-5 + 1j), gb.get_Gk_all(-5 + 1j), atol=2e-5, rtol=1e-4
            )
        a.calculate_all()
        b.calculate_all()
        assert_jdict_equal(a.exchange_Jdict, b.exchange_Jdict)
    finally:
        a.finalize()
        b.finalize()


@pytest.mark.parametrize("capability", ["self-energy", "split-soc"])
def test_plain_spectral_green_refuses_unavailable_model_operators(tmp_path, capability):
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.green import TBGreen

    model, _, _, points = source_model(tmp_path, "collinear_nspin2_full")
    if capability == "self-energy":
        model.get_sigma = lambda *args, **kwargs: np.eye(18)
    else:
        model.split_soc = True
    with pytest.raises(UnsupportedOperatorMode):
        TBGreen(
            model, kpts=points, kweights=np.full(5, 0.2), cache_path=tmp_path / "cache"
        )


def test_prepared_spinor_exchange_refuses_mae_method_before_integration(tmp_path):
    from HamiltonIO.siesta.wfsx import UnsupportedOperatorMode

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, _, _, _ = source_model(tmp_path, "noncollinear_nspin4_full")
    exchange = prepare_exchange_siesta_wfsx(
        model.eigenstate_source, **exchange_params(tmp_path)
    )
    try:
        with pytest.raises(UnsupportedOperatorMode):
            exchange.get_MAE([0], [0])
    finally:
        exchange.finalize()


def test_zero_temperature_cfr_requires_explicit_compatible_contour(
    tmp_path, monkeypatch
):
    import scipy.linalg
    from HamiltonIO.siesta.wfsx import SiestaWFSXParser, UnsupportedOperatorMode

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")
    metadata["electronic_temperature_k"] = 0
    source = SiestaWFSXParser(model.eigenstate_source.wfsx_path, metadata)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "zero-temperature CFR reached factorization instead of named refusal"
        )

    monkeypatch.setattr(scipy.linalg, "lu_factor", forbidden)
    params = exchange_params(tmp_path)
    params.pop("smearing")
    with pytest.raises(UnsupportedOperatorMode):
        prepare_exchange_siesta_wfsx(source, **params)


def test_zero_temperature_half_occupied_pole_has_finite_population(tmp_path):
    from HamiltonIO.siesta.wfsx import SiestaWFSXParser
    from test_siesta_wfsx_green import rewrite_test_records

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    model, metadata, _, _ = source_model(tmp_path, "collinear_nspin2_full")
    metadata["electronic_temperature_k"] = 0
    # Controlled limit of the complete pencil: move one occupied up-spin pole
    # on one of five equal-weight k points to Ef, so it becomes half occupied.
    metadata["electron_count"] = 11.9

    def put_first_pole_at_fermi(records):
        records[8].view(np.float64)[:] = metadata["fermi_energy_ev"]

    rewrite_test_records(model.eigenstate_source.wfsx_path, put_first_pole_at_fermi)
    source = SiestaWFSXParser(model.eigenstate_source.wfsx_path, metadata)
    params = exchange_params(tmp_path)
    params.pop("smearing")
    params["method"] = "semicircle"
    exchange = prepare_exchange_siesta_wfsx(source, **params)
    try:
        np.testing.assert_allclose(
            np.trace(exchange.rho_up + exchange.rho_dn), 11.9, atol=1e-4, rtol=1e-5
        )
        assert np.isfinite(exchange.Gup.get_rho_R([[0, 0, 1]])).all()
        assert np.isfinite(exchange.Gdn.get_rho_R([[0, 0, -1]])).all()
    finally:
        exchange.finalize()


@pytest.mark.parametrize("name", ["collinear_nspin2_full", "noncollinear_nspin4_full"])
@pytest.mark.parametrize("cached", [False, True])
def test_ordinary_exchange_finalization_preserves_user_cache_parent(
    tmp_path, name, cached
):
    parent = tmp_path / "user-cache"
    parent.mkdir()
    marker = parent / "unrelated.txt"
    marker.write_text("preserve user bytes")
    exchange = ordinary_exchange(
        name,
        cache_path=parent,
        use_cache=cached,
        **exchange_params(tmp_path / "output"),
    )
    greens = [exchange.G] if hasattr(exchange, "G") else [exchange.Gup, exchange.Gdn]
    owned = [green.cache_path for green in greens] if cached else []
    try:
        np.testing.assert_allclose(
            np.trace(
                exchange.rho
                if hasattr(exchange, "G")
                else exchange.rho_up + exchange.rho_dn
            ),
            12,
            atol=1e-4,
            rtol=1e-5,
        )
    finally:
        exchange.finalize()
    assert marker.read_text() == "preserve user bytes"
    assert list(parent.iterdir()) == [marker]
    assert all(not Path(path).exists() for path in owned)
