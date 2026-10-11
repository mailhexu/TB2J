"""Spinor projector data-model and Green-function validation."""

import numpy as np
import pytest
from utils.projector_filters import expected_channel_filter as _expected_channel_filter

from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
    spinor_tangent_trace,
)

RNG = np.random.default_rng(42)


def _make_spinor_data(nkpt=3, nband=5, nproj_per_site=2, nsite=2, hermitian=True):
    nproj = nproj_per_site * nsite
    kpoints = RNG.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = RNG.normal(size=(1, nkpt, nband))
    coefficients = RNG.normal(size=(1, nkpt, nband, 2, nproj)) + 1j * RNG.normal(
        size=(1, nkpt, nband, 2, nproj)
    )
    projector_site = np.repeat(np.arange(nsite), nproj_per_site)
    site_nproj = np.full(nsite, nproj_per_site)
    site_projector_indices = np.arange(nproj).reshape(nsite, nproj_per_site)
    # Hermitian site operators: (nsite, nproj_max, nproj_max, 2, 2)
    spinor_operator = RNG.normal(size=(nsite, nproj_per_site, nproj_per_site, 2, 2)) + (
        1j * RNG.normal(size=(nsite, nproj_per_site, nproj_per_site, 2, 2))
    )
    if hermitian:
        spinor_operator = (
            spinor_operator + spinor_operator.transpose(0, 2, 1, 4, 3).conj()
        ) / 2.0
    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.3,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        nspinor=2,
    )
    return data, spinor_operator


def test_spinor_data_model_validation():
    data, operators = _make_spinor_data(hermitian=False)
    assert data.nspinor == 2
    assert data.nproj == 4
    assert data.validate(exchange_ready=True)

    # wrong spinor coefficient shape
    bad = _make_spinor_data()[0]
    bad.coefficients = bad.coefficients[:, :, :, 0, :]
    with pytest.raises(ValueError, match="spinor coefficients"):
        bad.validate()

    # missing definition
    bad2, ops2 = _make_spinor_data()
    bad2.spinor_operator_definition = None
    with pytest.raises(ValueError, match="spinor_operator_definition"):
        bad2.validate()

    # bad operator shape
    bad3, ops3 = _make_spinor_data()
    bad3.spinor_operator = ops3[:, :1, :, :, :]
    with pytest.raises(ValueError, match="spinor_operator"):
        bad3.validate()

    # invalid nspinor
    with pytest.raises(ValueError, match="nspinor"):
        _make_spinor_data()[0].__class__(
            **{**_make_spinor_data()[0].__dict__, "nspinor": 3}
        )


def test_spinor_green_requires_spinor_data():
    data, _ = _make_spinor_data()
    data.nspinor = 1
    data.coefficients = data.coefficients[:, :, :, 0, :]
    data.spinor_operator = None
    green = ProjectorGreen(data)
    with pytest.raises(ValueError, match="nspinor=2"):
        green.get_Gk_spinor(0, 0.1)


def _spinor_overlap_k_data(mode_seed=105):
    """Nonorthogonal spinor data: k-dependent complex Hermitian overlap."""
    data, _ = _make_spinor_data()
    rng = np.random.default_rng(mode_seed)
    A = rng.normal(size=(data.nkpt, data.nproj, data.nproj)) + 1j * rng.normal(
        size=(data.nkpt, data.nproj, data.nproj)
    )
    data.overlap_k = A @ A.conj().swapaxes(-1, -2) + data.nproj * np.eye(data.nproj)
    return data


@pytest.mark.parametrize("mode", ["inverse", "svd", "lowdin", "tikhonov", "plain"])
def test_spinor_green_all_modes_match_reference_construction(mode):
    data = _spinor_overlap_k_data()
    rcond = 1.0e-3
    green = ProjectorGreen(data, overlap_mode=mode, overlap_rcond=rcond)
    energy = 0.25 + 0.4j

    evals = data.eigenvalues[0]
    coeff = data.coefficients[0]  # (nkpt, nband, 2, nproj)
    inv_denom = 1.0 / (energy + data.efermi - evals)
    covariant = np.einsum("knsp,kntq,kn->kpqst", coeff, coeff.conj(), inv_denom)
    if mode == "plain":
        expected = covariant
    else:
        filters = np.array(
            [_expected_channel_filter(S, mode, rcond) for S in data.overlap_k]
        )
        expected = np.einsum("kpa,kabst,kbq->kpqst", filters, covariant, filters)
    np.testing.assert_allclose(
        green.get_Gk_all_spinor(energy), expected, rtol=2e-12, atol=1e-12
    )
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    phase = (
        np.exp(green.k2Rfactor * np.einsum("ri,ki->rk", R, green.kpts))
        * green.kweights[None, :]
    )
    expected_r = np.einsum("kpqst,rk->rpqst", expected, phase, optimize="optimal")
    np.testing.assert_allclose(
        green.get_GR_spinor(R, energy),
        expected_r,
        rtol=2e-12,
        atol=1e-12,
    )


def test_spinor_green_cache_rebuilds_after_inplace_overlap_edit():
    data = _spinor_overlap_k_data()
    reference = ProjectorGreen(data, overlap_mode="inverse")
    green = ProjectorGreen(data, overlap_mode="inverse")
    energy = -0.8 + 0.3j
    green.get_Gk_all_spinor(energy)

    data.overlap_k[1] = data.overlap_k[1] * 1.4
    # Consumer-visible contract: results keep matching a live uncached backend.
    np.testing.assert_allclose(
        green.get_Gk_all_spinor(energy),
        reference.get_Gk_all_spinor(energy),
        rtol=2e-12,
        atol=1e-12,
    )


def test_spinor_tangent_trace_matches_uncached_backend_after_cutover():
    data = _spinor_overlap_k_data()
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    energy = -0.5 + 0.3j
    ref = ProjectorGreen(data)  # default inverse mode, cache active by design
    cached = ProjectorGreen(data, overlap_mode="inverse")
    cached.get_GR_spinor(R, energy)  # warm the cache
    a = spinor_tangent_trace(ref, R, energy)["K_ijR"]
    b = spinor_tangent_trace(cached, R, energy)["K_ijR"]
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=2e-12, atol=1e-12)
