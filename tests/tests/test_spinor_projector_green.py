"""Spinor projector data-model and Green-function validation."""

import numpy as np
import pytest

from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
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
