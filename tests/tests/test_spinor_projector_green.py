"""Spinor projector-Green extension tests (story 002, soc-spinor spec).

Covers the sympy-pinned kernel (docs/sympy/spinor_projector_green.md):
random-data equivalence against a brute-force implementation, the
collinear embedded reduction, and the nspinor=2 data-model validation.
"""

import numpy as np
import pytest

from TB2J.projector_green import (
    PAULI_MATRICES,
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
    spinor_projector_exchange_trace,
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


def _brute_force_tensor(green, Rpts, energy, operators):
    """Independent O(n^4) reference: loops over spins and projectors."""
    from TB2J.projector_green import _spinor_dense_block

    GR = green.get_GR_spinor(Rpts, energy)
    out = {}
    for iR, R in enumerate(map(tuple, Rpts)):
        iRm = [i for i, r in enumerate(map(tuple, Rpts)) if tuple(-x for x in r) == R][
            0
        ]
        for iatom in range(2):
            for jatom in range(2):
                Gij = _spinor_dense_block(
                    green.get_site_block_spinor(GR[iR], iatom, jatom)
                )
                Gji = _spinor_dense_block(
                    green.get_site_block_spinor(GR[iRm], jatom, iatom)
                )
                Di = _spinor_dense_block(operators[iatom])
                Dj = _spinor_dense_block(operators[jatom])
                J = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        value = 0.0 + 0.0j
                        Oi = np.kron(PAULI_MATRICES[a], np.eye(Di.shape[0] // 2)) @ Di
                        Oj = np.kron(PAULI_MATRICES[b], np.eye(Dj.shape[0] // 2)) @ Dj
                        value = -np.trace(Oi @ Gij @ Oj @ Gji)
                        J[a, b] = value.real / (4.0 * np.pi)
                out[(R, iatom, jatom)] = J
    return out


def test_random_spinor_tensor_matches_reference():
    data, operators = _make_spinor_data()
    green = ProjectorGreen(data)
    Rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    result = spinor_projector_exchange_trace(green, Rpts, energy=0.1)
    ref = _brute_force_tensor(green, Rpts, 0.1, {0: operators[0], 1: operators[1]})
    for key, J in ref.items():
        np.testing.assert_allclose(result["tensor"][key], J, atol=1e-12)
        Jiso, D, Jani = result["decomposition"][key]
        from TB2J.Jtensor import combine_J_tensor

        recon = combine_J_tensor(Jiso=Jiso, D=D, Jani=Jani)
        np.testing.assert_allclose(recon, J, atol=1e-12)


def test_collinear_embedded_reduction():
    """Spin-conserving data with Delta = delta sigma_z: isotropic cross-channel."""
    nkpt, nband, nproj = 4, 6, 4
    kpoints = RNG.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = RNG.normal(size=(1, nkpt, nband))
    coeff = RNG.normal(size=(1, nkpt, nband, 2, nproj)) + 1j * RNG.normal(
        size=(1, nkpt, nband, 2, nproj)
    )
    # spin-conserving: each band carries one spinor component only
    band_spin = RNG.integers(0, 2, size=nband)
    for n, s in enumerate(band_spin):
        coeff[0, :, n, 1 - s, :] = 0.0
    projector_site = np.repeat([0, 1], 2)
    site_nproj = np.array([2, 2])
    site_projector_indices = np.arange(4).reshape(2, 2)
    delta_i, delta_j = 0.8, 1.3
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    spinor_operator = np.zeros((2, 2, 2, 2, 2), dtype=complex)
    spinor_operator[0] = np.einsum("st,pq->pqst", sz, np.eye(2)) * delta_i
    spinor_operator[1] = np.einsum("st,pq->pqst", sz, np.eye(2)) * delta_j
    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coeff,
        efermi=0.2,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        nspinor=2,
    )
    green = ProjectorGreen(data)
    Rpts = np.array([[0, 0, 0], [1, 1, 0], [-1, -1, 0]], dtype=int)
    result = spinor_projector_exchange_trace(green, Rpts, energy=0.05)

    # collinear-channel reference via the standard collinear machinery
    collinear = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=np.stack([eigenvalues[0], eigenvalues[0]]),
        coefficients=np.stack([coeff[0, :, :, 0, :], coeff[0, :, :, 1, :]]),
        efermi=0.2,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
    )
    cg = ProjectorGreen(collinear)
    energy = 0.05
    Gup = cg.get_GR(Rpts, energy, ispin=0)
    Gdn = cg.get_GR(Rpts, energy, ispin=1)

    for key, Jtens in result["tensor"].items():
        R, iatom, jatom = key
        iR = [i for i, r in enumerate(map(tuple, Rpts)) if tuple(r) == tuple(R)][0]
        iRm = [
            i for i, r in enumerate(map(tuple, Rpts)) if tuple(r) == tuple(-np.array(R))
        ][0]
        # full sympy-pinned closed forms (docs/sympy/spinor_projector_green.md)
        pref = (delta_i if iatom == 0 else delta_j) * (
            delta_j if jatom == 1 else delta_i
        )
        Gu = cg.get_site_block(Gup[iR], iatom, jatom)
        Gd = cg.get_site_block(Gdn[iR], iatom, jatom)
        Hu = cg.get_site_block(Gup[iRm], jatom, iatom)
        Hd = cg.get_site_block(Gdn[iRm], jatom, iatom)
        cross = pref * (np.trace(Gu @ Hd) + np.trace(Gd @ Hu))
        same = pref * (np.trace(Gu @ Hu) + np.trace(Gd @ Hd))
        xy = (1j * pref * (np.trace(Gd @ Hu) - np.trace(Gu @ Hd))).real
        assert Jtens[0, 0] == pytest.approx(cross.real / (4.0 * np.pi), abs=1e-10)
        assert Jtens[1, 1] == pytest.approx(cross.real / (4.0 * np.pi), abs=1e-10)
        assert Jtens[2, 2] == pytest.approx(-same.real / (4.0 * np.pi), abs=1e-10)
        assert Jtens[0, 1] == pytest.approx(xy / (4.0 * np.pi), abs=1e-10)
        assert Jtens[1, 0] == pytest.approx(-xy / (4.0 * np.pi), abs=1e-10)
        assert Jtens[0, 2] == pytest.approx(0.0, abs=1e-10)
        assert Jtens[2, 0] == pytest.approx(0.0, abs=1e-10)
        assert Jtens[1, 2] == pytest.approx(0.0, abs=1e-10)
        assert Jtens[2, 1] == pytest.approx(0.0, abs=1e-10)
        # J^{zz} carries the same-channel piece at single-energy level; the
        # contour prescription removes it in the physical exchange (see
        # docs/sympy/spinor_projector_green.md).
        Jiso, D, Jani = result["decomposition"][key]
        assert Jiso == pytest.approx(np.trace(Jtens) / 3, abs=1e-12)


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
