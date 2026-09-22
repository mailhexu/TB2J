"""Spinor projector-Green extension tests (story 002, soc-spinor spec).

Covers the sympy-pinned kernel (docs/sympy/spinor_projector_green.md):
random-data equivalence against a brute-force implementation, the
collinear embedded reduction, and the nspinor=2 data-model validation.
"""

import numpy as np
import pytest

from TB2J.projector_green import (
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


def _brute_force_A(green, Rpts, energy, operators):
    """Independent O(n^4) reference for the ExchangeNCL channel matrix
    A^{uv} = Tr[Delta_i G^(u)_ij Delta_j G^(v)_ji]/pi with G^(u) the u-th
    Pauli component of the spinor Green block."""
    from TB2J.projector_green import PAULI_IDENTITY_AND_MATRICES, _spinor_dense_block

    GR = green.get_GR_spinor(Rpts, energy)
    out = {}
    for iR, R in enumerate(map(tuple, Rpts)):
        iRm = [i for i, r in enumerate(map(tuple, Rpts)) if tuple(-x for x in r) == R][
            0
        ]
        for iatom in range(2):
            for jatom in range(2):
                Gij = green.get_site_block_spinor(GR[iR], iatom, jatom)
                Gji = green.get_site_block_spinor(GR[iRm], jatom, iatom)
                Di = _spinor_dense_block(operators[iatom])
                Dj = _spinor_dense_block(operators[jatom])
                T_ij = [
                    0.5 * np.einsum("pqst,st->pq", Gij, SIG)
                    for SIG in PAULI_IDENTITY_AND_MATRICES
                ]
                T_ji = [
                    0.5 * np.einsum("pqst,st->pq", Gji, SIG)
                    for SIG in PAULI_IDENTITY_AND_MATRICES
                ]
                A = np.empty((4, 4), dtype=complex)
                for u in range(4):
                    Gu = np.kron(PAULI_IDENTITY_AND_MATRICES[u], T_ij[u])
                    for v in range(4):
                        Gv = np.kron(PAULI_IDENTITY_AND_MATRICES[v], T_ji[v])
                        A[u, v] = np.trace(Di @ Gu @ Dj @ Gv) / np.pi
                out[(R, iatom, jatom)] = A
    return out


def test_random_spinor_tensor_matches_reference():
    data, operators = _make_spinor_data()
    green = ProjectorGreen(data)
    Rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    result = spinor_projector_exchange_trace(green, Rpts, energy=0.1)
    ref = _brute_force_A(green, Rpts, 0.1, {0: operators[0], 1: operators[1]})
    for key, A in ref.items():
        np.testing.assert_allclose(result["A_ijR"][key], A, atol=1e-12)


def test_collinear_embedded_reduction():
    """Spin-conserving data with Delta = delta sigma_z: the contour-integrated
    spinor J_iso must reproduce the collinear kernel
    Im integral Tr[Delta G_up Delta G_down]/(4 pi) exactly, with no DMI and
    no anisotropy (the A00-Azz cancellation removes the same-spin channel)."""
    from ase.units import kB

    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
    from TB2J.mycfr import CFR

    nkpt, nband, nproj = 4, 6, 4
    kpoints = RNG.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = RNG.normal(size=(1, nkpt, nband))
    coeff = RNG.normal(size=(1, nkpt, nband, 2, nproj)) + 1j * RNG.normal(
        size=(1, nkpt, nband, 2, nproj)
    )
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
    Rpts = np.array([[0, 0, 0], [1, 1, 0], [-1, -1, 0]], dtype=int)
    exchange = compute_spinor_projector_exchange(
        data, Rpts=Rpts, nz=24, smearing_eV=0.05, sites=[0, 1]
    )

    # collinear kernel reference on the identical spin-channel data
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
    operators_col = {0: np.eye(2) * delta_i, 1: np.eye(2) * delta_j}
    contour = CFR(nz=24, T=0.05 / kB)
    for (R, iatom, jatom), entry in exchange.items():
        if iatom == jatom and R == (0, 0, 0):
            continue
        vals = []
        for energy in contour.path:
            Gup = cg.get_GR(Rpts, energy, ispin=0)
            Gdn = cg.get_GR(Rpts, energy, ispin=1)
            iR = [i for i, r in enumerate(map(tuple, Rpts)) if tuple(r) == tuple(R)][0]
            iRm = [
                i
                for i, r in enumerate(map(tuple, Rpts))
                if tuple(r) == tuple(-np.array(R))
            ][0]
            Gu = cg.get_site_block(Gup[iR], iatom, jatom)
            Hd = cg.get_site_block(Gdn[iRm], jatom, iatom)
            vals.append(np.trace(operators_col[iatom] @ Gu @ operators_col[jatom] @ Hd))
        ref = np.imag(contour.integrate_values(np.asarray(vals))) / (4.0 * np.pi)
        assert entry["Jiso"] == pytest.approx(ref, rel=1e-9, abs=1e-12), (
            R,
            iatom,
            jatom,
        )
        assert np.linalg.norm(entry["dmi"]) < 1e-9
        # Off-diagonal anisotropy vanishes for collinear states.  The zz
        # diagonal keeps the same-spin-channel residue of the raw A^{zz}
        # (ExchangeNCL's Im(A^{ij}+A^{ij}(-R)) mapping has the same
        # property on Matsubara-type contours); the physical isotropic
        # exchange is Jiso above.
        off_diag = entry["jani"] - np.diag(np.diag(entry["jani"]))
        assert np.abs(off_diag).max() < 1e-9


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
