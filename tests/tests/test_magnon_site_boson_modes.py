"""Consumer-facing tests for canonical physical site-boson modes."""

import numpy as np
import pytest

from TB2J.magnon.magnon3 import Magnon


def _magnon(nspin, spins, bonds):
    """Build isotropic exchange models; bonds are (R, i, j, J)."""
    Rlist = np.asarray([bond[0] for bond in bonds], dtype=int)
    JR = np.zeros((len(bonds), nspin, nspin, 3, 3), dtype=float)
    for ir, (_r, i, j, value) in enumerate(bonds):
        JR[ir, i, j] = np.eye(3) * value * spins[i] * spins[j]
    mags = np.zeros((nspin, 3))
    mags[:, 2] = 2.0 * np.asarray(spins)
    mags[1::2, 2] *= -1.0
    magnon = Magnon(
        nspin=nspin,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


def _fm_chain():
    # Isotropic ferromagnetic nearest-neighbour chain, including both directed bonds.
    return _magnon(
        1,
        [1.0],
        [((0, 0, 0), 0, 0, 1.0), ((1, 0, 0), 0, 0, 1.0), ((-1, 0, 0), 0, 0, 1.0)],
    )


def _two_sublattice_afm():
    # Collinear two-sublattice AFM, nearest-neighbour A-B bonds and their reverses.
    return _magnon(
        2,
        [1.0, 1.0],
        [
            ((0, 0, 0), 0, 1, -1.0),
            ((0, 0, 0), 1, 0, -1.0),
            ((1, 0, 0), 0, 1, -0.6),
            ((-1, 0, 0), 1, 0, -0.6),
        ],
    )


def _engineered_anomalous():
    magmom = np.array([[0.0, 0.0, 2.0]])
    JR = np.zeros((1, 1, 1, 3, 3))
    JR[0, 0, 0] = np.diag([-2.0, -1.0, 0.0])
    magnon = Magnon(
        nspin=1,
        magmom=magmom,
        Rlist=np.zeros((1, 3), dtype=int),
        JR=JR,
        cell=np.eye(3),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


def _direct_positive_energies(magnon, kpoints):
    """Independent generalized eigenvalues of g H, not Cholesky energies."""
    result = []
    for H in magnon.Hq(kpoints):
        n = H.shape[0] // 2
        g = np.diag(np.r_[np.ones(n), -np.ones(n)])
        eig = np.linalg.eigvals(g @ H)
        positive = np.sort(eig.real[eig.real > 1e-9])
        assert len(positive) == n
        result.append(positive)
    return np.asarray(result)


def test_site_boson_modes_match_direct_models_and_existing_energies():
    models = (_fm_chain(), _two_sublattice_afm())
    kpoints = np.array([[0.11, 0, 0], [0.23, 0, 0]])
    for magnon in models:
        modes = magnon.get_site_boson_modes(kpoints)
        np.testing.assert_allclose(
            modes.energies, magnon._magnon_energies(kpoints), atol=1e-12
        )
        np.testing.assert_allclose(
            modes.energies, _direct_positive_energies(magnon, kpoints), atol=1e-9
        )
        assert modes.u.shape == modes.v.shape == (2, magnon.nspin, magnon.nspin)
        assert modes.component_order == ("particle", "hole")
        assert (
            modes.reciprocal_convention
            == "fractional_reciprocal; J(q)=sum_R J(R) exp(-2*pi*i*q.R)"
        )
        assert modes.diagnostics.min_eig.shape == (2,)
        assert modes.diagnostics.cholesky_tier.shape == (2,)
        assert modes.diagnostics.shift.shape == (2,)
        assert np.all(modes.diagnostics.cholesky_tier == 0)
        np.testing.assert_allclose(modes.spin_lengths, magnon.Snorm)
        assert modes.U.shape == modes.V.shape == (magnon.nspin, 3)
        assert modes.amplitude_convention.startswith("z(k)=T(k)")
        for ik in range(len(kpoints)):
            T = modes.paraunitary[ik]
            n = magnon.nspin
            g = np.diag(np.r_[np.ones(n), -np.ones(n)])
            np.testing.assert_allclose(T.conj().T @ g @ T, g, atol=1e-9)


def test_anomalous_mode_normalization_and_rephasing_observables():
    magnon = _two_sublattice_afm()
    modes = magnon.get_site_boson_modes([[0.17, 0, 0]])
    assert np.max(np.abs(modes.v)) > 1e-8
    norms = np.sum(np.abs(modes.u) ** 2, axis=-1) - np.sum(
        np.abs(modes.v) ** 2, axis=-1
    )
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)
    phase = np.exp(1j * np.array([0.2, -0.7]))
    ur = modes.u * phase[None, :, None]
    vr = modes.v * phase[None, :, None]
    np.testing.assert_allclose(np.abs(ur) ** 2, np.abs(modes.u) ** 2)
    np.testing.assert_allclose(np.abs(vr) ** 2, np.abs(modes.v) ** 2)
    np.testing.assert_allclose(
        np.sum(np.abs(ur) ** 2, axis=-1) - np.sum(np.abs(vr) ** 2, axis=-1), norms
    )


def test_gamma_zero_mode_limit_and_diagnostics():
    magnon = _fm_chain()
    modes = magnon.get_site_boson_modes([[0.0, 0, 0]])
    np.testing.assert_allclose(
        modes.energies, magnon._magnon_energies([[0.0, 0, 0]]), atol=0, rtol=0
    )
    assert modes.diagnostics.cholesky_tier[0] in (0, 1)
    assert modes.diagnostics.shift[0] >= 0
    assert np.isfinite(modes.u).all() and np.isfinite(modes.v).all()
    n = magnon.nspin
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    np.testing.assert_allclose(
        modes.paraunitary[0].conj().T @ g @ modes.paraunitary[0],
        g,
        atol=1e-9,
    )
    assert modes.is_zero_mode[0, 0]


def test_unstable_reference_is_rejected():
    magnon = _fm_chain()
    magnon.JR *= -1
    with pytest.raises(ValueError, match="unstable"):
        magnon.get_site_boson_modes([[0.2, 0, 0]])


def test_mode_result_arrays_are_read_only_and_cannot_corrupt_paraunitary():
    modes = _two_sublattice_afm().get_site_boson_modes([[0.17, 0, 0]])
    before = modes.paraunitary.copy()
    for amplitudes in (modes.u, modes.v):
        with pytest.raises(ValueError):
            amplitudes[0, 0, 0] = amplitudes[0, 0, 0] + 1.0
    with pytest.raises(ValueError):
        modes.paraunitary[0, 0, 0] += 1.0
    with pytest.raises(ValueError):
        modes.diagnostics.min_eig[0] = 0.0
    np.testing.assert_array_equal(modes.paraunitary, before)


def test_engineered_anomalous_model_against_direct_diagonalization():
    magnon = _engineered_anomalous()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.21, 0.0, 0.0]])
    modes = magnon.get_site_boson_modes(kpoints)
    np.testing.assert_allclose(
        modes.energies, _direct_positive_energies(magnon, kpoints)
    )
    np.testing.assert_allclose(
        modes.energies, magnon._magnon_energies(kpoints), atol=0, rtol=0
    )
    assert np.max(np.abs(modes.v)) > 0
    np.testing.assert_allclose(
        np.sum(np.abs(modes.u) ** 2, axis=-1) - np.sum(np.abs(modes.v) ** 2, axis=-1),
        1.0,
        atol=1e-10,
    )
