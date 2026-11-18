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


def _nonreciprocal_chain():
    spin = 1.0
    J, D = 1.0, 0.2
    antisymmetric = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    positive_R = J * np.eye(3) + D * antisymmetric
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = positive_R * spin**2
    JR[1, 0, 0] = positive_R.T * spin**2
    magnon = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0.0, 2.0 * spin]]),
        Rlist=np.array([[1, 0, 0], [-1, 0, 0]]),
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


def _degenerate_anomalous_pair():
    magmoms = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    JR = np.zeros((1, 2, 2, 3, 3))
    JR[0, 0, 0] = np.diag([-2.0, -1.0, 0.0])
    JR[0, 1, 1] = np.diag([-2.0, -1.0, 0.0])
    magnon = Magnon(
        nspin=2,
        magmom=magmoms,
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


def _spin_transition_amplitudes(modes, u=None, v=None):
    """One-magnon local spin matrix elements for the declared HP convention."""
    u = modes.u[0].T if u is None else u[0].T
    v = modes.v[0].T if v is None else v[0].T
    amplitudes = np.sqrt(modes.spin_lengths / 2.0)[:, None, None] * (
        modes.U[:, :, None] * u[:, None, :] + modes.U.conj()[:, :, None] * v[:, None, :]
    )
    return amplitudes.reshape(-1, amplitudes.shape[-1])


def _fm_chain_8():
    # Periodic eight-site FM chain in one cell: zero-acoustic plus seven
    # gapped branches (omega = 4 J S [1 - cos(2 pi m / 8)] >= 1.17 eV).
    nspin = 8
    JR = np.zeros((1, nspin, nspin, 3, 3))
    for i in range(nspin):
        j = (i + 1) % nspin
        JR[0, i, j] = np.eye(3)
        JR[0, j, i] = np.eye(3)
    magmom = np.zeros((nspin, 3))
    magmom[:, 2] = 2.0
    magnon = Magnon(
        nspin=nspin,
        magmom=magmom,
        Rlist=np.array([[0, 0, 0]], dtype=int),
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


def _fm_chain_9():
    # Periodic nine-site FM chain: smallest gapped omega = 4[1-cos(2pi/9)]
    # ~ 0.936 eV under a ~7.76 eV Hessian scale.
    nspin = 9
    JR = np.zeros((1, nspin, nspin, 3, 3))
    for i in range(nspin):
        j = (i + 1) % nspin
        JR[0, i, j] = np.eye(3)
        JR[0, j, i] = np.eye(3)
    magmom = np.zeros((nspin, 3))
    magmom[:, 2] = 2.0
    magnon = Magnon(
        nspin=nspin,
        magmom=magmom,
        Rlist=np.array([[0, 0, 0]], dtype=int),
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
    residue = (
        _spin_transition_amplitudes(modes) @ _spin_transition_amplitudes(modes).conj().T
    )
    rephased_transition = _spin_transition_amplitudes(modes, ur, vr)
    np.testing.assert_allclose(
        rephased_transition @ rephased_transition.conj().T, residue, atol=1e-10
    )


def test_gamma_zero_mode_limit_and_diagnostics():
    magnon = _fm_chain()
    modes = magnon.get_site_boson_modes([[0.0, 0, 0]])
    assert modes.energies[0, 0] == 0.0
    assert modes.diagnostics.symmetry_residual[0] <= modes.diagnostics.stability_tol[0]
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
    assert int(modes.is_zero_mode[0].sum()) == 1
    small_k = magnon.get_site_boson_modes([[1e-4, 0, 0]])
    assert small_k.energies[0, 0] > 0.0
    assert not small_k.is_zero_mode[0, 0]
    np.testing.assert_allclose(
        small_k.energies, _direct_positive_energies(magnon, [[1e-4, 0, 0]]), rtol=1e-6
    )


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


def test_eight_site_fm_gamma_keeps_single_zero_branch():
    magnon = _fm_chain_8()
    modes = magnon.get_site_boson_modes([[0.0, 0, 0]])
    legacy = magnon._magnon_energies([[0.0, 0, 0]])
    assert int(modes.is_zero_mode[0].sum()) == 1
    assert modes.energies[0, 0] == 0.0
    assert np.all(modes.energies[0, 1:] > 0.1)
    np.testing.assert_array_equal(modes.energies[0, 1:], legacy[0, 1:])
    n = magnon.nspin
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    np.testing.assert_allclose(
        modes.paraunitary[0].conj().T @ g @ modes.paraunitary[0],
        g,
        atol=1e-9,
    )


def test_nine_site_fm_gamma_survives_underflow_sensitive_gaps():
    # Smallest gap 0.936 eV under a 7.76 eV scale: an exponentiating
    # (gH)^dim cross-check would underflow and veto the zeroing here.
    magnon = _fm_chain_9()
    modes = magnon.get_site_boson_modes([[0.0, 0, 0]])
    legacy = magnon._magnon_energies([[0.0, 0, 0]])
    assert int(modes.is_zero_mode[0].sum()) == 1
    assert modes.energies[0, 0] == 0.0
    assert np.all(modes.energies[0, 1:] > 0.9)
    np.testing.assert_array_equal(modes.energies[0, 1:], legacy[0, 1:])
    n = magnon.nspin
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    np.testing.assert_allclose(
        modes.paraunitary[0].conj().T @ g @ modes.paraunitary[0],
        g,
        atol=1e-9,
    )


def test_afm_gamma_zeroes_both_goldstone_branches_with_residuals():
    magnon = _two_sublattice_afm()
    modes = magnon.get_site_boson_modes([[0.0, 0, 0]])
    np.testing.assert_array_equal(modes.energies[0], [0.0, 0.0])
    assert modes.is_zero_mode[0].all()
    assert modes.diagnostics.symmetry_residual[0] <= modes.diagnostics.stability_tol[0]
    n = magnon.nspin
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    np.testing.assert_allclose(
        modes.paraunitary[0].conj().T @ g @ modes.paraunitary[0],
        g,
        atol=1e-9,
    )
    small = magnon.get_site_boson_modes([[1e-3, 0, 0]])
    assert np.all(small.energies[0] > 0.0)
    assert not small.is_zero_mode[0].any()


def test_scale_aware_stability_rejects_negative_hessian_within_legacy_shift():
    magnon = _engineered_anomalous()
    magnon.JR[0, 0, 0, 2, 2] = -1.00000005
    H = magnon.Hq([[0.0, 0.0, 0.0]])[0]
    minimum = np.linalg.eigvalsh(H)[0]
    assert minimum == pytest.approx(-1e-7, abs=1e-13)
    assert np.linalg.cholesky(H + 1e-6 * np.eye(H.shape[0])) is not None
    with pytest.raises(ValueError, match="unstable"):
        magnon.get_site_boson_modes([[0.0, 0.0, 0.0]])


def test_nonreciprocal_particle_hole_partner_uses_opposite_k():
    magnon = _nonreciprocal_chain()
    kpoints = np.array([[0.17, 0, 0], [-0.17, 0, 0]])
    modes = magnon.get_site_boson_modes(kpoints)
    spectra = []
    for kpoint in kpoints:
        H = magnon.Hq([kpoint])[0]
        g = np.diag([1.0, -1.0])
        spectra.append(np.linalg.eigvals(g @ H).real)
    positive_k = spectra[0][spectra[0] > 0]
    negative_k = spectra[0][spectra[0] < 0]
    positive_minus_k = spectra[1][spectra[1] > 0]
    assert not np.isclose(positive_k[0], positive_minus_k[0])
    np.testing.assert_allclose(modes.energies[0], np.sort(positive_k))
    np.testing.assert_allclose(modes.energies[1], np.sort(positive_minus_k))
    np.testing.assert_allclose(positive_minus_k, np.sort(-negative_k))


def test_mode_rephasing_preserves_physical_spin_spectral_residue():
    modes = _two_sublattice_afm().get_site_boson_modes([[0.17, 0, 0]])
    transition = _spin_transition_amplitudes(modes)
    residue = transition @ transition.conj().T
    phases = np.exp(1j * np.array([0.37, -0.61]))
    rephased = transition * phases[None, :]
    np.testing.assert_allclose(rephased @ rephased.conj().T, residue, atol=1e-10)


def test_degenerate_mode_unitary_rotation_preserves_spectral_projector():
    modes = _degenerate_anomalous_pair().get_site_boson_modes([[0.13, 0, 0]])
    np.testing.assert_allclose(modes.energies[0, 0], modes.energies[0, 1], atol=1e-12)
    transition = _spin_transition_amplitudes(modes)
    projector = transition @ transition.conj().T
    angle = 0.41
    phase = np.exp(0.29j)
    unitary = np.array(
        [
            [np.cos(angle), phase * np.sin(angle)],
            [-phase.conjugate() * np.sin(angle), np.cos(angle)],
        ]
    )
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-12)
    rotated = transition @ unitary
    np.testing.assert_allclose(rotated @ rotated.conj().T, projector, atol=1e-10)
