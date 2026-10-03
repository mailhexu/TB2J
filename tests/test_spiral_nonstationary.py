from __future__ import annotations

import numpy as np
import pytest

from TB2J.spiral_nonstationary import (
    NonstationaryCurvatureError,
    frozen_beta_gradient,
    j_fit_report,
    known_j_beta,
    known_j_comparison,
    nonstationary_curvature,
    pair_once_beta,
    pair_once_from_ordered,
    su2_rotation_y,
    ward_report,
)


class RingProvider:
    """Small independent planar-y folded-pencil provider for a one-orbital ring."""

    def __init__(self, q=1 / 5, t=-0.7, B=1.1):
        self.q_frac = np.array([q, 0.0, 0.0])
        self.taus = np.zeros((1, 3))
        self.phis = np.zeros(1)
        self.B_local = np.array([2 * B])
        self.Rlist = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        self.HR = np.array([[[0.13]], [[t]], [[t]]], dtype=complex)
        self.SR = np.array([[[1.0]], [[0.0]], [[0.0]]], dtype=complex)
        self.config = type(
            "Config", (), {"q_frac": self.q_frac, "taus": self.taus, "phis": self.phis}
        )()

    def gen_ham(self, k):
        from TB2J.spiral_nonstationary import su2_rotation_y

        k = np.asarray(k)
        H = np.zeros((2, 2), dtype=complex)
        S = np.zeros((2, 2), dtype=complex)
        _alpha = self.phis[0] + 2 * np.pi * np.dot(self.q_frac, self.taus[0])
        for r, h, s in zip(self.Rlist, self.HR[:, 0, 0], self.SR[:, 0, 0]):
            d = 2 * np.pi * np.dot(self.q_frac, r)  # one orbital: alpha cancels
            phase = np.exp(2j * np.pi * np.dot(k, r))
            H += phase * h * su2_rotation_y(d)
            S += phase * s * su2_rotation_y(d)
        H += 0.5 * self.B_local[0] * np.diag([1.0, -1.0])
        return H, S


def _kmesh(n):
    return np.column_stack((np.arange(n) / n, np.zeros((n, 2))))


def _reference(provider, n=30):
    qn = provider.q_frac[0] * n
    shift = 0.5 if abs(qn - round(qn)) < 1e-10 and int(round(qn)) % 2 else 0.0
    kpts = np.column_stack(((np.arange(n) + shift) / n, np.zeros((n, 2))))
    evals = []
    for k in kpts:
        H, S = provider.gen_ham(k)
        evals.append(np.linalg.eigvalsh(H))
    evals = np.asarray(evals)
    # Two occupied lower states per k, with a sharp gap at zero.
    occ = (evals < 0).astype(float)
    return kpts, np.ones(n) / n, occ


def test_primitive_curvature_matches_commensurate_lab_ring_fd_including_v2():
    provider = RingProvider(q=1 / 5)
    kpts, kweights, occupations = _reference(provider, n=5)
    curv = nonstationary_curvature(
        provider,
        kpts=kpts,
        kweights=kweights,
        occupations=occupations,
        q_frac=provider.q_frac,
        taus=provider.taus,
        phis=provider.phis,
        B_local=provider.B_local,
        translation_cutoff=5,
    )
    oracle = curv.lab_ring_fd_oracle
    assert oracle is not None
    import json

    json.dumps(curv.as_dict())
    assert curv.report["su2_unfold_max_residual"] < 1e-10
    assert curv.v2_diagonal.shape == (1,)
    for key in (("b", "b"), ("b", "d"), ("d", "b"), ("d", "d")):
        assert curv.blocks[key].shape == (5, 5)
        np.testing.assert_allclose(
            curv.blocks[key], oracle.blocks[key], atol=2e-5, rtol=2e-5
        )
    assert np.max(np.abs(curv.report["v2_diagonal"])) > 0


def test_noncommensurate_q_uses_primitive_mesh_and_translation_cutoff():
    provider = RingProvider(q=0.371)
    kpts, kweights, occupations = _reference(provider, n=48)
    curv = nonstationary_curvature(
        provider,
        kpts=kpts,
        kweights=kweights,
        occupations=occupations,
        q_frac=provider.q_frac,
        taus=provider.taus,
        phis=provider.phis,
        B_local=provider.B_local,
        translation_cutoff=3,
    )
    assert curv.report["primitive_only"] is True
    assert curv.report["q_commensurate_with_cutoff"] is False
    assert curv.blocks[("d", "d")].shape == (3, 3)


def test_gradient_corrected_ward_and_fixed_field_inapplicability():
    theta = np.array([0.0, 0.7, 1.8])
    j = {(0, 1): 0.8, (0, 2): -0.3, (1, 2): 0.5}
    # Pair-once Heisenberg Hessian, independently constructed from the model.
    cdd = np.zeros((3, 3))
    cbb = np.zeros((3, 3))
    for (i, k), value in j.items():
        delta = theta[i] - theta[k]
        cdd[i, k] = cdd[k, i] = -value
        cdd[i, i] += value * np.cos(delta)
        cdd[k, k] += value * np.cos(delta)
        cbb[i, k] = cbb[k, i] = -value * np.cos(delta)
        cbb[i, i] += value * np.cos(delta)
        cbb[k, k] += value * np.cos(delta)
    g = known_j_beta(j, theta)
    curv = type("Curv", (), {"blocks": {("b", "b"): cbb, ("d", "d"): cdd}})()
    report = ward_report(curv, g, thetas=theta, symmetry="co_rotating")
    assert report["applicable"] and report["passed"]
    broken = ward_report(curv, g, thetas=theta, symmetry="lab_field")
    assert not broken["applicable"]
    assert "laboratory" in broken["reason"]
    ordered = {(i, k): value / 2 for (i, k), value in j.items()}
    ordered.update({(k, i): value / 2 for (i, k), value in j.items()})
    pair_once = pair_once_from_ordered(ordered)
    np.testing.assert_allclose(known_j_beta(pair_once, theta), g)
    assert known_j_comparison(g, pair_once, theta)["matched"]
    assert not known_j_comparison(g + 0.01, pair_once, theta)["matched"]
    prediction = pair_once_beta(curv, theta)
    np.testing.assert_allclose(prediction, g)
    fit = j_fit_report(g, theta)
    assert fit["rank"] == 2 and fit["unique"] is False


def test_unordered_or_invalid_mesh_is_rejected_before_claiming_curvature():
    provider = RingProvider()
    kpts, kweights, occupations = _reference(provider, n=5)
    kpts[2, 0] += 0.013
    with pytest.raises(NonstationaryCurvatureError, match="uniform"):
        nonstationary_curvature(
            provider,
            kpts=kpts,
            kweights=kweights,
            occupations=occupations,
            q_frac=provider.q_frac,
            taus=provider.taus,
            phis=provider.phis,
            B_local=provider.B_local,
            translation_cutoff=2,
        )


def _ring_band_energy(provider, kpts, occupations, beta0=0.0):
    ncell = len(kpts)
    norb = len(provider.B_local)
    size = 2 * ncell * norb
    H = np.zeros((size, size), complex)

    def sl(a, mu):
        i = 2 * ((a % ncell) * norb + mu)
        return slice(i, i + 2)

    for a in range(ncell):
        for ir, R in enumerate(provider.Rlist):
            for mu in range(norb):
                for nu in range(norb):
                    H[sl(a, mu), sl(a + int(R[0]), nu)] += provider.HR[
                        ir, mu, nu
                    ] * np.eye(2)
        for mu in range(norb):
            theta = (
                2 * np.pi * provider.q_frac[0] * a
                + provider.phis[mu]
                + 2 * np.pi * provider.q_frac[0] * provider.taus[mu, 0]
            )
            field = (
                provider.B_local[mu]
                / 2
                * (
                    np.cos(theta + beta0) * np.diag([1.0, -1.0])
                    + np.sin(theta + beta0) * np.array([[0, 1], [1, 0]])
                )
            )
            H[sl(a, mu), sl(a, mu)] += field
    evals = np.linalg.eigvalsh(H)
    ref_e = []
    ref_f = []
    for ik, k in enumerate(kpts):
        e = np.linalg.eigvalsh(provider.gen_ham(k)[0])
        ref_e.extend(e)
        ref_f.extend(occupations[ik])
    f = np.asarray(ref_f)[np.argsort(np.asarray(ref_e))]
    return float(np.dot(f, evals)) / ncell


def test_periodic_curvature_normalization_is_cell_size_independent():
    estimates = []
    for ncell in (4, 8):
        provider = RingProvider(q=0.25)
        kpts, weights, occ = _reference(provider, ncell)
        curv = nonstationary_curvature(
            provider,
            kpts=kpts,
            kweights=weights,
            occupations=occ,
            q_frac=provider.q_frac,
            taus=provider.taus,
            phis=provider.phis,
            B_local=provider.B_local,
            translation_cutoff=ncell,
        )
        h = 2e-4
        fd = (
            _ring_band_energy(provider, kpts, occ, +h)
            + _ring_band_energy(provider, kpts, occ, -h)
            - 2 * _ring_band_energy(provider, kpts, occ, 0.0)
        ) / h**2
        assert curv.local_curvature[0, 0] == pytest.approx(fd, abs=2e-5, rel=2e-5)
        estimates.append(curv.local_curvature[0, 0])
    assert estimates[0] == pytest.approx(estimates[1], abs=2e-5, rel=2e-5)


def test_nonstationary_primitive_hessian_satisfies_gradient_corrected_ward():
    class ThreeSiteProvider:
        Rlist = np.zeros((1, 3), int)
        HR = np.array(
            [[[0.15, 0.35, -0.2], [0.35, -0.1, 0.28], [-0.2, 0.28, 0.07]]], complex
        )
        SR = np.array([np.eye(3)], complex)

        def gen_ham(self, k):
            q = np.array([0.31, 0.0, 0.0])
            taus = np.array([[0.0, 0.0, 0.0], [0.23, 0.0, 0.0], [0.61, 0.0, 0.0]])
            phis = np.array([0.0, 0.67, 1.91])
            fields = np.array([2.1, 1.8, 2.4])
            alpha = 2 * np.pi * taus @ q + phis
            H = np.zeros((6, 6), complex)
            S = np.zeros_like(H)
            for mu in range(3):
                for nu in range(3):
                    U = su2_rotation_y(alpha[nu] - alpha[mu])
                    im = slice(2 * mu, 2 * mu + 2)
                    jn = slice(2 * nu, 2 * nu + 2)
                    H[im, jn] += self.HR[0, mu, nu] * U
                    S[im, jn] += self.SR[0, mu, nu] * U
                H[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] += (
                    fields[mu] / 2 * np.diag([1.0, -1.0])
                )
            return H, S

    provider = ThreeSiteProvider()
    q = np.array([0.31, 0.0, 0.0])
    taus = np.array([[0.0, 0.0, 0.0], [0.23, 0.0, 0.0], [0.61, 0.0, 0.0]])
    phis = np.array([0.0, 0.67, 1.91])
    B = np.array([2.1, 1.8, 2.4])
    kpts = np.zeros((1, 3))
    weights = np.ones(1)
    occ = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    curv = nonstationary_curvature(
        provider,
        kpts=kpts,
        kweights=weights,
        occupations=occ,
        q_frac=q,
        taus=taus,
        phis=phis,
        B_local=B,
        translation_cutoff=1,
    )
    g = frozen_beta_gradient(
        provider, kpts=kpts, kweights=weights, occupations=occ, B_local=B
    )
    theta = 2 * np.pi * taus @ q + phis
    assert np.max(np.abs(g)) > 1e-3
    report = ward_report(curv, g, thetas=theta, symmetry="co_rotating", tol=1e-10)
    assert report["applicable"] and report["passed"]
    assert np.max(np.abs(curv.v2_diagonal)) > 1e-3
    np.testing.assert_allclose(pair_once_beta(curv, theta), g, atol=1e-10, rtol=1e-10)
    broken = ward_report(curv, g, thetas=theta, symmetry="lab_field")
    assert not broken["applicable"] and "laboratory" in broken["reason"]


def test_su2_pair_green_unfold_matches_explicit_lab_ring():
    from TB2J.spiral_nonstationary import pair_green_lab, su2_unfold_gate

    provider = RingProvider(q=1 / 5)
    kpts, weights, _ = _reference(provider, 5)
    ncell = 5
    H = np.zeros((2 * ncell, 2 * ncell), complex)
    theta = 2 * np.pi * provider.q_frac[0] * np.arange(ncell)
    for a in range(ncell):
        for ir, R in enumerate(provider.Rlist):
            for spin in range(2):
                H[2 * a + spin, 2 * ((a + int(R[0])) % ncell) + spin] += provider.HR[
                    ir, 0, 0
                ]
        H[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] += (
            provider.B_local[0]
            / 2
            * (
                np.cos(theta[a]) * np.diag([1.0, -1.0])
                + np.sin(theta[a]) * np.array([[0, 1], [1, 0]])
            )
        )
    energy = 0.2 + 0.3j
    expected = np.linalg.inv(energy * np.eye(2 * ncell) - H)[:2, 2:4]
    actual = pair_green_lab(
        provider,
        kpts=kpts,
        kweights=weights,
        q_frac=provider.q_frac,
        taus=provider.taus,
        phis=provider.phis,
        energy=energy,
        orbital_i=0,
        orbital_j=0,
        translation=[1, 0, 0],
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    gate = su2_unfold_gate(
        provider,
        kpts=kpts,
        kweights=weights,
        q_frac=provider.q_frac,
        taus=provider.taus,
        phis=provider.phis,
        energy=energy,
    )
    assert (
        gate["su2_unitary_residual"] < 1e-12
        and gate["scalar_phase_shortcut_used"] is False
    )


def _torus_kmesh(extents, q):
    ks = []
    for a in range(3):
        n = extents[a]
        if n == 0:
            ks.append(np.zeros(1))
        else:
            shift = (
                0.5
                if abs(q[a] * n - round(q[a] * n)) < 1e-10 and int(round(q[a] * n)) % 2
                else 0.0
            )
            ks.append((np.arange(n) + shift) / n)
    grids = np.meshgrid(*ks, indexing="ij")
    kpts = np.column_stack([g.ravel() for g in grids])
    return kpts[kpts[:, 2] == 0.0] if extents[2] == 0 else kpts


def test_two_dimensional_curvature_matches_explicit_lab_torus_fd():
    from TB2J.spiral_nonstationary import pair_green_lab

    class SquareProvider:
        Rlist = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        HR = np.array([[[0.05]], [[-0.6]], [[-0.6]], [[-0.45]], [[-0.45]]], complex)
        SR = np.array([[[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]], complex)

        def gen_ham(self, k):
            q = np.array([1 / 3, 1 / 3, 0.0])
            B = np.array([2.2])
            H = np.zeros((2, 2), complex)
            S = np.zeros((2, 2), complex)
            for ir, R in enumerate(self.Rlist):
                U = su2_rotation_y(2 * np.pi * np.dot(q, R))
                phase = np.exp(2j * np.pi * np.dot(k, R))
                H += phase * self.HR[ir, 0, 0] * U
                S += phase * self.SR[ir, 0, 0] * U
            H += B[0] / 2 * np.diag([1.0, -1.0])
            return H, S

    provider = SquareProvider()
    q = np.array([1 / 3, 1 / 3, 0.0])
    extents = np.array([3, 3, 0])
    kpts = _torus_kmesh(extents, q)
    assert len(kpts) == 9
    evals = np.array([np.linalg.eigvalsh(provider.gen_ham(k)[0]) for k in kpts])
    gap_center = 0.32
    assert evals[:, 0].max() < gap_center < evals[:, 1].min()
    occ = (evals < gap_center).astype(float)
    curv = nonstationary_curvature(
        provider,
        kpts=kpts,
        kweights=np.full(9, 1 / 9),
        occupations=occ,
        q_frac=q,
        taus=np.zeros((1, 3)),
        phis=np.zeros(1),
        B_local=np.array([2.2]),
        translation_cutoff=(3, 3, 0),
    )
    oracle = curv.lab_ring_fd_oracle
    assert oracle is not None
    assert curv.report["translation_extents"] == [3, 3, 0]
    assert curv.translations.shape == (9, 3)
    np.testing.assert_array_equal(
        curv.translations[:4], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]
    )
    for key in (("b", "b"), ("b", "d"), ("d", "b"), ("d", "d")):
        assert curv.blocks[key].shape == (9, 9)
        np.testing.assert_allclose(
            curv.blocks[key], oracle.blocks[key], atol=2e-5, rtol=2e-5
        )

    # Independent lab-torus resolvent block across a diagonal translation.
    H = np.zeros((18, 18), complex)

    def site(cell):
        idx = {tuple(int(v) for v in c): i for i, c in enumerate(curv.translations)}
        wrapped = [v % e if e else v for v, e in zip(cell, extents)]
        return idx[tuple(int(v) for v in wrapped)]

    def theta(cell):
        return 2 * np.pi * np.dot(q, cell)

    sx = np.array([[0, 1], [1, 0]], complex)
    for cell in curv.translations:
        a = site(cell)
        for ir, R in enumerate(provider.Rlist):
            b = site(cell + R)
            for spin in range(2):
                H[2 * a + spin, 2 * b + spin] += provider.HR[ir, 0, 0]
        th = theta(cell)
        H[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] += 1.1 * (
            np.cos(th) * np.diag([1.0, -1.0]) + np.sin(th) * sx
        )
    energy = 0.2 + 0.3j
    expected = np.linalg.inv(energy * np.eye(18) - H)[
        :2, 2 * site([1, 1, 0]) : 2 * site([1, 1, 0]) + 2
    ]
    actual = pair_green_lab(
        provider,
        kpts=kpts,
        kweights=np.full(9, 1 / 9),
        q_frac=q,
        taus=np.zeros((1, 3)),
        phis=np.zeros(1),
        energy=energy,
        orbital_i=0,
        orbital_j=0,
        translation=[1, 1, 0],
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
