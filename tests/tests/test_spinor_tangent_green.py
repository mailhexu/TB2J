"""Physical magnetic-tangent spinor exchange tests (split-soc tangent cutover).

Covers the physical shared-spinor cutover (docs/sympy/spinor_tangent_vertex_green.md):

- magnetic tangent vertices V^a = -(i/4)[((n x t_a).sigma) (x) I, M] with n the
  splitting-vector direction: |Delta|*sigma_a/2 transverse vertices for BOTH
  signs of a collinear splitting, exact zero on the longitudinal axis;
- the tangent trace matrix K^{ab} = Tr[V_i^a G_ij V_j^b G_ji] on full complex
  spinor Green blocks (independent brute-force reference);
- the contour prescription J^{ab} = Im contour K^{ab} dz/(2 pi) calibrated to
  the existing collinear kernel shell (J_xx = J_yy = J_collinear exactly, both
  site-sign parities), longitudinal row/column masked;
- two-site finite-angle energy anchors: isotropic J from E''/2 and the
  spin-phase bond DMI from the mixed curvature, through the real CFR contour;
- the FeO-class self-pair spurion: no longitudinal (A^{zz}-type) self-pair
  Jani survives the tangent construction;
- pair reversal K^{ab}_ij(R) = K^{ba}_ji(-R) and the transverse-leg frame.

The obsolete ExchangeNCL channel construction (spinor_pair_channels /
spinor_channels_to_exchange_tensor) is deliberately gone: its A0i-Ai0
differences vanish identically for collinear vertices and its A^{zz} channel
generated the self-pair Jani spurion.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.units import kB

from TB2J.mycfr import CFR
from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
    magnetic_tangent_vertices,
    projector_exchange_trace,
    spinor_dense_block,
    spinor_tangent_pair_matrix,
    spinor_tangent_trace,
)
from TB2J.split_soc_kernel import compute_ks_split_soc_exchange

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (SIGMA_X, SIGMA_Y, SIGMA_Z)


def _random_hermitian(n, rng, scale=1.0):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (a + a.conj().T) / np.sqrt(2.0 * n)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _collinear_pair_data(delta_signs=(1.0, 1.0), orbital_complex=True, seed=99):
    """Random metallic collinear spinor data + matching collinear channels.

    Returns ``(spinor_data, collinear_data, delta_orb)`` where the spinor
    operator is ``M_i = kron(Delta_i sigma_z, W_i)`` with signed
    ``Delta_i = delta_signs[i]`` and the collinear data shares the same
    band/projector content, so the tangent z-leg calibration against
    :func:`projector_exchange_trace` is exact.
    """
    rng = np.random.default_rng(seed)
    nkpt, nband, npj, nsite = 4, 4, 2, 2
    nproj = npj * nsite
    kpoints = rng.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eig = rng.normal(size=(2, nkpt, nband))
    eig[1] += 0.7  # spin-asymmetric spectra (spin-polarized collinear)
    coeff = rng.normal(size=(2, nkpt, nband, nproj)) + 1j * rng.normal(
        size=(2, nkpt, nband, nproj)
    )
    projector_site = np.repeat(np.arange(nsite), npj)
    site_nproj = np.full(nsite, npj)
    site_projector_indices = np.arange(nproj).reshape(nsite, npj)
    delta_orb = {}
    for site in range(nsite):
        d = rng.normal(size=(npj, npj))
        d = d + d.T
        if orbital_complex:
            d = d + 1j * rng.normal(size=(npj, npj))
            d = 0.5 * (d + d.conj().T)
        delta_orb[site] = d
    collinear = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eig,
        coefficients=coeff,
        efermi=0.2,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
    )
    nband2 = 2 * nband
    seig = np.empty((1, nkpt, nband2))
    scoeff = np.zeros((1, nkpt, nband2, 2, nproj), dtype=complex)
    for s in range(2):
        seig[0][:, s::2] = eig[s]
        scoeff[0][:, s::2, s, :] = coeff[s]
    sops = np.zeros((nsite, npj, npj, 2, 2), dtype=complex)
    for site in range(nsite):
        sops[site] = np.einsum(
            "st,pq->pqst", delta_signs[site] * SIGMA_Z, delta_orb[site]
        )
    spinor = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=seig,
        coefficients=scoeff,
        efermi=0.2,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        spinor_operator=sops,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        nspinor=2,
    )
    return spinor, collinear, delta_orb


def _chain_projector_data(nk=6, b=1.0, t1=0.3, t2=0.2, phi=0.0):
    """1D two-site-cell insulating chain as no-SOC spinor ProjectorGreenData.

    Sites A, B carry local fields ``-b sigma_z``; hopping is
    ``t1`` (intra-cell) and ``t2 e^{-2 pi i k}`` (inter-cell), each carrying
    the bond spin phase ``e^{+i phi sigma_z / 2}`` when ``phi != 0`` (the
    linked-dimer SOC model of the design). The lowest band is occupied
    and remains isolated by a finite gap over the chosen parameters.

    Returns ``(data, energy_function)`` where ``energy_function(theta_a,
    theta_b)`` evaluates the occupied band sum for tilts of the A/B fields
    by the given angles (rotating the FIELD only, hopping frozen).
    """
    kpoints = np.stack(
        [np.arange(nk, dtype=float) / nk, np.zeros(nk), np.zeros(nk)], axis=1
    )
    weights = np.full(nk, 1.0 / nk)

    def hamiltonian(k, theta_a=0.0, theta_b=0.0, b_axis=0):
        """Hermitian four-state (spin, site) Hamiltonian."""
        c = t1 + t2 * np.exp(-2j * np.pi * k)
        hopping = np.zeros((4, 4), dtype=complex)
        for s, sign in enumerate((1, -1)):
            h_ab = c * np.exp(0.5j * sign * phi)
            hopping[2 * s, 2 * s + 1] = h_ab
            hopping[2 * s + 1, 2 * s] = h_ab.conjugate()
        field = np.zeros((4, 4), dtype=complex)
        for theta, p, axis in (
            (theta_a, 0, SIGMA_X),
            (theta_b, 1, (SIGMA_X, SIGMA_Y)[b_axis]),
        ):
            sigma_n = np.sin(theta) * axis + np.cos(theta) * SIGMA_Z
            proj = np.zeros((2, 2), dtype=complex)
            proj[p, p] = 1.0
            field += np.kron(-b * sigma_n, proj)
        return hopping + field

    # reference no-SOC spectrum and eigenvectors at theta = 0
    nband = 4
    nproj = 2
    eigenvalues = np.empty((1, nk, nband))
    coefficients = np.empty((1, nk, nband, 2, nproj), dtype=complex)
    gaps = []
    for ik, k in enumerate(kpoints[:, 0]):
        h = hamiltonian(k)
        w, v = np.linalg.eigh(h)
        order = np.argsort(w)
        eigenvalues[0, ik] = w[order]
        coefficients[0, ik] = v[:, order].T.reshape(nband, 2, nproj)  # (n, s, p)
        gaps.append(w[1] - w[0])
    efermi = float(np.mean(0.5 * (eigenvalues[0, :, 0] + eigenvalues[0, :, 1])))
    sops = np.zeros((2, 1, 1, 2, 2), dtype=complex)
    for site in range(2):
        sops[site, 0, 0] = -2.0 * b * SIGMA_Z
    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=efermi,
        projector_site=np.repeat(np.arange(2), 1),
        projector_atom=np.arange(2),
        site_nproj=np.ones(2, dtype=int),
        site_projector_indices=np.arange(2).reshape(2, 1),
        spinor_operator=sops,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        nspinor=2,
        positions=np.array([[0.0, 0, 0], [1.5, 0, 0]]),
        cell=np.eye(3) * 10.0,
        atomic_numbers=np.array([26, 26]),
    )

    def energy_function(theta_a, theta_b, b_axis=0):
        total = 0.0
        for k in kpoints[:, 0]:
            w = np.linalg.eigvalsh(hamiltonian(k, theta_a, theta_b, b_axis))
            total += w[0] / nk
        return total

    return data, energy_function, float(min(gaps))


def _kernel_chain_entries(data, rpts, nz=64):
    res = compute_ks_split_soc_exchange(
        data,
        np.zeros((data.nkpt, data.nband, data.nband), dtype=complex),
        lam=1.0,
        Rpts=rpts,
        nz=nz,
        smearing_eV=0.005,
    )
    return res["exchange"]


# ---------------------------------------------------------------------------
# vertex construction
# ---------------------------------------------------------------------------


def test_vertices_collinear_both_signs():
    """V^a = |Delta| sigma_a/2 kron-ed with the orbital factor for +/-z."""
    rng = np.random.default_rng(5)
    w_orb = _random_hermitian(2, rng) + np.eye(2)  # positive trace
    for delta in (2.0, -2.0):
        block = np.einsum("st,pq->pqst", delta * SIGMA_Z, w_orb)
        info = magnetic_tangent_vertices(block)
        expected_n = np.array([0.0, 0.0, np.sign(delta)])
        np.testing.assert_allclose(info["n"], expected_n, atol=1e-12)
        assert info["magnitude"] == pytest.approx(abs(delta) * np.trace(w_orb))
        v = info["vertices"]
        np.testing.assert_allclose(v[0], 0.5 * abs(delta) * np.kron(SIGMA_X, w_orb))
        np.testing.assert_allclose(v[1], 0.5 * abs(delta) * np.kron(SIGMA_Y, w_orb))
        assert np.abs(v[2]).max() == 0.0  # exact longitudinal zero


def test_vertices_general_axis_transverse_plane():
    """x-axis reference: transverse vertices on (y, z), exact x zero."""
    w_orb = _random_hermitian(2, np.random.default_rng(6)) + np.eye(2)
    block = np.einsum("st,pq->pqst", 3.0 * SIGMA_X, w_orb)
    info = magnetic_tangent_vertices(block)
    np.testing.assert_allclose(info["n"], [1.0, 0.0, 0.0], atol=1e-12)
    v = info["vertices"]
    assert np.abs(v[0]).max() == 0.0
    np.testing.assert_allclose(v[1], 0.5 * 3.0 * np.kron(SIGMA_Y, w_orb))
    np.testing.assert_allclose(v[2], 0.5 * 3.0 * np.kron(SIGMA_Z, w_orb))


def test_vertices_reject_nonmagnetic_and_nonhermitian():
    block = np.zeros((1, 1, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="no magnetic splitting"):
        magnetic_tangent_vertices(block)
    nh = np.einsum("st,pq->pqst", SIGMA_Z, np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="Hermitian"):
        magnetic_tangent_vertices(nh)


# ---------------------------------------------------------------------------
# tangent trace matrix (independent brute force)
# ---------------------------------------------------------------------------


def test_tangent_pair_matrix_matches_brute_force():
    """K^{ab} from full complex G equals the dense trace chain."""
    rng = np.random.default_rng(11)
    ni, nj = 2, 3
    mi = np.einsum("st,pq->pqst", 1.3 * SIGMA_Z, _random_hermitian(ni, rng))
    mj = np.einsum("st,pq->pqst", -0.7 * SIGMA_Z, _random_hermitian(nj, rng))
    vi = magnetic_tangent_vertices(mi)["vertices"]
    vj = magnetic_tangent_vertices(mj)["vertices"]
    gij = rng.normal(size=(ni, nj, 2, 2)) + 1j * rng.normal(size=(ni, nj, 2, 2))
    gji = rng.normal(size=(nj, ni, 2, 2)) + 1j * rng.normal(size=(nj, ni, 2, 2))
    k = spinor_tangent_pair_matrix(
        vi, spinor_dense_block(gij), vj, spinor_dense_block(gji)
    )
    gij_d, gji_d = spinor_dense_block(gij), spinor_dense_block(gji)
    for a in range(3):
        for b in range(3):
            want = np.trace(vi[a] @ gij_d @ vj[b] @ gji_d)
            assert k[a, b] == pytest.approx(want, rel=1e-12, abs=1e-14)
    # pair reversal before integration: K^{ab}_ij = K^{ba}_ji (same blocks)
    k_rev = spinor_tangent_pair_matrix(vj, gji_d, vi, gij_d)
    np.testing.assert_allclose(k, k_rev.T, atol=1e-12)


# ---------------------------------------------------------------------------
# collinear kernel calibration (contour sign + normalization)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta_signs", [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_collinear_anchor_matches_kernel_shell(delta_signs):
    """z-leg tangent Jxx = Jyy = collinear-kernel J for BOTH site parities."""
    spinor, collinear, delta_orb = _collinear_pair_data(delta_signs=delta_signs)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    res = compute_ks_split_soc_exchange(
        spinor,
        np.zeros((spinor.nkpt, spinor.nband, spinor.nband), dtype=complex),
        lam=1.0,
        Rpts=rpts,
        nz=30,
        smearing_eV=0.05,
    )
    cg = ProjectorGreen(collinear)
    signs = {}
    for site, d in delta_orb.items():
        tr = float(np.real(np.trace(d)))
        signs[site] = 1.0 if tr >= 0 else -1.0
    contour = CFR(nz=30, T=0.05 / kB)
    values = {
        (tuple(int(x) for x in R), i, j): []
        for R in rpts
        for i in range(2)
        for j in range(2)
    }
    for energy in contour.path:
        trace = projector_exchange_trace(
            cg, rpts, energy=energy, local_operators=delta_orb, sites=[0, 1]
        )
        for key in values:
            values[key].append(trace["trace"][key])
    for (r, i, j), entry in res["exchange"].items():
        frame = entry["frame"]
        assert (frame["n"], frame["u"], frame["v"]) == (2, 0, 1)
        integrated = contour.integrate_values(np.asarray(values[(r, i, j)]))
        j_cl = float(np.imag(integrated)) / (signs[i] * signs[j])
        jl = entry["J_leg"]
        assert jl[0, 0] == pytest.approx(j_cl, rel=1e-8, abs=1e-10)
        assert jl[1, 1] == pytest.approx(j_cl, rel=1e-8, abs=1e-10)
        assert entry["mask_residual"] == pytest.approx(0.0, abs=1e-12)
        assert "Jiso" not in entry and "dmi" not in entry and "jani" not in entry


def test_collinear_anchor_contour_normalization_documented():
    """The /(2 pi) Im contour prescription is the calibrated contract."""
    spinor, _, _ = _collinear_pair_data(seed=7)
    green = ProjectorGreen(spinor)
    vertices = {
        site: magnetic_tangent_vertices(spinor.spinor_operator[site])["vertices"]
        for site in range(2)
    }
    rpts = np.array([[1, 0, 0], [-1, 0, 0]], dtype=int)
    trace = spinor_tangent_trace(green, rpts, energy=0.3 + 0.2j, vertices=vertices)
    assert trace["method"] == "spinor_tangent_trace"
    assert "Im contour" in trace["normalization"]
    k = trace["K_ijR"]
    assert k[((1, 0, 0), 0, 1)].shape == (3, 3)
    # longitudinal vertex is exactly zero: K^{zb} = K^{bz} = 0
    np.testing.assert_allclose(k[((1, 0, 0), 0, 1)][2, :], 0.0, atol=1e-14)
    np.testing.assert_allclose(k[((1, 0, 0), 0, 1)][:, 2], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# two-site finite-angle energy anchors (physical sign + factor)
# ---------------------------------------------------------------------------


def test_two_site_energy_second_derivative_isotropic():
    """Sum_R J_xx(R,0,1) x2 = -d2E/dthetaA dthetaB for the tilted chain."""
    data, energy_fn, gap = _chain_projector_data(nk=6, b=1.0, t1=0.3, t2=0.2)
    assert gap > 0.2
    nk = data.nkpt
    rpts = [[r, 0, 0] for r in range(nk)] + [[-r, 0, 0] for r in range(1, nk)]
    entries = _kernel_chain_entries(data, np.asarray(rpts, dtype=int))
    # sum over the ordered pairs of the full mesh (period nk: all distinct R)
    total = 0.0
    for r in range(nk):
        total += entries[((r, 0, 0), 0, 1)]["J_leg"][0, 0]
    # ordered sum counts (A,B) and (B,A): x2
    ordered_total = 2.0 * total

    h = 1e-3
    fd = (energy_fn(h, h) - energy_fn(h, -h) - energy_fn(-h, h) + energy_fn(-h, -h)) / (
        4.0 * h * h
    )
    assert ordered_total == pytest.approx(-fd, rel=2e-4)


def test_two_site_bond_phase_dmi():
    """Bond spin phase phi gives D_z; mixed curvature pins sign and factor."""
    phi = 0.7
    b, t1, t2 = 1.0, 0.0, 0.4
    data, energy_fn, gap = _chain_projector_data(nk=6, b=b, t1=t1, t2=t2, phi=phi)
    assert gap > 0.2
    nk = data.nkpt
    rpts = [[r, 0, 0] for r in range(nk)] + [[-r, 0, 0] for r in range(1, nk)]
    entries = _kernel_chain_entries(data, np.asarray(rpts, dtype=int))
    # With G(R) = sum_k exp(-2 pi i k.R)G(k), c(k) = t exp(-ik)
    # places the A->B bond at R=-1, equivalent to R=nk-1 on this mesh.
    analytic = b * t2 * np.sin(phi) / (8.0 * (b + t2))
    j_xy = entries[((-1, 0, 0), 0, 1)]["J_leg"][0, 1]
    assert j_xy == pytest.approx(analytic, rel=5e-3)
    for r in range(nk):
        if r == nk - 1:
            continue
        assert entries[((r, 0, 0), 0, 1)]["J_leg"][0, 1] == pytest.approx(0.0, abs=1e-9)
    # mixed curvature: 2 x sum_R J_xy = -d2E/dthetaAx dthetaBy
    h = 1e-3

    def e_mix(dta, dtb):
        return energy_fn(dta, dtb, b_axis=1)

    fd = (e_mix(h, h) - e_mix(h, -h) - e_mix(-h, h) + e_mix(-h, -h)) / (4.0 * h * h)
    total = sum(entries[((r, 0, 0), 0, 1)]["J_leg"][0, 1] for r in range(nk))
    assert 2.0 * total == pytest.approx(-fd, rel=5e-3)


def test_no_soc_chain_has_no_antisymmetric_part():
    """phi = 0 (no SOC): the transverse block is symmetric, D_z ~ 0."""
    data, _, _ = _chain_projector_data(nk=6, b=1.0, t1=0.3, t2=0.2, phi=0.0)
    entries = _kernel_chain_entries(
        data, np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=int)
    )
    entry = entries[((1, 0, 0), 0, 1)]["J_leg"]
    assert entry[0, 1] == pytest.approx(entry[1, 0], rel=1e-6)


# ---------------------------------------------------------------------------
# self-pair spurion (FeO class) and reciprocity
# ---------------------------------------------------------------------------


def test_self_pair_has_no_longitudinal_spurion():
    """The old A^{zz} self-pair Jani spurion cannot exist: V^n = 0 exactly."""
    spinor, _, _ = _collinear_pair_data(delta_signs=(1.0, -1.0), seed=13)
    rpts = np.array([[0, 0, 0]], dtype=int)
    res = compute_ks_split_soc_exchange(
        spinor,
        np.zeros((spinor.nkpt, spinor.nband, spinor.nband), dtype=complex),
        lam=1.0,
        Rpts=rpts,
        nz=30,
        smearing_eV=0.05,
    )
    entry = res["exchange"][((0, 0, 0), 0, 0)]
    jl = entry["J_leg"]
    # the longitudinal row/column is structurally zero: no Jani_zz-class
    # self-pair residue can be produced from a collinear reference
    assert entry["frame"]["n"] == 2
    np.testing.assert_allclose(jl[2, :], 0.0, atol=0.0)
    np.testing.assert_allclose(jl[:, 2], 0.0, atol=0.0)


def test_pair_reversal_after_integration():
    """J^{ab}_ij(R) = J^{ba}_ji(-R) on the contour-integrated blocks."""
    spinor, _, _ = _collinear_pair_data(seed=17)
    green = ProjectorGreen(spinor)
    vertices = {
        site: magnetic_tangent_vertices(spinor.spinor_operator[site])["vertices"]
        for site in range(2)
    }
    rpts = np.array([[1, 0, 0], [-1, 0, 0]], dtype=int)
    contour = CFR(nz=24, T=0.05 / kB)
    acc = {
        key: []
        for key in ((r, i, j) for r in map(tuple, rpts) for i in (0, 1) for j in (0, 1))
    }
    for energy in contour.path:
        trace = spinor_tangent_trace(green, rpts, energy=energy, vertices=vertices)
        for key, value in trace["K_ijR"].items():
            acc[key].append(value)
    for (r, i, j), vals in acc.items():
        integrated = contour.integrate_values(np.asarray(vals))
        rev = contour.integrate_values(np.asarray(acc[(tuple(-x for x in r), j, i)]))
        np.testing.assert_allclose(integrated, rev.T, rtol=1e-9, atol=1e-12)


def test_leg_frame_and_mask_in_kernel_output():
    """Kernel entries expose only the masked transverse block + frame."""
    spinor, _, _ = _collinear_pair_data(seed=23)
    res = compute_ks_split_soc_exchange(
        spinor,
        np.zeros((spinor.nkpt, spinor.nband, spinor.nband), dtype=complex),
        lam=1.0,
        Rpts=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int),
        nz=24,
        smearing_eV=0.05,
    )
    for key, entry in res["exchange"].items():
        frame = entry["frame"]
        assert frame["triad"] == (0, 1, 2)
        np.testing.assert_allclose(frame["axis"], [0.0, 0.0, 1.0])
        site_n = frame["site_n"]
        assert set(site_n) == {0, 1}
        n = frame["n"]
        jl = entry["J_leg"]
        np.testing.assert_allclose(jl[n, :], 0.0, atol=0.0)
        np.testing.assert_allclose(jl[:, n], 0.0, atol=0.0)
        assert np.isfinite(jl).all()


def test_frame_rotation_checks_full_green_and_site_vertex():
    from dataclasses import replace

    from TB2J.split_soc_kernel import spinor_frame_rotation_residual, su2_axis_rotation

    reference, _, _ = _collinear_pair_data(seed=29)
    u = su2_axis_rotation([1, 0, 0])
    rotated = replace(
        reference,
        coefficients=np.einsum("st,ckntp->cknsp", u, reference.coefficients),
        spinor_operator=np.einsum(
            "su,apquv,tv->apqst", u, reference.spinor_operator, u.conj()
        ),
    )
    report = spinor_frame_rotation_residual(rotated, reference, [1, 0, 0], sites=[0, 1])
    assert report["max_g_residual"] < 1e-11
    assert report["max_vertex_residual"] < 1e-11
    wrong = replace(rotated, coefficients=reference.coefficients)
    bad = spinor_frame_rotation_residual(wrong, reference, [1, 0, 0], sites=[0, 1])
    assert bad["max_g_residual"] > 1e-4


def test_full_tensor_requires_three_independent_references():
    from TB2J.split_soc_kernel import merge_transverse_legs

    tensor = np.array([[1.0, 0.3, -0.2], [0.1, 2.0, 0.4], [-0.5, 0.6, 3.0]])
    key = ((1, 0, 0), 0, 1)
    legs = {}
    for axis, name in enumerate("xyz"):
        measured = tensor.copy()
        measured[axis, :] = 0
        measured[:, axis] = 0
        legs[name] = {
            "exchange": {
                key: {"J_leg": measured, "frame": {"n": axis}, "mask_residual": 0.0}
            }
        }
    for subset in ({"z": legs["z"]}, {"x": legs["x"], "z": legs["z"]}):
        with pytest.raises(ValueError, match="three independent"):
            merge_transverse_legs(subset)
    merged = merge_transverse_legs(legs)
    np.testing.assert_allclose(merged["exchange"][key]["tensor"], tensor, atol=1e-12)
    assert merged["diagnostics"]["min_rank"] == 9


# NOTE: test_spinor_writer_reconstructs_three_reference_energy_curvature
# (the write_spinor_projector_exchange_out three-reference pin) migrates
# with the GPAW/exporter story that owns TB2J.interfaces.gpaw_spinor_
# projector; it is intentionally not carried on the VASP adapter branch.
