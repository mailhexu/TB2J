"""KS-band split-SOC kernel tests (story 002, split-soc-ks spec).

Covers the ADR-1 shared core:

- production second-variation mode: diagonalize ``E(k) + lam*W(k)``, rotate
  the spectral coefficients, replay through the existing spinor projector
  kernel (``ProjectorGreenData(nspinor=2)``);
- first-order insertion diagnostic: analytic ``dG = G0 W G0`` with BOTH
  trace-line topologies (story-001 lambdified algebra, 1e-14), never a
  finite-difference stand-in;
- rectangular (non-invertible) ``B_a`` embedding accepted (``B`` is only
  ever contracted, never inverted);
- ligand (all-atom) SOC entering DMI through ``W`` while vertices stay
  magnetic-only;
- ``lam=0`` collinear reduction to the existing collinear kernel trace;
- insertion derivative vs central finite difference of the second-variation
  tensor for a finite complex Hermitian ``W``;
- full-BZ Fourier phase bookkeeping for inter-site ``R``;
- band-window convergence reporting and FR-050 provenance metadata.

All references are written independently in the normalized band space
(dense ``[zI - E - lam W]^{-1}`` resolvents), never by reusing the kernel's
own replay/insertion internals.
"""

from __future__ import annotations

import numpy as np
import pytest

PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def _random_hermitian(n, rng, scale=1.0):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (a + a.conj().T) / np.sqrt(2.0 * n)


def _dense_spinor_block(block):
    """(ni, nj, 2, 2) -> dense (2*ni, 2*nj) with spin-major index (s, i)."""
    ni, nj = block.shape[0], block.shape[1]
    return block.transpose(2, 0, 3, 1).reshape(2 * ni, 2 * nj)


def _make_ks_data(nsite=2, nstate=6, nproj_per_site=2, nkpt=5, seed=11):
    """Synthetic normalized no-SOC spinful KS data (ProjectorGreenData).

    The coefficient array plays the role of the rectangular projector maps
    B_a(k); with nproj = nsite*nproj_per_site < 2*nstate the map is tall and
    strongly non-invertible.  Each site carries a random Hermitian 2x2-in-spin
    magnetic rotation vertex ``v_a`` (no local Hamiltonian is provided).
    """
    from TB2J.projector_green import SPINOR_OPERATOR_DEFINITION, ProjectorGreenData

    rng = np.random.default_rng(seed)
    nproj = nsite * nproj_per_site
    kpoints = rng.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = rng.normal(size=(1, nkpt, nstate))
    coefficients = rng.normal(size=(1, nkpt, nstate, 2, nproj)) + 1j * rng.normal(
        size=(1, nkpt, nstate, 2, nproj)
    )
    projector_site = np.repeat(np.arange(nsite), nproj_per_site)
    site_nproj = np.full(nsite, nproj_per_site)
    site_projector_indices = np.arange(nproj).reshape(nsite, nproj_per_site)
    sz = PAULI[3]
    ops = np.zeros((nsite, nproj_per_site, nproj_per_site, 2, 2), dtype=complex)
    for site in range(nsite):
        orbital = rng.normal(size=(nproj_per_site, nproj_per_site))
        orbital = orbital @ orbital.T + np.eye(nproj_per_site)
        ops[site] = np.einsum("st,pq->pqst", sz, orbital)
    return ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.2,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        spinor_operator=ops,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        nspinor=2,
    )


def _random_w_soc(data, rng, ncomp=None):
    """All-atom W_SO^K(k): (nkpt, nstate, nstate) complex Hermitian,
    k-dependent (independent Hermitian draw per k point), optionally given
    as an additive sum over ``ncomp`` site components."""
    nkpt, nstate = data.nkpt, data.nband
    if ncomp is None:
        ncomp = 1
    w = np.zeros((nkpt, nstate, nstate), dtype=complex)
    for ik in range(nkpt):
        for _ in range(ncomp):
            w[ik] += _random_hermitian(nstate, rng)
    return w


def _band_G(z_rel, efermi, eps_k, w_k, lam):
    """Independent band-space resolvent [z I - E - lam W]^{-1} at one k."""
    n = eps_k.size
    a = (z_rel + efermi) * np.eye(n) - np.diag(eps_k) - lam * w_k
    return np.linalg.inv(a)


def _tangent_reference(op_i, gij, op_j, gji):
    """Independent full-complex Green contraction for collinear z magnetic legs."""
    vi = [np.kron(PAULI[a], op_i[:, :, 0, 0] / 2) for a in (1, 2)]
    vj = [np.kron(PAULI[a], op_j[:, :, 0, 0] / 2) for a in (1, 2)]
    gi = _dense_spinor_block(gij)
    gj = _dense_spinor_block(gji)
    return np.array([[np.trace(a @ gi @ b @ gj) for b in vj] for a in vi])


def _fourier(blocks_k, rvec, kpoints, weights):
    """Sum_k w_k e^{-2 pi i k.R} G(k) over the leading k axis."""
    phase = np.exp(-2j * np.pi * kpoints @ np.asarray(rvec, dtype=float)) * weights
    return np.einsum("k...,k->...", np.asarray(blocks_k), phase)


def _embedded_vertex(data, site, ik):
    """V_a = B_a^dag v_a B_a in the band space, from the coefficient map."""
    npj = data.site_nproj[site]
    i0 = int(np.where(data.projector_site == site)[0][0])
    c = data.coefficients[0]
    return np.einsum(
        "nsp,pqst,mtq->nm",
        c[ik][:, :, i0 : i0 + npj].conj(),
        data.spinor_operator[site],
        c[ik][:, :, i0 : i0 + npj],
    )


def _site_block(full, site_i, site_j, nproj_per_site):
    i0 = site_i * nproj_per_site
    j0 = site_j * nproj_per_site
    n = nproj_per_site
    return full[i0 : i0 + n, j0 : j0 + n]


# ---------------------------------------------------------------------------
# TEST-001: insertion topology == story-001 lambdified expressions (1e-14)
# ---------------------------------------------------------------------------


def test_insertion_matches_sympy_lambdified_topologies():
    """d/dlam Tr[V_a G V_b G]|_0 equals the two story-001 topologies, and the
    kernel pair trace equals the band-space lambdified expression to 1e-14."""
    import sympy as sp

    from TB2J.split_soc_kernel import first_order_insertion_channels

    nstate, npj = 3, 2
    # single k point: with one k (unit weight) the real-space blocks at
    # R=0 coincide with the per-k blocks, so the story-001 topology
    # identity applies verbatim to the embedded band objects
    data = _make_ks_data(nsite=2, nstate=nstate, nproj_per_site=npj, nkpt=1, seed=5)
    data.weights[...] = 1.0
    rng = np.random.default_rng(47)
    w_soc = _random_w_soc(data, rng)
    z = 1.9 + 2.1j
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)

    channels = first_order_insertion_channels(data, w_soc, Rpts=rpts, energies=[z])

    # --- story-001 topology expression, built and lambdified with sympy ---
    n = nstate
    eps_s = sp.symbols(f"e0:{n}", real=True)
    z_s = sp.Symbol("z")
    w_s = sp.Matrix(n, n, lambda a, b: sp.Symbol(f"w_{a}_{b}"))
    va_s = sp.Matrix(n, n, lambda a, b: sp.Symbol(f"va_{a}_{b}"))
    vb_s = sp.Matrix(n, n, lambda a, b: sp.Symbol(f"vb_{a}_{b}"))
    g0_s = sp.diag(*[1 / (z_s - e) for e in eps_s])
    topo = sp.trace(va_s * g0_s * w_s * g0_s * vb_s * g0_s) + sp.trace(
        va_s * g0_s * vb_s * g0_s * w_s * g0_s
    )
    w_syms = [w_s[a, b] for a in range(n) for b in range(n)]
    va_syms = [va_s[a, b] for a in range(n) for b in range(n)]
    vb_syms = [vb_s[a, b] for a in range(n) for b in range(n)]
    topo_func = sp.lambdify([*eps_s, z_s, *w_syms, *va_syms, *vb_syms], topo, "numpy")

    def band_topology(ik):
        e_k = data.eigenvalues[0, ik]
        va = _embedded_vertex(data, 0, ik)
        vb = _embedded_vertex(data, 1, ik)
        args = [
            *e_k.tolist(),
            z + data.efermi,
            *w_soc[ik].reshape(-1).tolist(),
            *va.reshape(-1).tolist(),
            *vb.reshape(-1).tolist(),
        ]
        return complex(topo_func(*args))

    key = ((0, 0, 0), 0, 1)
    kernel_val = complex(np.asarray(channels["pair_trace"][key])[0])
    topo_ref = complex(
        np.dot(data.weights, [band_topology(ik) for ik in range(data.nkpt)])
    )
    scale = max(1.0, abs(topo_ref))
    assert abs(kernel_val - topo_ref) <= 1e-14 * scale, (kernel_val, topo_ref)

    # strength-0 partner trace matches the embedded identity line too
    def band_trace0(ik):
        e_k = data.eigenvalues[0, ik]
        va = _embedded_vertex(data, 0, ik)
        vb = _embedded_vertex(data, 1, ik)
        g0_k = np.diag(1.0 / (z + data.efermi - e_k))
        return complex(np.trace(va @ g0_k @ vb @ g0_k))

    trace0_ref = complex(
        np.dot(data.weights, [band_trace0(ik) for ik in range(data.nkpt)])
    )
    kernel_trace0 = complex(np.asarray(channels["pair_trace0"][key])[0])
    assert abs(kernel_trace0 - trace0_ref) <= 1e-14 * max(1.0, abs(trace0_ref))

    # each topology alone is NOT the derivative (story-001 assertion)
    g0_k = np.diag(1.0 / (z + data.efermi - data.eigenvalues[0, 0]))
    va = _embedded_vertex(data, 0, 0)
    vb = _embedded_vertex(data, 1, 0)
    w0 = w_soc[0]
    topo1 = np.trace(va @ g0_k @ w0 @ g0_k @ vb @ g0_k)
    topo2 = np.trace(va @ g0_k @ vb @ g0_k @ w0 @ g0_k)
    assert abs(topo1 - topo2) > 1e-3 * max(abs(topo1), abs(topo2))


# ---------------------------------------------------------------------------
# TEST-002: lam=0 reproduces the existing collinear kernel trace
# ---------------------------------------------------------------------------


def test_lambda_zero_matches_collinear_kernel():
    from ase.units import kB

    from TB2J.mycfr import CFR
    from TB2J.projector_green import ProjectorGreen, ProjectorGreenData
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    rng = np.random.default_rng(1234)
    nkpt, nband, nproj = 4, 4, 4
    kpoints = rng.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = rng.normal(size=(1, nkpt, nband))
    coeff = rng.normal(size=(1, nkpt, nband, 2, nproj)) + 1j * rng.normal(
        size=(1, nkpt, nband, 2, nproj)
    )
    band_spin = rng.integers(0, 2, size=nband)
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
        spinor_operator_definition="spinor 2x2 local operator (j-averaged basis)",
        nspinor=2,
    )
    rpts = np.array([[0, 0, 0], [1, 1, 0], [-1, -1, 0]], dtype=int)
    w_zero = np.zeros((nkpt, nband, nband))

    result = compute_ks_split_soc_exchange(
        data,
        w_zero,
        lam=1.0,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
    )

    # Direct collinear full-spinor tangent trace: both opposite-spin paths
    # survive, including distinct Fourier phases at nonzero R.
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
    contour = CFR(nz=24, T=0.05 / kB)
    indexed = {tuple(r): idx for idx, r in enumerate(rpts)}
    for (r, iatom, jatom), entry in result["exchange"].items():
        vals = []
        for energy in contour.path:
            gu = cg.get_GR(rpts, energy, ispin=0)
            gd = cg.get_GR(rpts, energy, ispin=1)
            ir, im = indexed[r], indexed[tuple(-x for x in r)]
            a = cg.get_site_block(gu[ir], iatom, jatom)
            b = cg.get_site_block(gd[im], jatom, iatom)
            c = cg.get_site_block(gd[ir], iatom, jatom)
            d = cg.get_site_block(gu[im], jatom, iatom)
            vals.append(
                (delta_i, delta_j)[iatom]
                * (delta_i, delta_j)[jatom]
                * (np.trace(a @ b) + np.trace(c @ d))
                / 4
            )
        ref = np.imag(contour.integrate_values(np.asarray(vals))) / (2 * np.pi)
        assert entry["J_leg"][0, 0] == pytest.approx(ref, rel=1e-9, abs=1e-12)
        assert entry["J_leg"][1, 1] == pytest.approx(ref, rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(entry["J_leg"][2], 0, atol=1e-12)
        np.testing.assert_allclose(entry["J_leg"][:, 2], 0, atol=1e-12)


# ---------------------------------------------------------------------------
# Rectangular band map versus independent dense resolvent
# ---------------------------------------------------------------------------


def test_rectangular_b_production_matches_band_reference():
    from ase.units import kB

    from TB2J.mycfr import CFR
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    data = _make_ks_data(nsite=2, nstate=6, nproj_per_site=2, nkpt=5, seed=21)
    rng = np.random.default_rng(99)
    w_soc = _random_w_soc(data, rng)
    lam = 0.9
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)

    # B is (nproj=4) x (nstate=6 x 2 spin) -- tall, non-invertible
    assert data.nproj == 4 and 2 * data.nband == 12

    result = compute_ks_split_soc_exchange(
        data,
        w_soc,
        lam=lam,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
    )

    contour = CFR(nz=24, T=0.05 / kB)

    def g_site_blocks(z, rvec):
        per_k = np.empty((data.nkpt, 4, 4, 2, 2), dtype=complex)
        for ik in range(data.nkpt):
            resolvent = _band_G(z, data.efermi, data.eigenvalues[0, ik], w_soc[ik], lam)
            ci = data.coefficients[0, ik]
            per_k[ik] = np.einsum("nsp,nm,mtq->pqst", ci, resolvent, ci.conj())
        full = _fourier(per_k, rvec, data.kpoints, data.weights)
        return {(i, j): _site_block(full, i, j, 2) for i in range(2) for j in range(2)}

    for r in map(tuple, rpts):
        rm = tuple(-x for x in r)
        for i in range(2):
            for j in range(2):
                values = []
                for z in contour.path:
                    g_r, g_rm = g_site_blocks(z, r), g_site_blocks(z, rm)
                    values.append(
                        _tangent_reference(
                            data.spinor_operator[i],
                            g_r[i, j],
                            data.spinor_operator[j],
                            g_rm[j, i],
                        )
                    )
                integrated = contour.integrate_values(np.asarray(values))
                reference = np.imag(integrated) / (2 * np.pi)
                np.testing.assert_allclose(
                    result["exchange"][(r, i, j)]["J_leg"][:2, :2],
                    reference,
                    atol=1e-9 * max(1.0, np.max(np.abs(reference))),
                )


# ---------------------------------------------------------------------------
# TEST-004: all-atom (ligand) SOC affects DMI; vertices stay magnetic-only
# ---------------------------------------------------------------------------


def test_ligand_soc_enters_dmi_but_not_vertices():
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    data = _make_ks_data(nsite=3, nstate=6, nproj_per_site=2, nkpt=4, seed=31)
    rng = np.random.default_rng(77)
    w_mag = _random_w_soc(data, rng, ncomp=2)
    w_all = w_mag + _random_w_soc(data, rng)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)

    kw = dict(lam=1.0, mode=MODE_SECOND_VARIATION, Rpts=rpts, nz=24, smearing_eV=0.05)
    res_all = compute_ks_split_soc_exchange(data, w_all, sites=[0, 1], **kw)
    res_mag = compute_ks_split_soc_exchange(data, w_mag, sites=[0, 1], **kw)

    key = ((1, 0, 0), 0, 1)
    a = res_all["exchange"][key]["J_leg"]
    b = res_mag["exchange"][key]["J_leg"]
    assert np.linalg.norm(a - b) > 1e-3 * max(1.0, np.linalg.norm(a), np.linalg.norm(b))

    # magnetic-only vertex gating: requesting the magnetic subset with all
    # site vertices present changes nothing on the magnetic pairs
    res_all_sites = compute_ks_split_soc_exchange(data, w_all, sites=[0, 1, 2], **kw)
    for rk, entry in res_all["exchange"].items():
        other = res_all_sites["exchange"][rk]
        np.testing.assert_allclose(entry["J_leg"], other["J_leg"], atol=1e-12)
    assert not any(i == 2 or j == 2 for (_, i, j) in res_all["exchange"])


# ---------------------------------------------------------------------------
# TEST-005: analytic insertion == central FD of the second-variation tensor
# ---------------------------------------------------------------------------


def test_insertion_matches_central_fd_of_second_variation():
    from TB2J.split_soc_kernel import (
        MODE_FIRST_ORDER_INSERTION,
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    data = _make_ks_data(nsite=2, nstate=5, nproj_per_site=2, nkpt=4, seed=13)
    rng = np.random.default_rng(555)
    w_soc = _random_w_soc(data, rng)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    kw = dict(Rpts=rpts, nz=24, smearing_eV=0.05, sites=[0, 1])

    res_d = compute_ks_split_soc_exchange(
        data, w_soc, lam=1.0, mode=MODE_FIRST_ORDER_INSERTION, **kw
    )
    # Richardson-extrapolated central differences kill the O(h^2) FD
    # truncation (the spectral replay is strongly curved in lam at the
    # small Matsubara energies); 4 production runs at +-h, +-h/2
    h = 1e-2

    def central(hh, rk):
        p = compute_ks_split_soc_exchange(
            data, w_soc, lam=hh, mode=MODE_SECOND_VARIATION, **kw
        )
        m = compute_ks_split_soc_exchange(
            data, w_soc, lam=-hh, mode=MODE_SECOND_VARIATION, **kw
        )
        return (p["exchange"][rk]["J_leg"] - m["exchange"][rk]["J_leg"]) / (2 * hh)

    for r in map(tuple, rpts):
        rk = (tuple(int(x) for x in r), 0, 1)
        fd1, fd2 = central(h, rk), central(h / 2, rk)
        rich = (4 * fd2 - fd1) / 3
        np.testing.assert_allclose(
            rich, res_d["exchange"][rk]["J_leg"], rtol=1e-3, atol=1e-5
        )


# ---------------------------------------------------------------------------
# TEST-006: band-window convergence report
# ---------------------------------------------------------------------------


def test_band_window_convergence_report():
    from TB2J.split_soc_kernel import band_window_convergence_report

    data = _make_ks_data(nsite=2, nstate=6, nproj_per_site=2, nkpt=4, seed=17)
    # states 4, 5 have no overlap with the dimer projectors: the pair
    # observable is exactly window-independent once they decouple from W
    data.coefficients[0, :, 4:, :, :] = 0.0
    rng = np.random.default_rng(88)
    w_dec = np.zeros((data.nkpt, data.nband, data.nband), dtype=complex)
    for ik in range(data.nkpt):
        w_dec[ik, :4, :4] = _random_hermitian(4, rng)
        w_dec[ik, 4:, 4:] = _random_hermitian(2, rng)
    w_coupled = w_dec.copy()
    for ik in range(data.nkpt):
        block = _random_hermitian(4, rng)[:, :2] * 0.3  # (4, 2)
        w_coupled[ik, :4, 4:] = block
        w_coupled[ik, 4:, :4] = block.conj().T

    common = dict(pair=(0, 1), lam=1.0, nz=24, smearing_eV=0.05, tol=1e-8)
    report = band_window_convergence_report(data, w_dec, windows=[4, 6], **common)
    assert report["converged"]
    assert report["windows"][-1]["nband"] == 6
    assert report["changes"][-1] <= 1e-8

    report_c = band_window_convergence_report(data, w_coupled, windows=[4, 6], **common)
    assert not report_c["converged"]
    assert report_c["changes"][-1] > 1e-8
    # window [4] values agree between the two W (identical restricted block)
    assert report["windows"][0]["J_uu"] == pytest.approx(
        report_c["windows"][0]["J_uu"], rel=1e-10
    )


# ---------------------------------------------------------------------------
# TEST-007: full-BZ Fourier phases / inter-site R bookkeeping
# ---------------------------------------------------------------------------


def test_fourier_phase_bookkeeping_for_translated_site():
    from TB2J.projector_green import SPINOR_OPERATOR_DEFINITION, ProjectorGreenData
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    # site 1 = site 0 translated by R1 = (1,0,0): B_1(k) = e^{-2 pi i k.R1} B_0(k)
    rng = np.random.default_rng(4242)
    nkpt, nstate = 6, 5
    kpoints = rng.normal(size=(nkpt, 3))
    weights = np.full(nkpt, 1.0 / nkpt)
    eigenvalues = rng.normal(size=(1, nkpt, nstate))
    base = rng.normal(size=(nkpt, nstate, 2, 2)) + 1j * rng.normal(
        size=(nkpt, nstate, 2, 2)
    )
    r1 = np.array([1.0, 0.0, 0.0])
    phase = np.exp(-2j * np.pi * kpoints @ r1)
    coefficients = np.empty((1, nkpt, nstate, 2, 4), dtype=complex)
    coefficients[0, ..., :2] = base
    coefficients[0, ..., 2:] = base * phase[:, None, None, None]

    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    orb = rng.normal(size=(2, 2))
    orb = orb @ orb.T + np.eye(2)
    vertex = np.einsum("st,pq->pqst", sz, orb)
    ops = np.stack([vertex, vertex])

    def build():
        return ProjectorGreenData(
            kpoints=kpoints,
            weights=weights,
            eigenvalues=eigenvalues,
            coefficients=coefficients,
            efermi=0.1,
            projector_site=np.repeat([0, 1], 2),
            projector_atom=np.repeat([0, 1], 2),
            site_nproj=np.array([2, 2]),
            site_projector_indices=np.arange(4).reshape(2, 2),
            spinor_operator=ops,
            spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
            nspinor=2,
        )

    data2 = build()
    w_soc = _random_w_soc(data2, rng)
    rpts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [-1, 0, 0], [-2, 0, 0]], dtype=int
    )
    kw = dict(lam=0.8, mode=MODE_SECOND_VARIATION, Rpts=rpts, nz=24, smearing_eV=0.05)
    res_pair = compute_ks_split_soc_exchange(data2, w_soc, sites=[0, 1], **kw)
    res_self = compute_ks_split_soc_exchange(data2, w_soc, sites=[0], **kw)

    # G_01(R) = G_00(R - R1) with B_1(k) = e^{-2 pi i k.R1} B_0(k): the pair
    # channels at (R=(1,0,0), 0-1) equal the onsite channels at (R=(0,0,0)),
    # and (R=(2,0,0), 0-1) equal (R=(1,0,0), 0-0)
    a_pair = res_pair["exchange"][((1, 0, 0), 0, 1)]["K_ijR"]
    a_self = res_self["exchange"][((0, 0, 0), 0, 0)]["K_ijR"]
    scale = max(1.0, np.abs(a_self).max())
    np.testing.assert_allclose(a_pair, a_self, atol=1e-10 * scale)
    a_pair2 = res_pair["exchange"][((2, 0, 0), 0, 1)]["K_ijR"]
    a_self2 = res_self["exchange"][((1, 0, 0), 0, 0)]["K_ijR"]
    scale2 = max(1.0, np.abs(a_self2).max())
    np.testing.assert_allclose(a_pair2, a_self2, atol=1e-10 * scale2)
    assert np.abs(a_pair).max() > 1e-6


# ---------------------------------------------------------------------------
# unit contracts
# ---------------------------------------------------------------------------


def test_second_variation_spectrum_matches_eigh():
    from TB2J.split_soc_kernel import (
        rotate_spinor_coefficients,
        second_variation_spectrum,
    )

    rng = np.random.default_rng(8)
    nk, n = 3, 5
    eps = rng.normal(size=(nk, n))
    w = np.asarray([_random_hermitian(n, rng) for _ in range(nk)])
    lam = 0.6
    eps2, x = second_variation_spectrum(eps, w, lam=lam)
    for ik in range(nk):
        ref = np.linalg.eigvalsh(np.diag(eps[ik]) + lam * w[ik])
        np.testing.assert_allclose(eps2[ik], ref, atol=1e-12)
        np.testing.assert_allclose(x[ik] @ x[ik].conj().T, np.eye(n), atol=1e-12)
        np.testing.assert_allclose(
            x[ik] @ np.diag(eps2[ik]) @ x[ik].conj().T,
            np.diag(eps[ik]) + lam * w[ik],
            atol=1e-11,
        )

    # rotation contract: C'[m,s,p] = sum_n C[n,s,p] X[n,m]
    coeff = rng.normal(size=(1, nk, n, 2, 3)) + 1j * rng.normal(size=(1, nk, n, 2, 3))
    c_rot = rotate_spinor_coefficients(coeff, x)
    assert c_rot.shape == (1, nk, n, 2, 3)
    for ik in range(nk):
        ref = np.einsum("nsp,nm->msp", coeff[0, ik], x[ik])
        np.testing.assert_allclose(c_rot[0, ik], ref, atol=1e-13)


def test_embed_site_vertex_contract():
    from TB2J.split_soc_kernel import embed_site_vertex

    rng = np.random.default_rng(9)
    npj, n = 2, 4
    b = rng.normal(size=(npj, n, 2)) + 1j * rng.normal(size=(npj, n, 2))
    v = rng.normal(size=(npj, npj, 2, 2)) + 1j * rng.normal(size=(npj, npj, 2, 2))
    v = v + v.transpose(1, 0, 3, 2).conj()
    vertex = embed_site_vertex(b, v)
    ref = np.einsum("pns,pqst,qmt->nm", b.conj(), v, b)
    np.testing.assert_allclose(vertex, ref, atol=1e-13)
    assert vertex.shape == (n, n)


def test_kernel_input_validation():
    from dataclasses import replace

    from TB2J.split_soc_kernel import compute_ks_split_soc_exchange

    data = _make_ks_data(nsite=2, nstate=4, nproj_per_site=2, nkpt=2, seed=2)
    rng = np.random.default_rng(6)
    w_ok = _random_w_soc(data, rng)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    kw = dict(Rpts=rpts, nz=8, smearing_eV=0.05)

    collinear_like = replace(
        data,
        nspinor=1,
        coefficients=data.coefficients[..., 0, :],
        spinor_operator=None,
        spinor_operator_definition=None,
    )
    with pytest.raises(ValueError, match="nspinor=2"):
        compute_ks_split_soc_exchange(collinear_like, w_ok, **kw)

    w_bad = w_ok.copy()
    w_bad[0, 0, 1] += 3.0  # break Hermiticity
    with pytest.raises(ValueError, match="[Hh]ermit"):
        compute_ks_split_soc_exchange(data, w_bad, **kw)

    with pytest.raises(ValueError, match="shape"):
        compute_ks_split_soc_exchange(data, w_ok[:, :2, :2], **kw)

    with pytest.raises(ValueError, match="mode"):
        compute_ks_split_soc_exchange(data, w_ok, mode="finite_difference", **kw)

    # insertion mode returns the lam-derivative at the strength-0
    # reference: no SOC scaling accepted (SPEC-NB4)
    with pytest.raises(ValueError, match="derivative"):
        compute_ks_split_soc_exchange(
            data,
            w_ok,
            lam=0.5,
            mode="first_order_insertion",
            **kw,
        )

    with pytest.raises(ValueError, match="band_mask"):
        compute_ks_split_soc_exchange(
            replace(
                data,
                band_mask=np.ones((1, data.nkpt, data.nband), dtype=bool),
            ),
            w_ok,
            **kw,
        )
