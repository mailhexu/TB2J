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
    spinor_operator = _random_hermitian(2, rng, 0.5)
    ops = np.zeros((nsite, nproj_per_site, nproj_per_site, 2, 2), dtype=complex)
    for site in range(nsite):
        orbital = rng.normal(size=(nproj_per_site, nproj_per_site))
        orbital = orbital + orbital.T
        ops[site] = np.einsum("st,pq->pqst", spinor_operator, orbital)
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


def _pair_channels(di, gij, dj, gji):
    """Independent ExchangeNCL channel matrix A^{uv} from full blocks.

    A^{uv} = Tr[D_i kron(sig_u, T^u_ij) D_j kron(sig_v, T^v_ji)] / pi with
    T^u the Pauli components of the spinor Green block (u, v in 0,x,y,z).
    """
    di_d = _dense_spinor_block(di)
    dj_d = _dense_spinor_block(dj)
    t_ij = [0.5 * np.einsum("pqst,st->pq", gij, s) for s in PAULI]
    t_ji = [0.5 * np.einsum("pqst,st->pq", gji, s) for s in PAULI]
    a = np.empty((4, 4), dtype=complex)
    for u in range(4):
        gu = np.kron(PAULI[u], t_ij[u])
        for v in range(4):
            gv = np.kron(PAULI[v], t_ji[v])
            a[u, v] = np.trace(di_d @ gu @ dj_d @ gv) / np.pi
    return a


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

    # collinear kernel reference: Im integral Tr[Delta G_up Delta G_down]/(4pi)
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
    ops_col = {0: np.eye(2) * delta_i, 1: np.eye(2) * delta_j}
    contour = CFR(nz=24, T=0.05 / kB)
    for (r, iatom, jatom), entry in result["exchange"].items():
        if iatom == jatom and r == (0, 0, 0):
            continue
        vals = []
        for energy in contour.path:
            gup = cg.get_GR(rpts, energy, ispin=0)
            gdn = cg.get_GR(rpts, energy, ispin=1)
            ir = [i for i, rr in enumerate(map(tuple, rpts)) if tuple(rr) == tuple(r)][
                0
            ]
            irm = [
                i
                for i, rr in enumerate(map(tuple, rpts))
                if tuple(rr) == tuple(-np.array(r))
            ][0]
            gu = cg.get_site_block(gup[ir], iatom, jatom)
            hd = cg.get_site_block(gdn[irm], jatom, iatom)
            vals.append(np.trace(ops_col[iatom] @ gu @ ops_col[jatom] @ hd))
        ref = np.imag(contour.integrate_values(np.asarray(vals))) / (4.0 * np.pi)
        assert entry["Jiso"] == pytest.approx(ref, rel=1e-9, abs=1e-12), (
            r,
            iatom,
            jatom,
        )
        assert np.linalg.norm(entry["dmi"]) < 1e-9


# ---------------------------------------------------------------------------
# TEST-003: rectangular (non-invertible) B_a accepted; production replay
# equals the independent band-space resolvent end-to-end
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

    # independent band-space reference through the pinned channel mapping
    contour = CFR(nz=24, T=0.05 / kB)
    npj = 2
    signs = {}
    for site in range(2):
        block = data.spinor_operator[site]
        signs[site] = (
            1.0
            if float(np.real(np.trace(block[:, :, 0, 0] - block[:, :, 1, 1]))) >= 0
            else -1.0
        )

    def g_site_blocks(z, rvec):
        """Fourier-transformed site blocks G(R) at relative energy z: dict
        (i, j) -> (npj, npj, 2, 2), from direct band-space inversion."""
        per_k = np.empty((data.nkpt, 4, 4, 2, 2), dtype=complex)
        for ik in range(data.nkpt):
            g = _band_G(z, data.efermi, data.eigenvalues[0, ik], w_soc[ik], lam)
            ci = data.coefficients[0, ik]
            # G[p,q,s,t] = sum_nm C[n,s,p] G[n,m] C*[m,t,q]  (<p s|G|q t>)
            per_k[ik] = np.einsum("nsp,nm,mtq->pqst", ci, g, ci.conj())
        full = _fourier(per_k, rvec, data.kpoints, data.weights)
        return {
            (i, j): _site_block(full, i, j, npj) for i in range(2) for j in range(2)
        }

    for r in map(tuple, rpts):
        r = tuple(int(x) for x in r)
        rm = tuple(-x for x in r)
        a_int = {
            (i, j): np.zeros((4, 4), dtype=complex) for i in range(2) for j in range(2)
        }
        a_int_m = {
            (i, j): np.zeros((4, 4), dtype=complex) for i in range(2) for j in range(2)
        }
        for ie, z in enumerate(contour.path):
            g_r = g_site_blocks(z, r)
            g_rm = g_site_blocks(z, rm)
            for i in range(2):
                for j in range(2):
                    a_int[(i, j)][...] += (
                        _pair_channels(
                            data.spinor_operator[i],
                            g_r[(i, j)],
                            data.spinor_operator[j],
                            g_rm[(j, i)],
                        )
                        * contour.weights[ie]
                    )
                    a_int_m[(i, j)][...] += (
                        _pair_channels(
                            data.spinor_operator[i],
                            g_rm[(i, j)],
                            data.spinor_operator[j],
                            g_r[(j, i)],
                        )
                        * contour.weights[ie]
                    )
        # CFR.integrate_values applies the contour normalization -pi/2 on
        # top of the quadrature weights; mirror it here
        a_int = {key: val * -np.pi / 2 for key, val in a_int.items()}
        a_int_m = {key: val * -np.pi / 2 for key, val in a_int_m.items()}
        for i in range(2):
            for j in range(2):
                sgn = signs[i] * signs[j]
                ai, am = a_int[(i, j)], a_int_m[(j, i)]
                jiso_ref = (
                    float(np.imag(ai[0, 0] - ai[1, 1] - ai[2, 2] - ai[3, 3]))
                    / 8.0
                    * sgn
                )
                d_ref = np.array(
                    [
                        float(np.real(ai[0, p + 1] - ai[p + 1, 0])) / 8.0 * sgn
                        for p in range(3)
                    ]
                )
                jani_ref = np.asarray(
                    [
                        [
                            float(np.imag(ai[a + 1, b + 1] + am[a + 1, b + 1]))
                            / 8.0
                            * sgn
                            for b in range(3)
                        ]
                        for a in range(3)
                    ]
                )
                entry = result["exchange"][(r, i, j)]
                scale = max(1.0, abs(jiso_ref), abs(entry["Jiso"]))
                assert entry["Jiso"] == pytest.approx(jiso_ref, abs=1e-9 * scale), (
                    r,
                    i,
                    j,
                    entry["Jiso"],
                    jiso_ref,
                )
                assert np.allclose(entry["dmi"], d_ref, atol=1e-9 * scale)
                assert np.allclose(entry["jani"], jani_ref, atol=1e-9 * scale)


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
    d_all = res_all["exchange"][key]["dmi"]
    d_mag = res_mag["exchange"][key]["dmi"]
    scale = max(1.0, np.linalg.norm(d_all), np.linalg.norm(d_mag))
    assert np.linalg.norm(d_all - d_mag) > 1e-3 * scale

    # magnetic-only vertex gating: requesting the magnetic subset with all
    # site vertices present changes nothing on the magnetic pairs
    res_all_sites = compute_ks_split_soc_exchange(data, w_all, sites=[0, 1, 2], **kw)
    for rk, entry in res_all["exchange"].items():
        other = res_all_sites["exchange"][rk]
        assert entry["Jiso"] == pytest.approx(other["Jiso"], abs=1e-12)
        np.testing.assert_allclose(entry["dmi"], other["dmi"], atol=1e-12)
        np.testing.assert_allclose(entry["jani"], other["jani"], atol=1e-12)
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

    def central(hh):
        p = compute_ks_split_soc_exchange(
            data, w_soc, lam=hh, mode=MODE_SECOND_VARIATION, **kw
        )
        m = compute_ks_split_soc_exchange(
            data, w_soc, lam=-hh, mode=MODE_SECOND_VARIATION, **kw
        )
        return {
            "Jiso": (p["exchange"][rk]["Jiso"] - m["exchange"][rk]["Jiso"]) / (2 * hh),
            "dmi": (p["exchange"][rk]["dmi"] - m["exchange"][rk]["dmi"]) / (2 * hh),
            "jani": (p["exchange"][rk]["jani"] - m["exchange"][rk]["jani"]) / (2 * hh),
        }

    for r in map(tuple, rpts):
        rk = (tuple(int(x) for x in r), 0, 1)
        fd1, fd2 = central(h), central(h / 2)
        rich = {q: (4 * np.asarray(fd2[q]) - np.asarray(fd1[q])) / 3 for q in fd1}
        an = res_d["exchange"][rk]
        assert rich["Jiso"] == pytest.approx(an["Jiso"], rel=1e-3, abs=1e-5), (
            rk,
            rich["Jiso"],
            an["Jiso"],
        )
        np.testing.assert_allclose(rich["dmi"], an["dmi"], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(rich["jani"], an["jani"], rtol=1e-3, atol=1e-5)


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
    assert report["windows"][0]["Jiso"] == pytest.approx(
        report_c["windows"][0]["Jiso"], rel=1e-10
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
    orb = orb + orb.T
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
    a_pair = res_pair["exchange"][((1, 0, 0), 0, 1)]["A_ijR"]
    a_self = res_self["exchange"][((0, 0, 0), 0, 0)]["A_ijR"]
    scale = max(1.0, np.abs(a_self).max())
    np.testing.assert_allclose(a_pair, a_self, atol=1e-10 * scale)
    a_pair2 = res_pair["exchange"][((2, 0, 0), 0, 1)]["A_ijR"]
    a_self2 = res_self["exchange"][((1, 0, 0), 0, 0)]["A_ijR"]
    scale2 = max(1.0, np.abs(a_self2).max())
    np.testing.assert_allclose(a_pair2, a_self2, atol=1e-10 * scale2)
    assert np.abs(a_pair).max() > 1e-6


# ---------------------------------------------------------------------------
# TEST-008: FR-050 provenance metadata
# ---------------------------------------------------------------------------


def test_provenance_metadata_emitted():
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    data = _make_ks_data(nsite=2, nstate=4, nproj_per_site=2, nkpt=3, seed=3)
    rng = np.random.default_rng(7)
    w_soc = _random_w_soc(data, rng)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    res = compute_ks_split_soc_exchange(
        data,
        w_soc,
        lam=0.5,
        mode=MODE_SECOND_VARIATION,
        Rpts=rpts,
        nz=12,
        smearing_eV=0.05,
        metadata={
            "strength0_reference": {"code": "synthetic", "leg": "collinear no-SOC"}
        },
    )
    meta = res["metadata"]
    assert meta["schema"] == "tb2j.split_soc_ks_provenance/1.0"
    assert meta["mode"] == "second_variation"
    assert meta["lambda"] == 0.5
    assert meta["strength0_reference"]["code"] == "synthetic"
    assert meta["soc_operator"]["units"] == "eV"
    assert meta["soc_operator"]["coverage"] == "all_atoms"
    assert meta["band_window"]["nband"] == 4
    assert meta["pauli_order"] == "x,y,z"
    assert "spinaxis" in meta["frame"]
    assert meta["vertices"]["magnetic_only"] is True
    assert meta["vertices"]["sites"] == [0, 1]
    assert "merge_mode" in meta


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
