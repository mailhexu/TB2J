"""Story-003 tests: GPAW split-SOC soc-leg adapter (ADR-4, psi gauge).

Layers (Kosmic Phase 2):
- gauge primitives ported from story-001 ``docs/sympy/split_soc_gauge.py``
  (verbatim ``add_soc`` C^dag H C chain, P^chi = C P^psi, T_lattice = O T_leg O^T);
- real old-API collinear Fe leg: full-BZ handling, W_SO^K assembly Hermiticity,
  second-variational spectrum reproduction, y-leg complex-rotation oracle;
- lambda=0 leg == collinear data (projection level and kernel level);
- refusals: ``projected=True``, already-SOC double counting, new-API calculators.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.units import Ha

gpaw = pytest.importorskip("gpaw")

from TB2J.interfaces.gpaw_spinor_split_soc import (  # noqa: E402
    apply_frame_rotation,
    axis_of,
    c_gpaw,
    collect_soc_leg,
    collect_three_legs,
    o_from_c,
    pack_sigma_dot_l,
    rotate_to_leg_basis,
    soc_leg_to_projector_green_data,
)
from TB2J.projector_green import (  # noqa: E402
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
)

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA = (SX, SY, SZ)

CANONICAL_LEGS = {"x": (90.0, 0.0), "y": (90.0, 90.0), "z": (0.0, 0.0)}

_RNG = np.random.default_rng(20260928)


def _random_hermitian(*shape):
    a = _RNG.normal(size=shape) + 1j * _RNG.normal(size=shape)
    return a + np.conj(np.swapaxes(a, -1, -2))


# ---------------------------------------------------------------------------
# session fixtures: one collinear SCF, cached read-only legs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fe_calc():
    """Collinear spin-polarized old-API bcc Fe (2-atom cell).

    Uses ``TB2J_GPAW_SPLIT_SOC_GPW`` when set (any legacy-loadable
    collinear .gpw), else runs a small in-test SCF (symmetry off so the
    stored k-set is the complete BZ).
    """
    import os

    gpw_path = os.environ.get("TB2J_GPAW_SPLIT_SOC_GPW")
    if gpw_path:
        from gpaw import GPAW

        if not Path(gpw_path).is_file():
            pytest.skip(f"GPAW split-SOC fixture not found: {gpw_path}")
        calc = GPAW(gpw_path, legacy_gpaw=True)
        atoms = calc.get_atoms()
        atoms.calc = calc
        return calc

    from ase import Atoms
    from gpaw import GPAW, PW
    from gpaw.mixer import Mixer

    a = 2.834
    atoms = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [a / 2] * 3],
        cell=[a] * 3,
        pbc=True,
    )
    atoms.set_initial_magnetic_moments([3.0, 3.0])
    calc = GPAW(
        mode=PW(300),
        xc="PBE",
        kpts=(2, 2, 2),
        symmetry="off",
        txt=None,
        occupations={"name": "fermi-dirac", "width": 0.1},
        convergence={"energy": 1e-3, "density": 5e-3},
        mixer=Mixer(0.05, 5, 100),
        maxiter=250,
        legacy_gpaw=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc


_LEG_CACHE: dict[tuple, object] = {}


def _leg(calc, theta, phi, scale=1.0):
    key = (id(calc), theta, phi, scale)
    if key not in _LEG_CACHE:
        _LEG_CACHE[key] = collect_soc_leg(calc, theta=theta, phi=phi, scale=scale)
    return _LEG_CACHE[key]


def _interleaved_calc_eigenvalues(calc):
    kd = calc.wfs.kd
    nb = calc.get_number_of_bands()
    out = np.empty((kd.nbzkpts, 2 * nb))
    for K in range(kd.nbzkpts):
        k = int(kd.bz2ibz_k[K])
        out[K, 0::2] = calc.get_eigenvalues(kpt=k, spin=0)
        out[K, 1::2] = calc.get_eigenvalues(kpt=k, spin=1)
    return out


# ---------------------------------------------------------------------------
# gauge primitives (story-001 ports, toy-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta,phi", [(90.0, 0.0), (90.0, 90.0), (0.0, 0.0)])
def test_c_matrix_unitary_and_axis_map_degrees(theta, phi):
    th, ph = np.deg2rad(theta), np.deg2rad(phi)
    c_mat = c_gpaw(th, ph)
    assert np.abs(c_mat.conj().T @ c_mat - np.eye(2)).max() < 1e-14
    n_vec = axis_of(th, ph)
    want = sum(n_vec[w] * SIGMA[w] for w in range(3))
    assert np.abs(c_mat @ SZ @ c_mat.conj().T - want).max() < 1e-14


def test_o_matrix_so3_and_ez_to_n():
    ez = np.array([0.0, 0.0, 1.0])
    for theta, phi in ((0.9, 1.3), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)):
        o_mat = o_from_c(c_gpaw(theta, phi))
        assert np.abs(o_mat @ o_mat.T - np.eye(3)).max() < 1e-14
        assert abs(np.linalg.det(o_mat) - 1.0) < 1e-14
        assert np.abs(o_mat @ ez - axis_of(theta, phi)).max() < 1e-14


def test_verbatim_chain_is_c_dag_h_c():
    """The verbatim two-tensordot add_soc chain equals plain C^dag H C."""
    ni = 3
    l_vec = [_random_hermitian(ni, ni) for _ in range(3)]
    h_packed = pack_sigma_dot_l(np.asarray(l_vec))
    assert h_packed.shape == (2, 2, ni, ni)
    for theta, phi in ((0.9, 1.3), (np.pi / 2, np.pi / 2)):
        c_mat = c_gpaw(theta, phi)
        got = rotate_to_leg_basis(h_packed, c_mat)
        want = np.empty_like(h_packed)
        wrong = np.empty_like(h_packed)
        for i in range(ni):
            for j in range(ni):
                want[:, :, i, j] = c_mat.conj().T @ h_packed[:, :, i, j] @ c_mat
                wrong[:, :, i, j] = c_mat.conj().T @ h_packed[:, :, i, j] @ c_mat.T
        assert np.abs(got - want).max() < 1e-13
        # negative control: C^dag H C^T is NOT what the chain computes
        assert np.abs(got - wrong).max() > 0.1
        # the rotated leg operator stays Hermitian
        got_flat = got.transpose(2, 0, 3, 1).reshape(2 * ni, 2 * ni)
        assert np.abs(got_flat - got_flat.conj().T).max() < 1e-13


def test_projection_gauge_identity_pchi_equals_c_ppsi():
    """P^chi = C P^psi on the spinor index; per-band norms preserved."""
    nb, nproj = 5, 3
    p_psi = _RNG.normal(size=(nb, nproj, 2)) + 1j * _RNG.normal(size=(nb, nproj, 2))
    c_mat = c_gpaw(np.pi / 2, np.pi / 2)  # y leg: complex mixing of S_z slots
    # P^chi = C P^psi on the spinor slot index
    p_chi = np.einsum("ab,mpa->mpb", c_mat, p_psi)
    norms_psi = np.abs(p_psi**2).sum(axis=(1, 2))
    norms_chi = np.abs(p_chi**2).sum(axis=(1, 2))
    assert np.abs(norms_psi - norms_chi).max() < 1e-13
    # y-leg C mixes the two S_z slots: no diagonal sign map reproduces it
    diag_map = np.stack([p_psi[:, :, 0] * c_mat[0, 0], p_psi[:, :, 1] * c_mat[1, 1]])
    diag_map = diag_map.transpose(1, 2, 0)
    assert np.abs(p_chi - diag_map).max() > 0.1


def test_tensor_gauge_identity_t_lattice_o_tleg_ot():
    """T_lattice = O T_leg O^T with O e_z = n (ADR-4 corrected direction)."""
    t_leg = _RNG.normal(size=(3, 3))
    t_leg = 0.5 * (t_leg + t_leg.T)
    o_mat = o_from_c(c_gpaw(0.7, -0.4))
    t_chi = apply_frame_rotation(t_leg, o_mat)
    want = np.einsum("wa,ab,ub->wu", o_mat, t_leg, o_mat)
    assert np.abs(t_chi - want).max() < 1e-14
    back = apply_frame_rotation(t_chi, o_mat.T)
    assert np.abs(back - t_leg).max() < 1e-14


# ---------------------------------------------------------------------------
# real-calc leg collection
# ---------------------------------------------------------------------------


def test_collect_soc_leg_shapes_and_full_bz(fe_calc):
    kd = fe_calc.wfs.kd
    leg = _leg(fe_calc, 0.0, 0.0)
    nk = kd.nbzkpts
    nb = fe_calc.get_number_of_bands()
    nproj = int(leg.site_nproj.sum())
    assert leg.kpoints.shape == (nk, 3)
    np.testing.assert_allclose(leg.kpoints, np.asarray(kd.bzk_kc), atol=1e-12)
    np.testing.assert_allclose(leg.weights, np.full(nk, 1.0 / nk), atol=1e-15)
    assert leg.eigenvalues_soc.shape == (nk, 2 * nb)
    assert leg.occupations_soc.shape == (nk, 2 * nb)
    assert leg.eigenvalues_strength0.shape == (nk, 2 * nb)
    assert leg.v_mn.shape == (nk, 2 * nb, 2 * nb)
    assert leg.p_amj_soc.shape == (nk, 2 * nb, nproj, 2)
    assert leg.w_soc.shape == (nk, 2 * nb, 2 * nb)
    ni0 = int(leg.site_nproj[0])
    assert leg.w_soc_atom.shape == (len(leg.site_nproj), ni0, ni0, 2, 2)
    assert leg.metadata["units"]["w_soc"] == "eV"
    assert leg.metadata["units"]["eigenvalues"] == "eV"
    # strength-0 fermi is the collinear calc fermi; the leg has its own
    assert leg.efermi == pytest.approx(fe_calc.get_fermi_level(), rel=1e-9)
    assert np.isfinite(leg.efermi_soc)
    # leg eigenvector matrices are unitary
    for K in range(nk):
        v = leg.v_mn[K]
        assert np.abs(v.conj().T @ v - np.eye(2 * nb)).max() < 1e-10


def test_w_soc_second_variation_spectrum_reproduction(fe_calc):
    leg = _leg(fe_calc, 90.0, 0.0)
    scale = max(1.0, float(np.abs(leg.w_soc).max()))
    herm = np.abs(leg.w_soc - np.conj(np.swapaxes(leg.w_soc, 1, 2))).max()
    assert herm < 1e-8 * scale
    ham = leg.w_soc.copy()
    idx = np.arange(leg.w_soc.shape[1])
    ham[:, idx, idx] += leg.eigenvalues_strength0
    eps2, _ = np.linalg.eigh(ham)
    assert np.abs(eps2 - leg.eigenvalues_soc).max() < 1e-8


def test_w_soc_atom_y_leg_verbatim_rotation_oracle(fe_calc):
    """y-leg W_SO blocks from the adapter == independently written chain."""
    from gpaw.spinorbit import soc, soc_eigenstates

    theta, phi = 90.0, 90.0
    leg = _leg(fe_calc, theta, phi)
    dvl = soc(
        fe_calc.wfs.setups[0],
        fe_calc.hamiltonian.xc,
        fe_calc.density.D_asp[0],
        False,
    )
    th, ph = np.deg2rad(theta), np.deg2rad(phi)
    c_inline = np.array(
        [
            [
                np.cos(th / 2) * np.exp(-1j * ph / 2),
                -np.sin(th / 2) * np.exp(-1j * ph / 2),
            ],
            [
                np.sin(th / 2) * np.exp(1j * ph / 2),
                np.cos(th / 2) * np.exp(1j * ph / 2),
            ],
        ]
    )
    h_ssii = np.zeros((2, 2) + dvl.shape[1:], complex)
    h_ssii[0, 0] = dvl[2]
    h_ssii[0, 1] = dvl[0] - 1j * dvl[1]
    h_ssii[1, 0] = dvl[0] + 1j * dvl[1]
    h_ssii[1, 1] = -dvl[2]
    h_ssii *= Ha  # Hartree -> eV, as GPAW add_soc does
    ni = dvl.shape[1]
    want = np.empty_like(h_ssii)
    for i in range(ni):
        for j in range(ni):
            want[:, :, i, j] = c_inline.conj().T @ h_ssii[:, :, i, j] @ c_inline
    got = leg.w_soc_atom[0, :ni, :ni]
    assert np.abs(got - want.transpose(2, 3, 0, 1)).max() < 1e-10

    # adapter eigenvalues equal a direct GPAW call at the same angles
    bzw = soc_eigenstates(fe_calc, theta=theta, phi=phi)
    assert np.abs(bzw.eigenvalues() - leg.eigenvalues_soc).max() < 1e-10


def test_lambda_zero_leg_equals_collinear_projections(fe_calc):
    leg = _leg(fe_calc, 90.0, 0.0, scale=0.0)
    eig_pre = _interleaved_calc_eigenvalues(fe_calc)
    assert np.abs(leg.w_soc).max() == 0.0
    assert np.abs(leg.eigenvalues_soc - np.sort(eig_pre, axis=1)).max() < 1e-10
    assert np.abs(leg.eigenvalues_strength0 - eig_pre).max() < 1e-12
    assert leg.efermi_soc == pytest.approx(fe_calc.get_fermi_level(), rel=1e-6)

    from gpaw.spinorbit import get_both_spins

    kpt_qs = get_both_spins(fe_calc.wfs)
    kd = fe_calc.wfs.kd
    nb = fe_calc.get_number_of_bands()
    data = soc_leg_to_projector_green_data(leg)
    for K in range(kd.nbzkpts):
        k = int(kd.bz2ibz_k[K])
        p_up = kpt_qs[k][0].projections.collect()
        p_dn = kpt_qs[k][-1].projections.collect()
        direct = np.zeros((2 * nb, p_up.shape[1], 2), complex)
        direct[0::2, :, 0] = p_up
        direct[1::2, :, 1] = p_dn
        # strength-0 psi basis == interleaved collinear projections
        assert np.abs(data.coefficients[0, K] - direct.transpose(0, 2, 1)).max() < 1e-8


def test_lambda_zero_kernel_matches_collinear_reduction(fe_calc):
    from ase.units import kB

    from TB2J.mycfr import CFR
    from TB2J.split_soc_kernel import compute_ks_split_soc_exchange

    leg = _leg(fe_calc, 0.0, 0.0, scale=0.0)
    data = soc_leg_to_projector_green_data(leg)
    rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    result = compute_ks_split_soc_exchange(
        data,
        leg.w_soc,
        lam=0.0,
        Rpts=rpts,
        nz=24,
        smearing_eV=0.05,
        sites=[0],
    )

    # independent collinear reduction: Im Tr[Delta G_up Delta G_dn]/(4 pi)
    ni = int(leg.site_nproj[0])
    block = data.spinor_operator[0, :ni, :ni]
    delta = block[:, :, 0, 0]  # psi-gauge vertex is sigma_z-diagonal
    collinear = ProjectorGreenData(
        kpoints=leg.kpoints,
        weights=leg.weights,
        eigenvalues=np.stack(
            [leg.eigenvalues_strength0[:, 0::2], leg.eigenvalues_strength0[:, 1::2]]
        ),
        coefficients=np.stack(
            [
                data.coefficients[0][:, 0::2, 0, :],
                data.coefficients[0][:, 1::2, 1, :],
            ]
        ),
        efermi=leg.efermi,
        projector_site=data.projector_site,
        projector_atom=data.projector_atom,
        site_nproj=data.site_nproj,
        site_projector_indices=data.site_projector_indices,
    )
    cg = ProjectorGreen(collinear)
    ops_col = {0: delta}
    contour = CFR(nz=24, T=0.05 / kB)
    compared = 0
    for (r, iatom, jatom), entry in result["exchange"].items():
        if not (iatom == 0 and jatom == 0):
            continue
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
                if tuple(rr) == tuple(-np.asarray(r))
            ][0]
            gu = cg.get_site_block(gup[ir], iatom, jatom)
            hd = cg.get_site_block(gdn[irm], jatom, iatom)
            vals.append(np.trace(ops_col[iatom] @ gu @ ops_col[jatom] @ hd))
        ref = np.imag(contour.integrate_values(np.asarray(vals))) / (4.0 * np.pi)
        assert entry["Jiso"] == pytest.approx(ref, rel=1e-8, abs=1e-12), (r, iatom)
        assert np.linalg.norm(entry["dmi"]) < 1e-9
        compared += 1
    assert compared >= 2  # R=(1,0,0) and (-1,0,0) on-site-restricted pairs


# ---------------------------------------------------------------------------
# ProjectorGreenData contract and metadata
# ---------------------------------------------------------------------------


def test_leg_data_validates_and_psi_gauge_vertices(fe_calc):
    leg = _leg(fe_calc, 90.0, 90.0)
    data = soc_leg_to_projector_green_data(leg)
    assert data.nspinor == 2
    nk = leg.kpoints.shape[0]
    nb2 = 2 * leg.nbands
    nproj = int(leg.site_nproj.sum())
    assert data.coefficients.shape == (1, nk, nb2, 2, nproj)
    assert data.eigenvalues.shape == (1, nk, nb2)
    assert data.occupations.shape == (1, nk, nb2)
    assert data.spinor_operator_definition == SPINOR_OPERATOR_DEFINITION
    assert data.validate(exchange_ready=True)
    ni = int(leg.site_nproj[0])
    block = data.spinor_operator[0, :ni, :ni]
    # psi gauge: collinear exchange vertex stays sigma_z-diagonal
    assert np.abs(block[:, :, 0, 1]).max() == 0.0
    assert np.abs(block[:, :, 1, 0]).max() == 0.0
    assert np.abs(block[:, :, 0, 0] - block[:, :, 1, 1]).max() > 0.0
    np.testing.assert_allclose(block, block.transpose(1, 0, 3, 2).conj(), atol=1e-12)
    assert data.metadata["frame"]["gauge"] == "psi"
    np.testing.assert_allclose(data.metadata["frame"]["spinaxis"], leg.axis, atol=1e-12)
    # single efermi on spinor leg data
    assert data.efermi_spin is None


def test_three_legs_driver_seam(fe_calc):
    legs = collect_three_legs(fe_calc)
    assert set(legs) == {"x", "y", "z"}
    ez = np.array([0.0, 0.0, 1.0])
    for name, (theta, phi) in CANONICAL_LEGS.items():
        leg = legs[name]
        np.testing.assert_allclose(
            leg.axis, axis_of(*np.deg2rad((theta, phi))), atol=1e-14
        )
        assert np.abs(leg.rotation @ ez - leg.axis).max() < 1e-12
        scale = max(1.0, float(np.abs(leg.w_soc).max()))
        herm = np.abs(leg.w_soc - np.conj(np.swapaxes(leg.w_soc, 1, 2))).max()
        assert herm < 1e-8 * scale
        assert leg.theta == theta and leg.phi == phi
        data = soc_leg_to_projector_green_data(leg)
        assert data.validate(exchange_ready=True)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_projected_true_refused(fe_calc):
    with pytest.raises(ValueError, match="projected"):
        collect_soc_leg(fe_calc, theta=90.0, phi=90.0, projected=True)


def test_new_api_calc_refused():
    from types import SimpleNamespace

    with pytest.raises(TypeError, match="old-API|legacy"):
        collect_soc_leg(SimpleNamespace(dft=object()))
