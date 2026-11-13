"""Regression tests for the spin-wave stiffness of the magnon code.

The stiffness is the coefficient of the long-wavelength expansion of the
acoustic magnon branch,

    omega(k) = omega0 + k^T A k     (quadratic branch, meV A^2)
    omega(k) = omega0 + v . k       (linear branch, meV A)

with the magnon convention of `TB2J.magnon.magnon3`
(H = -sum_<ij> J S_i.S_j, unordered bonds once, J in meV, |S| = |magmom|/2).

The reference values are the textbook ones and are pinned symbolically in
`docs/sympy/magnon_spin_stiffness.py`.  With the module convention
(`jiso > 0` is the ferromagnet) the stiffness of a stable reference is
positive:

    FM chain            A_xx = J S a^2
    FM honeycomb        A_xx = A_yy = J S a^2 / 4, A_xy = 0
    AFM chain           linear branch, speed v = |J| S a_cell
    AFM honeycomb       linear branch, v = sqrt(3/2) |J| S a
    mixed-spin honeycomb A_xx = J S1 S2 / (2 (S1 + S2))

Non-ferromagnetic reference states go through the same extraction: a
quadratic branch returns the tensor `A`, a linear one the directional
speeds (the spiral and Neel cases below).  The branch order is read from all
three cartesian axes, so a chain that is flat along x but disperses along y
is a linear branch.

An unusable reference state is refused instead of producing a number that
looks physical: a reference whose magnon frequencies are complex (the
modulus of a complex frequency is not an energy) and a reference whose
Hessian has a negative direction that the acoustic gap does not explain
(the state is a saddle point) both raise.
"""

from pathlib import Path

import numpy as np
import pytest

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.magnon_stiffness import stiffness_sum_rule

ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = (
    ROOT_DIR
    / "tests"
    / "data"
    / "tests"
    / "3_CrI3_wannier_SOC"
    / "refs"
    / "TB2J_results"
)


def _fm_chain(S, jiso, a=1.0):
    """FM chain, one site per cell, bonds to the neighbouring cells (jiso in eV).

    `Rlist` is in lattice coordinates (the cell carries the length a).
    """
    magmom = 2 * S
    Rlist = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = np.eye(3) * jiso
    JR[1, 0, 0] = np.eye(3) * jiso
    M = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0, magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3) * a,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom]]),
    )
    return M


def _fm_honeycomb(S, jiso, a=1.0):
    """FM honeycomb, two sites per cell, graphene nearest-neighbour bonds."""
    cellv = np.array([[a, 0, 0], [a / 2, np.sqrt(3) * a / 2, 0], [0, 0, 10.0]])
    nspin = 2
    magmom = 2 * S
    Rs, JR = [], []
    n = 2
    for r1 in range(-n, n + 1):
        for r2 in range(-n, n + 1):
            Rs.append(np.array([r1, r2, 0.0]))
            JR.append(np.zeros((nspin, nspin, 3, 3)))
            for i in range(nspin):
                for jj in range(nspin):
                    if i == jj:
                        continue
                    if (i, jj) == (0, 1) and (r1, r2) in [(0, 0), (-1, 0), (0, -1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
                    if (i, jj) == (1, 0) and (r1, r2) in [(0, 0), (1, 0), (0, 1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
    M = Magnon(
        nspin=nspin,
        magmom=np.array([[0.0, 0, magmom]] * nspin),
        Rlist=np.array(Rs),
        JR=np.array(JR),
        cell=cellv,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom]] * nspin),
    )
    return M


def _multisite_honeycomb(S1, S2, jiso, a=1.0):
    """Honeycomb with inequivalent sublattice spins (ferrimagnet)."""
    M = _fm_honeycomb(S1, jiso, a)
    magmoms = np.array([[0.0, 0, 2 * S1], [0.0, 0, 2 * S2]])
    M.magmom = magmoms
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=magmoms,
    )
    return M


def _exact_band(M, k_cart):
    """Positive eigenenergies (meV) of the exact eig(gH) spectrum."""
    k_frac = np.asarray(M.cell) @ np.asarray(k_cart, dtype=float) / (2 * np.pi)
    H = M.Hq(np.atleast_2d(k_frac))[0]
    n = H.shape[-1] // 2
    g = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    return np.sort(np.abs(np.linalg.eigvals(g @ H))) * 1e3


def test_multisite_band_matches_textbook():
    """A multi-site magnet is not the single-S case with a rescaled coupling.

    Textbook for the honeycomb with sublattice spins S1, S2:
    omega_pm(k) = (3J/2)[(S1+S2) +/- sqrt((S1-S2)^2 + 4 S1 S2 |gamma|^2)].
    """
    S1, S2, jiso, a = 1.5, 2.0, 1.0e-3, 1.0
    M = _multisite_honeycomb(S1, S2, jiso, a)
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2, np.sqrt(3) * a / 2, 0.0])
    for k in (np.array([0.05, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])):
        gam = (1 + np.exp(1j * k.dot(a1)) + np.exp(1j * k.dot(a2))) / 3
        root = np.sqrt((S1 - S2) ** 2 + 4 * S1 * S2 * abs(gam) ** 2)
        target = (3 * jiso * 1e3 / 2) * np.array([S1 + S2 - root,
                                                  S1 + S2 + root])
        band = _exact_band(M, k)
        assert abs(band[0] - target[0]) < 1e-6
        assert abs(band[-1] - target[1]) < 1e-6
        # both branches are doubly degenerate
        assert abs(band[1] - band[0]) < 1e-6
        assert abs(band[2] - band[-1]) < 1e-6


def test_multisite_stiffness_is_isotropic_and_symmetric_in_the_spins():
    """A_xx = A_yy = J S1 S2 a^2 / (2 (S1+S2)), A_xy = 0."""
    jiso, a = 1.0e-3, 1.0
    for S1, S2 in ((1.5, 2.0), (2.0, 2.5), (1.0, 3.0)):
        M = _multisite_honeycomb(S1, S2, jiso, a)
        st = M.get_spin_stiffness()
        theory = jiso * 1e3 * S1 * S2 * a**2 / (2 * (S1 + S2))
        assert st.order == 2
        assert abs(st.A[0, 0] - theory) / theory < 1e-3
        assert abs(st.A[1, 1] - theory) / theory < 1e-3
        assert abs(st.A[0, 1]) < 1e-6 * theory


def test_multisite_equal_spins_reduces_to_the_single_s_case():
    """With S1 = S2 the multi-site model is the FM honeycomb."""
    S, jiso, a = 1.5, 1.0e-3, 1.0
    st = _multisite_honeycomb(S, S, jiso, a).get_spin_stiffness()
    theory = abs(S * jiso * 1e3 * a**2) / 4
    assert abs(st.A[0, 0] - theory) / theory < 1e-3


def _afm_chain(S, jiso, a=1.0, rotated=False):
    """AFM chain, two-site Neel cell (magnetic cell 2a).

    With ``rotated=True`` the chain runs along the cartesian y axis (the
    first lattice vector is rotated into y), so the branch is flat along x
    and dispersive along y.
    """
    magmom = 2 * S
    Rlist = np.array([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((3, 2, 2, 3, 3))
    JR[0, 0, 1] = np.eye(3) * jiso
    JR[0, 1, 0] = np.eye(3) * jiso
    JR[1, 1, 0] = np.eye(3) * jiso
    JR[2, 0, 1] = np.eye(3) * jiso
    if rotated:
        cell = np.array([[0, 2 * a, 0], [2 * a, 0, 0], [0, 0, 10.0]])
    else:
        cell = np.array([[2 * a, 0, 0], [0, 2 * a, 0], [0, 0, 10.0]])
    M = Magnon(
        nspin=2,
        magmom=np.array([[0.0, 0, magmom], [0.0, 0, -magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=cell,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom], [0.0, 0, -magmom]]),
    )
    return M


def _pair_reversal_broken_chain(S, jiso, dmi, a=1.0):
    """FM chain whose exchange tensors break the pair-reversal symmetry.

    For a real exchange ``JR[-R]`` is ``JR[R].T`` (the antisymmetric part
    flips with R).  Using the same tensor for both directions leaves the
    transverse block non-Hermitian, so the LSWT frequencies are not real.
    """
    magmom = 2 * S
    Rlist = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    tensor = np.eye(3) * jiso + np.array(
        [[0.0, dmi, 0.0], [-dmi, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = tensor
    JR[1, 0, 0] = tensor
    M = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0, magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([a, a, 10.0]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom]]),
    )
    return M


def test_fm_chain_stiffness():
    """A_xx = J S a^2 for the FM chain (jiso > 0 is the ferromagnet)."""
    S, jiso, a = 2.0, 1.0e-3, 1.0
    st = _fm_chain(S, jiso, a).get_spin_stiffness()
    assert st.order == 2
    # in the module's convention jiso > 0 is the ferromagnet, so the
    # stiffness of the stable reference is positive
    assert st.A[0, 0] > 0
    assert abs(st.A[0, 0] - S * jiso * 1e3 * a**2) < 1e-3
    # no exchange in the transverse directions
    assert abs(st.A[1, 1]) < 1e-8
    assert abs(st.A[2, 2]) < 1e-8
    assert abs(st.A[0, 1]) < 1e-8


def test_fm_chain_stiffness_scales_with_lattice_constant():
    """The stiffness scales with a^2."""
    S, jiso = 2.0, 1.0e-3
    a1 = _fm_chain(S, jiso, 1.0).get_spin_stiffness().A[0, 0]
    a2 = _fm_chain(S, jiso, 2.0).get_spin_stiffness().A[0, 0]
    assert abs(a2 / a1 - 4.0) < 1e-3


def test_fm_honeycomb_stiffness_is_isotropic():
    """A_xx = A_yy = |J S a^2| / 4 and A_xy = 0 for the FM honeycomb."""
    S, jiso, a = 1.5, 1.0e-3, 1.0
    M = _fm_honeycomb(S, jiso, a)
    # in the module's convention (H = -sum_<ij> J S_i.S_j) jiso > 0 is the
    # ferromagnet, and then the reference state is stable.
    assert np.linalg.eigvalsh(M.Hq(np.zeros((1, 3)))[0]).min() > -1e-12
    st = M.get_spin_stiffness()
    theory = abs(S * jiso * 1e3 * a**2) / 4
    assert st.order == 2
    assert abs(abs(st.A[0, 0]) - theory) / theory < 1e-3
    assert abs(abs(st.A[1, 1]) - theory) / theory < 1e-3
    assert abs(st.A[0, 1]) < 1e-3 * theory


def test_afm_chain_reports_linear_branch_and_speed():
    """The AFM chain has no quadratic stiffness but a finite spin-wave speed."""
    S, jiso, a = 1.5, -1.0e-3, 1.0
    st = _afm_chain(S, jiso, a).get_spin_stiffness()
    theory = abs(S * jiso * 1e3 * 2 * a)  # |J| S a_cell, a_cell = 2a
    assert st.order == 1
    assert st.A is None
    # velocity holds one directional speed per cartesian axis
    assert abs(st.velocity[0] - theory) / theory < 1e-3


def test_rotated_afm_chain_reports_linear_branch():
    """A chain that is flat along x but disperses along y is a linear branch.

    Reading the branch order from the x axis alone would call this branch
    quadratic and fit a meaningless tensor to a linear dispersion.
    """
    S, jiso, a = 1.5, -1.0e-3, 1.0
    st = _afm_chain(S, jiso, a, rotated=True).get_spin_stiffness()
    theory = abs(S * jiso * 1e3 * 2 * a)  # |J| S a_cell, a_cell = 2a
    assert st.order == 1
    assert st.A is None
    # the speed sits on the direction the chain runs along
    assert abs(st.velocity[1] - theory) / theory < 1e-3
    # the flat directions carry no speed
    assert st.velocity[0] == 0.0
    assert st.velocity[2] == 0.0


def test_unstable_reference_state_is_refused():
    """A Neel reference with ferromagnetic exchange is a saddle point.

    The stiffness of such a reference is not a physical number, so the
    extraction refuses instead of reporting one.
    """
    with pytest.raises(ValueError, match="energetically unstable"):
        _afm_chain(1.5, 1.0e-3).get_spin_stiffness()


def test_unstable_ferromagnetic_reference_is_refused_away_from_gamma():
    """Antiferromagnetic exchange destabilizes the aligned chain at finite k."""
    with pytest.raises(ValueError, match="energetically unstable"):
        _fm_chain(1.5, -1.0e-3).get_spin_stiffness()


def test_complex_spectrum_is_refused():
    """A reference whose magnon frequencies are complex is refused.

    The modulus of a complex frequency is not a magnon energy, so the
    extraction must not silently report it as one.
    """
    M = _pair_reversal_broken_chain(2.0, 1.0e-3, 0.5e-3)
    with pytest.raises(ValueError, match="dynamically unstable"):
        M.get_spin_stiffness()


def _afm_honeycomb(S, jiso, a=2.0):
    """Neel honeycomb: two sites/cell with opposite moments, 3 NN bonds."""
    cellv = np.array([[a, 0, 0], [a / 2, np.sqrt(3) * a / 2, 0], [0, 0, 10.0]])
    nspin, n, magmom = 2, 2, 2 * S
    Rs, JR = [], []
    for r1 in range(-n, n + 1):
        for r2 in range(-n, n + 1):
            Rs.append(np.array([r1, r2, 0.0]))
            JR.append(np.zeros((nspin, nspin, 3, 3)))
            for i in range(nspin):
                for jj in range(nspin):
                    if i == jj:
                        continue
                    if (i, jj) == (0, 1) and (r1, r2) in [(0, 0), (-1, 0), (0, -1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
                    if (i, jj) == (1, 0) and (r1, r2) in [(0, 0), (1, 0), (0, 1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
    magmoms = np.array([[0.0, 0, magmom], [0.0, 0, -magmom]])
    M = Magnon(
        nspin=nspin,
        magmom=magmoms,
        Rlist=np.array(Rs),
        JR=np.array(JR),
        cell=cellv,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=magmoms,
    )
    return M


def test_afm_honeycomb_is_linear_and_isotropic():
    """The Neel honeycomb acoustic branch is linear with v = sqrt(3/2) J S a."""
    S, jiso, a = 1.5, -1.0e-3, 2.0
    st = _afm_honeycomb(S, jiso, a).get_spin_stiffness()
    theory = np.sqrt(1.5) * abs(S * jiso * 1e3 * a)
    assert st.order == 1
    assert st.A is None
    # directional speeds, one entry per cartesian axis
    assert abs(st.velocity[0] - theory) / theory < 1e-3
    assert abs(st.velocity[1] - theory) / theory < 1e-3
    assert abs(st.velocity[2]) < 1e-6


def test_afm_honeycomb_speed_scales_linearly_with_lattice_constant():
    """The linear-branch speed scales as a (not a^2)."""
    S, jiso = 1.5, -1.0e-3
    v1 = _afm_honeycomb(S, jiso, 1.0).get_spin_stiffness().velocity[0]
    v2 = _afm_honeycomb(S, jiso, 2.0).get_spin_stiffness().velocity[0]
    assert abs(v2 / v1 - 2.0) < 1e-3


def test_mixed_spin_honeycomb_stiffness():
    """A_xx = |J| S1 S2 / (2 (S1 + S2)) for inequivalent sublattice spins."""
    S1, S2, jiso, a = 1.5, 2.0, 1.0e-3, 1.0
    M = _fm_honeycomb(S1, jiso, a)
    magmoms = np.array([[0.0, 0, 2 * S1], [0.0, 0, 2 * S2]])
    M.magmom = magmoms
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=magmoms,
    )
    st = M.get_spin_stiffness()
    theory = abs(jiso * 1e3) * S1 * S2 / (2 * (S1 + S2))
    assert st.order == 2
    assert abs(abs(st.A[0, 0]) - theory) / theory < 1e-3
    assert abs(abs(st.A[1, 1]) - theory) / theory < 1e-3


def test_easy_axis_gap_does_not_change_the_stiffness():
    """A single-ion easy-axis term shifts the band but leaves the curvature."""
    S, jiso, a = 2.0, 1.0e-3, 1.0
    K = 0.5e-3

    def build(anis):
        M = _fm_chain(S, jiso, a)
        M.JR[0, 0, 0] += np.diag([0.0, 0.0, anis])
        M.set_reference(
            np.zeros(3),
            np.array([[0.0, 0, 1.0]]),
            np.array([1.0, 0, 0]),
            magmoms=np.array([[0.0, 0, 2 * S]]),
        )
        return M

    st_free = build(0.0).get_spin_stiffness()
    st_gap = build(K).get_spin_stiffness()
    assert st_gap.order == 2
    assert abs(st_gap.omega0 - abs(S * K) * 1e3) < 1e-6
    assert abs(abs(st_gap.A[0, 0]) - abs(st_free.A[0, 0])) < 1e-3 * abs(st_free.A[0, 0])


def test_spiral_reference_reports_linear_branch():
    """An incommensurate J1-J2 spiral has a linear Goldstone mode."""
    S, q = 1.0, np.arccos(0.25) / (2 * np.pi)
    Rlist = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    JR = np.zeros((4, 1, 1, 3, 3))
    # in the module's convention (H = -sum_<ij> J S_i.S_j) jiso > 0 is FM
    JR[0, 0, 0] = np.eye(3) * (-1.0e-3)   # J2, AFM (R = -2)
    JR[1, 0, 0] = np.eye(3) * 1.0e-3      # J1, FM  (R = -1)
    JR[2, 0, 0] = np.eye(3) * 1.0e-3
    JR[3, 0, 0] = np.eye(3) * (-1.0e-3)
    magmoms = np.array([[2 * S, 0.0, 0.0]])
    M = Magnon(
        nspin=1,
        magmom=magmoms,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([1.0, 10.0, 10.0]),
        _Q=np.array([q, 0.0, 0.0]),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([0.0, 0, 1.0]),
    )
    M.set_reference(
        np.array([q, 0.0, 0.0]),
        np.array([[0.0, 0, 1.0]]),
        np.array([0.0, 0, 1.0]),
        magmoms=magmoms,
    )
    st = M.get_spin_stiffness()
    assert st.order == 1
    assert st.A is None
    # the spiral runs along x: the only non-zero directional speed
    assert np.count_nonzero(st.velocity) == 1
    assert st.velocity[0] > 0


def test_sum_rule_matches_dispersion_for_bravais_chain():
    """The real-space sum rule reproduces the dispersion route for a Bravais lattice.

    The two routes must agree in sign as well: with `jiso > 0` the
    ferromagnet is the stable reference and its stiffness is positive.
    """
    S, jiso, a = 2.0, 1.0e-3, 1.0
    st = _fm_chain(S, jiso, a).get_spin_stiffness(
        J_iso=[jiso * 1e3, jiso * 1e3],
        bond_vectors=[[a, 0, 0], [-a, 0, 0]],
    )
    assert st.sum_rule is not None
    assert st.sum_rule[0, 0] > 0
    assert abs(st.sum_rule[0, 0] - st.A[0, 0]) / abs(st.A[0, 0]) < 1e-3


def test_sum_rule_is_not_the_multisublattice_answer():
    """The bond sum is not the exact LSWT stiffness of a multi-sublattice lattice."""
    S, jiso, a = 1.5, 1.0e-3, 1.0
    d01 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    a1v = np.array([a, 0.0, 0.0])
    a2v = np.array([a / 2, np.sqrt(3) * a / 2, 0.0])
    A_sum = stiffness_sum_rule(
        J_iso=[jiso * 1e3] * 3,
        S=S,
        bond_vectors=[d01, d01 - a1v, d01 - a2v],
    )
    st = _fm_honeycomb(S, jiso, a).get_spin_stiffness()
    assert not np.allclose(A_sum, st.A, rtol=1e-3, atol=1e-6)


def test_stiffness_rejects_non_positive_step():
    with pytest.raises(ValueError):
        _fm_chain(2.0, 1.0e-3).get_spin_stiffness(step=0.0)


def test_directional_speeds_are_not_reduced_to_a_norm():
    """A linear branch reports one speed per axis, not their Euclidean norm.

    The Euclidean norm of the axis projections is not a physical speed: the
    speed along a general direction is a projection of the speed vector, and
    the norm of its components overstates it.
    """
    st = _afm_chain(1.5, -1.0e-3).get_spin_stiffness()
    text = str(st)
    assert "spin-wave speeds" in text
    assert "x=" in text and "y=" in text and "z=" in text
    assert "|v|" not in text


def test_cli_stiffness_prints_block(monkeypatch, capsys):
    """The CLI loads TB2J results and prints the stiffness block."""
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_DIR}")
    from TB2J.magnon.magnon_cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["TB2J_magnon.py", "--stiffness", "-p", str(TEST_DATA_DIR)],
    )
    main()
    out = capsys.readouterr().out
    assert "Spin-wave stiffness" in out
    assert "stiffness tensor A (meV A^2)" in out or "spin-wave speeds" in out
