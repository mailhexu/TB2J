"""SymPy certificate: spin-wave stiffness from the magnon dispersion.

Pins, with executed symbolic assertions and implementation-anchored numeric
comparisons, the quantities that `Magnon.get_spin_stiffness` returns:

1. **Ferromagnetic chain** (one site per cell, nearest neighbours at +-a):
   ``omega(k) = 2 J S (1 - cos(k a))``, so the stiffness tensor has
   ``A_xx = J S a^2`` (meV A^2) and no other component.
2. **Ferromagnetic honeycomb** (two sites per cell, nearest-neighbour bonds
   with the graphene lattice vectors ``a1 = (a, 0)``, ``a2 = (a/2, sqrt(3) a/2)``):
   ``omega_-(k) = 3 J S (1 - |gamma|)`` with
   ``gamma = (1 + exp(i k.a1) + exp(i k.a2)) / 3``.  Expanding in the
   cartesian wavevector gives ``A_xx = A_yy = J S / 4`` and ``A_xy = 0``
   (the honeycomb is isotropic at long wavelength).
3. **Antiferromagnetic chain** (two-site Neel cell, magnetic cell 2a): the
   acoustic branch is linear, ``omega(k) = 2 J S |sin(pi k)|`` in the magnetic
   fractional coordinate, i.e. the spin-wave speed is ``v = J S a_cell``
   (meV A).  There is no quadratic coefficient.
4. The **real-space sum rule** ``A = -(S/2) sum_bonds J v v`` is the textbook
   route (Liechtenstein-Katsnelson-Gubanov; Pajda et al., PRB 64, 174402
   (2001)) and coincides with the dispersion route for the single-sublattice
   chain.  For the multi-sublattice honeycomb it does **not**: the exact LSWT
   band is not the bond sum.
5. **Non-ferromagnetic reference states** are handled by the same machinery:
   the branch order is read from the deviation scaling, so a quadratic branch
   yields the tensor ``A`` and a linear one the directional speeds
   ``d omega / d|k|``.  Pinned closed forms:
   mixed-spin FM honeycomb ``A = 6 J S1 S2 mu / sqrt(4 S1 S2 + (S1-S2)^2)``
   (reduces to ``3 J S mu``, i.e. ``|J S a^2|/4``, for equal spins);
   Neel honeycomb ``omega(k) = 3 J S sqrt(1 - |gamma|^2)`` so ``v = sqrt(3/2) J S a``;
   an easy-axis single-ion gap shifts the branch by a constant and leaves the
   curvature unchanged; an incommensurate J1-J2 spiral has a linear Goldstone
   mode with wavevector ``cos(2 pi q*) = -J1/(4 J2)``.

Run in the mydev environment::

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/magnon_spin_stiffness.py
"""

from pathlib import Path

import numpy as np
import sympy as sp

_np = np

from TB2J.magnon.magnon3 import Magnon as _Magnon  # noqa: E402

REPORT: list[str] = []


def note(text: str):
    REPORT.append(text)
    print(text)


def check(condition, label: str):
    assert condition, f"CERTIFICATE FAILED: {label}"
    note(f"  [OK] {label}")


def min_boson_energy(M, k0=None):
    """Smallest eigenvalue of the boson Hamiltonian (>= 0 iff the reference
    state is stable).  The magnon module uses ``H = -sum_<ij> J S_i.S_j`` with
    every bond counted once, so ``jiso > 0`` is the ferromagnet and
    ``jiso < 0`` the antiferromagnet."""
    k0 = np.zeros(3) if k0 is None else np.asarray(k0, dtype=float)
    return float(np.linalg.eigvalsh(M.Hq(np.atleast_2d(k0))[0]).min())


# ---------------------------------------------------------------------------
# 1. FM chain: omega(k) = 2 J S (1 - cos(k a))  =>  A_xx = J S a^2
# ---------------------------------------------------------------------------
J, S, a = sp.symbols("J S a", real=True)
k = sp.symbols("k", real=True)

omega_chain = 2 * J * S * (1 - sp.cos(k * a))
A_chain = sp.simplify(sp.limit(omega_chain / (k**2), k, 0))
check(A_chain == J * S * a**2, "FM chain stiffness A_xx = J S a^2")

# ---------------------------------------------------------------------------
# 2. FM honeycomb: omega_-(k) = 3 J S (1 - |gamma|), gamma = (1 + e^{i k.a1}
#    + e^{i k.a2}) / 3  =>  A_xx = A_yy = J S / 4, A_xy = 0
# ---------------------------------------------------------------------------
kx, ky, eps = sp.symbols("kx ky eps", real=True)
a1 = sp.Matrix([a, 0])
a2 = sp.Matrix([a / 2, sp.sqrt(3) * a / 2])
kvec = sp.Matrix([eps * kx, eps * ky])
gamma = (1 + sp.exp(sp.I * (kvec.dot(a1))) + sp.exp(sp.I * (kvec.dot(a2)))) / 3
gamma_sq = sp.simplify(sp.expand_complex(gamma * sp.conjugate(gamma)))
gamma_sq = sp.series(gamma_sq, eps, 0, 3).removeO().expand()
abs_gamma = sp.simplify(1 + (gamma_sq - 1) / 2)
omega_honey = sp.series(3 * J * S * (1 - abs_gamma), eps, 0, 3).removeO().expand()
second = sp.expand(omega_honey.coeff(eps, 2))
A_honey_xx = sp.simplify(second.coeff(kx, 2))
A_honey_yy = sp.simplify(second.coeff(ky, 2))
A_honey_xy = sp.simplify(second.coeff(kx, 1).coeff(ky, 1))
check(
    sp.simplify(A_honey_xx - J * S * a**2 / 4) == 0,
    "FM honeycomb stiffness A_xx = J S a^2 / 4",
)
check(
    sp.simplify(A_honey_yy - J * S * a**2 / 4) == 0,
    "FM honeycomb stiffness A_yy = J S a^2 / 4",
)
check(A_honey_xy == 0, "FM honeycomb stiffness A_xy = 0")

# ---------------------------------------------------------------------------
# 3. AFM chain: omega(k) = 2 J S |sin(pi k)|  =>  speed v = J S a_cell
# ---------------------------------------------------------------------------
kfrac = sp.symbols("kfrac", real=True)
a_cell = sp.symbols("a_cell", real=True, positive=True)
omega_afm = 2 * J * S * sp.Abs(sp.sin(sp.pi * kfrac))
domega_dkfrac = sp.limit(sp.diff(omega_afm, kfrac), kfrac, 0, dir="+")
speed = sp.simplify(domega_dkfrac / (2 * sp.pi / a_cell))
check(
    sp.simplify(speed - J * S * a_cell) == 0,
    "AFM chain spin-wave speed v = J S a_cell (no quadratic stiffness)",
)

# ---------------------------------------------------------------------------
# 4. Non-collinear / non-ferromagnetic reference states
#
#    Every reference state the LSWT route supports is fitted the same way:
#    the exact `eig(g H)` band is sampled about k0 and the deviation order is
#    read off the scaling.  Quadratic -> tensor A, linear -> directional
#    speed d omega / d|k|.  The statements below pin the two non-FM branches
#    that have a closed form, so the branch classification is checked, not
#    just the ferromagnetic case.
# ---------------------------------------------------------------------------

# 4a. Mixed-spin FM honeycomb: A = J S1 S2 / (2 (S1 + S2)) for the graphene
#     bond set (equal spins reduce to J S / 4).
S1s, S2s = sp.symbols("S1 S2", positive=True)
Jpos = sp.Symbol("J", positive=True)
mu = sp.symbols("mu", positive=True)
d_half = 3 * Jpos * (S2s - S1s) / 2
disc = sp.sqrt(d_half**2 + Jpos**2 * S1s * S2s * (3 * (1 - mu * k**2)) ** 2)
lam_minus = sp.series(3 * Jpos * (S1s + S2s) / 2 - disc, mu, 0, 2).removeO()
A_mixed = sp.simplify(sp.diff(lam_minus, mu) * mu / k**2)
check(
    sp.simplify(A_mixed.subs(S2s, S1s) - 3 * Jpos * S1s * mu) == 0,
    "mixed-spin honeycomb reduces to 3 J S mu for S1 = S2",
)
check(
    sp.simplify(
        A_mixed - 6 * Jpos * S1s * S2s * mu
        / sp.sqrt(4 * S1s * S2s + (S1s - S2s) ** 2)
    )
    == 0,
    "mixed-spin honeycomb A = 6 J S1 S2 mu / sqrt(4 S1 S2 + (S1-S2)^2)",
)

# 4b. Neel honeycomb: the acoustic branch is *linear*,
#     omega(k) = 3 J S sqrt(1 - |gamma|^2)  =>  v = sqrt(3/2) J S a.
kx2, ky2 = sp.symbols("kx ky", real=True)
kk1 = a * kx2 / (2 * sp.pi)
kk2 = a * (kx2 / 2 + sp.sqrt(3) * ky2 / 2) / (2 * sp.pi)
one_minus_g2 = sp.simplify(
    sp.Rational(2, 3)
    - sp.Rational(2, 9)
    * (sp.cos(2 * sp.pi * kk1) + sp.cos(2 * sp.pi * kk2)
       + sp.cos(2 * sp.pi * (kk1 - kk2)))
)
quad = sp.simplify(
    sp.expand(
        sp.series(sp.series(one_minus_g2, kx2, 0, 3).removeO(), ky2, 0, 3).removeO()
    )
)
quad_c = sp.Poly(sp.expand(quad), kx2, ky2)  # keep only O(k^2) terms
quad_xx = quad_c.coeff_monomial(kx2**2)
quad_yy = quad_c.coeff_monomial(ky2**2)
quad_xy = quad_c.coeff_monomial(kx2 * ky2)
check(
    sp.simplify(quad_xx - a**2 / 6) == 0
    and sp.simplify(quad_yy - a**2 / 6) == 0
    and sp.simplify(quad_xy) == 0,
    "Neel honeycomb 1 - |gamma|^2 = a^2 |k|^2 / 6  =>  v = sqrt(3/2) J S a",
)

# 4c. Gapped FM chain: a single-ion easy-axis term adds a constant to the
#     branch (the band shifts, the curvature does not).
K_anis = sp.Symbol("K", positive=True)
omega_gapped = 2 * J * S * (1 - sp.cos(k * a)) + K_anis
check(
    sp.simplify(sp.diff(omega_gapped, k, 2).subs(k, 0) - 2 * J * S * a**2) == 0,
    "easy-axis gap shifts the FM branch by a constant; A_xx = J S a^2 unchanged",
)

# 4d. Spiral reference (incommensurate): the Goldstone mode of a stationary
#     spiral is linear in the *rotating* frame, so the API reports a speed.
#     For the J1-J2 chain the stationary spiral wavevector obeys
#     cos(2 pi q) = -J1 / (4 J2), asserted as the minimum of the classical
#     energy.
J1, J2 = sp.symbols("J1 J2", real=True)
qq = sp.Symbol("q", real=True)
E_spiral = J1 * sp.cos(2 * sp.pi * qq) + J2 * sp.cos(4 * sp.pi * qq)
q_star = sp.acos(-J1 / (4 * J2)) / (2 * sp.pi)
check(
    sp.simplify(sp.diff(E_spiral, qq).subs(qq, q_star)) == 0,
    "J1-J2 spiral: cos(2 pi q*) = -J1/(4 J2) is a stationary point",
)

# ---------------------------------------------------------------------------
# 5. Implementation anchors: Magnon.get_spin_stiffness vs the certificate
# ---------------------------------------------------------------------------


def fm_chain_magnon(S_val, jiso):
    """FM chain, one site per cell, bonds +-a (a = 1 Angstrom)."""
    magmom = 2 * S_val
    Rlist = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = np.eye(3) * jiso
    JR[1, 0, 0] = np.eye(3) * jiso
    M = _Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0, magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3),
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


def fm_honeycomb_magnon(S_val, jiso):
    """FM honeycomb, two sites per cell, NN bonds (graphene lattice)."""
    a_val = 1.0
    cellv = np.array([[a_val, 0, 0], [a_val / 2, np.sqrt(3) * a_val / 2, 0], [0, 0, 10.0]])
    nspin = 2
    magmom = 2 * S_val
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
    M = _Magnon(
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


def afm_chain_magnon(S_val, jiso, a_val=1.0):
    """AFM chain, two-site Neel cell (magnetic cell 2a), bonds (0,1)@0, (1,0)@a."""
    magmom = 2 * S_val
    Rlist = np.array([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((3, 2, 2, 3, 3))
    JR[0, 0, 1] = np.eye(3) * jiso
    JR[0, 1, 0] = np.eye(3) * jiso
    JR[1, 1, 0] = np.eye(3) * jiso
    JR[2, 0, 1] = np.eye(3) * jiso
    M = _Magnon(
        nspin=2,
        magmom=np.array([[0.0, 0, magmom], [0.0, 0, -magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.array([[2 * a_val, 0, 0], [0, 2 * a_val, 0], [0, 0, 10.0]]),
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


# FM chain (S = 2, J = +1 meV, a = 1 A): jiso > 0 is the ferromagnet in the
# module's convention, so this reference is the stable one.
S_val, jiso = 2.0, 1.0e-3
check(
    min_boson_energy(fm_chain_magnon(S_val, jiso)) > -1e-12,
    "FM chain reference state is stable (jiso > 0)",
)
st = fm_chain_magnon(S_val, jiso).get_spin_stiffness()
theory = abs(S_val * jiso * 1e3)
check(
    st.order == 2 and abs(abs(st.A[0, 0]) - theory) / theory < 1e-4,
    f"implementation FM chain A_xx = {st.A[0, 0]:.6f} meV A^2 "
    f"(textbook |J S a^2| = {theory:.6f})",
)
check(
    abs(st.A[1, 1]) < 1e-8 and abs(st.A[0, 1]) < 1e-8,
    "implementation FM chain A is diagonal (no in-plane exchange)",
)

# FM honeycomb (S = 1.5, J = +1 meV): A = |J S| / 4, isotropic.
S_val, jiso = 1.5, 1.0e-3
check(
    min_boson_energy(fm_honeycomb_magnon(S_val, jiso)) > -1e-12,
    "FM honeycomb reference state is stable (jiso > 0)",
)
st = fm_honeycomb_magnon(S_val, jiso).get_spin_stiffness()
theory = abs(S_val * jiso * 1e3) / 4
check(
    st.order == 2
    and abs(abs(st.A[0, 0]) - theory) / theory < 1e-3
    and abs(abs(st.A[1, 1]) - theory) / theory < 1e-3
    and abs(st.A[0, 1]) < 1e-3 * theory,
    f"implementation FM honeycomb A = diag({st.A[0,0]:.6f}, {st.A[1,1]:.6f}) meV A^2 "
    f"(textbook |J S a^2|/4 = {theory:.6f})",
)

# AFM chain (S = 1.5, J = -1 meV, magnetic cell 2 A): linear branch.
S_val, jiso = 1.5, -1.0e-3
check(
    min_boson_energy(afm_chain_magnon(S_val, jiso)) > -1e-12,
    "AFM chain reference state is stable (jiso < 0)",
)
st = afm_chain_magnon(S_val, jiso).get_spin_stiffness()
theory = abs(S_val * jiso * 1e3 * 2.0)  # |J| S a_cell, a_cell = 2 A
check(
    st.order == 1 and st.A is None,
    "implementation AFM chain reports a linear branch (no stiffness tensor)",
)
check(
    abs(st.velocity[0] - theory) / theory < 1e-3,
    f"implementation AFM chain speed v_x = {st.velocity[0]:.6f} meV A "
    f"(textbook J S a_cell = {theory:.6f})",
)

# Sum rule: reproduces the dispersion route for the Bravais chain, and is
# *not* the general answer for the multi-sublattice honeycomb.
from TB2J.magnon.magnon_stiffness import stiffness_sum_rule  # noqa: E402

S_val, jiso = 2.0, 1.0e-3
A_sum = stiffness_sum_rule(
    J_iso=[jiso * 1e3, jiso * 1e3],
    S=S_val,
    bond_vectors=[[1.0, 0, 0], [-1.0, 0, 0]],
)
check(
    abs(abs(A_sum[0, 0]) - abs(S_val * jiso * 1e3)) < 1e-9,
    f"sum rule A_xx = {A_sum[0,0]:.6f} meV A^2 equals the Bravais textbook value",
)
S_val, jiso = 1.5, 1.0e-3
d01 = np.array([0.5, np.sqrt(3) / 2, 0.0])
a1v = np.array([1.0, 0.0, 0.0])
a2v = np.array([0.5, np.sqrt(3) / 2, 0.0])
A_sum_honey = stiffness_sum_rule(
    J_iso=[jiso * 1e3] * 3,
    S=S_val,
    bond_vectors=[d01, d01 - a1v, d01 - a2v],
)
st = fm_honeycomb_magnon(S_val, jiso).get_spin_stiffness()
check(
    not np.allclose(A_sum_honey, st.A, rtol=1e-3, atol=1e-6),
    "sum rule does not reproduce the multi-sublattice honeycomb stiffness "
    "(the exact LSWT band is not the bond sum)",
)

# ---------------------------------------------------------------------------
# 6. Non-FM implementation anchors
# ---------------------------------------------------------------------------


def afm_honeycomb_magnon(S_val, jiso, a_val=2.0):
    """Neel honeycomb: two sites/cell, opposite moments, 3 NN bonds."""
    cellv = np.array([[a_val, 0, 0], [a_val / 2, np.sqrt(3) * a_val / 2, 0], [0, 0, 10.0]])
    nspin, n, magmom = 2, 3, 2 * S_val
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
    M = _Magnon(
        nspin=nspin,
        magmom=magmoms,
        Rlist=np.array(Rs),
        JR=np.array(JR),
        cell=cellv,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(np.zeros(3), np.array([[0.0, 0, 1.0]]),
                    np.array([1.0, 0, 0]), magmoms=magmoms)
    return M


def gapped_fm_chain_magnon(S_val, jiso, k_anis, a_val=1.0):
    """FM chain with an on-site easy-axis term (constant gap)."""
    magmom = 2 * S_val
    Rlist = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = np.eye(3) * jiso + np.diag([0.0, 0.0, k_anis])
    JR[1, 0, 0] = np.eye(3) * jiso
    M = _Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0, magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3) * a_val,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(np.zeros(3), np.array([[0.0, 0, 1.0]]),
                    np.array([1.0, 0, 0]), magmoms=np.array([[0.0, 0, magmom]]))
    return M


# Neel honeycomb (S = 1.5, J = -1 meV, a = 2 A): linear branch,
# v = sqrt(3/2) |J| S a.
S_val, jiso, a_val = 1.5, -1.0e-3, 2.0
check(
    min_boson_energy(afm_honeycomb_magnon(S_val, jiso, a_val)) > -1e-12,
    "Neel honeycomb reference state is stable (jiso < 0)",
)
st = afm_honeycomb_magnon(S_val, jiso, a_val).get_spin_stiffness()
theory = np.sqrt(1.5) * abs(S_val * jiso * 1e3 * a_val)
check(
    st.order == 1 and st.A is None,
    "implementation Neel honeycomb reports a linear branch (no tensor)",
)
check(
    abs(st.velocity[0] - theory) / theory < 1e-3
    and abs(st.velocity[1] - theory) / theory < 1e-3
    and abs(st.velocity[2]) < 1e-6,
    f"implementation Neel honeycomb v_x = v_y = {st.velocity[0]:.6f} meV A, "
    f"v_z = {st.velocity[2]:.1e} (textbook sqrt(3/2) J S a = {theory:.6f})",
)

# Gapped FM chain: the gap shifts the branch, the curvature is unchanged.
S_val, jiso, k_anis = 2.0, 1.0e-3, 0.5e-3
st_free = gapped_fm_chain_magnon(S_val, jiso, 0.0).get_spin_stiffness()
st_gap = gapped_fm_chain_magnon(S_val, jiso, k_anis).get_spin_stiffness()
check(
    st_gap.order == 2
    and abs(st_gap.omega0 - abs(S_val * k_anis) * 1e3) < 1e-6,
    f"implementation gapped FM chain omega(Gamma) = {st_gap.omega0:.6f} meV "
    f"equals the anisotropy gap |S K| = {abs(S_val * k_anis) * 1e3:.6f} meV",
)
check(
    abs(abs(st_gap.A[0, 0]) - abs(st_free.A[0, 0])) / abs(st_free.A[0, 0]) < 1e-3,
    f"gapped FM chain A_xx = {st_gap.A[0,0]:.6f} equals the ungapped "
    f"{st_free.A[0,0]:.6f} meV A^2 (gap does not change the curvature)",
)

# Mixed-spin FM honeycomb: A = J S1 S2 / (2 (S1 + S2)) with the graphene bonds
# (equal spins reduce to J S / 4).
S1_val, S2_val, jiso = 1.5, 2.0, 1.0e-3
mixed = fm_honeycomb_magnon(S1_val, jiso)
mixed.magmom = np.array([[0.0, 0, 2 * S1_val], [0.0, 0, 2 * S2_val]])
mixed.set_reference(np.zeros(3), np.array([[0.0, 0, 1.0]]),
                    np.array([1.0, 0, 0]), magmoms=mixed.magmom)
st = mixed.get_spin_stiffness()
theory = abs(jiso * 1e3) * S1_val * S2_val / (2 * (S1_val + S2_val))
check(
    st.order == 2 and abs(abs(st.A[0, 0]) - theory) / theory < 1e-3,
    f"implementation mixed-spin honeycomb A_xx = {st.A[0,0]:.6f} meV A^2 "
    f"(textbook |J| S1 S2 / (2(S1+S2)) = {theory:.6f})",
)

# ---------------------------------------------------------------------------
# 4c. Multi-site magnons with inequivalent spins (S1 != S2).
#
#     The transverse block carries sqrt(S_i S_j) for every sublattice pair, so
#     a multi-site magnet is NOT the single-S case with a rescaled coupling.
#     For the honeycomb with sublattice spins S1, S2 the textbook branches are
#
#         omega_pm(k) = (3J/2) [ (S1+S2) +/- sqrt((S1-S2)^2 + 4 S1 S2 |gamma|^2) ],
#         |gamma|^2 = 1 - a^2 k^2 / 6,
#
#     hence omega_-(Gamma) = 0, omega_+(Gamma) = 3 J (S1+S2), the Dirac point
#     splits into 3 J min(S1,S2) / 3 J max(S1,S2), and the acoustic branch is
#     quadratic with
#
#         A_xx = A_yy = J S1 S2 a^2 / (2 (S1+S2))   (-> J S a^2/4 for S1 = S2).
# ---------------------------------------------------------------------------
Xsym = sp.Symbol("X", positive=True)
om_ferri = (3 * Jpos / 2) * (
    Xsym - sp.sqrt(Xsym**2 - 4 * S1s * S2s * (a**2 * k**2 / 6))
)
A_ferri = sp.simplify(
    sp.series(om_ferri, k, 0, 3).removeO().expand().coeff(k, 2)
)
check(
    sp.simplify(
        A_ferri.subs(Xsym, S1s + S2s) - Jpos * S1s * S2s * a**2 / (2 * (S1s + S2s))
    )
    == 0,
    "multi-site honeycomb A = J S1 S2 a^2 / (2 (S1+S2))",
)
check(
    sp.simplify(A_ferri.subs({Xsym: 2 * S, S1s: S, S2s: S}) - Jpos * S * a**2 / 4)
    == 0,
    "multi-site honeycomb reduces to J S a^2/4 for equal spins",
)
check(
    sp.simplify(om_ferri.subs(k, 0)) == 0,
    "multi-site honeycomb acoustic branch is gapless at Gamma",
)


def _exact_band(M, k_cart):
    """Positive eigenenergies (meV) of the exact eig(gH) spectrum."""
    cell = np.asarray(M.cell)
    k_frac = cell @ np.asarray(k_cart, dtype=float) / (2 * np.pi)
    H = M.Hq(np.atleast_2d(k_frac))[0]
    n = H.shape[-1] // 2
    g = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    return np.sort(np.abs(np.linalg.eigvals(g @ H))) * 1e3


S1_val, S2_val, jiso = 1.5, 2.0, 1.0e-3
a_val = 1.0
multi = fm_honeycomb_magnon(S1_val, jiso)
multi.magmom = np.array([[0.0, 0, 2 * S1_val], [0.0, 0, 2 * S2_val]])
multi.set_reference(np.zeros(3), np.array([[0.0, 0, 1.0]]),
                    np.array([1.0, 0, 0]), magmoms=multi.magmom)
a1v = np.array([a_val, 0.0, 0.0])
a2v = np.array([a_val / 2, np.sqrt(3) * a_val / 2, 0.0])
for k_vec in (np.array([0.05, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]),
              np.array([0.0, 0.1, 0.0])):
    gam = (1 + np.exp(1j * k_vec.dot(a1v)) + np.exp(1j * k_vec.dot(a2v))) / 3
    root = np.sqrt((S1_val - S2_val) ** 2 + 4 * S1_val * S2_val * abs(gam) ** 2)
    target = (3 * jiso * 1e3 / 2) * np.array([S1_val + S2_val - root,
                                              S1_val + S2_val + root])
    measured = _exact_band(multi, k_vec)
    check(
        abs(measured[0] - target[0]) < 1e-6
        and abs(measured[-1] - target[1]) < 1e-6
        and abs(measured[1] - measured[0]) < 1e-6
        and abs(measured[2] - measured[-1]) < 1e-6,
        f"implementation multi-site band at k={k_vec[:2].tolist()}: "
        f"omega_-/+ = {measured[0]:.6f}/{measured[-1]:.6f} meV "
        f"(textbook {target[0]:.6f}/{target[1]:.6f})",
    )
st = multi.get_spin_stiffness()
theory = abs(jiso * 1e3) * S1_val * S2_val * a_val**2 / (2 * (S1_val + S2_val))
check(
    st.order == 2 and abs(abs(st.A[0, 0]) - theory) / theory < 1e-3,
    f"implementation multi-site honeycomb A_xx = {st.A[0,0]:.6f} meV A^2 "
    f"(textbook |J| S1 S2 a^2 / (2(S1+S2)) = {theory:.6f})",
)

# Spiral reference: an incommensurate J1-J2 spiral has a linear Goldstone
# mode in the rotating frame => speed, no tensor.
def spiral_chain_magnon(S_val, j1, j2, q_frac):
    """1D J1-J2 spiral; j1, j2 in the module's convention (j > 0 = FM)."""
    Rlist = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    JR = np.zeros((4, 1, 1, 3, 3))
    JR[0, 0, 0] = np.eye(3) * j2
    JR[1, 0, 0] = np.eye(3) * j1
    JR[2, 0, 0] = np.eye(3) * j1
    JR[3, 0, 0] = np.eye(3) * j2
    magmoms = np.array([[2 * S_val, 0.0, 0.0]])
    M = _Magnon(
        nspin=1,
        magmom=magmoms,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([1.0, 10.0, 10.0]),
        _Q=np.array([q_frac, 0.0, 0.0]),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([0.0, 0, 1.0]),
    )
    M.set_reference(np.array([q_frac, 0.0, 0.0]), np.array([[0.0, 0, 1.0]]),
                    np.array([0.0, 0, 1.0]), magmoms=magmoms)
    return M


# J1 = 1 meV FM, J2 = 1 meV AFM; in the module's convention (H = -sum J S.S)
# the ferromagnet has jiso > 0 and the antiferromagnet jiso < 0.
S_val = 1.0
j1_tb2j, j2_tb2j = 1.0e-3, -1.0e-3
q_frac = np.arccos(1.0 / 4.0) / (2 * np.pi)   # cos(2 pi q) = -J1/(4 J2) = 1/4
check(
    min_boson_energy(spiral_chain_magnon(S_val, j1_tb2j, j2_tb2j, q_frac)) > -1e-12,
    "J1-J2 spiral reference state is stable",
)
st = spiral_chain_magnon(S_val, j1_tb2j, j2_tb2j, q_frac).get_spin_stiffness()
check(
    st.order == 1 and st.A is None,
    "implementation spiral reference reports a linear branch (no tensor)",
)
check(
    st.velocity is not None and np.count_nonzero(st.velocity) == 1,
    f"implementation spiral speed is one-directional: {np.round(st.velocity, 6)} meV A",
)

note("")
note("CERTIFICATE OK: all assertions passed.")
report_path = Path(__file__).with_suffix(".md")
header = """# Certificate: spin-wave stiffness from the magnon dispersion

Generated by `magnon_spin_stiffness.py` (all assertions executed and passed in
the `mydev` environment). Pinned statements:

1. FM chain (`H = -sum_<ij> J S_i.S_j`, bonds once, `|S| = |m|/2`):
   `omega(k) = 2 J S (1 - cos(k a))`, hence the stiffness tensor
   `A_xx = J S a^2` (meV A^2) with no other component.
2. FM honeycomb (graphene lattice vectors `a1 = (a,0)`, `a2 = (a/2, sqrt(3)a/2)`):
   `omega_-(k) = 3 J S (1 - |gamma|)`,
   `gamma = (1 + e^{i k.a1} + e^{i k.a2})/3`; the cartesian expansion gives
   `A_xx = A_yy = J S / 4` and `A_xy = 0`.
3. AFM chain (two-site Neel cell, magnetic cell `2a`): the acoustic branch is
   linear, `omega(k) = 2 J S |sin(pi k)|`, so the spin-wave speed is
   `v = J S a_cell` (meV A) and no quadratic stiffness exists.
4. The real-space sum rule `A = -(S/2) sum_bonds J v v` equals the dispersion
   route for the single-sublattice chain, and does **not** reproduce the
   multi-sublattice honeycomb (the exact LSWT band is not the bond sum); it
   therefore needs the true bond vectors and is a cross-check only.
5. Non-ferromagnetic reference states go through the same machinery: the
   branch order is read from the deviation scaling, so a quadratic branch
   yields the tensor `A` and a linear branch the directional speeds
   `d omega / d|k|`. Pinned closed forms:
   - mixed-spin FM honeycomb `A = 6 J S1 S2 mu / sqrt(4 S1 S2 + (S1-S2)^2)`
     (reduces to `3 J S mu`, i.e. `|J S a^2|/4`, for equal spins);
   - Neel honeycomb `omega(k) = 3 J S sqrt(1 - |gamma|^2)`, hence the linear
     branch has `v = sqrt(3/2) J S a`;
   - an easy-axis single-ion gap shifts the branch by `|S K|` and leaves the
     curvature unchanged;
   - an incommensurate J1-J2 spiral has a linear Goldstone mode at
     `cos(2 pi q*) = -J1/(4 J2)`.
6. Multi-site magnets with inequivalent sublattice spins: the transverse
   block carries `sqrt(S_i S_j)` per sublattice pair, so this is not the
   single-`S` model with a rescaled coupling. For the honeycomb with spins
   `S1, S2` the textbook branches are
   `omega_pm(k) = (3J/2)[(S1+S2) +/- sqrt((S1-S2)^2 + 4 S1 S2 |gamma|^2)]`,
   so `omega_-(Gamma) = 0`, `omega_+(Gamma) = 3 J (S1+S2)`, the Dirac point
   splits into `3 J min(S1,S2)` / `3 J max(S1,S2)`, and the branch stays
   quadratic with `A_xx = A_yy = J S1 S2 a^2 / (2 (S1+S2))`
   (`-> J S a^2/4` for equal spins).

## Console output of the executed script

```
"""
body = "\n".join(REPORT)
footer = "\n```\n"
report_path.write_text(header + body + footer)
print(f"report written to {report_path}")
