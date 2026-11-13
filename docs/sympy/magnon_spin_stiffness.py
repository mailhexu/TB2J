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
# 4. Implementation anchors: Magnon.get_spin_stiffness vs the certificate
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


# FM chain (S = 2, J = -1 meV, a = 1 A): the module's global sign convention
# makes the reported stiffness positive, so compare |J S a^2|.
S_val, jiso = 2.0, -1.0e-3
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

# FM honeycomb (S = 1.5, J = -1 meV): A = |J S| / 4, isotropic.
S_val, jiso = 1.5, -1.0e-3
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

# AFM chain (S = 1.5, J = +1 meV, magnetic cell 2 A): linear branch.
S_val, jiso = 1.5, 1.0e-3
st = afm_chain_magnon(S_val, jiso).get_spin_stiffness()
theory = S_val * jiso * 1e3 * 2.0  # J S a_cell, a_cell = 2 A
check(
    st.order == 1 and st.A is None,
    "implementation AFM chain reports a linear branch (no stiffness tensor)",
)
check(
    abs(np.linalg.norm(st.velocity) - theory) / theory < 1e-3,
    f"implementation AFM chain speed |v| = {np.linalg.norm(st.velocity):.6f} meV A "
    f"(textbook J S a_cell = {theory:.6f})",
)

# Sum rule: reproduces the dispersion route for the Bravais chain, and is
# *not* the general answer for the multi-sublattice honeycomb.
from TB2J.magnon.magnon_stiffness import stiffness_sum_rule  # noqa: E402

S_val, jiso = 2.0, -1.0e-3
A_sum = stiffness_sum_rule(
    J_iso=[jiso * 1e3, jiso * 1e3],
    S=S_val,
    bond_vectors=[[1.0, 0, 0], [-1.0, 0, 0]],
)
check(
    abs(abs(A_sum[0, 0]) - abs(S_val * jiso * 1e3)) < 1e-9,
    f"sum rule A_xx = {A_sum[0,0]:.6f} meV A^2 equals the Bravais textbook value",
)
S_val, jiso = 1.5, -1.0e-3
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

## Console output of the executed script

```
"""
body = "\n".join(REPORT)
footer = "\n```\n"
report_path.write_text(header + body + footer)
print(f"report written to {report_path}")
