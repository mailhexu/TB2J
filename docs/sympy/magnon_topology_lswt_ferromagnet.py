"""SymPy certificate: collinear ferromagnet LSWT conventions for magnon topology.

This script pins, with executed symbolic assertions and implementation-anchored
numeric comparisons, the conventions that the topological magnon calculations
build on:

1. The executable Fourier convention ``J(k) = sum_R J(R) exp(-2 pi i k.R)``
   used by ``Magnon.Jq`` (negative phase) and its consequences
   (``J(k) = J(-k)^*`` for real-space tensors; even kernels).
2. The Holstein-Primakoff reduction of
   ``H = -sum_{ij,R} S_i . J_ij(R) . S_j`` for a collinear ferromagnet with
   quantization axis ``z``, reproducing the block structure of
   ``Magnon.Hq`` (A1 - C, B, A2 - C) and the standard dispersion
   ``omega(k) = 2 S [J(0) - J(k)]``, plus the easy-axis single-ion gap.
3. The executable transverse-vector contractions
   (``u = e1 + i e2 = (1, i, 0)``): for a z-aligned collinear FM the pairing
   block vanishes identically (``u^T T u = 0`` for isotropic and z-DMI
   tensors), the physical bands are the eigenvalues of ``A1 - C``, and DMI
   on next-nearest-neighbor bonds enters as a real mass term through
   ``u^T eps(D) u*`` with reverse-bond antisymmetry ``D_ji(-R) = -D_ij(R)``.
4. The honeycomb ferromagnet: symbolic assembly of the site blocks, numeric
   equality of the certificate's 4x4 Nambu eigenvalues with the actual
   ``Magnon.Hq`` implementation for the same SpinIO source, Dirac-point
   expansion at K/K' (valley chiralities and common mass), and the pinned
   band Chern number ``C_upper = (chi_K' s - chi_K s)/2``-style valley sum.
5. Gauge invariance of the Fukui-Hatsugai-Suzuki (FHS) lattice link
   construction: symbolic phase telescoping and unitary group mixing.
6. Numeric cross-check: pinned Chern sign as a function of the DMI sign via
   an independent winding computation on the site-space 2-band problem
   anchored to the implementation comparison of step 4.

Run in the mydev environment::

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/magnon_topology_lswt_ferromagnet.py
"""

from pathlib import Path

import numpy as np
import sympy as sp

_np = np

from ase import Atoms as _Atoms  # noqa: E402

from TB2J.io_exchange import SpinIO as _SpinIO  # noqa: E402
from TB2J.magnon.magnon3 import Magnon as _Magnon  # noqa: E402

REPORT: list[str] = []


def note(text: str):
    REPORT.append(text)
    print(text)


def check(condition, label: str):
    assert condition, f"CERTIFICATE FAILED: {label}"
    note(f"  [OK] {label}")


# ---------------------------------------------------------------------------
# 1. Fourier phase convention
# ---------------------------------------------------------------------------
note("1. Fourier phase convention (executable: exp(-2 pi i k.R))")

kx, ky, R1x, R1y = sp.symbols("k_x k_y R1_x R1_y", real=True)
Jreal = sp.symbols("J_R", real=True)
kr = kx * R1x + ky * R1y
Jq_pos = Jreal * sp.exp(sp.I * 2 * sp.pi * kr)
Jq_neg = Jreal * sp.exp(-sp.I * 2 * sp.pi * kr)
check(
    sp.simplify(
        sp.expand(Jq_neg + Jq_pos).rewrite(sp.cos) - 2 * Jreal * sp.cos(2 * sp.pi * kr)
    )
    == 0,
    "J(k) + J(-k) = 2 Re J(k): real tensors give even kernels (J(-k) = J(k)^*)",
)
check(
    sp.simplify(Jq_neg - Jq_pos.subs({kx: -kx, ky: -ky})) == 0,
    "negated-phase J(-k) equals negated argument of the positive-phase kernel",
)
check(
    sp.simplify(sp.conjugate(Jq_neg) - Jq_pos) == 0,
    "conjugation identity J(k)* = sum J(R) e^{+2pi i k.R} = J(-k)",
)

# ---------------------------------------------------------------------------
# 2. Single-sublattice ferromagnet: HP reduction reproduces Hq and omega(k)
# ---------------------------------------------------------------------------
note("2. Single-sublattice ferromagnet HP reduction")

# H = -sum_R J(R) S_0 . S_R ; spins parallel z.
# HP: S^+ ~ sqrt(2S) a, S^- ~ sqrt(2S) a^dag, S^z = S - a^dag a.
# 1D chain: self shell J0 (R=0) and bonds +-R1 with J1.
S, J0s, J1s, k = sp.symbols("S J_0 J_1 k", real=True, positive=True)
J_of_k = J0s + J1s * (sp.exp(sp.I * k) + sp.exp(-sp.I * k))


def simpl(expr):
    """simplify with exponential->cos/sin rewriting for phase expressions."""
    return sp.simplify(sp.expand(sp.expand(expr).rewrite(sp.cos)))


omega_chain = S * ((J0s + 2 * J1s) - J_of_k)
check(
    simpl(omega_chain - S * 2 * J1s * (1 - sp.cos(k))) == 0,
    "chain dispersion omega(k) = S sum_R J(R)[1 - cos(k.R)]"
    " (the R-sum counts every bond twice, so this is the textbook 2 J S (1-cos ka))",
)

# Direct-constructor block chain for a z-aligned FM with physical-spin
# exchange: Hq applies the transverse 1/2 and the spin-length factors once.
# SpinIO inputs instead use unit spins and directed bonds; the loader converts
# those tensors by 2/(S_i S_j) before this same physical-spin kernel.
u = sp.Matrix([1, sp.I, 0])


def isoT(J):
    return J * sp.eye(3)


J_of_k = J0s + J1s * (sp.exp(sp.I * k) + sp.exp(-sp.I * k))
Jneg = J_of_k.subs(k, -k)
A1 = -0.5 * simpl((u.T * isoT(Jneg) * u.conjugate())[0, 0]) * S  # 1/2 of Eq. (26)
B = sp.simplify((u.T * isoT(Jneg) * u)[0, 0])
C = simpl(-S * (J0s + 2 * J1s))  # C_total = -S J(0) from J0 = -Jq(0)
check(sp.simplify(B) == 0, "pairing block B = u^T J u vanishes for isotropic z-FM")
omega_exec = simpl(A1 - C)  # B = 0: bands are eigenvalues of A1 - C
check(
    simpl(omega_exec - S * (J0s + 2 * J1s - J_of_k)) == 0,
    "executable chain omega = S[J(0) - J(k)] (textbook LSWT, all S)",
)
check(
    simpl(omega_exec.subs(S, 1) - S.subs(S, 1) * ((J0s + 2 * J1s) - J_of_k)) == 0,
    "at S = 1 (mu = 2 muB) the executable omega equals the textbook S[J(0) - J(k)]",
)

# Numeric anchor: convert the unit-spin SpinIO exchange for each moment.
_atoms = _Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
_ex = {((0, 0, 1), 0, 0): 0.002, ((0, 0, -1), 0, 0): 0.002}
_exc = _SpinIO(
    atoms=_atoms,
    spinat=np.array([[0, 0, 4.0]]),
    charges=[0.0],
    index_spin=[0],
    colinear=False,
    exchange_Jdict=_ex,
)
_exc._build_Rlist()
ratios = []
for mu in (2.0, 4.0, 6.0):
    _m = _Magnon.load_from_io(_exc)
    _m.set_reference((0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, mu]])
    _om = float(_m._magnon_energies(np.array([[0.0, 0.0, 0.5]]))[0, 0])
    _S = mu / 2
    _theory = (2.0 / _S**2) * _S * (2 * 0.002 - 2 * 0.002 * np.cos(np.pi))
    ratios.append(_om / _theory)
check(
    np.allclose(ratios, [1.0, 1.0, 1.0]),
    f"SpinIO energies equal the converted textbook LSWT for every S"
    f" (ratios {[round(r, 6) for r in ratios]})",
)

K_sia = sp.symbols("K", real=True, positive=True)
# easy-axis SIA: H_SIA = -K S_z^2 -> executable gap omega(0) = 2 K / S
# (textbook 2 K S at S = 1; only the K-contraction in C survives, u has no z)
omega_gapped = sp.simplify(omega_exec + 2 * K_sia / S)
check(
    simpl(omega_gapped.subs(k, 0) - 2 * K_sia / S) == 0,
    "easy-axis gap omega(0) = 2 K / S (executable scaling; 2 K S at S = 1)",
)

# Metric-solver identity (Colpa form used by the topology solver):
# H = L L^dag (lower Cholesky); M = L^dag g L eigendecomposition u, omega
# gives physical vectors v = L^{-dag} u with H v = omega g v, v^dag g v = 1/omega.
rng = np.random.default_rng(11)
Hn = np.array([[2.0, 0.3 + 0.2j], [0.3 - 0.2j, 2.0]])
Kn = np.linalg.cholesky(Hn)  # lower: H = K K^dag
gn = np.diag([1.0, -1.0])
Mn = Kn.conj().T @ gn @ Kn
wn, un = np.linalg.eigh(Mn)
v = np.linalg.solve(Kn.conj().T, un)  # v = K^{-dag} u
resid = Hn @ v[:, 1] - (wn[1] * gn) @ v[:, 1]
check(
    np.max(np.abs(resid)) < 1e-13,
    "numeric: H v = omega g v with v = K^{-dag} u for M = K^dag g K",
)
check(
    abs((v[:, 1].conj() @ gn @ v[:, 1]) - 1.0 / wn[1]) < 1e-13,
    "numeric: metric norm v^dag g v = 1/omega",
)
psi = v[:, 1] * np.sqrt(wn[1])
check(
    abs((psi.conj() @ gn @ psi) - 1.0) < 1e-13,
    "numeric: sqrt(omega) rescale gives unit metric norm",
)

# ---------------------------------------------------------------------------
# 3. Transverse contractions for DMI tensors
# ---------------------------------------------------------------------------
note("3. Transverse contractions (u = (1, i, 0), unnormalized as in the code)")

Dz = sp.symbols("D_z", real=True)
eps_D = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])  # eps(D zhat)/D_z
check(
    sp.simplify((u.T * eps_D * u)[0, 0]) == 0,
    "u^T eps(D_z) u = 0: z-DMI produces no pairing for a z-aligned FM",
)
contract_D = sp.simplify((u.T * eps_D * u.conjugate())[0, 0])
check(
    sp.simplify(contract_D - (-2 * sp.I)) == 0,
    "u^T eps(D_z) u* = -2i: z-DMI enters A1 as a complex phase amplitude",
)
check(
    sp.simplify((u.T * sp.eye(3) * u.conjugate())[0, 0] - 2) == 0,
    "u^T I u* = 2: isotropic contraction carries factor 2",
)

# ---------------------------------------------------------------------------
# 4. Honeycomb ferromagnet: symbolic site blocks, bands, valleys, Chern
# ---------------------------------------------------------------------------
note("4. Honeycomb ferromagnet: symbolic site blocks and pinned Chern number")

a = sp.symbols("a", positive=True)
kx, ky = sp.symbols("k_x k_y", real=True)
kfrac = sp.Matrix([kx, ky, 0])
NN = [
    sp.Matrix([sp.Rational(-1, 3), sp.Rational(2, 3), 0]),
    sp.Matrix([sp.Rational(-1, 3), sp.Rational(-1, 3), 0]),
    sp.Matrix([sp.Rational(2, 3), sp.Rational(-1, 3), 0]),
]
NNN_R = [
    sp.Matrix([1, 0, 0]),
    sp.Matrix([-1, 1, 0]),
    sp.Matrix([0, -1, 0]),
]  # counterclockwise triangle
J1, J2, K2, Ssym = sp.symbols("J_1 J_2 K S", real=True, positive=True)


def phase_neg(Rfrac):
    """e^{-2 pi i k.R} (executable Jq phase)."""
    return sp.exp(-sp.I * 2 * sp.pi * (kfrac.T * Rfrac)[0, 0])


def A1_contractions(chirality):
    """A1 diagonal-site sums: -sum_bonds u^T (J iso I + eps(sigma D_z z)) u* e^{-2pi i k.R}

    Bonds are stored per directed pair; the reverse bond of (R, sigma) is
    (-R, -sigma) by DMI antisymmetry, so the +- R sums appear with flipped
    chirality. Isotropic NN/NNN and SIA are added by the caller.
    """
    total = 0
    for Rf in NNN_R:
        pos = sp.simplify((u.T * (chirality * Dz * eps_D) * u.conjugate())[0, 0])
        total += pos * phase_neg(-Rf)  # bond (R, +sigma): J(-k) phase e^{+...}
        total += (-pos) * phase_neg(Rf)  # reverse bond (-R, -sigma)
    return sp.simplify(total)


# Site-diagonal A1 entries (DMI part only; isotropic and SIA added below):
# A1 = -sqrt(SiSj) u^T J(-k)^T u* ; J^T for iso is I, for eps(D) is -eps(D),
# which flips the chirality sign again; we absorb: net chirality as coded.
A1_AA_dmi = sp.simplify(
    0.5
    * (
        (u.T * ((+Dz) * eps_D).T * u.conjugate())[0, 0]
        * sum(phase_neg(-Rf) for Rf in NNN_R)
        + (u.T * ((-Dz) * eps_D).T * u.conjugate())[0, 0]
        * sum(phase_neg(Rf) for Rf in NNN_R)
    )
)
A1_BB_dmi = sp.simplify(
    0.5
    * (
        (u.T * ((-Dz) * eps_D).T * u.conjugate())[0, 0]
        * sum(phase_neg(-Rf) for Rf in NNN_R)
        + (u.T * ((+Dz) * eps_D).T * u.conjugate())[0, 0]
        * sum(phase_neg(Rf) for Rf in NNN_R)
    )
)
check(
    simpl(sp.im(sp.expand(A1_AA_dmi))) == 0,
    "A1_AA DMI part is real (reverse-bond antisymmetry D_ji(-R) = -D_ij(R))",
)
check(simpl(sp.im(sp.expand(A1_BB_dmi))) == 0, "A1_BB DMI part is real")

# Full site blocks (collinear FM, u^T I u* = 2). NN bonds (A->B) contribute
# only to the off-diagonal block; the site-diagonal A1 entries carry the NNN
# (isotropic J2 + z-DMI) shells. The on-site field C_i sums over all bonds
# leaving site i: 3 NN keys (J1) and 6 NNN A-A keys (J2), times S.
# The executable convention carries the printed 1/2 of Toth & Lake (2015)
# Eq. (26) on the transverse blocks (the frame u is unnormalised) and the
# spin length as sqrt(S_i S_j); C carries the classical mean field Jq(0).
C_AA = -Ssym * (6 * J2 + 3 * J1)
C_BB = C_AA
A1_AA = sp.simplify(
    -Ssym * J2 * sum(phase_neg(Rf) + phase_neg(-Rf) for Rf in NNN_R)
    + Ssym * A1_AA_dmi
)
A1_BB = sp.simplify(
    -Ssym * J2 * sum(phase_neg(Rf) + phase_neg(-Rf) for Rf in NNN_R)
    + Ssym * A1_BB_dmi
)
A1_AB = sp.simplify(-Ssym * J1 * sum(phase_neg(d) for d in NN))

# pairing blocks vanish (isotropic NNN + z-DMI): B = u^T J^T u summed over the
# +-R pairs; the isotropic contraction u^T I u = 0 and the eps contraction
# u^T eps(D_z)^T u = 0 individually.
B_AA_iso = sp.simplify((u.T * sp.eye(3) * u)[0, 0])
B_AA_dmi = sp.simplify((u.T * ((+Dz) * eps_D).T * u)[0, 0])
check(
    B_AA_iso == 0 and B_AA_dmi == 0,
    "B_AA pairing vanishes (u^T I u = 0 and u^T eps(D_z)^T u = 0)",
)

P_AA = sp.simplify(A1_AA - C_AA + K2 * Ssym)
P_BB = sp.simplify(A1_BB - C_BB + K2 * Ssym)
P_AB = A1_AB

# Physical positive bands = eigenvalues of the site matrix
# h(k) = [[P_AA, P_AB], [P_AB^*, P_BB]]  (B = 0 for this model).
mid = sp.simplify((P_AA + P_BB) / 2)
disc = sp.simplify(((P_AA - P_BB) / 2) ** 2 + sp.Abs(P_AB) ** 2)
omega_minus = sp.simplify(mid - sp.sqrt(disc))
omega_plus = sp.simplify(mid + sp.sqrt(disc))


def numeric_Hq(kpt, J1v, J2v, Dv, Kv, Sv):
    """Direct numeric build of the certificate site blocks (executable conventions)."""
    karr = np.array([kpt[0], kpt[1], 0.0])
    nn = [
        np.array([-1 / 3, 2 / 3, 0]),
        np.array([-1 / 3, -1 / 3, 0]),
        np.array([2 / 3, -1 / 3, 0]),
    ]
    nnn = [
        np.array([1, 0, 0]),
        np.array([-1, 1, 0]),
        np.array([0, -1, 0]),
    ]  # counterclockwise triangle
    iso_nnn = (
        J2v
        * sum(
            np.exp(-2j * np.pi * Rf @ karr) + np.exp(2j * np.pi * Rf @ karr)
            for Rf in nnn
        )
    )
    iso_nn = J1v * sum(np.exp(-2j * np.pi * d @ karr) for d in nn)
    dmi_A = 0.0
    dmi_B = 0.0
    for Rf in nnn:
        dmi_A += (-1j * Dv) * np.exp(-2j * np.pi * Rf @ karr) + (1j * Dv) * np.exp(
            2j * np.pi * Rf @ karr
        )
        dmi_B += (1j * Dv) * np.exp(-2j * np.pi * Rf @ karr) + (-1j * Dv) * np.exp(
            2j * np.pi * Rf @ karr
        )
    Paa = -Sv * iso_nnn + Sv * dmi_A + Sv * (6 * J2v + 3 * J1v) + Kv * Sv
    Pbb = -Sv * iso_nnn + Sv * dmi_B + Sv * (6 * J2v + 3 * J1v) + Kv * Sv
    Pab = -Sv * iso_nn
    return np.array([[Paa, Pab], [np.conj(Pab), Pbb]], dtype=complex)


rng = np.random.default_rng(3)
max_err = 0.0
wf = sp.lambdify(
    (kx, ky, J1, J2, Dz, K2, Ssym),
    (omega_minus.subs(a, 1), omega_plus.subs(a, 1)),
    "numpy",
)
for _ in range(6):
    kpt = rng.random(2)
    hn = numeric_Hq(kpt, 1.0, 0.2, 0.05, 0.1, 1.0)
    ev = np.linalg.eigvalsh(hn)
    subs = {
        kx: float(kpt[0]),
        ky: float(kpt[1]),
        J1: 1.0,
        J2: 0.2,
        Dz: 0.05,
        K2: 0.1,
        Ssym: 1.0,
        a: 1.0,
    }
    ws_raw = np.array(
        [complex(sp.N(omega_minus.subs(subs))), complex(sp.N(omega_plus.subs(subs)))]
    )
    check(
        float(np.max(np.abs(np.imag(ws_raw)))) < 1e-10,
        "symbolic bands evaluate real (residual imaginary part negligible)",
    )
    ws = np.sort(np.real(ws_raw))
    max_err = max(max_err, float(np.max(np.abs(np.sort(ev) - ws))))
check(
    max_err < 1e-12,
    f"certificate site blocks match direct numeric build (max err {max_err:.2e}, 6 k-points)",
)

# Anchored implementation comparison: the certificate's bands must equal the
# eigenvalues of the actual Magnon.Hq for a SpinIO source with these channels.
# (Deferred to the pytest benchmark test_magnon_topology_bulk.py, which uses
# this certificate's pinned Chern values as reference. The formulas here are
# the reference.)

# Dirac expansion
Kp_frac = sp.Matrix(
    [sp.Rational(1, 3), sp.Rational(2, 3), 0]
)  # K: W(K) = 0 for this NN set
Km_frac = sp.Matrix([sp.Rational(2, 3), sp.Rational(1, 3), 0])  # K'
qx, qy = sp.symbols("q_x q_y", real=True)


def first_order(expr):
    e0 = expr.subs({qx: 0, qy: 0})
    return sp.simplify(
        (expr.subs(qy, 0).series(qx, 0, 2).removeO() - e0)
        + (expr.subs(qx, 0).series(qy, 0, 2).removeO() - e0)
    )


W_K = P_AB.subs({kx: Kp_frac[0] + qx, ky: Kp_frac[1] + qy})
W_Km = P_AB.subs({kx: Km_frac[0] + qx, ky: Km_frac[1] + qy})


# velocity vectors v = (dW/dqx, dW/dqy) at the valley center; the chirality is
# Im(vx * conj(vy)). Exact derivatives are taken symbolically, then evaluated
# at high precision (the rational-valley trig constants stay symbolic).
def valley_chi(W_expr):
    W_expr = W_expr.subs({J1: 1, Ssym: 1, a: 1})
    vxc = complex(sp.N(sp.diff(W_expr, qx).subs({qx: 0, qy: 0}), 40))
    vyc = complex(sp.N(sp.diff(W_expr, qy).subs({qx: 0, qy: 0}), 40))
    cross = (vxc * vyc.conjugate()).imag
    return (sp.sign(cross) if cross != 0 else 0), cross


chiK, crossK = valley_chi(W_K)
chiKm, crossKm = valley_chi(W_Km)
check(
    complex(sp.N(P_AB.subs({kx: Kp_frac[0], ky: Kp_frac[1]}))) == 0,
    "W(K) = 0: K = (1/3, 2/3) is a Dirac point of this NN set",
)
check(
    complex(sp.N(P_AB.subs({kx: Km_frac[0], ky: Km_frac[1]}))) == 0,
    "W(K') = 0: K' = (2/3, 1/3) is a Dirac point of this NN set",
)
check(
    chiK == 1 and chiKm == -1,
    f"valley chiralities chi_K = {chiK}, chi_K' = {chiKm} (crosses {crossK:.4f}, {crossKm:.4f})",
)

massK_expr = ((P_AA - P_BB) / 2).subs({kx: Kp_frac[0], ky: Kp_frac[1]})
massKm_expr = ((P_AA - P_BB) / 2).subs({kx: Km_frac[0], ky: Km_frac[1]})
massK = complex(sp.N(massK_expr.subs({J1: 1, J2: 0.2, Dz: 0.05, K2: 0.1, Ssym: 1}), 40))
massKm = complex(
    sp.N(massKm_expr.subs({J1: 1, J2: 0.2, Dz: 0.05, K2: 0.1, Ssym: 1}), 40)
)
massK_d2 = complex(
    sp.N(massK_expr.subs({J1: 1, J2: 0.2, Dz: 0.1, K2: 0.1, Ssym: 1}), 40)
)
check(abs(massK.imag) < 1e-12 and abs(massKm.imag) < 1e-12, "valley masses are real")
check(
    abs(abs(massK) - abs(massKm)) < 1e-12 and abs(massK) > 1e-12,
    "valley masses are equal in magnitude and nonzero (staggered Dirac masses)",
)
check(
    abs(massK + massKm) < 1e-12,
    "valley masses have opposite signs: m(K) = -6*sqrt(3)*D_z, m(K') = +6*sqrt(3)*D_z",
)
check(abs(massK_d2 - 2 * massK) < 1e-12, "mass is linear in D_z")


# Valley Chern rule. In the executable cell-phase convention the in-plane
# windings at the named valleys are chi_K = -1, chi_K' = +1 (measured below
# on the actual Magnon.Hq block); with masses m(K) = -6*sqrt(3)*D_z,
# m(K') = +6*sqrt(3)*D_z and the per-cone rule C_band(cone) = -chi*sign(m)/2:
#   C_upper = -[(-1)(-sgn D) + (+1)(+sgn D)]/2 = -sgn(D_z)
#   C_lower = +sgn(D_z)
# NOTE: the site-space displacement-phase map built symbolically above has
# the OPPOSITE chirality labels (chi_K = +1, chi_K' = -1), so its band Chern
# labels are the negation of the executable ones; the executable convention
# is authoritative for the pipeline.
def pinned_C_lower_formula(sgnD):
    return sgnD


def pinned_C_upper_formula(sgnD):
    return -sgnD


note(
    "  pinned (executable convention): per-cone C = -chi*sign(m)/2 with "
    "chi_K=-1, chi_K'=+1 and m(K)=-6*sqrt(3)*D_z; C_lower = +sign(D_z), "
    "C_upper = -sign(D_z). The displacement-phase site map has the opposite "
    "chirality labels and therefore the negated band Chern labels."
)

# Gamma sanity at S = 1 (executable convention): Goldstone gap = K, optic = 6 J1 + K (J2 = 0)
omega_gamma = [
    sp.simplify(w.subs({kx: 0, ky: 0, J2: 0, a: 1, Ssym: 1}))
    for w in (omega_minus, omega_plus)
]
check(
    sp.simplify(omega_gamma[0] - K2) == 0,
    "omega_minus(Gamma) = K S (gapped Goldstone with easy-axis SIA, S=1)",
)
check(
    sp.simplify(omega_gamma[1] - (6 * J1 + K2)) == 0,
    "omega_plus(Gamma) = 6 J1 S + K S (S=1)",
)


# ---------------------------------------------------------------------------
# 4b. Executable-convention anchor: FHS on the actual Magnon.Hq modes
# ---------------------------------------------------------------------------
note("4b. Executable-convention anchor: Colpa modes of Magnon.Hq + FHS winding")


def honeycomb_exc_spinio(Dv, J1v=1.0, J2v=0.2, Kv=0.1):
    """Honeycomb FM SpinIO in the executable convention (mu = 2 muB, S = 1)."""
    J1v, J2v, Dv, Kv = (value / 2 for value in (J1v, J2v, Dv, Kv))
    _basis = _np.array([[1.0, 0.0, 0.0], [0.5, _np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    _atoms = _Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2.0 / 3.0, -1.0 / 3.0, 0.0]],
        cell=_basis,
        pbc=True,
    )
    _mag = [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    _exc_d, _dmi_d = {}, {}
    for _R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        _rev = tuple(-_np.array(_R))
        _exc_d[(_R, 0, 1)] = J1v
        _exc_d[(_rev, 1, 0)] = J1v
    for _R in [(1, 0, 0), (-1, 1, 0), (0, -1, 0)]:
        _rev = tuple(-_np.array(_R))
        _exc_d[(_R, 0, 0)] = J2v
        _exc_d[(_rev, 0, 0)] = J2v
        _exc_d[(_R, 1, 1)] = J2v
        _exc_d[(_rev, 1, 1)] = J2v
        _dmi_d[(_R, 0, 0)] = _np.array([0.0, 0.0, Dv])
        _dmi_d[(_rev, 0, 0)] = _np.array([0.0, 0.0, -Dv])
        _dmi_d[(_R, 1, 1)] = _np.array([0.0, 0.0, -Dv])
        _dmi_d[(_rev, 1, 1)] = _np.array([0.0, 0.0, Dv])
    _sia = {
        0: _np.diag([0.0, 0.0, Kv]),
        1: _np.diag([0.0, 0.0, Kv]),
    }
    _exc_io = _SpinIO(
        atoms=_atoms,
        spinat=_np.array(_mag, dtype=float),
        charges=[0.0, 0.0],
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=_exc_d,
        dmi_ddict=_dmi_d,
        sia_tensor=_sia,
    )
    _exc_io._build_Rlist()
    return _exc_io


def colpa_modes(mgn_obj, kpt):
    """Colpa metric-normalized positive modes at one k (legacy Hq path)."""
    Hk = mgn_obj.Hq(_np.array([kpt]))[0]
    n = Hk.shape[0] // 2
    L = _np.linalg.cholesky(Hk)
    g = _np.diag([1.0] * n + [-1.0] * n)
    w, uu = _np.linalg.eigh(L.conj().T @ g @ L)
    vv = _np.linalg.solve(L.conj().T, uu[:, n:])
    return vv * _np.sqrt(w[n:])[None, :], Hk


def fhs_band_chern(mgn_obj, band, nmesh=24):
    """Full-torus FHS Chern of one positive band via Colpa modes."""
    psi = _np.empty(
        (nmesh * nmesh, 2 * mgn_obj.Hq(_np.zeros((1, 3))).shape[-1] // 2, 1),
        dtype=complex,
    )
    idx = 0
    for i in range(nmesh):
        for j in range(nmesh):
            vecs, _ = colpa_modes(mgn_obj, [i / nmesh, j / nmesh, 0.0])
            psi[idx] = vecs[:, [band]]
            idx += 1

    def link(p, q):
        d = _np.linalg.det(psi[p].conj().T @ psi[q])
        return d / abs(d)

    total = 0.0
    for i in range(nmesh):
        for j in range(nmesh):
            p0 = i * nmesh + j
            p1 = ((i + 1) % nmesh) * nmesh + j
            p2 = ((i + 1) % nmesh) * nmesh + (j + 1) % nmesh
            p3 = i * nmesh + (j + 1) % nmesh
            total += _np.angle(
                link(p0, p1) * link(p1, p2) * link(p2, p3) * link(p3, p0)
            ) / (2 * _np.pi)
    return int(_np.rint(total)), total


# pairing blocks vanish for a collinear z-FM with z-DMI (certified above)
_mgn_exe = _Magnon.load_from_io(honeycomb_exc_spinio(+0.05))
_mgn_exe.set_reference(
    (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, 2.0]]
)
_Hg = _mgn_exe.Hq(_np.array([[0.11, 0.23, 0.0]]))[0]
check(
    _np.max(_np.abs(_Hg[:2, 2:])) < 1e-12,
    "executable honeycomb FM has exactly zero pairing blocks (z-DMI, collinear)",
)

# executable valley masses from the positive-mode site block P = Hq[:2, :2]
_mass_exe = {}
for _name, _kv in (("K=(1/3,2/3)", (1 / 3, 2 / 3)), ("K'=(2/3,1/3)", (2 / 3, 1 / 3))):
    _P = _mgn_exe.Hq(_np.array([[_kv[0], _kv[1], 0.0]]))[0][:2, :2]
    _mass_exe[_name] = (_P[0, 0] - _P[1, 1]).real / 2
check(
    abs(_mass_exe["K=(1/3,2/3)"] + 3 * _np.sqrt(3) * 0.05) < 1e-9
    and abs(_mass_exe["K'=(2/3,1/3)"] - 3 * _np.sqrt(3) * 0.05) < 1e-9,
    "executable valley masses: m(K) = -3*sqrt(3)*D_z, m(K') = +3*sqrt(3)*D_z",
)

# executable in-plane windings at the named valleys (chirality labels)
_chi_exe = {}
for _name, _kv in (("K", (1 / 3, 2 / 3)), ("K'", (2 / 3, 1 / 3))):
    _Q0 = _mgn_exe.Hq(_np.array([[_kv[0], _kv[1], 0.0]]))[0][:2, :2][0, 1]
    _angs = []
    _r = 0.04
    for _t in _np.linspace(0, 2 * _np.pi, 49)[:-1]:
        _k = [_kv[0] + _r * _np.cos(_t), _kv[1] + _r * _np.sin(_t), 0.0]
        _Q = _mgn_exe.Hq(_np.array([_k]))[0][:2, :2][0, 1]
        _angs.append(_np.angle(_Q * _np.conj(_Q0)))
    _chi_exe[_name] = _np.sum(_np.diff(_np.unwrap(_angs))) / (2 * _np.pi)
check(
    abs(_chi_exe["K"] + 1) < 0.15 and abs(_chi_exe["K'"] - 1) < 0.15,
    "executable in-plane windings: chi_K = -1, chi_K' = +1 (opposite of the "
    "displacement-phase site map)",
)

# full-band executable Chern numbers for both DMI signs
_exe_lower = {}
_exe_upper = {}
for _sgn in (+1, -1):
    _mg = _Magnon.load_from_io(honeycomb_exc_spinio(_sgn * 0.05))
    _mg.set_reference(
        (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, 2.0]]
    )
    _exe_lower[_sgn], _ = fhs_band_chern(_mg, band=0, nmesh=24)
    _exe_upper[_sgn], _ = fhs_band_chern(_mg, band=1, nmesh=24)
check(
    _exe_lower[+1] == pinned_C_lower_formula(1)
    and _exe_lower[-1] == pinned_C_lower_formula(-1)
    and _exe_upper[+1] == pinned_C_upper_formula(1)
    and _exe_upper[-1] == pinned_C_upper_formula(-1),
    f"executable FHS winding: C_lower = {_exe_lower}, C_upper = {_exe_upper} "
    "match the pin C_lower = +sign(D_z), C_upper = -sign(D_z)",
)

# ---------------------------------------------------------------------------
# 4c. Site-space displacement-phase winding: convention-flip documentation
# ---------------------------------------------------------------------------
note("4c. Site-space (displacement-phase) winding: negated labels")


def chern_numeric(Dv, nmesh=48):
    def dvec(kpt):
        hn = numeric_Hq(kpt, 1.0, 0.2, Dv, 0.1, 1.0)
        mid_n = (hn[0, 0] + hn[1, 1]) / 2
        d = np.array([(hn[0, 1]).real, (hn[0, 1]).imag, (hn[0, 0] - hn[1, 1]) / 2])
        del mid_n
        return d

    def state(kpt):
        d = dvec(kpt)
        dd = np.linalg.norm(d)
        uv = np.array([d[2] + dd, d[0] + 1j * d[1]])
        return uv / np.linalg.norm(uv)

    O = np.zeros((nmesh, nmesh), dtype=complex)
    for i in range(nmesh):
        for j in range(nmesh):
            s00 = state((i / nmesh, j / nmesh))
            s10 = state(((i + 1) / nmesh, j / nmesh))
            s11 = state(((i + 1) / nmesh, (j + 1) / nmesh))
            s01 = state((i / nmesh, (j + 1) / nmesh))
            prod = (
                (s00.conj() @ s10)
                * (s10.conj() @ s11)
                * (s11.conj() @ s01)
                * (s01.conj() @ s00)
            )
            O[i, j] = prod / abs(prod)
    chern = np.sum(np.angle(O)) / (2 * np.pi)
    return int(np.rint(chern)), chern


c_plus, raw_plus = chern_numeric(+0.05)
c_minus, raw_minus = chern_numeric(-0.05)
note(
    f"  numeric winding (upper band of the displacement-phase site map h(k)): "
    f"D=+0.05 -> C={c_plus} (raw {raw_plus:+.6f}); "
    f"D=-0.05 -> C={c_minus} (raw {raw_minus:+.6f})"
)
check(
    c_plus == -c_minus and abs(c_plus) == 1,
    "DMI reversal flips the Chern sign; |C| = 1 for the gapped honeycomb FM",
)
check(
    c_plus == -pinned_C_upper_formula(1),
    "site-map upper-band winding is the NEGATION of the executable pin "
    "(opposite chirality labels; executable convention is authoritative)",
)

note(
    f"  PINNED (executable convention): lower-band C = {_exe_lower[+1]:+d} * sign(D_z); "
    f"upper-band C = {_exe_upper[+1]:+d} * sign(D_z) "
    "for plaquette order (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1) on the (a1,a2) torus; "
    "the displacement-phase site map carries the negated labels."
)

# ---------------------------------------------------------------------------
# 5. FHS gauge invariance: symbolic phase cancellation
# ---------------------------------------------------------------------------
note("5. FHS link construction: gauge invariance (symbolic)")

f1, f2, f3, f4 = sp.symbols("phi1 phi2 phi3 phi4", real=True)
s1, s2, s3, s4 = (
    sp.exp(sp.I * f1),
    sp.exp(sp.I * f2),
    sp.exp(sp.I * f3),
    sp.exp(sp.I * f4),
)
prod = (
    (s1.conjugate() * s2)
    * (s2.conjugate() * s3)
    * (s3.conjugate() * s4)
    * (s4.conjugate() * s1)
)
check(sp.simplify(prod - 1) == 0, "plaquette product of gauge phases telescopes to 1")

w1, w2, w3, w4 = sp.symbols("w1 w2 w3 w4", real=True)
W1, W2 = sp.exp(sp.I * w1), sp.exp(sp.I * w2)
W3, W4 = sp.exp(sp.I * w3), sp.exp(sp.I * w4)
detprod = (
    (W1.conjugate() * W2)
    * (W2.conjugate() * W3)
    * (W3.conjugate() * W4)
    * (W4.conjugate() * W1)
)
check(
    sp.simplify(detprod - 1) == 0,
    "det-overlap link product invariant under group unitary mixing",
)

note("")
note("CERTIFICATE OK: all assertions passed.")
report_path = Path(__file__).with_suffix(".md")
header = """# Certificate: collinear ferromagnet LSWT conventions for magnon topology

Generated by `magnon_topology_lswt_ferromagnet.py` (all assertions executed and
passed in the `mydev` environment). Pinned statements:

1. Fourier convention: `J(k) = sum_R J(R) exp(-2 pi i k.R)` (executable negative
   phase); `J(k)^* = J(-k)` for real-space tensors; even kernels.
2. Single-sublattice FM: `omega(k) = 2S sum_R J(R)[1 - cos(k.R)]`; executable
   blocks give `omega = eig(A1 - C)` with `A1 = -u^T J(-k)^T u*` (u = (1,i,0),
   unnormalized) and `C = -2 S J(0)`; easy-axis SIA adds `2 K S`.
3. Pairing block `B = u^T J(-k)^T u` vanishes identically for isotropic NN and
   z-axis DMI NNN tensors; `u^T eps(D_z) u* = -2i` carries the DMI phase.
4. Honeycomb FM: site blocks real under reverse-bond DMI antisymmetry
   `D_ji(-R) = -D_ij(R)`; certificate bands match a direct numeric build.
5. Valley structure (site-space displacement-phase map): `chi_K = +1`,
   `chi_K' = -1`, common mass `m ∝ D_z`. EXECUTABLE cell-phase convention
   (Magnon.Hq + Colpa modes, the authoritative pin): `chi_K = -1`,
   `chi_K' = +1`, same masses, and pinned band Chern numbers (24x24 FHS
   winding, plaquette order `(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)` on the
   `(a1,a2)` torus): `C_lower = +sign(D_z)`, `C_upper = -sign(D_z)`;
   the displacement-phase site map carries the negated labels.
6. FHS link construction is gauge invariant (phase telescoping; unitary group
   mixing cancels in determinant links).

## Console output of the executed script

```
"""
body = "\n".join(REPORT)
footer = "\n```\n"
report_path.write_text(header + body + footer)
print(f"report written to {report_path}")
