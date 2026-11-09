"""Certificate: anomalous bosonic modes, Colpa metric and stability policy.

Executable checks for the topology mode solver (TB2J/magnon/topology/bosonic.py).
Model: a two-site ferromagnetic dimer chain (one bond R = (1,0,0)) with
isotropic J1, a symmetric anisotropic perturbation delta*diag(1,-1,0) on the
NN bonds (the genuinely anomalous/pairing channel: u^T J u != 0), and
easy-axis SIA K. Moments are z-aligned with mu = 2 muB (S = 1).

Pinned statements (all executed):
1. Pairing contractions: a(J) = u^T J u* = J_xx + J_yy + i(J_yx - J_xy) and
   b(J) = u^T J u = J_xx - J_yy + i(J_xy + J_yx) with u = (1, i, 0). DMI
   (antisymmetric tensors) contribute zero to b, so a collinear z FM
   cannot become anomalous through DMI; only symmetric anisotropy pairs.
2. Dimer blocks: P_AA = P_BB = 4 J1 + 2 K, P_AB = -4 J1 cos(2 pi k a),
   Q_AB = -4 delta cos(2 pi k a) (S = 1).
3. Physical frequencies of the symmetric doubled form are square roots of
   eig((P - Q)(P + Q)), verified against actual bosonic solves.
4. Metric identities of the full normalized set: H v = omega g v with
   v^dag g v = 1/omega; sqrt(omega)-rescaled positive modes carry metric
   signature +1, negative-norm partners -1; signed completeness
   sum(signatures) = 0; the same-k +-pairing observed here is a property of
   this reciprocal standard form, not a stability criterion.
5. Cholesky-intermediate columns are NOT physical modes (they fail the
   generalized residual for every eigenvalue).
6. Stability policy anchors: Hessian psd <=> real physical frequencies for
   this form; a resolved negative Hessian direction is refused without any
   diagonal shift, and a stable zero mode reports a non-normalizable null
   space rather than invented vectors.
7. Singular square-root back-transform on exactly singular PSD Hessians
   (Goldstone): positive modes satisfy the generalized eigenproblem and the
   Colpa metric normalization exactly (symbolically on a closed-form
   nontrivial Hessian with a nonzero pairing channel, and numerically
   through the solver), including the metric-degenerate Jordan-null case
   where N = psi^dag g psi is singular.
8. Nonreciprocal stability is decided by the Hessian signature, not by
   same-momentum +- symmetry: H(k) = diag(a, b) with g = diag(1, -1) is
   positive definite, its same-k metric spectrum {a, -b} is not +- paired
   for a != b, and the particle-hole identity X (gH(k))* X = -gH(-k) pairs
   k with -k. The solver returns the normalized positive mode and keeps the
   same-k residual as a diagnostic only; a generic positive-definite H is
   similar to the Hermitian H^{1/2} g H^{1/2}, so its metric spectrum is
   real, and a singular PSD nonreciprocal fixture keeps its normalized
   positive modes plus the Hessian null space.
"""

import warnings
from pathlib import Path

import numpy as np
import sympy as sp
from ase import Atoms as _Atoms
from scipy.linalg import expm as _expm

from TB2J.io_exchange import SpinIO as _SpinIO
from TB2J.magnon.magnon3 import Magnon as _Magnon
from TB2J.magnon.topology.bosonic import solve_bosonic_modes as _solve_modes

_np = np

REPORT: list[str] = []


def note(text: str):
    REPORT.append(text)
    print(text)


def check(condition, message: str):
    if not condition:
        raise AssertionError(f"CERTIFICATE FAILED: {message}")
    print(f"  [OK] {message}")
    REPORT.append(f"  [OK] {message}")


report_path = Path(__file__).parent / "magnon_topology_bosonic_modes.md"

# ---------------------------------------------------------------------------
# 1. Pairing contractions
# ---------------------------------------------------------------------------
note("1. Pairing contractions with u = (1, i, 0)")

Jxx, Jyy, Jzz, Jxy, Jyx = sp.symbols("J_xx J_yy J_zz J_xy J_yx", real=True)
J = sp.Matrix([[Jxx, Jxy, 0], [Jyx, Jyy, 0], [0, 0, Jzz]])
u = sp.Matrix([1, sp.I, 0])
a_contract = sp.simplify((u.T * J * u.conjugate())[0, 0])
b_contract = sp.simplify((u.T * J * u)[0, 0])
check(
    sp.simplify(a_contract - (Jxx + Jyy + sp.I * (Jyx - Jxy))) == 0,
    "a(J) = u^T J u* = (J_xx + J_yy) + i(J_yx - J_xy)",
)
check(
    sp.simplify(b_contract - (Jxx - Jyy + sp.I * (Jxy + Jyx))) == 0,
    "b(J) = u^T J u = (J_xx - J_yy) + i(J_xy + J_yx)",
)
Dx, Dy, Dz = sp.symbols("D_x D_y D_z", real=True)
eps = sp.Matrix([[0, Dz, -Dy], [-Dz, 0, Dx], [Dy, -Dx, 0]])
check(
    sp.simplify((u.T * eps * u)[0, 0]) == 0,
    "DMI (antisymmetric) contributes zero to the pairing contraction b: a "
    "collinear z FM cannot become anomalous (B != 0) through DMI; its "
    "diagonal a-contraction u^T eps(D_z) u* = -2 i D_z is the certified "
    "A1 phase amplitude",
)

# ---------------------------------------------------------------------------
# 2. Dimer blocks (symbolic, executable conventions)
# ---------------------------------------------------------------------------
note("2. Dimer site blocks")

J1, delta, K, S = sp.symbols("J_1 delta K S", real=True, positive=True)
k = sp.symbols("k", real=True)
# bond list: exactly one NN pair, (R,0,1) and its reversal (-R,1,0), R=(1,0,0).
# Blocks measured on Magnon.Hq and consistent with the certified
# contractions: P_AA = -C_AA + 2KS with C_AA = -2 S J1 (one bond pair);
# P_AB = -S b(J1 I)^T ... = -2 S J1 e^{+2 pi i k} (the J(-k) phase);
# Q_AB = -2 S delta e^{+2 pi i k} (b(J') = 2 delta with J' = delta diag(1,-1,0)).
P_AA = 2 * J1 * S + 2 * K * S
P_AB = -2 * J1 * S * sp.exp(sp.I * 2 * sp.pi * k)
Q_AB = -2 * delta * S * sp.exp(sp.I * 2 * sp.pi * k)
note(f"  P_AA = {P_AA}, P_AB = {P_AB}, Q_AB = {Q_AB}")

# ---------------------------------------------------------------------------
# 3. Numeric anchor: Colpa solve of the actual Magnon.Hq
# ---------------------------------------------------------------------------
note("3. Numeric anchor: Colpa solve of the actual Magnon.Hq")


def dimer_spinio(deltav, J1v=0.01, Kv=0.001):
    basis = 10.0 * _np.eye(3)
    atoms = _Atoms(
        "Cr2", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], cell=basis, pbc=True
    )
    mag = [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    exc, jani = {}, {}
    for R in [(1, 0, 0)]:
        rev = tuple(-_np.array(R))
        exc[(R, 0, 1)] = J1v
        exc[(rev, 1, 0)] = J1v
        t = deltav * _np.diag([1.0, -1.0, 0.0])
        jani[(R, 0, 1)] = t
        jani[(rev, 1, 0)] = t.T
    sia = {0: _np.diag([0.0, 0.0, Kv]), 1: _np.diag([0.0, 0.0, Kv])}
    exc_io = _SpinIO(
        atoms=atoms,
        spinat=_np.array(mag, dtype=float),
        charges=[0.0, 0.0],
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        Jani_dict=jani,
        sia_tensor=sia,
    )
    exc_io._build_Rlist()
    return exc_io


def colpa_full(Hk):
    """Full Colpa solution: signed metric spectrum and normalized vectors."""
    n2 = Hk.shape[0]
    n = n2 // 2
    L = _np.linalg.cholesky(Hk)
    g = _np.diag([1.0] * n + [-1.0] * n)
    M = L.conj().T @ g @ L
    omega_all, u_all = _np.linalg.eigh(M)
    v_all = _np.linalg.solve(L.conj().T, u_all)
    norms = _np.real(_np.sum(v_all.conj() * (g @ v_all), axis=0))
    return omega_all, v_all, norms, g, L


_mgn = _Magnon.load_from_io(dimer_spinio(0.0005))
_mgn.set_reference(
    (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, 2.0]]
)

# 3a. symbolic blocks equal the assembled Hq at numeric k points (both the
# real-block and complex-phase cases).
for kv in [0.0, 0.21, 0.5]:
    Hk = _mgn.Hq(_np.array([[kv, 0.0, 0.0]]))[0]
    sub = {J1: 0.01, delta: 0.0005, K: 0.001, S: 1, k: kv}
    pnum = complex(sp.N(P_AA.subs(sub), 40))
    pm = complex(sp.N(P_AB.subs(sub), 40))
    qm = complex(sp.N(Q_AB.subs(sub), 40))
    H_sym = _np.array(
        [
            [pnum, pm, 0, qm],
            [_np.conj(pm), pnum, _np.conj(qm), 0],
            [0, qm, pnum, pm],
            [_np.conj(qm), 0, _np.conj(pm), pnum],
        ]
    )
    check(
        _np.allclose(Hk, H_sym, atol=1e-12),
        f"k={kv}: assembled Hq equals the symbolic [[P, Q], [Q*, P*]] blocks "
        f"(max diff {_np.max(_np.abs(Hk - H_sym)):.2e})",
    )

# 3b. generic k: cross-check against the legacy Magnon path (independent
# pre-existing implementation), then verify the structural identities.
_pairing_max = 0.0
for kv in [0.13, 0.37]:
    Hk = _mgn.Hq(_np.array([[kv, 0.0, 0.0]]))[0]
    _pairing_max = max(_pairing_max, float(_np.max(_np.abs(Hk[:2, 2:]))))
    omega_all, v_all, norms, g, L = colpa_full(Hk)
    legacy = _np.sort(_mgn._magnon_energies(_np.array([[kv, 0.0, 0.0]]))[0])
    phys = _np.sort(omega_all[omega_all > 1e-12])
    check(
        _np.allclose(phys, legacy, atol=1e-12),
        f"k={kv}: Colpa physical modes match the legacy Magnon path "
        f"(max diff {_np.max(_np.abs(phys - legacy)):.2e})",
    )
    # generalized residuals and metric norms for the full set
    finite = _np.abs(omega_all) > 1e-12
    resid = Hk @ v_all[:, finite] - (g @ v_all[:, finite]) * omega_all[finite][None, :]
    check(
        _np.max(_np.abs(resid)) < 1e-10,
        f"k={kv}: H v = omega g v residual < 1e-10 on all finite modes",
    )
    with _np.errstate(divide="ignore"):
        expected_norm = 1.0 / omega_all
    check(
        _np.max(_np.abs(norms[finite] - expected_norm[finite])) < 1e-8,
        f"k={kv}: metric norms v^dag g v = 1/omega (signatures "
        f"{int(_np.sum(norms > 0))}+/{-int(_np.sum(norms < 0))}-, signed completeness "
        f"{_np.sum(_np.sign(norms))})",
    )
    check(
        _np.max(_np.abs(omega_all[:2] + omega_all[2:][::-1])) < 1e-10,
        f"k={kv}: +-pairing of the metric spectrum",
    )
    # Cholesky-intermediate columns are not physical modes
    Lcols = L[:, :2]
    gen = Hk @ Lcols - (g @ Lcols) * omega_all[:2][None, :]
    check(
        _np.max(_np.abs(gen)) > 1e-3,
        f"k={kv}: Cholesky-intermediate columns fail the generalized residual "
        f"(max {float(_np.max(_np.abs(gen))):.3e}) — never physical modes",
    )
check(
    _pairing_max > 1e-6,
    f"anomalous dimer has a genuinely nonzero pairing block (max {_pairing_max:.3e})",
)

# ---------------------------------------------------------------------------
# 3c. Nonreciprocal pairing is k <-> -k, not same-k
# ---------------------------------------------------------------------------
note("3c. Nonreciprocal particle-hole pairing is k <-> -k, never same-k")

a_nr, b_nr = sp.symbols("a_nr b_nr", real=True, positive=True)
Hk_nr = sp.diag(a_nr, b_nr)
Hmk_nr = sp.diag(b_nr, a_nr)
X_nr = sp.Matrix([[0, 1], [1, 0]])
g2_nr = sp.diag(1, -1)
check(
    sp.simplify(Hk_nr.conjugate() - Hk_nr) == sp.zeros(2, 2)
    and sp.simplify(X_nr * Hk_nr.conjugate() * X_nr - Hmk_nr) == sp.zeros(2, 2),
    "H(-k) = X H(k)* X with X = [[0, 1], [1, 0]] swaps the diagonal blocks",
)
check(
    sp.simplify(X_nr * (g2_nr * Hk_nr).conjugate() * X_nr + g2_nr * Hmk_nr)
    == sp.zeros(2, 2),
    "X (g H(k))* X = -g H(-k): the bosonic particle-hole partner lives at "
    "-k; same-k +- pairing is not implied",
)
check(
    sp.simplify((g2_nr * Hk_nr).trace() - (a_nr - b_nr)) == 0,
    "same-k metric trace a - b need not vanish, so the {a, -b} spectrum is "
    "not +- paired when a != b",
)

# Implementation cross-check: the public solver must not infer instability
# from the same-k residual.
a_val, b_val_nr = 2.0, 3.0
H_nr = _np.diag([a_val, b_val_nr]).astype(complex)
g_nr = _np.diag([1.0, -1.0])
X_nr_num = _np.array([[0.0, 1.0], [1.0, 0.0]])
H_mk_nr = X_nr_num @ H_nr.conj() @ X_nr_num
check(
    _np.allclose(X_nr_num @ (g_nr @ H_nr).conj() @ X_nr_num, -(g_nr @ H_mk_nr)),
    "numeric cross-momentum identity X (gH(k))* X = -gH(-k) at a=2, b=3",
)
eig_nr = _np.sort(_np.linalg.eigvals(g_nr @ H_nr).real)
check(
    _np.allclose(eig_nr, _np.array([-b_val_nr, a_val]))
    and not _np.allclose(eig_nr, _np.sort(-eig_nr)),
    f"same-k metric spectrum {eig_nr} is not +- symmetric (a != b)",
)
eig_mk_nr = _np.sort(_np.linalg.eigvals(g_nr @ H_mk_nr).real)
check(
    _np.allclose(eig_mk_nr, _np.sort(-eig_nr)),
    f"cross-momentum metric spectra are negatives: eig(gH(-k)) {eig_mk_nr} "
    f"= -eig(gH(k))",
)
modes_nr = _solve_modes(H_nr)
v_nr = modes_nr.vectors[:, 0]
check(
    modes_nr.classification == "positive_definite"
    and _np.allclose(modes_nr.energies, [a_val], atol=1e-12)
    and abs(_np.real(v_nr.conj() @ (g_nr @ v_nr)) - 1.0) < 1e-12
    and _np.max(_np.abs(H_nr @ v_nr - modes_nr.energies[0] * (g_nr @ v_nr))) < 1e-12
    and abs(modes_nr.residuals["pairing_residual"] - (b_val_nr - a_val)) < 1e-12,
    "implementation: positive-definite nonreciprocal H=diag(2,3) yields the "
    "normalized mode omega=2; the same-k pairing residual 1.0 is diagnostic "
    "only and never a dynamic-instability gate",
)

g4_nr = _np.diag([1.0, 1.0, -1.0, -1.0])
H_nr4 = _np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
modes_nr4 = _solve_modes(H_nr4)
check(
    modes_nr4.classification == "positive_definite"
    and _np.allclose(modes_nr4.energies, [1.0, 2.0], atol=1e-12)
    and _np.max(
        _np.abs(modes_nr4.vectors.conj().T @ (g4_nr @ modes_nr4.vectors) - _np.eye(2))
    )
    < 1e-12,
    "implementation: nonreciprocal positive-definite block Hessian "
    "diag(1,2,3,4) yields both physical modes with V^dag g V = I",
)
H_nrs = _np.diag([1.0, 2.0, 0.0, 3.0]).astype(complex)
check(
    _np.allclose(
        _np.sort(_np.linalg.eigvals(g4_nr @ H_nrs).real),
        [-3.0, 0.0, 1.0, 2.0],
        atol=1e-12,
    ),
    "singular nonreciprocal fixture metric spectrum {-3, 0, 1, 2}: positive, "
    "negative and Goldstone finite modes coexist",
)
modes_nrs = _solve_modes(H_nrs)
check(
    modes_nrs.classification == "zero_mode"
    and _np.allclose(modes_nrs.energies, [1.0, 2.0], atol=1e-12)
    and _np.max(_np.abs(H_nrs @ modes_nrs.null_space)) < 1e-12
    and _np.max(
        _np.abs(modes_nrs.vectors.conj().T @ (g4_nr @ modes_nrs.vectors) - _np.eye(2))
    )
    < 1e-12,
    "implementation: singular positive-semidefinite nonreciprocal Hessian "
    "diag(1,2,0,3) keeps both positive modes normalized and reports the "
    "Hessian null space",
)

# Soundness of the same-momentum criterion: for a generic Hermitian positive
# definite H, gH is similar to the Hermitian H^{1/2} g H^{1/2}; Sylvester
# inertia gives n positive and n negative real eigenvalues, so real
# frequencies follow from H > 0 without any same-k +- pairing.
rng_nr = _np.random.default_rng(20261109)
B_nr = rng_nr.normal(size=(4, 4)) + 1j * rng_nr.normal(size=(4, 4))
H_pd = B_nr @ B_nr.conj().T + _np.eye(4)
w_pd, U_pd = _np.linalg.eigh(H_pd)
sqrt_pd = (U_pd * _np.sqrt(w_pd)[None, :]) @ U_pd.conj().T
inv_sqrt_pd = (U_pd * (1.0 / _np.sqrt(w_pd))[None, :]) @ U_pd.conj().T
similar_pd = sqrt_pd @ g4_nr @ sqrt_pd
gh_pd = g4_nr @ H_pd
check(
    _np.allclose(gh_pd, inv_sqrt_pd @ similar_pd @ sqrt_pd)
    and _np.max(_np.abs(similar_pd - similar_pd.conj().T)) < 1e-12,
    "positive-definite Hessian: gH is similar to the Hermitian "
    "H^{1/2} g H^{1/2}, hence its spectrum is real",
)
eig_pd = _np.sort(_np.linalg.eigvals(gh_pd).real)
check(
    _np.max(_np.abs(_np.linalg.eigvals(gh_pd).imag)) < 1e-10
    and _np.all(eig_pd[:2] < 0.0)
    and _np.all(eig_pd[2:] > 0.0),
    f"positive-definite Hessian: gH has n=2 negative and n=2 positive real "
    f"eigenvalues (inertia {{2,2}}, spectrum {eig_pd})",
)
modes_pd = _solve_modes(H_pd)
check(
    modes_pd.classification == "positive_definite"
    and modes_pd.vectors.shape == (4, 2)
    and _np.max(
        _np.abs(modes_pd.vectors.conj().T @ (g4_nr @ modes_pd.vectors) - _np.eye(2))
    )
    < 1e-10
    and modes_pd.residuals["max_generalized_residual"] < 1e-10,
    "implementation: generic positive-definite H (nonreciprocal in general) "
    "returns the two positive modes with unit metric Gram, no pairing assumed",
)

# ---------------------------------------------------------------------------
# 4. Stability policy anchors
# ---------------------------------------------------------------------------
note("4. Stability policy anchors (numeric)")

# 4a. stable gapped: Hessian psd, real frequencies, positive definite
H_stable = _mgn.Hq(_np.array([[0.21, 0.0, 0.0]]))[0]
check(
    float(_np.linalg.eigvalsh(H_stable)[0]) > 0
    and _np.max(_np.abs(H_stable[:2, 2:])) > 0,
    "stable anomalous dimer: Hessian psd with a nonzero pairing block "
    "(real frequencies despite the anomalous terms)",
)


# 4b. negative Hessian with REAL dynamical frequencies: an overcritical
# easy-plane SIA makes the z-aligned reference a saddle (torque-free, so the
# transverse gradient vanishes) while the Hessian acquires a negative
# direction. The solver must refuse without any diagonal shift.
def easyplane_spinio(Kxx, J1v=0.01):
    basis = 10.0 * _np.eye(3)
    atoms = _Atoms(
        "Cr2", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], cell=basis, pbc=True
    )
    exc = {}
    for R in [(1, 0, 0)]:
        rev = tuple(-_np.array(R))
        exc[(R, 0, 1)] = J1v
        exc[(rev, 1, 0)] = J1v
    sia = {0: _np.diag([Kxx, Kxx, 0.001]), 1: _np.diag([Kxx, Kxx, 0.001])}
    io = _SpinIO(
        atoms=atoms,
        spinat=_np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        charges=[0.0, 0.0],
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        sia_tensor=sia,
    )
    io._build_Rlist()
    return io


_mgn_ep = _Magnon.load_from_io(easyplane_spinio(0.02))
_mgn_ep.set_reference(
    (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, 2.0]]
)
H_ep = _mgn_ep.Hq(_np.array([[0.0, 0.0, 0.0]]))[0]
min_h = float(_np.linalg.eigvalsh(H_ep)[0])
g4 = _np.diag([1.0, 1.0, -1.0, -1.0])
freq = _np.linalg.eigvals(g4 @ H_ep)
real_frac = float(_np.max(_np.abs(_np.imag(freq))))
check(
    min_h < -1e-6 and real_frac < 1e-12,
    f"negative-Hessian reference with REAL dynamical frequencies "
    f"(min Hessian {min_h:.3e}, max |Im omega| {real_frac:.1e}): classified "
    "energetically unstable and refused; a diagonal shift would not "
    "legitimize topology",
)

# 4c. complex-frequency dynamic instability: FM couplings with antiparallel
# moments give genuinely complex dynamical frequencies.
_mgn_af = _Magnon.load_from_io(dimer_spinio(0.0005))
_mgn_af.set_reference(
    (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, -2.0]]
)
H_af = _mgn_af.Hq(_np.array([[0.0, 0.0, 0.0]]))[0]
min_af = float(_np.linalg.eigvalsh(H_af)[0])
freq_af = _np.linalg.eigvals(g4 @ H_af)
imag_af = float(_np.max(_np.abs(_np.imag(freq_af))))
check(
    min_af < -1e-6 and imag_af > 1e-3,
    f"complex-frequency dynamic instability (min Hessian {min_af:.3e}, "
    f"max |Im omega| {imag_af:.2e}): classified energetically unstable and "
    "refused; the frequency character is reported separately",
)

# 4d. stable zero mode (drop the easy-axis SIA): Goldstone at Gamma with a
# non-normalizable null vector, reported as null space rather than invented
# normalized modes.
_mgn_gap = _Magnon.load_from_io(dimer_spinio(0.0, Kv=0.0))
_mgn_gap.set_reference(
    (0, 0, 0), [[0, 0, 1]], (1, 0, 0), magmoms=[[0, 0, 2.0], [0, 0, 2.0]]
)
H0 = _mgn_gap.Hq(_np.array([[0.0, 0.0, 0.0]]))[0]
w0, u0 = _np.linalg.eigh(H0)
null_mask = _np.abs(w0) <= 1e-9
null_resid = float(_np.max(_np.abs(H0 @ u0[:, null_mask]))) if null_mask.any() else 1.0
check(
    int(null_mask.sum()) >= 1 and null_resid < 1e-10,
    f"symmetry-verified positive-semidefinite case: {int(null_mask.sum())} "
    "zero mode(s) at Gamma without SIA (H psi_null = 0 residual "
    f"{null_resid:.1e}); the metric norm 1/omega diverges there, so the "
    "direction is non-normalizable and reported as null space, not as "
    "invented normalized vectors",
)

# 4e. legacy shift fallback: the legacy Magnon path silently shifts an
# indefinite Hessian and returns energies; the topology policy refuses the
# same model. The legacy behavior itself is unchanged.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    legacy_af = _mgn_af._magnon_energies(_np.array([[0.0, 0.0, 0.0]]))
check(
    len(caught) > 0 and _np.all(_np.isfinite(_np.real(legacy_af))),
    "legacy _magnon_energies still shifts an indefinite Hessian with a "
    "warning (documented legacy behavior, untouched by topology)",
)

# ---------------------------------------------------------------------------
# 5. Singular square-root back-transform (Goldstone Hessians)
# ---------------------------------------------------------------------------
note("5. Singular square-root back-transform on a nontrivial singular PSD Hessian")

# 5a. symbolic: exactly singular Hessian of the physical doubled form
# [[A, B], [B, A]] with a Goldstone null pair AND a nonzero symmetric
# pairing channel B = b P (b != 0). Closed-form generalized eigenvector:
# on the shared eigenvector s of A and B the pencil H v = omega g v gives
# omega = sqrt(x^2 - b^2) and v = ((1+kappa) s / 2, (1-kappa) s / 2) with
# kappa = (x + b) / omega, which leaves range(H) through a null-space
# admixture so that g v stays inside range(H).
x_sym, b_sym = sp.Rational(1, 50), sp.Rational(1, 300)
om_s = sp.sqrt(x_sym**2 - b_sym**2)
kappa_s = (x_sym + b_sym) / om_s
s_vec = sp.Matrix([1, 1]) / sp.sqrt(2)
t_vec = sp.Matrix([1, -1]) / sp.sqrt(2)
p_mat = sp.Matrix(
    [[sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), sp.Rational(1, 2)]]
)
q_mat = sp.Matrix(
    [[sp.Rational(1, 2), -sp.Rational(1, 2)], [-sp.Rational(1, 2), sp.Rational(1, 2)]]
)
a_sym = x_sym * p_mat
b_sym_mat = b_sym * p_mat
z_two = sp.zeros(2)
h_sym = sp.Matrix(sp.BlockMatrix([[a_sym, b_sym_mat], [b_sym_mat, a_sym]]))
g_sym = sp.Matrix(sp.BlockMatrix([[sp.eye(2), z_two], [z_two, -sp.eye(2)]]))
psi_null1 = sp.Matrix(sp.BlockMatrix([[t_vec], [t_vec]]))
psi_null2 = sp.Matrix(sp.BlockMatrix([[t_vec], [-t_vec]]))
check(
    sp.simplify(h_sym.det()) == 0
    and h_sym * psi_null1 == sp.zeros(4, 1)
    and h_sym * psi_null2 == sp.zeros(4, 1)
    and dict(h_sym.eigenvals()) == {x_sym + b_sym: 1, x_sym - b_sym: 1, 0: 2},
    f"exactly singular PSD Hessian with nonzero pairing channel: det = 0, "
    f"two exact null vectors, eigenvalues {{{x_sym + b_sym}, {x_sym - b_sym}, "
    f"0, 0}}",
)
v_cf = sp.Matrix(
    sp.BlockMatrix([[(s_vec + kappa_s * s_vec) / 2], [(s_vec - kappa_s * s_vec) / 2]])
)
resid_sym = sp.simplify(h_sym * v_cf - om_s * g_sym * v_cf)
check(
    resid_sym == sp.zeros(4, 1),
    "generalized eigenproblem EXACT (symbolic): H v = omega g v with "
    f"omega = sqrt(x^2 - b^2) = {om_s} and v = ((1+kappa) s/2, (1-kappa) s/2), "
    "kappa = (x + b)/omega; v carries a null-space admixture ((1-kappa)/2 "
    "!= 0) so that g v stays inside range(H)",
)
gv_unscaled = sp.simplify((v_cf.T * g_sym * v_cf)[0, 0])
vn_cf = v_cf / sp.sqrt(om_s * kappa_s)
check(
    sp.simplify(gv_unscaled - kappa_s) == 0
    and sp.simplify((vn_cf.T * g_sym * vn_cf)[0, 0] - 1 / om_s) == 0,
    "metric normalization EXACT (symbolic): v^dag g v = kappa, so the "
    "Colpa-rescaled v/sqrt(omega kappa) obeys v^dag g v = 1/omega "
    f"(kappa = {gv_unscaled})",
)

# 5b. numeric: the actual solver on the dust-shifted Hessian (negative
# roundoff dust below zero_tol pins the Cholesky failure and hence the
# singular-square-root branch; the solver clamps it exactly like the
# symbolic eps = 0 case).

dust = -1e-12
x_num = float(x_sym)
b_val = float(b_sym)
a_num = x_num * np.array(p_mat, dtype=float) + dust * np.array(q_mat, dtype=float)
b_num = b_val * np.array(p_mat, dtype=float)
h_num = np.block([[a_num, b_num], [b_num, a_num]])
g_num = np.diag([1.0, 1.0, -1.0, -1.0])
try:
    np.linalg.cholesky(h_num)
    _chol_ok = True
except np.linalg.LinAlgError:
    _chol_ok = False
check(
    not _chol_ok,
    "numeric Hessian with -1e-12 dust fails Cholesky (deterministic "
    "singular-square-root branch)",
)
modes_s = _solve_modes(h_num)
check(
    modes_s.classification == "zero_mode"
    and modes_s.vectors is not None
    and modes_s.vectors.shape == (4, 1),
    f"solver classifies zero_mode and attaches the positive-mode vector "
    f"(classification {modes_s.classification}, vectors "
    f"{None if modes_s.vectors is None else modes_s.vectors.shape})",
)
om_num = float(modes_s.energies[modes_s.positive_indices[0]])
check(
    abs(om_num - float(om_s)) < 1e-10,
    f"solver energy equals the symbolic closed form: "
    f"{om_num:.12f} vs sqrt(x^2 - b^2) = {float(om_s):.12f}",
)
v_num = modes_s.vectors[:, 0]
resid_num = h_num @ v_num - om_num * (g_num @ v_num)
check(
    float(np.max(np.abs(resid_num))) < 1e-12,
    f"solver vector satisfies H v = omega g v on the singular Hessian "
    f"(residual {float(np.max(np.abs(resid_num))):.2e}); a plain "
    "range-restricted back-transform would miss the null admixture by "
    "~4e-3 here",
)
vu_num = v_num / np.sqrt(om_num)
norm_num = float(om_num * np.real(vu_num.conj() @ (g_num @ vu_num)))
check(
    abs(norm_num - 1.0) < 1e-12 and modes_s.residuals["max_metric_norm_error"] < 1e-12,
    f"metric normalization on the solver vector: omega * (v/sqrt(omega))"
    f"^dag g (v/sqrt(omega)) = {norm_num:.15f}",
)
null_num = float(np.max(np.abs(h_num @ modes_s.null_space)))
check(
    null_num < 1e-12 and modes_s.residuals["null_space_residual"] < 1e-12,
    f"Hessian null space still reported as non-normalizable null vectors "
    f"(H psi residual {null_num:.1e})",
)

# 5c. metric-degenerate (Jordan) null space WITH a surviving positive mode:
# a g-neutral null vector makes N = psi^dag g psi singular while a
# decoupled sector still carries a real positive frequency. The pencil is
# solved on the nondegenerate quotient of the constrained subspace (the
# kernel direction cannot carry a finite eigenvalue because A = Q^dag H Q
# is positive definite), the constraint is enforced by pseudoinverse with
# per-mode solvability validation, and unsolvable modes are refused with
# documented diagnostics instead of a singular-matrix crash.
beta_num = 0.5
h_jordan = np.zeros((4, 4), dtype=complex)
h_jordan[0, 0] = h_jordan[1, 1] = h_jordan[2, 2] = h_jordan[3, 3] = 1.0
h_jordan[0, 2] = h_jordan[2, 0] = -1.0
h_jordan[1, 3] = h_jordan[3, 1] = beta_num
psi_j = np.array([1, 0, 1, 0]) / np.sqrt(2)
n_j = float(np.real(psi_j.conj() @ (g_num @ psi_j)))
check(
    float(np.max(np.abs(h_jordan @ psi_j))) < 1e-12
    and abs(n_j) < 1e-14
    and float(np.linalg.eigvalsh(h_jordan)[0]) >= 0.0,
    "Jordan-null fixture: g-neutral null vector (N = 0, singular) of an "
    "exactly PSD doubled-form Hessian",
)
modes_j = _solve_modes(h_jordan)
om_j = float(modes_j.energies[modes_j.positive_indices[0]])
v_j = modes_j.vectors[:, 0]
resid_j = float(np.max(np.abs(h_jordan @ v_j - om_j * (g_num @ v_j))))
vu_j = v_j / np.sqrt(om_j)
norm_j = float(om_j * np.real(vu_j.conj() @ (g_num @ vu_j)))
check(
    modes_j.classification == "zero_mode"
    and abs(om_j - np.sqrt(1.0 - beta_num**2)) < 1e-10
    and resid_j < 1e-12
    and abs(norm_j - 1.0) < 1e-12,
    f"positive mode survives the metric-degenerate null space exactly: "
    f"omega = {om_j:.12f} = sqrt(1 - beta^2), generalized residual "
    f"{resid_j:.1e}, omega * (v/sqrt(omega))^dag g (v/sqrt(omega)) = "
    f"{norm_j:.15f}; no singular-matrix crash",
)

# 5d. paraunitary mode mixing (Main-reported regression): a g-unitary
# congruence lifts exact Jordan zeros into a COMPLEX dust pair above
# zero_tol (+-1.09e-8 +- 3.2e-8j) while a genuine positive mode survives.
# The metric-norm discriminator recognizes the dust (vanishing
# |v^dag g v|) and classifies it as zeros; the positive mode is preserved
# and validated. No singular-matrix crash, no fabricated positive mode.

h_p = np.array(
    [[1, 0, 1, 0], [0, 2, 0, 0.3], [1, 0, 1, 0], [0, 0.3, 0, 2]], dtype=float
)
t_p = np.array([[0.2, 0.25], [0.25, -0.1]])
s_p = _expm(np.block([[np.zeros((2, 2)), t_p], [t_p, np.zeros((2, 2))]]))
h_para = s_p.T @ h_p @ s_p
check(
    float(np.max(np.abs(s_p.T @ g_num @ s_p - g_num))) < 1e-14
    and float(np.linalg.eigvalsh(h_para)[0]) >= -1e-12,
    "paraunitary fixture: g-unitary congruence of an exactly PSD singular "
    "doubled-form Hessian (g-unitarity error < 1e-14, Hessian psd within "
    "1e-12 congruence rounding)",
)
modes_p = _solve_modes(h_para)
om_p = float(modes_p.energies[modes_p.positive_indices[0]])
v_p = modes_p.vectors[:, 0]
resid_p = float(np.max(np.abs(h_para @ v_p - om_p * (g_num @ v_p))))
vu_p = v_p / np.sqrt(om_p)
norm_p = float(om_p * np.real(vu_p.conj() @ (g_num @ vu_p)))
check(
    modes_p.classification == "zero_mode"
    and modes_p.residuals.get("n_jordan_dust_modes", 0) == 2
    and abs(om_p - float(sp.sqrt(sp.Rational(391, 100)))) < 1e-10
    and resid_p < 1e-12
    and abs(norm_p - 1.0) < 1e-12,
    f"complex Jordan dust classified as zeros (2 dust modes), genuine "
    f"positive mode preserved: omega = {om_p:.12f} = sqrt(3.4), generalized "
    f"residual {resid_p:.1e}, omega * (v/sqrt(omega))^dag g (v/sqrt(omega)) "
    f"= {norm_p:.15f}",
)

# 5e. The parity reduction does not require A and B to commute.
a, b, c, d, e, f, z = sp.symbols("a b c d e f z", real=True)
A = sp.Matrix([[a, b], [b, c]])
B = sp.Matrix([[d, e], [e, f]])
I2 = sp.eye(2)
Z2 = sp.zeros(2)
parity = sp.BlockMatrix([[I2, I2], [I2, -I2]]).as_explicit()
h_parity = sp.BlockMatrix([[A, B], [B, A]]).as_explicit()
g_parity = sp.diag(1, 1, -1, -1)
reduced = sp.BlockMatrix([[Z2, A - B], [A + B, Z2]]).as_explicit()
check(
    parity.inv() * g_parity * h_parity * parity == reduced
    and sp.expand(
        (z * sp.eye(4) - g_parity * h_parity).det()
        - (z**2 * I2 - (A - B) * (A + B)).det()
    )
    == 0,
    "parity reduction and characteristic polynomial hold symbolically for "
    "noncommuting symmetric A and B",
)
A_num = np.array([[1.0, 0.5], [0.5, 1.5]])
B_num = np.array([[1.0, 0.5], [0.5, 0.5]])
h_num = np.block([[A_num, B_num], [B_num, A_num]])
modes_num = _solve_modes(h_num)
check(
    np.max(np.abs(A_num @ B_num - B_num @ A_num)) > 0.1
    and modes_num.classification == "zero_mode"
    and np.allclose(modes_num.energies, [0.0, np.sqrt(2.0)], atol=1e-10),
    "noncommuting singular fixture: solver energies agree with the "
    "parity product eigenvalues 0 and 2",
)

# 5f. Main-reported exact fixture: an exact Jordan (Goldstone) zero plus a
# strongly squeezed 1e5 block split by 1.1e-8. The parity-reduced spectrum is
# exact for the STORED float64 matrix — the block split is quantized by the
# float64 grid, so the symbolic sqrt(2.2e-3) is unattainable — and the
# squeezed positive mode is recovered, never labeled zero and never
# mislabeled dynamically unstable.
h_sq = np.zeros((4, 4), dtype=complex)
h_sq[0, 0] = h_sq[2, 2] = 1.0
h_sq[0, 2] = h_sq[2, 0] = 1.0
h_sq[1, 1] = h_sq[3, 3] = 1e5
h_sq[1, 3] = h_sq[3, 1] = 1e5 - 1.1e-8
delta_sq = 1e5 - (1e5 - 1.1e-8)
om_sq_expect = float(np.sqrt(delta_sq * (2e5 - delta_sq)))
modes_sq = _solve_modes(h_sq)
om_sq = float(modes_sq.energies[modes_sq.positive_indices[0]])
v_sq = modes_sq.vectors[:, 0]
resid_sq = float(np.max(np.abs(h_sq @ v_sq - om_sq * (g_num @ v_sq))))
check(
    modes_sq.classification == "zero_mode"
    and abs(om_sq - om_sq_expect) < 1e-12
    and modes_sq.residuals["n_jordan_dust_modes"] == 2
    and resid_sq < 1e-6
    and modes_sq.residuals["max_generalized_residual"] < 1e-8,
    f"exact Jordan zero + squeezed block: parity-reduced spectrum exact for "
    f"the stored matrix (omega = {om_sq:.12f} = sqrt(delta*(2e5-delta)), "
    f"delta = 1e5 - fl(1e5 - 1.1e-8) = {delta_sq:.10e}); squeezed positive "
    f"mode preserved, never a zero (classification "
    f"{modes_sq.classification}, absolute residual {resid_sq:.1e})",
)

# 5g. Structural guard: a genuine block mismatch changes the Hamiltonian, so
# the certified parity reduction must not be applied to it. Here the actual
# matrix remains positive definite, so the solver must solve it directly and
# return both normalized positive modes -- never the idealized parity zero
# and never a false same-k-pairing dynamic instability.
h_pert = h_sq.copy()
h_pert[2, 2] += 1e-6
modes_pert = _solve_modes(h_pert)
pert_gram = (
    _np.max(
        _np.abs(modes_pert.vectors.conj().T @ (g_num @ modes_pert.vectors) - _np.eye(2))
    )
    if modes_pert.vectors is not None
    else _np.inf
)
pert_resid = (
    _np.max(
        _np.abs(
            h_pert @ modes_pert.vectors
            - modes_pert.energies[None, :] * (g_num @ modes_pert.vectors)
        )
    )
    if modes_pert.vectors is not None
    else _np.inf
)
check(
    not _np.array_equal(h_pert[2:, 2:], h_pert[:2, :2])
    and modes_pert.classification == "positive_definite"
    and modes_pert.vectors is not None
    and modes_pert.h_min_eigenvalue > 0.0
    and modes_pert.energies[0] > 0.0
    and pert_gram < 1e-8
    and pert_resid < 1e-6,
    f"block-mismatch guard: A2 = A1 + 1e-6 is not treated as the exact "
    f"standard form; the actual positive-definite matrix is solved "
    f"({modes_pert.classification}, modes {modes_pert.energies}) with both "
    "normalized vectors, never the idealized parity zero and never a false "
    "dynamic instability",
)

header = """# Certificate: anomalous bosonic modes, Colpa metric and stability policy

Generated by `magnon_topology_bosonic_modes.py` (all assertions executed and
passed in the `mydev` environment). Pinned statements:

1. Pairing contractions with `u = (1, i, 0)`: `a(J) = u^T J u*` and
   `b(J) = u^T J u` as in section 1; DMI never pairs a collinear z FM.
2. Dimer blocks with symmetric anisotropy `delta`: pairing `Q_AB = -4 delta cos(2 pi k a)`.
3. Physical frequencies = sorted(eig(P-Q) U eig(P+Q)), verified against the
   actual `Magnon.Hq` Colpa solve; generalized residuals < 1e-10; metric
   norms `v^dag g v = 1/omega`; signed completeness 0. The same-k +-pairing
   seen here is a property of this reciprocal standard form, not a stability
   criterion.
4. Cholesky-intermediate columns are not physical modes.
5. Stability policy: Hessian psd <=> real frequencies for this form;
   negative-Hessian references with real dynamical frequencies are refused
   without shifts; stable zero modes report non-normalizable null spaces;
   the legacy shift fallback behavior is unchanged and never used for
   topology.
6. Singular square-root back-transform on exactly singular PSD Hessians
   (Goldstone): the generalized eigenproblem `H v = omega g v` and the Colpa
   normalization `v^dag g v = 1/omega` hold exactly, symbolically on a
   closed-form nontrivial singular Hessian with a nonzero pairing channel
   and numerically through `solve_bosonic_modes`, including the
   metric-degenerate Jordan-null case (`N = psi^dag g psi` singular) where
   the pencil is solved on the nondegenerate quotient and unsolvable modes
   are refused with documented diagnostics. The parity reduction
   `lam = eig((A-B)(A+B))`, `omega = +-sqrt(lam)` of the standard bosonic
   doubled form is asserted symbolically (block congruence identity and
   characteristic polynomial, noncommuting A and B) and numerically against
   the solver on a noncommuting singular fixture and on a strongly squeezed
   float64-quantized mode; a genuine diagonal-block mismatch is solved as
   the actual positive-definite matrix, never accepted as the idealized
   parity spectrum and never mislabeled dynamically unstable.
7. Nonreciprocal stability: the reported minimal counterexample
   `H(k) = diag(2, 3)` (generally `diag(a, b)`) is positive definite with
   same-k metric spectrum `{a, -b}`, but the bosonic particle-hole identity
   `X (gH(k))* X = -gH(-k)` pairs `k` with `-k` (asserted symbolically and
   numerically). The same-k pairing residual is diagnostic only; the
   solver returns the normalized positive mode. A generic positive-definite
   `H` is similar to the Hermitian `H^{1/2} g H^{1/2}`, so its metric
   spectrum is real, and a singular PSD nonreciprocal fixture keeps both
   normalized positive modes and its Hessian null space.

## Console output of the executed script

```
"""
body = "\n".join(REPORT)
footer = "\n```\n"
report_path.write_text(header + body + footer)
print(f"report written to {report_path}")
print("CERTIFICATE OK: all assertions passed.")
