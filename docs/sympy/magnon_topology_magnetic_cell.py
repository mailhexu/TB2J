"""Certificate: commensurate magnetic-cell expansion and embedding phases.

Executable checks for ticket 05 (matched embeddings and commensurate
magnetic cells). The symbolic reference is a one-dimensional three-site
120-degree antiferromagnetic chain with isotropic nearest-neighbour exchange.

The certificate derives the reference truth for the magnetic-cell expansion
symbolically (section 2) and then verifies the executable construction
against it (section 3). Pinned statements (all executed):

1. Phase-composition identity: on a commensurate expansion m (m*q integer),
   the real-space displacement d (in original cells) of any expanded bond
   satisfies exp(2 pi i Q d) == exp(2 pi i Q R) with R = d mod m, so the
   single-Q bond twist is well defined on the expanded cell.
2. Twisted-ring reference (symbolic): for the N=3 ring with moments twisted
   by `phi_s = s*2*pi/3`, the linear classical gradient vanishes. The full
   bosonic A block is Hermitian, B is symmetric and nonzero, and the k'=0
   Hessian has an exact Goldstone null direction. This one-dimensional
   nearest-neighbour reference is dynamically unstable at the other folded
   momenta; the pipeline detects and refuses those modes rather than claiming
   a stable folded spectrum.
For a Hermitian Fourier A block, the Nambu hole block is `A(-k)^* = A^T(-k)`: conjugate each real-space A coefficient before the Fourier phase. Conjugating the already transformed `A(k)` is not equivalent.
3. Construction verification: the expanded-cell SpinIO model with twisted
   frames and unchanged re-keyed tensors is stationary and receives the same
   zero-mode/unstable classifications from the executable topology assembler.
   In-plane SIA on the twisted reference breaks stationarity and is refused.
4. Stationarity: the twisted expanded reference passes the constrained
   transverse classical gradient check of the topology pipeline.
5. No double rotation: solving the expanded model with a second Q rotation
   changes the Hamiltonian; the expanded reference must be solved with
   Q = 0 (the ordering is real space now).
6. Embedding identity: LatticeEmbedding maps embedding-fractional momenta
   through the reciprocal dual k_in = k_embed @ basis^{-T}, with fixed
   normal_k appended; nonprimitive bases (|det| != 1) are refused.
"""

from pathlib import Path

import numpy as np
import sympy as sp

_np = np

from ase import Atoms as _Atoms  # noqa: E402

from TB2J.io_exchange import SpinIO as _SpinIO  # noqa: E402

REPORT: list[str] = []


def note(text: str):
    REPORT.append(text)
    print(text)


def check(condition, message: str):
    if not condition:
        raise AssertionError(f"CERTIFICATE FAILED: {message}")
    print(f"  [OK] {message}")
    REPORT.append(f"  [OK] {message}")


report_path = Path(__file__).parent / "magnon_topology_magnetic_cell.md"

# ---------------------------------------------------------------------------
# 1. Phase-composition identity
# ---------------------------------------------------------------------------
note("1. Phase-composition identity on the commensurate expansion")
# commensurate: q = p/m exactly; d = R + n*m  =>  Q*d = Q*R + p*n
q_frac = sp.Rational(1, 3)
m_val = 3
deltas = [(1, 1), (2, -1), (3, 0), (4, 1), (5, -1)]
ok = all(
    sp.exp(sp.I * 2 * sp.pi * q_frac * dd) - sp.exp(sp.I * 2 * sp.pi * q_frac * rr) == 0
    for dd, rr in deltas
)
check(
    ok,
    "exp(2 pi i Q d) == exp(2 pi i Q R) with R = d mod m for q = p/m "
    "(all sampled displacements)",
)

# ---------------------------------------------------------------------------
# 2. Symbolic twisted-ring linear spin-wave reference
# ---------------------------------------------------------------------------
note("2. Symbolic twisted-ring reference (N = 3, isotropic J1)")

S_sym, J1_sym, delta = sp.symbols("S J1 delta", positive=True, real=True)
nsites = 3
a_sym = sp.symbols(f"a0:{nsites}")  # annihilation
b_sym = sp.symbols(f"b0:{nsites}")  # creation (a dagger)
x_sym = sp.symbols(f"x0:{nsites}", real=True)
y_sym = sp.symbols(f"y0:{nsites}", real=True)


def _frames(phi):
    """Right-handed HP frame around a moment twisted about the z axis."""
    c, s = sp.cos(phi), sp.sin(phi)
    e1 = sp.Matrix([-s, c, 0])  # in-plane tangent
    e2 = sp.Matrix([0, 0, 1])  # spiral axis
    e3 = sp.Matrix([c, s, 0])  # classical moment
    return e1, e2, e3


e1_ref, e2_ref, e3_ref = _frames(delta)
check(
    e1_ref.cross(e2_ref) == e3_ref,
    "local transverse axes are right-handed about the classical moment",
)


def _spin_vec(site, xs, ys):
    """Classical spin with transverse components in the local frame.

    Normalized to |S| through second order: n3 = 1 - (xs^2 + ys^2)/2.
    """
    e1, e2, e3 = _frames(sp.Integer(site) * delta)
    n3 = 1 - (xs**2 + ys**2) / 2
    return S_sym * (e1 * xs + e2 * ys + e3 * n3)


E_full = sp.Integer(0)
for s in range(nsites):
    t = (s + 1) % nsites
    Ss = _spin_vec(s, x_sym[s], y_sym[s])
    St = _spin_vec(t, x_sym[t], y_sym[t])
    E_full = E_full + J1_sym * Ss.dot(St)
E_full = sp.expand(E_full)

# keep terms through second order in the transverse variables
E2 = sp.Integer(0)
E1 = sp.Integer(0)
gens = tuple(x_sym) + tuple(y_sym)
for term in sp.Add.make_args(E_full):
    if term.is_number:
        continue
    deg = sp.Poly(term, *gens).total_degree()
    if deg == 1:
        E1 += term
    elif deg == 2:
        E2 += term

check(
    sp.simplify(E1.subs(delta, 2 * sp.pi / 3)) == 0,
    "linear classical gradient vanishes at the commensurate pitch " "delta = 2*pi/3",
)

# Holstein-Primakoff linearization: with the transverse components defined
# in units of the full moment magnitude, mu_perp = S_sym * x, the textbook
# amplitude sqrt(S/2) corresponds to x = (a + b) / sqrt(2 S_sym).
c_hp = 1 / sp.sqrt(2 * S_sym)
E2_hp = E2.subs(
    {x_sym[s]: c_hp * (a_sym[s] + b_sym[s]) for s in range(nsites)}
    | {y_sym[s]: -sp.I * c_hp * (a_sym[s] - b_sym[s]) for s in range(nsites)}
)
E2_hp = sp.expand(E2_hp)

Amat = sp.Matrix(
    nsites, nsites, lambda i, j: sp.simplify(sp.diff(E2_hp, b_sym[i], a_sym[j]))
)
Bmat = sp.Matrix(
    nsites, nsites, lambda i, j: sp.simplify(sp.diff(E2_hp, b_sym[i], b_sym[j]))
)


def _mat_zero(mat):
    return all(sp.simplify(sp.expand(el)) == 0 for el in mat)


check(_mat_zero(Amat - Amat.H), "bosonic A block is Hermitian")
check(_mat_zero(Bmat - Bmat.T), "bosonic B block is complex symmetric (pairing)")


# Bloch BdG matrix at the folded ring momenta. The wrap bond carries the
# reciprocal phase in both number-conserving and anomalous blocks.
def _bloch(M, kp):
    phase = sp.exp(sp.I * 2 * sp.pi * kp)
    out = M.copy()
    out[0, 2] *= phase
    out[2, 0] /= phase
    return out


def _bdg(kp):
    A = _bloch(Amat, kp)
    B = _bloch(Bmat, kp)
    return A, B, _bloch(Amat.conjugate(), kp)


ring_k = [sp.Integer(0), sp.Rational(1, 3), sp.Rational(2, 3)]
G_metric = sp.diag(1, 1, 1, -1, -1, -1)
H0_blocks = _bdg(ring_k[0])
A0, B0, A0bar = H0_blocks
H0 = A0.row_join(B0).col_join(B0.conjugate().T.row_join(A0bar))
check(
    sp.simplify(H0 - H0.conjugate().T) == sp.zeros(6),
    "full k=0 bosonic BdG matrix is Hermitian",
)
gold_det = sp.simplify(sp.factor(H0.det().subs(delta, 2 * sp.pi / 3)))
check(
    gold_det == 0,
    "full bosonic Hessian has an exact Goldstone null direction at k'=0",
)
check(
    not _mat_zero(Bmat),
    "isotropic noncollinear reference has a nonzero anomalous pairing block",
)

# Numeric check exposes the limitation of this one-dimensional 120-degree
# nearest-neighbour chain: it is stationary but has unstable folded momenta.
anchor_subs = {S_sym: 2.0, J1_sym: 0.01, delta: 2 * sp.pi / 3}
ring_imaginary_parts = []
for kp in ring_k:
    A_k, B_k, Abar_k = _bdg(kp)
    H_k = A_k.row_join(B_k).col_join(B_k.conjugate().T.row_join(Abar_k))
    check(
        sp.simplify(H_k - H_k.conjugate().T) == sp.zeros(6),
        f"full bosonic BdG matrix is Hermitian at k'={float(kp):g}",
    )
    H_num = np.array(H_k.subs(anchor_subs).evalf(), dtype=complex)
    frequencies = np.linalg.eigvals(np.asarray(G_metric, dtype=complex) @ H_num)
    ring_imaginary_parts.append(float(np.max(np.abs(frequencies.imag))))
check(
    ring_imaginary_parts[0] < 1e-9,
    "the stationary reference has real frequencies at k'=0",
)
check(
    ring_imaginary_parts[1] > 1e-8 and ring_imaginary_parts[2] > 1e-8,
    "nearest-neighbour 120-degree chain is dynamically unstable away from k'=0",
)
check(
    np.isclose(ring_imaginary_parts[1], ring_imaginary_parts[2], atol=1e-9),
    "the two symmetry-related unstable folded momenta have equal growth rates",
)
check(
    not _mat_zero(Bmat),
    "isotropic noncollinear reference has a nonzero anomalous pairing block",
)

# ---------------------------------------------------------------------------
# 3. Executable construction verification
# ---------------------------------------------------------------------------
note("3. Expanded-cell construction verification (twisted frames)")


def expanded_twisted_chain(q_num=1, q_den=3, J1=-0.01, K=0.0):
    """Expanded N-cell chain: twisted moments, unchanged re-keyed tensors.

    Bond (s -> s+1) with wrap displacement kept integer via R = +-1 keys.
    """
    m_cells = q_den
    cell_a = 10.0
    atoms = _Atoms(
        f"Cr{m_cells}",
        positions=[[c * cell_a / m_cells, 0.0, 0.0] for c in range(m_cells)],
        cell=[[m_cells * cell_a, 0, 0], [0, 10, 0], [0, 0, 10]],
        pbc=True,
    )
    exc = {}
    # +1 direction (displacement +1/3 original cell): (0->1,1->2,R=0), (2->0,R=1)
    # -1 direction (displacement -1/3): (1->0,2->1,R=0), (0->2,R=-1)
    for key in [
        ((0, 0, 0), 0, 1),
        ((0, 0, 0), 1, 0),
        ((0, 0, 0), 1, 2),
        ((0, 0, 0), 2, 1),
        ((1, 0, 0), 2, 0),
        ((-1, 0, 0), 0, 2),
    ]:
        exc[key] = J1
    sia = {c: _np.diag([K, 0.0, 0.0]) for c in range(m_cells)} if K else None
    twist = 2 * np.pi * q_num / q_den
    magmoms = [
        [2.0 * np.cos(twist * s), 2.0 * np.sin(twist * s), 0.0] for s in range(m_cells)
    ]
    io = _SpinIO(
        atoms=atoms,
        spinat=_np.array(magmoms),
        charges=[0.0] * m_cells,
        index_spin=list(range(m_cells)),
        colinear=False,
        exchange_Jdict=exc,
        sia_tensor=sia,
    )
    io._build_Rlist()
    return io


def _ring_classifications(io, q_den=3):
    from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian
    from TB2J.magnon.topology.bosonic import solve_bosonic_modes
    from TB2J.magnon.topology.preparation import prepare_topology_model

    model = prepare_topology_model(io, source_path="cert")
    classifications = []
    for n in range(q_den):
        kv = n / q_den
        H = local_frame_bosonic_hamiltonian(
            model.JR, model.Rlist, model.magmoms, _np.array([[kv, 0.0, 0.0]])
        )
        classifications.append(solve_bosonic_modes(H[0]).classification)
    return classifications


twisted_io = expanded_twisted_chain()
twisted_classifications = _ring_classifications(twisted_io)
note(f"   executable 120-degree chain classifications: {twisted_classifications}")
check(
    twisted_classifications[0] == "zero_mode"
    and twisted_classifications[1:] == ["energetically_unstable"] * 2,
    "the numerical topology pipeline refuses the same unstable folded modes "
    "identified by the symbolic metric spectrum",
)


# in-plane SIA on the twisted reference must break stationarity
from TB2J.magnon.topology.assembly import classical_transverse_gradient  # noqa: E402
from TB2J.magnon.topology.preparation import prepare_topology_model  # noqa: E402

sia_io = expanded_twisted_chain(K=0.001)
model_sia = prepare_topology_model(sia_io, source_path="cert")
stat_sia = classical_transverse_gradient(
    model_sia.JR, model_sia.Rlist, model_sia.magmoms, rtol=1e-6
)
check(
    not stat_sia.stationary,
    f"in-plane SIA on the twisted reference breaks stationarity "
    f"(max transverse gradient {stat_sia.max_transverse:.2e} > tolerance "
    f"{stat_sia.tolerance:.2e}); the pipeline guard refuses it",
)

# ---------------------------------------------------------------------------
# 4. Stationarity of the verified construction
# ---------------------------------------------------------------------------
note("4. Stationarity of the twisted expanded reference")
model_tw = prepare_topology_model(twisted_io, source_path="cert")
stat = classical_transverse_gradient(
    model_tw.JR, model_tw.Rlist, model_tw.magmoms, rtol=1e-6
)
check(
    stat.stationary,
    f"twisted expanded reference is stationary (max transverse gradient "
    f"{stat.max_transverse:.2e} within tolerance {stat.tolerance:.2e})",
)

# ---------------------------------------------------------------------------
# 5. No double rotation
# ---------------------------------------------------------------------------
note("5. No double rotation")
H_exp = model_tw.magnon().Hq(_np.array([[0.1, 0.0, 0.0]]))[0]
_refused = False
try:
    m_double = model_tw.magnon(Q=(1 / 3, 0.0, 0.0))
    H_double = m_double.Hq(_np.array([[0.1, 0.0, 0.0]]))[0]
    _refused = not _np.allclose(H_exp, H_double, atol=1e-8)
except ValueError as _exc:
    _refused = "non-uniform reference moment pattern" in str(_exc)
check(
    _refused,
    "a second Q rotation on the expanded (non-uniform) reference is refused "
    "(the expanded reference must be solved with Q = 0)",
)

# ---------------------------------------------------------------------------
# 6. Embedding identity
# ---------------------------------------------------------------------------
note("6. Embedding identity")
from TB2J.magnon.topology.embedding import (  # noqa: E402
    LatticeEmbedding,
    identity_embedding,
)

B_sym = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
A_sym = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"a{i}{j}"))
G_input = A_sym.inv().T
A_embed = B_sym * A_sym
G_embed = A_embed.inv().T
dual_sym = B_sym.inv().T
k_embed_sym = sp.Matrix([[sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(1, 4)]])
check(
    sp.simplify(A_embed * G_embed.T - sp.eye(3)) == sp.zeros(3),
    "declared direct basis and reciprocal dual satisfy A_embed G_embed^T = I",
)
check(
    sp.simplify(k_embed_sym * dual_sym * G_input - k_embed_sym * G_embed)
    == sp.zeros(1, 3),
    "reciprocal-coordinate map k_input = k_embed B^{-T} preserves physical k",
)
emb = LatticeEmbedding(basis=B_sym.tolist(), normal_k=0.25)
k_in = emb.to_input_k([0.1, 0.2])
check(
    _np.allclose(k_in, [-0.1, 0.2, 0.25]),
    "to_input_k uses the reciprocal dual and appends normal_k "
    f"([0.1, 0.2] -> {np.round(k_in, 6).tolist()})",
)
ident = identity_embedding()
check(
    _np.allclose(ident.to_input_k([0.1, 0.2]), [0.1, 0.2, 0.0]),
    "identity embedding preserves the input momentum",
)
try:
    LatticeEmbedding(basis=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    raise AssertionError("CERTIFICATE FAILED: nonprimitive basis was accepted")
except ValueError as exc:
    check(
        "nonprimitive" in str(exc),
        f"nonprimitive basis refused " f"(|det| = 2): '{exc}'",
    )

header = """# Certificate: commensurate magnetic-cell expansion and embedding phases

Generated by `magnon_topology_magnetic_cell.py` (all assertions executed and
passed in the `mydev` environment). Pinned statements:

1. Phase composition: `exp(2 pi i Q d) == exp(2 pi i Q R)` with `R = d mod m`
   for commensurate `Q = p/m`; the single-Q bond twist is well defined on the
   expanded cell.
2. Twisted-ring reference (symbolic): for the N=3 ring with moments twisted
   by `phi_s = s*2*pi/3`, the linear classical gradient vanishes. The full
   bosonic A block is Hermitian, B is symmetric and nonzero, and the k'=0
   Hessian has an exact Goldstone null direction. This one-dimensional
   nearest-neighbour reference is dynamically unstable at the other folded
   momenta; the topology pipeline detects and refuses those modes rather than
   claiming a stable folded spectrum.
For a Hermitian Fourier A block, the Nambu hole block is `A(-k)^* = A^T(-k)`: conjugate real-space coefficients before the Fourier phase.
3. Construction verification: the expanded-cell SpinIO model with twisted
   frames is stationary and the executable assembler reports the same
   zero-mode/unstable classifications. In-plane SIA breaks stationarity and
   is refused by the pipeline guard.
4. The twisted expanded reference is stationary (vanishing constrained
   transverse classical gradient).
5. Applying a second Q rotation to the expanded model changes the
   Hamiltonian: refused by construction (the expanded reference is Q = 0).
6. Embedding: `k_in = k_embed @ basis^{-T}` is the reciprocal dual of the
   declared direct basis, with fixed `normal_k`; nonprimitive bases refused.

## Console output of the executed script

```
"""
body = "\n".join(REPORT)
footer = "\n```\n"
report_path.write_text(header + body + footer)
print(f"report written to {report_path}")
print("CERTIFICATE OK: all assertions passed.")
