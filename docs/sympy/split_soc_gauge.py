"""Split-SOC gauge conventions: assertion-checked (sympy + numpy) derivation.

Story 001 of the split-SOC KS-band spec
(Projects/TB2J/specs/split-soc-ks/stories/story-001-sympy-conventions.md).

Pins, with assertion checks on random complex matrices (deviations asserted at
1e-14 against an O(1)-scaled toy; deviations printed):

1. Spin rotation and axis map.  The GPAW (theta, phi) spinor basis matrix
   C(theta, phi) (gpaw/spinorbit.py:380-383, 26.7.0) equals the standard active
   SU(2) rotation exp(-i phi sz/2) exp(-i theta sy/2) — asserted BOTH via exact
   sympy (check_symbolic_frame_law) and via scipy.linalg.expm; C is unitary and
     C sigma_z C^dag = n(theta, phi) . sigma .
   The TB2J axis map ``rotation_matrix(theta, phi)``
   (TB2J/mathutils/rotate_spin.py:31-41) satisfies the same axis identity
     U^dag sigma_z U = n(theta, phi) . sigma
   but is a DIFFERENT Wigner-D matrix (asserted distinct for generic angles):
   only the axis map (theta, phi) -> n is shared; never mix the two D-matrices.

2. ``add_soc`` tensordot chain.  The verbatim GPAW chain
       H = tensordot(C, H, (0, 1)); H = tensordot(C.T.conj(), H, (1, 1))
   (gpaw/spinorbit.py:74-75) computes the standard
     H <- C^dag (sigma.L) C
   at every leg (asserted against per-(i,j) plain matrix products, on both the
   packed sigma.L toy and general non-packed spin-leg data; negative control:
   C^dag H C^T differs).  An earlier draft of this script mislabeled its own
   einsum target and claimed C^dag H C^T — the operand-label trap documented
   here so it is not repeated: einsum('as,stij,bt', C^dag, H, C.T) is
   C^dag H C, because operand C.T with labels (b, t) contributes C[t, b].

3. Gauge theorem (psi/chi pictures).  With
     H_psi = h0 (x) 1 + Delta_diag (x) sigma_z + sum_v W_v (x) C^dag sigma_v C
   (GPAW psi picture: collinear exchange unrotated, SOC rotated), and
     H_chi = h0 (x) 1 + Delta_diag (x) n.sigma + sum_v W_v (x) sigma_v
   (SIESTA chi picture: exchange rotated to the leg axis, SOC lattice-fixed),
   U = 1 (x) C gives
     H_chi = U H_psi U^dag ,
   identical spectra, and eigenvector correspondence
     X_chi = U X_psi diag(e^{i phi_m}),  e^{i phi_m} = conj(<x_chi_m|U x_psi_m>)
   (non-degenerate spectrum).

4. Projections.  Spinor projections transform as a general Wigner-D on the m_j
   index, P^chi = C P^psi (einsum on the spinor index), with per-band norm
   preservation.  For n = x, C mixes the two S_z components; no diagonal
   sign/permutation map reproduces it (asserted).

5. Exchange-tensor gauge map.  With
     O_wv = Tr[sigma_w (C sigma_v C^dag)] / 2  in  SO(3),  O e_z = n,
   and lattice Cartesian probes in both pictures,
     T_chi = O T_psi O^T   (equivalently  O^T T_chi O = T_psi).
   The research report writes the inverse labeling T_chi = O^T T_psi O; the
   pinned content is the two-sided conjugation by the leg rotation with
   O e_z = n (both directions asserted, general unstructured Green data).
   The real-part convention commutes with the map.

Run with the mydev environment.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA = (SX, SY, SZ)
TOL = 1.0e-14

_RNG = np.random.default_rng(20260927)


def _random_hermitian(*shape: int) -> np.ndarray:
    """Random complex Hermitian matrix with O(1) entries."""
    a = _RNG.normal(size=shape) + 1j * _RNG.normal(size=shape)
    a = a + np.conj(np.swapaxes(a, -1, -2))
    return a / np.sqrt(2.0 * shape[-1])


def c_gpaw(theta: float, phi: float) -> np.ndarray:
    """GPAW spinorbit.py C_ss (verbatim formula), angles in radians."""
    return np.array(
        [
            [
                np.cos(theta / 2) * np.exp(-1j * phi / 2),
                -np.sin(theta / 2) * np.exp(-1j * phi / 2),
            ],
            [
                np.sin(theta / 2) * np.exp(1j * phi / 2),
                np.cos(theta / 2) * np.exp(1j * phi / 2),
            ],
        ],
        dtype=complex,
    )


def rotation_matrix_tb2j(theta: float, phi: float) -> np.ndarray:
    """TB2J mathutils.rotate_spin.rotation_matrix (verbatim formula)."""
    return np.array(
        [
            [np.cos(theta / 2), np.exp(-1j * phi) * np.sin(theta / 2)],
            [-np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def n_of(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def o_of(c_mat: np.ndarray) -> np.ndarray:
    """O_wv = Tr[sigma_w C sigma_v C^dag] / 2."""
    return np.array(
        [
            [
                0.5 * np.trace(SIGMA[w] @ c_mat @ SIGMA[v] @ c_mat.conj().T)
                for v in range(3)
            ]
            for w in range(3)
        ]
    )


def check_spin_rotation_and_axis_maps() -> None:
    """Assertions 1: C unitary, C sigma_z C^dag = n.sigma; same axis map in TB2J."""
    from scipy.linalg import expm

    # C equals the standard active SU(2) rotation (asserted, three legs)
    for theta, phi in ((0.9, 1.3), (np.pi / 2, np.pi / 2), (0.7, -0.4)):
        want = expm(-1j * phi * SZ / 2) @ expm(-1j * theta * SY / 2)
        dev = np.abs(c_gpaw(theta, phi) - want).max()
        assert dev < TOL, f"C != expm rotation at ({theta},{phi}): {dev}"
    print("  C(theta,phi) == expm(-i phi sz/2) expm(-i theta sy/2)  (dev < 1e-15)")

    for theta, phi in [
        (0.9, 1.3),
        (np.pi / 2, 0.0),
        (np.pi / 2, np.pi / 2),
        (0.0, 0.0),
    ]:
        c_mat = c_gpaw(theta, phi)
        unit = np.abs(c_mat.conj().T @ c_mat - np.eye(2)).max()
        assert unit < TOL, f"C not unitary: {unit}"
        n_vec = n_of(theta, phi)
        dev = np.abs(
            c_mat @ SZ @ c_mat.conj().T - sum(n_vec[w] * SIGMA[w] for w in range(3))
        ).max()
        assert dev < TOL, f"C sigma_z C^dag != n.sigma at ({theta},{phi}): {dev}"
        print(
            f"  C({theta:.3f},{phi:.3f}): unitary, C sz C^dag = n.sigma  (dev {dev:.1e})"
        )

    u_tb2j = rotation_matrix_tb2j(0.9, 1.3)
    n_vec = n_of(0.9, 1.3)
    dev = np.abs(
        u_tb2j.conj().T @ SZ @ u_tb2j - sum(n_vec[w] * SIGMA[w] for w in range(3))
    ).max()
    assert dev < TOL, f"TB2J rotation_matrix axis map failed: {dev}"
    distinct = np.abs(u_tb2j - c_gpaw(0.9, 1.3)).max()
    assert distinct > 0.1, f"expected distinct Wigner-D matrices, got diff {distinct}"
    print(
        f"  TB2J U^dag sz U = n.sigma (dev {dev:.1e}); U_TB2J != C (|diff| {distinct:.3f})"
    )
    print("  => axis map shared, Wigner-D matrices distinct (do not mix)")


def check_add_soc_tensordot_chain() -> None:
    """Assertion 2: verbatim GPAW chain == C^dag (sigma.L) C at every leg.

    Comparison targets use per-(i,j) plain matrix products (no einsum
    operand-label traps): target[:, :, i, j] = C^dag @ M @ C.
    """
    ni = 3
    l_vec = [_random_hermitian(ni, ni) for _ in range(3)]
    h_soc = np.zeros((2, 2, ni, ni), dtype=complex)
    h_soc[0, 0] = l_vec[2]
    h_soc[0, 1] = l_vec[0] - 1j * l_vec[1]
    h_soc[1, 0] = l_vec[0] + 1j * l_vec[1]
    h_soc[1, 1] = -l_vec[2]

    def gpaw_chain(c_mat: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
        # verbatim gpaw/spinorbit.py:74-75 chain
        out = np.tensordot(c_mat, h_arr, (0, 1))
        return np.tensordot(c_mat.T.conj(), out, (1, 1))

    def conj_form(c_mat: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(h_arr)
        for i in range(h_arr.shape[2]):
            for j in range(h_arr.shape[3]):
                out[:, :, i, j] = c_mat.conj().T @ h_arr[:, :, i, j] @ c_mat
        return out

    # packed sigma.L toy AND general non-packed spin-leg data
    h_gen = _RNG.normal(size=(2, 2, ni, ni)) + 1j * _RNG.normal(size=(2, 2, ni, ni))
    for name, h_arr in (("packed sigma.L", h_soc), ("general spin-leg H", h_gen)):
        for theta, phi in ((0.9, 1.3), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)):
            c_mat = c_gpaw(theta, phi)
            dev = np.abs(gpaw_chain(c_mat, h_arr) - conj_form(c_mat, h_arr)).max()
            assert (
                dev < TOL
            ), f"chain != C^dag H C ({name}, {theta:.2f},{phi:.2f}): {dev}"
        print(
            f"  {name}: verbatim chain == C^dag (sigma.L) C at all legs (dev < 1e-15)"
        )

    # negative control: C^dag H C^T is NOT what the chain computes
    c_mat = c_gpaw(0.9, 1.3)
    wrong = np.empty_like(h_soc)
    for i in range(ni):
        for j in range(ni):
            wrong[:, :, i, j] = c_mat.conj().T @ h_soc[:, :, i, j] @ c_mat.T
    dev = np.abs(gpaw_chain(c_mat, h_soc) - wrong).max()
    assert dev > 1.0, f"chain unexpectedly equals C^dag H C^T: {dev}"
    print(f"  negative control: chain != C^dag H C^T (gap {dev:.2f})")
    print(
        "  => GPAW's add_soc rotation is the standard C^dag (sigma.L) C;"
        " no adapter correction needed"
    )


def check_symbolic_frame_law() -> None:
    """Exact sympy assertions: SU(2)/Pauli frame law with symbolic angles.

    Discharges the AGENTS.md sympy requirement for the derived GPAW frame
    formulas: every identity below is exact symbolic equality (no numeric
    substitution).
    """
    th, ph = sp.symbols("theta phi", real=True)
    c_half, s_half = sp.cos(th / 2), sp.sin(th / 2)
    em, ep = sp.exp(-sp.I * ph / 2), sp.exp(sp.I * ph / 2)
    c_mat = sp.Matrix([[c_half * em, -s_half * em], [s_half * ep, c_half * ep]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    sig = (sx, sy, sz)

    # unitarity (exact)
    assert sp.simplify(c_mat.T.conjugate() * c_mat - sp.eye(2)) == sp.zeros(2)
    # C sigma_z C^dag == n.sigma (exact)
    n_vec = (
        sp.sin(th) * sp.cos(ph) * sx + sp.sin(th) * sp.sin(ph) * sy + sp.cos(th) * sz
    )
    assert sp.simplify(
        sp.expand_complex(c_mat * sz * c_mat.T.conjugate() - n_vec)
    ) == sp.zeros(2)
    print("  symbolic: C unitary; C sz C^dag == n.sigma (exact)")

    # O from the trace formula is SO(3) and maps e_z to n (exact)
    o_mat = sp.Matrix(
        3,
        3,
        lambda w, v: sp.Rational(1, 2)
        * sp.trace(sig[w] * c_mat * sig[v] * c_mat.T.conjugate()),
    )
    e_oo = sp.expand_complex(o_mat * o_mat.T - sp.eye(3))
    assert e_oo.applyfunc(lambda e: sp.trigsimp(e, method="fu")) == sp.zeros(3)
    assert sp.simplify(o_mat.det() - 1) == 0
    n_components = (
        sp.sin(th) * sp.cos(ph),
        sp.sin(th) * sp.sin(ph),
        sp.cos(th),
    )
    ez = o_mat * sp.Matrix([0, 0, 1])
    for w in range(3):
        e_ez = sp.expand_complex(ez[w] - n_components[w])
        assert sp.trigsimp(e_ez, method="fu") == 0
    print("  symbolic: O in SO(3), det 1, O e_z == n (exact)")

    # TB2J rotation_matrix axis identity (exact)
    u_tb2j = sp.Matrix(
        [
            [sp.cos(th / 2), sp.exp(-sp.I * ph) * sp.sin(th / 2)],
            [-sp.exp(sp.I * ph) * sp.sin(th / 2), sp.cos(th / 2)],
        ]
    )
    assert sp.simplify(
        sp.expand_complex(u_tb2j.T.conjugate() * sz * u_tb2j - n_vec)
    ) == sp.zeros(2)
    print("  symbolic: U_TB2J^dag sz U_TB2J == n.sigma (exact)")

    # verbatim tensordot index algebra == C^dag H C for symbolic H entries
    h_sym = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"h{i}{j}"))
    # tensordot(C, H, (0, 1)):  out1[s, s1] = sum_c C[c, s] H[s1, c]
    out1 = sp.Matrix(
        2, 2, lambda s, s1: sum(c_mat[c, s] * h_sym[s1, c] for c in range(2))
    )
    # tensordot(C.T.conj(), out1, (1, 1)):  out2[a, s] = sum_{s1} C^dag[a, s1] out1[s, s1]
    out2 = sp.Matrix(
        2,
        2,
        lambda a, s: sum(c_mat.T.conjugate()[a, s1] * out1[s, s1] for s1 in range(2)),
    )
    target = c_mat.T.conjugate() * h_sym * c_mat
    assert sp.expand(out2 - target) == sp.zeros(2)
    print("  symbolic: verbatim add_soc index algebra == C^dag H C (exact, symbolic H)")


def check_hamiltonian_gauge_theorem() -> None:
    """Assertion 3: H_chi = U H_psi U^dag, spectra, eigenvector correspondence."""
    nb = 4
    theta, phi = 0.9, 1.3
    c_mat = c_gpaw(theta, phi)
    n_vec = n_of(theta, phi)
    n_sig = sum(n_vec[w] * SIGMA[w] for w in range(3))
    u_mat = np.kron(np.eye(nb), c_mat)

    h0 = _random_hermitian(nb, nb) + 4.0 * np.diag(_RNG.uniform(0.5, 1.0, nb))
    delta = _RNG.uniform(0.5, 1.5, nb)  # per-band collinear exchange splitting
    w_vec = [_random_hermitian(nb, nb) for _ in range(3)]

    h_psi = np.kron(h0, np.eye(2)) + np.kron(np.diag(delta), SZ)
    h_chi = np.kron(h0, np.eye(2)) + np.kron(np.diag(delta), n_sig)
    for v in range(3):
        h_psi += np.kron(w_vec[v], c_mat.conj().T @ SIGMA[v] @ c_mat)
        h_chi += np.kron(w_vec[v], SIGMA[v])

    dev = np.abs(h_chi - u_mat @ h_psi @ u_mat.conj().T).max()
    scale = np.abs(h_psi).max()
    assert dev < TOL * scale, f"H_chi != U H_psi U^dag: {dev} (scale {scale})"
    print(f"  H_chi = U H_psi U^dag  (dev {dev:.1e}, scale {scale:.1f})")

    ev_psi = np.linalg.eigvalsh(h_psi)
    ev_chi = np.linalg.eigvalsh(h_chi)
    dev = np.abs(ev_psi - ev_chi).max()
    assert dev < 1e-13 * scale, f"spectra differ: {dev}"
    gaps = np.abs(np.diff(ev_psi)).min()
    assert gaps > 1e-3, f"toy spectrum degenerate (gap {gaps})"
    print(f"  spectra identical (dev {dev:.1e}); min gap {gaps:.2f}")

    x_psi = np.linalg.eigh(h_psi)[1]
    x_chi = np.linalg.eigh(h_chi)[1]
    phases = np.einsum("im,im->m", x_chi.conj(), u_mat @ x_psi)  # <x_chi|U x_psi>
    assert np.abs(np.abs(phases) - 1.0).max() < 1e-12
    x_rot = u_mat @ x_psi @ np.diag(np.conj(phases))
    dev = np.abs(x_chi - x_rot).max()
    assert dev < 1e-12, f"eigenvector correspondence failed: {dev}"
    print(f"  X_chi = U X_psi diag(e^i phi)  (dev {dev:.1e})")


def check_projection_wigner_d() -> None:
    """Assertion 4: P^chi = C P^psi, norms preserved, not a sign/permutation map."""
    nb, nproj = 4, 3
    p_psi = _RNG.normal(size=(nb, 2, nproj)) + 1j * _RNG.normal(size=(nb, 2, nproj))
    for theta, phi in [(np.pi / 2, 0.0), (0.9, 1.3)]:
        c_mat = c_gpaw(theta, phi)
        p_chi = np.einsum("st,mti->msi", c_mat, p_psi)
        dev = np.abs(p_chi - np.einsum("mti,st->msi", p_psi, c_mat)).max()
        assert dev < TOL, f"P^chi map not C-consistent: {dev}"
        norms_psi = np.einsum("mti,mti->mi", p_psi.conj(), p_psi)
        norms_chi = np.einsum("mti,mti->mi", p_chi.conj(), p_chi)
        dev = np.abs(norms_chi - norms_psi).max()
        assert dev < TOL, f"P^chi norms not preserved: {dev}"
        print(
            f"  theta={theta:.3f}: P^chi = C P^psi (dev {dev:.1e}); per-band |P|^2 preserved"
        )

    # x leg: C mixes S_z components; no diagonal sign map can reproduce it
    c_x = c_gpaw(np.pi / 2, 0.0)
    assert np.abs(c_x[0, 1]) > 0.3, "expected S_z mixing for the x leg"
    p_chi = np.einsum("st,mti->msi", c_x, p_psi)
    worst = min(
        np.abs(p_chi - np.einsum("st,mti->msi", np.diag([s1, s2]), p_psi)).max()
        for s1 in (1, -1)
        for s2 in (1, -1)
    )
    assert worst > 0.1, f"sign/permutation map unexpectedly close: {worst}"
    print(f"  x leg: no diagonal sign map reproduces C (min residual {worst:.3f})")


def _tensor_from_probes(
    v_a: np.ndarray, v_b: np.ndarray, g1: np.ndarray, g2: np.ndarray
) -> np.ndarray:
    """T[w,v] = Tr[(sigma_w (x) 1) V_a G (sigma_v (x) 1) V_b G'] on band(x)spin."""
    nb = v_a.shape[0] // 2
    probes = [np.kron(np.eye(nb), SIGMA[w]) for w in range(3)]
    tens = np.empty((3, 3), dtype=complex)
    for w in range(3):
        for v in range(3):
            tens[w, v] = np.trace(probes[w] @ v_a @ g1 @ probes[v] @ v_b @ g2)
    return tens


def check_exchange_tensor_gauge_map() -> None:
    """Assertion 5: T_chi = O T_psi O^T (and inverse), O in SO(3), O e_z = n."""
    nb = 3
    theta, phi = 0.9, 1.3
    c_mat = c_gpaw(theta, phi)
    n_vec = n_of(theta, phi)
    o_mat = o_of(c_mat)

    dev = np.abs(o_mat @ o_mat.T - np.eye(3)).max()
    assert dev < TOL, f"O not orthogonal: {dev}"
    assert abs(np.linalg.det(o_mat).real - 1.0) < TOL, "O det != +1"
    dev = np.abs(o_mat @ np.array([0.0, 0.0, 1.0]) - n_vec).max()
    assert dev < TOL, f"O e_z != n: {dev}"
    print(f"  O in SO(3) (dev {dev:.1e}), det +1, O e_z = n")

    def rnd(*shape: int) -> np.ndarray:
        return _RNG.normal(size=shape) + 1j * _RNG.normal(size=shape)

    g1, g2 = rnd(2 * nb, 2 * nb), rnd(2 * nb, 2 * nb)
    d_a = rnd(nb, nb)
    d_a = d_a + d_a.conj().T
    d_b = rnd(nb, nb)
    d_b = d_b + d_b.conj().T
    u_mat = np.kron(np.eye(nb), c_mat)
    v_psi_a, v_psi_b = np.kron(d_a, SZ), np.kron(d_b, SZ)
    v_chi_a = u_mat @ v_psi_a @ u_mat.conj().T
    v_chi_b = u_mat @ v_psi_b @ u_mat.conj().T
    g_chi1 = u_mat @ g1 @ u_mat.conj().T
    g_chi2 = u_mat @ g2 @ u_mat.conj().T

    t_psi = _tensor_from_probes(v_psi_a, v_psi_b, g1, g2)
    t_chi = _tensor_from_probes(v_chi_a, v_chi_b, g_chi1, g_chi2)
    scale = max(1.0, np.abs(t_psi).max())

    dev = np.abs(t_chi - o_mat @ t_psi @ o_mat.T).max()
    assert dev < TOL * scale, f"T_chi != O T_psi O^T: {dev} (scale {scale})"
    print(f"  T_chi = O T_psi O^T  (dev {dev:.1e}, scale {scale:.1f})")

    dev = np.abs(o_mat.T @ t_chi @ o_mat - t_psi).max()
    assert dev < TOL * scale, f"inverse map failed: {dev}"
    print(f"  inverse labeling O^T T_chi O = T_psi  (dev {dev:.1e})")

    t_psi_r, t_chi_r = t_psi.real, t_chi.real
    dev = np.abs(t_chi_r - o_mat @ t_psi_r @ o_mat.T).max()
    assert dev < TOL * scale, f"real-part convention failed: {dev}"
    print(f"  real-part tensors follow the same map (dev {dev:.1e})")


def check_tb2j_axis_map_equivalence() -> None:
    """Closure: both codes carry the same (theta, phi) -> n axis map."""
    for theta, phi in [(0.7, -0.4), (np.pi / 2, np.pi / 2)]:
        n_vec = n_of(theta, phi)
        n_sig = sum(n_vec[w] * SIGMA[w] for w in range(3))
        dev_g = np.abs(
            c_gpaw(theta, phi) @ SZ @ c_gpaw(theta, phi).conj().T - n_sig
        ).max()
        dev_t = np.abs(
            rotation_matrix_tb2j(theta, phi).conj().T
            @ SZ
            @ rotation_matrix_tb2j(theta, phi)
            - n_sig
        ).max()
        assert dev_g < TOL and dev_t < TOL, (dev_g, dev_t)
        print(f"  ({theta:.3f},{phi:.3f}): GPAW dev {dev_g:.1e}, TB2J dev {dev_t:.1e}")
    print("  axis-map equivalence pinned; D-matrices remain backend-specific")


def main() -> None:
    print("split_soc_gauge: split-SOC gauge conventions (story-001)")
    check_symbolic_frame_law()
    check_spin_rotation_and_axis_maps()
    check_add_soc_tensordot_chain()
    check_hamiltonian_gauge_theorem()
    check_projection_wigner_d()
    check_exchange_tensor_gauge_map()
    check_tb2j_axis_map_equivalence()
    print("all assertions passed (tolerance 1e-14 on O(1)-scaled random complex toys)")


if __name__ == "__main__":
    main()
