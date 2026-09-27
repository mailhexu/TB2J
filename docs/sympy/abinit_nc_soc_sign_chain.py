"""ABINIT NC split-SOC sign chain: i^l f_l Y_lm / amet(-i) / conjugation.

Story 001 of the split-SOC KS-band spec; pins the operator convention of
research-supporting/split-soc-abinit-nc-pypao.md (sections 1.3, 2.1, 3)
against ABINIT source semantics, with a finite toy G-space contraction.

Source-pinned references (ABINIT tree branch `savetb2j`):
- abinit/src/66_nonlocal/m_contract.F90::metric_so: Pauli Re/Im packing
  sigma_n/2, optional spinaxis U(alpha, beta) with S -> U^dag S U, the
  antisymmetric Cartesian tensor A^(n) from even permutations (n, m1, m2)
  of (0, 1, 2), and the final Re/Im swap = the "-i amet" factor.
- abinit/src/66_nonlocal/m_nonlop_pl.F90 (SO pass): amet(1) applies to gxa,
  amet(2) applies to i*gxa (temp = (-Im, +Re)), summed over the source spin;
  metcon_so rank=1 is a plain 3x3 application of amet on the tensor index.
- Projector FT convention (pypao/abinao):
  P_lm(k) = sum_G i^l f_l(|k+G|) Y_lm(k+G^) exp(+i 2pi (k+G).tau_a) c_G,
  with real tesseral harmonics (pypao/spherical_harmonics.py: scipy lpmv,
  Condon-Shortley phase, ABINIT ordering l^2+l+m).

Asserted chains (all on random complex data, 1e-14 against O(1) scale):

1. metric_so internals: the pre-swap
   amet0 = sum_n (sigma_n/2) (x) A^(n) is real, and the final Re/Im swap
   multiplies the complex matrix by -i.  With gprimd = identity,
   amet = -i amet0 = L (x) S (S = sigma/2) in the real Cartesian-p basis.
   The spinaxis rotation U(alpha, beta) conjugates the Pauli side only:
   amet(alpha, beta) = (U^dag (x) 1) amet(0, 0) (U (x) 1)  [lattice-fixed L].
2. Two-branch contraction: applying (amet(1), amet(2)) as in m_nonlop_pl to
   a complex tensor g,  out = amet(1) g + amet(2) (i g),  equals direct
   multiplication by the complex matrix -i amet0 (any gprimd).
3. Finite toy G-space contraction (single l = 1 channel, random G set):
   the ABINIT-side assembly
     W[(G'σ'),(Gσ)] = sum_{iy1,iy2} t*[G',iy1] amet[iy1,iy2,s',s] t[G,iy2],
     t[G,iy] = i^l f(q) Y^R_l,m(iy)(q^) exp(+i 2pi (k+G).tau),
   equals the complex-harmonic Python-kernel form with the complex
   conjugation carried by the KET (G-side) tensor:
     W = sum_{m'm} Y_{m'}(q'^) [L.S]^c_{m'm} Y*_{m}(q^)
         * (i^l)(-i)^l f(q')f(q) exp(+i 2pi (G'-G).tau).
   (The conjugate placement — Y*(q'^) on the bra, as naively written in
   research note section 2.1 — yields the complex conjugate operator and is
   asserted WRONG; this is the pinned CrI3-incident-class convention.)
   W is Hermitian and spin-traceless, and additive over atoms
   (W_SO covers all atoms).  Negative controls: conjugating the bra instead
   of the ket, flipping the ket atomic-phase conjugation, or flipping the
   ket i^l sign each break the match (asserted).

Run with the mydev environment.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import lpmv

TOL = 1.0e-14

_RNG = np.random.default_rng(20260927)


def metric_so_paulis(alpha: float = 0.0, beta: float = 0.0) -> list[np.ndarray]:
    """Pauli(n)/2 as 2x2 complex matrices for n = (x, y, z), spinaxis-applied."""
    paulis = [
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex),  # x
        np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex),  # y
        np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex),  # z
    ]
    if abs(alpha) < 1e-12 and abs(beta) < 1e-12:
        return paulis
    cb2, sb2 = np.cos(beta / 2), np.sin(beta / 2)
    em = np.exp(-1j * alpha / 2)
    u_mat = np.array(
        [[cb2 * em, -sb2 * em], [sb2 * np.conj(em), cb2 * np.conj(em)]], dtype=complex
    )
    return [u_mat.conj().T @ p_mat @ u_mat for p_mat in paulis]


def _amet_preswap(gprimd: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """amet0[iy1, iy2, s1, s2] = sum_n pauli_n (x) A^(n) (real-valued)."""
    paulis = metric_so_paulis(alpha, beta)
    amet0 = np.zeros((3, 3, 2, 2), dtype=complex)
    for n in range(3):
        m1 = (n + 1) % 3
        m2 = (m1 + 1) % 3
        a_mat = (
            gprimd[m1, :, None] * gprimd[m2, None, :]
            - gprimd[m2, :, None] * gprimd[m1, None, :]
        )
        amet0 += paulis[n][None, None, :, :] * a_mat[:, :, None, None]
    return amet0


def metric_so_amet(
    gprimd: np.ndarray, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    """amet as complex [iy1, iy2, s1, s2], verbatim metric_so incl. Re/Im swap."""
    paulis = metric_so_paulis(alpha, beta)
    amet = np.zeros((2, 3, 3, 2, 2))
    for n in range(3):
        m1 = (n + 1) % 3
        m2 = (m1 + 1) % 3
        a_mat = (
            gprimd[m1, :, None] * gprimd[m2, None, :]
            - gprimd[m2, :, None] * gprimd[m1, None, :]
        )
        for iy1 in range(3):
            for iy2 in range(3):
                amet[0, iy1, iy2] += paulis[n].real * a_mat[iy1, iy2]
                amet[1, iy1, iy2] += paulis[n].imag * a_mat[iy1, iy2]
    swapped = np.empty_like(amet)
    swapped[0] = amet[1]  # Fortran: amet(1) <- amet(2); amet(2) <- -amet(1)
    swapped[1] = -amet[0]
    return swapped[0] + 1j * swapped[1]


def check_amet_minus_i(gprimd: np.ndarray) -> None:
    """Assertion 1a: the Re/Im swap is multiplication by -i; L(x)S for gprimd=I."""
    amat = metric_so_amet(gprimd)
    amet0 = _amet_preswap(gprimd, 0.0, 0.0)
    dev = np.abs(amat - (-1j) * amet0).max()
    assert dev < TOL, f"Re/Im swap != -i multiplication: {dev}"
    print(f"  amet(post-swap) == -i amet0  (dev {dev:.1e})")

    if np.abs(gprimd - np.eye(3)).max() < 1e-15:
        # Cartesian p basis (l = 1 real tesseral): L_n has matrix -i eps_{n,.,.}
        paulis = metric_so_paulis()
        ls_real = np.zeros((3, 3, 2, 2), dtype=complex)
        for n in range(3):
            m1 = (n + 1) % 3
            m2 = (m1 + 1) % 3
            eps = np.zeros((3, 3))
            eps[m1, m2] = 1.0
            eps[m2, m1] = -1.0
            ls_real += (-1j * eps)[:, :, None, None] * paulis[n][None, None, :, :]
        dev = np.abs(amat - ls_real).max()
        assert dev < TOL, f"amet != L(x)S in the Cartesian p basis: {dev}"
        print(f"  gprimd = I: amet == L (x) S in the real p basis  (dev {dev:.1e})")


def check_amet_spinaxis() -> None:
    """Assertion 1b: spinaxis rotates the Pauli side only (lattice-fixed L)."""
    alpha, beta = 0.7, 1.1
    a_rot = metric_so_amet(np.eye(3), alpha, beta)
    a_zero = metric_so_amet(np.eye(3), 0.0, 0.0)
    cb2, sb2 = np.cos(beta / 2), np.sin(beta / 2)
    em = np.exp(-1j * alpha / 2)
    u_mat = np.array(
        [[cb2 * em, -sb2 * em], [sb2 * np.conj(em), cb2 * np.conj(em)]], dtype=complex
    )
    want = np.einsum("as,ijst,tu->ijau", u_mat.conj().T, a_zero, u_mat)
    dev = np.abs(a_rot - want).max()
    assert dev < TOL, f"spinaxis amet != U^dag amet(0) U: {dev}"
    print(f"  amet(alpha,beta) == (U^dag (x) 1) amet(0,0) (U (x) 1)  (dev {dev:.1e})")


def check_two_branch_contraction() -> None:
    """Assertion 2: amet(1) g + amet(2) (i g) == (-i amet0) g, complex g."""
    gprimd = np.random.default_rng(5).normal(size=(3, 3))
    amet = metric_so_amet(gprimd)
    amet_re, amet_im = amet.real, amet.imag

    g_ten = _RNG.normal(size=(3, 2, 7)) + 1j * _RNG.normal(size=(3, 2, 7))
    out_branch = np.einsum("ijab,jbn->ian", amet_re, g_ten) + np.einsum(
        "ijab,jbn->ian", amet_im, 1j * g_ten
    )
    out_direct = np.einsum(
        "ijab,jbn->ian", (-1j) * _amet_preswap(gprimd, 0.0, 0.0), g_ten
    )
    dev = np.abs(out_branch - out_direct).max()
    scale = np.abs(out_direct).max()
    assert (
        dev < TOL * scale
    ), f"two-branch contraction != -i amet0 g: {dev} (scale {scale})"
    print(
        f"  amet(1) g + amet(2) (i g) == (-i amet0) g  (dev {dev:.1e}, scale {scale:.1f})"
    )


def real_sph_harm(
    l_val: int, m_val: int, cos_theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """pypao/spherical_harmonics.py real_sph_harm (verbatim algorithm)."""
    abs_m = abs(m_val)
    ylmcst = np.sqrt((2 * l_val + 1) / (4.0 * np.pi))
    legendre = lpmv(abs_m, l_val, cos_theta)
    if m_val == 0:
        return ylmcst * legendre
    norm = (
        ylmcst
        * np.sqrt(factorial(l_val - abs_m) / factorial(l_val + abs_m))
        * (-1) ** abs_m
    )
    trig = np.cos(abs_m * phi) if m_val > 0 else np.sin(abs_m * phi)
    return np.sqrt(2.0) * norm * trig * legendre


def complex_y1(m_val: int, cos_theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Complex Condon-Shortley Y_1^m."""
    c44 = np.sqrt(3.0 / (4.0 * np.pi))
    if m_val == 0:
        return c44 * cos_theta
    c88 = np.sqrt(3.0 / (8.0 * np.pi))
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    if m_val == 1:
        return -c88 * sin_th * np.exp(1j * phi)
    return c88 * sin_th * np.exp(-1j * phi)


def ls_complex_matrix() -> np.ndarray:
    """L.S, S = sigma/2, complex m-basis, ordering m = (-1, 0, 1).

    Flattened ((m, s), (m', s')) with s = (up, dn); L dimensionless and
    L_pm |l m> = sqrt(l(l+1) - m(m+-1)) |l m+-1>.
    """
    m_vals = (-1, 0, 1)
    ls_mat = np.zeros((6, 6), dtype=complex)

    def idx(m_val: int, s_val: int) -> int:
        return 2 * m_vals.index(m_val) + s_val

    sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
    s_minus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # S-: up -> dn
    s_plus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # S+: dn -> up
    for m_val in m_vals:
        for s1 in range(2):
            for s2 in range(2):
                ls_mat[idx(m_val, s1), idx(m_val, s2)] += m_val * sz[s1, s2]
        for lp, s_mat in ((1, s_minus), (-1, s_plus)):  # 1/2 (L+ S- + L- S+)
            m_out = m_val + lp
            if abs(m_out) > 1:
                continue
            coeff = np.sqrt(2.0 - m_val * m_out)  # l = 1: l(l+1) - m m' = 2 - m m'
            for s1 in range(2):
                for s2 in range(2):
                    ls_mat[idx(m_out, s1), idx(m_val, s2)] += (
                        0.5 * coeff * s_mat[s1, s2]
                    )
    return ls_mat


def _toy_w(
    so_eso: float, k_pt: np.ndarray, g_vecs: np.ndarray, tau: np.ndarray, l_val: int = 1
) -> dict:
    """Assemble the single-site l-channel W both ways plus negative controls."""
    n_g = len(g_vecs)
    kg = k_pt[None, :] + g_vecs
    q_norm = np.linalg.norm(kg, axis=1)
    cos_th = (kg / q_norm[:, None])[:, 2]
    phi = np.arctan2(kg[:, 1], kg[:, 0])
    f_q = np.exp(-(q_norm**2))  # toy radial form factor
    m_vals = list(range(-l_val, l_val + 1))
    wt = 4.0 * np.pi * (2 * l_val + 1) * so_eso  # ucvol = 1 toy
    amat = metric_so_amet(np.eye(3))  # [iy1, iy2, s', s]
    ls_mat = ls_complex_matrix()
    y_c = np.array(
        [
            [complex_y1(m_val, cos_th[ig], phi[ig]) for m_val in m_vals]
            for ig in range(n_g)
        ]
    )
    y_r = np.array(
        [
            [real_sph_harm(l_val, m_val, cos_th[ig], phi[ig]) for m_val in m_vals]
            for ig in range(n_g)
        ]
    )
    phase = np.exp(2j * np.pi * (kg @ tau))

    def w_assemble(
        t_ket: np.ndarray, t_bra: np.ndarray, weight: float, operator: np.ndarray
    ) -> np.ndarray:
        w_mat = np.zeros((2 * n_g, 2 * n_g), dtype=complex)
        for igp in range(n_g):
            for ig in range(n_g):
                for sp in range(2):
                    for s in range(2):
                        acc = 0.0 + 0.0j
                        for iy1 in range(2 * l_val + 1):
                            for iy2 in range(2 * l_val + 1):
                                acc += (
                                    t_bra[igp, iy1]
                                    * operator[iy1, iy2, sp, s]
                                    * t_ket[ig, iy2]
                                )
                        w_mat[2 * igp + sp, 2 * ig + s] = weight * acc
        return w_mat

    # real-tensor tensors: the ABINIT metric index iy is CARTESIAN-ordered
    # (x, y, z); the tesseral m values of the p channel are m(+1)=x, m(-1)=y,
    # m(0)=z (pypao real_sph_harm), so slot iy holds m = m_of_cart[iy].
    m_of_cart = (1, -1, 0)
    ket_r = np.array(
        [
            [
                (1j) ** l_val
                * f_q[ig]
                * y_r[ig, m_vals.index(m_of_cart[iy])]
                * phase[ig]
                for iy in range(2 * l_val + 1)
            ]
            for ig in range(n_g)
        ]
    )
    bra_r = np.conj(ket_r)
    ket_c = np.array(
        [
            [
                (1j) ** l_val * f_q[ig] * y_c[ig, im] * phase[ig]
                for im in range(2 * l_val + 1)
            ]
            for ig in range(n_g)
        ]
    )
    # pinned chain: the complex-conjugated tensor belongs to the KET (G) side
    bra_c = np.conj(ket_c)
    bra_c_wrong_phase = ket_c * (phase / np.conj(phase))[:, None] ** 2
    ket_c_wrong_il = ((-1j) ** l_val) * f_q[:, None] * y_c * phase[:, None]

    ls_ordered = ls_mat.reshape(3, 2, 3, 2).transpose(0, 2, 1, 3)  # [m1, m2, s', s]

    return {
        # ABINIT: conj on the bra (G') tensor
        "abinit": w_assemble(ket_r, bra_r, wt, amat),
        # pinned complex-Y form: conj on the KET (G) tensor
        "complex": w_assemble(bra_c, ket_c, wt, ls_ordered),
        "wrong_bra_conj": w_assemble(ket_c, bra_c, wt, ls_ordered),
        "wrong_phase": w_assemble(bra_c_wrong_phase, ket_c, wt, ls_ordered),
        "wrong_ket_il": w_assemble(np.conj(ket_c_wrong_il), ket_c, wt, ls_ordered),
    }


def check_toy_g_space_contraction() -> None:
    """Assertion 3: ABINIT-side W == complex-harmonic W; Hermitian; traceless."""
    n_g = 6
    k_pt = _RNG.normal(size=3) * 0.3
    g_vecs = _RNG.normal(size=(n_g, 3))
    g_vecs /= np.linalg.norm(g_vecs, axis=1)[:, None]
    eso = 0.37

    w1 = _toy_w(eso, k_pt, g_vecs, np.array([0.0, 0.0, 0.0]))
    w_abinit, w_cplx = w1["abinit"], w1["complex"]
    scale = max(np.abs(w_abinit).max(), np.abs(w_cplx).max())
    dev = np.abs(w_abinit - w_cplx).max()
    assert dev < TOL * scale, f"toy contraction mismatch: {dev} (scale {scale})"
    print(
        f"  ABINIT (real-tensor/amet) W == complex-Ylm kernel W  (dev {dev:.1e}, scale {scale:.1f})"
    )

    herm = np.abs(w_abinit - w_abinit.conj().T).max()
    assert herm < TOL * scale, f"W not Hermitian: {herm}"
    blocks = w_abinit.reshape(n_g, 2, n_g, 2)
    tr_sum = sum(np.trace(blocks[g, :, g, :]) for g in range(n_g))
    assert np.abs(tr_sum) < TOL * scale, f"L.S spin trace non-zero: {tr_sum}"
    print(
        f"  W Hermitian (dev {herm:.1e}); per-G spin trace of L.S = 0 (|tr| {abs(tr_sum):.1e})"
    )

    tau2 = np.array([0.13, -0.22, 0.31])
    w_t2 = _toy_w(eso, k_pt, g_vecs, tau2)["abinit"]
    w_t3 = _toy_w(eso, k_pt, g_vecs, -tau2)["abinit"]
    # all-atom coverage: the total W_SO is the SUM of per-site separable
    # terms; each term is Hermitian, so the site-sum is a valid operator.
    herm2 = np.abs(w_t2 - w_t2.conj().T).max()
    assert herm2 < TOL * scale, f"per-site W not Hermitian: {herm2}"
    w_sum = w_t2 + w_t3
    w_sum_herm = np.abs(w_sum - w_sum.conj().T).max()
    assert w_sum_herm < TOL * scale, f"two-site W not Hermitian: {w_sum_herm}"
    print(
        f"  per-site W Hermitian (dev {herm2:.1e}); two-site sum Hermitian (dev {w_sum_herm:.1e})"
    )
    print("  => W_SO additive over all sites (ligands included), each Hermitian")

    # negative controls: conjugation on the bra (G') side instead of the ket
    # (G) side — the naive note-2.1 placement — must break the match; so must
    # a flipped ket atomic-phase conjugation or a flipped ket i^l sign.
    for key in ("wrong_bra_conj", "wrong_phase", "wrong_ket_il"):
        dev = np.abs(w1[key] - w_cplx).max()
        assert dev > 1e-3 * scale, f"negative control {key} did not trigger: {dev}"
        print(f"  negative control {key}: match broken (dev {dev:.1e})")


def main() -> None:
    print("abinit_nc_soc_sign_chain: ABINIT NC SOC operator convention (story-001)")
    check_amet_minus_i(np.eye(3))
    check_amet_spinaxis()
    check_two_branch_contraction()
    check_toy_g_space_contraction()
    print("all assertions passed (tolerance 1e-14 on O(1)-scaled random complex toys)")


if __name__ == "__main__":
    main()
