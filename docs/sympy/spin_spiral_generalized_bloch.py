"""Spin-spiral generalized Bloch theorem: sympy-verified derivation.

Research phase of the spin-spiral MFT spec (TB2J <-> TBUpy interface).

Conventions.  Sites carry a spinor index s in {0,1} (interleaved layout,
2*(cell*L + mu) + s, matching TB2J/pauli.py and TBUpy 2*orb+spin).  A
flat spiral about +z has spin angle

    theta[a, mu] = 2 pi q (a + tau_mu) + phi_mu     (cell a, sublattice mu)

and SU(2) rotation Rz(theta) = exp(-i theta sigma_z / 2).  Hopping
classes are keyed by signed lattice vector R:  t(mu, nu, R) is the
amplitude <0 mu sigma|H|R nu sigma>.  For a collinear, spin-diagonal
reference Hamiltonian the spiral Hamiltonian is H_s = U H0 U^dagger:

* the on-site exchange field B_mu sigma_z is INVARIANT (rotation
  commutes with sigma_z),
* every hopping gets a spin-dependent phase
      t(mu,nu,R) exp(-i s (theta[R nu] - theta[0 mu]) / 2).

Pinned, with assertion checks:

1. SU(2) conventions: unitarity, composition, sigma_z invariance,
   active moment rotation (1,0,0) -> (cos t, sin t, 0).

2. Generalized Bloch theorem (central identity): the supercell ring
   Hamiltonian (all classes summed at their residue) has a spectrum
   equal to the union over primitive k-points k_n = n/N of the folded
   twisted Hamiltonian
       Hq(k)_{mu s, nu s'} =
           sum_classes e^{2 pi i k R} t(mu,nu,R)
               exp(+i s alpha_mu/2) exp(-i s' alpha_nu(R)/2),
       alpha_mu = 2 pi q tau_mu + phi_mu,
       alpha_nu(R) = 2 pi q (R + tau_nu) + phi_nu.
   Verified symbolically (characteristic polynomials, symbolic q/tau/phi)
   and numerically for random multi-sublattice models.

3. Hermiticity: with the physical pairing t(nu,mu,-R) = conj(t(mu,nu,R))
   both H_s and Hq(k) are Hermitian.

4. q -> -q (time reversal): spectra of H_s(q) and H_s(-q) coincide, so
   the force-theorem E(q) is even in q, as a Heisenberg map requires.

5. TBUpy convention: one orbital per cell (tau = phi = 0) gives
       Hq(k) = diag(H0(k - q/2), H0(k + q/2)),
   exactly the PHASE_CONVENTION of tbupy.generalized_bloch.

6. Force theorem: at a common reference Fermi level with a smooth
   occupation function, the occupied band energy from the folded
   Hq(k_n) mesh equals the supercell value.

7. Nonorthogonal basis: the rotated overlap pencil (Hq(k_n), Sq(k_n))
   has the same spectrum as the supercell pencil (H_s, S_s).

Run with the mydev environment:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/spin_spiral_generalized_bloch.py
"""

import itertools

import numpy as np
import sympy as sp

# --------------------------------------------------------------------------
# 1. SU(2) rotation conventions
# --------------------------------------------------------------------------

t_sym = sp.Symbol("t", real=True)
SZ = sp.Matrix([[1, 0], [0, -1]])
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])


def Rz(angle):
    """SU(2) rotation exp(-i angle sigma_z / 2)."""
    return sp.Matrix([[sp.exp(-sp.I * angle / 2), 0], [0, sp.exp(sp.I * angle / 2)]])


def check_su2_basics():
    R = Rz(t_sym)
    assert sp.simplify(R.H * R - sp.eye(2)) == sp.zeros(2)
    a, b = sp.symbols("a b", real=True)
    assert sp.simplify(Rz(a) * Rz(b) - Rz(a + b)) == sp.zeros(2)
    assert sp.simplify(R.H * SZ * R - SZ) == sp.zeros(2)
    rho_x = (sp.eye(2) + SX) / 2
    rot = sp.simplify(R * rho_x * R.H)
    expect = (sp.eye(2) + sp.cos(t_sym) * SX + sp.sin(t_sym) * SY) / 2
    assert sp.simplify(rot - expect) == sp.zeros(2), rot
    print(
        "[1] SU(2) basics: unitarity, composition, sigma_z invariance, "
        "active moment rotation (1,0,0) -> (cos t, sin t, 0)  ... OK"
    )


check_su2_basics()


# --------------------------------------------------------------------------
# 2. + 3. Spiral Hamiltonians: symbolic builders
# --------------------------------------------------------------------------


def bond_phase(amu, bnu, R, tau_mu, tau_nu, phi_mu, phi_nu, q, s):
    """exp(-i s (theta[b,nu] - theta[a,mu]) / 2) with signed class R = b - a."""
    theta_diff = 2 * sp.pi * q * (R + tau_nu - tau_mu) + phi_nu - phi_mu
    return sp.exp(-sp.I * s * theta_diff / 2)


def h_class_symbolic(mu, nu, R, amp, taus, phis, q, s_rows=None):
    """Class matrix h(R) for one bond class: entries (s, s') = delta_ss' phase."""
    size = 2 * len(taus)
    h = sp.zeros(size, size)
    for s in (0, 1):
        ph = bond_phase(0, 0, R, taus[mu], taus[nu], phis[mu], phis[nu], q, s)
        h[2 * mu + s, 2 * nu + s] = amp * ph
    return h


def supercell_symbolic(N, L, classes, taus, phis, q):
    """Ring Hamiltonian: H[(a mu),(b nu)] sums classes at residue b-a (mod N).

    classes: list of (mu, nu, R_signed, amplitude).
    """
    size = 2 * L * N
    H = sp.zeros(size, size)

    def idx(a, mu, s):
        return 2 * (a % N) * L + 2 * mu + s

    # group classes by residue
    by_res = {}
    for mu, nu, R, amp in classes:
        by_res.setdefault(R % N, []).append((mu, nu, R, amp))
    for a in range(N):
        for rep, lst in by_res.items():
            b = (a + rep) % N
            for mu, nu, R, amp in lst:
                for s in (0, 1):
                    ph = bond_phase(
                        a, b, R, taus[mu], taus[nu], phis[mu], phis[nu], q, s
                    )
                    H[idx(a, mu, s), idx(b, nu, s)] += amp * ph
    return H


def folded_symbolic(N, L, classes, taus, phis, q, k):
    """Twisted Hamiltonian Hq(k) = sum_classes e^{2 pi i k R} h(R)."""
    size = 2 * L
    H = sp.zeros(size, size)
    for mu, nu, R, amp in classes:
        for s in (0, 1):
            amu = 2 * sp.pi * q * taus[mu] + phis[mu]
            anu = 2 * sp.pi * q * (R + taus[nu]) + phis[nu]
            e = (
                amp
                * sp.exp(2 * sp.pi * sp.I * k * R)
                * sp.exp(sp.I * s * amu / 2)
                * sp.exp(-sp.I * s * anu / 2)
            )
            H[2 * mu + s, 2 * nu + s] += e
    return H


def check_bloch_symbolic():
    """Symbolic characteristic-polynomial equality, small rings."""
    q = sp.Symbol("q", real=True)
    x = sp.Symbol("x")

    # Case A: N=2 cells, L=1, NN classes R=+1 and R=-1 (C2 multigraph ring).
    N, L = 2, 1
    tv = sp.Symbol("t", real=True)
    classes = [(0, 0, 1, tv), (0, 0, -1, tv)]
    Hs = supercell_symbolic(N, L, classes, taus=[0], phis=[0], q=q)
    cp_super = Hs.charpoly(x).as_expr()
    cp_fold = sp.Integer(1)
    for n in range(N):
        Hq = folded_symbolic(
            N, L, classes, taus=[0], phis=[0], q=q, k=sp.Rational(n, N)
        )
        cp_fold *= Hq.charpoly(x).as_expr()
    assert sp.simplify(sp.expand(cp_super - cp_fold)) == 0, "N=2 L=1 mismatch"
    print(
        "[2a] generalized Bloch theorem, N=2 L=1 (C2 ring), symbolic q: "
        "charpoly equality ... OK"
    )

    # Case B: N=4, L=1, classes +-1 and +-2.
    N, L = 4, 1
    classes = [
        (0, 0, 1, tv),
        (0, 0, -1, tv),
        (0, 0, 2, sp.Symbol("t2", real=True)),
        (0, 0, -2, sp.Symbol("t2", real=True)),
    ]
    Hs = supercell_symbolic(N, L, classes, taus=[0], phis=[0], q=q)
    cp_super = Hs.charpoly(x).as_expr()
    cp_fold = sp.Integer(1)
    for n in range(N):
        Hq = folded_symbolic(
            N, L, classes, taus=[0], phis=[0], q=q, k=sp.Rational(n, N)
        )
        cp_fold *= Hq.charpoly(x).as_expr()
    assert sp.simplify(sp.expand(cp_super - cp_fold)) == 0, "N=4 L=1 mismatch"
    print(
        "[2b] generalized Bloch theorem, N=4 L=1, symbolic q: "
        "charpoly equality ... OK"
    )

    # Case C: N=2, L=2 sublattices, symbolic tau_B, phi_B, three bond classes
    # (small enough for a fully symbolic characteristic polynomial).
    N, L = 2, 2
    tauB, phiB = sp.symbols("tau_B phi_B", real=True)
    tAB0 = sp.Symbol("tAB0", real=True)  # A->B inside cell
    tAB1 = sp.Symbol("tAB1", real=True)  # A->B next cell
    tAA = sp.Symbol("tAA", real=True)  # A->A NN
    classes = [
        (0, 1, 0, tAB0),
        (0, 1, 1, tAB1),
        (0, 0, 1, tAA),
        (1, 0, 0, tAB0),
        (1, 0, -1, tAB1),
        (0, 0, -1, tAA),
    ]
    taus = [0, tauB]
    phis = [0, phiB]
    Hs = supercell_symbolic(N, L, classes, taus=taus, phis=phis, q=q)
    cp_super = sp.expand(sp.det(sp.Matrix(x * sp.eye(2 * L * N)) - Hs))
    cp_fold = sp.Integer(1)
    for n in range(N):
        Hq = folded_symbolic(
            N, L, classes, taus=taus, phis=phis, q=q, k=sp.Rational(n, N)
        )
        cp_fold *= sp.expand(sp.det(sp.Matrix(x * sp.eye(2 * L)) - Hq))
    assert sp.simplify(sp.expand(cp_super - cp_fold)) == 0, "N=2 L=2 mismatch"
    print(
        "[2c] generalized Bloch theorem, N=2 L=2, symbolic q/tau_B/phi_B: "
        "charpoly equality ... OK"
    )


check_bloch_symbolic()


# --------------------------------------------------------------------------
# Numeric general-model checks
# --------------------------------------------------------------------------


def hermitian_classes(N, L, seed):
    """Random Hermitian class set for a collinear reference model.

    Signed classes with the physical pairing t(nu,mu,-R) = conj(t(mu,nu,R)),
    on-site eps_mu + B_mu sigma_z.  Returns (bond_classes, eps, Bf) where
    bond_classes is a list of (mu, nu, R_signed, complex amplitude).
    """
    rng = np.random.default_rng(seed)
    classes = []
    Rs = list(range(1, (N - 1) // 2 + 1))
    half = N // 2 if N % 2 == 0 else None
    reps = Rs + ([half] if half else [])
    for mu, nu in itertools.combinations_with_replacement(range(L), 2):
        for R in reps:
            # REAL amplitudes: a TR-invariant (real-hopping) collinear
            # reference.  For complex-hopping references the two chiralities
            # q and -q are genuinely inequivalent and E(q) is not even.
            amp = round(float(rng.normal()), 6) + 0j
            classes.append((mu, nu, R, amp))
            classes.append((nu, mu, -R, amp.conjugate()))
        if mu != nu:
            amp = round(float(rng.normal()), 6) + 0j
            classes.append((mu, nu, 0, amp))
            classes.append((nu, mu, 0, amp.conjugate()))
    eps = [round(rng.normal(), 6) for _ in range(L)]
    Bf = [round(rng.normal(), 6) for _ in range(L)]
    return classes, eps, Bf


def supercell_numeric(N, L, classes, taus, phis, qval, eps, Bf, overlap=False):
    """Numeric supercell H (and optional S) in the absolute gauge."""
    size = 2 * L * N
    H = np.zeros((size, size), dtype=complex)
    S = np.eye(size, dtype=complex) if overlap else None

    def idx(a, mu, s):
        return 2 * (a % N) * L + 2 * mu + s

    by_res = {}
    for mu, nu, R, amp in classes:
        by_res.setdefault(R % N, []).append((mu, nu, R, amp))
    rng = np.random.default_rng(12345)  # noqa: F841 (fixture parity)
    for a in range(N):
        for rep, lst in by_res.items():
            b = (a + rep) % N
            for mu, nu, R, amp in lst:
                for s in (0, 1):
                    dth = (
                        2 * np.pi * qval * (R + float(taus[nu]) - float(taus[mu]))
                        + float(phis[nu])
                        - float(phis[mu])
                    )
                    ph = np.exp(-0.5j * (1 if s == 0 else -1) * dth)
                    H[idx(a, mu, s), idx(b, nu, s)] += amp * ph
                    if overlap:
                        ov = 0.3 * np.exp(-0.7 * (abs(mu - nu) + abs(R))) + 0.1
                        S[idx(a, mu, s), idx(b, nu, s)] += ov * ph
    for a in range(N):
        for mu in range(L):
            for s in (0, 1):
                H[idx(a, mu, s), idx(a, mu, s)] += eps[mu] + (1 - 2 * s) * Bf[mu]
    return (H, S) if overlap else (H, None)


def folded_numeric(N, L, classes, taus, phis, qval, eps, Bf, overlap=False):
    """All N folded Hq(k_n) (and optional Sq(k_n)) matrices."""
    Hs, Ss = [], []
    for n in range(N):
        kval = n / N
        size = 2 * L
        H = np.zeros((size, size), dtype=complex)
        S = np.eye(size, dtype=complex) if overlap else None
        for mu, nu, R, amp in classes:
            for s in (0, 1):
                amu = 2 * np.pi * qval * float(taus[mu]) + float(phis[mu])
                anu = 2 * np.pi * qval * (R + float(taus[nu])) + float(phis[nu])
                ep = (
                    np.exp(2j * np.pi * kval * R)
                    * np.exp(0.5j * (1 if s == 0 else -1) * amu)
                    * np.exp(-0.5j * (1 if s == 0 else -1) * anu)
                )
                H[2 * mu + s, 2 * nu + s] += amp * ep
                if overlap:
                    ov = 0.3 * np.exp(-0.7 * (abs(mu - nu) + abs(R))) + 0.1
                    S[2 * mu + s, 2 * nu + s] += ov * ep
        for mu in range(L):
            for s in (0, 1):
                H[2 * mu + s, 2 * mu + s] += eps[mu] + (1 - 2 * s) * Bf[mu]
                if overlap:
                    pass  # identity part already present
        Hs.append(H)
        Ss.append(S)
    return Hs, Ss


def check_numeric_spectra():
    for seed in (1, 2, 3):
        N, L = 5, 2
        classes, eps, Bf = hermitian_classes(N, L, seed)
        taus = [0.0, 0.25]
        phis = [0.0, np.pi / 3]
        for qval in (0.13, 0.37, 0.5):
            Hs, _ = supercell_numeric(N, L, classes, taus, phis, qval, eps, Bf)
            assert np.allclose(Hs, Hs.conj().T, atol=1e-12), "H_s not hermitian"
            evals_s = np.linalg.eigvalsh(Hs)
            Hs2, _ = supercell_numeric(N, L, classes, taus, phis, -qval, eps, Bf)
            evals_s2 = np.linalg.eigvalsh(Hs2)
            assert np.allclose(evals_s, evals_s2, atol=1e-9), "E(q) != E(-q)"
            Hlist, _ = folded_numeric(N, L, classes, taus, phis, qval, eps, Bf)
            evals_fold = np.sort(np.concatenate([np.linalg.eigvalsh(H) for H in Hlist]))
            assert np.allclose(
                evals_s, evals_fold, atol=1e-9
            ), f"folded spectrum mismatch seed={seed} q={qval}"
            Hq0, _ = folded_numeric(N, L, classes, taus, phis, qval, eps, Bf)
            for H in Hq0:
                assert np.allclose(H, H.conj().T, atol=1e-12), "Hq not hermitian"
    print(
        "[2d] general models (N=5, L=2, 3 seeds x 3 q): folded == supercell "
        "spectra; H_s, Hq hermitian; E(q) = E(-q)  ... OK"
    )


check_numeric_spectra()


def check_tbupy_convention():
    """One orbital per cell: Hq(k) == diag(H0(k-q/2), H0(k+q/2))."""
    N, L = 4, 1
    classes, eps, Bf = hermitian_classes(N, L, 7)
    taus, phis = [0.0], [0.0]

    def H0(kval, s):
        h = eps[0] + (1 - 2 * s) * Bf[0] + 0j
        for mu, nu, R, amp in classes:
            assert (mu, nu) == (0, 0)
            h += amp * np.exp(2j * np.pi * kval * R)
        return h

    qval = 0.23
    Hlist, _ = folded_numeric(N, L, classes, taus, phis, qval, eps, Bf)
    for n in range(N):
        kval = n / N
        expect = np.array(
            [[H0(kval - qval / 2, 0), 0.0], [0.0, H0(kval + qval / 2, 1)]],
            dtype=complex,
        )
        assert np.allclose(
            Hlist[n], expect, atol=1e-12
        ), "up/down shifted-block convention violated"
    print(
        "[5] TBUpy convention: Hq(k) = diag(H0(k-q/2), H0(k+q/2)) "
        "(PHASE_CONVENTION reproduced)  ... OK"
    )


check_tbupy_convention()


def check_force_theorem_band_energy():
    """Occupied band energy: folded mesh == supercell at fixed efermi."""
    N, L = 5, 2
    classes, eps, Bf = hermitian_classes(N, L, 11)
    taus, phis = [0.0, 0.25], [0.0, np.pi / 3]
    qval = 0.3
    Hs, _ = supercell_numeric(N, L, classes, taus, phis, qval, eps, Bf)
    evals_s = np.linalg.eigvalsh(Hs)
    Hlist, _ = folded_numeric(N, L, classes, taus, phis, qval, eps, Bf)
    evals_f = np.concatenate([np.linalg.eigvalsh(H) for H in Hlist])
    width = 0.05

    def eband(ev):
        mu0 = float(np.quantile(np.sort(np.concatenate([evals_s, evals_f])), 0.63))
        occ = 1.0 / (1.0 + np.exp((ev - mu0) / width))
        return float(np.sum(ev * occ))

    e_s, e_f = eband(evals_s), eband(evals_f)
    assert abs(e_s - e_f) < 1e-9 * max(1.0, abs(e_s)), (e_s, e_f)
    print(
        "[6] force theorem: folded k-mesh occupied band energy == supercell "
        f"value (diff {abs(e_s - e_f):.2e})  ... OK"
    )


check_force_theorem_band_energy()


def _pencil_eigs(H, S):
    """Hermitian generalized eigenproblem via Cholesky of S."""
    Lc = np.linalg.cholesky(S)
    Li = np.linalg.inv(Lc)
    A = Li @ H @ Li.conj().T
    return np.linalg.eigvalsh(0.5 * (A + A.conj().T))


def check_nonorthogonal_pencil():
    """Generalized pencil spectra: (Hq, Sq) folded == (H_s, S_s) supercell."""
    N, L = 4, 2
    classes, eps, Bf = hermitian_classes(N, L, 17)
    taus, phis = [0.0, 0.25], [0.0, np.pi / 3]
    qval = 0.29
    Hs, Ss = supercell_numeric(N, L, classes, taus, phis, qval, eps, Bf, overlap=True)
    Ss = 0.5 * (Ss + Ss.conj().T)
    assert np.min(np.linalg.eigvalsh(Ss)) > 0, "overlap not positive definite"
    evals_s = np.sort(_pencil_eigs(Hs, Ss))
    Hlist, Slist = folded_numeric(
        N, L, classes, taus, phis, qval, eps, Bf, overlap=True
    )
    evals_f = []
    for H, S in zip(Hlist, Slist):
        S = 0.5 * (S + S.conj().T)
        evals_f.extend(_pencil_eigs(H, S))
    evals_f = np.sort(np.array(evals_f))
    assert np.allclose(
        evals_s, evals_f, atol=1e-9
    ), "nonorthogonal pencil spectra mismatch"
    print(
        "[7] nonorthogonal basis: folded pencil spectra == supercell "
        "pencil spectra  ... OK"
    )


check_nonorthogonal_pencil()

print("\nAll assertions passed.")
