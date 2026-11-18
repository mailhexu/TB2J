"""Explicit magnon-phonon scattering vertex (Story 009).

Normative derivation: ``docs/sympy/magnon_phonon_vertex.py`` (+ ``.md``); the
formulas below are asserted there against a mechanical real-space finite-model
reference.  Numeric kernels in this module are filesystem- and SCF-free.

Conventions
-----------
- Spin Hamiltonian ``H = -sum_b J_b S_i . S_j`` over a **directed** bond list
  ``(i, j, R)`` with both directions present; ``J`` in the full-moment
  SpinIO convention (the ``Magnon.JR`` entries), energies eV, displacements
  Angstrom.  ``J > 0`` is ferromagnetic.
- Fourier ``A(q) = sum_R A(R) exp(-2 pi i q . R)`` (fractional reciprocal).
- Collinear reference (all local ``V`` frames parallel); canonical Story-006
  site-boson modes ``u, v`` (unit-commutator HP bosons, ``sqrt(2 S)`` carried
  by the transverse spin).
- Phonon displacement for mode ``(q, nu)``:
  ``tau_{a alpha}(L) = ZPA_{a nu} e^{mw}_{a alpha, nu}(q) e^{2 pi i q L}
  (b_{q nu} + b^dag_{-q nu}) / sqrt(N_cell)`` with
  ``ZPA = sqrt(hbar / (2 M_a omega))``: the mass/zero-point factor is applied
  exactly once, here, when classical ``dJ/dtau`` rows meet the mode pattern.
- Vertex (manuscript eq:mp-vertex):

  ``H_mp = sum_{k q} g a^dag_{k+q} a_k (b_q + b^dag_{-q}) / sqrt(N_cell)``
  ``g_{mu nu nu'}(k, q) = - sum_b O^pat_{b, nu'}(q) F_b[mu, nu](k, q)``

  with the asserted bond factor (``w(p)`` = Story-006 ``v`` array at ``p``)::

    F_b[mu, nu](k, q) =
        e^{+2 pi i k.R}     t_ij  [u*_{mu i}(k+q) u_{nu j}(k)
                                    + w*_{nu i}(-k) w_{mu j}(-k-q)]
      + e^{-2 pi i (k+q).R} t_ij* [u*_{mu j}(k+q) u_{nu i}(k)
                                    + w*_{nu j}(-k) w_{mu i}(-k-q)]
      - S_j [u*_{mu i}(k+q) u_{nu i}(k) + w*_{nu i}(-k) w_{mu i}(-k-q)]
      - S_i e^{-2 pi i q.R} [u*_{mu j}(k+q) u_{nu j}(k)
                                    + w*_{nu j}(-k) w_{mu j}(-k-q)]
    t_ij = sqrt(S_i S_j) / 2 (U_i . U_j*)

  The longitudinal partner-spin term of the far endpoint carries
  ``e^{-2 pi i q.R}`` because that fluctuation sits in cell ``L+R`` while the
  displacement modulation is tied to the bond's home cell.  Per directed bond
  ``F_b[mu, nu](k, q) = [F_b[nu, mu](k+q, -q)]*``, so with ``O(-q) = O(q)*``
  the vertex satisfies the manuscript Hermiticity
  ``g_{mu nu nu'}(k, q) = [g_{nu mu nu'}(k+q, -q)]*``.  On-site bonds have an
  identically zero vertex; the scalar collinear coupling cannot generate a
  bilinear ``a^dag b`` mixing block (both asserted in the derivation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from TB2J.magnon.eigenstates import SiteBosonModes

#: angular-frequency prefactor: omega = 2 pi f_THZ_TO_HZ * f[THz]
THZ_TO_HZ = 1e12
#: displacements below this zero-point amplitude are rejected as unstable
FREQUENCY_TOL_THZ = 1e-8
#: tolerance for the collinearity control ``U_i . U_j`` and ``V_i x V_j``
COLLINEARITY_TOL = 1e-8
#: pattern momentum below which home-cell rows are exact
GAMMA_TOL = 1e-8


@dataclass(frozen=True)
class PhononModeInput:
    """Primitive-cell phonon mode data consumed by the vertex.

    Duck-type contract (``TB2J.magnon_phonon.loaders.as_phonon_mode_input``
    builds this from e.g. an ``unfold_epc.phonon.PhononModes`` object):

    - ``q``: fractional reciprocal momentum, shape ``(3,)``.
    - ``frequencies_thz``: ordinary frequencies in THz, shape ``(nmodes,)``;
      non-positive frequencies are rejected (an exactly-zero Gamma acoustic
      branch has no finite zero-point amplitude and is out of scope here).
    - ``masses_amu``: primitive-cell masses in u, shape ``(natom,)``.
    - ``eigenvectors``: complex array ``(3 * natom, nmodes)`` in phonopy's
      mass-weighted, unit-L2 convention, rows ordered ``(atom, axis)``,
      including the basis-position phase.  The zero-point factor
      ``sqrt(hbar / 2 M omega)`` must NOT be folded into these rows; it is
      applied exactly once by this module.
    """

    q: np.ndarray
    frequencies_thz: np.ndarray
    masses_amu: np.ndarray
    eigenvectors: np.ndarray
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrientedBondDerivatives:
    """Structure-only classical exchange-derivative container.

    ``dJ_dtau[b, a, alpha] = d J_b / d tau_{a alpha}`` in eV/Angstrom in the
    full-moment ``JR`` convention, for directed bonds ``(i_b, j_b, R_b)``.
    ``convention`` is one of

    - ``"home_cell"``: rows for displacing atom ``a`` of the home primitive
      cell.  Exact for building vertices at ``q = 0`` only (all pattern phases
      unity, as in the periodic matched-row evidence); ``vertex`` enforces
      this.
    - ``"pattern_projected"``: rows already contracted with a frozen pattern
      at the stored momentum ``source_q`` (complex, eV per unit pattern
      amplitude); usable at commensurate finite ``q``.
    """

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    dJ_dtau: np.ndarray
    convention: str = "home_cell"
    source_q: np.ndarray | None = None
    units: str = "eV/Angstrom"


@dataclass(frozen=True)
class PatternBondDerivatives:
    """Per-directed-bond pattern-projected complex rows (eV) at one momentum."""

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    dJ_pattern: np.ndarray  # (n_bonds, n_phonon_modes) complex
    q: np.ndarray


@dataclass(frozen=True)
class ModeProjectedDerivatives:
    """Finite-q S002 exchange derivatives projected onto a named mode.

    ``O_pattern`` and ``FD_pattern`` are complex, full-moment ``dJ/dtau``
    coefficients in eV/Angstrom for the report's unit-Cartesian polarization.
    The polarization, mode label and source identity are retained; the vertex
    multiplies the derivative by the corresponding generalized-coordinate
    zero-point amplitude exactly once.
    """

    q: np.ndarray
    mode: str
    mode_index: int
    polarization: np.ndarray
    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    O_pattern: np.ndarray
    FD_pattern: np.ndarray
    source: dict = field(default_factory=dict)
    mode_evidence_sha256: str = ""


@dataclass(frozen=True)
class VertexResult:
    """Materialized vertex block at one ``(k, q)``; arrays are read-only."""

    g: np.ndarray  # (n_modes_kq, n_modes_k, n_phonon_modes), complex, eV
    kpoints: np.ndarray  # the four queried points: k+q, k, -k, -(k+q)
    phonon_q: np.ndarray
    magnon_energies: np.ndarray  # (4, n_modes) eV, order of ``kpoints``
    phonon_frequencies_thz: np.ndarray
    O_pattern: np.ndarray  # (n_bonds, n_phonon_modes) complex, eV
    n_bonds: int
    collinearity_residual: float
    conventions: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, dtype in (
            ("g", complex),
            ("kpoints", float),
            ("phonon_q", float),
            ("magnon_energies", float),
            ("phonon_frequencies_thz", float),
            ("O_pattern", complex),
        ):
            array = np.asarray(getattr(self, name), dtype=dtype)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _fold(k):
    return np.mod(np.asarray(k, dtype=float), 1.0)


def _zero_point_amplitudes(frequencies_thz, masses_amu):
    """sqrt(hbar / (2 M omega)) in Angstrom, per atom and mode."""
    from ase.units import _amu, _hbar

    freq = np.asarray(frequencies_thz, dtype=float)
    if np.any(freq <= FREQUENCY_TOL_THZ):
        raise ValueError(
            "Vertex requires positive phonon frequencies; got minimum "
            f"{freq.min():.3g} THz. Gamma acoustic zero-point amplitudes are "
            "divergent and need the documented limiting treatment."
        )
    omega = 2 * np.pi * freq * THZ_TO_HZ
    masses = np.asarray(masses_amu, dtype=float) * _amu
    zpa_m = np.sqrt(_hbar / (2.0 * masses[:, None] * omega[None, :]))
    return zpa_m * 1e10  # (natom, nmodes) Angstrom


def project_home_cell_rows(bonds: OrientedBondDerivatives, phonon: PhononModeInput):
    """Contract home-cell ``dJ/dtau`` rows with a phonon pattern at ``q ~ 0``.

    All pattern phases are unity at Gamma, so the contraction is exact there;
    finite-``q`` vertices require pattern-projected rows.  Returns
    :class:`PatternBondDerivatives` with the zero-point factor applied once.
    """
    if bonds.convention != "home_cell":
        raise ValueError(f"expected home_cell convention, got {bonds.convention!r}")
    q = np.asarray(phonon.q, dtype=float)
    if not np.allclose(q, 0.0, atol=GAMMA_TOL):
        raise ValueError(
            "home_cell derivative rows are exact only at q = 0; supply "
            "pattern_projected rows from the finite-q campaign for "
            f"q = {q.tolist()}"
        )
    dJ = np.asarray(bonds.dJ_dtau, dtype=complex)
    if dJ.ndim != 3 or dJ.shape[2] != 3:
        raise ValueError("home_cell dJ_dtau must contain all three Cartesian axes")
    eig = np.asarray(phonon.eigenvectors, dtype=complex)
    natom = dJ.shape[1]
    if eig.shape[0] != 3 * natom:
        raise ValueError(
            f"eigenvector rows {eig.shape[0]} do not match 3 * natom {3 * natom}"
        )
    zpa = _zero_point_amplitudes(phonon.frequencies_thz, phonon.masses_amu)
    n_modes = eig.shape[1]
    rows = np.zeros((dJ.shape[0], n_modes), dtype=complex)
    for mode in range(n_modes):
        pattern = eig[:, mode]  # (3 * natom,)
        # rows[b, m] = sum_{a, alpha} dJ[b, a, alpha] * e_{a alpha} * ZPA_{a m}
        rows[:, mode] = np.einsum(
            "bax,ax,a->b", dJ, pattern.reshape(natom, 3), zpa[:, mode]
        )
    return PatternBondDerivatives(
        i=np.asarray(bonds.i, dtype=int),
        j=np.asarray(bonds.j, dtype=int),
        R=np.asarray(bonds.R, dtype=int),
        dJ_pattern=rows,
        q=q,
    )


def _projected_mode_zero_point(phonon: PhononModeInput, mode_index: int):
    """Return unit-Cartesian polarization and its generalized ZPA in Angstrom."""
    from ase.units import _amu, _hbar

    frequencies = np.asarray(phonon.frequencies_thz, dtype=float)
    masses = np.asarray(phonon.masses_amu, dtype=float)
    if mode_index < 0 or mode_index >= len(frequencies):
        raise ValueError(f"phonon mode index {mode_index} is out of range")
    mass_weighted = np.asarray(phonon.eigenvectors[:, mode_index], dtype=complex)
    cart = mass_weighted.reshape(len(masses), 3) / np.sqrt(masses[:, None])
    norm = np.linalg.norm(cart)
    if norm <= 0:
        raise ValueError("phonon mode has a zero Cartesian polarization")
    cart /= norm
    omega = 2 * np.pi * frequencies[mode_index] * THZ_TO_HZ
    effective_mass_kg = np.sum(masses * np.sum(np.abs(cart) ** 2, axis=1)) * _amu
    zpa_ang = np.sqrt(_hbar / (2 * effective_mass_kg * omega)) * 1e10
    return cart, float(zpa_ang)


def _validate_bonds(bonds: OrientedBondDerivatives, nspin: int):
    i = np.asarray(bonds.i, dtype=int)
    j = np.asarray(bonds.j, dtype=int)
    if i.min() < 0 or j.min() < 0 or i.max() >= nspin or j.max() >= nspin:
        raise ValueError("bond site index out of range for the magnon model")
    if np.asarray(bonds.dJ_dtau).shape[0] != len(i):
        raise ValueError("dJ_dtau rows do not match the bond list")
    return i, j


def _collinearity_residual(U, V):
    """max |U_i . U_j| (no conjugate) and pairwise non-parallelism of V frames.

    For collinear frames U_i . U_j = 0 identically (real orthonormal
    transverse pairs related by a rotation about the common V), while
    U_i . U_j* = 2 for i = j.
    """
    dot_U = np.einsum("ix,jx->ij", U, U)
    res_U = np.max(np.abs(dot_U))
    cross = np.cross(V[:, None, :], V[None, :, :])
    res_V = np.max(np.linalg.norm(cross, axis=-1))
    return float(max(res_U, res_V))


class MagnonPhononModel:
    """ADR-006 model seam: ``vertex(k, q)`` materializes the requested block.

    Parameters
    ----------
    magnon:
        A :class:`TB2J.magnon.magnon3.Magnon` whose ``get_site_boson_modes``
        provides the canonical site-boson modes (Story 006 contract).
    bond_derivatives:
        :class:`OrientedBondDerivatives` in the full-moment ``JR`` convention,
        or a hash-joined :class:`ModeProjectedDerivatives` from the S002 loader.
        The latter represents one report mode and requires ``mode_index`` in
        :meth:`vertex`.
    phonons:
        Optional provider with ``modes_at(q) -> PhononModeInput``; otherwise
        pass ``phonon_modes`` explicitly to :meth:`vertex`.
    """

    def __init__(
        self,
        magnon,
        bond_derivatives: OrientedBondDerivatives | ModeProjectedDerivatives,
        phonons=None,
    ):
        self.magnon = magnon
        self.bond_derivatives = bond_derivatives
        self.phonons = phonons

    def _pattern_rows(self, phonon: PhononModeInput, mode_index=None):
        bonds = self.bond_derivatives
        if isinstance(bonds, ModeProjectedDerivatives):
            if mode_index is None:
                raise ValueError(
                    "finite-q mode-projected reports require an explicit phonon mode_index"
                )
            if int(mode_index) != bonds.mode_index:
                raise ValueError(
                    f"phonon mode index {mode_index} does not match report mode "
                    f"{bonds.mode!r} index {bonds.mode_index}"
                )
            q = np.asarray(phonon.q, dtype=float)
            if not np.allclose(bonds.q, q, atol=GAMMA_TOL, rtol=0):
                raise ValueError(
                    f"mode-projected report q {bonds.q.tolist()} does not match requested q {q.tolist()}"
                )
            provenance = dict(phonon.provenance)
            if provenance.get("mode_evidence_sha256") != bonds.mode_evidence_sha256:
                raise ValueError(
                    "unsupported finite-q join: phonon mode evidence provenance "
                    "does not match the S002 report source"
                )
            if provenance.get("mode") != bonds.mode:
                raise ValueError("phonon mode identity does not match S002 report mode")
            cart, zpa = _projected_mode_zero_point(phonon, int(mode_index))
            O = np.array(bonds.O_pattern, dtype=complex)
            if bonds.polarization is not None:
                polarization = np.array(bonds.polarization, dtype=complex)
                if polarization.shape != cart.shape:
                    raise ValueError(
                        "report polarization shape does not match phonon mode"
                    )
                polarization /= np.linalg.norm(polarization)
                overlap = np.vdot(polarization.reshape(-1), cart.reshape(-1))
                if abs(overlap) < 1 - 1e-6:
                    raise ValueError(
                        f"report polarization does not match phonon mode {mode_index}"
                    )
                # O_pattern is the derivative along the REPORT's polarization;
                # the requested mode direction is cart = overlap * P_report
                # (linear directional derivative), so rotate the pattern rows.
                O = O * overlap
            return PatternBondDerivatives(
                i=np.asarray(bonds.i, dtype=int),
                j=np.asarray(bonds.j, dtype=int),
                R=np.asarray(bonds.R, dtype=int),
                dJ_pattern=(O * zpa)[:, None],
                q=q,
            )
        if bonds.convention == "pattern_projected":
            q = np.asarray(phonon.q, dtype=float)
            if bonds.source_q is None or not np.allclose(
                np.asarray(bonds.source_q, dtype=float), q, atol=GAMMA_TOL
            ):
                raise ValueError(
                    "pattern_projected rows do not match the requested phonon "
                    f"momentum {q.tolist()}"
                )
            return PatternBondDerivatives(
                i=np.asarray(bonds.i, dtype=int),
                j=np.asarray(bonds.j, dtype=int),
                R=np.asarray(bonds.R, dtype=int),
                dJ_pattern=np.asarray(bonds.dJ_dtau, dtype=complex),
                q=q,
            )
        return project_home_cell_rows(bonds, phonon)

    def _reference_frames(self):
        """Spin lengths and local frames of the magnon reference + collinearity."""
        from TB2J.magnon.magnon_math import get_rotation_arrays

        magmoms = np.asarray(self.magnon.magmom, dtype=float)
        spins = np.linalg.norm(magmoms, axis=-1) / 2.0
        normalized = magmoms / np.linalg.norm(magmoms, axis=-1)[:, None]
        U, V = get_rotation_arrays(
            normalized, u=np.asarray(self.magnon._uz, dtype=float)
        )
        return (
            spins,
            np.asarray(U, dtype=complex),
            np.asarray(V, dtype=float),
            _collinearity_residual(U, V),
        )

    def vertex(
        self,
        k,
        q,
        phonon_modes: PhononModeInput | None = None,
        site_boson_modes: SiteBosonModes | None = None,
        mode_index: int | None = None,
    ) -> VertexResult:
        """Materialize ``g_{mu nu nu'}(k, q)`` for all branches.

        ``site_boson_modes`` optionally injects precomputed (possibly
        rephased) Story-006 modes; it must contain exactly the four folded
        momenta ``[k+q, k, -(k+q), -k]`` in that order.  Any degenerate
        subspace gauge of the injected modes yields the covariant vertex
        ``g -> Theta g Theta^dag`` with the asserted invariants.

        A :class:`ModeProjectedDerivatives` input represents one S002 report
        mode and requires its explicit ``mode_index``.
        """
        k = np.asarray(k, dtype=float).reshape(3)
        q = np.asarray(q, dtype=float).reshape(3)
        kp = k + q
        points = [kp, k, -kp, -k]
        folded = np.array([_fold(p) for p in points])

        if phonon_modes is None:
            if self.phonons is None:
                raise ValueError("no phonon modes: pass phonon_modes or a provider")
            phonon = self.phonons.modes_at(q)
        else:
            phonon = phonon_modes
        if not np.allclose(np.asarray(phonon.q, dtype=float), q, atol=GAMMA_TOL):
            raise ValueError("phonon mode momentum does not match the requested q")
        frequencies = np.asarray(phonon.frequencies_thz, dtype=float)
        masses = np.asarray(phonon.masses_amu, dtype=float)
        eigenvectors = np.asarray(phonon.eigenvectors)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(
                "phonon frequencies must be a nonempty one-dimensional array"
            )
        if np.any(frequencies <= FREQUENCY_TOL_THZ):
            raise ValueError(
                "Vertex requires positive phonon frequencies; Gamma acoustic "
                "zero-point amplitudes are divergent."
            )
        if masses.ndim != 1 or np.any(masses <= 0):
            raise ValueError("phonon masses must be a positive one-dimensional array")
        if eigenvectors.shape != (3 * len(masses), len(frequencies)):
            raise ValueError("phonon eigenvector shape must be (3 * natom, n_modes)")

        spins, U, V, collinear_residual = self._reference_frames()
        if collinear_residual > COLLINEARITY_TOL:
            raise ValueError(
                "Noncollinear reference detected (frame residual "
                f"{collinear_residual:.3g}); the explicit scalar-collinear "
                "vertex supports collinear references only."
            )

        if site_boson_modes is None:
            modes = self.magnon.get_site_boson_modes(folded)
        else:
            modes = site_boson_modes
            if modes.kpoints.shape != folded.shape or not np.allclose(
                modes.kpoints, folded, atol=1e-12
            ):
                raise ValueError(
                    "injected site_boson_modes must be evaluated at the four "
                    f"folded points {folded.tolist()}"
                )
        spins = np.asarray(modes.spin_lengths, dtype=float)

        if isinstance(self.bond_derivatives, ModeProjectedDerivatives):
            i_b = np.asarray(self.bond_derivatives.i, dtype=int)
            j_b = np.asarray(self.bond_derivatives.j, dtype=int)
            if (
                i_b.size == 0
                or j_b.size != i_b.size
                or i_b.min() < 0
                or j_b.min() < 0
                or i_b.max() >= len(spins)
                or j_b.max() >= len(spins)
            ):
                raise ValueError(
                    "mode-projected report bond sites do not match spin model"
                )
            rows = self._pattern_rows(phonon, mode_index=mode_index)
        else:
            i_b, j_b = _validate_bonds(self.bond_derivatives, len(spins))
            rows = self._pattern_rows(phonon, mode_index=mode_index)
        R_b = np.asarray(rows.R, dtype=float)
        O = np.asarray(rows.dJ_pattern, dtype=complex)
        # full-moment rows -> unit-direction exchange derivative consumed by F
        O = O / (spins[i_b][:, None] * spins[j_b][:, None])

        u_kp = np.asarray(modes.u[0])  # at k+q
        u_k = np.asarray(modes.u[1])  # at k
        w_mkp = np.asarray(modes.v[2])  # v-array at -(k+q)
        w_mk = np.asarray(modes.v[3])  # v-array at -k

        p1 = np.exp(2j * np.pi * (R_b @ k))
        p2 = np.exp(-2j * np.pi * (R_b @ kp))
        p3 = np.exp(-2j * np.pi * (R_b @ q))
        t_b = (
            np.sqrt(spins[i_b] * spins[j_b])
            / 2
            * np.einsum("bx,bx->b", U[i_b], np.conj(U[j_b]))
        )

        def pair(p, site, conj):
            """(n_mode, n_bonds) amplitudes gathered per bond site."""
            block = p[:, site]
            return np.conj(block) if conj else block

        # each structure is an outer product over (mu, nu) per bond
        A_ij = np.einsum("mb,nb->mnb", pair(u_kp, i_b, True), pair(u_k, j_b, False))
        W_ij = np.einsum("nb,mb->mnb", pair(w_mk, i_b, True), pair(w_mkp, j_b, False))
        A_ji = np.einsum("mb,nb->mnb", pair(u_kp, j_b, True), pair(u_k, i_b, False))
        W_ji = np.einsum("nb,mb->mnb", pair(w_mk, j_b, True), pair(w_mkp, i_b, False))
        D_i = np.einsum(
            "mb,nb->mnb", pair(u_kp, i_b, True), pair(u_k, i_b, False)
        ) + np.einsum("nb,mb->mnb", pair(w_mk, i_b, True), pair(w_mkp, i_b, False))
        D_j = np.einsum(
            "mb,nb->mnb", pair(u_kp, j_b, True), pair(u_k, j_b, False)
        ) + np.einsum("nb,mb->mnb", pair(w_mk, j_b, True), pair(w_mkp, j_b, False))

        F = (
            p1[None, None, :] * t_b[None, None, :] * (A_ij + W_ij)
            + p2[None, None, :] * np.conj(t_b)[None, None, :] * (A_ji + W_ji)
            - spins[j_b][None, None, :] * D_i
            - spins[i_b][None, None, :] * p3[None, None, :] * D_j
        )
        g = -np.einsum("bv,mnb->mnv", O, F)
        reported_mode = isinstance(self.bond_derivatives, ModeProjectedDerivatives)
        output_frequencies = np.asarray(phonon.frequencies_thz, dtype=float)
        if reported_mode:
            output_frequencies = output_frequencies[[int(mode_index)]]
        conventions = {
            "fourier": "A(q) = sum_R A(R) exp(-2 pi i q.R)",
            "spin_hamiltonian": "H = -sum_b J_b S_i.S_j (full-moment JR convention)",
            "phonon_displacement": (
                "tau = ZPA e^{mw} e^{2 pi i q L} (b_q + b^dag_-q)/sqrt(N_cell); "
                "zero-point factor applied once here"
            ),
            "vertex": "H_mp = sum g a^dag_{k+q} a_k (b_q + b^dag_-q)/sqrt(N_cell)",
            "rows_convention": (
                "mode_projected_report"
                if reported_mode
                else self.bond_derivatives.convention
            ),
        }
        if reported_mode:
            conventions["mode_identity"] = self.bond_derivatives.mode
            conventions["mode_evidence_sha256"] = (
                self.bond_derivatives.mode_evidence_sha256
            )
        result = VertexResult(
            g=np.asarray(g, dtype=complex),
            kpoints=np.array(points, dtype=float),
            phonon_q=np.asarray(phonon.q, dtype=float).copy(),
            magnon_energies=np.asarray(modes.energies, dtype=float).copy(),
            phonon_frequencies_thz=output_frequencies.copy(),
            O_pattern=np.asarray(O, dtype=complex),
            n_bonds=len(i_b),
            collinearity_residual=collinear_residual,
            conventions=conventions,
        )
        return result
