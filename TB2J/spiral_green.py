"""Folded-pencil Green functions for spin-spiral frozen states (story 003).

Normative conventions: ``docs/sympy/spiral_green_function_mft.py``
(sections 1-2) and the spiral frozen-bundle contract
(``tbupy_spiral_state`` v1):

* interleaved spin basis ``(orb0 up, orb0 down, orb1 up, ...)``, eV;
* folded pencils ``Hq(k), Sq(k)`` from the contract rebuild rule, with
  the tbupy multi-sublattice assembler as single source of truth;
* resolvent ``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}`` with the S-weighted
  pole sum ``Tr[Sq Gq] = sum_n 1/(E - eps_n)``;
* unfolding to real-space spin blocks by the inverse twisted-Bloch
  transform, Eq. (4) of the derivation script: the basis carries the
  cell half-twist ``exp(+i sigma_s pi q a)`` only, while the sublattice
  phases (``taus``, ``phis``) live in ``Hq(k)`` - the unfold factors
  are therefore tau/phi-independent.

:class:`SpiralState` mirrors ``tbupy.spiral_state.SpiralState``
field-for-field so bundles round-trip between the repos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SpiralState",
    "SpiralGreen",
    "SpiralGreenAtE",
    "assemble_folded_pencil",
    "load_spiral_state",
    "spectral_density",
]


@dataclass
class SpiralState:
    """TB2J mirror of the ``tbupy_spiral_state`` bundle (schema v1).

    Collinear reference channels ``HR_up``/``HR_dn`` and the
    spin-independent overlap ``SR`` in ``(nR, norb, norb)`` layout over
    the stored signed cells ``Rlist``; spiral wavevector ``q_frac``;
    per-orbital fractional positions ``taus`` and sublattice phase
    offsets ``phis``; twisted-gauge local exchange field magnitudes
    ``B_local``; rotating-frame density matrix ``rho`` and Hubbard
    potential ``V_U`` (both spin-interleaved ``(2 norb, 2 norb)``);
    frozen Fermi level ``efermi``; per-orbital torque residuals
    ``torque_norms``; JSON metadata string (dc type, smearing width,
    guard flags, ...).
    """

    HR_up: np.ndarray
    HR_dn: np.ndarray
    SR: np.ndarray
    Rlist: np.ndarray
    q_frac: np.ndarray
    taus: np.ndarray
    phis: np.ndarray
    B_local: np.ndarray
    rho: np.ndarray | None = None
    V_U: np.ndarray | None = None
    efermi: float = 0.0
    torque_norms: np.ndarray | None = None
    metadata_json: str = "{}"
    schema_version: int = 1
    kpts: np.ndarray | None = None
    kweights: np.ndarray | None = None
    occupations: np.ndarray | None = None
    orbital_to_atom: np.ndarray | None = None
    constraint_potential: np.ndarray | None = None
    evals: np.ndarray | None = None
    evecs: np.ndarray | None = None
    field_symmetry: dict | None = None
    # NOTE: density_discrepancy is set dynamically by v2 validation
    # (mirroring tbupy.spiral_state), NOT a declared field, so the
    # dataclass field lists of the two mirrors stay identical.

    @property
    def norb(self) -> int:
        return int(np.asarray(self.HR_up).shape[1])

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)

    @property
    def width(self) -> float:
        """Occupation smearing width (eV) from the bundle metadata."""
        return float(self.metadata.get("width", 0.1))

    def validate(self) -> None:
        """Check bundle shapes against the contract schema."""
        self.schema_version = int(self.schema_version)
        if self.schema_version == 2:
            _validate_spiral_state_v2(self)
            return
        HR = np.asarray(self.HR_up)
        if HR.ndim != 3 or HR.shape[1] != HR.shape[2]:
            raise ValueError(f"HR_up must have shape (nR, norb, norb), got {HR.shape}")
        nR, norb = HR.shape[0], HR.shape[1]
        for name in ("HR_dn", "SR"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != HR.shape:
                raise ValueError(
                    f"{name} must have shape {HR.shape} like HR_up, got {arr.shape}"
                )
        rlist = np.asarray(self.Rlist)
        if rlist.shape != (nR, 3):
            raise ValueError(f"Rlist must have shape (nR, 3), got {rlist.shape}")
        for name, shape in (
            ("q_frac", (3,)),
            ("taus", (norb, 3)),
            ("phis", (norb,)),
            ("B_local", (norb,)),
        ):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        for name in ("rho", "V_U", "torque_norms"):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            want = (norb,) if name == "torque_norms" else (2 * norb, 2 * norb)
            if arr.shape != want:
                raise ValueError(f"{name} must have shape {want}, got {arr.shape}")


def _sigma_z_field(B_local) -> np.ndarray:
    """Interleaved on-site field ``diag(+B/2 on up, -B/2 on down)``."""
    B = np.asarray(B_local, dtype=float)
    out = np.zeros((2 * len(B), 2 * len(B)))
    out[0::2, 0::2] += np.diag(0.5 * B)
    out[1::2, 1::2] -= np.diag(0.5 * B)
    return out


def assemble_folded_pencil(state: SpiralState, k_frac) -> tuple[np.ndarray, np.ndarray]:
    """Contract rebuild rule at one k.

    ``Hq(k) = assemble_multisublattice_hq_sq(HR_up, HR_dn, SR, Rlist,
    config, k) + interleave(B_local x sigma_z) + V_U``, with
    ``config = MultiSublatticeSpiralConfig(q_frac, taus, phis)``; ``V_U``
    is a rotating-frame operator added cell-periodically (R=0 block).
    Returns the interleaved spinor pair ``(Hq, Sq)`` of shape
    ``(2 norb, 2 norb)``.
    """
    from tbupy.generalized_bloch import (
        MultiSublatticeSpiralConfig,
        assemble_multisublattice_hq_sq,
    )

    config = MultiSublatticeSpiralConfig(state.q_frac, state.taus, state.phis)
    Hq, Sq = assemble_multisublattice_hq_sq(
        state.HR_up, state.HR_dn, state.SR, state.Rlist, config, k_frac
    )
    Hq = Hq + _sigma_z_field(state.B_local)
    if state.V_U is not None:
        Hq = Hq + np.asarray(state.V_U, dtype=complex)
    return Hq, Sq


@dataclass
class SpiralGreenAtE:
    """Energy snapshot of :class:`SpiralGreen` (returned by ``at(E)``)."""

    green: "SpiralGreen"
    E: complex
    evals: np.ndarray  # (nk, n) pencil eigenvalues
    evecs: np.ndarray  # (nk, n, n) standardized eigenvectors, C^dag Sq C = I
    f: np.ndarray  # (nk, n) frozen occupations at efermi/width
    Gq: np.ndarray  # (nk, n, n) resolvent (E Sq - Hq)^{-1}

    def pole_sum(self) -> np.ndarray:
        """``sum_n 1/(E - eps_n(k))`` per k point, shape ``(nk,)``."""
        return np.sum(1.0 / (self.E - self.evals), axis=1)

    def pole_sum_trace(self, S=None) -> np.ndarray:
        """Per-k S-weighted pole sum ``Tr[S Gq(k, E)]``, shape ``(nk,)``.

        With ``S = Sq(k)`` this equals :meth:`pole_sum`
        (derivation-script section 1); with ``S=None`` it is the plain
        trace ``Tr Gq(k, E)``.  ``S`` may be a single ``(n, n)`` operator
        or a stacked ``(nk, n, n)`` array.
        """
        if S is None:
            return np.trace(self.Gq, axis1=1, axis2=2)
        S = np.asarray(S, dtype=complex)
        if S.ndim == 2:
            return np.einsum("ij,kji->k", S, self.Gq)
        if S.shape != self.Gq.shape:
            raise ValueError(f"S must have shape {self.Gq.shape}, got {S.shape}")
        return np.einsum("kij,kji->k", S, self.Gq)

    def unfold(self, i, j_cells) -> np.ndarray:
        """Real-space spin blocks ``G_q[(0, i), (R, j)](E)``.

        ``i`` is the orbital index in cell 0; ``j_cells`` is an iterable
        of ``(j, R)`` pairs with orbital index ``j`` and integer lattice
        displacement ``R = b - a``.  Returns complex array
        ``(len(j_cells), 2, 2)`` indexed by spin ``(s, s')``.

        Inverse twisted-Bloch transform, Eq. (4)::

            G_s[(a i s), (b j s')](E) = (1/N) sum_m e^{2 pi i k_m (a-b)}
                conj(tw(s, m)) Gq(k_m)[2i+s, 2j+s'] tw(s', m),
            tw(s, m) = exp(+i sigma_s pi q m),  sigma_up = +1.

        The mesh must be the uniform commensurate mesh
        ``k_m = (m/N, 0, 0)`` of the N-cell supercell.
        """
        green = self.green
        nk = green.kmesh.shape[0]
        m = np.arange(nk)
        if not (
            np.allclose(green.kmesh[:, 0], m / nk)
            and np.allclose(green.kmesh[:, 1:], 0.0)
        ):
            raise ValueError(
                "unfold requires the uniform commensurate mesh k_m = (m/N, 0, 0)"
            )
        q = float(np.asarray(green.state.q_frac, dtype=float)[0])
        sigma = np.array([1.0, -1.0])  # sigma_up = +1, sigma_dn = -1
        tw = np.exp(1j * np.pi * sigma[:, None] * q * m[None, :])  # (2, nk)
        i = int(i)
        out = np.empty((len(j_cells), 2, 2), dtype=complex)
        for row, item in enumerate(j_cells):
            j, R = item
            j = int(j)
            R = np.asarray(R, dtype=float)
            if R.ndim == 0:  # scalar displacement along the ring axis
                R = np.array([R, 0.0, 0.0])
            bloch = np.exp(-2j * np.pi * (green.kmesh @ R)) / nk  # a - b = -R
            sub = self.Gq[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            out[row] = np.einsum("m,sm,mst,tm->st", bloch, np.conj(tw), sub, tw)
        return out


class SpiralGreen:
    """Folded-pencil Green function of a frozen spiral bundle.

    Prebuilds ``Hq(k), Sq(k)`` on the given k mesh (contract rebuild
    rule) and the standardized pencil eigenbasis
    ``Hq C = eps Sq C, C^dag Sq C = I`` once (batched over k); per
    energy it exposes the frozen occupations, the resolvent
    ``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}``, S-weighted pole sums, and
    unfolding to real-space spin blocks.
    """

    def __init__(self, state: SpiralState, kmesh, kweights=None, assembler=None):
        state.validate()
        kmesh = np.asarray(kmesh, dtype=float)
        if kmesh.ndim != 2 or kmesh.shape[1] != 3:
            raise ValueError(f"kmesh must have shape (nk, 3), got {kmesh.shape}")
        nk = kmesh.shape[0]
        if kweights is None:
            kweights = np.full(nk, 1.0 / nk)
        kweights = np.asarray(kweights, dtype=float)
        if kweights.shape != (nk,):
            raise ValueError(
                f"kweights must have shape (nk,) = ({nk},), got {kweights.shape}"
            )
        if (
            state.schema_version == 2
            and state.kpts is not None
            and not np.array_equal(kmesh, np.asarray(state.kpts, dtype=float))
        ):
            raise ValueError(
                "v2 bundles must be evaluated on a k mesh that exactly "
                "match persisted kpts (frozen occupations are bound to them)"
            )
        self.state = state
        self.kmesh = kmesh
        self.kweights = kweights
        self.norb = state.norb
        self.n = 2 * self.norb
        if assembler is None:
            assembler = assemble_folded_pencil
        Hq = np.empty((nk, self.n, self.n), dtype=complex)
        Sq = np.empty((nk, self.n, self.n), dtype=complex)
        for ik, k in enumerate(kmesh):
            Hq[ik], Sq[ik] = assembler(state, k)
        self.Hq = Hq
        self.Sq = Sq
        # Standardized pencil eigenbasis, batched over k: S = L L^dag,
        # A = L^{-1} Hq L^{-dag}, C = L^{-dag} U with A U = eps U.
        L = np.linalg.cholesky(Sq)
        Li = np.linalg.inv(L)
        Li_dag = np.conj(np.swapaxes(Li, 1, 2))
        A = Li @ Hq @ Li_dag
        A = 0.5 * (A + np.conj(np.swapaxes(A, 1, 2)))
        evals, U = np.linalg.eigh(A)
        self.evals = evals
        self.evecs = Li_dag @ U

    def frozen_occupations(self, evals=None) -> np.ndarray:
        """Frozen occupations.

        Schema v2: the persisted per-band occupations (bound to the
        persisted evals/evecs provenance) are authoritative. Schema v1:
        Fermi occupations from the frozen efermi and smearing width.
        """
        if self.state.schema_version == 2 and evals is None:
            return np.array(self.state.occupations)
        if evals is None:
            evals = self.evals
        return 1.0 / (1.0 + np.exp((evals - self.state.efermi) / self.state.width))

    def at(self, E: complex) -> SpiralGreenAtE:
        """Energy snapshot: eigenbasis quantities, occupations, resolvent."""
        E = complex(E)
        f = self.frozen_occupations()
        Gq = np.linalg.inv(E * self.Sq - self.Hq)
        return SpiralGreenAtE(
            green=self, E=E, evals=self.evals, evecs=self.evecs, f=f, Gq=Gq
        )

    def eigenbasis(self, E: complex):
        """Eigenbasis quantities at ``E``: ``(evals, evecs, frozen f_n)``."""
        at = self.at(E)
        return at.evals, at.evecs, at.f

    def resolvent(self, E: complex) -> np.ndarray:
        """``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}``, vectorized over k."""
        return self.at(E).Gq


# ---------------------------------------------------------------------------
# Schema v2 mirror: persisted-eigenpair provenance (TBUpy planar states)
# ---------------------------------------------------------------------------

_V2_REQUIRED_META = (
    "schema_version",
    "gauge",
    "q_frac",
    "electron_count",
    "occupation_rule",
    "width",
    "hubbard",
    "constraint",
    "field_symmetry",
    "field_role",
    "field_rotation_policy",
)


def _validate_spiral_state_v2(state: "SpiralState") -> None:
    """Strict v2 provenance checks (mirrors tbupy.spiral_state)."""
    required = (
        state.kpts,
        state.kweights,
        state.occupations,
        state.orbital_to_atom,
        state.constraint_potential,
        state.evals,
        state.evecs,
        state.field_symmetry,
    )
    if any(x is None for x in required):
        raise ValueError(
            "schema v2 requires kpts, kweights, occupations, "
            "orbital_to_atom, constraint_potential, evals, evecs, and "
            "field_symmetry"
        )
    state.kpts = np.asarray(state.kpts, dtype=float)
    state.kweights = np.asarray(state.kweights, dtype=float)
    state.occupations = np.asarray(state.occupations, dtype=float)
    state.orbital_to_atom = np.asarray(state.orbital_to_atom, dtype=np.int64)
    state.constraint_potential = np.asarray(state.constraint_potential, dtype=complex)
    state.evals = np.asarray(state.evals, dtype=float)
    state.evecs = np.asarray(state.evecs, dtype=complex)
    norb, nk = len(np.asarray(state.taus)), len(state.kpts)
    n = 2 * norb
    if state.kpts.shape != (nk, 3) or state.kweights.shape != (nk,):
        raise ValueError("kpts/kweights must have shapes (nk,3)/(nk,)")
    if not np.isclose(state.kweights.sum(), 1.0, atol=1e-8):
        raise ValueError("kweights must sum to one")
    if state.orbital_to_atom.shape != (norb,) or np.any(state.orbital_to_atom < 0):
        raise ValueError("orbital_to_atom must be a nonnegative (norb,) mapping")
    if state.constraint_potential.shape != (n, n):
        raise ValueError(f"constraint_potential must have shape ({n},{n})")
    if (
        state.evals.ndim != 2
        or state.evals.shape[0] != nk
        or state.evecs.shape != (nk, n, state.evals.shape[1])
    ):
        raise ValueError("evals/evecs must have shapes (nk,nband)/(nk,nbasis,nband)")
    if (
        state.occupations.shape != state.evals.shape
        or np.any(~np.isfinite(state.occupations))
        or np.any((state.occupations < 0) | (state.occupations > 1))
    ):
        raise ValueError("occupations must match evals and lie in [0,1]")
    meta = json.loads(state.metadata_json)
    for key in _V2_REQUIRED_META:
        if key not in meta:
            raise ValueError(f"v2 metadata missing '{key}'")
    functional = meta["hubbard"]
    if not isinstance(functional, dict) or not all(
        key in functional for key in ("hubbard_dict", "hubbard_type", "dc_type")
    ):
        raise ValueError("v2 Hubbard/DC functional provenance is incomplete")
    if (
        meta["schema_version"] != 2
        or meta["gauge"] != "planar_y"
        or not np.allclose(meta["q_frac"], np.asarray(state.q_frac, dtype=float))
    ):
        raise ValueError("v2 metadata schema/gauge/q_frac mismatch")
    if (
        meta.get("field_role")
        not in (
            "external",
            "constraint_proxy",
            "intrinsic_exchange",
        )
        or meta.get("field_rotation_policy") is None
    ):
        raise ValueError("v2 field role and rotation policy must be explicit")
    if bool(meta.get("hard_rotation_applied", False)):
        raise ValueError("hard-rotated rho cannot be used as a spectral projector")
    if not isinstance(meta["constraint"], dict) or not meta["constraint"]:
        raise ValueError("constraint must explicitly identify kind='none' or operators")
    if (
        meta["constraint"].get("kind") == "none"
        and np.max(np.abs(state.constraint_potential)) > 1e-12
    ):
        raise ValueError("kind='none' requires a zero constraint_potential")
    if meta["constraint"].get("kind") != "none":
        records = meta["constraint"].get("sites")
        if not isinstance(records, list) or any(
            not all(
                k in r
                for k in (
                    "operator",
                    "type",
                    "target",
                    "multiplier",
                    "rotation",
                    "hold_fixed",
                )
            )
            for r in records
        ):
            raise ValueError(
                "constrained v2 provenance requires per-site "
                "operator/type/target/multiplier/rotation/hold_fixed"
            )
    if meta["field_symmetry"] != state.field_symmetry:
        raise ValueError("field_symmetry field disagrees with v2 metadata")
    if (
        abs(
            float(meta["electron_count"])
            - float(np.sum(state.kweights[:, None] * state.occupations))
        )
        > 1e-8
    ):
        raise ValueError("electron_count disagrees with persisted fixed occupations")
    if state.rho is None:
        raise ValueError("v2 requires a same-frame stored rho")
    rho_spec = _rho_from_eigenpairs(state)
    state.density_discrepancy = float(np.linalg.norm(rho_spec - state.rho))
    tol = float(meta.get("density_tolerance", 1e-10))
    if state.density_discrepancy > tol:
        raise ValueError(
            f"spectral density mismatch: ||rho_spec-rho||="
            f"{state.density_discrepancy:.6g} > {tol:.6g}"
        )


def spectral_density(state: "SpiralState") -> np.ndarray:
    """Reconstruct the same-frame occupied spinor density from the
    persisted eigenpairs and occupations (schema v2 only). Pure
    function; provenance validation lives in :meth:`SpiralState.validate`."""
    if state.schema_version != 2:
        raise ValueError("spectral density reconstruction requires schema v2")
    return _rho_from_eigenpairs(state)


def _rho_from_eigenpairs(state: "SpiralState") -> np.ndarray:
    return np.einsum(
        "k,kn,kbn,kcn->bc",
        state.kweights,
        state.occupations,
        state.evecs,
        state.evecs.conj(),
        optimize=True,
    )


def load_spiral_state(path) -> "SpiralState":
    """Read a ``tbupy_spiral_state`` v1 or v2 sidecar into the mirror."""
    from scipy.io import netcdf_file

    with netcdf_file(str(path), "r", mmap=False) as nc:
        name = getattr(nc, "schema_name", "")
        if isinstance(name, bytes):
            name = name.decode()
        version = int(getattr(nc, "schema_version", -1))
        if name != "tbupy_spiral_state" or version not in (1, 2):
            raise ValueError(
                f"Unsupported spiral state schema {name!r} version {version}"
            )

        def _f8(key):
            return np.array(nc.variables[key][:], dtype=float)

        def _cx(key):
            return _f8(f"{key}_real") + 1j * _f8(f"{key}_imag")

        # v1 stores the collinear channels as real f8; v2 stores every
        # matrix as a complex (real, imag) pair (tbupy save_spiral_state).
        _chan = _cx if version == 2 else _f8
        common = dict(
            HR_up=_chan("HR_up"),
            HR_dn=_chan("HR_dn"),
            SR=_chan("SR"),
            Rlist=np.array(nc.variables["Rlist"][:], dtype=np.int64),
            q_frac=_f8("q_frac"),
            taus=_f8("taus"),
            phis=_f8("phis"),
            B_local=_f8("B_local"),
            rho=_cx("rho"),
            V_U=_cx("V_U"),
            efermi=float(_f8("efermi").ravel()[0]),
            metadata_json=_metadata(nc),
        )
        if version == 2:
            meta = json.loads(common["metadata_json"])
            common.update(
                schema_version=2,
                kpts=_f8("kpts"),
                kweights=_f8("kweights"),
                occupations=_f8("occupations"),
                orbital_to_atom=np.array(
                    nc.variables["orbital_to_atom"][:], dtype=np.int64
                ),
                constraint_potential=_cx("constraint_potential"),
                evals=_f8("evals"),
                evecs=_cx("evecs"),
                field_symmetry=meta.get("field_symmetry"),
            )
        if "torque_norms" in nc.variables:
            common["torque_norms"] = _f8("torque_norms")
        return SpiralState(**common)


def _metadata(nc) -> str:
    raw = np.array(nc.variables["metadata_json"][:])
    if raw.size == 0:
        return "{}"
    return bytes(raw).decode("utf-8").strip()
