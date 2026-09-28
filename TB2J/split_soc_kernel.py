"""Shared KS-band split-SOC exchange kernel (ADR-1, story 002).

One exchange core consumed by all split-SOC backend adapters (GPAW,
ABINIT NC/PAW, VASP).  The kernel never sees an LCAO Hamiltonian:

- it consumes *normalized* no-SOC spinful KS band data (a
  ``ProjectorGreenData`` with ``nspinor=2``: eigenvalues ``E(k)``,
  rectangular all-atom projector maps ``B_a(k)`` stored as the coefficient
  array, and magnetic rotation vertices ``v_a`` stored as the spinor
  operator) plus the all-atom SOC operator in the band window,
  ``W^K_SO(k) = <psi_n|W_SO|psi_m>``, an ``(nkpt, N, N)`` Hermitian matrix
  (the spin structure of W lives inside the spinor states; there is no
  separate 2x2 block index on the band space);
- **second variation** (production): diagonalize ``E(k) + lam*W(k)`` per
  k, rotate the spectral coefficients ``C'[m,s,p] = sum_n C[n,s,p] X[n,m]``
  (band-space rotation on the ``(n, s)``-indexed coefficients; the ket-side
  ``<p s|psi>`` contract of the spectral replay), and re-enter the existing
  spinor projector kernel — all powers of lam within the retained window;
- **first-order insertion** (diagnostic): analytic ``dG = G0 W G0`` with
  both trace-line topologies (docs/sympy/split_soc_insertion.md), never a
  finite-difference stand-in;
- magnetic vertices are only ever contracted, ``V_a = B_a^dag v_a B_a``
  embedded or (equivalently, and as implemented) left in projector space
  inside the site blocks ``B_a^dag G B_b``; ``B_a`` is a transition map and
  is **never inverted**;
- W_SO is all-atom (ligand SOC enters the DMI), vertices are magnetic-only.

Every output carries an FR-050/ADR-8 provenance metadata block
(:func:`split_soc_provenance`).  The kernel is gauge-independent: the
psi/chi frame choice and any output-frame rotation belong to the adapters.
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace

import numpy as np

from TB2J.Jtensor import decompose_J_tensor
from TB2J.projector_green import (
    PAULI_MATRICES,
    ProjectorGreen,
    magnetic_tangent_vertices,
    spinor_dense_block,
    spinor_tangent_pair_matrix,
    spinor_tangent_trace,
)

MODE_SECOND_VARIATION = "second_variation"
MODE_FIRST_ORDER_INSERTION = "first_order_insertion"
SPLIT_SOC_MODES = (MODE_SECOND_VARIATION, MODE_FIRST_ORDER_INSERTION)

PROVENANCE_SCHEMA = "tb2j.split_soc_ks_provenance/1.0"
SOC_UNITS = "eV"
PAULI_ORDER = "x,y,z"

#: right-handed (u, v, n) triads: the transverse plane measured by a leg
#: whose magnetic reference axis is ``n`` (leg z measures (x, y), leg x
#: measures (y, z), leg y measures (z, x)).
TRANSVERSE_AXES = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

#: contour prescription applied to the tangent traces
TANGENT_NORMALIZATION = "J^{ab} = Im contour K^{ab} dz / (2 pi)"

_HERMITIAN_TOL = 1.0e-8
_AXIS_ALIGN_TOL = 1.0e-6
_CARTESIAN_AXES = np.eye(3)


# ---------------------------------------------------------------------------
# band-space primitives (pinned contracts)
# ---------------------------------------------------------------------------


def second_variation_spectrum(eigenvalues, w_soc, lam=1.0):
    """Diagonalize ``E(k) + lam*W(k)`` per k point.

    ``eigenvalues``: ``(nkpt, N)`` (or ``(1, nkpt, N)``) normalized no-SOC
    spinor band energies (eV).  ``w_soc``: ``(nkpt, N, N)`` Hermitian.
    Returns ``(eps2, x)`` with ``eps2`` ``(nkpt, N)`` second-variational
    eigenvalues (ascending) and ``x`` ``(nkpt, N, N)`` unitary rotations
    whose *columns* are the new states in the no-SOC band basis.  The
    spectral replay built from ``(eps2, x)`` is gauge-invariant under
    degenerate-subspace rotations of ``x``.
    """
    eps = np.asarray(eigenvalues, dtype=float)
    if eps.ndim == 3:
        if eps.shape[0] != 1:
            raise ValueError("eigenvalues batch dimension must be 1 (nspinor=2)")
        eps = eps[0]
    if eps.ndim != 2:
        raise ValueError("eigenvalues must have shape (nkpt, nstate)")
    nkpt, nstate = eps.shape
    w = np.asarray(w_soc, dtype=complex)
    if w.shape != (nkpt, nstate, nstate):
        raise ValueError(
            f"w_soc must have shape {(nkpt, nstate, nstate)}, got {w.shape}"
        )
    ham = lam * w
    ham[:, np.arange(nstate), np.arange(nstate)] += eps
    eps2, x = np.linalg.eigh(ham)
    return eps2, x


def rotate_spinor_coefficients(coefficients, x):
    """Rotate spectral coefficients into the second-variational basis.

    ``C'[k, m, s, p] = sum_n C[k, n, s, p] X[k, n, m]`` — the band-space
    rotation acts on the ``(n, s)`` state index of the ket-side coefficient
    ``<p s|psi>``; the projector index is untouched (the rectangular map
    stays a map).  Accepts and returns the ``(1, nkpt, N, 2, nproj)`` (or
    ``(nkpt, N, 2, nproj)``) ProjectorGreenData layout.
    """
    c = np.asarray(coefficients, dtype=complex)
    x = np.asarray(x, dtype=complex)
    if c.ndim == 5:
        if c.shape[0] != 1:
            raise ValueError("coefficients batch dimension must be 1 (nspinor=2)")
        return np.einsum("knsp,knm->kmsp", c[0], x)[None, ...]
    if c.ndim == 4:
        return np.einsum("knsp,knm->kmsp", c, x)
    raise ValueError(
        "coefficients must have shape (1, nkpt, nstate, 2, nproj); got " f"{c.shape}"
    )


def embed_site_vertex(b_a, v_a):
    """Embed a site vertex into the band window: ``V_a = B_a^dag v_a B_a``.

    ``b_a``: ``(nproj_a, N, 2)`` rectangular projector map
    (``B^a_{mu, n sigma} = <p_{a mu}|psi_{n sigma k}>``); ``v_a``:
    ``(nproj_a, nproj_a, 2, 2)`` local spin-rotation vertex.  Returns the
    ``(N, N)`` band-space vertex.  ``B_a`` is only contracted, never
    inverted — no squareness or completeness is assumed.
    """
    b = np.asarray(b_a, dtype=complex)
    v = np.asarray(v_a, dtype=complex)
    if b.ndim != 3 or b.shape[2] != 2:
        raise ValueError("b_a must have shape (nproj, nstate, 2)")
    if v.shape != (b.shape[0], b.shape[0], 2, 2):
        raise ValueError("v_a must have shape (nproj, nproj, 2, 2) matching b_a")
    return np.einsum("pns,pqst,qmt->nm", b.conj(), v, b)


# ---------------------------------------------------------------------------
# FR-050 provenance
# ---------------------------------------------------------------------------


def split_soc_provenance(
    *,
    mode,
    lam,
    nband,
    soc_operator=None,
    strength0_reference=None,
    frame=None,
    vertices_sites=None,
    merge_mode=None,
    extra=None,
):
    """Build the FR-050/ADR-8 provenance metadata block for one output.

    Records the strength-0 reference, SOC-operator source/units/coverage,
    frame/spinaxis, Pauli order, band-window definition, magnetic-only
    vertices, and merge mode.  Adapters merge backend-specific fields via
    ``soc_operator``/``strength0_reference``/``frame``/``extra``.

    Attestation semantics (fields the kernel cannot verify from its
    inputs): ``all_atom_coverage`` and ``soc_operator.coverage`` are
    *adapter-attested* — the backend must guarantee that ``W_SO`` covers
    all SOC-bearing atoms (ligands included) before claiming them;
    ``band_window.convergence_study`` starts as ``None`` and MUST be
    attached by the adapter from
    :func:`band_window_convergence_report` (or an equivalent study)
    before the output is published.
    """
    meta = {
        "schema": PROVENANCE_SCHEMA,
        "mode": mode,
        "quantity": (
            "exchange"
            if mode == MODE_SECOND_VARIATION
            else "exchange_lambda_derivative"
        ),
        "lambda": float(lam),
        "units": SOC_UNITS,
        "pauli_order": PAULI_ORDER,
        "strength0_reference": dict(strength0_reference or {}),
        "soc_operator": {
            "units": SOC_UNITS,
            "coverage": "all_atoms",
            "coverage_attestation": "adapter-attested (not verifiable by the kernel)",
            "site_decomposition": False,
            **dict(soc_operator or {}),
        },
        "frame": {"spinaxis": "provider_frame", **dict(frame or {})},
        "band_window": {
            "nband": int(nband),
            "definition": "normalized no-SOC spinor band window (as provided)",
            "convergence_study": None,
        },
        "vertices": {
            "magnetic_only": True,
            "sites": [int(s) for s in (vertices_sites or [])],
            "soc_free": True,
        },
        "merge_mode": merge_mode or "single_leg",
        "all_atom_coverage": True,
    }
    if extra:
        meta.update(extra)
    return meta


def json_safe_provenance(value):
    """Retain numeric FR-050 studies while encoding R-tuple keys for JSON."""
    if isinstance(value, dict):
        return {
            json.dumps(key, separators=(",", ":"))
            if isinstance(key, tuple)
            else str(key): json_safe_provenance(item)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return json_safe_provenance(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_safe_provenance(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# shared validation
# ---------------------------------------------------------------------------


def _validate_kernel_data(data):
    if getattr(data, "nspinor", 1) != 2:
        raise ValueError("the split-SOC kernel requires nspinor=2 data")
    if getattr(data, "band_mask", None) is not None:
        raise ValueError(
            "band_mask is not supported: materialize the band window into "
            "eigenvalues/coefficients before calling the split-SOC kernel"
        )
    if getattr(data, "overlap_k", None) is not None:
        raise ValueError(
            "the split-SOC kernel consumes normalized band data: overlap_k "
            "must be None (use the generalized-S form for non-orthogonal "
            "metrics, docs/sympy/split_soc_insertion.md)"
        )
    if getattr(data, "spinor_operator", None) is None:
        raise ValueError(
            "magnetic rotation vertices are required: set spinor_operator "
            "(the split-SOC kernel has no local-Hamiltonian fallback)"
        )


def _validated_w_soc(data, w_soc):
    w = np.asarray(w_soc, dtype=complex)
    if w.ndim == 2:
        w = w[None, :, :]
    if w.shape != (data.nkpt, data.nband, data.nband):
        raise ValueError(
            f"w_soc must have shape {(data.nkpt, data.nband, data.nband)}, "
            f"got {w.shape}"
        )
    scale = max(1.0, float(np.abs(w).max(initial=0.0)))
    if not np.allclose(
        w,
        np.conj(np.swapaxes(w, 1, 2)),
        rtol=_HERMITIAN_TOL,
        atol=_HERMITIAN_TOL * scale,
    ):
        raise ValueError("w_soc must be Hermitian at every k point")
    return w


def _validated_sites(data, sites):
    nsite = len(data.site_nproj)
    if sites is None:
        return list(range(nsite))
    out = [int(s) for s in sites]
    if any(s < 0 or s >= nsite for s in out):
        raise ValueError(f"sites out of range [0, {nsite}): {sites}")
    return out


def _validated_rpts(Rpts):
    if Rpts is None:
        # same default as the existing spinor kernel
        # (compute_spinor_projector_exchange): the full nmax=1 grid
        from TB2J.interfaces.gpaw_projector import _R_grid

        arr = _R_grid(nmax=1)
    else:
        arr = np.asarray(Rpts, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Rpts must have shape (nR, 3)")
    keys = [tuple(int(x) for x in r) for r in arr]
    have = set(keys)
    missing = [r for r in keys if tuple(-x for x in r) not in have]
    if missing:
        raise ValueError(
            "Rpts must include each negative R vector; missing: " f"{missing[0]}"
        )
    return arr, keys


# ---------------------------------------------------------------------------
# first-order insertion (diagnostic mode)
# ---------------------------------------------------------------------------


def first_order_insertion_channels(data, w_soc, Rpts, energies, sites=None):
    """Analytic first-order SOC insertion at explicit complex energies.

    Builds ``G0(z,k)`` from the strength-0 eigenvalues and the insertion
    ``dG(k) = G0(k) W(k) G0(k)`` in the band window, contracts both into
    the projector blocks through the rectangular maps, restores the
    full-BZ Fourier phases per inter-site ``R`` (same ``exp(-2 pi i k.R)``
    convention and k weights as the production replay), and evaluates the
    two trace-line topologies of the *magnetic tangent* matrix per
    (R, i, j)::

        dK^{ab}/dlam|_0 = Tr[V_i^a dG_ij V_j^b G0_ji]
                        + Tr[V_i^a G0_ij V_j^b dG_ji]

    with ``V_i^a`` the physical magnetic tangent vertices
    (:func:`TB2J.projector_green.magnetic_tangent_vertices`) and ``G0``
    the full complex spinor strength-0 Green blocks.  This is the
    *derivative at lam = 0*: ``dK`` is ``d/dlam`` of the tangent matrix
    and ``pair_trace`` of the dense pair trace — no SOC scaling is applied
    here.

    Returns ``{"energies", "K0", "dK", "pair_trace0", "pair_trace"}`` with
    ``(nE, 3, 3)`` tangent-matrix arrays and ``(nE,)`` dense pair traces —
    the story-001 topology objects ``Tr[v_a dg_ij v_b g0_ji] +
    Tr[v_a g0_ij v_b dg_ji]`` (and their strength-0 partner) evaluated per
    energy on the full magnetic vertex ``v_a``.  This is the
    diagnostic/audit mode; the production mode is
    :func:`compute_ks_split_soc_exchange` with ``mode="second_variation"``.
    """
    _validate_kernel_data(data)
    w = _validated_w_soc(data, w_soc)
    sites = _validated_sites(data, sites)
    rpts, rkeys = _validated_rpts(Rpts)
    energies = np.asarray(energies, dtype=complex)
    r_index = {r: ir for ir, r in enumerate(rkeys)}

    green = ProjectorGreen(data)
    eps = data.eigenvalues[0]  # (nkpt, N)
    coeff = data.coefficients[0]  # (nkpt, N, 2, nproj)
    ops = green.get_local_operators_spinor(sites=sites)
    vert_info = {site: magnetic_tangent_vertices(op) for site, op in ops.items()}
    vertices = {site: info["vertices"] for site, info in vert_info.items()}
    dense_ops = {site: spinor_dense_block(op) for site, op in ops.items()}
    phase = np.exp(green.k2Rfactor * np.einsum("ri,ki->rk", rpts, green.kpts))
    phase = phase * green.kweights[None, :]

    pairs = [(r, i, j) for r in rkeys for i in sites for j in sites]
    k0 = {key: np.empty((energies.size, 3, 3), dtype=complex) for key in pairs}
    dk = {key: np.empty((energies.size, 3, 3), dtype=complex) for key in pairs}
    tr0 = {key: np.empty(energies.size, dtype=complex) for key in pairs}
    tr = {key: np.empty(energies.size, dtype=complex) for key in pairs}
    for ie, energy in enumerate(energies):
        g0_diag = 1.0 / (energy + green.efermi - eps)  # (nkpt, N)
        g0_full = green.get_Gk_all_spinor(energy)  # (nkpt, np, np, 2, 2)
        d_m = g0_diag[..., :, None] * w * g0_diag[..., None, :]  # G0 W G0
        dg_full = np.einsum(
            "knsp,knm,kmtq->kpqst", coeff, d_m, coeff.conj(), optimize="optimal"
        )
        gr0 = np.einsum("kpqst,rk->rpqst", g0_full, phase, optimize="optimal")
        dgr = np.einsum("kpqst,rk->rpqst", dg_full, phase, optimize="optimal")
        for ir, r in enumerate(rkeys):
            irm = r_index[tuple(-x for x in r)]
            for i in sites:
                dense_i = dense_ops[i]
                for j in sites:
                    dense_j = dense_ops[j]
                    g0ij = spinor_dense_block(
                        green.get_site_block_spinor(gr0[ir], i, j)
                    )
                    g0ji = spinor_dense_block(
                        green.get_site_block_spinor(gr0[irm], j, i)
                    )
                    dgij = spinor_dense_block(
                        green.get_site_block_spinor(dgr[ir], i, j)
                    )
                    dgji = spinor_dense_block(
                        green.get_site_block_spinor(dgr[irm], j, i)
                    )
                    key = (r, i, j)
                    k0[key][ie] = spinor_tangent_pair_matrix(
                        vertices[i], g0ij, vertices[j], g0ji
                    )
                    dk[key][ie] = spinor_tangent_pair_matrix(
                        vertices[i], dgij, vertices[j], g0ji
                    ) + spinor_tangent_pair_matrix(vertices[i], g0ij, vertices[j], dgji)
                    # topology pair traces (the story-001 object): dense
                    # Tr[v_a dg_ij v_b g0_ji] + Tr[v_a g0_ij v_b dg_ji] and
                    # its strength-0 partner, on the FULL magnetic vertex.
                    tr[key][ie] = np.trace(dense_i @ dgij @ dense_j @ g0ji) + np.trace(
                        dense_i @ g0ij @ dense_j @ dgji
                    )
                    tr0[key][ie] = np.trace(dense_i @ g0ij @ dense_j @ g0ji)
    return {
        "energies": energies,
        "K0": k0,
        "dK": dk,
        "pair_trace0": tr0,
        "pair_trace": tr,
    }


# ---------------------------------------------------------------------------
# magnetic reference frame (one leg)
# ---------------------------------------------------------------------------


def _leg_frame_from_vertices(vert_info, sites, align_tol=_AXIS_ALIGN_TOL):
    """Right-handed (u, v, n) triad shared by all magnetic sites of a leg.

    Every site's measured splitting direction must be parallel to a common
    Cartesian axis (parallel OR antiparallel — site signs are recorded in
    ``site_n`` and never enter the exchange); the leg axis itself is the
    POSITIVE Cartesian unit vector ``e_n``.
    """
    n0 = np.asarray(vert_info[int(sites[0])]["n"], dtype=float)
    for site in sites:
        n_site = np.asarray(vert_info[int(site)]["n"], dtype=float)
        if abs(abs(float(np.dot(n0, n_site))) - 1.0) > align_tol:
            raise ValueError(
                "magnetic splitting directions are not collinear across sites "
                f"({site}: {n_site.tolist()} vs {n0.tolist()}): the tangent "
                "leg frame requires a common reference axis"
            )
    n_axis = int(np.argmax(np.abs(n0)))
    if abs(abs(n0[n_axis]) - 1.0) > align_tol:
        raise ValueError(
            "the magnetic reference axis is not aligned with a Cartesian axis; "
            "the transverse-leg merge requires x/y/z references "
            f"(got {n0.tolist()})"
        )
    u_axis, v_axis = TRANSVERSE_AXES[n_axis]
    return {
        "n": n_axis,
        "u": u_axis,
        "v": v_axis,
        "axis": _CARTESIAN_AXES[n_axis].copy(),
        "triad": (u_axis, v_axis, n_axis),
        "site_n": {
            int(site): np.asarray(vert_info[int(site)]["n"], float) for site in sites
        },
        "normalization": TANGENT_NORMALIZATION,
    }


# ---------------------------------------------------------------------------
# production driver
# ---------------------------------------------------------------------------


def _integrate_k_matrices(vals, contour):
    arr = np.asarray(vals, dtype=complex)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"expected (nE, 3, 3) tangent matrices, got {arr.shape}")
    out = np.empty((3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            out[a, b] = contour.integrate_values(arr[:, a, b])
    return out


def _integrate_traces(vals, contour):
    return contour.integrate_values(np.asarray(vals, dtype=complex))


def compute_ks_split_soc_exchange(
    data,
    w_soc,
    lam=1.0,
    mode=MODE_SECOND_VARIATION,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    sites=None,
    overlap_mode=None,
    overlap_rcond=None,
    metadata=None,
):
    """KS-band split-SOC exchange tensor per (R, i, j) (ADR-1).

    Parameters
    ----------
    data : ProjectorGreenData
        Normalized no-SOC spinful band data (``nspinor=2``): eigenvalues
        ``E(k)``, all-atom rectangular projector maps ``B_a(k)`` (the
        coefficient array), and magnetic rotation vertices ``v_a`` (the
        spinor operator).  No local Hamiltonian is required.
    w_soc : (nkpt, N, N) array_like
        All-atom SOC operator in the band window, ``W^K_SO_nm(k)`` (eV),
        Hermitian at every k.  Complex Hermitian is fully supported.
    lam : float
        Dimensionless SOC scaling.  Only meaningful for the
        second-variation mode; the insertion mode *returns the
        lam-derivative at the strength-0 reference* and rejects any
        other value.
    mode : str
        ``"second_variation"`` (production: diagonalize ``E + lam W``,
        rotate the spectral coefficients, replay the existing spinor
        kernel — all orders of lam in-window) or
        ``"first_order_insertion"`` (diagnostic: analytic ``G0 W G0``
        insertion with both trace-line topologies; outputs are lam
        *derivatives* of the tensors and of the channel matrix
        ``A_ijR``).
    Rpts : (nR, 3) array_like of int, optional
        Lattice vectors; each negative R must be present.  Defaults to
        the same ``_R_grid(nmax=1)`` full grid as the existing spinor
        kernel.  ``kpoints`` must form a uniform full-BZ grid (the
        R-Fourier in ``get_GR_spinor`` is only meaningful on a periodic
        mesh) and ``Rpts`` should cover every R reachable within the
        cutoff: a partial R sum is a Dirichlet-windowed q-average, not
        the q=0 response.  Self-pairs ``(R, i, i)`` are on-site
        curvature entries; inter-site exchange is the ``(R, i, j)``,
        ``i != j`` block.
    sites : list[int]
        Magnetic sites carrying vertices (magnetic-only gating); default
        all sites.
    metadata : dict, optional
        Backend provenance merged into the FR-050 block
        (``strength0_reference``, ``soc_operator_source``, ``frame``,
        ``merge_mode``, plus free-form top-level fields).

    Returns
    -------
    dict with ``"exchange"``: ``{(R, i, j): {"J_leg", "frame", "K_ijR",
    "mask_residual", ...}}`` — the MEASURED transverse 2x2 only, zero-
    masked on the leg's ``n`` row/column, with the explicit right-handed
    ``(u, v, n)`` frame (``frame["axis"]`` the positive Cartesian unit
    vector, ``frame["site_n"]`` the signed per-site splitting directions) —
    and ``"metadata"``: FR-050 provenance block.  A single reference never
    determines Jiso/DMI/Jani; the full lattice tensor requires
    :func:`merge_transverse_legs` over the three x/y/z references.  The
    insertion mode additionally carries the dense ``"pair_trace"`` (lam
    derivative of ``Tr[v_a G v_b G]``, the story-001 topology object); its
    ``J_leg`` entries are lam *derivatives* (``metadata["quantity"]`` says
    so).
    """
    from ase.units import kB

    from TB2J.mycfr import CFR

    _validate_kernel_data(data)
    w = _validated_w_soc(data, w_soc)
    if mode not in SPLIT_SOC_MODES:
        raise ValueError(
            f"unsupported split-SOC kernel mode: {mode!r} (expected one of "
            f"{SPLIT_SOC_MODES})"
        )
    if mode == MODE_FIRST_ORDER_INSERTION and float(lam) != 1.0:
        raise ValueError(
            "the first-order insertion mode returns the lam-derivative of the "
            "exchange at the strength-0 reference; it takes no SOC scaling "
            f"(lam must be 1.0, got {lam})"
        )
    sites = _validated_sites(data, sites)
    rpts, rkeys = _validated_rpts(Rpts)

    contour = CFR(nz=nz, T=smearing_eV / kB)
    green0 = ProjectorGreen(
        data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond
    )
    ops = green0.get_local_operators_spinor(sites=sites)
    vert_info = {site: magnetic_tangent_vertices(op) for site, op in ops.items()}
    vertices = {site: info["vertices"] for site, info in vert_info.items()}
    frame = _leg_frame_from_vertices(vert_info, sites)

    if mode == MODE_SECOND_VARIATION:
        eps2, x = second_variation_spectrum(data.eigenvalues, w, lam=lam)
        rotated = dataclass_replace(
            data,
            eigenvalues=eps2[None, ...],
            coefficients=rotate_spinor_coefficients(data.coefficients, x),
            metadata={
                **data.metadata,
                "split_soc": {
                    "second_variation": True,
                    "lambda": float(lam),
                    "unit_strength_reference": "lam=0 replay of this data",
                },
            },
        )
        green = ProjectorGreen(
            rotated, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond
        )
        values = {
            key: [] for key in ((r, i, j) for r in rkeys for i in sites for j in sites)
        }
        for energy in contour.path:
            trace = spinor_tangent_trace(
                green, rpts, energy=energy, vertices=vertices, sites=sites
            )
            for key in values:
                values[key].append(trace["K_ijR"][key])
    else:
        insertion = first_order_insertion_channels(
            data, w, Rpts=rpts, energies=contour.path, sites=sites
        )
        values = {key: np.asarray(k) for key, k in insertion["dK"].items()}

    # contour-integrate each pair's tangent matrix exactly once
    integrated_k = {
        key: _integrate_k_matrices(vals, contour) for key, vals in values.items()
    }
    insertion_traces = (
        {
            key: _integrate_traces(trace, contour)
            for key, trace in insertion["pair_trace"].items()
        }
        if mode == MODE_FIRST_ORDER_INSERTION
        else {}
    )

    # one reference determines ONLY the transverse 2x2 of its (u, v, n)
    # triad: mask the n row/column and never emit Jiso/DMI/Jani here
    exchange = {}
    n_axis = int(frame["n"])
    for key, integrated in integrated_k.items():
        j_leg = np.imag(integrated) / (2.0 * np.pi)
        mask_residual = float(
            max(
                np.abs(j_leg[n_axis, :]).max(initial=0.0),
                np.abs(j_leg[:, n_axis]).max(initial=0.0),
            )
        )
        j_leg[n_axis, :] = 0.0
        j_leg[:, n_axis] = 0.0
        entry = {
            "J_leg": j_leg,
            "frame": frame,
            "K_ijR": integrated,
            "mask_residual": mask_residual,
        }
        if mode == MODE_FIRST_ORDER_INSERTION:
            # story-001 topology object: d/dlam Tr[v_a G v_b G]|_0 as the
            # dense trace of both insertion lines (see
            # first_order_insertion_channels); not recoverable from the
            # tangent matrix, which uses the rotated tangent vertices.
            entry["pair_trace"] = complex(insertion_traces[key])
        exchange[key] = entry

    extra = dict(metadata or {})
    strength0 = {
        "data_schema": f"{data.schema_name}/{data.schema_version}",
        "coefficient_source": data.coefficient_source or "unspecified",
        "description": "normalized no-SOC spinor band window (ProjectorGreenData nspinor=2)",
    }
    strength0.update(extra.pop("strength0_reference", {}) or {})
    soc_operator = {
        "source": extra.pop("soc_operator_source", None)
        or "argument w_soc (all-atom KS-band matrix elements)",
    }
    frame = extra.pop("frame", None)
    merge_mode = extra.pop("merge_mode", None)
    provenance = split_soc_provenance(
        mode=mode,
        lam=lam,
        nband=data.nband,
        soc_operator=soc_operator,
        strength0_reference=strength0,
        frame=frame,
        vertices_sites=sites,
        merge_mode=merge_mode,
        extra=extra or None,
    )
    return {"exchange": exchange, "metadata": provenance}


# ---------------------------------------------------------------------------
# three-leg raw-tensor merge (rank-9)
# ---------------------------------------------------------------------------


def rotate_transverse_leg(exchange, rotation, axis):
    """Map a psi-gauge z-leg's measured 2x2 block to a lattice axis.

    A change of spin coordinates rotates the full raw matrix O J O^T;
    unmeasured longitudinal entries stay masked, never treated as zeros in
    the eventual rank-nine solve. The original per-site magnetic signs do
    not multiply the tangent vertices a second time.
    """
    o = np.asarray(rotation, dtype=float)
    n = int(axis)
    if o.shape != (3, 3) or n not in (0, 1, 2):
        raise ValueError("rotation must be 3x3 and axis must be x/y/z index")
    if not np.allclose(o @ o.T, np.eye(3), atol=1e-8) or not np.allclose(
        o[:, 2], np.eye(3)[n], atol=1e-8
    ):
        raise ValueError(
            "leg frame must be a proper rotation taking z to the lattice axis"
        )
    rotated = {}
    for key, entry in exchange.items():
        if int(entry["frame"]["n"]) != 2:
            raise ValueError("psi-gauge input leg must have a z magnetic reference")
        measured = o @ np.asarray(entry["J_leg"], dtype=float) @ o.T
        residual = max(np.max(np.abs(measured[n, :])), np.max(np.abs(measured[:, n])))
        measured[n, :] = 0.0
        measured[:, n] = 0.0
        rotated[key] = {
            "J_leg": measured,
            "frame": {"n": n, "axis": np.eye(3)[n], "rotation": o.tolist()},
            "mask_residual": max(float(entry["mask_residual"]), float(residual)),
        }
    return rotated


def merge_transverse_legs(legs, consistency_atol=1.0e-8):
    """Solve the raw 3x3 lattice exchange tensor from transverse leg blocks.

    ``legs`` maps a leg name to ``{"exchange": {(R, i, j): entry}}`` with
    entries as produced by :func:`compute_ks_split_soc_exchange`, already
    rotated to the LATTICE frame (``J_leg -> O J_leg O^T`` by the caller
    when the kernel ran in a rotated spin frame).  All legs must carry the
    identical ``(R, i, j)`` key set.

    One magnetic reference determines only the transverse 2x2 block of its
    right-handed ``(u, v, n)`` triad; the x/y/z references together give 12
    constraints for the 9 entries of the real lattice tensor (each diagonal
    measured twice) with design-matrix rank 9 — the solve is an exact
    least squares on a consistent system.  :func:`io_merge.merge` is
    deliberately NOT used: its independently averaged scalar/traceless
    decomposition biases anisotropy (a true diag(1, 2, 3) target returns
    diag(7/6, 2, 17/6) through that route).  The decomposition here is the
    TB2J :func:`Jtensor.decompose_J_tensor` (Levi-Civita DMI convention)
    applied to the raw solved tensor.

    The repeated-diagonal gate (``consistency_atol``) is a physics gate,
    not a data-integrity check: at finite SOC strength the three leg
    references are perturbed about non-stationary O(lam^2)-split states,
    so an exactly-specified full-strength fixture can legitimately
    refuse a tight tolerance (e.g. a few 1e-4 eV spread at lam=1).  A
    refusal at small lam instead indicates the references do not share
    one strength-0 state - chase the producer, not the merge.

    Returns ``{"exchange": {(R, i, j): {"tensor", "Jiso", "dmi", "jani",
    "diagnostics"}}, "diagnostics": {...worst values...}}``.
    """
    if not legs:
        raise ValueError("merge_transverse_legs requires at least one leg")
    exchanges = {}
    for name, leg in legs.items():
        exc = leg.get("exchange") if isinstance(leg, dict) else None
        if not isinstance(exc, dict) or not exc:
            raise ValueError(f"leg {name!r} must carry a non-empty 'exchange' mapping")
        exchanges[name] = exc
    names = list(exchanges)
    reference_keys = set(exchanges[names[0]])
    for name in names[1:]:
        keys = set(exchanges[name])
        missing = reference_keys ^ keys
        if missing:
            raise ValueError(
                f"leg {name!r} does not share the (R, i, j) key set of leg "
                f"{names[0]!r}; first mismatch: {sorted(missing)[0]}"
            )
    keys = sorted(reference_keys)

    tensors = {}
    diagnostics = {}
    for key in keys:
        rows = []
        values = []
        diag_seen = {}
        mask_residual = 0.0
        for name in names:
            entry = exchanges[name][key]
            frame = entry["frame"]
            n = int(frame["n"])
            j_leg = np.asarray(entry["J_leg"], dtype=float)
            if j_leg.shape != (3, 3):
                raise ValueError(f"J_leg must be (3, 3), got {j_leg.shape} in {name}")
            mask_residual = max(mask_residual, float(entry.get("mask_residual", 0.0)))
            for a in range(3):
                if a == n:
                    continue
                for b in range(3):
                    if b == n:
                        continue
                    row = np.zeros(9)
                    row[3 * a + b] = 1.0
                    rows.append(row)
                    values.append(float(j_leg[a, b]))
                    if a == b:
                        diag_seen.setdefault(a, []).append(float(j_leg[a, a]))
        design = np.asarray(rows)
        target = np.asarray(values)
        if np.linalg.matrix_rank(design) != 9:
            raise ValueError(
                f"transverse legs cannot determine the full tensor for {key}: "
                "three independent x/y/z reference axes are required"
            )
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        tensor = solution.reshape(3, 3)
        repeat_dev = 0.0
        for entries in diag_seen.values():
            if len(entries) > 1:
                repeat_dev = max(repeat_dev, max(entries) - min(entries))
        if repeat_dev > consistency_atol:
            raise ValueError(
                f"leg diagonals disagree for pair {key}: max spread "
                f"{repeat_dev:.3e} exceeds consistency_atol {consistency_atol:.1e}"
            )
        tensors[key] = tensor
        diagnostics[key] = {
            "rank": int(np.linalg.matrix_rank(design)),
            "max_repeat_deviation": float(repeat_dev),
            "max_transverse_mask_residual": float(mask_residual),
            "reciprocity_residual": None,
        }

    # reciprocity J_ij(R) = J_ji(-R)^T holds exactly before integration
    # (cyclicity of the tangent trace); report the worst numerical residual
    for key, tensor in tensors.items():
        r, i, j = key
        rev = (tuple(-x for x in r), j, i)
        if rev in tensors and diagnostics[key]["reciprocity_residual"] is None:
            residual = float(np.abs(tensor - tensors[rev].T).max())
            diagnostics[key]["reciprocity_residual"] = residual
            diagnostics[rev]["reciprocity_residual"] = residual

    exchange = {}
    for key, tensor in tensors.items():
        jiso, dmi, jani = decompose_J_tensor(tensor)
        exchange[key] = {
            "tensor": tensor,
            "Jiso": float(jiso),
            "dmi": np.asarray(dmi, dtype=float),
            "jani": np.asarray(jani, dtype=float),
            "diagnostics": diagnostics[key],
        }
    worst = {
        "min_rank": min(d["rank"] for d in diagnostics.values()),
        "max_repeat_deviation": max(
            d["max_repeat_deviation"] for d in diagnostics.values()
        ),
        "max_transverse_mask_residual": max(
            d["max_transverse_mask_residual"] for d in diagnostics.values()
        ),
        "max_reciprocity_residual": max(
            d["reciprocity_residual"]
            for d in diagnostics.values()
            if d["reciprocity_residual"] is not None
        )
        if any(d["reciprocity_residual"] is not None for d in diagnostics.values())
        else None,
        "legs": list(names),
    }
    return {"exchange": exchange, "diagnostics": worst}


# ---------------------------------------------------------------------------
# magnetic frame / axis validation
# ---------------------------------------------------------------------------


def su2_axis_rotation(axis):
    """SU(2) representative of the spin rotation taking ``e_z`` to ``axis``.

    Returns the 2x2 unitary ``U`` with ``U sigma_z U^dag = (axis/|axis|).sigma``
    (asserted internally).  Adapters use this SAME matrix to build rotated
    legs (rotate the full no-SOC spinor H/G/projections AND the magnetic
    vertex, leaving the lattice-frame SOC operator untouched), so
    :func:`spinor_frame_rotation_residual` can verify proper rotation.
    """
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        raise ValueError("axis must be a nonzero vector")
    n = axis / norm
    sigma_n = np.einsum("kab,k->ab", PAULI_MATRICES, n)
    if abs(n[2]) > 1.0 - 1.0e-12:
        u = (
            np.eye(2, dtype=complex)
            if n[2] > 0
            else np.array([[0, -1], [1, 0]], dtype=complex)
        )
        if not np.allclose(u @ PAULI_MATRICES[2] @ u.conj().T, sigma_n, atol=1.0e-10):
            raise RuntimeError("SU(2) axis rotation self-check failed")
        return u
    w = np.cross(_CARTESIAN_AXES[2], n)
    w /= np.linalg.norm(w)
    theta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    sigma_w = np.einsum("kab,k->ab", PAULI_MATRICES, w)
    for sign in (1.0, -1.0):
        u = (
            np.cos(theta / 2) * np.eye(2, dtype=complex)
            - 1j * sign * np.sin(theta / 2) * sigma_w
        )
        if np.allclose(u @ PAULI_MATRICES[2] @ u.conj().T, sigma_n, atol=1.0e-10):
            return u
    raise RuntimeError("SU(2) axis rotation self-check failed")


def split_soc_frame_report(data, axis, tol=1.0e-6, reference_data=None, sites=None):
    """Validate one leg's magnetic reference frame (report, never mutate).

    Checks, per leg:
    1. ``axis`` is a unit vector;
    2. every site's measured splitting direction (from
       ``data.spinor_operator``) is parallel to ``axis`` — parallel OR
       antiparallel, with the per-site sign recorded (metadata-only
       ``spinaxis`` edits that leave the vertex untouched FAIL here);
    3. when ``reference_data`` (another leg built from the same
       scalar-relativistic reference) is supplied, the band energies must
       be identical per k point: a genuine global SU(2) rotation of the
       no-SOC spinor Hamiltonian leaves the spectrum unchanged, while a
       leg whose *Hamiltonian* was not rotated (only relabelled) generally
       fails either this or the deep
       :func:`spinor_frame_rotation_residual` G-covariance check.

    Returns ``{"ok", "axis_norm", "site_cosine", "site_sign",
    "eigenvalue_residual"}``.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-12:
        raise ValueError("axis must be a nonzero vector")
    axis_unit = axis / axis_norm
    if getattr(data, "nspinor", 1) != 2 or data.spinor_operator is None:
        raise ValueError(
            "frame validation requires nspinor=2 data with spinor_operator"
        )
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    site_cosine = {}
    site_sign = {}
    for site in sites:
        info = magnetic_tangent_vertices(data.spinor_operator[int(site)])
        cosine = float(np.dot(info["n"], axis_unit))
        site_cosine[int(site)] = cosine
        site_sign[int(site)] = 1.0 if cosine >= 0 else -1.0
    worst = max(abs(c) for c in site_cosine.values())
    eigenvalue_residual = None
    if reference_data is not None:
        ref = np.asarray(reference_data.eigenvalues, dtype=float)
        leg = np.asarray(data.eigenvalues, dtype=float)
        if ref.shape != leg.shape:
            raise ValueError(
                f"reference eigenvalue shape {ref.shape} != leg shape {leg.shape}"
            )
        eigenvalue_residual = float(np.abs(leg - ref).max())
    return {
        "ok": bool((1.0 - worst) <= tol)
        and (eigenvalue_residual is None or eigenvalue_residual <= tol),
        "axis_norm": axis_norm,
        "site_cosine": site_cosine,
        "site_sign": site_sign,
        "eigenvalue_residual": eigenvalue_residual,
    }


def spinor_frame_rotation_residual(
    data, reference_data, axis, energy=0.0, rpts=((0, 0, 0),), sites=None, rotation=None
):
    """Deep G-covariance proof that one leg is a rotated reference.

    With ``U`` from :func:`su2_axis_rotation`, a leg generated by a global
    SU(2) rotation of the *entire* no-SOC spinor problem satisfies, at
    every z and R::

        G_leg(R, z) = (kron(U, I) ) G_ref(R, z) (kron(U, I))^dag,
        M_leg(site) = (kron(U, I)) M_ref(site) (kron(U, I))^dag.

    Rotating only the SOC operator, or only relabelling ``spinaxis``
    metadata without rotating H/G/projections and the vertex, produces a
    large residual.  Returns ``{"max_g_residual", "max_vertex_residual",
    "energy", "axis"}``.
    """
    if getattr(data, "nspinor", 1) != 2 or getattr(reference_data, "nspinor", 1) != 2:
        raise ValueError("rotation residual requires nspinor=2 data")
    if data.coefficients.shape[-1] != reference_data.coefficients.shape[-1]:
        raise ValueError("both legs must share the same projector basis size")
    u = (
        su2_axis_rotation(axis)
        if rotation is None
        else np.asarray(rotation, dtype=complex)
    )
    if u.shape != (2, 2) or not np.allclose(u.conj().T @ u, np.eye(2), atol=1e-10):
        raise ValueError("spinor frame rotation must be a 2x2 unitary")
    rotated_axis = np.array(
        [
            0.5 * np.trace(s @ u @ PAULI_MATRICES[2] @ u.conj().T).real
            for s in PAULI_MATRICES
        ]
    )
    if not np.allclose(
        rotated_axis, np.asarray(axis) / np.linalg.norm(axis), atol=1e-10
    ):
        raise ValueError("spinor frame rotation does not map z to the requested axis")
    green_leg = ProjectorGreen(data)
    green_ref = ProjectorGreen(reference_data)
    if green_leg.nbasis != green_ref.nbasis:
        raise ValueError("both legs must share the same projector basis size")
    u_full = np.kron(u, np.eye(green_leg.nbasis))
    rpts_arr = np.asarray(rpts, dtype=int)
    keys = [tuple(int(x) for x in r) for r in rpts_arr]
    have = set(keys)
    missing = [r for r in keys if tuple(-x for x in r) not in have]
    if missing:
        raise ValueError(f"rpts must include each negative R vector: {missing[0]}")
    gr_leg = green_leg.get_GR_spinor(rpts_arr, energy)
    gr_ref = green_ref.get_GR_spinor(rpts_arr, energy)
    g_residual = 0.0
    for ir in range(len(keys)):
        dense_leg = spinor_dense_block(gr_leg[ir])
        dense_ref = spinor_dense_block(gr_ref[ir])
        scale = max(1.0, float(np.abs(dense_leg).max(initial=0.0)))
        rotated = u_full @ dense_ref @ u_full.conj().T
        g_residual = max(g_residual, float(np.abs(dense_leg - rotated).max()) / scale)
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    vertex_residual = 0.0
    for site in sites:
        site = int(site)
        m_leg = spinor_dense_block(
            green_leg.get_local_operators_spinor(sites=[site])[site]
        )
        m_ref = spinor_dense_block(
            green_ref.get_local_operators_spinor(sites=[site])[site]
        )
        scale = max(1.0, float(np.abs(m_leg).max(initial=0.0)))
        u_site = np.kron(u, np.eye(int(data.site_nproj[site])))
        rotated = u_site @ m_ref @ u_site.conj().T
        vertex_residual = max(
            vertex_residual, float(np.abs(m_leg - rotated).max()) / scale
        )
    return {
        "max_g_residual": float(g_residual),
        "max_vertex_residual": float(vertex_residual),
        "energy": float(energy),
        "axis": np.asarray(axis, dtype=float) / float(np.linalg.norm(axis)),
    }


# ---------------------------------------------------------------------------
# band-window convergence study
# ---------------------------------------------------------------------------


def band_window_convergence_report(
    data,
    w_soc,
    windows,
    pair,
    lam=1.0,
    mode=MODE_SECOND_VARIATION,
    nz=24,
    smearing_eV=0.05,
    Rpts=None,
    tol=1.0e-6,
    overlap_mode=None,
    overlap_rcond=None,
    sites=None,
):
    """Window-enlargement study for one ordered site pair (FR-050).

    Runs :func:`compute_ks_split_soc_exchange` on growing prefix windows
    (first ``nw`` band-window states per k) and reports per-window
    MEASURED transverse observables for the pair (the ``J_uu/J_uv/J_vu/
    J_vv`` entries of the leg's right-handed (u, v, n) triad and their
    2-norm, keyed by R), the relative change between consecutive windows,
    and convergence flags.  No Jiso/DMI/Jani is reported from a single
    reference — only the measured transverse 2x2 exists per leg.
    Truncating the window removes both SOC couplings to omitted bands and
    magnetic-vertex transitions through them (Feshbach O(lam^2) leakage
    plus vertex truncation), so the report must accompany exchange outputs
    (ADR-8).

    Returns ``{"windows": [{"nband", "J_uu", "J_uv", "J_vu", "J_vv",
    "transverse_norm"}, ...], "changes": [...], "tol", "converged",
    "converged_from"}`` where ``converged_from`` is the first window size
    from which all later changes stay within ``tol`` (``None`` if not
    converged).
    """
    windows = sorted(int(nw) for nw in windows)
    if not windows or windows[0] < 1 or windows[-1] > data.nband:
        raise ValueError(
            f"windows must be a non-empty increasing list within "
            f"[1, {data.nband}]: {windows}"
        )
    i, j = (int(pair[0]), int(pair[1]))
    rpts, rkeys = _validated_rpts(Rpts)
    site_list = sites if sites is not None else [i, j]
    if i not in site_list or j not in site_list:
        site_list = sorted(set(site_list) | {i, j})

    windows_out = []
    for nw in windows:
        sub = dataclass_replace(
            data,
            eigenvalues=data.eigenvalues[:, :, :nw],
            coefficients=data.coefficients[:, :, :nw],
            occupations=None,
        )
        res = compute_ks_split_soc_exchange(
            sub,
            np.asarray(w_soc, dtype=complex)[:, :nw, :nw],
            lam=lam,
            mode=mode,
            Rpts=rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=site_list,
            overlap_mode=overlap_mode,
            overlap_rcond=overlap_rcond,
        )
        j_uu = {}
        j_uv = {}
        j_vu = {}
        j_vv = {}
        transverse_norm = {}
        for key, entry in res["exchange"].items():
            r = key[0]
            if key[1] == i and key[2] == j:
                frame = entry["frame"]
                u, v = int(frame["u"]), int(frame["v"])
                j_leg = np.asarray(entry["J_leg"], dtype=float)
                j_uu[r] = float(j_leg[u, u])
                j_uv[r] = float(j_leg[u, v])
                j_vu[r] = float(j_leg[v, u])
                j_vv[r] = float(j_leg[v, v])
                transverse_norm[r] = float(
                    np.linalg.norm([j_leg[u, u], j_leg[u, v], j_leg[v, u], j_leg[v, v]])
                )
        windows_out.append(
            {
                "nband": nw,
                "J_uu": j_uu,
                "J_uv": j_uv,
                "J_vu": j_vu,
                "J_vv": j_vv,
                "transverse_norm": transverse_norm,
            }
        )

    def _values(window):
        return [
            window["J_uu"],
            window["J_uv"],
            window["J_vu"],
            window["J_vv"],
            window["transverse_norm"],
        ]

    changes = []
    for prev, cur in zip(windows_out[:-1], windows_out[1:]):
        worst = 0.0
        for vp, vc in zip(_values(prev), _values(cur)):
            for r in vc:
                scale = max(1.0, abs(vc[r]), abs(vp[r]))
                worst = max(worst, abs(vc[r] - vp[r]) / scale)
        changes.append(float(worst))
    converged = bool(changes) and all(c <= tol for c in changes)
    converged_from = None
    if converged:
        converged_from = windows_out[0]["nband"]
        for idx, change in enumerate(changes):
            if change > tol:
                converged_from = windows_out[idx + 1]["nband"]
    return {
        "windows": windows_out,
        "changes": changes,
        "tol": float(tol),
        "converged": converged,
        "converged_from": converged_from,
        "pair": (i, j),
        "mode": mode,
        "lam": float(lam),
    }
