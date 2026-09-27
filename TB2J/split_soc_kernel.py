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

from dataclasses import replace as dataclass_replace

import numpy as np

from TB2J.projector_green import (
    ProjectorGreen,
    _spinor_dense_block,
    site_magnetization_sign,
    spinor_channels_to_exchange_tensor,
    spinor_pair_channels,
    spinor_projector_exchange_trace,
)

MODE_SECOND_VARIATION = "second_variation"
MODE_FIRST_ORDER_INSERTION = "first_order_insertion"
SPLIT_SOC_MODES = (MODE_SECOND_VARIATION, MODE_FIRST_ORDER_INSERTION)

PROVENANCE_SCHEMA = "tb2j.split_soc_ks_provenance/1.0"
SOC_UNITS = "eV"
PAULI_ORDER = "x,y,z"

_HERMITIAN_TOL = 1.0e-8


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
    arr = (
        np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
        if Rpts is None
        else np.asarray(Rpts, dtype=int)
    )
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


def first_order_insertion_channels(data, w_soc, Rpts, energies, lam=1.0, sites=None):
    """Analytic first-order SOC insertion at explicit complex energies.

    Builds ``G0(z,k)`` from the strength-0 eigenvalues and the insertion
    ``dG(k) = G0(k) W(k) G0(k)`` in the band window, contracts both into
    the projector blocks through the rectangular maps, restores the
    full-BZ Fourier phases per inter-site ``R`` (same ``exp(-2 pi i k.R)``
    convention and k weights as the production replay), and evaluates the
    two trace-line topologies per (R, i, j)::

        d/dlam Tr[V_a G V_b G]|_0
            = Tr[v_a dG_ij v_b G0_ji] + Tr[v_a G0_ij v_b dG_ji]

    through the shared :func:`spinor_pair_channels` kernel.

    Returns ``{"energies", "A0", "dA", "pair_trace0", "pair_trace"}`` with
    ``(nE, 4, 4)`` channel arrays and ``(nE,)`` pair traces — the dense
    trace-line objects ``Tr[v_a dg_ij v_b g0_ji] + Tr[v_a g0_ij v_b dg_ji]``
    (and their strength-0 partner) evaluated per energy.  This is the
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
    phase = np.exp(green.k2Rfactor * np.einsum("ri,ki->rk", rpts, green.kpts))
    phase = phase * green.kweights[None, :]

    pairs = [(r, i, j) for r in rkeys for i in sites for j in sites]
    a0 = {key: np.empty((energies.size, 4, 4), dtype=complex) for key in pairs}
    da = {key: np.empty((energies.size, 4, 4), dtype=complex) for key in pairs}
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
                for j in sites:
                    g0ij = green.get_site_block_spinor(gr0[ir], i, j)
                    g0ji = green.get_site_block_spinor(gr0[irm], j, i)
                    dgij = green.get_site_block_spinor(dgr[ir], i, j)
                    dgji = green.get_site_block_spinor(dgr[irm], j, i)
                    key = (r, i, j)
                    a0[key][ie] = spinor_pair_channels(ops[i], g0ij, ops[j], g0ji)
                    da[key][ie] = spinor_pair_channels(
                        ops[i], dgij, ops[j], g0ji
                    ) + spinor_pair_channels(ops[i], g0ij, ops[j], dgji)
                    # topology pair traces (the story-001 object): dense
                    # Tr[v_a dg_ij v_b g0_ji] + Tr[v_a g0_ij v_b dg_ji] and
                    # its strength-0 partner.  Note pi*sum(A^{uv}) is NOT
                    # this trace: the pinned channel construction
                    # reconstructs the spin-transposed Green block.
                    dense_i = _spinor_dense_block(ops[i])
                    dense_j = _spinor_dense_block(ops[j])
                    tr[key][ie] = np.trace(
                        dense_i
                        @ _spinor_dense_block(dgij)
                        @ dense_j
                        @ _spinor_dense_block(g0ji)
                    ) + np.trace(
                        dense_i
                        @ _spinor_dense_block(g0ij)
                        @ dense_j
                        @ _spinor_dense_block(dgji)
                    )
                    tr0[key][ie] = np.trace(
                        dense_i
                        @ _spinor_dense_block(g0ij)
                        @ dense_j
                        @ _spinor_dense_block(g0ji)
                    )
    pair0 = tr0
    pair = tr
    return {
        "energies": energies,
        "A0": a0,
        "dA": da,
        "pair_trace0": pair0,
        "pair_trace": pair,
    }


# ---------------------------------------------------------------------------
# production driver
# ---------------------------------------------------------------------------


def _integrate_channels(vals, contour):
    arr = np.asarray(vals, dtype=complex)
    out = np.empty((4, 4), dtype=complex)
    for a in range(4):
        for b in range(4):
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
        Dimensionless SOC scaling.
    mode : str
        ``"second_variation"`` (production: diagonalize ``E + lam W``,
        rotate the spectral coefficients, replay the existing spinor
        kernel — all orders of lam in-window) or
        ``"first_order_insertion"`` (diagnostic: analytic ``G0 W G0``
        insertion with both trace-line topologies; outputs are lam
        *derivatives* of the tensors).
    Rpts : (nR, 3) array_like of int
        Lattice vectors; each negative R must be present.
    sites : list[int]
        Magnetic sites carrying vertices (magnetic-only gating); default
        all sites.
    metadata : dict, optional
        Backend provenance merged into the FR-050 block
        (``strength0_reference``, ``soc_operator_source``, ``frame``,
        ``merge_mode``, plus free-form top-level fields).

    Returns
    -------
    dict with ``"exchange"``: ``{(R, i, j): {"Jiso", "dmi", "jani",
    "tensor", "A_ijR", ...}}`` and ``"metadata"``: FR-050 provenance
    block.  The insertion mode additionally carries the dense
    ``"pair_trace"`` (lam derivative of ``Tr[V_a G V_b G]``, the
    story-001 topology object); its exchange entries are lam
    *derivatives* (``metadata["quantity"]`` says so).
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
    sites = _validated_sites(data, sites)
    rpts, rkeys = _validated_rpts(Rpts)

    contour = CFR(nz=nz, T=smearing_eV / kB)
    green0 = ProjectorGreen(
        data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond
    )
    ops = green0.get_local_operators_spinor(sites=sites)

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
            trace = spinor_projector_exchange_trace(
                green, rpts, energy=energy, local_operators=ops, sites=sites
            )
            for key in values:
                values[key].append(trace["A_ijR"][key])
    else:
        insertion = first_order_insertion_channels(
            data, w, Rpts=rpts, energies=contour.path, lam=lam, sites=sites
        )
        values = {key: list(np.asarray(a)) for key, a in insertion["dA"].items()}

    signs = {site: site_magnetization_sign(op) for site, op in ops.items()}
    exchange = {}
    for key, vals in values.items():
        r, i, j = key
        integrated = _integrate_channels(vals, contour)
        integrated_rev = _integrate_channels(
            values[(tuple(-x for x in r), j, i)], contour
        )
        entry = spinor_channels_to_exchange_tensor(
            integrated, integrated_rev, signs[i] * signs[j]
        )
        entry["A_ijR"] = integrated
        if mode == MODE_FIRST_ORDER_INSERTION:
            # story-001 topology object: d/dlam Tr[V_a G V_b G]|_0 as the
            # dense trace of both insertion lines (see
            # first_order_insertion_channels); not recoverable from the
            # channel matrix, whose Pauli sum reconstructs the
            # spin-transposed Green block.
            entry["pair_trace"] = complex(
                _integrate_traces(insertion["pair_trace"][key], contour)
            )
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
    ``Jiso``/DMI/Jani norms per R, the relative change between consecutive
    windows, and convergence flags.  Truncating the window removes both
    SOC couplings to omitted bands and magnetic-vertex transitions
    through them (Feshbach O(lam^2) leakage plus vertex truncation), so
    the report must accompany exchange outputs (ADR-8).

    Returns ``{"windows": [{"nband", "Jiso", "dmi_norm", "jani_norm"},
    ...], "changes": [...], "tol", "converged", "converged_from"}`` where
    the per-window observables are keyed by R and ``converged_from`` is
    the first window size from which all later changes stay within
    ``tol`` (``None`` if not converged).
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
        jiso = {}
        dmi_norm = {}
        jani_norm = {}
        for key, entry in res["exchange"].items():
            r = key[0]
            if key[1] == i and key[2] == j:
                jiso[r] = float(entry["Jiso"])
                dmi_norm[r] = float(np.linalg.norm(entry["dmi"]))
                jani_norm[r] = float(np.linalg.norm(entry["jani"]))
        windows_out.append(
            {"nband": nw, "Jiso": jiso, "dmi_norm": dmi_norm, "jani_norm": jani_norm}
        )

    def _values(window):
        return [window["Jiso"], window["dmi_norm"], window["jani_norm"]]

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
