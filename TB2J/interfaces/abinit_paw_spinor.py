"""ABINIT spinor PAW channel (nspden=4 Dij + nspinor=2 WFK) — Story 011.

End-to-end spinor projector-Green exchange from ABINIT PAW outputs:

* :func:`parse_pawprt_dij_spinor` — read the four spin-component Dij blocks
  (``up-up``, ``dwn-dwn``, ``up-dwn``, ``dwn-up``) that ABINIT prints per
  atom in the pawprt ``Total pseudopotential strength Dij`` section when
  ``ndij=4`` (nspden=4, complex ``cplex_dij=2`` storage).
* :func:`pauli_delta_blocks_from_components` — assemble the Pauli splitting
  ``Delta = 2 B.sigma`` per site, dropping the scalar ``(D1+D2)/2`` part.
* :func:`build_abinit_paw_spinor_data` — build the spinor-native
  :class:`~TB2J.projector_green.ProjectorGreenData` from raw dual-projector
  cprj plus the Pauli Delta.
* :func:`gen_exchange_abinit_paw_spinor` — end-to-end entry point writing
  the noncollinear TB2J outputs (J_iso, DMI, Jani) via the ExchangeNCL
  spinor kernel.

Source-verified conventions (ABINIT sources, libpaw ``m_pawdij.F90``):

* Storage: ``paw_ij%dij(cplex_dij*qphase*lmn2_size, ndij)`` with
  ``dij(:,1)=D^{up-up}``, ``dij(:,2)=D^{dn-dn}``, ``dij(:,3)=D^{up-dn}``
  and ``dij(:,4)=D^{dn-up}=conj(dij(:,3))`` elementwise — enforced by
  construction in ``pawdijfock`` (``dijfock_vv(klmn1+1,4) = -dij_updn_i``)
  and every other Dij contributor.  For ndij=4, ``cplex_dij=2``: complex
  values interleaved real/imag along the packed lmn2 axis.
* Printing (``pawdij_print_dij`` + ``pawio_print_ij``, ``opt_sym=2``): each
  component is printed as a FULL square lmn x lmn matrix under a
  ``=== REAL PART:`` / ``=== IMAGINARY PART:`` pair of headers with
  ``(1x,f9.5)`` rows.  The lower triangle of block ``c`` is reconstructed
  from ``conj(block 7-c)``, which reproduces the same packed values, so the
  full printed matrix of every block equals the symmetric-packed complex
  Dij elementwise.  Blocks are labeled ``Atom # N - Component up-up`` etc.

Pauli packing contract (same as ``abinao.spinor_export`` for the nspden=4
V_xc grid, and the paper contract): nspden=4 components are
``(V11, V22, Re V12, Im V12)`` of the 2x2 operator, hence with
``V = v*1 + B.sigma``:

    v  = (D1 + D2) / 2           (scalar part, dropped)
    Bx = Re D3,   By = -Im D3,   Bz = (D1 - D2) / 2
    Delta = 2 * (Bx*sigma_x + By*sigma_y + Bz*sigma_z)

i.e. per projector pair ``(i, j)`` the 2x2 spin operator is

    Delta_00 = D1 - D2,   Delta_01 = 2*D3,
    Delta_10 = 2*D4,      Delta_11 = D2 - D1,

whose zz element ``Delta_00 = D_up - D_dn`` matches the collinear Delta
convention.

Coefficient convention: RAW dual-projector cprj ``<~p|psi>`` with NO
conjugation anywhere on the seam.  Conjugating the coefficients flips the
dataset's k-gauge (``conj(c)`` is the ``-k`` projection set) and the
Im-prescription exchange kernels are NOT invariant under it — the abinao
abe4152 lesson; the same defect was briefly present on this side of the
seam (TB2J commit 42f4dfc, reverted for the same reason).

Circularity pitfall (do not "verify" the gauge with collinear-derived
synthetic data): collinear-interleaved spinor fixtures agree with the
collinear kernel under EITHER coefficient convention as long as both sides
use the same one; only genuine nspinor=2 WFK data can validate the k-gauge.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from TB2J.interfaces.abinit_paw import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    _atomic_numbers_from_symbols,
    _load_paw_pseudo,
    build_abinit_paw_site_layout,
    normalize_paw_xml_mapping,
)
from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreenData,
)

__all__ = [
    "parse_pawprt_dij_spinor",
    "pauli_delta_blocks_from_components",
    "build_abinit_paw_spinor_data",
    "adapt_paw_projection_to_spinor_coefficients",
    "save_spinor_projected_data",
    "load_spinor_projected_data",
    "gen_exchange_abinit_paw_spinor",
]

_BLOCK_ORDER = ("up-up", "dwn-dwn", "up-dwn", "dwn-up")

# Header, e.g. " Total pseudopotential strength Dij (hartree):".
_HEADER_RE = re.compile(
    r"pseudopotential strength Dij\s*\((hartree|eV)\)\s*:", re.IGNORECASE
)

# ndij=4 marker: " Atom #  1 - Component up-up" (pawdij_print_dij dspin labels).
_COMPONENT_RE = re.compile(
    r"Atom\s*#\s*(\d+)\s*-\s*Component\s+(up-up|dwn-dwn|up-dwn|dwn-up)\s*$",
    re.IGNORECASE,
)

# Collinear marker present in a section we are trying to read as spinor.
_SPIN_COMPONENT_RE = re.compile(r"Atom\s*#\s*\d+\s*-\s*Spin\s+component", re.IGNORECASE)

_REAL_PART_RE = re.compile(r"===\s*REAL\s+PART", re.IGNORECASE)
_IMAG_PART_RE = re.compile(r"===\s*IMAGINARY\s+PART", re.IGNORECASE)

# Print resolution of pawio_print_ij: (1x, f9.5) in the section's unit.
_PRINT_ATOL = 5.0e-5


def _row_values(line: str) -> list[float] | None:
    """Float tokens on *line*, or ``None`` for non-data lines.

    A trailing ``"..."`` truncation sentinel (pawio_print_ij, >maxprt
    columns) is dropped so the leading sub-matrix still parses; shape
    validation downstream then rejects truncated blocks.
    """
    tokens = [t for t in line.split() if t != "..."]
    if not tokens:
        return None
    values: list[float] = []
    for tok in tokens:
        try:
            values.append(float(tok))
        except ValueError:
            return None
    return values


def _collect_rows(lines: list[str], start: int) -> tuple[list[list[float]], int]:
    """Collect consecutive numeric rows from *start*; return (rows, next_i)."""
    i = start
    n = len(lines)
    while i < n and lines[i].strip() == "":
        i += 1
    rows: list[list[float]] = []
    while i < n:
        vals = _row_values(lines[i])
        if vals is None:
            break
        rows.append(vals)
        i += 1
    return rows, i


def _square_matrix(rows: list[list[float]], what: str) -> np.ndarray:
    """Assemble a full square matrix; only the real pawio_print_ij layout."""
    if not rows:
        raise ValueError(f"no data rows found for {what}")
    lengths = {len(r) for r in rows}
    if len(lengths) != 1 or len(rows) not in lengths:
        raise ValueError(
            f"{what}: expected a full square matrix of numeric rows "
            f"(pawio_print_ij full print), got {len(rows)} rows of "
            f"lengths {sorted(lengths)}; "
            "increase maxprt/pawprtvol so the full Dij is printed"
        )
    return np.array(rows, dtype=float)


def _parse_dij_section(
    lines: list[str], start: int, collinear_seen: list[bool]
) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    """Parse one ``Total pseudopotential strength Dij`` section (ndij=4).

    Returns ``(components, next_i)`` where ``components[atom][block]`` is the
    complex lmn x lmn matrix for block in {up-up, dwn-dwn, up-dwn, dwn-up}.
    Collinear ``Spin component`` markers are skipped and reported through
    *collinear_seen* so the caller can produce an informative error when the
    log has no nspden=4 block at all.
    """
    components: dict[int, dict[str, np.ndarray]] = {}
    i = start
    n = len(lines)
    while i < n:
        if _HEADER_RE.search(lines[i]):
            break
        marker = _COMPONENT_RE.search(lines[i])
        if marker is None:
            if _SPIN_COMPONENT_RE.search(lines[i]):
                collinear_seen.append(True)
            i += 1
            continue
        atom = int(marker.group(1)) - 1
        block = marker.group(2).lower()
        i += 1

        # --- real part -----------------------------------------------------
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n or not _REAL_PART_RE.search(lines[i]):
            raise ValueError(
                f"atom {atom} block '{block}': expected '=== REAL PART:' "
                "header after the component marker"
            )
        i += 1
        real_rows, i = _collect_rows(lines, i)
        real = _square_matrix(real_rows, f"atom {atom} block '{block}' real part")

        # --- imaginary part (cplex_dij=2 always prints it) ------------------
        while i < n and lines[i].strip() == "":
            i += 1
        imag = np.zeros_like(real)
        if i < n and _IMAG_PART_RE.search(lines[i]):
            i += 1
            imag_rows, i = _collect_rows(lines, i)
            imag = _square_matrix(imag_rows, f"atom {atom} block '{block}' imag part")

        components.setdefault(atom, {})[block] = real + 1j * imag
    return components, i


def _cross_check_units(by_unit: dict[str, dict[int, dict[str, np.ndarray]]]) -> None:
    """hartree and eV sections in one log must describe the same Dij."""
    if not {"hartree", "eV"} <= set(by_unit):
        return
    ha = by_unit["hartree"]
    ev = by_unit["eV"]
    if sorted(ha) != sorted(ev):
        raise ValueError(
            "pawprt log prints inconsistent hartree/eV Dij atom sets: "
            f"{sorted(ha)} vs {sorted(ev)}"
        )
    for atom in ha:
        for block in ha[atom]:
            if ha[atom][block].shape != ev[atom][block].shape:
                raise ValueError(
                    f"atom {atom} block '{block}': hartree/eV Dij shapes differ"
                )
            delta = np.max(np.abs(ha[atom][block] * HARTREE_TO_EV - ev[atom][block]))
            if delta > 4.0 * _PRINT_ATOL * HARTREE_TO_EV:
                raise ValueError(
                    f"atom {atom} block '{block}': hartree and eV Dij sections "
                    f"disagree by {delta:.3e} eV; the log is inconsistent"
                )


def parse_pawprt_dij_spinor(
    log_path: str | Path, unit: str | None = None
) -> tuple[dict[int, dict[str, np.ndarray]], str]:
    """Parse the nspden=4 (ndij=4) pawprt Dij blocks from an ABINIT log.

    Parameters
    ----------
    log_path:
        ABINIT log/``.abo`` file containing the ``Total pseudopotential
        strength Dij`` section with four ``Component ...`` blocks per atom.
    unit:
        ``"hartree"`` or ``"eV"`` to select the printed section.  ``None``
        (default) auto-detects: the eV section is preferred when present
        (finer print resolution at the fixed f9.5 format), otherwise
        hartree.  When both sections are printed they are cross-checked.

    Returns
    -------
    (components, unit)
        ``components[atom][block]`` are complex lmn x lmn matrices for
        ``block`` in ``("up-up", "dwn-dwn", "up-dwn", "dwn-up")`` in the
        returned unit.
    """
    lines = Path(log_path).read_text().splitlines()
    by_unit: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    collinear_seen: list[bool] = []
    i = 0
    n = len(lines)
    while i < n:
        match = _HEADER_RE.search(lines[i])
        if match is None:
            i += 1
            continue
        section_unit = match.group(1).lower()
        section_unit = "eV" if section_unit == "ev" else section_unit
        components, i = _parse_dij_section(lines, i + 1, collinear_seen)
        if components:
            by_unit[section_unit] = components  # last section per unit wins
    if not by_unit:
        if collinear_seen:
            raise ValueError(
                f"no nspden=4 Dij block found in {log_path}; the log contains "
                "collinear 'Spin component' Dij blocks — use the collinear "
                "gen_exchange_abinit_paw path"
            )
        raise ValueError(
            f"no nspden=4 Dij block ('Component up-up' markers) found in {log_path}"
        )

    _cross_check_units(by_unit)

    if unit is not None:
        wanted = str(unit).strip().lower()
        if wanted.startswith(("ha", "hartree")):
            wanted = "hartree"
        elif wanted == "ev":
            wanted = "eV"
        if wanted not in by_unit:
            raise ValueError(
                f"log has no Dij section in unit {wanted!r}; found "
                f"{sorted(by_unit)}"
            )
        return by_unit[wanted], wanted

    if "eV" in by_unit:
        return by_unit["eV"], "eV"
    return by_unit["hartree"], "hartree"


# ---------------------------------------------------------------------------
# Pauli decomposition
# ---------------------------------------------------------------------------


def _require_block(atom: int, block: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in block:
        raise ValueError(
            f"atom {atom}: missing '{name}' Dij component; the nspden=4 print "
            f"must contain all of {_BLOCK_ORDER}"
        )
    return np.asarray(block[name], dtype=complex)


def pauli_delta_blocks_from_components(
    components_by_atom: Mapping[int, Mapping[str, np.ndarray]],
    unit: str,
    *,
    hermiticity_tol: float | None = None,
) -> tuple[np.ndarray, str]:
    """Assemble per-site Pauli splitting blocks from the four Dij components.

    ``components_by_atom[atom]`` maps ``("up-up", "dwn-dwn", "up-dwn",
    "dwn-up")`` to complex lmn x lmn matrices (as returned by
    :func:`parse_pawprt_dij_spinor`).  The returned blocks carry the same
    energy *unit* as the input components; convert to eV before building
    exchange data (the spinor kernel contract is eV).

    The packed m_pawdij conventions are verified at parse level:

    * ``dwn-up == conj(up-dwn)`` on the packed (upper-triangular) values —
      the elementwise conjugation every Dij contributor enforces;
    * ``up-up`` / ``dwn-dwn`` are real (Hermitian symmetric-packed storage).

    Returns ``(blocks, unit)`` with ``blocks`` of shape
    ``(natom, nproj_max, nproj_max, 2, 2)`` in units of ``Delta = 2 B.sigma``
    (scalar part dropped, identity not stored).
    """
    if not components_by_atom:
        raise ValueError("no Dij components to decompose")
    atoms = sorted(int(a) for a in components_by_atom)
    shapes = {
        tuple(np.asarray(matrix).shape)
        for atom in atoms
        for matrix in components_by_atom[atom].values()
    }
    if len(shapes) != 1:
        raise ValueError(f"inconsistent Dij block shapes across atoms: {shapes}")
    ni = shapes.pop()[0]
    nmax = ni  # per-site matrices arrive full width; padding is applied below

    tol = hermiticity_tol
    if tol is None:
        # pawio_print_ij prints (1x, f9.5) in the section's unit: per-element
        # rounding <= 5e-6, symmetrised reads and conj checks stay below 4x.
        tol = 4.0 * _PRINT_ATOL

    blocks = np.zeros((len(atoms), nmax, nmax, 2, 2), dtype=complex)
    for row, atom in enumerate(atoms):
        site = components_by_atom[atom]
        d1 = _require_block(atom, site, "up-up")
        d2 = _require_block(atom, site, "dwn-dwn")
        d3 = _require_block(atom, site, "up-dwn")
        d4 = _require_block(atom, site, "dwn-up")
        if d1.shape != (ni, ni):
            raise ValueError(f"atom {atom}: non-square Dij block {d1.shape}")

        iu = np.triu_indices(ni)
        # m_pawdij: component 4 (dn-up) = conj(component 3 (up-dn)) elementwise.
        conj_deviation = np.max(np.abs(d4[iu] - np.conj(d3[iu])))
        if conj_deviation > tol:
            raise ValueError(
                f"atom {atom}: dwn-up block is not conj(up-dwn) "
                f"(max deviation {conj_deviation:.3e} {unit}); the parsed "
                "blocks violate the m_pawdij packing convention"
            )
        imag_deviation = max(np.max(np.abs(d1.imag)), np.max(np.abs(d2.imag)))
        if imag_deviation > tol:
            raise ValueError(
                f"atom {atom}: up-up/dwn-dwn blocks have significant imaginary "
                f"parts (max {imag_deviation:.3e} {unit}); expected real "
                "Hermitian symmetric-packed storage"
            )

        blocks[row, :ni, :ni, 0, 0] = d1 - d2
        blocks[row, :ni, :ni, 0, 1] = 2.0 * d3
        blocks[row, :ni, :ni, 1, 0] = 2.0 * d4
        blocks[row, :ni, :ni, 1, 1] = d2 - d1

        dense = blocks[row].transpose(2, 0, 3, 1).reshape(2 * ni, 2 * ni)
        hermiticity = np.max(np.abs(dense - dense.conj().T))
        if hermiticity > tol:
            raise ValueError(
                f"atom {atom}: assembled Delta is not Hermitian "
                f"(max deviation {hermiticity:.3e} {unit}); check the parsed "
                "Dij blocks for print corruption"
            )
    return blocks, unit


# ---------------------------------------------------------------------------
# Spinor ProjectorGreenData builder
# ---------------------------------------------------------------------------


def build_abinit_paw_spinor_data(
    coefficients: np.ndarray,
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    spinor_operator: np.ndarray,
    site_nproj: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
    occupations: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProjectorGreenData:
    """Build the spinor-native exchange-ready :class:`ProjectorGreenData`.

    ``coefficients`` must have shape ``(1, nkpt, nband, 2, nproj_total)`` —
    RAW dual-projector values ``<~p|psi_{n,s}>``, NO conjugation (k-gauge;
    see module docstring).  ``spinor_operator`` has shape
    ``(natom, nproj_max, nproj_max, 2, 2)`` in eV with the
    ``Delta = 2 B.sigma`` contract (identity part dropped).  ``eigenvalues``
    accepts ``(nkpt, nband)`` or ``(1, nkpt, nband)`` in eV.
    """
    coefficients = np.asarray(coefficients, dtype=complex)
    if (
        coefficients.ndim != 5
        or coefficients.shape[0] != 1
        or coefficients.shape[3] != 2
    ):
        raise ValueError(
            "spinor coefficients must have shape (1, nkpt, nband, 2, nproj_total); "
            f"got {coefficients.shape}"
        )
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    if eigenvalues.shape[0] != 1:
        raise ValueError(
            "spinor data has a single (nsppol=1) eigenvalue channel; got shape "
            f"{eigenvalues.shape}"
        )
    site_nproj = np.asarray(site_nproj, dtype=int)
    natom = len(site_nproj)
    if spinor_operator.shape[0] != natom:
        raise ValueError(
            f"spinor_operator covers {spinor_operator.shape[0]} sites, expected {natom}"
        )
    nproj_total = coefficients.shape[-1]
    if int(site_nproj.sum()) != nproj_total:
        raise ValueError(
            f"site_nproj sums to {int(site_nproj.sum())} but coefficients carry "
            f"{nproj_total} projector channels"
        )
    starts = np.cumsum([0, *site_nproj[:-1]])
    site_projector_indices = np.full((natom, int(site_nproj.max())), -1, dtype=int)
    for atom, (start, width) in enumerate(zip(starts, site_nproj, strict=True)):
        site_projector_indices[atom, :width] = np.arange(start, start + width)
    projector_site = np.repeat(np.arange(natom), site_nproj)

    meta = dict(metadata or {})
    meta.setdefault("code", "abinit")
    meta.setdefault("spinor_export", "abinit_paw_spinor")
    meta.setdefault(
        "coefficient_convention",
        "raw dual-projector cprj <~p|psi>, no conjugation (k-gauge, abe4152)",
    )
    meta.setdefault(
        "delta_source",
        "ABINIT pawprt nspden=4 Dij, Pauli decomposition Delta=2*B.sigma",
    )
    meta["units"] = {
        **meta.get("units", {}),
        "cell": "Angstrom",
        "positions": "Angstrom",
        "eigenvalues": "eV",
        "efermi": "eV",
        "spinor_operator": "eV",
    }
    data = ProjectorGreenData(
        kpoints=np.asarray(kpoints, dtype=float),
        weights=np.asarray(kweights, dtype=float),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=float(efermi),
        occupations=occupations,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        nspinor=2,
        spinor_operator=np.asarray(spinor_operator, dtype=complex),
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source="abinao.project_wfk_paw (nspinor=2 adapter, raw cprj)",
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_projector_channel",
        operator_basis=(
            "ABINIT pawprt nspden=4 Dij Pauli decomposition "
            "(up-up, dwn-dwn, up-dwn, dwn-up) -> Delta=2*B.sigma"
        ),
        metadata=meta,
    )
    data.validate(exchange_ready=True)
    return data


# ---------------------------------------------------------------------------
# abinao projection adapter (nspinor=2)
# ---------------------------------------------------------------------------


def adapt_paw_projection_to_spinor_coefficients(
    cprj_by_k: Sequence,
    nproj_total: int,
    nband: int | None = None,
) -> np.ndarray:
    """Adapt abinao ``PawProjectionResult.cprj`` to spinor coefficient layout.

    abinao ``project_wfk_paw`` (with an explicit ``nspinor=2`` result field,
    per-site ``(nproj, 2, nband)`` blocks, and the species-flattened path
    stacking those blocks on axis 0) emits ``cprj[ik][0]`` as
    ``(nproj_total, 2, nband)``; TB2J wants ``(nband, 2, nproj_total)``.
    This adapter only permutes axes — the values stay RAW (no conjugation),
    so the projector Green assembly applies the exchange convention itself.
    """
    adapted = []
    for ik, by_spin in enumerate(cprj_by_k):
        if len(by_spin) != 1:
            raise ValueError(
                f"k-point {ik}: a spinor WFK has nsppol=1; got {len(by_spin)} "
                "spin channels"
            )
        cprj = np.asarray(by_spin[0], dtype=complex)
        if cprj.ndim != 3 or cprj.shape != (nproj_total, 2, cprj.shape[2]):
            raise ValueError(
                f"k-point {ik}: expected abinao spinor cprj of shape "
                f"({nproj_total}, 2, nband) — (nproj_total, nspinor, nband) "
                f"with site blocks stacked along axis 0 — got {cprj.shape}. "
                "This is not a nspinor=2 projection result."
            )
        nb = cprj.shape[2]
        if nband is not None and nb != nband:
            raise ValueError(
                f"k-point {ik}: nband {nb} does not match eigenvalue channel ({nband})"
            )
        adapted.append(np.transpose(cprj, (2, 1, 0)))  # (nband, 2, nproj_total)
    return np.asarray(adapted, dtype=complex)[None]  # (1, nkpt, nband, 2, nproj)


def _project_wfk_spinor_in_process(
    wfk_path: str | Path,
    paw_xml_path: str | Path | Mapping[str, str | Path],
) -> dict:
    """Project an nspinor=2 WFK onto PAW projectors via abinao (read-only)."""
    try:
        from abinao.paw_projection import project_wfk_paw
        from abinao.wfk import read_wfk
    except ImportError as exc:
        raise ImportError(
            "abinao is required for in-process spinor WFK projection. "
            "Either install abinao or pass projected_data_path."
        ) from exc

    wfk = read_wfk(wfk_path)
    if wfk.nspinor != 2:
        raise ValueError(
            f"gen_exchange_abinit_paw_spinor expects an nspinor=2 WFK; got "
            f"nspinor={wfk.nspinor}"
        )
    if wfk.nsppol != 1:
        raise ValueError(f"a spinor WFK has nsppol=1; got nsppol={wfk.nsppol}")
    if any(int(v) != 1 for v in wfk.istwfk):
        raise NotImplementedError(
            "spinor WFK requires istwfk=1 (the istwfk>=2 time-reversal mirror "
            "expansion is not valid for spinors)"
        )
    if wfk.symrel is not None and len(wfk.symrel) > 1:
        raise NotImplementedError(
            "symmetry-reduced spinor WFK is not supported; abinao's BZ "
            "expansion is not spinor-aware. Run the full BZ (nsym 1, kptopt 0)."
        )

    xml_by_species = {
        str(symbol): Path(path)
        for symbol, path in normalize_paw_xml_mapping(
            wfk.atom_species, paw_xml_path
        ).items()
    }
    paw_pseudo = _load_paw_pseudo(xml_by_species)

    result = project_wfk_paw(wfk, paw_pseudo)

    site_layout = build_abinit_paw_site_layout(
        wfk.atom_species,
        xml_by_species,
        paw_pseudo,
        site_slices=getattr(result, "site_slices", None),
    )
    site_nproj = np.array(
        [
            site.projector_slice.stop - site.projector_slice.start
            for site in site_layout
        ],
        dtype=int,
    )
    nproj_total = int(site_nproj.sum())
    eigenvalues = np.asarray(
        [np.asarray(result.eigenvalues[ik][0], dtype=float) for ik in range(wfk.nkpt)],
        dtype=float,
    )  # (nkpt, nband), eV

    coefficients = adapt_paw_projection_to_spinor_coefficients(
        result.cprj,
        nproj_total=nproj_total,
        nband=eigenvalues.shape[1],
    )

    cell = np.asarray(wfk.rprimd, dtype=float) * BOHR_TO_ANGSTROM
    positions = np.asarray(wfk.xred, dtype=float) @ cell
    return {
        "coefficients": coefficients,
        "eigenvalues": eigenvalues,
        "kweights": np.asarray(result.kweights, dtype=float),
        "kpoints": np.asarray(result.kpoints, dtype=float),
        "efermi": float(result.efermi) if result.efermi is not None else 0.0,
        "site_nproj": site_nproj,
        "cell": cell,
        "positions": positions,
        "atomic_numbers": _atomic_numbers_from_symbols(list(wfk.atom_species)),
        "atom_species": tuple(wfk.atom_species),
        "site_slices": tuple(site.projector_slice for site in site_layout),
        "provenance": {
            "wfk_path": str(wfk_path),
            "paw_xml": {k: str(v) for k, v in xml_by_species.items()},
        },
    }


# ---------------------------------------------------------------------------
# Projected-data persistence (pickle)
# ---------------------------------------------------------------------------

_SPINOR_PROJECTED_KEYS = (
    "coefficients",
    "eigenvalues",
    "kweights",
    "kpoints",
    "efermi",
    "site_nproj",
)


def save_spinor_projected_data(
    path: str | Path,
    *,
    coefficients: np.ndarray,
    eigenvalues: np.ndarray,
    kweights: np.ndarray,
    kpoints: np.ndarray,
    efermi: float,
    site_nproj: np.ndarray,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    atomic_numbers: np.ndarray | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist spinor PAW projection results for later assembly."""
    payload: dict[str, Any] = {
        "coefficients": np.asarray(coefficients, dtype=complex),
        "eigenvalues": np.asarray(eigenvalues, dtype=float),
        "kweights": np.asarray(kweights, dtype=float),
        "kpoints": np.asarray(kpoints, dtype=float),
        "efermi": float(efermi),
        "site_nproj": np.asarray(site_nproj, dtype=int),
        "nspinor": 2,
    }
    for key, value in (
        ("cell", cell),
        ("positions", positions),
        ("atomic_numbers", atomic_numbers),
    ):
        if value is not None:
            payload[key] = np.asarray(value)
    if extra:
        payload["extra"] = dict(extra)
    path = Path(path)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_spinor_projected_data(path: str | Path) -> dict:
    """Load a pickle written by :func:`save_spinor_projected_data`."""
    path = Path(path)
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    missing = [k for k in _SPINOR_PROJECTED_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"spinor projected-data file {path} is missing keys: {missing}"
        )
    if int(payload.get("nspinor", 1)) != 2:
        raise ValueError(
            f"{path} is not spinor (nspinor=2) projected data; use the "
            "collinear abinit_paw path"
        )
    return payload


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def gen_exchange_abinit_paw_spinor(
    wfk_path: str | None = None,
    paw_xml_path: str | Mapping[str, str | Path] | None = None,
    log_path: str | None = None,
    projected_data_path: str | None = None,
    *,
    delta_ij: dict[int, np.ndarray] | None = None,
    delta_unit: str | None = None,
    magnetic_elements: list[str] | None = None,
    index_magnetic_atoms: list[int] | None = None,
    output_path: str = "TB2J_results_abinit_paw_spinor",
    nz: int = 30,
    smearing_eV: float = 0.05,
    Rcut: float = 10.0,
    description: str | None = None,
    hermiticity_tol: float | None = None,
) -> tuple[Path, dict]:
    """Run the spinor (nspden=4 / nspinor=2) PAW exchange calculation.

    Data sources (exactly one projection source):

    * ``wfk_path`` + ``paw_xml_path`` — in-process abinao projection of the
      spinor WFK (raw cprj adapter);
    * ``projected_data_path`` — a pickle written by
      :func:`save_spinor_projected_data`.

    Delta sources (exactly one):

    * ``log_path`` — ABINIT log with the nspden=4 pawprt Dij blocks; parsed
      and Pauli-decomposed into the per-site 2x2 splitting;
    * ``delta_ij`` — pre-assembled per-site ``(ni, ni, 2, 2)`` Delta blocks
      in ``delta_unit`` (default eV).

    A single spinor reference cannot determine Jiso/DMI/Jani. This loader
    validates the projection and magnetic operator, but the scalar output
    writer refuses unless supplied independent x/y/z magnetic references;
    use the PAW split-SOC three-leg driver for production exchange.
    """
    if projected_data_path is not None:
        proj = load_spinor_projected_data(projected_data_path)
    elif wfk_path is not None:
        if paw_xml_path is None:
            raise ValueError("paw_xml_path is required when wfk_path is given")
        proj = _project_wfk_spinor_in_process(wfk_path, paw_xml_path)
    else:
        raise ValueError(
            "Provide either projected_data_path or wfk_path (+paw_xml_path)."
        )

    if delta_ij is not None:
        atoms = sorted(int(a) for a in delta_ij)
        max_width = max(int(np.asarray(delta_ij[a]).shape[0]) for a in atoms)
        blocks = np.zeros((len(atoms), max_width, max_width, 2, 2), dtype=complex)
        for row, atom in enumerate(atoms):
            block = np.asarray(delta_ij[atom], dtype=complex)
            if block.ndim != 4 or block.shape[2:] != (2, 2):
                raise ValueError(
                    f"delta_ij[{atom}] must be an (ni, ni, 2, 2) Pauli block; "
                    f"got {block.shape}"
                )
            blocks[row, : block.shape[0], : block.shape[1]] = block
        resolved_unit = delta_unit or "eV"
        if resolved_unit.strip().lower().startswith(("ha", "hartree")):
            blocks = blocks * HARTREE_TO_EV
            resolved_unit = "eV"
    elif log_path is not None:
        components, parsed_unit = parse_pawprt_dij_spinor(log_path, unit=delta_unit)
        blocks, resolved_unit = pauli_delta_blocks_from_components(
            components,
            parsed_unit,
            hermiticity_tol=hermiticity_tol,
        )
        if resolved_unit.lower().startswith(("ha", "hartree")):
            blocks = blocks * HARTREE_TO_EV
            resolved_unit = "eV"
    else:
        raise ValueError(
            "Either log_path or delta_ij must be provided to obtain Delta_ij."
        )
    if resolved_unit != "eV":
        raise ValueError(f"internal error: Delta unit {resolved_unit!r} is not eV")

    site_nproj = np.asarray(proj["site_nproj"], dtype=int)
    if blocks.shape[0] != len(site_nproj):
        raise ValueError(
            f"Delta covers {blocks.shape[0]} atoms but the projection has "
            f"{len(site_nproj)} sites"
        )
    for atom, width in enumerate(site_nproj):
        if blocks.shape[1] < width:
            raise ValueError(
                f"atom {atom}: Delta block width {blocks.shape[1]} is smaller "
                f"than the {width} projector channels of the PAW dataset"
            )

    data = build_abinit_paw_spinor_data(
        coefficients=proj["coefficients"],
        eigenvalues=proj["eigenvalues"],
        kweights=proj["kweights"],
        kpoints=proj["kpoints"],
        efermi=proj["efermi"],
        spinor_operator=blocks,
        site_nproj=site_nproj,
        cell=proj.get("cell"),
        positions=proj.get("positions"),
        atomic_numbers=proj.get("atomic_numbers"),
        metadata=proj.get("extra") or proj.get("provenance"),
    )

    from TB2J.interfaces.gpaw_spinor_projector import (
        write_spinor_projector_exchange_out,
    )

    if description is None:
        description = (
            "ABINIT PAW spinor projector Green data from nspinor=2 WFK and "
            "nspden=4 PAW Dij magnetic operator. Full exchange requires "
            "three independent magnetic reference axes.\n"
        )
    return write_spinor_projector_exchange_out(
        data,
        path=output_path,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        Rcut=Rcut,
    )
