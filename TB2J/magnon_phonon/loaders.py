"""Structure-only data adapters for the magnon-phonon vertex (Story 009).

These loaders read existing evidence formats and map them onto the
filesystem-free containers of :mod:`TB2J.magnon_phonon.vertex`.  They never
run SCFs and never write computational state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from TB2J.magnon_phonon.vertex import ModeProjectedDerivatives, PhononModeInput

#: matched signed-row schema produced by the Story-001 recovery
MATCHED_ROWS_SCHEMA = "matched-periodic-J-v1"
#: derivative unit column consumed from matched rows
_DERIVATIVE_COLUMN = "FD_meV_A"


def _bond_key(row):
    return (int(row["i"]), int(row["j"]), tuple(int(x) for x in row["R"]))


def load_matched_rows(paths) -> "tuple[object, object]":
    """Load Story-001/002 matched periodic rows into derivative containers.

    Parameters
    ----------
    paths:
        Iterable of JSON paths (or loaded dicts), one per displaced Cartesian
        axis, in the ``matched-periodic-J-v1`` schema.  Each file carries
        ``rows`` with ``i``, ``j``, ``R`` and the central-difference
        derivative ``FD_meV_A`` for the displacement recorded in
        ``geometry.displaced_atom_zero_based`` / ``cartesian_axis_one_based``.

    Returns
    -------
    (OrientedBondDerivatives, metadata dict)
        The derivative container uses the full-moment ``JR`` convention with
        units converted to eV/Angstrom; the metadata preserves per-file
        provenance (campaign, step, hashes) for downstream records.

    Raises
    ------
    ValueError
        On schema mismatch or inconsistent bond sets across axis files.
    """
    records = []
    for path in paths:
        if isinstance(path, (str, Path)):
            with open(path) as stream:
                records.append(json.load(stream))
        else:
            records.append(path)
    if not records:
        raise ValueError("no matched-row files supplied")

    bond_order = []
    seen = None
    for rec in records:
        if rec.get("schema") != MATCHED_ROWS_SCHEMA:
            raise ValueError(
                f"expected schema {MATCHED_ROWS_SCHEMA!r}, got {rec.get('schema')!r}"
            )
        keys = [_bond_key(row) for row in rec["rows"]]
        if seen is None:
            seen = set(keys)
            bond_order = keys
        elif set(keys) != seen:
            raise ValueError("axis files carry inconsistent directed-bond sets")

    axis_by_atom = {}
    atom_axis_records = set()
    for rec in records:
        geometry = rec["provenance"]["geometry"]
        atom = int(geometry["displaced_atom_zero_based"])
        axis = int(geometry["cartesian_axis_one_based"])
        if atom < 0 or axis not in (1, 2, 3):
            raise ValueError(
                "matched rows require zero-based atoms and Cartesian axes 1, 2, 3"
            )
        key = (atom, axis)
        if key in atom_axis_records:
            raise ValueError(
                f"duplicate matched displacement record for atom/axis {key}"
            )
        atom_axis_records.add(key)
        axis_by_atom.setdefault(atom, set()).add(axis)
    incomplete = {
        atom: sorted({1, 2, 3} - axes)
        for atom, axes in axis_by_atom.items()
        if axes != {1, 2, 3}
    }
    if incomplete:
        raise ValueError(
            "matched home-cell rows require all three Cartesian axes per displaced atom; "
            f"missing axes: {incomplete}"
        )
    atoms = sorted(axis_by_atom)
    if atoms != list(range(max(atoms) + 1)):
        raise ValueError("matched home-cell rows omit one or more intervening atoms")
    natom = max(atoms) + 1
    naxis = 3
    n_bonds = len(bond_order)
    dJ = np.zeros((n_bonds, natom, naxis), dtype=float)
    index = {key: pos for pos, key in enumerate(bond_order)}
    for rec in records:
        geometry = rec["provenance"]["geometry"]
        atom = int(geometry["displaced_atom_zero_based"])
        axis = int(geometry["cartesian_axis_one_based"]) - 1
        step = float(geometry.get("half_step_A", rec.get("h_Ang", 0.0)))
        if step <= 0:
            raise ValueError("matched rows record no positive displacement step")
        for row in rec["rows"]:
            pos = index[_bond_key(row)]
            dJ[pos, atom, axis] = float(row[_DERIVATIVE_COLUMN]) * 1e-3  # meV -> eV

    from TB2J.magnon_phonon.vertex import OrientedBondDerivatives

    bonds = OrientedBondDerivatives(
        i=np.array([key[0] for key in bond_order], dtype=int),
        j=np.array([key[1] for key in bond_order], dtype=int),
        R=np.array([key[2] for key in bond_order], dtype=int),
        dJ_dtau=dJ,
        convention="home_cell",
        units="eV/Angstrom",
    )
    metadata = {
        "schema": MATCHED_ROWS_SCHEMA,
        "sources": [
            {
                "campaign": rec.get("campaign"),
                "provenance": rec.get("provenance"),
            }
            for rec in records
        ],
    }
    return bonds, metadata


def load_s002_report(
    report_path,
    *,
    campaign: str,
    mode: str,
    expected_bonds,
    mode_evidence_path,
) -> ModeProjectedDerivatives:
    """Load one finite-q S002 report and join its hash-pinned mode evidence.

    The report supplies complex ``O`` and finite-difference rows; its companion
    mode-evidence JSON supplies the Cartesian polarization and eigenvector
    index. Both the bond set and the mode/evidence SHA must be explicitly
    matched. Missing polarization, complex components, rows, or provenance is
    rejected instead of inferred.
    """
    report = json.loads(Path(report_path).read_text())
    try:
        campaign_record = report["campaigns"][campaign]
    except KeyError as exc:
        raise ValueError(f"S002 report has no campaign {campaign!r}") from exc
    reports = [
        entry
        for entry in campaign_record.get("reports", [])
        if entry.get("mode") == mode
    ]
    if len(reports) != 1:
        raise ValueError(
            f"S002 report must contain exactly one mode {mode!r}; found {len(reports)}"
        )
    entry = reports[0]
    q = np.asarray(entry.get("q"), dtype=float)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("S002 report has invalid q")
    evidence_bytes = Path(mode_evidence_path).read_bytes()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    if evidence_hash != entry.get("source_sha256"):
        raise ValueError("S002 source mode evidence hash mismatch")
    evidence = json.loads(evidence_bytes)
    if evidence.get("mode") != mode:
        raise ValueError("S002 source mode evidence label mismatch")
    evidence_q = np.asarray(evidence.get("q"), dtype=float)
    if evidence_q.shape != (3,) or not np.allclose(evidence_q, q, atol=1e-8, rtol=0):
        raise ValueError("S002 source mode q does not match report q")
    if "polarization_real" not in evidence or "mode_eigenvector_index" not in evidence:
        raise ValueError(
            "unsupported S002 mode evidence: missing polarization or eigenvector index"
        )
    polarization = np.asarray(
        evidence["polarization_real"], dtype=float
    ) + 1j * np.asarray(
        evidence.get(
            "polarization_imag",
            np.zeros_like(evidence["polarization_real"], dtype=float),
        ),
        dtype=float,
    )
    if (
        polarization.ndim != 2
        or polarization.shape[1] != 3
        or not np.isfinite(polarization).all()
    ):
        raise ValueError(
            "S002 source polarization must be finite with shape (natom, 3)"
        )
    if not np.isclose(np.linalg.norm(polarization), 1.0, atol=1e-6, rtol=0):
        raise ValueError("S002 source polarization must have unit Cartesian norm")

    rows = entry.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("S002 report contains no directed-bond rows")
    required = (
        "iatom",
        "jatom",
        "R",
        "O_real_meV_per_Ang",
        "O_imag_meV_per_Ang",
        "FD_real_meV_per_Ang",
        "FD_imag_meV_per_Ang",
    )
    keys = []
    O = []
    FD = []
    for row in rows:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"S002 row missing component(s): {', '.join(missing)}")
        key = (
            int(row["iatom"]),
            int(row["jatom"]),
            tuple(int(value) for value in row["R"]),
        )
        keys.append(key)
        O.append(
            complex(
                float(row["O_real_meV_per_Ang"]),
                float(row["O_imag_meV_per_Ang"]),
            )
            * 1e-3
        )
        FD.append(
            complex(
                float(row["FD_real_meV_per_Ang"]),
                float(row["FD_imag_meV_per_Ang"]),
            )
            * 1e-3
        )
    if len(set(keys)) != len(keys):
        raise ValueError("S002 report contains duplicate directed bonds")
    expected = {
        (int(i), int(j), tuple(int(value) for value in R)) for i, j, R in expected_bonds
    }
    if set(keys) != expected:
        raise ValueError(
            "S002 report bond set does not match expected directed bond set"
        )

    return ModeProjectedDerivatives(
        q=q,
        mode=mode,
        mode_index=int(evidence["mode_eigenvector_index"]),
        polarization=polarization,
        i=np.asarray([key[0] for key in keys], dtype=int),
        j=np.asarray([key[1] for key in keys], dtype=int),
        R=np.asarray([key[2] for key in keys], dtype=int),
        O_pattern=np.asarray(O, dtype=complex),
        FD_pattern=np.asarray(FD, dtype=complex),
        source={
            "campaign": campaign,
            "campaign_record": campaign_record.get("campaign"),
            "source_evidence": entry.get("source_evidence"),
            "source_sha256": evidence_hash,
            "mode_evidence": evidence,
        },
        mode_evidence_sha256=evidence_hash,
    )


def as_phonon_mode_input(obj) -> PhononModeInput:
    """Adapt a duck-typed producer object to :class:`PhononModeInput`.

    Accepts any object exposing ``q``, ``frequencies_thz``, ``masses_amu`` and
    ``eigenvectors`` (e.g. ``unfold_epc.phonon.PhononModes``, imported lazily
    by the caller and never required by this module).  The eigenvector row
    count must match ``3 * len(masses_amu)``: supercell-ordered patterns must
    be folded to the primitive cell before reaching the vertex seam.
    """
    for name in ("q", "frequencies_thz", "masses_amu", "eigenvectors"):
        if not hasattr(obj, name):
            raise ValueError(f"phonon producer object lacks attribute {name!r}")
    q = np.asarray(obj.q, dtype=float).reshape(3)
    freq = np.asarray(obj.frequencies_thz, dtype=float)
    masses = np.asarray(obj.masses_amu, dtype=float)
    eig = np.asarray(obj.eigenvectors, dtype=complex)
    if eig.ndim != 2 or eig.shape[0] != 3 * len(masses):
        raise ValueError(
            f"eigenvector shape {eig.shape} does not match 3 x natom "
            f"({3 * len(masses)} rows); fold supercell patterns to the "
            "primitive cell before the vertex seam"
        )
    if eig.shape[1] != len(freq):
        raise ValueError("eigenvector columns do not match the frequency count")
    provenance = getattr(obj, "provenance", {})
    return PhononModeInput(
        q=q,
        frequencies_thz=freq,
        masses_amu=masses,
        eigenvectors=eig,
        provenance=dict(provenance),
    )
