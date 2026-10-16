"""Structure-only data adapters for the magnon-phonon vertex (Story 009).

These loaders read existing evidence formats and map them onto the
filesystem-free containers of :mod:`TB2J.magnon_phonon.vertex`.  They never
run SCFs and never write computational state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from TB2J.magnon_phonon.vertex import PhononModeInput

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

    atoms = sorted(
        {
            int(rec["provenance"]["geometry"]["displaced_atom_zero_based"])
            for rec in records
        }
    )
    axes = sorted(
        {
            int(rec["provenance"]["geometry"]["cartesian_axis_one_based"])
            for rec in records
        }
    )
    natom = max(atoms) + 1
    naxis = max(axes)

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
    return PhononModeInput(
        q=q, frequencies_thz=freq, masses_amu=masses, eigenvectors=eig
    )
