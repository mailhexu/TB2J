"""Shared ``SpinIO`` assertion helpers for E2E validation tests (story 010-1).

Each E2E case is a plain pytest function that loads the canonical ``SpinIO``
result and calls these helpers. They implement the layered oracle of ADR-001 /
ADR-005: schema, then per-quantity toleranced comparison, then independent
physical invariants.

Conventions (verified against the exchange core):
- ``exchange_Jdict`` / ``dmi_ddict`` / ``Jani_dict`` keys are ``(R, i, j)`` where
  ``R`` is a 3-tuple cell index and ``i``, ``j`` are spin indices. (The
  ``SpinIO`` docstring's ``(i, j, R)`` order is incorrect; the writer
  ``io_txt.py`` unpacks ``R, i, j = key``.)
- Both ``(R, i, j)`` and ``(-R, j, i)`` are stored, so pair reversal is
  ``J[(R, i, j)] == J[(-R, j, i)]`` and DMI antisymmetry is
  ``D[(R, i, j)] == -D[(-R, j, i)]``.
- Stored exchange values are in **eV** (the text writer multiplies by 1e3 for
  meV). ``compare_J(unit="meV")`` converts expected meV values to eV first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

__all__ = [
    "check_schema",
    "compare_J",
    "check_pair_reversal",
    "check_dmi_antisymmetry",
    "check_jani_hermiticity",
    "check_exchange_out_section",
]

_REQUIRED_ATTRS = ("atoms", "index_spin", "spinat", "exchange_Jdict", "TB2J_version")
_MEV_PER_EV = 1000.0


def check_schema(sio) -> None:
    """Assert ``sio`` is a well-formed ``SpinIO`` with finite exchange data.

    Fails if a required provenance field is missing/None or any stored
    isotropic exchange value is non-finite.
    """
    for attr in _REQUIRED_ATTRS:
        if not hasattr(sio, attr):
            raise AssertionError(f"SpinIO missing required attribute '{attr}'")
        if getattr(sio, attr) is None:
            raise AssertionError(f"SpinIO attribute '{attr}' is None")
    jdict = sio.exchange_Jdict
    for key, val in jdict.items():
        if not np.isfinite(val):
            raise AssertionError(f"non-finite J at {key}: {val}")


def _to_ev(value: float, unit: str) -> float:
    """Convert an expected quantity to the eV units used in storage."""
    unit = unit.lower()
    if unit == "ev":
        return float(value)
    if unit == "mev":
        return float(value) / _MEV_PER_EV
    raise ValueError(f"unknown unit '{unit}' (expected 'eV' or 'meV')")


def compare_J(
    sio,
    expected: Mapping,
    tol: float = 1e-6,
    unit: str = "meV",
) -> None:
    """Compare selected ``J[(R, i, j)]`` against reviewed references.

    ``expected`` maps ``(R, i, j)`` -> reference value in ``unit``. Each value is
    compared to the stored eV value with an absolute ``tol`` (in eV). Reports the
    offending key and unit on failure; fails if a requested key is absent.
    """
    jdict = sio.exchange_Jdict
    for key, ref in expected.items():
        if key not in jdict:
            raise AssertionError(f"J key {key} not in exchange_Jdict (unit={unit})")
        actual = jdict[key]
        expected_ev = _to_ev(ref, unit)
        if not np.isfinite(actual):
            raise AssertionError(f"J at {key} is non-finite: {actual}")
        diff = abs(actual - expected_ev)
        if diff > tol:
            raise AssertionError(
                f"J at {key}: stored {actual:.6g} eV vs expected "
                f"{ref:g} {unit} ({expected_ev:.6g} eV); |diff|={diff:.3e} eV "
                f"> tol={tol:.3e} eV"
            )


def _rev_key(key):
    """Return the pair-reversed key ``(R, i, j) -> (-R, j, i)``."""
    R, i, j = key
    return (tuple(-x for x in R), j, i)


def check_pair_reversal(sio, tol: float = 1e-6) -> None:
    """Assert isotropic exchange obeys ``J[(R,i,j)] == J[(-R,j,i)]``."""
    jdict = sio.exchange_Jdict
    for key, val in jdict.items():
        rev = _rev_key(key)
        if rev not in jdict:
            raise AssertionError(f"pair reversal: reverse key {rev} missing for {key}")
        if abs(val - jdict[rev]) > tol:
            raise AssertionError(
                f"pair reversal violated at {key}: {val} vs {jdict[rev]} at {rev}"
            )


def check_dmi_antisymmetry(sio, tol: float = 1e-6) -> None:
    """Assert DMI obeys ``D[(R,i,j)] == -D[(-R,j,i)]`` (skips if no DMI)."""
    ddict = getattr(sio, "dmi_ddict", None) or {}
    for key, vec in ddict.items():
        rev = _rev_key(key)
        if rev not in ddict:
            raise AssertionError(
                f"DMI antisymmetry: reverse key {rev} missing for {key}"
            )
        if not np.allclose(vec, -np.asarray(ddict[rev]), atol=tol):
            raise AssertionError(
                f"DMI antisymmetry violated at {key}: {vec} vs {ddict[rev]} at {rev}"
            )


def check_jani_hermiticity(sio, tol: float = 1e-6) -> None:
    """Assert each anisotropy matrix is Hermitian (no-op when none stored)."""
    jdict = getattr(sio, "Jani_dict", None) or {}
    for key, mat in jdict.items():
        mat = np.asarray(mat)
        if mat.shape != (3, 3):
            raise AssertionError(
                f"Jani at {key} has shape {mat.shape}, expected (3, 3)"
            )
        if not np.allclose(mat, mat.conj().T, atol=tol):
            raise AssertionError(f"Jani at {key} is not Hermitian:\n{mat}")


def check_exchange_out_section(path, section: str) -> None:
    """Assert a named section/substring is present in an ``exchange.out`` file."""
    text = Path(path).read_text()
    if section not in text:
        raise AssertionError(f"exchange.out section '{section}' not found in {path}")
