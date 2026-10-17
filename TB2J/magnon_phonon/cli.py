"""Thin CLI for the magnon-phonon package (ADR-006: explicit calculation
methods, JSON-result conventions).  Story 010 exposes the ``scattering``
calculation (one-shot Born kernels); other calculations are separate
entries owned by their stories.

Usage::

    python -m TB2J.magnon_phonon.cli scattering manifest.json

The manifest selects the frequency grid, temperature, broadening and one
or both kernel blocks with explicit arrays (kernels are filesystem-free):

    {
      "calculation": "scattering",
      "omega_eV": [...],
      "broadening_eV": 1e-3,
      "temperature_K": 150.0,
      "magnon": {"g": [...], "eps_kq": [...], "omega_q_ev": [...],
                 "n_bose_q": [...], "weights": [...]},
      "phonon": {"g": [...], "eps_k": [...], "eps_kq": [...],
                 "omega_nu_ev": [...], "weights": [...]}
    }

``magnon`` runs :func:`magnon_self_energy` (result ``sigma``, shape
``(nw, n, n)``); ``phonon`` runs :func:`phonon_hot_band_kernel` (results
``delta_d`` — the additive coordinate Green-function correction, not a
self-energy — and ``sigma_nu``, the Dyson-partitioned self-energy).  The
output JSON echoes the metadata conventions of the seam.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _json_clean(value):
    """Convert NumPy values and non-finite diagnostics to strict JSON types."""
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": _json_clean(value.real), "imag": _json_clean(value.imag)}
        return _json_clean(value.tolist())
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _run_scattering(manifest: dict) -> dict:
    from TB2J.magnon_phonon.scattering import (
        bose_occupation,
        magnon_self_energy,
        phonon_hot_band_kernel,
    )

    omega = np.asarray(manifest["omega_eV"], dtype=float)
    eta = float(manifest["broadening_eV"])
    temperature = float(manifest.get("temperature_K", 0.0))
    out = {
        "calculation": "scattering",
        "omega_eV": omega,
        "broadening_eV": eta,
        "temperature_K": temperature,
        "metadata": {
            "approximation": (
                "one-shot O(g^2) Born (bare internal propagators, not SCBA)"
            ),
            "phonon_sigma_convention": (
                "sigma_nu is the Dyson-partitioned self-energy "
                "D0^-1 deltaD D0^-1; delta_d is the additive coordinate "
                "Green-function correction, not a self-energy"
            ),
        },
    }
    if "magnon" in manifest:
        block = manifest["magnon"]
        g = np.asarray(block["g"], dtype=complex)
        eps_kq = np.asarray(block["eps_kq"], dtype=float)
        om_q = np.asarray(block["omega_q_ev"], dtype=float)
        n_bose = (
            np.asarray(block["n_bose_q"], dtype=float)
            if "n_bose_q" in block
            else bose_occupation(om_q, temperature)
        )
        weights = np.asarray(block["weights"], dtype=float)
        sigma = magnon_self_energy(g, eps_kq, om_q, n_bose, weights, omega, eta)
        out["magnon"] = {"sigma": sigma}
    if "phonon" in manifest:
        block = manifest["phonon"]
        g = np.asarray(block["g"], dtype=complex)
        eps_k = np.asarray(block["eps_k"], dtype=float)
        eps_kq = np.asarray(block["eps_kq"], dtype=float)
        om_nu = np.asarray(block["omega_nu_ev"], dtype=float)
        weights = np.asarray(block["weights"], dtype=float)
        n_in = (
            np.asarray(block["n_bose_k"], dtype=float)
            if "n_bose_k" in block
            else bose_occupation(eps_k, temperature)
        )
        n_out = (
            np.asarray(block["n_bose_kq"], dtype=float)
            if "n_bose_kq" in block
            else bose_occupation(eps_kq, temperature)
        )
        delta_d, sigma_nu = phonon_hot_band_kernel(
            g, eps_k, eps_kq, n_in, n_out, om_nu, weights, omega, eta
        )
        out["phonon"] = {"delta_d": delta_d, "sigma_nu": sigma_nu}
    if "magnon" not in manifest and "phonon" not in manifest:
        raise ValueError(
            "scattering manifest requires a 'magnon' and/or 'phonon' kernel block"
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m TB2J.magnon_phonon.cli",
        description="Magnon-phonon calculations (JSON manifest in, JSON result out).",
    )
    sub = parser.add_subparsers(dest="calculation", required=True)
    scattering_parser = sub.add_parser(
        "scattering", help="one-shot Born magnon/phonon scattering kernels"
    )
    scattering_parser.add_argument(
        "manifest", help="JSON manifest path ('-' for stdin)"
    )
    args = parser.parse_args(argv)

    if args.calculation == "scattering":
        if args.manifest == "-":
            manifest = json.load(sys.stdin)
        else:
            with open(args.manifest, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        if manifest.get("calculation") not in (None, "scattering"):
            raise ValueError("manifest calculation must be 'scattering'")
        print(
            json.dumps(
                _json_clean(_run_scattering(manifest)), indent=2, allow_nan=False
            )
        )


if __name__ == "__main__":
    main()
