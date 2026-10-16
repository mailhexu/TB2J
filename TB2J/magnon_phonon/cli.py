"""Thin CLI for the magnon-phonon package (ADR-006: explicit calculation
methods, JSON-result conventions).  Story 011 exposes the ``hybrid``
calculation; other calculations are separate entries owned by their
stories.

Usage::

    python -m TB2J.magnon_phonon.cli hybrid manifest.json

Manifest (JSON) for the controlled two-boson model::

    {
        "calculation": "hybrid",
        "model": {"type": "controlled_two_boson",
                  "alpha_eV": 2.0, "beta_eV": 1.0, "g_eV": 0.1},
        "points": [[0.0, 0.0, 0.0]]
    }

The output JSON carries energies (eV), magnetic/displacement weights,
stability flags and per-point diagnostics (shifts are always zero: the
stability threshold is flagged, never regularized).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _run_hybrid(manifest: dict) -> dict:
    from TB2J.magnon_phonon.harmonic import (
        ControlledTwoBosonModel,
        hybrid_modes,
    )

    model_spec = manifest.get("model")
    if not model_spec or model_spec.get("type") != "controlled_two_boson":
        raise ValueError(
            "hybrid manifest requires model.type = 'controlled_two_boson'; "
            "material mixed-Hessian producers supply their own typed input."
        )
    model = ControlledTwoBosonModel(
        alpha_eV=float(model_spec["alpha_eV"]),
        beta_eV=float(model_spec["beta_eV"]),
        g_eV=float(model_spec["g_eV"]),
    )
    points = np.asarray(manifest["points"], dtype=float)
    result = hybrid_modes(points, model.to_mixed_hessian())
    d = result.diagnostics
    return {
        "calculation": "hybrid",
        "model": model_spec,
        "units": {"energy": "eV"},
        "kpoints": points.tolist(),
        "energies_eV": result.energies.tolist(),
        "magnetic_weight": result.magnetic_weight.tolist(),
        "displacement_weight": result.displacement_weight.tolist(),
        "is_stable": result.is_stable.tolist(),
        "diagnostics": {
            "min_eig": d.min_eig.tolist(),
            "stability_tol": d.stability_tol.tolist(),
            "stability_flag": d.stability_flag.tolist(),
            "shift": d.shift.tolist(),
            "pairing_residual": d.pairing_residual.tolist(),
            "hermiticity_residual": d.hermiticity_residual.tolist(),
        },
        "provenance": model.to_mixed_hessian().provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m TB2J.magnon_phonon.cli",
        description="Magnon-phonon calculations (JSON manifest in, JSON result out).",
    )
    sub = parser.add_subparsers(dest="calculation", required=True)
    hybrid_parser = sub.add_parser("hybrid", help="harmonic mixed spin/lattice modes")
    hybrid_parser.add_argument("manifest", help="JSON manifest path ('-' for stdin)")
    args = parser.parse_args(argv)

    if args.calculation == "hybrid":
        if args.manifest == "-":
            manifest = json.load(sys.stdin)
        else:
            with open(args.manifest, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        if manifest.get("calculation") not in (None, "hybrid"):
            raise ValueError("manifest calculation must be 'hybrid'")
        print(json.dumps(_run_hybrid(manifest), indent=2))


if __name__ == "__main__":
    main()
