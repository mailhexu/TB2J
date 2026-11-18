"""Thin CLI for the magnon-phonon package (ADR-006: explicit calculation
methods, JSON-result conventions).  Story 010 exposes the ``scattering``
calculation (one-shot Born kernels) and Story 011 exposes the ``hybrid``
calculation (harmonic mixed spin/lattice modes); other calculations are
separate entries owned by their stories.

Usage::

    python -m TB2J.magnon_phonon.cli scattering manifest.json
    python -m TB2J.magnon_phonon.cli hybrid manifest.json

Scattering manifests select the frequency grid, temperature, broadening and
one or both kernel blocks with explicit arrays (kernels are filesystem-free):

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

Hybrid manifests support the controlled model or pre-supplied typed Hessian
blocks.  For ``mixed_hessian_npz``, the NPZ stores ``magnon_bare``,
``phonon_bare`` and ``coupling`` arrays, optionally with its matching
``points`` array.  The JSON model entry supplies ``npz_path``,
``n_mag_sites``, ``spin_coordinate_labels``, ``displacement_labels``,
``producer`` and ``provenance``.  Arrays may be static matrices or carry a
leading point axis.  Scalar-collinear producers remain rejected by the
numerical contract.  The output JSON carries energies (eV),
magnetic/displacement weights, stability flags and per-point diagnostics
(shifts are always zero: the stability threshold is flagged, never
regularized).
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


def _typed_hessian_from_npz(model_spec: dict, points: np.ndarray):
    """Load already-qualified numerical blocks outside the kernel layer."""
    from TB2J.magnon_phonon.harmonic import MixedSpinDisplacementHessian

    archive_path = model_spec["npz_path"]
    with np.load(archive_path, allow_pickle=False) as archive:
        magnon = np.array(archive["magnon_bare"], dtype=complex, copy=True)
        phonon = np.array(archive["phonon_bare"], dtype=complex, copy=True)
        coupling = np.array(archive["coupling"], dtype=complex, copy=True)
        archived_points = (
            np.array(archive["points"], dtype=float, copy=True)
            if "points" in archive
            else None
        )
    if archived_points is not None and (
        archived_points.shape != points.shape
        or not np.allclose(archived_points, points, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("NPZ points do not match the hybrid manifest points")

    def point_provider(array):
        if array.ndim == 2:
            return lambda query: np.broadcast_to(array, (len(query),) + array.shape)
        if array.shape[0] != len(points):
            raise ValueError(
                "per-point NPZ Hessians must have one leading entry per manifest point"
            )

        def provide(query):
            indices = []
            for point in query:
                distances = np.linalg.norm(points - point, axis=1)
                index = int(np.argmin(distances))
                if distances[index] > 1e-10:
                    raise ValueError(f"NPZ Hessian has no entry for point {point}")
                indices.append(index)
            return array[indices]

        return provide

    n_mag = int(model_spec["n_mag_sites"])
    displacement_coordinate = model_spec.get("displacement_coordinate", "mass_weighted")
    return MixedSpinDisplacementHessian(
        n_mag_sites=n_mag,
        displacement_labels=tuple(
            tuple(label) for label in model_spec["displacement_labels"]
        ),
        spin_coordinate_labels=tuple(
            tuple(label) for label in model_spec["spin_coordinate_labels"]
        ),
        magnon_bare=point_provider(magnon),
        phonon_bare=point_provider(phonon),
        mixed_derivative=point_provider(coupling),
        spin_lengths=tuple(float(s) for s in model_spec["spin_lengths"]),
        displacement_coordinate=displacement_coordinate,
        masses_amu=(
            tuple(float(m) for m in model_spec["masses_amu"])
            if "masses_amu" in model_spec
            else None
        ),
        producer=model_spec["producer"],
        provenance=model_spec["provenance"],
        units=model_spec.get(
            "units",
            {
                "energy": "eV",
                "spin_coordinate": "dimensionless local transverse spin components (hbar=1)",
                "displacement_coordinate": "mass_weighted | cartesian_angstrom",
            },
        ),
        fourier_convention=model_spec.get(
            "fourier_convention",
            "gamma(k) = sum_R gamma(R) exp(+2 pi i k . R)",
        ),
    )


def _run_hybrid(manifest: dict) -> dict:
    from TB2J.magnon_phonon.harmonic import (
        ControlledTwoBosonModel,
        hybrid_modes,
    )

    model_spec = manifest.get("model")
    if not model_spec:
        raise ValueError("hybrid manifest requires a model")
    points = np.atleast_2d(np.asarray(manifest["points"], dtype=float))
    model_type = model_spec.get("type")
    if model_type == "controlled_two_boson":
        controlled = ControlledTwoBosonModel(
            alpha_eV=float(model_spec["alpha_eV"]),
            beta_eV=float(model_spec["beta_eV"]),
            g_eV=float(model_spec["g_eV"]),
        )
        mixed = controlled.to_mixed_hessian()
    elif model_type == "mixed_hessian_npz":
        mixed = _typed_hessian_from_npz(model_spec, points)
    else:
        raise ValueError(
            "hybrid model.type must be 'controlled_two_boson' or 'mixed_hessian_npz'"
        )
    result = hybrid_modes(points, mixed)
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
        "provenance": mixed.provenance,
    }


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
    hybrid_parser = sub.add_parser("hybrid", help="harmonic mixed spin/lattice modes")
    hybrid_parser.add_argument("manifest", help="JSON manifest path ('-' for stdin)")
    args = parser.parse_args(argv)

    if args.manifest == "-":
        manifest = json.load(sys.stdin)
    else:
        with open(args.manifest, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    if manifest.get("calculation") not in (None, args.calculation):
        raise ValueError(f"manifest calculation must be '{args.calculation}'")
    if args.calculation == "scattering":
        result = _run_scattering(manifest)
    else:
        result = _run_hybrid(manifest)
    print(json.dumps(_json_clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
