"""Fail-closed manifest workflow for certified SIESTA spiral exchange."""

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


class WorkflowRefusal(ValueError):
    """A manifest cannot produce a certified exchange artifact."""


def _keys(record, required, optional=(), *, where="manifest"):
    if not isinstance(record, dict):
        raise WorkflowRefusal(f"{where}: supply a JSON object")
    missing = set(required) - record.keys()
    extra = record.keys() - set(required) - set(optional)
    if missing or extra:
        raise WorkflowRefusal(
            f"{where}: missing keys {sorted(missing)}; unsupported keys {sorted(extra)}; "
            "use the documented version-1 manifest schema"
        )


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowRefusal(
                f"duplicate JSON key {key!r}; declare each setting once"
            )
        result[key] = value
    return result


def _read_json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _validate_manifest(manifest):
    _keys(
        manifest,
        (
            "version",
            "inputs",
            "operator",
            "axes_by_site",
            "ensemble",
            "response",
            "mapping",
            "spin",
            "required_gates",
            "requires_plusu",
            "producer",
        ),
        ("displacement", "gate_reports"),
    )
    if manifest["version"] != 1:
        raise WorkflowRefusal("manifest version: only version 1 is supported")
    _keys(
        manifest["operator"], ("mode", "orbitals_by_site"), where="operator provenance"
    )
    operator = manifest["operator"]
    if (
        operator["mode"] not in ("onsite_delta", "weighted_total_h")
        or not operator["orbitals_by_site"]
    ):
        raise WorkflowRefusal(
            "missing_operator_provenance: declare operator.mode and nonempty "
            "operator.orbitals_by_site; the CLI never guesses orbital ownership"
        )
    _keys(
        manifest["response"],
        ("sampling", "finite_steps"),
        ("contour", "contour_node_rungs"),
        where="response",
    )
    if manifest["response"]["sampling"] != "recorded":
        raise WorkflowRefusal(
            "unsupported sampling: use response.sampling='recorded'; provide a new "
            "producer-certified bundle to change the k mesh"
        )
    steps = manifest["response"]["finite_steps"]
    if (
        not isinstance(steps, list)
        or len(steps) < 2
        or any(
            not isinstance(step, (float, int)) or not 0 < step < float("inf")
            for step in steps
        )
    ):
        raise WorkflowRefusal(
            "AC-005: supply at least two positive finite_steps in radians"
        )
    # AC-005 needs two *distinct* rotation points; a duplicated step is a
    # mislabeled single-point ladder and cannot support a cross-check.
    _STEP_DUPLICATE_TOL_RAD = 1e-12
    duplicates = sorted(
        {
            tuple(sorted((steps[i], steps[j])))
            for i in range(len(steps))
            for j in range(i + 1, len(steps))
            if abs(steps[i] - steps[j]) <= _STEP_DUPLICATE_TOL_RAD
        }
    )
    if duplicates:
        shown = ", ".join(f"{lo!r}/{hi!r}" for lo, hi in duplicates)
        raise WorkflowRefusal(
            "AC-005: finite_steps must contain at least two distinct rotation "
            f"angles; duplicate step(s) within {_STEP_DUPLICATE_TOL_RAD:g} rad: {shown}"
        )
    gates = manifest["required_gates"]
    supported = {"AC-003", "AC-005", "AC-007", "AC-008", "AC-009"}
    if (
        not isinstance(gates, list)
        or not {"AC-003", "AC-005", "AC-007"} <= set(gates)
        or set(gates) - supported
    ):
        raise WorkflowRefusal(
            "required_gates must include AC-003, AC-005, AC-007; only AC-008/AC-009 may be added"
        )
    if not isinstance(manifest["requires_plusu"], bool):
        raise WorkflowRefusal("requires_plusu must explicitly declare true or false")
    if manifest["requires_plusu"] or "AC-008" in gates:
        raise WorkflowRefusal(
            "AC-008: certified Hubbard ingestion is unsupported by the CLI NPZ loader; "
            "use the Python ingest_dftu_run/run_ac008_gate/build_exchange_export workflow; "
            "do not relabel a +U calculation as XC-only"
        )
    if "displacement" in manifest:
        _keys(
            manifest["displacement"],
            ("pattern", "gauge", "reference_fermi_ev"),
            where="displacement",
        )
        if manifest["displacement"]["gauge"] != "periodic-image-sum":
            raise WorkflowRefusal(
                "unsupported localized gauge: AC-009 needs displaced-projector dP endpoints; "
                "the supported displaced-SCF path certifies only periodic-image-sum"
            )
        if "AC-009" not in gates:
            raise WorkflowRefusal("displacement requires AC-009 in required_gates")
        if operator["mode"] != "onsite_delta":
            raise WorkflowRefusal(
                "AC-009: weighted_total_h partition variants under displacement are "
                "unsupported; use a certified onsite_delta fixed-partition workflow"
            )
    elif "AC-009" in gates:
        raise WorkflowRefusal(
            "AC-009 requires displacement settings and matching +/- HSX inputs"
        )
    _keys(manifest["spin"], ("spinat", "charges", "index_spin"), where="spin")
    _keys(
        manifest["mapping"],
        ("reference_directions_by_site",),
        ("chi2_floor_eV_per_rad2", "torque_chi2_floor_eV_per_rad"),
        where="mapping",
    )
    _keys(
        manifest["ensemble"],
        (
            "kind",
            "n_occupied",
            "electron_count",
            "fermi_energy_ev",
            "smearing_width_ev",
            "gap_ev",
        ),
        ("declaration",),
        where="ensemble",
    )
    required_inputs = {"bundle", "structure"}
    producer = manifest["producer"]
    _keys(
        producer,
        (
            "kind",
            "software_revisions",
            "pseudopotentials",
            "basis_definitions",
            "numerical_settings",
            "input_files",
        ),
        where="producer",
    )
    if producer["kind"] not in ("siesta", "analytic"):
        raise WorkflowRefusal(
            "producer.kind: declare 'siesta' or an explicitly analytic model"
        )
    revisions = producer["software_revisions"]
    if (
        not isinstance(revisions, dict)
        or not revisions
        or any(
            not isinstance(value, str) or not value.strip()
            for value in revisions.values()
        )
    ):
        raise WorkflowRefusal(
            "producer.software_revisions: record the actual nonempty producer revisions"
        )
    if (
        not isinstance(producer["numerical_settings"], dict)
        or not producer["numerical_settings"]
    ):
        raise WorkflowRefusal(
            "producer.numerical_settings: record the actual producer numerical settings"
        )
    for field in ("pseudopotentials", "basis_definitions", "input_files"):
        roles = producer[field]
        if (
            not isinstance(roles, list)
            or any(not isinstance(role, str) or not role for role in roles)
            or (
                not roles
                and (field != "pseudopotentials" or producer["kind"] != "analytic")
            )
        ):
            raise WorkflowRefusal(
                f"producer.{field}: list the digest-pinned input roles"
            )
        required_inputs.update(roles)
    if "displacement" in manifest:
        required_inputs.update(("displacement_plus", "displacement_minus"))
    reports = manifest.get("gate_reports", [])
    if not isinstance(reports, list) or any(
        not isinstance(role, str) for role in reports
    ):
        raise WorkflowRefusal("gate_reports must list input role names")
    required_inputs.update(reports)
    _keys(manifest["inputs"], required_inputs, where="inputs")
    for role, descriptor in manifest["inputs"].items():
        _keys(descriptor, ("path", "sha256"), where=f"inputs.{role}")
        if role in ("displacement_plus", "displacement_minus") and not descriptor[
            "path"
        ].lower().endswith((".hsx", ".hsx.gz")):
            raise WorkflowRefusal(
                f"inputs.{role}: {descriptor['path']} is not a displaced SIESTA HSX "
                "producer interface; supply the actual +/- HSX exports, not NPZ endpoints"
            )
        digest = descriptor["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise WorkflowRefusal(
                f"inputs.{role}.sha256: record the lowercase SHA-256 of the input bytes"
            )


def _snapshot_inputs(manifest, base, directory):
    """Consume only immutable digest-checked copies, including HSX endpoints."""
    paths = {}
    for index, (role, descriptor) in enumerate(manifest["inputs"].items()):
        source = base / descriptor["path"]
        data = source.read_bytes()
        if _sha256(data) != descriptor["sha256"]:
            raise WorkflowRefusal(
                f"inputs.{role}: SHA-256 digest mismatch for {source}; restore the "
                "recorded input or regenerate the manifest from the intended input"
            )
        target = directory / f"{index}-{source.name}"
        target.write_bytes(data)
        paths[role] = target
    return paths


def _check_spin(manifest, atoms, bundle, site_order):
    import numpy as np
    from ase.units import Bohr

    spin = manifest["spin"]
    spinat = np.asarray(spin["spinat"], dtype=float)
    charges = np.asarray(spin["charges"], dtype=float)
    indices = spin["index_spin"]
    if (
        spinat.shape != (len(atoms), 3)
        or charges.shape != (len(atoms),)
        or not np.isfinite(spinat).all()
        or not np.isfinite(charges).all()
        or len(indices) != len(atoms)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < -1
            for index in indices
        )
        or sorted(index for index in indices if index >= 0)
        != list(range(len(site_order)))
    ):
        raise WorkflowRefusal(
            "spin: provide finite per-atom spinat/charges and a one-to-one index_spin "
            "mapping (-1 for nonmagnetic atoms), in response site_order"
        )
    if not np.allclose(atoms.cell.array / Bohr, bundle.cell, atol=1e-6, rtol=0):
        raise WorkflowRefusal(
            "structure cell differs from bundle: supply the matching producer geometry"
        )
    ownership = manifest["operator"]["orbitals_by_site"]
    if sorted(
        orbital for orbitals in ownership.values() for orbital in orbitals
    ) != list(range(bundle.norb)):
        raise WorkflowRefusal(
            "operator provenance: declare every bundle orbital exactly once"
        )
    references = manifest["mapping"]["reference_directions_by_site"]
    for ispin, site in enumerate(site_order):
        atom = indices.index(ispin)
        positions = np.asarray(bundle.orbital_positions)[ownership[site]]
        if not np.allclose(positions, atoms.positions[atom] / Bohr, atol=1e-6, rtol=0):
            raise WorkflowRefusal(
                f"structure: orbital centers for site {site} do not match index_spin"
            )
        direction = np.asarray(references[site], dtype=float)
        moment = spinat[atom]
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or np.linalg.norm(direction) == 0
            or np.linalg.norm(moment) == 0
            or not np.allclose(
                moment / np.linalg.norm(moment),
                direction / np.linalg.norm(direction),
                atol=1e-8,
                rtol=0,
            )
        ):
            raise WorkflowRefusal(
                f"spin: nonzero moment direction for {site} must match mapping reference"
            )
    return spinat, charges, indices


def _pipeline(manifest, paths):
    import numpy as np
    from ase.io import read

    from TB2J.interfaces.siesta_spiral_api import build_exchange_export
    from TB2J.interfaces.siesta_spiral_displacement import (
        DisplacedReferenceFDProvider,
        DisplacementPattern,
    )
    from TB2J.interfaces.siesta_spiral_io import (
        KB_EV,
        load_fixture,
        run_eigenvalue_oracle,
        run_electron_count_oracle,
    )
    from TB2J.interfaces.siesta_spiral_mapping import map_response_set
    from TB2J.interfaces.siesta_spiral_metallic import evaluate_metallic_response_set
    from TB2J.interfaces.siesta_spiral_response import (
        CURVATURE_BUDGET_EV_PER_RAD2,
        ENSEMBLE_GAPPED_T0,
        ENSEMBLE_METALLIC_FD,
        TORQUE_BUDGET_EV_PER_RAD,
        EnergyContour,
        Ensemble,
        FrozenResponseProblem,
        evaluate_response_set,
    )
    from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex
    from TB2J.interfaces.siesta_spiral_vertex_displacement import (
        run_ac009_validation,
        vertex_displacement_derivative,
    )

    bundle, reference = load_fixture(paths["bundle"])
    if "displacement" in manifest:
        # The NPZ reader deliberately does not infer a displacement observable.
        # Preserve the explicit manifest declaration for both independent routes.
        bundle.metadata["displacement_gauge"] = manifest["displacement"]["gauge"]
    ensemble = Ensemble(**manifest["ensemble"])
    if ensemble.kind == ENSEMBLE_METALLIC_FD:
        # The AC-003 oracle occupies with the producer temperature; the
        # metallic response must smear with the *same* width. Fail closed
        # rather than assume a width the producer never recorded.
        temperature = reference.get("temperature")
        if (
            not isinstance(temperature, (float, int))
            or float(temperature) <= 0
            or not float(temperature) < float("inf")
        ):
            raise WorkflowRefusal(
                "metallic smearing consistency: the producer bundle records no "
                "positive electronic temperature; refuse to assume a smearing "
                "width it did not record"
            )
        producer_width = float(temperature) * KB_EV
        if abs(float(ensemble.smearing_width_ev) - producer_width) > 1e-10:
            raise WorkflowRefusal(
                "metallic smearing consistency: ensemble.smearing_width_ev "
                f"{float(ensemble.smearing_width_ev)!r} eV does not match the "
                f"producer temperature-derived width {producer_width!r} eV "
                "(tolerance 1e-10 eV); align it with the recorded temperature"
            )
    if not np.isclose(
        ensemble.electron_count, reference["electron_count"], atol=1e-8, rtol=0
    ) or not np.isclose(ensemble.fermi_energy_ev, reference["ef"], atol=1e-8, rtol=0):
        raise WorkflowRefusal(
            "incompatible_ensemble: electron_count/Fermi energy must match the "
            "recorded producer reference; use a matched bundle and ensemble"
        )
    eigen = run_eigenvalue_oracle(bundle, reference, strict=True)
    count = run_electron_count_oracle(bundle, reference, strict=True)
    gates = [
        {
            "gate": "AC-003",
            "passed": True,
            "eigenvalue_oracle": asdict(eigen),
            "electron_count_oracle": asdict(count),
            "declared_ensemble": ensemble.to_record(),
        }
    ]
    operator = manifest["operator"]
    vertex = extract_magnetic_vertex(
        bundle, orbitals_by_site=operator["orbitals_by_site"], operator=operator["mode"]
    )
    axes = {
        site: list(values.items()) for site, values in manifest["axes_by_site"].items()
    }
    problem = FrozenResponseProblem(bundle, vertex, axes_by_site=axes)
    atoms = read(paths["structure"])
    spinat, charges, indices = _check_spin(manifest, atoms, bundle, problem.site_order)
    settings = manifest["response"]
    kwargs = dict(
        ensemble=ensemble,
        coords=[
            (site, label)
            for site in problem.site_order
            for label in manifest["axes_by_site"][site]
        ],
        kpoints=reference["kp"],
        k_weights=reference["kp_weight"],
        finite_steps=settings["finite_steps"],
    )
    if ensemble.kind == ENSEMBLE_GAPPED_T0:
        if "contour" not in settings:
            raise WorkflowRefusal(
                "gapped ensemble requires response.contour; declare the occupied-pole contour"
            )
        response = evaluate_response_set(
            problem,
            **kwargs,
            contour=EnergyContour(**settings["contour"]),
            contour_node_rungs=settings.get("contour_node_rungs", ()),
        )
    else:
        if "contour" in settings or "contour_node_rungs" in settings:
            raise WorkflowRefusal(
                "metallic ensemble: remove gapped contour settings; the fixed-N evaluator has no contour"
            )
        response = evaluate_metallic_response_set(problem, **kwargs)
    cross = dict(response.finite_rotation_cross_check)
    budgets = {
        "torque": TORQUE_BUDGET_EV_PER_RAD,
        "curvature": CURVATURE_BUDGET_EV_PER_RAD2,
    }
    passed = all(
        isinstance(cross.get(f"{field}_max_abs_mismatch"), (float, int))
        and 0 <= cross[f"{field}_max_abs_mismatch"] <= budget
        for field, budget in budgets.items()
    )
    finite = response.ladder.get("finite_rotation", [])
    passed = (
        passed
        and len(finite) >= 2
        and all(
            finite[-1].get(f"{field}_max_delta_vs_previous") is not None
            and finite[-1][f"{field}_max_delta_vs_previous"] <= budget
            for field, budget in budgets.items()
        )
    )
    cross["passed"] = bool(passed)
    response = replace(response, finite_rotation_cross_check=cross)
    gates.append({"gate": "AC-005", **cross, "declared_ensemble": response.ensemble})
    mapping = map_response_set(response, **manifest["mapping"])
    gates.append(
        {
            "gate": "AC-007",
            "passed": mapping.compatibility["label"] == "heisenberg_compatible",
            "compatibility": mapping.compatibility,
            "declared_ensemble": response.ensemble,
        }
    )
    derivative, ac009 = None, None
    if "displacement" in manifest:
        settings = manifest["displacement"]
        pattern = DisplacementPattern(**settings["pattern"])
        fd = DisplacedReferenceFDProvider(
            pattern=pattern,
            plus_path=paths["displacement_plus"],
            minus_path=paths["displacement_minus"],
            reference_fermi_ev=settings["reference_fermi_ev"],
            reference_bundle=bundle,
        ).provide()
        derivative = vertex_displacement_derivative(
            fd,
            bundle,
            orbitals_by_site=operator["orbitals_by_site"],
            operator=operator["mode"],
        )
        ac009 = run_ac009_validation(
            bundle,
            pattern,
            plus_path=paths["displacement_plus"],
            minus_path=paths["displacement_minus"],
            reference_fermi_ev=settings["reference_fermi_ev"],
            orbitals_by_site=operator["orbitals_by_site"],
            operator=operator["mode"],
            gauge=settings["gauge"],
        )
    gates.extend(
        _read_json(paths[role].read_bytes())
        for role in manifest.get("gate_reports", [])
    )
    result = build_exchange_export(
        response_set=response,
        mapping_output=mapping,
        displacement=derivative,
        ac009_verdict=ac009,
        gate_verdicts=gates,
        requires_plusu=manifest["requires_plusu"],
    )
    if not result["spinio_kwargs"]["exchange_Jdict"]:
        raise WorkflowRefusal(
            "AC-007: no identifiable signed exchange pairs; supply certified reference directions"
        )
    return result, atoms, spinat, charges, indices


def _code_sha256():
    package = Path(__file__).resolve().parents[1]
    paths = sorted((package / "interfaces").glob("siesta_spiral*.py"))
    paths.extend((Path(__file__).resolve(), package / "io_exchange" / "io_exchange.py"))
    return _sha256(
        b"".join(
            path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes()
            for path in paths
        )
    )


def _publish(stage, output):
    """Atomically publish a complete directory without ever replacing a target."""
    import ctypes
    import os

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if not sys.platform.startswith("linux") or rename is None:
        raise WorkflowRefusal(
            "atomic no-replace publication requires Linux renameat2; "
            "no output was published"
        )
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. Unlike Path.rename, this cannot
    # replace an empty directory created by another process after our check.
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1):
        error = ctypes.get_errno()
        raise OSError(
            error,
            f"atomic no-replace publication refused: {os.strerror(error)}",
            str(output),
        )


def _run(manifest_path, output):
    from TB2J.io_exchange import SpinIO

    if output.exists() or output.is_symlink():
        raise WorkflowRefusal(
            f"output {output} already exists; choose a new directory (existing data are never overwritten)"
        )
    if not output.parent.is_dir():
        raise WorkflowRefusal(
            f"output parent {output.parent} does not exist; create it first"
        )
    data = manifest_path.read_bytes()
    manifest = _read_json(data)
    _validate_manifest(manifest)
    with tempfile.TemporaryDirectory(prefix="tb2j-spiral-inputs-") as input_directory:
        paths = _snapshot_inputs(manifest, manifest_path.parent, Path(input_directory))
        result, atoms, spinat, charges, indices = _pipeline(manifest, paths)
    kwargs = result.pop("spinio_kwargs")
    result["cli"] = {
        "manifest_sha256": _sha256(data),
        "input_digests": {
            role: record["sha256"] for role, record in manifest["inputs"].items()
        },
        "settings": manifest,
        "producer": manifest["producer"],
        "settings_sha256": _sha256(
            _canonical({k: v for k, v in manifest.items() if k != "inputs"})
        ),
        "code_revision": result["provenance"]["code_revision"],
        "code_sha256": _code_sha256(),
    }
    stable = _read_json(_canonical(result))
    del stable["provenance"]["created_utc"]
    del stable["provenance"]["content_sha256"]
    result["cli"]["reproducibility_sha256"] = _sha256(_canonical(stable))
    kwargs["debug_dict"] = result
    spinio = SpinIO(
        atoms=atoms, spinat=spinat, charges=charges, index_spin=indices, **kwargs
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as stage_name:
        stage = Path(stage_name) / "publish"
        stage.mkdir()
        spinio.write_pickle(path=str(stage))
        loaded = SpinIO.load_pickle(path=str(stage))
        if (
            loaded.exchange_Jdict != spinio.exchange_Jdict
            or loaded.debug_dict != result
        ):
            raise WorkflowRefusal(
                "SpinIO writer round-trip failed; no output published"
            )
        (stage / "provenance.json").write_bytes(_canonical(result) + b"\n")
        run = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": {
                name: _sha256((stage / name).read_bytes())
                for name in ("TB2J.pickle", "provenance.json")
            },
        }
        (stage / "run.json").write_bytes(_canonical(run) + b"\n")
        _publish(stage, output)
    print(
        f"Certified SpinIO export: {output} (reproducibility_sha256={result['cli']['reproducibility_sha256']})"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Certified SIESTA spiral workflow from a digest-pinned JSON manifest"
    )
    parser.add_argument("manifest", help="JSON workflow manifest (version 1)")
    parser.add_argument(
        "--output", required=True, help="New output directory; never overwritten"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        _run(Path(args.manifest).resolve(), Path(args.output).absolute())
    except Exception as exc:
        # KeyboardInterrupt/SystemExit deliberately escape this fail-closed boundary.
        print(f"tb2j-siesta-spiral: refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
