"""Certified SIESTA spiral consumer API (story-020, ADR-009).

``build_exchange_export`` consumes ResponseSet, MappingOutput, optional
DisplacementDerivatives/VertexDisplacementDerivatives and the verdict dictionaries
returned by run_ac008_gate/run_ac009_validation. No files are written. The
returned dictionary contains the preserved raw response, complete mapping record
(conventions, identifiability, residuals included), units, reference and operator
metadata, ensemble declarations, convergence evidence and SHA-256 provenance.

``export['spinio_kwargs']`` uses the EXISTING SpinIO constructor, not a new IO
schema. Supply your own atoms, spinat, charges and index_spin to
``SpinIO(**export['spinio_kwargs'], atoms=..., spinat=..., charges=...,
index_spin=...)``. Spin indices follow reference site_order. The exchange keys
are (R, i, j), R=(0,0,0), for the intra-reference site pairs, both ordered
directions at j_stored_eV (half the pair-once coupling). This is a local-reference
mapping, NOT a real-space Fourier inversion. Partially identifiable chart DMI
and Gamma are preserved in mapping metadata, never passed off as full Cartesian
SpinIO tensors. dH/dR, dS/dR and dV/dR are NOT dJ/dx; SpinIO.dJdx remains unset.
Structure and moment magnitudes are never invented. ``debug_dict`` carries the
same certified metadata, so existing SpinIO pickling preserves it.

``ProvenanceSchema`` is a frozen record of input_digests (SHA-256 of every
consumed artifact payload), code_revision (package git HEAD, or a SHA-256 of
module digests outside a checkout), created_utc (UTC ISO-8601), and gate_verdicts
(the consumed verdict chain). to_dict/from_dict use the upstream canonical-JSON
content_sha256 convention; from_dict refuses tampering. ``content_sha256`` is
also available as a property. Nested inputs are detached on construction.

Refusals (ExportRefusal subclasses, with stable ``reason`` identifiers):
MissingOperatorProvenance / missing_operator_provenance: restore the certified
vertex/source-bundle declaration; FailedGate / failed_gate: rerun the named
consumed gate; IncompatibleEnsemble / incompatible_ensemble: use one matched
ADR-005 ensemble; NonHeisenbergCompatibleResponse /
non_heisenberg_compatible_response: obtain an AC-007 compatible mapping or
request raw_response_only; UncertifiedDisplacement / uncertified_displacement:
supply the matching AC-009 pass; UncertifiedPlusU / uncertified_plusu: supply
an AC-008 pass (required when requires_plusu=True or +U is declared).
Explicit failing AC-009/generic verdicts raise FailedGate, while absent AC-009
certification raises UncertifiedDisplacement. AC-008 refusals are always
UncertifiedPlusU. All checks precede assembly of exchange dictionaries.

``raw_response_only=True`` is an explicit diagnostic escape: ONLY the preserved
raw response and its metadata are exported, with uncertified_exchange_absent
labeling, no mapping, displacement or SpinIO parameters. Operator provenance and
ensemble consistency are still required. Failed verdicts remain visible in the
provenance chain; they cannot certify exchange through this escape. No numerical
convergence or material validation is inferred from a pipeline-consistency pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from .siesta_spiral_displacement import GAUGE_LABELS, DisplacementDerivatives
from .siesta_spiral_mapping import HEISENBERG_COMPATIBLE, MappingOutput
from .siesta_spiral_response import ResponseSet
from .siesta_spiral_vertex import OPERATOR_MODES
from .siesta_spiral_vertex_displacement import VertexDisplacementDerivatives

API_VERSION = "1.0"


class ExportRefusal(RuntimeError):
    """An export was refused; no exchange object has been constructed."""

    reason = "refused"

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.refusal_class = self.reason
        super().__init__(f"{self.reason}: {diagnostic}")


class MissingOperatorProvenance(ExportRefusal):
    reason = "missing_operator_provenance"


class FailedGate(ExportRefusal):
    reason = "failed_gate"


class IncompatibleEnsemble(ExportRefusal):
    reason = "incompatible_ensemble"


class NonHeisenbergCompatibleResponse(ExportRefusal):
    reason = "non_heisenberg_compatible_response"


class UncertifiedDisplacement(ExportRefusal):
    reason = "uncertified_displacement"


class UncertifiedPlusU(ExportRefusal):
    reason = "uncertified_plusu"


REFUSAL_CLASSES = {
    cls.reason: cls
    for cls in (
        MissingOperatorProvenance,
        FailedGate,
        IncompatibleEnsemble,
        NonHeisenbergCompatibleResponse,
        UncertifiedDisplacement,
        UncertifiedPlusU,
    )
}


def _digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_payload(value):
    """Serialize dataclass records, including the sparse tuple-keyed seam."""
    if is_dataclass(value):
        return {f.name: _json_payload(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {
            (k if isinstance(k, str) else json.dumps(_json_payload(k))): _json_payload(
                v
            )
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_payload(v) for v in value]
    return value


@dataclass(frozen=True)
class ProvenanceSchema:
    """Digest-stamped provenance; see the module API reference for fields."""

    input_digests: dict
    code_revision: str
    created_utc: str
    gate_verdicts: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "input_digests", copy.deepcopy(self.input_digests))
        object.__setattr__(
            self, "gate_verdicts", tuple(copy.deepcopy(self.gate_verdicts))
        )
        for name, digest in self.input_digests.items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name}: expected a SHA-256 payload digest")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(f"{name}: expected a SHA-256 payload digest") from exc
        stamp = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("created_utc must declare UTC")
        if not self.code_revision:
            raise ValueError("code_revision must identify package code")

    def to_dict(self):
        payload = _json_payload(self)
        payload["version"] = API_VERSION
        payload["content_sha256"] = _digest(payload)
        return payload

    @property
    def content_sha256(self):
        return self.to_dict()["content_sha256"]

    @classmethod
    def from_dict(cls, data):
        payload = {k: v for k, v in data.items() if k != "content_sha256"}
        if payload.get("version") != API_VERSION:
            raise ValueError("unsupported provenance version")
        if data.get("content_sha256") != _digest(payload):
            raise ValueError("provenance payload digest mismatch")
        return cls(
            input_digests=payload["input_digests"],
            code_revision=payload["code_revision"],
            created_utc=payload["created_utc"],
            gate_verdicts=tuple(payload["gate_verdicts"]),
        )


def _code_revision():
    package = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        modules = sorted(package.joinpath("interfaces").glob("siesta_spiral*.py"))
        modules.append(package / "io_exchange" / "io_exchange.py")
        return "module-sha256:" + _digest(
            {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in modules
            }
        )


def _gate_name(verdict):
    return verdict.get("gate", verdict.get("acceptance_criterion", "unnamed gate"))


def _passed(verdict):
    if not isinstance(verdict, dict):
        return False
    if "passed" in verdict:
        return verdict["passed"] is True and verdict.get("status", "pass") == "pass"
    return verdict.get("status") == "pass"


def _operator_audit(response, mapping, displacement):
    if response is not None:
        provenance = response.provenance
        amplitudes = response.frozen_declarations.get("magnetic_amplitudes", {})
        certificate = amplitudes.get("certificate", {})
        if (
            response.operator_mode not in OPERATOR_MODES
            or provenance.get("operator_mode") != response.operator_mode
            or certificate.get("operator_mode") != response.operator_mode
            or any(
                not provenance.get(k) for k in ("vertex_sha256", "source_bundle_sha256")
            )
        ):
            raise MissingOperatorProvenance(
                "operator provenance gate (AC-004): restore the certified vertex, "
                "source_bundle_sha256 and matching operator_mode before export"
            )
        for key in ("vertex_sha256", "source_bundle_sha256"):
            if key in amplitudes and amplitudes[key] != provenance[key]:
                raise MissingOperatorProvenance(
                    f"AC-004 operator provenance: {key} contradicts frozen declarations; "
                    "regenerate the response from the certified vertex"
                )
    if mapping is not None:
        for key in ("vertex_sha256", "source_bundle_sha256", "operator_mode"):
            if mapping.provenance.get(key) != response.provenance.get(key):
                raise MissingOperatorProvenance(
                    f"AC-007 mapping {key} does not match its raw response; remap it"
                )
        if (
            mapping.provenance.get("response_sha256")
            != response.to_dict()["content_sha256"]
        ):
            raise MissingOperatorProvenance(
                "AC-007 mapping response digest differs; remap the response"
            )
    if isinstance(displacement, VertexDisplacementDerivatives) and (
        displacement.operator_mode not in OPERATOR_MODES or not displacement.provenance
    ):
        raise MissingOperatorProvenance(
            "AC-009 vertex operator provenance is missing; regenerate the certified dV/dR"
        )


def _ensemble_audit(declarations):
    compared = {}
    for source, record in declarations.items():
        for field in (
            "kind",
            "n_occupied",
            "electron_count",
            "fermi_energy_ev",
            "gap_ev",
            "smearing_width_ev",
            "occupation_smearing_width_ev",
            "ensemble_declaration",
        ):
            if field not in record or record[field] is None:
                continue
            value = record[field]
            if field in compared and compared[field][1] != value:
                other, _ = compared[field]
                raise IncompatibleEnsemble(
                    f"ADR-005 ensemble gate: {source}.{field} disagrees with {other}; "
                    "regenerate the chain with one matched ensemble declaration"
                )
            compared[field] = (source, value)


def _declares_plusu(record):
    """Recognize explicit producer declarations, never guess from splittings."""
    if not isinstance(record, dict):
        return False
    for key, value in record.items():
        if key in ("vertex_kind", "declared_vertex_kind") and value in (
            "xc+hubbard",
            "hubbard",
        ):
            return True
        if key in ("requires_plusu", "dftu", "hubbard_provenance") and value:
            return True
        if isinstance(value, dict) and _declares_plusu(value):
            return True
    return False


def _displacement_audit(displacement, verdict):
    if verdict is None or not _passed(verdict):
        raise UncertifiedDisplacement(
            "AC-009 pass missing: run run_ac009_validation on the matched displaced "
            "references before exporting dH/dR, dS/dR or dV/dR"
        )
    pattern = _json_payload(displacement.pattern)
    declared = verdict.get("pattern", {})
    for key in ("atom_index", "axis", "step_bohr", "branch"):
        if declared.get(key) != pattern[key]:
            raise UncertifiedDisplacement(
                f"AC-009 pattern {key} does not match the derivative; rerun the matching gate"
            )
    gauges = verdict.get("gauge", {})
    if displacement.gauge not in GAUGE_LABELS or any(
        gauges.get(key) != displacement.gauge
        for key in (
            "requested",
            "seam_record",
            "vertex_derivative_record",
            "displaced_reference_route",
        )
    ):
        raise UncertifiedDisplacement(
            "AC-009 gauge mismatch: certify the same periodic-image-sum or localized "
            "observable; localized data require displaced-projector endpoints"
        )
    if isinstance(displacement, VertexDisplacementDerivatives):
        if verdict.get("operator_mode") != displacement.operator_mode:
            raise UncertifiedDisplacement(
                "AC-009 operator_mode mismatch; rerun the matching gate"
            )
    else:
        references = verdict.get("geometry", {}).get("reference_digests", ())
        if tuple(references) != tuple(displacement.reference_digests):
            raise UncertifiedDisplacement(
                "AC-009 reference digests differ; validate the same displaced pair"
            )
    agreement = verdict.get("agreement", {})
    if not agreement:
        raise UncertifiedDisplacement(
            "AC-009 agreement evidence missing; rerun the gate"
        )
    if isinstance(displacement, VertexDisplacementDerivatives):
        if set(agreement) != {str(key) for key in displacement.dV}:
            raise UncertifiedDisplacement(
                "AC-009 dV coverage differs; rerun the matching gate"
            )
        values = {str(key): value for key, value in displacement.dV.items()}
    else:
        values = {}
        for key, (d1, d2, d3, d4) in displacement.dH.items():
            if str(key) in agreement:
                v_up = (d1 - d2) / 2.0
                values[str(key)] = (v_up, -v_up, d3, d4)
        if set(values) != set(agreement):
            raise UncertifiedDisplacement(
                "AC-009 onsite coverage differs; rerun the matching gate"
            )
    if any(
        list(values[key]) != entry.get("derived") for key, entry in agreement.items()
    ):
        raise UncertifiedDisplacement(
            "AC-009 derivative payload differs from its agreement evidence; rerun the matching gate"
        )


def build_exchange_export(
    response_set=None,
    mapping_output=None,
    displacement=None,
    plusu_gate=None,
    ac009_verdict=None,
    *,
    raw_response_only=False,
    gate_verdicts=(),
    requires_plusu=False,
    created_utc=None,
):
    """Audit first, then assemble the export described in the module reference."""
    if response_set is not None and not isinstance(response_set, ResponseSet):
        raise TypeError("response_set must be a ResponseSet")
    if mapping_output is not None and not isinstance(mapping_output, MappingOutput):
        raise TypeError("mapping_output must be a MappingOutput")
    if displacement is not None and not isinstance(
        displacement, (DisplacementDerivatives, VertexDisplacementDerivatives)
    ):
        raise TypeError(
            "displacement must be a certified displacement derivative record"
        )
    response = response_set
    if mapping_output is not None:
        mapped_response = mapping_output.raw_response
        if response is not None:
            _ensemble_audit(
                {
                    "response": response.ensemble,
                    "mapping.raw_response": mapped_response.ensemble,
                }
            )
        if response is not None and response.to_dict() != mapped_response.to_dict():
            raise MissingOperatorProvenance(
                "AC-007 mapping consumes a different raw response; supply the matching ResponseSet"
            )
        response = mapped_response
    if response is None and (raw_response_only or displacement is None):
        raise ValueError("provide a ResponseSet, MappingOutput or displacement record")
    _operator_audit(response, mapping_output, displacement)
    consumed = list(gate_verdicts)
    for gate in (plusu_gate, ac009_verdict):
        if gate is not None:
            consumed.append(gate)
    if response is not None and "passed" in response.finite_rotation_cross_check:
        consumed.append({"gate": "AC-005", **response.finite_rotation_cross_check})
    for verdict in consumed:
        if not isinstance(verdict, dict):
            raise FailedGate(
                "consumed gate verdict is not a dictionary; supply its complete report"
            )
    ensembles = {}
    if response is not None:
        ensembles["response"] = response.ensemble
    for i, verdict in enumerate(consumed):
        ensembles[f"{_gate_name(verdict)}[{i}]"] = verdict.get(
            "declared_ensemble", verdict.get("ensemble", {})
        )
    _ensemble_audit(ensembles)
    if not raw_response_only:
        plusu_required = requires_plusu or (
            response is not None
            and (
                _declares_plusu(response.provenance)
                or _declares_plusu(response.frozen_declarations)
            )
        )
        if plusu_gate is not None or plusu_required:
            if (
                plusu_gate is None
                or _gate_name(plusu_gate) != "AC-008"
                or not _passed(plusu_gate)
            ):
                raise UncertifiedPlusU(
                    "AC-008 pass missing or failed: run run_ac008_gate with the complete "
                    "XC+Hubbard provenance and matched frozen rotations; "
                    f"failure reasons: {(plusu_gate or {}).get('failure_reasons', [])}"
                )
        for verdict in consumed:
            if not _passed(verdict):
                if _gate_name(verdict) == "AC-008":
                    raise UncertifiedPlusU(
                        "AC-008 consumed verdict did not pass; rerun run_ac008_gate "
                        "with the complete XC+Hubbard provenance"
                    )
                raise FailedGate(
                    f"{_gate_name(verdict)} did not pass; rerun that gate and resolve "
                    f"its evidence: {verdict.get('failure_reasons', verdict.get('status'))}"
                )
        if (
            mapping_output is not None
            and mapping_output.compatibility.get("label") != HEISENBERG_COMPATIBLE
        ):
            raise NonHeisenbergCompatibleResponse(
                "AC-007 Heisenberg compatibility gate is not passed "
                f"({mapping_output.compatibility.get('label')!r}); obtain a compatible "
                "mapping or request raw_response_only=True (no exchange parameters)"
            )
        if displacement is not None:
            if ac009_verdict is not None and _gate_name(ac009_verdict) != "AC-009":
                raise UncertifiedDisplacement(
                    "AC-009 verdict identity missing; run run_ac009_validation"
                )
            _displacement_audit(displacement, ac009_verdict)

    payloads = {}
    if response is not None:
        payloads["response_set"] = response.to_dict()
    if mapping_output is not None:
        payloads["mapping_output"] = mapping_output.to_dict()
    if displacement is not None:
        payloads["displacement"] = _json_payload(displacement)
    for i, verdict in enumerate(consumed):
        payloads[f"gate:{_gate_name(verdict)}:{i}"] = copy.deepcopy(verdict)
    input_digests = {name: _digest(value) for name, value in payloads.items()}
    provenance = ProvenanceSchema(
        input_digests=input_digests,
        code_revision=_code_revision(),
        created_utc=created_utc or datetime.now(timezone.utc).isoformat(),
        gate_verdicts=tuple(copy.deepcopy(consumed)),
    )
    result = {
        "version": API_VERSION,
        "export_kind": "raw_response_only" if raw_response_only else "certified_chain",
        "provenance": provenance.to_dict(),
        "raw_response": payloads.get("response_set"),
        "units": {
            "energy": "eV",
            "gradient": "eV/rad",
            "torque": "eV/rad",
            "curvature": "eV/rad^2",
        },
        "reference_metadata": {
            "site_order": list(response.site_order) if response is not None else [],
            "q": copy.deepcopy(response.provenance.get("q"))
            if response is not None
            else None,
            "frozen_declarations": copy.deepcopy(response.frozen_declarations)
            if response is not None
            else None,
        },
        "vertex_provenance": copy.deepcopy(response.provenance)
        if response is not None
        else None,
        "ensemble": copy.deepcopy(ensembles),
        "convergence_evidence": {
            "integration": copy.deepcopy(response.integration)
            if response is not None
            else None,
            "ladder": copy.deepcopy(response.ladder) if response is not None else None,
            "finite_rotation_cross_check": copy.deepcopy(
                response.finite_rotation_cross_check
            )
            if response is not None
            else None,
            "contour_vs_band_max_abs": response.contour_vs_band_max_abs
            if response is not None
            else None,
        },
        "labels": {
            "uncertified_exchange_absent": bool(raw_response_only),
            "exchange_scope": "intra-reference site pairs, not real-space Fourier inversion",
        },
    }
    if raw_response_only:
        return result
    if displacement is not None:
        result["displacement"] = payloads["displacement"]
        result["labels"]["gauge"] = displacement.gauge
        result["units"].update(displacement.units)
    if ac009_verdict is not None:
        result["labels"]["ac009_gauge"] = copy.deepcopy(ac009_verdict.get("gauge"))
    exchange = {}
    if mapping_output is not None:
        result["mapping"] = payloads["mapping_output"]
        site_index = {site: i for i, site in enumerate(response.site_order)}
        for pair in mapping_output.parameters.values():
            value = pair["j_stored_eV"]
            if value is not None:
                a, b = (site_index[s] for s in pair["sites"])
                exchange[((0, 0, 0), a, b)] = value
                exchange[((0, 0, 0), b, a)] = value
    result["spinio_kwargs"] = {
        "colinear": False,
        "exchange_Jdict": exchange or None,
        "dmi_ddict": None,
        "Jani_dict": None,
        "dJdx": None,
        "debug_dict": copy.deepcopy(result),
    }
    return result
