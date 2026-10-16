"""Internal Story-022 report helpers; not part of the installed consumer API.

The seam accepts CLI ``provenance.json`` curvature mappings and caller-owned
rung observations. It performs only numeric comparison/report assembly; it
never computes responses or changes the production vertex.
"""

import math

BASIS_RELATIVE_THRESHOLD = 0.05
GRID_RELATIVE_THRESHOLD = 0.02
KMESH_RELATIVE_THRESHOLD = 0.02
CONTOUR_ABSOLUTE_THRESHOLD_EV_PER_RAD2 = 1.0e-4


def _curvature_values(mapping):
    if not hasattr(mapping, "keys"):
        raise TypeError("curvature must be a mapping")
    values = {}
    for key, value in mapping.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"curvature value for {key!r} is not finite")
        values[key] = number
    return values


def compare_response_curvatures(reference, refined, *, relative):
    """Compare CLI curvature mappings with identical finite component keys.

    Relative response change uses the Story-022 normalization
    ``max(abs(refined-reference)) / max(1e-12, max(abs(refined)))``.
    With ``relative=False`` the returned maximum absolute delta is in
    eV/rad^2, for contour-integration convergence.
    """
    reference_values = _curvature_values(reference)
    refined_values = _curvature_values(refined)
    if reference_values.keys() != refined_values.keys():
        raise ValueError("curvature mappings must have identical component keys")
    if not reference_values:
        raise ValueError("curvature mappings must not be empty")
    maximum_delta = max(
        abs(reference_values[key] - refined_values[key]) for key in reference_values
    )
    if not relative:
        return maximum_delta
    scale = max(1.0e-12, max(abs(value) for value in refined_values.values()))
    return maximum_delta / scale


def build_convergence_ladder(rungs, *, metric, threshold, units):
    """Build one independent convergence ladder from ordered rung records.

    Each input rung has ``rung_id``, a complete JSON-safe ``observation``
    (including its ``curvature`` mapping), and an optional additional
    ``metadata`` mapping. Adjacent comparisons are stored on the later rung;
    the first rung has no adjacent metric and therefore no individual verdict.
    """
    if metric not in ("relative_response", "absolute_response"):
        raise ValueError("metric must be relative_response or absolute_response")
    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if len(rungs) < 2:
        raise ValueError("a convergence ladder requires at least two rungs")
    relative = metric == "relative_response"
    entries = []
    previous_curvature = None
    for rung in rungs:
        observation = dict(rung["observation"])
        curvature = observation["curvature"]
        change = (
            None
            if previous_curvature is None
            else compare_response_curvatures(
                previous_curvature, curvature, relative=relative
            )
        )
        entries.append(
            {
                "rung_id": rung["rung_id"],
                "observation": observation,
                "adjacent_change": change,
                "units": units,
                "threshold": float(threshold),
                "passed": None if change is None else bool(change <= threshold),
            }
        )
        previous_curvature = curvature
    return {
        "metric": metric,
        "units": units,
        "threshold": float(threshold),
        "rungs": entries,
        "passed": all(entry["passed"] is not False for entry in entries[1:]),
    }


def assemble_campaign_report(
    *,
    basis,
    grid,
    kmesh,
    response_integration,
    derivative_agreement,
    partition_sensitivity,
):
    """Assemble JSON-serializable Story-022 evidence with isolated axes.

    Convergence accepts only the four named ladder reports; derivative and
    partition evidence are top-level sibling sections and cannot affect any
    ladder metric or verdict. Inputs are copied into fresh JSON-compatible
    containers so the result has no aliases to caller-owned mappings.
    """
    import json

    convergence = {
        "basis": basis,
        "grid": grid,
        "kmesh": kmesh,
        "response_integration": response_integration,
    }
    return json.loads(
        json.dumps(
            {
                "convergence_ladders": convergence,
                "derivative_agreement": derivative_agreement,
                "partition_sensitivity": partition_sensitivity,
            },
            allow_nan=False,
        )
    )


def story022_convergence_ladders(
    *, basis_rungs, grid_rungs, kmesh_rungs, response_integration_rungs
):
    """Build all fixed-budget Story-022 ladders from ordered rung observations."""
    return {
        "basis": build_convergence_ladder(
            basis_rungs,
            metric="relative_response",
            threshold=BASIS_RELATIVE_THRESHOLD,
            units="relative",
        ),
        "grid": build_convergence_ladder(
            grid_rungs,
            metric="relative_response",
            threshold=GRID_RELATIVE_THRESHOLD,
            units="relative",
        ),
        "kmesh": build_convergence_ladder(
            kmesh_rungs,
            metric="relative_response",
            threshold=KMESH_RELATIVE_THRESHOLD,
            units="relative",
        ),
        "response_integration": build_convergence_ladder(
            response_integration_rungs,
            metric="absolute_response",
            threshold=CONTOUR_ABSOLUTE_THRESHOLD_EV_PER_RAD2,
            units="eV/rad^2",
        ),
    }
