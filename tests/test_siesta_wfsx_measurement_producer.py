"""TRS partner merging must stay faithful and keep the oracle view unique."""

import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent / "siesta_wfsx_fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from fixture_driver import FixtureError  # noqa: E402
from measurement_producer import (  # noqa: E402
    _require_unique_union_matches,
    _trs_merge_partners,
)


def _case_rows():
    # Rows 1 and 2 are the +/-half-boundary TRS partner double-keep SIESTA
    # leaves in even-mesh irreducible sets; rows 0, 3, 4 are ordinary.
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.5, -0.4, 0.0],
            [0.5, 0.4, 0.0],
            [-0.3, 0.1, 0.4],
            [0.4, 0.4, 0.4],
            [0.5, 0.0, 0.0],
        ]
    )
    weights = np.array([0.2, 0.125, 0.125, 0.25, 0.3, 0.1])
    return points, weights


def test_partner_pairs_merge_with_summed_weights_and_kept_order():
    points, weights = _case_rows()
    kept, merged_points, merged_weights, pairs, stats = _trs_merge_partners(
        points, weights
    )
    assert pairs == [(1, 2)]
    assert kept.tolist() == [0, 1, 3, 4, 5]
    assert merged_points.tolist() == points[kept].tolist()
    assert merged_weights.tolist() == [0.2, 0.25, 0.25, 0.3, 0.1]
    assert stats["partner_pairs"] == 1
    assert stats["original_kpoint_count"] == 6
    assert stats["merged_kpoint_count"] == 5
    assert stats["max_partner_weight_asymmetry"] == 0.0
    assert stats["weight_sum_absolute_error"] == pytest.approx(0.0, abs=1e-15)
    # The merged view must give every original row exactly one canonical match.
    _require_unique_union_matches(merged_points, points)


def test_self_inverse_rows_stay_alone_and_dropped_partners_vanish():
    points, weights = _case_rows()
    kept, merged_points, _, pairs, _ = _trs_merge_partners(points, weights)
    # The self-inverse boundary row (index 5) is never paired, and the dropped
    # partner (index 2) must not survive as a duplicate class.
    assert all(5 not in pair for pair in pairs)
    assert 2 not in kept.tolist()
    _require_unique_union_matches(merged_points, points)


def test_ambiguous_view_is_refused():
    points, _ = _case_rows()
    duplicated = np.vstack([points, points[0]])
    with pytest.raises(FixtureError) as error:
        _require_unique_union_matches(duplicated, duplicated)
    assert error.value.code == "KP_TRS_AMBIGUOUS"
