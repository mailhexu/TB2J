"""Scientific refinement must not hide unaligned tensor elements."""

import numpy as np
import pytest
from siesta_wfsx_fixtures.run_convergence import ConvergenceError, _refinement_step


def test_refinement_refuses_changed_full_tensor_domain():
    onsite = {"R": [0, 0, 0], "i": 0, "j": 0, "value": np.eye(3).tolist()}
    extra = {"R": [0, 0, 1], "i": 0, "j": 0, "value": (2 * np.eye(3)).tolist()}
    # The shared onsite block is identical, but a whole nonzero fine-mesh
    # interaction has no coarse counterpart. An intersection would falsely
    # report zero change and convergence instead of refusing the domain shift.
    with pytest.raises(ConvergenceError) as error:
        _refinement_step([onsite], [onsite, extra], [5, 5, 5], [10, 10, 10])
    assert error.value.code == "TENSOR_DOMAIN_CHANGED"
