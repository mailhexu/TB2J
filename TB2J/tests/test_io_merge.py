import numpy as np

from TB2J.io_merge import get_projections


def test_get_projections_can_treat_weak_canting_as_collinear():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([1e-3, 0.0, 1.0])
    b /= np.linalg.norm(b)

    noncollinear_proj = get_projections(a, b, tol=1e-6)
    collinear_proj = get_projections(a, b, tol=1e-2)

    assert np.allclose(noncollinear_proj[1], np.zeros(3))
    assert np.linalg.norm(collinear_proj[0]) > 0.0
    assert np.linalg.norm(collinear_proj[1]) > 0.0
