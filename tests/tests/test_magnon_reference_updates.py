"""H(q) must use the current public exchange and reference state."""

import numpy as np

from TB2J.magnon.magnon3 import Magnon


def chain():
    model = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0.0, 2.0]]),
        Rlist=np.array([[1, 0, 0], [-1, 0, 0]]),
        JR=np.tile(np.eye(3), (2, 1, 1, 1, 1)),
        cell=np.eye(3),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 0.0, 1.0]),
    )
    model.set_reference(np.zeros(3), [[0.0, 0.0, 1.0]], [0.0, 0.0, 1.0])
    return model


def test_goldstone_survives_moment_and_exchange_updates():
    model = chain()
    q = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    initial = model.Hq(q)
    model.set_reference(
        np.zeros(3), [[0.0, 0.0, 1.0]], [0.0, 0.0, 1.0], magmoms=[[0.0, 0.0, 4.0]]
    )
    updated = model.Hq(q)
    np.testing.assert_allclose(updated[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(updated[1], initial[1] / 2, atol=1e-14)
    model.JR *= 1.5
    np.testing.assert_allclose(model.Hq(q), updated * 1.5, atol=1e-14)


def test_reference_axis_and_propagation_updates_match_fresh_model():
    model = chain()
    q = np.array([[0.13, 0.05, 0.0]])
    model.Hq(q)
    fresh = chain()
    for candidate in (model, fresh):
        candidate.set_reference([0.2, 0.0, 0.0], [[0.0, 0.0, 1.0]], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(model.Hq(q), fresh.Hq(q), atol=1e-14)
