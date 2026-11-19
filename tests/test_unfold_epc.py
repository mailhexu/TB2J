import numpy as np

from TB2J.interfaces.unfold_epc import (
    exchange_product_derivative,
    load_unfold_epc,
    mixed_green_response,
)


def test_nonzero_q_store_resolvent_matches_independent_real_pattern_fd(tmp_path):
    from unfold_epc.store import SparseDerivative, TwoIndexStore
    from unfold_epc.transform import transform

    n = 3
    cells = np.column_stack((np.arange(n), np.zeros((n, 2), dtype=int)))
    row, col = np.indices((n, n))
    local_h = np.array(
        [[0.7, 0.2 + 0.03j, -0.1j], [0.2 - 0.03j, -0.2, 0.04], [0.1j, 0.04, 0.1]]
    )
    local_s = np.array(
        [[0.03, 0.01j, 0.005], [-0.01j, -0.02, 0.004], [0.005, 0.004, 0.01]]
    )
    block = SparseDerivative(
        row.ravel(),
        col.ravel(),
        cells[row.ravel()],
        cells[col.ravel()],
        local_h.reshape(-1, 1),
        local_s.ravel(),
    )
    store = TwoIndexStore(
        np.eye(3),
        np.diag([3, 1, 1]),
        np.zeros(n, dtype=int),
        cells,
        1,
        1,
        1,
        1.3,
        0.005,
        {(0, 0): block},
        {"input_hashes": {}},
    )
    path = tmp_path / "epc.nc"
    store.write(path)
    store = load_unfold_epc(path)
    k, q = np.array([0.0, 0.0, 0.0]), np.array([1 / 3, 0.0, 0.0])
    dh, ds = transform(store, k, q, atom=0, axis=0)
    h0 = np.eye(n) * 0.5 + 0.12 * (np.ones((n, n)) - np.eye(n))
    s0 = np.eye(n) + 0.04 * (np.ones((n, n)) - np.eye(n))
    z, ef = 0.9 + 0.4j, store.reference_fermi
    uk = np.exp(2j * np.pi * (cells @ k)) / np.sqrt(n)
    uq = np.exp(2j * np.pi * (cells @ (k + q))) / np.sqrt(n)
    g = np.linalg.inv((z + ef) * s0 - h0)
    gk, gq = np.vdot(uk, g @ uk), np.vdot(uq, g @ uq)
    actual = mixed_green_response(np.array([[gk]]), np.array([[gq]]), dh[0], ds, z, ef)[
        0, 0
    ]
    cos_h, sin_h, cos_s, sin_s = (np.zeros((n, n), complex) for _ in range(4))
    for image in range(n):
        translated_h = np.roll(np.roll(local_h, image, axis=0), image, axis=1)
        translated_s = np.roll(np.roll(local_s, image, axis=0), image, axis=1)
        angle = 2 * np.pi * image / 3
        cos_h += np.cos(angle) * translated_h
        sin_h += np.sin(angle) * translated_h
        cos_s += np.cos(angle) * translated_s
        sin_s += np.sin(angle) * translated_s
    step = 1e-6

    def fd(dh, ds):
        plus = np.linalg.inv((z + ef) * (s0 + step * ds) - (h0 + step * dh))
        minus = np.linalg.inv((z + ef) * (s0 - step * ds) - (h0 - step * dh))
        return np.vdot(uk, ((plus - minus) / (2 * step)) @ uq)

    expected = fd(cos_h, cos_s) - 1j * fd(sin_h, sin_s)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    wrong = mixed_green_response(
        np.array([[gk]]), np.array([[gq]]), dh[0], ds, z, ef, drop_overlap=True
    )[0, 0]
    assert abs(wrong - expected) > 1e-3


def test_both_endpoint_splittings_and_oriented_green_terms():
    rng = np.random.default_rng(7)
    values = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(8)]
    pi, gij, pj, gji, dpi, dgij, dpj, dgji = values
    # A finite-q second endpoint has a nontrivial complex Bloch phase.
    dpj *= np.exp(-2j * np.pi / 3)
    actual = exchange_product_derivative(pi, gij, pj, gji, dpi, dgij, dpj, dgji)
    step = 1e-6

    def product(sign):
        return np.trace(
            (pi + sign * step * dpi)
            @ (gij + sign * step * dgij)
            @ (pj + sign * step * dpj)
            @ (gji + sign * step * dgji)
        )

    expected = (product(1) - product(-1)) / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
