import math

import numpy as np
import pytest

from TB2J.mryasov import mryasov_downfold

Z = (0, 0, 0)


def test_signed_displaced_bonds_and_minimized_energy():
    # The ligand in cell n sees metal 0 at n+1 and metal 1 at n-1.
    # Its FM field is -1, so the normalized soft-moment energy is -h²/2.
    exchange = {
        ((1, 0, 0), 4, 0): -2.0,
        ((-1, 0, 0), 0, 4): -2.0,
        ((-1, 0, 0), 4, 2): 1.0,
        ((1, 0, 0), 2, 4): 1.0,
        ((-2, 0, 0), 0, 2): 0.25,
        ((2, 0, 0), 2, 0): 0.25,
        (Z, 4, 4): float("nan"),  # ligand-only interactions are ignored
        (Z, 0, 0): 3.0,  # a rigid-spin constant
    }
    effective = mryasov_downfold(exchange, [0, 2], [4])
    assert effective == {
        ((-2, 0, 0), 0, 2): -1.75,
        ((2, 0, 0), 2, 0): -1.75,
    }
    dressed = mryasov_downfold(exchange, [0, 2], [4], dressing=3.0)
    assert dressed[((-2, 0, 0), 0, 2)] == -5.75
    assert dressed[((2, 0, 0), 2, 0)] == -5.75

    spins = {
        0: [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)],
        2: [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0)],
    }
    aligned = {i: [(1.0, 0.0)] * 3 for i in spins}

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def minimized_energy(configuration):
        energy = 0.0
        for n in range(3):
            a = configuration[0][(n + 1) % 3]
            b = configuration[2][(n - 1) % 3]
            field = tuple(-2 * x + y for x, y in zip(a, b))
            energy += -0.5 * dot(field, field) - 0.25 * dot(a, b)
        return energy

    def effective_energy(configuration):
        return -0.5 * sum(
            value * dot(configuration[i][n], configuration[j][(n + R[0]) % 3])
            for (R, i, j), value in effective.items()
            for n in range(3)
        )

    assert minimized_energy(spins) - minimized_energy(aligned) == pytest.approx(
        effective_energy(spins) - effective_energy(aligned)
    )


def test_same_metal_distinct_cells_generates_reciprocal_translations():
    right = (np.int8(100), 0, 0)
    left = (-right[0], 0, 0)
    exchange = {
        (left, 3, 0): 2.0,
        (right, 0, 3): 2.0,
        (right, 3, 0): 1.0,
        (left, 0, 3): 1.0,
    }
    assert mryasov_downfold(exchange, [0], [3]) == {
        ((200, 0, 0), 0, 0): pytest.approx(2 / 3),
        ((-200, 0, 0), 0, 0): pytest.approx(2 / 3),
    }


@pytest.mark.parametrize(
    "exchange",
    [
        {},
        {(Z, 3, 0): 1.0},
        {(Z, 3, 0): 1.0, (Z, 0, 3): 2.0},
        {(Z, 3, 0): 0.0, (Z, 0, 3): 0.0},
        {(Z, 3, 0): 1.0, (Z, 0, 3): 1.0, (Z, 3, 1): -1.0, (Z, 1, 3): -1.0},
        {(Z, 3, 0): float("nan"), (Z, 0, 3): float("nan")},
        {(Z, 3, 0): 1j, (Z, 0, 3): 1j},
        {(Z, 0, 1): float("inf")},
    ],
)
def test_rejects_invalid_couplings_and_fields(exchange):
    with pytest.raises(ValueError):
        mryasov_downfold(exchange, [0, 1], [3])


@pytest.mark.parametrize(
    "metals,ligands",
    [
        ([], [3]),
        ([0], []),
        ([0, 0], [3]),
        ([0], [0]),
        ([-1], [3]),
        ([0.5], [3]),
        ([True], [3]),
    ],
)
def test_rejects_invalid_indices(metals, ligands):
    with pytest.raises(ValueError):
        mryasov_downfold({}, metals, ligands)


@pytest.mark.parametrize("dressing", [0, -1, math.inf, math.nan, True, 1j])
def test_rejects_invalid_dressing(dressing):
    with pytest.raises(ValueError):
        mryasov_downfold({}, [0], [3], dressing=dressing)


def test_rejects_dressing_that_overflows_effective_exchange():
    exchange = {(Z, i, j): 4.0 for i, j in ((0, 3), (3, 0), (1, 3), (3, 1))}
    with pytest.raises(ValueError, match="non-finite effective exchange"):
        mryasov_downfold(exchange, [0, 1], [3], dressing=1e308)
