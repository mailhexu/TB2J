"""Verify the native transverse-to-Cartesian frozen-Hubbard tangent map.

Run with mydev. This extends the existing native-contract implementation pins;
it does not alter the vault MathDoc or certify a material/exchange tensor.
"""

import sympy as sp


# MathDoc: tb2j/qe-hubbard-native-contract
# The native derivative is physical dV, while the stored splitting is 2 V_mag.
def check_cartesian_map():
    theta, phi = sp.symbols("theta phi", real=True)
    n = sp.Matrix(
        [
            sp.sin(theta) * sp.cos(phi),
            sp.sin(theta) * sp.sin(phi),
            sp.cos(theta),
        ]
    )
    tangent = sp.Matrix(
        [
            [sp.cos(theta) * sp.cos(phi), -sp.sin(phi)],
            [sp.cos(theta) * sp.sin(phi), sp.cos(phi)],
            [-sp.sin(theta), 0],
        ]
    )

    def simplify(matrix):
        return matrix.applyfunc(sp.trigsimp)

    assert simplify(tangent.T * tangent) == sp.eye(2)
    assert simplify(tangent * tangent.T - (sp.eye(3) - n * n.T)) == sp.zeros(3)
    for axis in range(3):
        target = sp.eye(3)[:, axis]
        rebuilt = sum(
            (tangent[axis, j] * n.cross(tangent[:, j]) for j in range(2)),
            sp.zeros(3, 1),
        )
        assert simplify(rebuilt - n.cross(target)) == sp.zeros(3, 1)

    # An exact tilted native x-reference basis discriminates all Cartesian axes.
    n = sp.Matrix([1, 2, 3]) / sp.sqrt(14)
    t1 = sp.eye(3)[:, 0] - n * n[0]
    t1 = sp.simplify(t1 / sp.sqrt(t1.dot(t1)))
    tangent = t1.row_join(sp.simplify(n.cross(t1)))
    pauli = (
        sp.Matrix([[0, 1], [1, 0]]),
        sp.Matrix([[0, -sp.I], [sp.I, 0]]),
        sp.diag(1, -1),
    )
    xyz = sp.symbols("x y z", real=True)
    potential = sum((c * sigma for c, sigma in zip(xyz, pauli)), sp.zeros(2))

    def derivative(direction):
        axis = n.cross(direction)
        generator = sum((axis[k] * pauli[k] for k in range(3)), sp.zeros(2))
        return -sp.I * (generator * potential - potential * generator) / 2

    for axis in range(3):
        rebuilt = sum(
            (tangent[axis, j] * derivative(tangent[:, j]) for j in range(2)),
            sp.zeros(2),
        )
        assert sp.simplify(rebuilt - derivative(sp.eye(3)[:, axis])) == sp.zeros(2)
    longitudinal = sum(
        (n[a] * derivative(sp.eye(3)[:, a]) for a in range(3)), sp.zeros(2)
    )
    assert sp.simplify(longitudinal) == sp.zeros(2)
    print("PASS: native transverse basis maps to Cartesian physical dV tangents")


if __name__ == "__main__":
    check_cartesian_map()
