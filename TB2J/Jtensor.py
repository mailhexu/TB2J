import numpy as np
from dataclasses import dataclass


def decompose_J_tensor(Jtensor):
    """ decompose a exchange tensor into the isotropic exchange (scalar)
    DMI (vector) and the anisotropic exchange (3x3 tensor).

    :param Jtensor:  a 3x3 matrix.
    :returns: 

    """
    Jiso = np.average(np.diag(Jtensor))
    Dm = (Jtensor - Jtensor.T) / 2.0
    D = np.array((Dm[1, 2], Dm[2, 0], Dm[0, 1]), dtype=float)
    Jani = (Jtensor + Jtensor.T) / 2 - np.eye(3) * Jiso
    return Jiso, D, Jani


def combine_J_tensor(Jiso=0.0,
                     D=np.zeros(3),
                     Jani=np.zeros((3, 3), dtype=float)):
    """ Combine isotropic exchange, DMI, and anisotropic exchange into tensor form

    :param Jiso: scalar, isotropice exchange
    :param D: vector, DMI.
    :param Jani: 3x3 matrix anisotropic exchange
    :returns:  A 3x3 matrix, the exchange paraemter in tensor form.
    """
    Jtensor = np.zeros((3, 3), dtype=float)
    if Jiso is not None:
        Jtensor += np.eye(3, dtype=float) * Jiso
    if Jani is not None:
        Jtensor += np.array(Jani, dtype=float)
    if D is not None:
        Jtensor += np.array([[0, D[2], -D[1]],
                             [-D[2], 0, D[0]],
                             [D[1], -D[0], 0]], dtype=float)
    return Jtensor


def test_J_tensor():
    Jiso = 5.0
    DMI = [0.1, 0.2, 0.3]
    Jani = [[0.1, 0.4, 0.3],
            [0.4, 0.1, 0.2],
            [0.3, 0.2, -0.2]]
    Jtensor = combine_J_tensor(Jiso, DMI, Jani)

    Jiso2, DMI2, Jani2 = decompose_J_tensor(Jtensor)
    # print(Jtensor)
    #print(f"{Jiso=},\n {DMI=}, \n{Jani=}")
    assert(np.isclose(Jiso, Jiso2))
    assert(np.allclose(DMI, DMI2))
    assert(np.allclose(Jani, Jani2))


if __name__ == "__main__":
    test_J_tensor()
