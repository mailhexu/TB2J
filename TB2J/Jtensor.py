import numpy as np


def Jiso_to_Jtensor(Jiso):
    """Convert isotropic exchange to tensor form

    :param Jiso: scalar, isotropice exchange
    :returns:  A 3x3 matrix, the exchange paraemter in tensor form.
    """
    return np.eye(3, dtype=float) * Jiso


def Jani_to_Jtensor(Jani):
    """Convert anisotropic exchange to tensor form

    :param Jani: 3x3 matrix anisotropic exchange
    :returns:  A 3x3 matrix, the exchange paraemter in tensor form.
    """
    return np.array(Jani, dtype=float)


def DMI_to_Jtensor(D):
    """Convert DMI to tensor form

    :param D: vector, DMI.
    :returns:  A 3x3 matrix, the exchange paraemter in tensor form.
    """
    return np.array([[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]], dtype=float)


def Jtensor_to_Jiso(Jtensor):
    """Convert tensor form to isotropic exchange

    :param Jtensor:  A 3x3 matrix, the exchange paraemter in tensor form.
    :returns: scalar, isotropice exchange
    """
    return np.average(np.diag(Jtensor))


def Jtensor_to_Jani(Jtensor):
    """Convert tensor form to anisotropic exchange

    :param Jtensor:  A 3x3 matrix, the exchange paraemter in tensor form.
    :returns: 3x3 matrix anisotropic exchange
    """
    return (Jtensor + Jtensor.T) / 2 - np.eye(3) * Jtensor_to_Jiso(Jtensor)


def Jtensor_to_DMI(Jtensor):
    """Convert tensor form to DMI

    :param Jtensor:  A 3x3 matrix, the exchange paraemter in tensor form.
    :returns: vector, DMI.
    """
    Dm = (Jtensor - Jtensor.T) / 2.0
    return np.array((Dm[1, 2], Dm[2, 0], Dm[0, 1]), dtype=float)


def decompose_J_tensor(Jtensor):
    """decompose a exchange tensor into the isotropic exchange (scalar)
    DMI (vector) and the anisotropic exchange (3x3 tensor).

    :param Jtensor:  a 3x3 matrix.
    :returns:

    """
    Jtensor = Jtensor.real
    Jiso = np.average(np.diag(Jtensor))
    Dm = (Jtensor - Jtensor.T).real / 2.0
    D = np.array((Dm[1, 2], Dm[2, 0], Dm[0, 1]), dtype=float)
    Jani = (Jtensor + Jtensor.T) / 2 - np.eye(3) * Jiso
    return Jiso, D, Jani


def combine_J_tensor(Jiso=None, D=None, Jani=None, dtype=float):
    """Combine isotropic exchange, DMI, and anisotropic exchange into tensor form
    :param Jiso: scalar, isotropice exchange
    :param D: vector, DMI.
    :param Jani: 3x3 matrix anisotropic exchange
    :returns:  A 3x3 matrix, the exchange paraemter in tensor form.
    """
    Jtensor = np.zeros((3, 3), dtype=dtype)
    if Jiso is not None:
        Jtensor += np.eye(3, dtype=dtype) * Jiso
    if Jani is not None:
        Jtensor += np.array(Jani, dtype=dtype)
    if D is not None:
        Jtensor += np.array(
            [[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]], dtype=dtype
        )
    return Jtensor


def test_J_tensor():
    Jiso = 5.0
    DMI = [0.1, 0.2, 0.3]
    Jani = [[0.1, 0.4, 0.3], [0.4, 0.1, 0.2], [0.3, 0.2, -0.2]]
    Jtensor = combine_J_tensor(Jiso, DMI, Jani)

    Jiso2, DMI2, Jani2 = decompose_J_tensor(Jtensor)
    # print(Jtensor)
    # print(f"{Jiso=},\n {DMI=}, \n{Jani=}")
    assert np.isclose(Jiso, Jiso2)
    assert np.allclose(DMI, DMI2)
    assert np.allclose(Jani, Jani2)


if __name__ == "__main__":
    test_J_tensor()
