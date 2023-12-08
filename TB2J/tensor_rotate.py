import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, det


# Rotation from a to b
Rxx = Rotation.from_euler("x", 0, degrees=True)
Rxy = Rotation.from_euler("z", 90, degrees=True)
Rxz = Rotation.from_euler("y", -90, degrees=True)
# rotation from y to z
Ryx = Rotation.from_euler("z", -90, degrees=True)
Ryy = Rotation.from_euler("x", 0, degrees=True)
Ryz = Rotation.from_euler("x", 90, degrees=True)

# Inverse of the Rotation from x to z
Rzx = Rotation.from_euler("y", 90, degrees=True)
# Inverse of the Rotation from y to z
Rzy = Rotation.from_euler("x", -90, degrees=True)
Rzz = Rotation.from_euler("z", 0, degrees=True)


def normalize(vec):
    """
    Normalize vector
    :param vec: 3D vector
    :return: normalized vector
    """
    return vec / norm(vec)


def find_rotation_matrix_to_vec2(vec1, vec2=np.array([0, 0, 1])):
    """
    Find rotation matrix to align vector with vec2
    :param vec: 3D vector
    :param vec2: 3D vector
    :return: rotation matrix
    """
    # Normalize vector
    v1 = vec1 / norm(vec1)
    v2 = vec2 / norm(vec2)
    # Find rotation axis
    axis = np.cross(v1, np.array(v2))
    if norm(axis) < 1e-8:
        return np.eye(3, dtype=float)
    axis = axis / norm(axis)
    # Find rotation angle
    angle = np.arccos(np.dot(v1, v2))
    # Find rotation matrix
    rot = Rotation.from_rotvec(angle * axis)
    return rot.as_matrix()


def rotate_vec_to_vec2(vec, vec2=np.array([0, 0, 1])):
    """
    Rotate vector to align with z-axis
    :param vec: 3D vector
    :return: rotated vector
    """
    rot = find_rotation_matrix_to_vec2(vec, vec2)
    return np.dot(rot, vec)


def rotate_tensor(vec1, J, vec2):
    """
    Let H= vec1@T@vec2, rotate vec1 and vec2 to the z-axis, and return the rotated tensor
    """
    z = np.array([0, 0, 1])
    rot1 = find_rotation_matrix_to_vec2(vec1, z)
    rot2 = find_rotation_matrix_to_vec2(vec2, z)
    return rot1, rot2, rot1 @ J @ rot2.T


def rotate_back(rot1, rot2, J_rotated):
    """
    Rotate back the rotated tensor
    """
    return rot1.T @ J_rotated @ rot2


def remove_components(J, vec1, vec2, remove_indices=[[2, 2]]):
    """
    Remove some  component of the roated tensor and rotate back.
    :param J: tensor
    :param vec1: vector 1 which the tensor acts on (vec1@J@vec2)
    :param vec2: vector 2 which the tensor acts on (vec1@J@vec2)
    :param remove_indices: list of indices to remove, e.g. [[2,2]] to remove the zz component.
    Returns:
        Jback: the tensor with  removed components
        weigth_back: the weight of the Jback.
    """
    z = np.array([0, 0, 1])
    w = np.ones_like(J)
    rot1, rot2, J_rotated = rotate_tensor(vec1, J, vec2)
    w_rotated = rot1 @ w @ rot2.T
    for i, j in remove_indices:
        w_rotated[i, j] = 0
        J_rotated[i, j] = 0
    Jback = rotate_back(rot1, rot2, J_rotated)
    weight_back = rotate_back(rot1, rot2, w_rotated)
    # if np.linalg.norm(J)>1e-4:
    #    print(f"{vec1=}, {vec2=}")
    #    print(f"{J=}")
    #    print(f"{J_rotated=}")
    #    print(f"{Jback=}")
    #    print(f"{w_rotated=}")
    #    print(f"{weight_back=}")
    return Jback, weight_back


def get_weight_back(vec1, vec2):
    z = np.array([0, 0, 1])
    rot1 = find_rotation_matrix_to_vec2(vec1, z)
    rot2 = find_rotation_matrix_to_vec2(vec2, z)
    w = np.ones((3, 3))
    # w=np.eye(3)
    w_rotated = rot1 @ w @ rot2.T
    w_rotated[:, 2] = 0
    w_rotated[2, :] = 0
    w_rotated[2, 2] = 0
    return rot1.T @ w_rotated @ rot2


def test_weight_back():
    vec1 = [0.0, 0.0, 0.1]
    vec2 = [0.0, 0.0, 0.1]
    vec1 = normalize(vec1)
    vec2 = normalize(vec2)
    w = 0
    # Rotations=[Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz]
    # wR=[0, 1, 1, 0, 0, 1, 0, 0, 0]
    # Rotations=[ Rxy, Rxz, Ryx,  Ryz, Rzx, Rzy]
    Rotations = [Rxy, Ryz, Rzx]
    # Rotations=[ Ryx, Rzy, Rxz]
    wR = [1] * len(Rotations)
    # Rotations=[Rxy, Ryz, Rzx]
    for wR, R in zip(wR, Rotations):
        v1 = R.apply(vec1)
        v2 = R.apply(vec2)
        wi = get_weight_back(v1, v2)
        print(wi * wR)
        w += wi * wR
    print((w + w.T) / 2)


# test_weight_back()


def test_remove_component():
    # vec1=[0.3, 0.4, 0.5]
    # vec2=[0.3, 0.4, -0.5]
    vec1 = [0.0, 0.1, 0]
    vec2 = [0.0, 0.1, 0]
    z = np.array([0, 0, 1])
    J = [[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]]
    J = [[0.0, 0.1, 0.2], [-0.1, 0.0, 0.3], [-0.2, -0.3, 0.0]]
    print(f"{J=}")
    w = np.ones_like(J)
    # w = np.eye(3)
    rot1, rot2, J_rotated = rotate_tensor(vec1, J, vec2)
    J_rotated = rot1 @ J @ rot2.T
    w_rotated = rot1 @ w @ rot2.T
    m = np.zeros((3, 3))
    m[0, 0] = 1
    m[0, 1] = 1
    # w_rotated = np.ones_like(J)
    print(f"{w_rotated=}")
    print(f"{J_rotated=}")
    print(f"{J*w_rotated=}")
    m = np.linalg.inv(J_rotated)
    remove = False
    if remove:
        w_rotated[:, 2] = 0
        w_rotated[2, :] = 0
        J_rotated[2, :] = 0
        J_rotated[:, 2] = 0
    m = m @ J_rotated
    Jback = rotate_back(rot1, rot2, J_rotated)
    # weight_back = rotate_back(rot1, rot2, w_rotated)
    weight_back = rot2 @ m @ rot2.T
    print(f"{weight_back=}")
    print(f"{Jback=}")
    print(f"{J@weight_back=}")
    print(f"{np.sum(weight_back)=}")


# test_remove_component()


def remove_zz_component(J, vec1, vec2):
    """
    Remove the zz component of the tensor
    """
    # return remove_components(J, vec1, vec2, remove_indices=[[2, 2]])
    return remove_components(
        J, vec1, vec2, remove_indices=[[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]
    )


def test_remove_zz_component():
    """
    Test remove_zz_component
    """
    J = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    J2, weight = remove_zz_component(J, vec1, vec2)
    print(J2)
    print(weight)
    print(vec1 @ J @ vec2)
    print(vec1 @ J2 @ vec2)
    print(vec1 @ weight @ vec2)
    # print(vec1@J2@vec2/vec1@weight@vec2)

    # Test with random vectors
    for i in range(10):
        vec1 = np.random.rand(3)
        vec2 = np.random.rand(3)
        J = np.random.rand(3, 3)
        J2, weight = remove_zz_component(J, vec1, vec2)
        # assert np.allclose(vec1@J@vec2, vec1@J2@vec2)
        # assert np.allclose(vec1@weight@vec2, 0)
        # assert np.allclose(weight[2, 2], 0)


def test_rotate_tensor():
    """
    Test rotate_tensor
    """
    J = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    rot1, rot2, J2 = rotate_tensor(vec1, J, vec2)
    print(J2)
    print(vec1 @ J @ vec2)
    print(vec1 @ rot1.T @ J2 @ rot2 @ vec2)

    # Test with random vectors
    for i in range(10):
        vec1 = np.random.rand(3)
        vec2 = np.random.rand(3)
        rot1, rot2, J2 = rotate_tensor(vec1, J, vec2)
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        assert np.allclose(np.linalg.norm(vec1 @ rot1.T), norm(vec1))
        assert np.allclose(vec1 @ J @ vec2, vec1 @ rot1.T @ J2 @ rot2 @ vec2)
        assert np.allclose(rot1 @ rot1.T, np.eye(3))
        assert np.allclose(rot2 @ rot2.T, np.eye(3))


def test_find_rotation_matrix_to_z():
    """
    Test find_rotation_matrix_to_z
    """
    v = rotate_vec_to_vec2(np.array([1, 0, 0]), np.array([0, 0, 1]))
    assert np.allclose(v, np.array([0, 0, 1]))

    v = rotate_vec_to_vec2(np.array([-1, 0, 0]), np.array([0, 0, 1]))
    assert np.allclose(v, np.array([0, 0, 1]))

    v = rotate_vec_to_vec2(np.array([-1.0, 3, 8]), np.array([1, 0, 0]))
    print(v)
    assert np.allclose(v, np.array([1, 0, 0]))

    v = rotate_vec_to_vec2(np.array([-1.0, 3, 8]), np.array([3, 0.04, -0.3]))
    print(v)
    assert np.allclose(v, np.array([3, 0.04, -0.3]))


def test_rotate_isotropic_tensor():
    a = np.eye(3)
    b = rotate_tensor([0, 0, 1], np.eye(3), [1, 0, 0])
    print(b)


if __name__ == "__main__":
    # test_rotate_tensor()
    # test_find_rotation_matrix_to_z()
    # test_remove_zz_component()
    # test_rotate_isotropic_tensor()
    pass
