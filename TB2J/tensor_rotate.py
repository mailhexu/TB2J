import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, det

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
    axis = axis/norm(axis)
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
    z=np.array([0, 0, 1])
    rot1 = find_rotation_matrix_to_vec2(vec1, z)
    rot2 = find_rotation_matrix_to_vec2(vec2, z)
    return rot1, rot2, rot1@J@rot2.T

def rotate_back(rot1, rot2, J_rotated):
    """
    Rotate back the rotated tensor
    """
    return rot1.T@J_rotated@rot2

def remove_components(J, vec1, vec2, indices=[[2,2]]):
    """
    Remove the zz component of the tensor
    """
    z=np.array([0, 0, 1])
    w=np.ones_like(J)
    rot1, rot2, J_rotated=rotate_tensor(vec1, J, vec2)
    w_rotated= rot1@w@rot2.T
    for (i,j) in indices:
        w_rotated[i, j]=0
        J_rotated[i, j]=0
    Jback= rotate_back(rot1, rot2, J_rotated)
    weigth_back= rotate_back(rot1, rot2, w_rotated)
    return Jback, weigth_back

def remove_zz_component(J, vec1, vec2):
    """
    Remove the zz component of the tensor
    """
    return remove_components(J, vec1, vec2, indices=[[2,2]])

def test_remove_zz_component():
    """
    Test remove_zz_component
    """
    J=np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    vec1=np.array([0, -1, 0])
    vec2=np.array([1, 0, 0])
    J2, weight=remove_zz_component(J, vec1, vec2)
    print(J2)
    print(weight)
    print(vec1@J@vec2)
    print(vec1@J2@vec2)
    print(vec1@weight@vec2)
    #print(vec1@J2@vec2/vec1@weight@vec2)

    # Test with random vectors
    for i in range(10):
        vec1=np.random.rand(3)
        vec2=np.random.rand(3)
        J=np.random.rand(3, 3)
        J2, weight=remove_zz_component(J, vec1, vec2)
        #assert np.allclose(vec1@J@vec2, vec1@J2@vec2)
        #assert np.allclose(vec1@weight@vec2, 0)
        #assert np.allclose(weight[2, 2], 0)


def test_rotate_tensor():
    """
    Test rotate_tensor
    """
    J=np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    vec1=np.array([1, 0, 0])
    vec2=np.array([0, 1, 0])
    rot1, rot2, J2=rotate_tensor(vec1, J, vec2)
    print(J2)
    print(vec1@J@vec2)
    print(vec1@rot1.T@J2@rot2@vec2)

    # Test with random vectors
    for i in range(10):
        vec1=np.random.rand(3)
        vec2=np.random.rand(3)
        rot1, rot2, J2=rotate_tensor(vec1, J, vec2)
        norm1=norm(vec1)
        norm2=norm(vec2)
        assert np.allclose(np.linalg.norm(vec1@rot1.T) , norm(vec1))
        assert np.allclose(vec1@J@vec2, vec1@rot1.T@J2@rot2@vec2)
        assert np.allclose(rot1@rot1.T, np.eye(3))
        assert np.allclose(rot2@rot2.T, np.eye(3))  
    



def test_find_rotation_matrix_to_z():
    """
    Test find_rotation_matrix_to_z
    """
    v=rotate_vec_to_vec2(np.array([1, 0, 0]), np.array([0, 0, 1]))
    assert np.allclose(v, np.array([0, 0, 1]))

    v=rotate_vec_to_vec2(np.array([-1, 0, 0]), np.array([0, 0, 1]))
    assert np.allclose(v, np.array([0, 0, 1]))

    v=rotate_vec_to_vec2(np.array([-1.0, 3, 8]), np.array([1, 0, 0]))
    print(v)
    assert np.allclose(v, np.array([1, 0, 0]))

    v=rotate_vec_to_vec2(np.array([-1.0, 3, 8]), np.array([3, 0.04, -0.3]))
    print(v)
    assert np.allclose(v, np.array([3, 0.04, -0.3]))




#test_rotate_tensor()
#test_find_rotation_matrix_to_z()
test_remove_zz_component()