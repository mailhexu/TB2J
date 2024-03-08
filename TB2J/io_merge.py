import os
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from TB2J.io_exchange import SpinIO
from TB2J.tensor_rotate import remove_components
from TB2J.Jtensor import DMI_to_Jtensor, Jtensor_to_DMI
from TB2J.tensor_rotate import Rzx, Rzy, Rzz, Ryz, Rxz


def test_rotation_matrix():
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]
    print(Rxz.apply(x))
    print(Ryz.apply(y))
    print(Rzx.apply(z))
    print(Rzy.apply(z))


def recover_DMI_from_rotated_structure(Ddict, rotation):
    """
    Recover the DMI vector from the rotated structure.
    D: the dictionary of DMI vector in the rotated structure.
    rotation: the rotation operator from the original structure to the rotated structure.
    """
    for key, val in Ddict.items():
        Ddict[key] = rotation.apply(val, inverse=True)
    return Ddict


def recover_Jani_fom_rotated_structure(Janidict, rotation):
    """
    Recover the Jani tensor from the rotated structure.
    Janidict: the dictionary of Jani tensor in the rotated structure.
    rotation: the  from the original structure to the rotated structure.
    """
    R = rotation.as_matrix()
    RT = R.T
    for key, Jani in Janidict.items():
        # Note: E=Si J Sj , Si'= Si RT, Sj' = R Sj,
        #       Si' J' Sj' = Si RT R J RT R Sj => J' = R J RT
        #       But here we are doing the opposite rotation back to
        #       the original axis, so we replace R with RT.
        Janidict[key] = RT @ Jani @ R
    return Janidict


def recover_spinat_from_rotated_structure(spinat, rotation):
    """
    Recover the spinat from the rotated structure.
    spinat: the spinat in the rotated structure.
    rotation: the rotation operator from the original structure to the rotated structure.
    """
    for i, spin in enumerate(spinat):
        spinat[i] = rotation.apply(spin, inverse=True)
    return spinat


# test_rotation_matrix()

# R_xyz = [Rxz.as_matrix(), Ryz.as_matrix(), np.eye(3, dtype=float)]


def rot_merge_DMI(Dx, Dy, Dz):
    Dx_z = Rzx.apply(Dx)
    Dy_z = Rzy.apply(Dy)
    D = (
        np.array([0.0, 0.5, 0.5]) * Dx_z
        + np.array([0.5, 0.0, 0.5]) * Dy_z
        + np.array([0.5, 0.5, 0.0]) * Dz
    )
    return D


def rot_merge_DMI2(Dx, Dy, Dz):
    Dx_z = Rzx.apply(Dx)
    Dy_z = Rzy.apply(Dy)
    D = (
        np.array([1, 0, 0]) * Dx_z
        + np.array([0, 1, 0]) * Dy_z
        + np.array([0, 0, 1]) * Dz
    )
    return D


def merge_DMI(Dx, Dy, Dz):
    D = (
        np.array([0.0, 0.5, 0.5]) * Dx
        + np.array([0.5, 0.0, 0.5]) * Dy
        + np.array([0.5, 0.5, 0.0]) * Dz
    )
    return D


def merge_DMI2(Dx, Dy, Dz):
    D = np.array([1, 0, 0]) * Dx + np.array([0, 1, 0]) * Dy + np.array([0, 0, 1]) * Dz
    return D


def swap_direction(m, idirections):
    """
    swap two directions of a tensor m.
    idirections: the index of the two directions.
    """
    idirections = list(idirections)
    inv = [idirections[1], idirections[0]]
    n = np.copy(m)
    n[:, idirections] = n[:, inv]
    n[idirections, :] = n[inv, :]
    return n


def test_swap():
    m = np.reshape(np.arange(9), (3, 3))
    print(m)
    print(swap_direction(m, [0, 1]))


def merge_Jani(Janix, Janiy, Janiz):
    # This is wrong.
    # Jani = (
    #    np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) * Janix
    #    + np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) * Janiy
    #    + np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]) * Janiz
    # ) / 2.0
    wx = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
    wy = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    wz = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    Jani = (wx * Janix + wy * Janiy + wz * Janiz) / (wx + wy + wz)
    return Jani


def read_pickle(path):
    p1 = os.path.join(path, "TB2J_results", "TB2J.pickle")
    p2 = os.path.join(path, "TB2J.pickle")
    if os.path.exists(p1) and os.path.exists(p2):
        print(f" WARNING!: Both file {p1} and {p2} exist. Use default {p1}.")
    if os.path.exists(p1):
        ret = SpinIO.load_pickle(os.path.join(path, "TB2J_results"))
    elif os.path.exists(p2):
        ret = SpinIO.load_pickle(path)
    else:
        raise FileNotFoundError(f"Cannot find either file {p1} or {p2}")
    return ret


class Merger2:
    def __init__(self, paths, method):
        self.method = method
        if method.lower() == "spin":
            self.load_with_rotated_spin(paths)
        elif method.lower() == "structure":
            self.load_with_rotated_structure(paths)
        else:
            raise ValueError("method should be either 'spin' or 'structure'")

    def load_with_rotated_structure(self, paths):
        """
        Merge TB2J results from multiple calculations.
        :param paths: a list of paths to the TB2J results.
        :param method: 'structure' or 'spin'
        """
        self.paths = paths
        if len(self.paths) != 3:
            raise ValueError(
                "The number of paths should be 3, with structure rotated from z to x, y, z"
            )
        for i, path in enumerate(self.paths):
            read_pickle(path)
        self.indata = [read_pickle(path) for path in paths]

        self.dat = copy.deepcopy(self.indata[-1])
        # self.dat.description += (
        #    "Merged from TB2J results in paths: \n  " + "\n  ".join(paths) + "\n"
        # )
        Rotations = [Rzx, Rzy, Rzz]
        for dat, rotation in zip(self.indata, Rotations):
            dat.spinat = recover_spinat_from_rotated_structure(dat.spinat, rotation)
            dat.dmi_ddict = recover_DMI_from_rotated_structure(dat.dmi_ddict, rotation)
            dat.Jani_dict = recover_Jani_fom_rotated_structure(dat.Jani_dict, rotation)

    def load_with_rotated_spin(self, paths):
        """
        Merge TB2J results from multiple calculations.
        :param paths: a list of paths to the TB2J results.
        :param method: 'structure' or 'spin'
        """
        self.paths = paths
        self.indata = [read_pickle(path) for path in paths]
        self.dat = copy.deepcopy(self.indata[-1])
        # self.dat.description += (
        #    "Merged from TB2J results in paths: \n  " + "\n  ".join(paths) + "\n"
        # )

    def merge_Jani(self):
        """
        Merge the anisotropic exchange tensor.
        """
        Jani_dict = {}
        for key, Jani in self.dat.Jani_dict.items():
            R, i, j = key
            weights = np.zeros((3, 3), dtype=float)
            Jani_sum = np.zeros((3, 3), dtype=float)
            for dat in self.indata:
                Si = dat.get_spin_ispin(i)
                Sj = dat.get_spin_ispin(j)
                # print(f"{Si=}, {Sj=}")
                Jani = dat.get_Jani(i, j, R, default=np.zeros((3, 3), dtype=float))
                Jani_removed, w = remove_components(
                    Jani,
                    Si,
                    Sj,
                    remove_indices=[[0, 2], [1, 2], [2, 2], [2, 1], [2, 0]],
                )
                w = Jani_removed / Jani
                Jani_sum += Jani * w  # Jani_removed
                # print(f"{Jani* w=}")
                weights += w
            # print(f"{weights=}")
            if np.any(weights == 0):
                raise RuntimeError(
                    "The data set to be merged does not give a complete anisotropic J tensor, please add more data"
                )
            Jani_dict[key] = Jani_sum / weights
        self.dat.Jani_dict = Jani_dict

    def merge_DMI(self):
        """
        merge the DMI vector
        """
        DMI = {}
        for key, D in self.dat.dmi_ddict.items():
            R, i, j = key
            weights = np.zeros((3, 3), dtype=float)
            Dtensor_sum = np.zeros((3, 3), dtype=float)
            for dat in self.indata:
                Si = dat.get_spin_ispin(i)
                Sj = dat.get_spin_ispin(j)
                D = dat.get_DMI(i, j, R, default=np.zeros((3,), dtype=float))
                Dtensor = DMI_to_Jtensor(D)
                Dtensor_removed, w = remove_components(
                    Dtensor, Si, Sj, remove_indices=[[0, 1], [1, 0]]
                )
                Dtensor_sum += Dtensor * w  # Dtensor_removed
                weights += w
            if np.any(weights == 0):
                raise RuntimeError(
                    "The data set to be merged does not give a complete DMI vector, please add more data"
                )
            DMI[key] = Jtensor_to_DMI(Dtensor_sum / weights)
        self.dat.dmi_ddict = DMI

    def merge_Jiso(self):
        """
        merge the isotropic exchange
        """
        Jiso = {}
        for key, J in self.dat.exchange_Jdict.items():
            R, i, j = key
            weights = 0.0
            Jiso_sum = 0.0
            for dat in self.indata:
                Si = dat.get_spin_ispin(i)
                Sj = dat.get_spin_ispin(j)
                J = dat.get_Jiso(i, j, R, default=0.0)
                Jiso_sum += J  # *np.eye(3, dtype=float)
                weights += 1.0
            if np.any(weights == 0):
                raise RuntimeError(
                    "The data set to be merged does not give a complete isotropic exchange, please add more data"
                )
            Jiso[key] = Jiso_sum / weights
        self.dat.exchange_Jdict = Jiso

    def write(self, path="TB2J_results"):
        """
        Write the merged TB2J results to a folder.
        :param path: the path to the folder to write the results.
        """
        self.dat.description += (
            "Merged from TB2J results in paths: \n  " + "\n  ".join(self.paths) + "\n"
        )
        if self.method == "spin":
            self.dat.description += (
                ", which are from DFT data with various spin orientations. \n"
            )
        elif self.method == "structure":
            self.dat.description += ", which are from DFT data with structure with z axis rotated to x, y, z\n"
        self.dat.description += "\n"
        self.dat.write_all(path=path)


class Merger:
    def __init__(self, path_x, path_y, path_z, method="structure"):
        assert method in ["structure", "spin"]
        self.dat_x = read_pickle(path_x)
        self.dat_y = read_pickle(path_y)
        self.dat_z = read_pickle(path_z)
        self.dat = copy.copy(self.dat_z)
        self.paths = [path_x, path_y, path_z]
        self.method = method

    def merge_Jani(self):
        Jani_dict = {}
        Janixdict = self.dat_x.Jani_dict
        Janiydict = self.dat_y.Jani_dict
        Janizdict = self.dat_z.Jani_dict
        for key, Janiz in Janizdict.items():
            try:
                R, i, j = key
                keyx = R
                keyy = R
                Janix = Janixdict[(tuple(keyx), i, j)]
                Janiy = Janiydict[(tuple(keyy), i, j)]
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err
                )
            if self.method == "spin":
                Jani_dict[key] = merge_Jani(Janix, Janiy, Janiz)
            else:
                Jani_dict[key] = merge_Jani(
                    swap_direction(Janix, (0, 2)), swap_direction(Janiy, (1, 2)), Janiz
                )
        self.dat.Jani_dict = Jani_dict

    def merge_Jiso(self):
        Jdict = {}
        Jxdict = self.dat_x.exchange_Jdict
        Jydict = self.dat_y.exchange_Jdict
        Jzdict = self.dat_z.exchange_Jdict
        for key, J in Jzdict.items():
            try:
                Jx = Jxdict[key]
                Jy = Jydict[key]
                Jz = Jzdict[key]
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err
                )
            Jdict[key] = (Jx + Jy + Jz) / 3.0
        self.dat.exchange_Jdict = Jdict

    def merge_DMI(self):
        dmi_ddict = {}
        if self.dat_x.has_dmi and self.dat_y.has_dmi and self.dat_z.has_dmi:
            Dxdict = self.dat_x.dmi_ddict
            Dydict = self.dat_y.dmi_ddict
            Dzdict = self.dat_z.dmi_ddict
            for key, Dz in Dzdict.items():
                try:
                    R, i, j = key
                    keyx = R
                    keyy = R
                    Dx = Dxdict[(tuple(keyx), i, j)]
                    Dy = Dydict[(tuple(keyy), i, j)]
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err
                    )
                if self.method == "structure":
                    dmi_ddict[key] = rot_merge_DMI(Dx, Dy, Dz)
                else:
                    dmi_ddict[key] = merge_DMI(Dx, Dy, Dz)
            self.dat.dmi_ddict = dmi_ddict

        dmi_ddict = {}
        try:
            Dxdict = self.dat_x.debug_dict["DMI2"]
            Dydict = self.dat_y.debug_dict["DMI2"]
            Dzdict = self.dat_z.debug_dict["DMI2"]
            for key, Dz in Dzdict.items():
                try:
                    R, i, j = key
                    keyx = R
                    keyy = R
                    Dx = Dxdict[(tuple(keyx), i, j)]
                    Dy = Dydict[(tuple(keyy), i, j)]
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err
                    )
                if self.method == "structure":
                    dmi_ddict[key] = rot_merge_DMI2(Dx, Dy, Dz)
                elif self.method == "spin":
                    dmi_ddict[key] = merge_DMI2(Dx, Dy, Dz)

            self.dat.debug_dict["DMI2"] = dmi_ddict
        except:
            pass

    def write(self, path="TB2J_results"):
        self.dat.description += (
            "Merged from TB2J results in paths: \n  " + "\n  ".join(self.paths) + "\n"
        )
        if self.method == "spin":
            self.dat.description += (
                ", which are from DFT data with spin along x, y, z orientation\n"
            )
        elif self.method == "structure":
            self.dat.description += ", which are from DFT data with structure with z axis rotated to x, y, z\n"
        self.dat.description += "\n"
        self.dat.write_all(path=path)


def merge(path_x, path_y, path_z, method, save=True, path="TB2J_results"):
    m = Merger(path_x, path_y, path_z, method)
    m.merge_Jiso()
    m.merge_DMI()
    m.merge_Jani()
    if save:
        m.write(path=path)
    return m.dat


def merge2(paths, method, save=True, path="TB2J_results"):
    """
    Merge TB2J results from multiple calculations.
    :param paths: a list of paths to the TB2J results.
    :param method: 'structure' or 'spin'
    :param save: whether to save the merged results.
    :param path: the path to the folder to write the results.
    """
    m = Merger2(paths, method)
    m.merge_Jiso()
    m.merge_DMI()
    m.merge_Jani()
    if save:
        m.write(path=path)
    return m.dat
