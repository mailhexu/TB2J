import os
import pickle
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from TB2J.io_exchange import SpinIO

# Rotation from x to z
Rxz = Rotation.from_euler('y', -90, degrees=True)
# Rotation from y to z
Ryz = Rotation.from_euler('x', -90, degrees=True)


def merge_DMI(Dx, Dy, Dz):
    Dx_z = Rxz.apply(Dx)
    Dy_z = Ryz.apply(Dy)
    D = (np.array([0.0, 0.5, 0.5]) * Dx_z + np.array([0.5, 0.0, 0.5]) * Dy_z +
         np.array([0.5, 0.5, 0.0]) * Dz)
    return D


def merge_DMI2(Dx, Dy, Dz):
    Dx_z = Rxz.apply(Dx)
    Dy_z = Ryz.apply(Dy)
    D = (np.array([1, 0, 0]) * Dx_z + np.array([0, 1, 0]) * Dy_z +
         np.array([0, 0, 1]) * Dz)
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


class Merger():
    def __init__(self, path_x, path_y, path_z):
        self.dat_x = SpinIO.load_pickle(os.path.join(path_x, 'TB2J_results'),
                                        'TB2J.pickle')
        self.dat_y = SpinIO.load_pickle(os.path.join(path_y, 'TB2J_results'),
                                        'TB2J.pickle')
        self.dat_z = SpinIO.load_pickle(os.path.join(path_z, 'TB2J_results'),
                                        'TB2J.pickle')
        self.dat = copy.copy(self.dat_z)

    def merge_DMI(self):
        dmi_ddict = {}
        if self.dat_x.has_dmi and self.dat_y.has_dmi and self.dat_z.has_dmi:
            Dxdict = self.dat_x.dmi_ddict
            Dydict = self.dat_y.dmi_ddict
            Dzdict = self.dat_z.dmi_ddict
            for key, Dz in Dzdict.items():
                try:
                    R, i, j = key
                    #keyx=tuple(map(int, np.round(Rxz.apply(R))))
                    #keyy=tuple(map(int, np.round(Ryz.apply(R))))
                    keyx = R
                    keyy = R
                    Dx = Dxdict[(tuple(keyx), i, j)]
                    Dy = Dydict[(tuple(keyy), i, j)]
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err)
                dmi_ddict[key] = merge_DMI2(Dx, Dy, Dz)
            self.dat.dmi_ddict = dmi_ddict

        dmi_ddict = {}
        try:
            Dxdict = self.dat_x.debug_dict['DMI2']
            Dydict = self.dat_y.debug_dict['DMI2']
            Dzdict = self.dat_z.debug_dict['DMI2']
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
                        % err)
                dmi_ddict[key] = merge_DMI2(Dx, Dy, Dz)
            self.dat.debug_dict['DMI2'] = dmi_ddict
        except:
            pass

    def write(self, path='TB2J_results'):
        self.dat.write_all(path=path)


def merge(path_x, path_y, path_z, save=True, path='TB2J_results'):
    m = Merger(path_x, path_y, path_z)
    m.merge_DMI()
    if save:
        m.write(path=path)
    return m.dat
