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


def rot_merge_DMI(Dx, Dy, Dz):
    Dx_z = Rxz.apply(Dx)
    Dy_z = Ryz.apply(Dy)
    D = (np.array([0.0, 0.5, 0.5]) * Dx_z + np.array([0.5, 0.0, 0.5]) * Dy_z +
         np.array([0.5, 0.5, 0.0]) * Dz)
    return D


def rot_merge_DMI2(Dx, Dy, Dz):
    Dx_z = Rxz.apply(Dx)
    Dy_z = Ryz.apply(Dy)
    D = (np.array([1, 0, 0]) * Dx_z + np.array([0, 1, 0]) * Dy_z +
         np.array([0, 0, 1]) * Dz)
    return D


def merge_DMI(Dx, Dy, Dz):
    D = (np.array([0.0, 0.5, 0.5]) * Dx + np.array([0.5, 0.0, 0.5]) * Dy +
         np.array([0.5, 0.5, 0.0]) * Dz)
    return D


def merge_DMI2(Dx, Dy, Dz):
    D = (np.array([1, 0, 0]) * Dx + np.array([0, 1, 0]) * Dy +
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


def merge_Jani(Janix, Janiy, Janiz):
    Jani = (np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) * Janix +
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) * Janiy +
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]) * Janiz) / 2.0
    return Jani


def read_pickle(path):
    p1 = os.path.join(path, 'TB2J_results', 'TB2J.pickle')
    p2 = os.path.join(path, 'TB2J.pickle')
    if os.path.exists(p1) and os.path.exists(p2):
        print(f" WARNING!: Both file {p1} and {p2} exist. Use default {p1}.")
    if os.path.exists(p1):
        ret = SpinIO.load_pickle(os.path.join(path, 'TB2J_results'))
    elif os.path.exists(p2):
        ret = SpinIO.load_pickle(path)
    else:
        raise FileNotFoundError(f"Cannot find either file {p1} or {p2}")
    return ret


class Merger():
    def __init__(self, path_x, path_y, path_z, method='structure'):
        assert (method in ['structure', 'spin'])
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
                    % err)
            if self.method == 'spin':
                Jani_dict[key] = merge_Jani(Janix, Janiy, Janiz)
            else:
                Jani_dict[key] = merge_Jani(swap_direction(Janix, (0, 2)),
                                            swap_direction(Janiy, (1, 2)),
                                            Janiz)
        self.dat.Jani_dict = Jani_dict

    def merge_Jiso(self):
        Jdict={}
        Jxdict=self.dat_x.exchange_Jdict
        Jydict=self.dat_y.exchange_Jdict
        Jzdict=self.dat_z.exchange_Jdict
        for key, J in Jzdict.items():
            try:
               Jx = Jxdict[key]
               Jy = Jydict[key]
               Jz = Jzdict[key]
            except KeyError as err:
                raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err)
            Jdict[key]=(Jx+Jy+Jz)/3.0
        self.dat.exchange_Jdict=Jdict
               

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
                        % err)
                if self.method == 'structure':
                    dmi_ddict[key] = rot_merge_DMI(Dx, Dy, Dz)
                else:
                    dmi_ddict[key] = merge_DMI(Dx, Dy, Dz)
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
                if self.method == 'structure':
                    dmi_ddict[key] = rot_merge_DMI2(Dx, Dy, Dz)
                elif self.method == 'spin':
                    dmi_ddict[key] = merge_DMI2(Dx, Dy, Dz)

            self.dat.debug_dict['DMI2'] = dmi_ddict
        except:
            pass

    def write(self, path='TB2J_results'):
        self.dat.description += 'Merged from TB2J results in paths: \n  ' + '\n  '.join(
            self.paths) + '\n'
        if self.method == 'spin':
            self.dat.description += ', which are from DFT data with spin along x, y, z orientation\n'
        elif self.method == 'structure':
            self.dat.description += ', which are from DFT data with structure with z axis rotated to x, y, z\n'
        self.dat.description += '\n'
        self.dat.write_all(path=path)


def merge(path_x, path_y, path_z, method, save=True, path='TB2J_results'):
    m = Merger(path_x, path_y, path_z, method)
    m.merge_Jiso()
    m.merge_DMI()
    m.merge_Jani()
    if save:
        m.write(path=path)
    return m.dat
