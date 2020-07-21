import os
import pickle
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from TB2J.io_exchange import SpinIO

# Rotation from x to z
Rxz = Rotation.from_euler('y', -90, degrees=True)
# Rotation from y to z
Ryz = Rotation.from_euler('x', 90, degrees=True)

def merge_DMI(Dx, Dy, Dz):
    Dx_z = Rxz.apply(Dx)
    Dy_z = Ryz.apply(Dy)
    D = (np.array([0.0, 0.5, 0.5]) * Dx_z + np.array([0.5, 0.0, 0.5]) * Dy_z +
         np.array([0.5, 0.5, 0.0]) * Dz)
    return D


class Merger():
    def __init__(self, path_x, path_y, path_z):
        self.dat_x=SpinIO.load_pickle(os.path.join(path_x, 'TB2J_results'), 'TB2J.pickle')
        self.dat_y=SpinIO.load_pickle(os.path.join(path_y, 'TB2J_results'), 'TB2J.pickle')
        self.dat_z=SpinIO.load_pickle(os.path.join(path_z, 'TB2J_results'), 'TB2J.pickle')
        self.dat=copy.copy(self.dat_z)

    def merge_DMI(self):
        dmi_ddict={}
        if self.dat_x.has_dmi and self.dat_y.has_dmi and self.dat_z.has_dmi:
            Dxdict=self.dat_x.dmi_ddict
            Dydict=self.dat_y.dmi_ddict
            Dzdict=self.dat_z.dmi_ddict
            for key, Dz in Dzdict.items():
                try:
                    Dx= Dxdict[key]
                    Dy= Dydict[key]
                except KeyError as err:
                    raise KeyError("%s, Please make sure the three calculations use the same k-mesh and same Rcut."%err)
                dmi_ddict[key] = merge_DMI(Dx, Dy, Dz)
            self.dat.dmi_ddict=dmi_ddict

    def write(self, path='TB2J_results'):
        self.dat.write_all(path=path)



def merge(path_x, path_y, path_z, save=True, path='TB2J_results'):
    m=Merger(path_x, path_y, path_z)
    m.merge_DMI()
    if save:
        m.write(path=path)
    return m.dat

