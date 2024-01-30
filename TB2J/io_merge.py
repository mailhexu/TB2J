import os
import copy
import numpy as np
from itertools import product
from TB2J.io_exchange import SpinIO

I = np.eye(3)
uz = np.array([[0.0, 0.0, 1.0]])

def get_rotation_matrix(magmoms):

    dim = magmoms.shape[0]
    v = magmoms / np.linalg.norm(magmoms, axis=-1).reshape(-1, 1)
    n = v[:, [1, 0, 2]]
    n[:, 0] *= -1
    n[:, -1] *= 0
    n /= np.linalg.norm(n, axis=-1).reshape(dim, 1)
    z = np.repeat(uz, dim, axis=0)
    A = np.stack([z, np.cross(n, z), n], axis=1)
    B = np.stack([v, np.cross(n, v), n], axis=1)
    R = np.einsum('nki,nkj->nij', A, B)

    Rnan = np.isnan(R)
    if Rnan.any():
        nanidx = np.where(Rnan)[0]
        R[nanidx] = I
        R[nanidx, 2] = v[nanidx]

    return R

def transform_Jani(Jani, Ri, Rj):

    new_Jani = Ri @ Jani @ Rj.T
    new_Jani[:, 2] *= 0
    new_Jani[2, :] *= 0
    new_Jani = Ri.T @ new_Jani @ Rj

    new_Jani = Rj @ new_Jani @ Ri.T
    new_Jani[:, 2] *= 0
    new_Jani[2, :] *= 0
    new_Jani = Rj.T @ new_Jani @ Ri

    return new_Jani.round(8)

def transform_DMI(DMI, Ri, Rj):

    new_DMI = Ri @ DMI
    new_DMI[2] *= 0
    new_DMI = Ri.T @ new_DMI

    new_DMI = Rj @ new_DMI
    new_DMI[2] *= 0
    new_DMI = Rj.T @ new_DMI

    return new_DMI.round(8)

class SpinIO_merge(SpinIO):
    def __init__(self, *args, **kwargs):
        super(SpinIO_merge, self).__init__(*args, **kwargs)
        self.projv = None
        self._set_ind_atoms()

    def _set_ind_atoms(self):

        if not hasattr(self, 'nspin'):
            self.nspin = len([i for i in self.index_spin if i >= 0])
        if not self.ind_atoms:
            self.ind_atoms = dict([(i, i) for i in range(self.nspin)])

    def _set_rotation_matrix(self, reference_cell=None):

        if reference_cell is None:
            self.sR = np.eye(3)
        else:
            self.sR = np.linalg.solve(reference_cell, self.atoms.cell.array)

        self._set_ind_atoms()
        spinat = self.spinat
        ind_atoms = self.ind_atoms
        
        magmoms = spinat[list(ind_atoms.values())]
        magmoms = magmoms @ self.sR

        self.R = get_rotation_matrix(magmoms)

    @classmethod
    def load_pickle(cls, path='TB2J_results', fname='TB2J.pickle', reference_cell=None):
        obj = super(SpinIO_merge, cls).load_pickle(path=path, fname=fname)
        obj._set_rotation_matrix(reference_cell=reference_cell)

        return obj

def read_pickle(path, reference_cell=None):
    p1 = os.path.join(path, 'TB2J_results', 'TB2J.pickle')
    p2 = os.path.join(path, 'TB2J.pickle')
    if os.path.exists(p1) and os.path.exists(p2):
        print(f" WARNING!: Both file {p1} and {p2} exist. Use default {p1}.")
    if os.path.exists(p1):
        ret = SpinIO_merge.load_pickle(os.path.join(path, 'TB2J_results'), reference_cell=reference_cell)
    elif os.path.exists(p2):
        ret = SpinIO_merge.load_pickle(path, reference_cell=reference_cell)
    else:
        raise FileNotFoundError(f"Cannot find either file {p1} or {p2}")
    return ret

class Merger():
    def __init__(self, *paths, main_path=None, method='structure'):
        if method not in ['structure', 'spin']:
            raise ValueError(f"Unrecognized method '{method}'. Available options are: 'structure' or 'spin'.")
        self.method = method

        if main_path is None:
            self.main_dat = read_pickle(paths[-1])
        else:
            self.main_dat = read_pickle(main_path)

        self.dat = [read_pickle(path, reference_cell=self.main_dat.atoms.cell.array) for path in paths]
        if main_path is not None:
            self.dat.append(self.main_dat) 

    def merge_Jani(self):
        Jani_dict = {}
        for key in self.main_dat.Jani_dict.keys():
            try:
                _, i, j = key
                Jani_list = []
                for obj in self.dat:
                    Jani = obj.sR @ obj.Jani_dict[key] @ obj.sR.T
                    Jani_list.append(transform_Jani(Jani, obj.R[i], obj.R[j]))
                Jani_list = np.stack(Jani_list)
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err)
            Jani_dict[key] = np.nan_to_num(np.mean(Jani_list, axis=0, where=Jani_list != 0.0))
        self.main_dat.Jani_dict = Jani_dict

    def merge_Jiso(self):
        Jdict={}
        for key in self.main_dat.exchange_Jdict.keys():
            try:
               J = np.mean([obj.exchange_Jdict[key] for obj in self.dat])
            except KeyError as err:
                raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err)
            Jdict[key] = J
        self.main_dat.exchange_Jdict = Jdict           

    def merge_DMI(self):
        dmi_ddict = {}
        if all(obj.has_dmi for obj in self.dat):
            for key in self.main_dat.dmi_ddict.keys():
                try:
                    _, i, j = key
                    DMI_list = []
                    for obj in self.dat:
                        DMI = obj.sR @ obj.dmi_ddict[key]
                        DMI_list.append(transform_DMI(DMI, obj.R[i], obj.R[j]))
                    DMI_list = np.stack(DMI_list)
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err)
                DMI = np.nan_to_num(np.mean(DMI_list, axis=0, where=DMI_list != 0.0))
                dmi_ddict[key] = DMI
            self.main_dat.dmi_ddict = dmi_ddict

def merge(*paths, main_path=None, method='structure', save=True, write_path='TB2J_results'):
    m = Merger(*paths, main_path=main_path, method=method)
    m.merge_Jiso()
    m.merge_DMI()
    m.merge_Jani()
    if save:
        m.main_dat.write_all(path=write_path)
    return m.dat
