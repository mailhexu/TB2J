import os
import copy
import numpy as np
from itertools import product
from TB2J.io_exchange import SpinIO

uy = np.array([0.0, 1.0, 0.0])
uz = np.array([0.0, 0.0, 1.0])

def orthplane(a):

    b = np.cross(uy, a)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-10:
        b = np.cross(uz, a)
    c = np.cross(a, b)
    norm_c = np.linalg.norm(c)

    return np.array([
        b / norm_b,
        c / norm_c
    ])

def flat_projections(a):

    vectors = []
    for u, v in product(a, a):
        vectors.append(
            np.hstack([u*v, np.roll(u, -1)*v + np.roll(v, -1)*u])
        )

    return np.vstack(vectors)

class SpinIO_merge(SpinIO):
    def __init__(self, *args, **kwargs):
        super(SpinIO_merge, self).__init__(*args, **kwargs)
        self.projv = None

    def _set_projection_vectors(self):

        spinat = self.spinat
        ind_atoms = self.ind_atoms
        projv = {}
        for i in range(self.nspin):
            for j in range(i, self.nspin):
                a, b = spinat[(ind_atoms[i], ind_atoms[j]), :]
                v = np.cross(a, b)
                norm = np.linalg.norm(v)
                if norm < 1e-6:
                    projv[i, j] = orthplane(a)
                else:
                    projv[i, j] = (v / norm).reshape((1, 3))
                projv[j, i] = projv[i, j]

        self.projv = projv

    @classmethod
    def load_pickle(cls, path='TB2J_results', fname='TB2J.pickle'):
        obj = super(SpinIO_merge, cls).load_pickle(path=path, fname=fname)
        obj._set_projection_vectors()

        return obj

def read_pickle(path):
    p1 = os.path.join(path, 'TB2J_results', 'TB2J.pickle')
    p2 = os.path.join(path, 'TB2J.pickle')
    if os.path.exists(p1) and os.path.exists(p2):
        print(f" WARNING!: Both file {p1} and {p2} exist. Use default {p1}.")
    if os.path.exists(p1):
        ret = SpinIO_merge.load_pickle(os.path.join(path, 'TB2J_results'))
    elif os.path.exists(p2):
        ret = SpinIO_merge.load_pickle(path)
    else:
        raise FileNotFoundError(f"Cannot find either file {p1} or {p2}")
    return ret

class Merger():
    def __init__(self, *paths, main_path=None, method='structure'):
        if method not in ['structure', 'spin']:
            raise ValueError(f"Unrecognized method '{method}'. Available options are: 'structure' or 'spin'.")
        self.method = method
        self.dat = [read_pickle(path) for path in paths]

        if main_path is None:
            self.main_dat = copy.deepcopy(self.dat[-1])
        else:
            self.main_dat = read_pickle(main_path)
            self.dat.append(copy.deepcopy(self.main_dat))

        self._set_projv()
        
    def _set_projv(self):

        rotated = self.method == 'structure'
        if rotated:
            cell = self.main_dat.atoms.cell.array
            rotated_cells = np.stack(
                [obj.atoms.cell.array for obj in self.dat], axis=0
            )
            R = np.linalg.solve(cell, rotated_cells)

        projv = {}; projv_Jani = {};
        for key in self.main_dat.projv.keys():
            vectors = [obj.projv[key] for obj in self.dat]
            if rotated:
                vectors = [(R[i] @ vectors[i].T).T for i in range(len(R))]
            Jani_projections = [flat_projections(v) for v in vectors]
            projv[key] = np.vstack(vectors)
            projv_Jani[key] = np.vstack(Jani_projections)
        
        self.projv = projv
        self.projv_Jani = projv_Jani

    def merge_Jani(self):
        Jani_dict = {}
        projv_Jani = self.projv_Jani
        for key in self.main_dat.Jani_dict.keys():
            try:
                R, i, j = key
                proj_list = []
                for obj in self.dat:
                    v = obj.projv[i, j]
                    proj_list.append((v @ obj.Jani_dict[key] @ v.T).flatten())
                projections = np.hstack(proj_list)
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err)
            Jani = np.linalg.lstsq(projv_Jani[i, j], projections, rcond=None)[0]
            Jani_dict[key] = np.array([
                [Jani[0], Jani[3], Jani[5]],
                [Jani[3], Jani[1], Jani[4]],
                [Jani[5], Jani[4], Jani[2]]
            ])
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
            projv = self.projv
            for key in self.main_dat.dmi_ddict.keys():
                try:
                    R, i, j = key
                    proj_list = [obj.projv[i, j] @ obj.dmi_ddict[key] for obj in self.dat]
                    projections = np.hstack(proj_list)
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err)
                DMI = np.linalg.lstsq(projv[i, j], projections, rcond=None)[0]
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
