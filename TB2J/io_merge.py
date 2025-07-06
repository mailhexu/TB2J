import copy
import os
import warnings
from itertools import combinations_with_replacement

import numpy as np

from TB2J.io_exchange import SpinIO

u0 = np.zeros(3)
uy = np.array([0.0, 1.0, 0.0])
uz = np.array([0.0, 0.0, 1.0])


def get_Jani_coefficients(a, R=np.eye(3)):
    if len(a) == 1:
        u = a
        v = a
    else:
        u = a[[0, 0, 1]]
        v = a[[0, 1, 1]]

    ur = u @ R.T
    vr = v @ R.T
    coefficients = np.hstack(
        [ur * vr, np.roll(ur, -1, axis=-1) * vr + np.roll(vr, -1, axis=-1) * ur]
    )

    return coefficients, u, v


def get_projections(a, b, tol=1e-2):
    projections = np.empty((2, 3))
    if np.linalg.matrix_rank([a, b], tol=tol) == 1:
        if np.linalg.matrix_rank([a, uy], tol=tol) == 1:
            projections[0] = np.cross(a, uz)
        else:
            projections[0] = np.cross(a, uy)
        projections[1] = np.cross(a, projections[0])
        projections /= np.linalg.norm(projections, axis=-1).reshape(-1, 1)
    else:
        projections[0] = np.cross(a, b)
        projections[0] /= np.linalg.norm(projections[0])
        projections[1] = u0

    return projections


class SpinIO_merge(SpinIO):
    def __init__(self, *args, **kwargs):
        super(SpinIO_merge, self).__init__(*args, **kwargs)
        self.projv = None

    def _set_projection_vectors(self):
        norm = np.linalg.norm(self.spinat, axis=-1).reshape(-1, 1)
        spinat = self.spinat / norm
        idx = [self.ind_atoms[i] for i in self.index_spin if i >= 0]
        projv = {}
        for i, j in combinations_with_replacement(range(self.nspin), 2):
            a, b = spinat[idx][[i, j]]
            projv[i, j] = get_projections(a, b)
            projv[j, i] = projv[i, j]

        self.projv = projv

    @classmethod
    def load_pickle(cls, path="TB2J_results", fname="TB2J.pickle"):
        obj = super(SpinIO_merge, cls).load_pickle(path=path, fname=fname)
        obj._set_projection_vectors()

        return obj


def read_pickle(path):
    p1 = os.path.join(path, "TB2J_results", "TB2J.pickle")
    p2 = os.path.join(path, "TB2J.pickle")
    if os.path.exists(p1) and os.path.exists(p2):
        print(f" WARNING!: Both file {p1} and {p2} exist. Use default {p1}.")
    if os.path.exists(p1):
        ret = SpinIO_merge.load_pickle(os.path.join(path, "TB2J_results"))
    elif os.path.exists(p2):
        ret = SpinIO_merge.load_pickle(path)
    else:
        raise FileNotFoundError(f"Cannot find either file {p1} or {p2}")
    return ret


class Merger:
    def __init__(self, *paths, main_path=None):
        self.dat = [read_pickle(path) for path in paths]

        if main_path is None:
            self.main_dat = copy.deepcopy(self.dat[-1])
        else:
            self.main_dat = read_pickle(main_path)
            self.dat.append(copy.deepcopy(self.main_dat))

        self._set_projv()

    def _set_projv(self):
        cell = self.main_dat.atoms.cell.array
        rotated_cells = np.stack([obj.atoms.cell.array for obj in self.dat], axis=0)
        R = np.linalg.solve(cell, rotated_cells)
        indices = range(len(self.dat))

        proju = {}
        projv = {}
        coeff_matrix = {}
        projectors = {}
        for key in self.main_dat.projv.keys():
            vectors = [obj.projv[key] for obj in self.dat]
            coefficients, u, v = zip(
                *[get_Jani_coefficients(vectors[i], R=R[i]) for i in indices]
            )
            projectors[key] = np.vstack([u[i] @ R[i].T for i in indices])
            coeff_matrix[key] = np.vstack(coefficients)
            proju[key] = np.stack(u)
            projv[key] = np.stack(v)
            if np.linalg.matrix_rank(coeff_matrix[key], tol=1e-2) < 6:
                warnings.warn("""
                    WARNING: The matrix of equations to reconstruct the exchange tensors is
                    close to being singular. This happens when the magnetic moments between
                    different configurations are cloes to being parallel. You need to consider 
                    more rotated spin configurations, otherwise the results might have a large 
                    error.""")

        self.proju = proju
        self.projv = projv
        self.coeff_matrix = coeff_matrix
        self.projectors = projectors

    def merge_Jani(self):
        Jani_dict = {}
        proju = self.proju
        projv = self.projv
        coeff_matrix = self.coeff_matrix
        for key in self.main_dat.Jani_dict.keys():
            try:
                R, i, j = key
                u = proju[i, j]
                v = projv[i, j]
                Jani = np.stack([sio.Jani_dict[key] for sio in self.dat])
                projections = np.einsum("nmi,nij,nmj->nm", u, Jani, v).flatten()
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err
                )
            newJani = np.linalg.lstsq(coeff_matrix[i, j], projections, rcond=1e-2)[0]
            Jani_dict[key] = np.array(
                [
                    [newJani[0], newJani[3], newJani[5]],
                    [newJani[3], newJani[1], newJani[4]],
                    [newJani[5], newJani[4], newJani[2]],
                ]
            )
        self.main_dat.Jani_dict = Jani_dict

    def merge_Jiso(self):
        Jdict = {}
        for key in self.main_dat.exchange_Jdict.keys():
            try:
                J = np.mean([obj.exchange_Jdict[key] for obj in self.dat])
            except KeyError as err:
                raise KeyError(
                    "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                    % err
                )
            Jdict[key] = J
        self.main_dat.exchange_Jdict = Jdict

    def merge_DMI(self):
        dmi_ddict = {}
        if all(obj.has_dmi for obj in self.dat):
            projectors = self.projectors
            proju = self.proju
            for key in self.main_dat.dmi_ddict.keys():
                try:
                    R, i, j = key
                    u = proju[i, j]
                    DMI = np.stack([sio.dmi_ddict[key] for sio in self.dat])
                    projections = np.einsum("nmi,ni->nm", u, DMI).flatten()
                except KeyError as err:
                    raise KeyError(
                        "Can not find key: %s, Please make sure the three calculations use the same k-mesh and same Rcut."
                        % err
                    )
                newDMI = np.linalg.lstsq(projectors[i, j], projections, rcond=4e-1)[0]
                dmi_ddict[key] = newDMI
            self.main_dat.dmi_ddict = dmi_ddict

    def standardize(self):
        # make sure that the Jani has the trace of zero
        Jdict = self.main_dat.exchange_Jdict
        Jani_dict = self.main_dat.Jani_dict
        for key in self.main_dat.Jani_dict.keys():
            Jani = self.main_dat.Jani_dict[key]
            shift = np.trace(Jani) / 3.0
            Jani_dict[key] -= shift * np.eye(3)
            Jdict[key] += shift
        self.main_dat.Jani_dict = Jani_dict
        self.main_dat.exchange_Jdict = Jdict


def merge(*paths, main_path=None, save=True, write_path="TB2J_results"):
    m = Merger(*paths, main_path=main_path)
    m.merge_Jiso()
    m.merge_DMI()
    m.merge_Jani()
    m.standardize()

    if save:
        m.main_dat.write_all(path=write_path)
    return m.dat
