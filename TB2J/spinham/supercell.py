"""
A helper class for building supercells
"""
import numpy as np
from collections import OrderedDict
from itertools import product
from functools import lru_cache


class SupercellMaker(object):
    def __init__(self, sc_matrix):
        """
        a helper class for making supercells.
        sc_matrix, supercell matrix. sc_matrix .dot. unitcell = supercell
        """
        sc_matrix = np.array(sc_matrix, dtype=int)
        if len(sc_matrix.flatten()) == 3:
            sc_matrix = np.diag(sc_matrix)
        elif sc_matrix.shape == (3, 3):
            pass
        else:
            raise ValueError("sc_matrix should be 3 or 3*3 matrix")
        self.sc_matrix = sc_matrix
        self.inv_scmat = np.linalg.inv(self.sc_matrix.T)
        self.build_sc_vec()
        self.ncell = int(round(abs(np.linalg.det(sc_matrix))))

    def to_red_sc(self, x):
        return np.dot(self.inv_scmat, x)

    def rotate_vector(self, vec):
        """
        a vector in the new axis.
        """

    def build_sc_vec(self):
        eps_shift = (
            np.sqrt(2.0) * 1.0e-8
        )  # shift of the grid, so to avoid double counting
        # max_R = np.max(np.abs(self.sc_matrix)) * 3
        sc_vec = []
        newcell = self.sc_matrix
        scorners_newcell = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        corners = np.dot(scorners_newcell, newcell)
        scorners = corners
        rep = np.ceil(scorners.ptp(axis=0)).astype("int") + 1

        # sc_vec: supercell vector (map atom from unit cell to supercell)
        for vec in product(range(rep[0]), range(rep[1]), range(rep[2])):
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = self.to_red_sc(vec)
            # check if in the interior
            if (not (tmp_red <= -1.0 * eps_shift).any()) and (
                not (tmp_red > 1.0 - eps_shift).any()
            ):
                sc_vec.append(np.array(vec))

        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(self.sc_matrix)))) != num_sc:
            raise Exception(
                "\n\nSuper-cell generation failed! Wrong number of super-cell vectors found."
            )

        self.sc_vec = sc_vec

    def get_R(self):
        return self.sc_vec

    @property
    def R_sc(self):
        return self.sc_vec

    def build_sc_vec2(self):
        """
        TODO rewrite this to reduce memory usage.
        """
        max_R = np.max(np.abs(self.sc_matrix)) * 3
        sc_cands = []
        for i in range(-max_R, max_R + 1):
            for j in range(-max_R, max_R + 1):
                for k in range(-max_R, max_R + 1):
                    sc_cands.append(np.array([i, j, k]))

        # sc_vec: supercell vector (map atom from unit cell to supercell)
        sc_vec = []
        eps_shift = (
            np.sqrt(2.0) * 1.0e-8
        )  # shift of the grid, so to avoid double counting
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = self.to_red_sc(vec)
            # check if in the interior
            inside = True
            for t in tmp_red:
                if t <= -1.0 * eps_shift or t > 1.0 - eps_shift:
                    inside = False
            if inside:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)
        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(self.sc_matrix)))) != num_sc:
            raise Exception(
                "\n\nSuper-cell generation failed! Wrong number of super-cell vectors found."
            )

        self.sc_vec = sc_vec

    def sc_cell(self, cell):
        cell = np.array(cell)
        if len(cell.flatten()) == 3:
            cell = np.diag(cell)
        return np.dot(self.sc_matrix, cell)

    def sc_pos(self, positions, return_R=False):
        """
        pos -> pos in supercell (reduced.)
        """
        sc_pos = []
        sc_R = []
        for cur_sc_vec in self.sc_vec:  # go over all super-cell vectors
            for pos in positions:
                # shift orbital and compute coordinates in
                # reduced coordinates of super-cell
                sc_pos.append(self.to_red_sc(pos + cur_sc_vec))
                sc_R.append(cur_sc_vec)
        if return_R:
            return sc_pos, sc_R
        else:
            return sc_pos

    def sc_trans_invariant(self, q, return_R=False):
        """
        translation invariant quantities. Like on-site energy of tight binding,
        chemical symbols, magnetic moments of spin.
        """
        sc_q = []
        sc_R = []  # supercell R
        for cur_sc_vec in self.sc_vec:  # go over all super-cell vectors
            for qitem in q:
                sc_q.append(qitem)
                sc_R.append(cur_sc_vec)
        if return_R:
            return sc_q, sc_R
        else:
            return sc_q

    def sc_index(self, indices, n_ind=None):
        """
        Note that the number of indices could be inequal to the repeat period.
        e.g. for n_orb of orbitals, the indices of atoms iatom for each orbital.
        In that case, in the second unit cell (c=1 here), iatom-> iatom+n_ind,
        where n_ind=natoms in primitive cell.
        """
        sc_ind = []
        if n_ind is None:
            n_ind = len(indices)
        for c, cur_sc_vec in enumerate(self.sc_vec):  # go over all super-cell vectors
            for ind in indices:
                sc_ind.append(ind + c * n_ind)
        return sc_ind

    @lru_cache(maxsize=10000)
    def _sc_R_to_pair_ind(self, R_plus_Rv):
        """
        R: initial R vector (e.g. R in (i, jR) pair)
        Rv: translation vector. (e.g. for Rv in self.sc_vec)

        Returns:
        sc_part: R in
        pair_ind:
        """
        R_plus_Rv = np.array(R_plus_Rv)
        sc_part = np.floor(self.to_red_sc(R_plus_Rv))  # round down!
        sc_part = np.array(sc_part, dtype=int)
        # find remaining vector in the original reduced coordinates
        orig_part = R_plus_Rv - np.dot(sc_part, self.sc_matrix)
        # remaining vector must equal one of the super-cell vectors
        pair_ind = None
        for p, pair_sc_vec in enumerate(self.sc_vec):
            if False not in (pair_sc_vec == orig_part):
                if pair_ind is not None:
                    raise Exception("\n\nFound duplicate super cell vector!")
                pair_ind = p
        if pair_ind is None:
            raise Exception("\n\nDid not find super cell vector!")
        return sc_part, pair_ind

    @lru_cache(maxsize=10000)
    def sc_jR_to_scjR(self, j, R, Rv, n_basis):
        """
        (j, R) in primitive cell to (j', R') in supercell.
        """
        Rprim, pair_ind = self._sc_R_to_pair_ind(tuple(np.array(R) + np.array(Rv)))
        return j + pair_ind * n_basis, tuple(Rprim)

    @lru_cache(maxsize=10000)
    def sc_i_to_sci(self, i, ind_Rv, n_basis):
        return i + ind_Rv * n_basis

    def sc_ijR_only(self, i, j, R, n_basis):
        ret = []
        for c, cur_sc_vec in enumerate(self.sc_vec):  # go over all super-cell vectors
            sc_part, pair_ind = self._sc_R_to_pair_ind(tuple(R + cur_sc_vec))
            sc_i = i + c * n_basis
            sc_j = j + pair_ind * n_basis
            ret.append((sc_i, sc_j, tuple(sc_part)))
        return ret

    def sc_jR(self, jlist, Rjlist, n_basis):
        sc_jlist = []
        sc_Rjlist = []
        for c, cur_sc_vec in enumerate(self.sc_vec):  # go over all super-cell vectors
            # for i , j, ind_R, val in
            for j, Rj in zip(jlist, Rjlist):
                sc_part, pair_ind = self._sc_R_to_pair_ind(tuple(Rj + cur_sc_vec))
                sc_j = j + pair_ind * n_basis
                # ret_dict[(sc_i, sc_j, tuple(sc_part))] = val
                sc_jlist.append(sc_j)
                sc_Rjlist.append(tuple(sc_part))
        return sc_jlist, sc_Rjlist

    def sc_ijR(self, terms, n_basis):
        """
        # TODO very slow when supercell is large, should improve it.
        map Val(i, j, R) which is a funciton of (R+rj-ri) to supercell.
        e.g. hopping in Tight binding. exchange in heisenberg model,...
        Args:
        ========================
        terms: either list of [i, j, R, val] or  dict{(i,j, R): val}
        pos: reduced positions in the unit cell.
        Returns:
        =======================
        """
        ret_dict = OrderedDict()
        for c, cur_sc_vec in enumerate(self.sc_vec):  # go over all super-cell vectors
            # for i , j, ind_R, val in
            for (i, j, ind_R), val in terms.items():
                # sc_part = np.floor(
                # self.to_red_sc(ind_R + cur_sc_vec))  # round down!

                # pair_ind=self._sc_R_to_pair_ind(tuple(ind_R), tuple(cur_sc_vec))
                sc_part, pair_ind = self._sc_R_to_pair_ind(tuple(ind_R + cur_sc_vec))
                # index of "from" and "to" hopping indices
                sc_i = i + c * n_basis
                sc_j = j + pair_ind * n_basis

                # hi = self._hoppings[h][1] + c * self._norb
                # hj = self._hoppings[h][2] + pair_ind * self._norb
                ret_dict[(sc_i, sc_j, tuple(sc_part))] = val
        return ret_dict

    def sc_atoms(self, atoms):
        """
        This function is compatible with ase.build.make_supercell.
        They should produce the same result.
        """
        from ase.atoms import Atoms

        sc_cell = self.sc_cell(atoms.get_cell())
        sc_pos = self.sc_pos(atoms.get_scaled_positions())
        sc_numbers = self.sc_trans_invariant(atoms.get_atomic_numbers())
        sc_magmoms = self.sc_trans_invariant(atoms.get_initial_magnetic_moments())
        return Atoms(
            cell=sc_cell,
            scaled_positions=sc_pos,
            numbers=sc_numbers,
            magmoms=sc_magmoms,
        )

    def phase(self, qpoint):
        return np.exp(2j * np.pi * np.einsum("i, ji -> j", qpoint, self.sc_vec))


def smod(x):
    x = np.mod(x, 1)
    return x if x < 0.5 else x - 1


smod = np.vectorize(smod)


def map_to_primitive(atoms, primitive_atoms, offset=(0, 0, 0)):
    ilist = []
    Rlist = []
    offset = np.array(offset, dtype=float)
    ppos = primitive_atoms.get_positions()
    pos = atoms.get_positions()
    cell = primitive_atoms.get_cell()
    for p in pos:
        found = False
        for i, pp in enumerate(ppos):
            res0 = np.linalg.solve(cell, p - pp)
            res = smod(res0)
            if np.linalg.norm(res) < 0.01:
                found = True
                R = res0 - res
                ilist.append(i)
                Rlist.append(R)
        if not found:
            print("Not found")
    return np.array(ilist, dtype=int), np.array(Rlist, dtype=int)


def test():
    sc_mat = np.diag([1, 1, 2])
    # sc_mat[0, 1] = 2
    spm = SupercellMaker(sc_matrix=sc_mat)
    print(spm.sc_cell([1, 1, 1]))
    print(spm.sc_pos([[0.5, 1, 1]]))
    print(spm.sc_trans_invariant(["Fe"]))
    print(
        spm.sc_ijR(
            {
                (0, 0, (0, 0, 1)): 1.2,
                (1, 1, (0, 0, 1)): 1.2,
            },
            [(0, 0, 0)],
        )
    )
    print(spm.sc_index(indices=(1, 2)))
    print(spm.sc_index(indices=(1, 2), n_ind=4))
    from ase.atoms import Atoms

    atoms = Atoms("HO", positions=[[0, 0, 0], [0, 0.2, 0]], cell=[1, 1, 1])
    from ase.build import make_supercell

    atoms2 = make_supercell(atoms, sc_mat)
    atoms3 = spm.sc_atoms(atoms)
    # print(atoms2.get_positions())
    # print(atoms3.get_positions())
    assert atoms2 == atoms3
    assert (atoms2.get_positions() == atoms3.get_positions()).all()


if __name__ == "__main__":
    test()
