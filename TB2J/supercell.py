"""
A helper class for building supercells
"""
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import product
from functools import lru_cache
from ase.build import make_supercell
from ase.atoms import Atoms


def close_to_int(x, tol=1e-4):
    return np.allclose(x, np.round(x), atol=tol)


class SupercellMaker(object):

    def __init__(self, sc_matrix, center=False):
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
            raise ValueError('sc_matrix should be 3 or 3*3 matrix')
        self.sc_matrix = sc_matrix
        self.center = center
        if center:
            self.shift = np.array([0.5, 0.5, 0.5])
        else:
            self.shift = np.zeros(3, dtype=float)
        self.inv_scmat = np.linalg.inv(self.sc_matrix.T)
        self.build_sc_vec()
        self.ncell = int(round(abs(np.linalg.det(sc_matrix))))

    def to_red_sc(self, x):
        return np.dot(self.inv_scmat, x) + self.shift

    def build_sc_vec(self):
        eps_shift = np.sqrt(
            2.0) * 1.0E-8  # shift of the grid, so to avoid double counting
        #max_R = np.max(np.abs(self.sc_matrix)) * 3
        if self.center:
            minr = -0.5
            maxr = 0.5
        else:
            minr = 0.0
            maxr = 1.0
        sc_vec = []
        newcell = self.sc_matrix
        scorners_newcell = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                     [0., 1., 1.], [1., 0., 0.], [1., 0., 1.],
                                     [1., 1., 0.], [1., 1., 1.]])
        corners = np.dot(scorners_newcell - self.shift, newcell)
        scorners = corners
        #rep = np.ceil(scorners.ptp(axis=0)).astype('int') + 1
        minrep = np.ceil(np.min(scorners, axis=0)).astype('int')
        maxrep = np.ceil(np.max(scorners, axis=0)).astype('int') + 1

        # sc_vec: supercell vector (map atom from unit cell to supercell)
        # for vec in product(range(rep[0]), range(rep[1]), range(rep[2])):
        for vec in product(range(minrep[0], maxrep[0]),
                           range(minrep[1], maxrep[1]),
                           range(minrep[2], maxrep[2])):
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = self.to_red_sc(vec)
            # check if in the interior
            if (not (tmp_red <= -eps_shift).any()) and (
                    not (tmp_red > 1.0 - eps_shift).any()):
                sc_vec.append(np.array(vec))

        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(self.sc_matrix)))) != num_sc:
            raise Exception(
                "\n\nSuper-cell generation failed! Wrong number of super-cell vectors found."
            )

        sc_vec = np.array(sc_vec)
        self.sc_vec = np.array(sc_vec)
        svtuple = (tuple(s) for s in sc_vec)
        self.sc_vec_dict = dict(zip(svtuple, range(sc_vec.shape[0])))

    def get_R(self):
        return self.sc_vec

    @property
    def R_sc(self):
        return self.sc_vec

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

    def sc_trans_kvector(self, x, kpt, phase=0.0, real=False):
        """
        x is a vector of quantities inside the primitive cell. 
        qpoint is the wavevector
        phase
        x_sc= x * exp(i(qR + phase))

        Note: if x is a 2D m*n array, the first m*n block is the primitive cell.
        [block1, block2, block3, ... block_ncell]
        """
        factor = self.phase_factor(kpt, phase=phase)
        ret = np.kron(factor, x)
        if real:
            ret = np.real(ret)
        return ret

    def Rvector_for_each_element(self, n_ind=1):
        """
        repeat the R_sc vectors.
        """
        return np.kron(self.sc_vec, np.ones((n_ind, 1)))

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
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            for ind in indices:
                sc_ind.append(ind + c * n_ind)
        return sc_ind

    @lru_cache(maxsize=5000)
    def _sc_R_to_pair_ind(self, R_plus_Rv):
        """
        R: initial R vector (e.g. R in (i, jR) pair)
        Rv: translation vector. (e.g. for Rv in self.sc_vec)

        Returns:
        sc_part: R in
        pair_ind:
        """
        R_plus_Rv = np.asarray(R_plus_Rv)
        sc_part = np.floor(self.to_red_sc(R_plus_Rv))  # round down!
        sc_part = np.array(sc_part, dtype=int)
        # find remaining vector in the original reduced coordinates
        orig_part = R_plus_Rv - np.dot(sc_part, self.sc_matrix)
        pair_ind1 = self.sc_vec_dict[tuple(orig_part)]
        return sc_part, pair_ind1

    def sc_jR_to_scjR(self, j, R, Rv, n_basis):
        """
        (j, R) in primitive cell to (j', R') in supercell.
        """
        Rprim, pair_ind = self._sc_R_to_pair_ind(
            tuple(np.array(R) + np.array(Rv)))
        return j + pair_ind * n_basis, tuple(Rprim)

    def sc_i_to_sci(self, i, ind_Rv, n_basis):
        return i + ind_Rv * n_basis

    def sc_ijR_only(self, i, j, R, n_basis):
        ret = []
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            sc_part, pair_ind = self._sc_R_to_pair_ind(tuple(R + cur_sc_vec))
            sc_i = i + c * n_basis
            sc_j = j + pair_ind * n_basis
            ret.append((sc_i, sc_j, tuple(sc_part)))
        return ret

    def sc_jR(self, jlist, Rjlist, n_basis):
        sc_jlist = []
        sc_Rjlist = []
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            # for i , j, ind_R, val in
            for j, Rj in zip(jlist, Rjlist):
                sc_part, pair_ind = self._sc_R_to_pair_ind(
                    tuple(Rj + cur_sc_vec))
                sc_j = j + pair_ind * n_basis
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
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            # for i , j, ind_R, val in
            for (i, j, ind_R), val in terms.items():
                sc_part, pair_ind = self._sc_R_to_pair_ind(
                    tuple(ind_R + cur_sc_vec))
                # index of "from" and "to" hopping indices
                sc_i = i + c * n_basis
                sc_j = j + pair_ind * n_basis

                # hi = self._hoppings[h][1] + c * self._norb
                # hj = self._hoppings[h][2] + pair_ind * self._norb
                ret_dict[(sc_i, sc_j, tuple(sc_part))] = val
        return ret_dict

    def sc_Rlist_HR(self, Rlist, HR, n_basis):
        """
        terms: H[R][i,j] = val
        ========================
        terms: either list of [i, j, R, val] or  dict{(i,j, R): val}
        pos: reduced positions in the unit cell.
        Returns:
        =======================
        """
        sc_Rlist = []
        sc_HR = []
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            # for i , j, ind_R, val in
            for iR, R in enumerate(Rlist):
                H = HR[iR]
                sc_part, pair_ind = self._sc_R_to_pair_ind(
                    tuple(R + cur_sc_vec))
                sc_Rlist.append(sc_part)
                sc_val = np.zeros((n_basis * self.ncell, n_basis * self.ncell),
                                  dtype=HR.dtype)
                for i in range(n_basis):
                    for j in range(n_basis):
                        sc_i = i + c * n_basis
                        sc_j = j + pair_ind * n_basis
                        sc_val[sc_i, sc_j] = H[i, j]
                sc_HR.append(sc_val)
        return np.array(sc_Rlist, dtype=int), np.array(sc_HR)

    def sc_RHdict(self, RHdict, n_basis):
        """
        terms: H[R][i,j] = val
        ========================
        terms: either list of [i, j, R, val] or  dict{(i,j, R): val}
        pos: reduced positions in the unit cell.
        Returns:
        =======================
        """
        sc_RHdict = defaultdict(lambda: np.zeros(
            (n_basis * self.ncell, n_basis * self.ncell), dtype=H.dtype))
        for c, cur_sc_vec in enumerate(
                self.sc_vec):  # go over all super-cell vectors
            for R, H in RHdict.items():
                sc_part, pair_ind = self._sc_R_to_pair_ind(
                    tuple(R + cur_sc_vec))
                ii = c * n_basis
                jj = pair_ind * n_basis
                sc_RHdict[tuple(sc_part)][ii:ii + n_basis,
                                          jj:jj + n_basis] += H
        return sc_RHdict

    def sc_RHdict_notrans(self, RHdict, n_basis, Rshift=(0, 0, 0)):
        sc_RHdict = defaultdict(lambda: np.zeros(
            (n_basis * self.ncell, n_basis * self.ncell), dtype=H.dtype))
        cur_sc_vec = np.array(Rshift)
        for R, H in RHdict.items():
            sc_part, pair_ind = self._sc_R_to_pair_ind(
                tuple(np.array(R) + cur_sc_vec))
            c = self.sc_vec_dict[Rshift]
            ii = c * n_basis
            jj = pair_ind * n_basis
            sc_RHdict[tuple(sc_part)][ii:ii + n_basis, jj:jj + n_basis] += H
        return sc_RHdict

    def sc_H_RpRk_notrans(self, Rplist, Rklist, n_basis, Rpprime, H):
        """
        For a given perturbation at Rp',
        <Rm|Rp'=Rp+Rm|Rk+Rm>
        =H(Rp,Rk)=<0|Rp|Rk> is a matrix of nbasis*nbasis
        First: Rm = Rp'-Rp, Rk+Rm = Rp'-Rp+Rm
        Input: Rplist, Rklist, H
        H: [iRg, iRk, ibasis, ibasis]
        """
        sc_RHdict = defaultdict(lambda: np.zeros(
            (n_basis * self.ncell, n_basis * self.ncell), dtype=H.dtype))
        for iRp, Rp in enumerate(Rplist):
            Rm = np.array(Rpprime) - np.array(Rp)

            sc_part_i, pair_ind_i = self._sc_R_to_pair_ind(tuple(np.array(Rm)))
            ii = pair_ind_i * n_basis
            if tuple(sc_part_i) == (0, 0, 0):
                for iRk, Rk in enumerate(Rklist):
                    sc_part_j, pair_ind_j = self._sc_R_to_pair_ind(
                        tuple(np.array(Rk) + np.array(Rm)))
                    jj = pair_ind_j * n_basis
                    sc_RHdict[tuple(sc_part_j)][ii:ii + n_basis, jj:jj +
                                                n_basis] += H[iRp, iRk, :, :]
                # elif tuple(sc_part_j) == (0, 0, 0):
                #    sc_RHdict[tuple(-sc_part_j)][jj:jj + n_basis,
                #                                 ii:ii + n_basis] += H[iRp, iRk, :, :].T.conj()

        return sc_RHdict

    def sc_atoms(self, atoms):
        """
        This function is compatible with ase.build.make_supercell.
        They should produce the same result.
        """
        sc_cell = self.sc_cell(atoms.get_cell())
        sc_pos = self.sc_pos(atoms.get_scaled_positions())
        sc_numbers = self.sc_trans_invariant(atoms.get_atomic_numbers())
        sc_magmoms = self.sc_trans_invariant(
            atoms.get_initial_magnetic_moments())
        return Atoms(cell=sc_cell,
                     scaled_positions=sc_pos,
                     numbers=sc_numbers,
                     magmoms=sc_magmoms)

    def phase_factor(self, qpoint, phase=0, real=True):
        f = np.exp(2j * np.pi * np.einsum('i, ji -> j', qpoint, self.sc_vec) +
                   1j * phase)
        if real:
            f = np.real(f)
        return f

    def modulation_function_R(self, func):
        return [func(R) for R in self.R_sc]

    def _make_translate_maps(positions, basis, sc_mat, tol_r=1e-4):
        """
        find the mapping between supercell and translated cell.
        Returns:
        ===============
        A N * nbasis array.
        index[i] is the mapping from supercell to translated supercell so that
        T(r_i) psi = psi[indices[i]].

        """
        a1 = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
        sc = make_supercell(a1, self._scmat)
        rs = sc.get_scaled_positions()

        indices = np.zeros([len(rs), len(positions)], dtype='int32')
        for i, ri in enumerate(rs):
            inds = []
            Tpositions = positions + np.array(ri)
            for i_basis, pos in enumerate(positions):
                for j_basis, Tpos in enumerate(Tpositions):
                    dpos = Tpos - pos
                    if close_to_int(dpos, tol_r) and (self._basis[i_basis]
                                                      == self._basis[j_basis]):
                        indices[i, j_basis] = i_basis

        self._trans_rs = rs
        self._trans_indices = indices


def smod(x):
    x = np.mod(x, 1)
    return x if x < 0.5 else x - 1


smod = np.vectorize(smod)


def map_to_primitive(atoms, primitive_atoms, offset=(0, 0, 0)):
    """
    Find the mapping of a supercell to a primitive cell.
    :param atoms: the positions of atoms
    :param primitive_atoms: 
    :param offset: 
    :param 0: 
    :param 0): 

    """
    ilist = []
    Rlist = []
    offset = np.array(offset, dtype=float)
    ppos = primitive_atoms.get_positions()
    pos = atoms.get_positions()
    cell = primitive_atoms.get_cell()
    for p in pos:
        found = False
        for i, pp in enumerate(ppos):
            res0 = np.linalg.solve(cell.T, p - pp)
            res = smod(res0)
            if np.linalg.norm(res) < 0.01:
                found = True
                R = res0 - res
                ilist.append(i)
                Rlist.append(R)
                break
        if not found:
            print("Not found")
            ilist.append(-1)
            Rlist.append([-999, -999, -999])
    return np.array(ilist, dtype=int), np.array(Rlist, dtype=int)


def find_primitive_cell(atoms,
                        sc_matrix,
                        origin_atom_id=None,
                        thr=1e-5,
                        perfect=True):
    """
    Find a primitive cell atoms from the supercell atom structure.
    :param atoms: the supercell structure.
    :param sc_matrix: the matrix which maps the primitive cell to supercell.
    :param origin: the origin of the primitive cell.
    :param thr: the atoms which the reduced position is within -thr to 1.0+thr are considered as inside the primitive atoms
    :params perfect: True|False, whether the primitive cell should contains the same number of atoms .
    :returns: (patoms, selected)
    patoms: the primitive cell atoms
    selected: the selected indices of the atoms in the supercell.
    """
    scell = atoms.get_cell().array
    inv_scmat = np.linalg.inv(sc_matrix)
    pcell = scell@inv_scmat
    print(f"{inv_scmat=}")

    xcart = atoms.get_positions()
    xred = atoms.get_scaled_positions()
    print(xred)
    if origin_atom_id is not None:
        origin = xred[origin_atom_id]
    else:
        origin = np.zeros(3)
    # check if some atom is exactly at the origin.
    # if so, shift the positions by thr so that this atom is inside the cell
    # if np.any(np.linalg.norm(xred - origin[None, :], axis=1) < thr):
    #    xred += thr
    #xred += 0.05

    sc_xred = xred@sc_matrix
    print(sc_xred)
    #np.all(sc_xred<1 and sc_xred>=0.0)
    # print(sc_xred<1)
    x = np.logical_and(sc_xred < 1+thr, sc_xred >= -thr)
    print(np.all(x, axis=1))
    selected = np.where(np.all(x, axis=1))[0]
    print(selected)
    symbols = atoms.get_chemical_symbols()
    psymbols = [symbols[i] for i in selected]
    patoms = Atoms(symbols=psymbols, positions=xcart[selected], cell=pcell)
    ncell = abs(np.linalg.det(sc_matrix))
    natom = len(atoms)
    if perfect:
        assert len(symbols) == int(
            natom/ncell), "The number of atoms in the found primitive cell does not equal to natom in the supercell divided by the size of the cell"
    return patoms, selected


def test_find_primitive_cell():
    atoms = Atoms('HO', positions=[[0, 0, 0], [0, 0.2, 0]], cell=[1, 1, 1])
    sc_matrix = np.diag([1, 1, 3])
    atoms2 = make_supercell(atoms, sc_matrix)
    patoms = find_primitive_cell(atoms2, sc_matrix)


def test():
    sc_mat = np.diag([1, 1, 3])
    #sc_mat[0, 1] = 2
    spm = SupercellMaker(sc_matrix=sc_mat, center=False)
    print("sc_vec", spm.sc_vec)
    print(spm.sc_cell([1, 1, 1]))
    print(spm.sc_pos([[0.5, 1, 1]]))
    print(spm.sc_pos([[0.5, 1, 0.5]]))
    print(spm.sc_trans_invariant(['Fe']))
    print(spm.sc_ijR({
        (0, 0, (0, 0, 1)): 1.2,
        (1, 1, (0, 0, 1)): 1.2,
    }, 2))
    print(spm.sc_index(indices=(1, 2)))
    print(spm.sc_index(indices=(1, 2), n_ind=4))
    from ase.atoms import Atoms
    atoms = Atoms('HO', positions=[[0, 0, 0], [0, 0.2, 0]], cell=[1, 1, 1])
    from ase.build import make_supercell
    atoms2 = make_supercell(atoms, sc_mat)
    atoms3 = spm.sc_atoms(atoms)
    # print(atoms2.get_positions())
    # print(atoms3.get_positions())
    assert (atoms2 == atoms3)
    assert (atoms2.get_positions() == atoms3.get_positions()).all()


if __name__ == '__main__':
    # test()
    test_find_primitive_cell()
