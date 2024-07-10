from sympair import SymmetryPairFinder, SymmetryPairGroupDict
import numpy as np


class Symmetrizer:
    def __init__(
        self, sympair_group_dict=None, atoms=None, pairs=None, Rlist=None, symprec=1e-8
    ):
        """
        Given a dictionary of J_ijR, symmetrize it.
        """
        if sympair_group_dict is None:
            sympair_group_dict = SymmetryPairFinder(
                atoms=atoms, paris=pairs, Rlist=Rlist, symprec=symprec
            ).group_pairs_by_tag_and_distance()
        self.sympair_group_dict = sympair_group_dict


class TB2JSymmetrizer(Symmetrizer):
    def __init__(self, exc):
        # list of pairs with the index of atoms
        ijRs = exc.ijR_list_index_atom()
        self.pgdict = SymmetryPairFinder(
            atoms=exc.atoms, pairs=ijRs
        ).get_symmetry_pair_group_dict()
        print(self.pgdict)
        self.exc = exc

    def get_pairs_atom_indices(self):
        pass

    def symmetrize_J(self):
        symJdict = {}
        Jdict = self.exc.exchange_Jdict
        ngroup = self.sympair_group_dict
        for igroup, pairgroup in self.pgdict.groups:
            ijRs = pairgroup.get_all_ijR()
            ijRs_spin = (self.exc.ijR_index_atom_to_spin(*ijR) for ijR in ijRs)
            Js = [self.exc.get_J(*ijR_spin) for ijR_spin in ijRs_spin]
            Javg = np.average(Js)
            for i, j, R in ijRs_spin:
                symJdict[R, i, j] = Javg


def symmetrize_J(exc):
    """
    symmetrize the exchange parameters
    parameters:
    exc: exchange
    """
    symmetrizer = TB2JSymmetrizer(exc)

    # symmetrizer.symmetrize_J(exc.Jdict)


def test():
    from TB2J.io_exchange import SpinIO

    path = "/Users/hexu/projects/TB2J_examples/Wannier/SrMnO3_ABINIT_Wannier90/TB2J_results"
    exc = SpinIO.load_pickle(path=path, fname="TB2J.pickle")
    symmetrize_J(exc)


if __name__ == "__main__":
    test()
