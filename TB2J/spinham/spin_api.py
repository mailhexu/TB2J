import numpy as np
from .hamiltonian import SpinHamiltonian, read_spin_ham_from_file

# from minimulti.spin.mover import SpinMover


class SpinModel:
    def __init__(self, fname=None, sc_matrix=None):
        if fname is not None:
            self.read_from_file(fname)
        else:
            self._ham = SpinHamiltonian()

        if sc_matrix is not None:
            self.make_supercell(sc_matrix)

        # self.mover = SpinMover(self._ham)

    @property
    def ham(self):
        return self._ham

    def add_term(self, term, name=None):
        self._ham.add_Hamiltonian_term(term, name=name)

    @ham.setter
    def ham(self, ham):
        self._ham = ham

    # @property
    # def S(self):
    #    return self.mover.s

    @property
    def nspin(self):
        return self._ham.nspin

    def read_from_file(self, fname):
        self._ham = read_spin_ham_from_file(fname)

    def set(self, **kwargs):
        self.params.set(**kwargs)
        # self.mover.set(
        #     time_step=self.params.time_step,
        #     temperature=self.params.temperature,
        #     total_time=self.params.total_time,
        # )

    def set_ham(self, **kwargs):
        self._ham.set(**kwargs)

    def make_supercell(self, sc_matrix=None, supercell_maker=None):
        self._ham = self._ham.make_supercell(
            sc_matrix=sc_matrix, supercell_maker=supercell_maker
        )
        # self.mover = SpinMover(self._ham)
        return self

    # def run_one_step(self):
    #     self.mover.run_one_step()

    # def run_time(self):
    #     self.mover.run()

    def plot_magnon_band(
        self,
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        supercell_matrix=None,
        npoints=100,
        color="red",
        ax=None,
        kpath_fname=None,
        Jq=False,
    ):
        self._ham.plot_magnon_band(
            kvectors=kvectors,
            knames=knames,
            supercell_matrix=supercell_matrix,
            npoints=npoints,
            color=color,
            ax=ax,
            Jq=Jq,
            kpath_fname=kpath_fname,
        )
