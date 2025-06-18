import numpy as np

from TB2J.io_exchange import SpinIO


class MagnonIO:
    """Handle IO operations for magnon calculations"""

    def __init__(self, exc: SpinIO):
        """
        Initialize MagnonIO with a SpinIO instance

        Parameters
        ----------
        exc : SpinIO
            SpinIO instance containing exchange information
        """
        self.exc = exc

    def get_nspin(self):
        """Get number of spins"""
        return self.exc.get_nspin()

    def get_ind_atoms(self):
        """Get atom indices"""
        return self.exc.ind_atoms

    def get_magmom(self):
        """Get magnetic moments"""
        nspin = self.get_nspin()
        return np.array([self.exc.spinat[self.exc.iatom(i)] for i in range(nspin)])

    def get_rlist(self):
        """Get R-vectors list"""
        return self.exc.Rlist

    def get_jtensor(self, asr=False, iso_only=False):
        """
        Get full J tensor for R-list

        Parameters
        ----------
        asr : bool, optional
            Acoustic sum rule, default False
        iso_only : bool, optional
            Only isotropic interactions, default False
        """
        return self.exc.get_full_Jtensor_for_Rlist(asr=asr, iso_only=iso_only)
