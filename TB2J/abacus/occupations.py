"""
This file is stolen from the hotbit programm, with some modification.
"""

import numpy as np
from scipy.optimize import brentq
import sys

from ase.dft.dos import DOS
from scipy import integrate

# import numba

# from numba import float64, int32

MAX_EXP_ARGUMENT = np.log(sys.float_info.max)

# @numba.vectorize(nopython=True)
# def myfermi(e, mu, width, nspin):
#    x = (e - mu) / width
#    if x < -10:
#        ret = 2.0 / nspin
#    elif x > 10:
#        ret = 0.0
#    else:
#        ret = 2.0 / nspin / (math.exp(x) + 1)
#    return ret


def myfermi(e, mu, width, nspin):
    x = (e - mu) / width
    return np.where(x < 10, 2.0 / (nspin * (np.exp(x) + 1.0)), 0.0)


class Occupations(object):
    def __init__(self, nel, width, wk, nspin=1):
        """
        Initialize parameters for occupations.
        :param nel: Number of electrons
        :param width: Fermi-broadening
        :param wk: k-point weights. eg. If only gamma, [1.0]
        :param nspin(optional): number of spin, if spin=1 multiplicity=2 else, multiplicity=1.
        """
        self.nel = nel
        self.width = width
        self.wk = wk
        self.nk = len(wk)
        self.nspin = nspin

    def get_mu(self):
        """Return the Fermi-level (or chemical potential)."""
        return self.mu

    def fermi(self, mu):
        """
        Occupy states with given chemical potential.
        Occupations are 0...2; without k-point weights
        """
        return myfermi(self.e, mu, self.width, self.nspin)

    def root_function(self, mu):
        """This function is exactly zero when mu is right."""
        f = self.fermi(mu)
        return np.einsum("i, ij->", self.wk, f) - self.nel

    def occupy(self, e, xtol=1e-11):
        """
        Calculate occupation numbers with given Fermi-broadening.

        @param e: e[ind_k,ind_orb] energy of k-point, state a
            Note added by hexu:  With spin=2,e[k,a,sigma], it also work. only the *2 should be removed.
        @param wk: wk[:] weights for k-points
        @param width: The Fermi-broadening

        Returns: fermi[ind_k, ind_orb]
        """
        self.e = e
        eflat = e.flatten()
        ind = np.argsort(eflat)
        e_sorted = eflat[ind]
        if self.nspin == 1:
            m = 2
        elif self.nspin == 2:
            m = 1
        n_sorted = (self.wk[:, None, None] * np.ones_like(e) * m).flatten()[ind]

        sum = n_sorted.cumsum()
        if self.nel < sum[0]:
            ifermi = 0
        elif self.nel > sum[-1]:
            raise ("number of electrons larger than number of orbital*spin")
        else:
            ifermi = np.searchsorted(sum, self.nel)
        try:
            if ifermi == 0:
                elo = e_sorted[0]
            else:
                elo = e_sorted[ifermi - 1]
            if ifermi == len(e_sorted) - 1:
                ehi = e_sorted[-1]
            else:
                ehi = e_sorted[ifermi + 1]
            guess = e_sorted[ifermi]
            dmu = np.max((self.width, guess - elo, ehi - guess))
            mu = brentq(self.root_function, guess - dmu, guess + dmu, xtol=xtol)
            # mu = brent(
            #    self.root_function,
            #    brack=(guess - elo, guess, guess + dmu),
            #    tol=xtol)
        except Exception as E:
            # probably a bad guess
            print("Error in finding Fermi level: ", E)
            dmu = self.width
            if self.nel < 1e-3:
                mu = min(e_sorted) - dmu * 20
            elif self.nel - sum[-1] > -1e-3:
                mu = max(e_sorted) + dmu * 20
            else:
                # mu = brent(
                #         self.root_function,
                #         brack=(e_sorted[0] - dmu * 10,
                #                guess,
                #                e_sorted[-1] + dmu * 10),
                #         tol=xtol)
                mu = brentq(
                    self.root_function,
                    e_sorted[0] - dmu * 20,
                    e_sorted[-1] + dmu * 20,
                    xtol=xtol,
                )

        if np.abs(self.root_function(mu)) > xtol * 1e4:
            # raise RuntimeError(
            #    'Fermi level could not be assigned reliably. Has the system fragmented?'
            # )
            print(
                "Fermi level could not be assigned reliably. Has the system fragmented?"
            )

        f = self.fermi(mu)
        # rho=(self.eigenvecs*f).dot(self.eigenvecs.transpose())

        self.mu, self.f = mu, f
        return f

    def plot(self):
        import pylab as pl

        for ik in range(self.nk):
            pl.plot(self.e[ik, :], self.f[ik, :])
            pl.scatter(self.e[ik, :], self.f[ik, :])
        pl.title("occupations")
        pl.xlabel("energy (Ha)")
        pl.ylabel("occupation")
        pl.show()


class GaussOccupations(Occupations):
    def get_mu(self):
        return self.mu

    def delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -(((self.e - energy) / self.width) ** 2)
        return np.exp(x) / (np.sqrt(np.pi) * self.width)

    def get_dos(self, npts=500):
        eflat = self.e.flatten()
        ind = np.argsort(eflat)
        ##e_sorted = eflat[ind]
        if self.nspin == 1:
            m = 2
        elif self.nspin == 2:
            m = 1
        # n_sorted = (self.wk * np.ones_like(self.e) * m).flatten()[ind]
        dos = np.zeros(npts)
        for w, e_n in zip(self.w_k, self.e_skn[0]):
            for e in e_n:
                dos += w * self.delta(e)

    def root_function(self, mu):
        pass

    # @profile
    def occupy(self, e, xtol=1e-8, guess=0.0):
        self.e = e
        dos = myDOS(kweights=self.wk, eigenvalues=e, width=self.width, npts=501)
        edos = dos.get_energies()
        d = dos.get_dos()
        idos = integrate.cumtrapz(d, edos, initial=0) - self.nel
        # f_idos = interpolate.interp1d(edos, idos)
        # ret = optimize.fmin(f_idos, x0=edos[400], xtol=xtol, disp=True)
        ifermi = np.searchsorted(idos, 0.0)
        # self.mu = ret[0]
        self.mu = edos[ifermi]
        self.f = self.fermi(self.mu)
        return self.f


class myDOS(DOS):
    def __init__(
        self, kweights, eigenvalues, nspin=1, width=0.1, window=None, npts=1001
    ):
        """Electronic Density Of States object.

        calc: calculator object
            Any ASE compliant calculator object.
        width: float
            Width of guassian smearing.  Use width=0.0 for linear tetrahedron
            interpolation.
        window: tuple of two float
            Use ``window=(emin, emax)``.  If not specified, a window
            big enough to hold all the eigenvalues will be used.
        npts: int
            Number of points.

        """
        self.npts = npts
        self.width = width
        # self.w_k = calc.get_k_point_weights()
        self.w_k = kweights
        self.nspins = nspin
        # self.e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
        #                        for k in range(len(self.w_k))]
        #                       for s in range(self.nspins)])
        # self.e_skn -= calc.get_fermi_level()
        self.e_skn = np.array([eigenvalues.T])  # eigenvalues: iband, ikpt

        if window is None:
            emin = None
            emax = None
        else:
            emin, emax = window

        if emin is None:
            emin = self.e_skn.min() - 10 * self.width
        if emax is None:
            emax = self.e_skn.max() + 10 * self.width

        self.energies = np.linspace(emin, emax, npts)

        # if width == 0.0: # To use tetrahedron method
        #    bzkpts = calc.get_bz_k_points()
        #    size, offset = get_monkhorst_pack_size_and_offset(bzkpts)
        #    bz2ibz = calc.get_bz_to_ibz_map()
        #    shape = (self.nspins,) + tuple(size) + (-1,)
        #    self.e_skn = self.e_skn[:, bz2ibz].reshape(shape)
        #    self.cell = calc.atoms.cell

    def get_idos(self):
        e, d = self.get_dos()
        return np.trapz(d, e)

    def delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -(((self.energies - energy) / self.width) ** 2)
        return np.exp(x) / (np.sqrt(np.pi) * self.width)

    def get_dos(self, spin=None):
        """Get array of DOS values.

        The *spin* argument can be 0 or 1 (spin up or down) - if not
        specified, the total DOS is returned.
        """

        if spin is None:
            if self.nspins == 2:
                # Spin-polarized calculation, but no spin specified -
                # return the total DOS:
                return self.get_dos(spin=0) + self.get_dos(spin=1)
            else:
                spin = 0

        dos = np.zeros(self.npts)
        for w, e_n in zip(self.w_k, self.e_skn[spin]):
            for e in e_n:
                dos += w * self.delta(e)
        return dos
