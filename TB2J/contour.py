import numpy as np
from scipy.special import roots_legendre
from TB2J.utils import simpson_nonuniform, trapezoidal_nonuniform


class Contour():
    def __init__(self, emin, emax=0.0):
        self.emin = emin
        self.emax = emax
        self.path = None

    def build_path_semicircle(self, npoints, endpoint=True):
        R = (self.emax - self.emin) / 2.0
        R0 = (self.emin + self.emax) / 2.0
        phi = np.linspace(np.pi, 0, num=npoints + 1, endpoint=endpoint)
        p = R0 + R * np.exp(1.0j * phi)
        if endpoint:
            self.path = p
            self.de = np.diff(p)
        else:
            self.path = (p[:-1] + p[1:]) / 2
            self.de = p[1:] - p[:-1]

    def build_path_legendre(self, npoints, endpoint=True):
        p = 13
        x, w = roots_legendre(npoints)
        R = (self.emax - self.emin) / 2.0
        R0 = (self.emin + self.emax) / 2.0
        y1 = -np.log(1 + np.pi * p)
        y2 = 0
        y = (y2 - y1) / 2 * x + (y2 + y1) / 2
        phi = (np.exp(-y) - 1) / p
        path = R0 + R * np.exp(1.0j * phi)
        #weight= -(y2-y1)/2*np.exp(-y)/p*1j*(path-R0)*w
        if endpoint:
            self.path = path
            self.de = np.diff(path)
        else:
            self.path = (path[:-1] + path[1:]) / 2
            self.de = path[1:] - path[:-1]

    def build_path_rectangle(self, height=0.1, nz1=50, nz2=200, nz3=50):
        """
        prepare list of energy for integration.
        The path has three segments:
         emin --1-> emin + 1j*height --2-> emax+1j*height --3-> emax
        """
        nz1, nz2, nz3 = nz1, nz2, nz3
        nz = nz1 + nz2 + nz3
        p = np.zeros(nz + 1, dtype='complex128')
        p[:nz1] = self.emin + np.linspace(0, height, nz1, endpoint=False) * 1j
        p[nz1:nz1 + nz2] = self.emin + height * 1j + np.linspace(
            0, self.emax - self.emin, nz2, endpoint=False)
        p[nz1 + nz2:nz] = self.emax + height * 1j + np.linspace(
            0, -height, nz3, endpoint=False) * 1j
        p[-1] = self.emax  # emax
        self.path = (p[:-1] + p[1:]) / 2
        self.de = p[1:] - p[:-1]

    def integrate(self, f, method='simpson'):
        if method == "trapezoidal":
            integrate = trapezoidal_nonuniform
        elif method == 'simpson':
            integrate = simpson_nonuniform
        else:
            raise ValueError("method")
        return integrate(self.path, f=f)

    @property
    def npoints(self):
        return len(self.path)

    def de(self):
        return self.de

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        plt.plot(self.path.real, self.path.imag, marker='.')
        plt.show()


def test():
    ct = Contour(emin=-16, emax=0)
    #ct.build_path_semicircle(npoints=100)
    #ct.build_path_rectangle()
    ct.build_path_legendre(npoints=50)
    print(ct.npoints)
    ct.plot()


if __name__ == '__main__':
    test()
