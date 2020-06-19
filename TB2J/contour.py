import numpy as np

class Contour():
    def __init__(self, emin, emax=0.0):
        self.emin=emin
        self.emax=emax

    def build_path_semicircle(self, npoints):
        R0= (self.emin+self.emax)/2.0
        R= (self.emax-self.emin)/2.0
        phi=np.linspace(np.pi, 0, num=npoints+1, endpoint=True)
        p=R0+R*np.exp(1.0j * phi)
        self.path=(p[:-1]+p[1:])/2
        self.de=p[1:]-p[:-1]

    def build_path_rectangle(self, height=0.1, nz1=50, nz2=200, nz3=50):
        """
        prepare list of energy for integration.
        The path has three segments:
         emin --1-> emin + 1j*height --2-> emax+1j*height --3-> emax
        """
        nz1, nz2, nz3 = nz1, nz2, nz3
        nz = nz1+nz2+nz3
        p= np.zeros(nz + 1, dtype='complex128')
        p[:nz1] = self.emin + np.linspace(
            0, height, nz1, endpoint=False) * 1j
        p[nz1:nz1 + nz2] = self.emin + height * 1j + np.linspace(
            0, self.emax - self.emin, nz2, endpoint=False)
        p[nz1 + nz2:nz] = self.emax + height * 1j + np.linspace(
            0, -height, nz3, endpoint=False) * 1j
        p[-1] = self.emax  # emax
        self.path=(p[:-1]+p[1:])/2
        self.de=p[1:]-p[:-1]


    @property
    def npoints(self):
        return len(self.path)

    def de(self):
        return self.de

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax=plt.subplots()
        plt.plot(self.path.real, self.path.imag)
        plt.show()



def test():
    ct=Contour(emin=-16, emax=0)
    ct.build_path_semicircle(npoints=100)
    #ct.build_path_rectangle()
    ct.plot()
if __name__ == '__main__':
    test()
