import matplotlib.pyplot as plt
from TB2J.spinham.spin_api import SpinModel

def plot_magnon_band(fname='exchange.xml',
         lattice_type=None,
         npoints=301,
         show=True,
         kvectors=None,
         knames=None,
         kpath_fname='exchange_kpth.txt'):
    m = SpinModel(fname=fname, sc_matrix=None)
    m.plot_magnon_band( lattice_type=lattice_type,
        kvectors=kvectors, knames=knames, npoints=npoints, kpath_fname=kpath_fname)
    plt.savefig('exchange_magnon.png')
    if show:
        plt.show()

