import matplotlib.pyplot as plt
from TB2J.spinham.spin_api import SpinModel


def plot_magnon_band(fname='exchange.xml',
                     npoints=301,
                     show=True,
                     kvectors=None,
                     knames=None,
                     figfname="magnon_band.pdf",
                     Jq=False,
                     kpath_fname='exchange_kpth.txt'):
    m = SpinModel(fname=fname, sc_matrix=None)
    m.plot_magnon_band(kvectors=kvectors,
                       knames=knames,
                       npoints=npoints,
                       kpath_fname=kpath_fname,
                       Jq=Jq,
)
    plt.savefig(figfname)
    if show:
        plt.show()
