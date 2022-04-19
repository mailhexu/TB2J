import matplotlib.pyplot as plt
from TB2J.spinham.spin_api import SpinModel
from TB2J.io_exchange.io_exchange import SpinIO
from TB2J.io_exchange.io_txt import write_Jq_info
from ase.dft.kpoints import monkhorst_pack
from ase.cell import Cell
import numpy as np
from TB2J import __version__


def write_eigen(qmesh, gamma=True, path='./', output_fname='EigenJq.txt', **kwargs):
    m = SpinIO.load_pickle(path)
    m.write_Jq(kmesh=qmesh, path=path, gamma=gamma,
               output_fname=output_fname, **kwargs)


def plot_magnon_band(fname='exchange.xml',
                     path='./',
                     npoints=301,
                     show=True,
                     kvectors=None,
                     knames=None,
                     figfname="magnon_band.pdf",
                     supercell_matrix=np.eye(3),
                     Jq=False,
                     kpath_fname='exchange_kpth.txt',
                     ax=None,
                     **kwargs
                     ):
    m = SpinModel(fname=fname, sc_matrix=None)
    m.set_ham(**kwargs)
    m.plot_magnon_band(kvectors=kvectors,
                       knames=knames,
                       npoints=npoints,
                       kpath_fname=kpath_fname,
                       Jq=Jq,
                       supercell_matrix=supercell_matrix,
                       ax=ax
                       )

    plt.savefig(figfname)
    if show:
        plt.show()
