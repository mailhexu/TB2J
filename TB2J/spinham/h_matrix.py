import numpy as np
import matplotlib.pyplot as plt
import aiida
from aiida_tb2j.data import ExchangeData


def plot_dispersion(bands, kpoint_labels, color="blue", title=None):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(6, 6 / 1.618)

    """
    Plot the bands
    """
    kpoints = np.arange(len(bands))
    axs.plot(kpoints, bands, color=color, linewidth=1.5)

    """
    Plot the symmetry points
    """
    bmin = bands.min()
    bmax = bands.max()
    ymin = bmin - 0.05 * np.abs(bmin - bmax)
    ymax = bmax + 0.05 * np.abs(bmax - bmin)
    axs.set_xticks(kpoint_labels[0], kpoint_labels[1], fontsize=10)
    axs.vlines(x=kpoint_labels[0], ymin=ymin, ymax=ymax, color="black", linewidth=0.5)
    axs.set_xlim([0, len(kpoints)])
    axs.set_ylim([ymin, ymax])

    if title is not None:
        plt.title(title, fontsize=10)

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--pickle_filename",
        type=str,
        help="Path of the 'TB2J.pickle' file.",
        required=True,
    )
    args = parser.parse_args()

    """
    Right now the implementation depends on AiiDA and so we must create and load an AiiDA profile,
    even if we do not store any information on a data base.
    """
    aiida.load_profile()
    """
    Create an ExchangeData object with the informations from the TB2J.pickle file
    """
    exchange = ExchangeData.load_tb2j(
        pickle_file=args.pickle_filename, isotropic=False, pbc=(True, True, True)
    )
    """
    Compute the magnon band structure along a high symmetry path generated with
    the ASE package. The informations is stored in an AiiDA BandsData object.
    Here tol is the symmetry tolerance to determine the space group of the system.
    They are in units of eV
    """
    magnon_data = exchange.get_magnon_bands(
        npoints=300, tol=1e-1, with_DMI=True, with_Jani=True
    )
    magnon_bands = 1000 * magnon_data.get_bands()  # Convert to meV
    raw_labels = [
        (k, "$\Gamma$") if s == "GAMMA" else (k, s) for k, s in magnon_data.labels
    ]
    kpoint_labels = list(zip(*raw_labels))
    plot_dispersion(magnon_bands, kpoint_labels, color="blue", title="Magnon Bands")
    """
    We can also obtain the dynamical matrix h instead of the actual magnon bands. The result
    is stored in a numpy array with shape (number of kpoints, 2*natoms, 2*natoms)
    """
    kpoints = (
        magnon_data.get_kpoints()
    )  # The shape of the kpoints must be (nkpoints, 3)
    h_matrix = 1000 * exchange._H_matrix(
        kpoints, with_DMI=True, with_Jani=True
    )  # Convert to meV
    h_dispersion = np.linalg.eigvalsh(
        h_matrix
    )  # We can also get the eigenvectors with np.linalg.eigh
    plot_dispersion(
        h_dispersion, kpoint_labels, color="red", title="h matrix dispersion"
    )
