import matplotlib.pyplot as plt
import numpy as np


class BandsPlot:
    _UNITS = "meV"
    _NSYSTEMS = 1

    def __init__(self, bands, kpath, xlist=None, **kwargs):
        """Initialize BandsPlot.

        Parameters
        ----------
        bands : array_like
            Band energies
        kpath : list of (index, label) tuples
            K-point labels and their indices
        xlist : list of arrays, optional
            X-coordinates for segmented paths. If None, uses range(nbands)
        **kwargs : dict
            Additional plotting options
        """
        self.bands = bands * 1000  # Convert to meV
        self.kpath = kpath
        self.xlist = xlist

        plot_options = kwargs
        self.linewidth = plot_options.pop("linewidth", 1.5)
        self.color = plot_options.pop("color", "blue")
        self.fontsize = plot_options.pop("fontsize", 12)
        self.ticksize = plot_options.pop("ticksize", 10)
        self.plot_options = plot_options

    def plot(self, filename=None):
        """Plot the band structure.

        Parameters
        ----------
        filename : str, optional
            If provided, saves the plot to this file
        """
        fig, axs = plt.subplots(1, self._NSYSTEMS, constrained_layout=True)

        # Get min/max for y-axis limits
        bmin, bmax = self.bands.min(), self.bands.max()
        ymin = bmin - 0.05 * np.abs(bmin - bmax)
        ymax = bmax + 0.05 * np.abs(bmax - bmin)
        axs.set_ylim([ymin, ymax])

        # Plot bands
        if self.xlist is not None:
            # Plot segments
            start_idx = 0
            for x in self.xlist:
                nbands = x.shape[0]
                segment_bands = self.bands[start_idx : start_idx + nbands].T
                for band in segment_bands:
                    axs.plot(
                        x,
                        band[start_idx : start_idx + nbands],
                        linewidth=self.linewidth,
                        color=self.color,
                        **self.plot_options,
                    )
                start_idx += nbands
            # Set xlim to cover all segments
            axs.set_xlim([self.xlist[0][0], self.xlist[-1][-1]])
        else:
            # Standard continuous plotting
            kdata = np.arange(self.bands.shape[0])
            for band in self.bands.T:
                axs.plot(
                    kdata,
                    band,
                    linewidth=self.linewidth,
                    color=self.color,
                    **self.plot_options,
                )
            axs.set_xlim([0, kdata[-1]])

        # Add k-point labels and vertical lines
        kpoint_labels = list(zip(*self.kpath))
        axs.set_xticks(kpoint_labels[0], kpoint_labels[1], fontsize=self.ticksize)
        axs.vlines(
            x=kpoint_labels[0],
            ymin=ymin,
            ymax=ymax,
            color="black",
            linewidth=self.linewidth / 5,
        )

        axs.set_ylabel(f"Energy ({self._UNITS})", fontsize=self.fontsize)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
