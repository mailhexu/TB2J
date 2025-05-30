import matplotlib.pyplot as plt
import numpy as np


class BandsPlot:
    _UNITS = "meV"
    _NSYSTEMS = 1

    def __init__(self, bands, kpath, **kwargs):
        self.bands = bands
        self.kpath = kpath
        self.bands *= 1000

        plot_options = kwargs
        self.linewidth = plot_options.pop("linewidth", 1.5)
        self.color = plot_options.pop("color", "blue")
        self.fontsize = plot_options.pop("fontsize", 12)
        self.ticksize = plot_options.pop("ticksize", 10)
        self.plot_options = plot_options

    def plot(self, filename=None):
        fig, axs = plt.subplots(1, self._NSYSTEMS, constrained_layout=True)

        kdata = np.arange(self.bands.shape[0])
        for band in self.bands.T:
            axs.plot(
                kdata,
                band,
                linewidth=self.linewidth,
                color=self.color,
                **self.plot_options,
            )

        bmin, bmax = self.bands.min(), self.bands.max()
        ymin, ymax = (
            bmin - 0.05 * np.abs(bmin - bmax),
            bmax + 0.05 * np.abs(bmax - bmin),
        )

        axs.set_ylim([ymin, ymax])
        axs.set_xlim([0, kdata[-1]])

        kpoint_labels = list(zip(*self.kpath))
        axs.set_xticks(*kpoint_labels, fontsize=self.ticksize)
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
            fig.save(filename, dpi=300, bbox_inches="tight")
