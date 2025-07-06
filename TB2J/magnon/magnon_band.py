"""Module for handling magnon band structure data and plotting."""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MagnonBand:
    """Magnon band structure data and plotting functionality.

    Parameters
    ----------
    energies : np.ndarray
        Band energies of shape (nkpts, nbands)
    kpoints : np.ndarray
        k-points coordinates of shape (nkpts, 3)
    kpath_labels : List[Tuple[int, str]]
        List of (index, label) tuples for special k-points
    special_points : dict
        Dictionary mapping k-point names to their coordinates
    xcoords : Optional[Union[np.ndarray, List[np.ndarray]]]
        x-coordinates for plotting. Can be continuous or segmented.
    """

    energies: np.ndarray
    kpoints: np.ndarray
    kpath_labels: List[Tuple[int, str]]
    special_points: dict
    xcoords: Optional[Union[np.ndarray, List[np.ndarray]]] = None

    def __post_init__(self):
        """Convert input arrays to numpy arrays and set default x-coordinates."""
        self.energies = np.array(self.energies)
        self.kpoints = np.array(self.kpoints)

        if self.xcoords is None:
            self.xcoords = np.arange(len(self.kpoints))

    def plot(self, ax=None, filename=None, show=False, shift=0.0, **kwargs):
        """Plot the magnon band structure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes for plotting. If None, creates new figure.
        filename : str, optional
            If provided, saves plot to this file.
        show : bool, optional
            Whether to show the plot on screen. Default is False.
        **kwargs : dict
            Additional arguments passed to plot function:
            - linewidth: float, default 1.5
            - color: str, default 'blue'
            - linestyle: str, default '-'

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        # Plot settings
        linewidth = kwargs.pop("linewidth", 1.5)
        color = kwargs.pop("color", "blue")
        linestyle = kwargs.pop("linestyle", "-")

        # Plot bands
        if isinstance(self.xcoords, list):  # Segmented path
            start_idx = 0
            for x in self.xcoords:
                nbands = x.shape[0]
                segment_bands = self.energies[start_idx : start_idx + nbands].T
                for band in segment_bands:
                    ax.plot(
                        x,
                        band[start_idx : start_idx + nbands] + shift,
                        linewidth=linewidth,
                        color=color,
                        linestyle=linestyle,
                        **kwargs,
                    )
                start_idx += nbands
            ax.set_xlim([self.xcoords[0][0], self.xcoords[-1][-1]])
        else:  # Continuous path
            for band in self.energies.T:
                ax.plot(
                    self.xcoords,
                    band + shift,
                    linewidth=linewidth,
                    color=color,
                    linestyle=linestyle,
                    **kwargs,
                )
            ax.set_xlim([self.xcoords[0], self.xcoords[-1]])

        # Set y-limits with padding
        bmin, bmax = self.energies.min(), self.energies.max()
        ymin = bmin - 0.05 * abs(bmin - bmax)
        ymax = bmax + 0.05 * abs(bmax - bmin)
        ax.set_ylim([ymin, ymax])

        # Add k-point labels and vertical lines
        kpoint_pos = [i for i, _ in self.kpath_labels]
        kpoint_labels = [label for _, label in self.kpath_labels]
        ax.set_xticks(kpoint_pos)
        ax.set_xticklabels(kpoint_labels)
        ax.vlines(
            x=kpoint_pos,
            ymin=ymin,
            ymax=ymax,
            color="black",
            linewidth=linewidth / 5,
        )

        ax.set_ylabel("Energy (meV)")

        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return ax

    def save(self, filename: str):
        """Save band structure data to a JSON file.

        Parameters
        ----------
        filename : str
            Output filename (will append .json if needed)
        """
        if not filename.endswith(".json"):
            filename = filename + ".json"

        data = {
            "kpoints": self.kpoints.tolist(),
            "energies": self.energies.tolist(),
            "kpath_labels": [(int(i), str(l)) for i, l in self.kpath_labels],
            "special_points": {k: v.tolist() for k, v in self.special_points.items()},
            "xcoords": self.xcoords.tolist()
            if isinstance(self.xcoords, np.ndarray)
            else [x.tolist() for x in self.xcoords]
            if self.xcoords is not None
            else None,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> "MagnonBand":
        """Load band structure from a JSON file.

        Parameters
        ----------
        filename : str
            Input JSON filename

        Returns
        -------
        MagnonBand
            Loaded band structure object
        """
        with open(filename) as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        data["kpoints"] = np.array(data["kpoints"])
        data["energies"] = np.array(data["energies"])
        if data["xcoords"] is not None:
            if isinstance(data["xcoords"][0], list):  # Segmented path
                data["xcoords"] = [np.array(x) for x in data["xcoords"]]
            else:  # Continuous path
                data["xcoords"] = np.array(data["xcoords"])
        data["special_points"] = {
            k: np.array(v) for k, v in data["special_points"].items()
        }

        return cls(**data)
