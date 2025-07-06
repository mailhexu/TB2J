"""Module for magnon density of states calculations and plotting."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from ase.dft.dos import DOS

from TB2J.kpoints import monkhorst_pack


@dataclass
class MagnonDOS:
    """Data class for storing magnon DOS data"""

    energies: np.ndarray  # DOS energy points in meV
    dos: np.ndarray  # DOS values in states/meV
    weights: Optional[np.ndarray] = None  # k-point weights
    kpoints: Optional[np.ndarray] = None  # k-points used for DOS

    def save(self, filename: str):
        """Save DOS data to a JSON file.

        Parameters
        ----------
        filename : str
            Output filename (should end in .json)
        """
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "energies": self.energies.tolist(),
            "dos": self.dos.tolist(),
        }
        if self.weights is not None:
            data["weights"] = self.weights.tolist()
        if self.kpoints is not None:
            data["kpoints"] = self.kpoints.tolist()

        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "MagnonDOS":
        """Load DOS data from a JSON file.

        Parameters
        ----------
        filename : str
            Input JSON filename

        Returns
        -------
        MagnonDOS
            Loaded DOS object
        """
        with open(filename) as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        data["energies"] = np.array(data["energies"])
        data["dos"] = np.array(data["dos"])
        if "weights" in data:
            data["weights"] = np.array(data["weights"])
        if "kpoints" in data:
            data["kpoints"] = np.array(data["kpoints"])

        return cls(**data)

    def plot(self, ax=None, color="blue", show=True, filename=None, **plot_kwargs):
        """Plot the magnon DOS.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure
        color : str, optional
            Color for DOS line
        show : bool, optional
            Whether to show plot
        filename : str, optional
            If provided, saves plot to file
        **plot_kwargs : dict
            Additional keyword arguments passed to plot

        Returns
        -------
        matplotlib.axes.Axes
            The plotting axes
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.energies, self.dos, color=color, **plot_kwargs)
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("DOS (states/meV)")
        ax.set_title("Magnon DOS")

        if filename:
            plt.savefig(filename)
        if show:
            plt.show()

        return ax


class MagnonDOSCalculator:
    """Calculator for magnon density of states"""

    def __init__(self, magnon):
        """Initialize DOS calculator

        Parameters
        ----------
        magnon : Magnon
            Magnon object containing exchange parameters
        """
        self.magnon = magnon
        self.kpts = None
        self.weights = None
        self.dos_args = {}

    def estimate_energy_range(self, padding_factor=1.2):
        """Estimate the energy range of eigenvalues.

        Computes eigenvalues at zone center and high-symmetry points at zone boundaries
        to estimate the full range of magnon energies.

        Parameters
        ----------
        padding_factor : float, optional
            Factor to extend the energy window beyond min/max values.
            Default is 1.2 (20% padding).

        Returns
        -------
        tuple
            (min_energy, max_energy) in eV
        """
        # Generate high-symmetry points
        kpoints = np.array(
            [
                [0.0, 0.0, 0.0],  # Γ (zone center)
                [0.5, 0.0, 0.0],  # X
                [0.5, 0.5, 0.0],  # M
                [0.5, 0.5, 0.5],  # R (zone corner)
                [0.0, 0.5, 0.0],  # Y
                [0.0, 0.0, 0.5],  # Z
            ]
        )

        # Calculate eigenvalues at these points
        evals = self.magnon._magnon_energies(kpoints)
        min_energy = evals.min()
        max_energy = evals.max()

        # Add padding and convert to eV
        window_size = max_energy - min_energy
        min_energy = min_energy - (padding_factor - 1) * window_size
        max_energy = max_energy + (padding_factor - 1) * window_size

        return min_energy, max_energy

    def set_kmesh(self, kmesh=[9, 9, 9], gamma=True):
        """Set k-point mesh for DOS calculation.

        Parameters
        ----------
        kmesh : list, optional
            Number of k-points along each direction
        gamma : bool, optional
            Whether to include Gamma point
        """
        self.kpts = monkhorst_pack(kmesh, gamma_center=gamma)
        self.weights = np.ones(len(self.kpts)) / len(self.kpts)

    def get_fermi_level(self):
        return 0.0  # Fermi energy is not used in magnon calculations

    def get_eigenvalues(self, kpt, spin=0):
        """Get eigenvalues at a k-point.

        Parameters
        ----------
        kpt : int
            K-point index
        spin : int, optional
            Spin index (unused)

        Returns
        -------
        numpy.ndarray
            Eigenvalues in eV
        """
        kpoint = self.kpts[kpt]
        evals = self.magnon._magnon_energies(np.array([kpoint]))[0]
        return evals

    def get_dos(self, width=0.1, window=None, npts=1001):
        """Calculate DOS using ASE's DOS module.

        Parameters
        ----------
        width : float, optional
            Gaussian smearing width in eV
        window : tuple, optional
            Energy window (min, max) in eV
        npts : int, optional
            Number of energy points

        Returns
        -------
        MagnonDOS
            Calculated DOS object
        """
        if self.kpts is None:
            self.set_kmesh()

        # Estimate energy window if not provided
        if window is None:
            window = self.estimate_energy_range()

        dos_calc = DOS(self, width=width, window=window, npts=npts)
        energies = dos_calc.get_energies()
        dos_vals = dos_calc.get_dos()

        # Convert to meV
        energies = energies * 1000  # eV to meV
        dos_vals = dos_vals / 1000  # states/eV to states/meV

        return MagnonDOS(
            energies=energies,
            dos=dos_vals,
            weights=self.weights,
            kpoints=self.kpts,
        )

    def get_number_of_spins(self):
        """Required by ASE DOS calculator."""
        return 1

    def get_k_point_weights(self):
        """Required by ASE DOS calculator."""
        return self.weights

    def get_bz_k_points(self):
        """Required by ASE DOS calculator."""
        return self.kpts


def plot_magnon_dos(
    magnon,
    kmesh=[9, 9, 9],
    gamma=True,
    width=0.0005,
    window=None,
    xlim=None,
    npts=1001,
    filename=None,
    save_data=True,
    show=True,
):
    """Convenience function to calculate and plot magnon DOS.

    Parameters
    ----------
    magnon : Magnon
        Magnon object containing exchange parameters
    kmesh : list, optional
        Number of k-points along each direction
    gamma : bool, optional
        Whether to include Gamma point
    width : float, optional
        Gaussian smearing width in eV
    window : tuple, optional
        Energy window (min, max) in eV
    npts : int, optional
        Number of energy points
    filename : str, optional
        Output filename for plot
    save_data : bool, optional
        Whether to save DOS data to JSON
    show : bool, optional
        Whether to show plot

    Returns
    -------
    MagnonDOS
        The calculated DOS object
    """
    calculator = MagnonDOSCalculator(magnon)
    calculator.set_kmesh(kmesh=kmesh, gamma=gamma)
    dos = calculator.get_dos(width=width, window=window, npts=npts)

    # Plot DOS
    dos.plot(filename=filename, show=show)

    # Save data if requested
    if save_data:
        data_file = (
            Path(filename).with_suffix(".json") if filename else Path("magnon_dos.json")
        )
        dos.save(data_file)
        print(f"DOS data saved to {data_file}")

    return dos
