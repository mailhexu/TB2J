"""Module for magnon density of states calculations and plotting."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from ase.dft.dos import DOS

from TB2J.kpoints import monkhorst_pack
from TB2J.magnon.eigenstates import MagnonEigenstateData
from TB2J.magnon.magnon_parameters import (
    MagnonParameters,
    add_common_magnon_args,
    add_dos_specific_args,
    prepare_magnon_from_params,
)


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
        kpoints = self.kpoints if self.kpoints is not None else np.zeros((0, 3))
        energies = np.zeros((len(kpoints), 0))
        data = MagnonEigenstateData(
            calculation_type="dos",
            kpoints=kpoints,
            energies=energies,
            weights=self.weights,
            metadata={"units": {"energies": "eV", "dos_energies": "meV"}},
            plot={
                "kind": "dos",
                "dos": self.dos,
                "dos_energies_mev": self.energies,
            },
        )
        data.save_json(filename)

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

        if data.get("schema_name") == "tb2j.magnon.eigenstates":
            eig = MagnonEigenstateData.from_dict(data)
            plot = eig.plot or {}
            return cls(
                energies=np.array(plot["dos_energies_mev"]),
                dos=np.array(plot["dos"]),
                weights=eig.weights,
                kpoints=eig.kpoints,
            )

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
    export_formats=None,
    export_prefix=None,
    save_wavefunctions=False,
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
    export_formats = ["json"] if export_formats is None else export_formats
    if save_data and "json" in export_formats:
        if export_prefix is not None:
            data_file = Path(export_prefix + ".json")
        else:
            data_file = (
                Path(filename).with_suffix(".json")
                if filename
                else Path("magnon_dos.json")
            )
        dos.save(data_file)
        print(f"DOS data saved to {data_file}")

    if "netcdf" in export_formats or save_wavefunctions:
        if export_prefix is not None:
            prefix = export_prefix
        elif filename:
            prefix = str(Path(filename).with_suffix(""))
        else:
            prefix = "magnon_dos"
        eigenstates = magnon.get_magnon_eigenstates(
            calculator.kpts,
            calculation_type="dos",
            include_wavefunctions=save_wavefunctions,
            weights=calculator.weights,
            plot={
                "kind": "dos",
                "dos": dos.dos,
                "dos_energies_mev": dos.energies,
            },
        )
        if "json" in export_formats and save_wavefunctions:
            eigenstates.save_json(prefix + ".json")
        if "netcdf" in export_formats:
            eigenstates.save_netcdf(prefix + ".nc")

    return dos


def plot_magnon_dos_from_TB2J(params: MagnonParameters):
    """Calculate and plot magnon DOS from TB2J results.

    Parameters
    ----------
    params : MagnonParameters
        Parameters for the calculation

    Returns
    -------
    MagnonDOS
        The calculated DOS object
    """
    magnon = prepare_magnon_from_params(params)

    window = None
    if params.window is not None:
        window = (params.window[0] / 1000, params.window[1] / 1000)

    print("\nCalculating magnon DOS...")
    dos = plot_magnon_dos(
        magnon,
        kmesh=params.kmesh,
        gamma=params.gamma,
        width=params.width,
        window=window,
        npts=params.npts,
        filename=params.filename,
        show=params.show,
        export_formats=params.export_formats,
        export_prefix=params.export_prefix,
        save_wavefunctions=params.save_wavefunctions,
    )

    print(f"\nPlot saved to {params.filename}")
    data_file = Path(params.filename).with_suffix(".json")
    print(f"DOS data saved to {data_file}")

    return dos


def main():
    """Command-line interface for magnon DOS calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate and plot magnon DOS from TB2J results"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=str,
        help="Path to TOML configuration file",
    )
    group.add_argument(
        "--save-config",
        type=str,
        help="Save default configuration to specified TOML file",
    )

    add_common_magnon_args(parser)
    add_dos_specific_args(parser)

    args = parser.parse_args()

    if args.save_config:
        params = MagnonParameters()
        params.to_toml(args.save_config)
        print(f"Saved default configuration to {args.save_config}")
        return

    if args.config:
        params = MagnonParameters.from_toml(args.config)
    else:
        window = None
        if args.window is not None:
            window = tuple(args.window)
        params = MagnonParameters(
            path=args.path,
            filename=args.output,
            Jiso=args.Jiso,
            Jani=args.Jani,
            SIA=getattr(args, "SIA", True),
            DMI=args.DMI,
            Q=args.Q,
            uz_file=args.uz_file,
            spin_conf_file=args.spin_conf_file,
            n=getattr(args, "n", None),
            show=args.show,
            kmesh=args.kmesh,
            gamma=args.gamma,
            width=args.width,
            window=window,
            npts=args.npts,
        )

    plot_magnon_dos_from_TB2J(params)


if __name__ == "__main__":
    main()
