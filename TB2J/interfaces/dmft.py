import os
from abc import ABC, abstractmethod

import numpy as np

# Try to import h5py, but don't fail if not present (unless used)
try:
    import h5py
except ImportError:
    h5py = None


class DMFTParser(ABC):
    """
    Abstract base class for parsing DMFT output files.
    """

    @abstractmethod
    def read_self_energy(self):
        """
        Reads the self-energy from file.

        Returns:
            sigma (np.ndarray): Self-energy array of shape (n_spin, n_freq, n_orb, n_orb).
                                Note: Currently ignoring k-dependence for local DMFT.
            mesh (np.ndarray): Frequency mesh (complex array of Matsubara frequencies).
        """
        return None, None

    @abstractmethod
    def get_chemical_potential(self):
        """
        Returns the chemical potential (mu).

        Returns:
            mu (float): The chemical potential.
        """
        return 0.0


class W2DynamicsParser(DMFTParser):
    """
    Concrete parser for w2dynamics HDF5 output files.
    """

    def __init__(self, filename):
        if h5py is None:
            raise ImportError(
                "h5py is required for reading DMFT HDF5 files. Please install it."
            )

        self.filename = filename
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"DMFT file not found: {self.filename}")

    def _open_file(self):
        return h5py.File(self.filename, "r")

    def get_chemical_potential(self):
        with self._open_file() as f:
            # Try typical locations for mu
            if "atoms/mu" in f:
                return f["atoms/mu"][0]
            # Fallback or other locations can be added here
            raise KeyError("Could not find chemical potential 'mu' in HDF5 file.")

    def _get_beta(self):
        with self._open_file() as f:
            if "parameters/beta" in f:
                return f["parameters/beta"][0]
            raise KeyError("Could not find inverse temperature 'beta' in HDF5 file.")

    def read_self_energy(self):
        """
        Reads Sigma(iwn) from w2dynamics format.
        Structure: /dmft_output/iterations/last_iter/self_energy/ineq-001/sigma_freq
        """
        with self._open_file() as f:
            # Navigate to the self energy data
            # Assuming the first inequivalent atom for now
            base_path = "dmft_output/iterations/last_iter/self_energy/ineq-001"
            if base_path not in f:
                raise KeyError(f"Could not find self-energy group at {base_path}")

            sigma_ds = f[f"{base_path}/sigma_freq"]
            sigma_data = sigma_ds[:]
            # w2dyn shape: [n_spin, n_freq, n_orb, n_orb] (Verify this matches mock)

            # Reconstruct frequency mesh
            beta = self._get_beta()
            n_freq = sigma_data.shape[1]
            n_points = np.arange(n_freq)
            mesh = 1j * (2 * n_points + 1) * np.pi / beta

            return sigma_data, mesh


class DMFTManager:
    """
    Manager for DMFT calculations.
    Combines static Wannier90 model with dynamic self-energy.
    """

    def __init__(self, path, prefix, atoms, dmft_file, **kwargs):
        from ase.io import read

        from TB2J.interfaces.wannier90_interface import WannierHam
        from TB2J.utils import auto_assign_basis_name
        from TB2J.wannier import parse_atoms

        self.path = path
        self.prefix = prefix
        self.dmft_file = dmft_file
        self.parser = W2DynamicsParser(dmft_file)
        self.output_path = kwargs.get("output_path", "TB2J_results")

        if atoms is None:
            posfile = kwargs.get("posfile", None)
            if posfile is not None:
                try:
                    atoms = read(os.path.join(path, posfile))
                except Exception:
                    atoms = parse_atoms(os.path.join(path, f"{prefix}.win"))
            else:
                atoms = parse_atoms(os.path.join(path, f"{prefix}.win"))
        self.atoms = atoms

        # 1. Read static model
        print("Reading static Wannier90 model...")
        nspin = kwargs.get("nspin", 2)
        self.static_model = WannierHam.read_from_wannier_dir(
            path=path, prefix=prefix, atoms=atoms, nls=(nspin == 2)
        )

        # 2. Generate basis
        basis, _ = auto_assign_basis_name(
            self.static_model.xred,
            atoms,
            write_basis_file=os.path.join(self.output_path, "assigned_basis.txt"),
        )
        self.basis = basis

    def description(self):
        desc = f""" Input from DMFT calculation.
Tight binding data from {self.path}.
Prefix of Wannier90 files:{self.prefix}.
DMFT self-energy from:{self.dmft_file}.
Warning: Please check if the noise level of Wannier function Hamiltonian is much smaller than the exchange values.
"""
        return desc

    def __call__(self):
        # Wrap static model with DMFT data
        from TB2J.dmft_model import TBModelDMFT

        print("Wrapping static model with DMFT self-energy...")
        dmft_model = TBModelDMFT(self.static_model, self.parser)

        return dmft_model, self.basis, self.description()
