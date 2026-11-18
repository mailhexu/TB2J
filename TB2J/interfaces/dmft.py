import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# Try to import h5py, but don't fail if not present (unless used)
try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


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

            return sigma_data, mesh


class SigInpParser(DMFTParser):
    """
    Concrete parser for text-based DMFT self-energy files (sig.inp format).

    This parser handles the format commonly used by TRIQS, EDMFTF, and custom DMFT solvers.

    File format:
    - Header lines starting with # containing metadata
    - Data section with Matsubara frequencies and self-energy values

    Example header:
        # nom,ncor_orb= 4774 10
        # T=  0.010000000000000
        # s_oo-Vdc=  2.595...  3.291...  ... (10 values)
        # s_oo= [23.112..., 23.808..., ... (10 values)]
        # Vdc= [20.517..., ... (10 values)]

    Example data line:
        iωₙ  Re(Σ₁) Im(Σ₁) Re(Σ₂) Im(Σ₂) ... Re(Σ₁₀) Im(Σ₁₀)
    """

    def __init__(
        self,
        sig_file: str,
        params_file: Optional[str] = None,
        mu_file: Optional[str] = None,
        use_total_sigma: bool = False,
    ):
        """
        Initialize SigInpParser.

        Args:
            sig_file: Path to sig.inp file containing self-energy data
            params_file: Path to dmft_params.dat (optional, auto-detected if not provided)
            mu_file: Path to DMFT_mu.out (optional, auto-detected if not provided)
            use_total_sigma: If True, add Σ(∞) to self-energy. Default: False
                            (use as-is, assuming file contains Σ(iω) - Σ(∞))
        """
        self.sig_file = sig_file
        self.params_file = params_file or self._find_params_file(sig_file)
        self.mu_file = mu_file or self._find_mu_file(sig_file)
        self.use_total_sigma = use_total_sigma

        # Instance attributes (populated during parsing)
        self.sigma: Optional[np.ndarray] = None  # [n_spin, n_freq, n_orb, n_orb]
        self.mesh: Optional[np.ndarray] = None  # Matsubara frequency mesh [n_freq]
        self.mu: float = 0.0  # chemical potential
        self.header: dict = {}  # parsed header info
        self.orbital_map: dict = {}  # orbital mapping from params file

        # Validate and parse
        if not os.path.exists(self.sig_file):
            raise FileNotFoundError(f"Self-energy file not found: {self.sig_file}")

        self._parse_all()

    def _find_params_file(self, sig_file: str) -> Optional[str]:
        """Auto-detect dmft_params.dat file in same directory."""
        dir_path = os.path.dirname(sig_file) or "."
        candidates = [
            os.path.join(dir_path, "dmft_params.dat"),
            os.path.join(dir_path, "params.dat"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                logger.info(f"Auto-detected params file: {candidate}")
                return candidate
        logger.warning("dmft_params.dat not found, will use default orbital mapping")
        return None

    def _find_mu_file(self, sig_file: str) -> Optional[str]:
        """Auto-detect DMFT_mu.out file in same directory."""
        dir_path = os.path.dirname(sig_file) or "."
        candidates = [
            os.path.join(dir_path, "DMFT_mu.out"),
            os.path.join(dir_path, "mu.out"),
            os.path.join(dir_path, "chemical_potential.dat"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                logger.info(f"Auto-detected mu file: {candidate}")
                return candidate
        logger.warning("Chemical potential file not found, will use 0.0")
        return None

    def _parse_all(self):
        """Parse all files and construct self-energy."""
        logger.info(f"Parsing DMFT self-energy from {self.sig_file}")

        # Parse header
        self.header = self._parse_header()
        logger.info(
            f"Header: nom={self.header.get('nom')}, "
            f"ncor_orb={self.header.get('ncor_orb')}, "
            f"T={self.header.get('temperature')} eV"
        )

        # Parse data section
        self.sigma_diag, self.mesh = self._parse_data()
        logger.info(
            f"Parsed {len(self.mesh)} frequency points, "
            f"{self.sigma_diag.shape[1]} orbitals"
        )

        # Apply double-counting correction if requested
        if self.use_total_sigma:
            self.sigma_diag = self._apply_double_counting(self.sigma_diag)

        # Parse orbital mapping from params file
        if self.params_file and os.path.exists(self.params_file):
            self.orbital_map = self._parse_params()
            logger.info(
                f"Parsed orbital map: {self.orbital_map.get('n_correlated_atoms', 0)} atoms"
            )
        else:
            # Default mapping: all orbitals are correlated
            self.orbital_map = self._create_default_orbital_map()
            if self.params_file:
                logger.warning(
                    f"Params file not found: {self.params_file}. "
                    f"Using default orbital mapping."
                )

        # Read chemical potential
        self.mu = self._read_mu()
        logger.info(f"Chemical potential: {self.mu} eV")

        # Expand to full matrix form
        self.sigma = self._expand_to_matrix()
        logger.info(f"Self-energy matrix shape: {self.sigma.shape}")

        # Initialize atom-level mapping (for multi-atom support)
        self.atom_sigma_map: dict = {}  # {atom_idx: sigma_block}
        self.wannier_to_atom_map: list = []  # [atom_idx for each wannier orbital]

    def assign_to_atoms(
        self,
        n_wannier_orbitals: int,
        wannier_to_atom_map: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Map self-energy to specific atoms in the Wannier model.

        Args:
            n_wannier_orbitals: Total number of Wannier orbitals
            wannier_to_atom_map: List mapping wannier orbital index to atom index
                                (length = n_wannier_orbitals)

        Returns:
            atom_sigma_map: Dictionary mapping atom index to self-energy block
                           {atom_idx: sigma[n_spin, n_freq, n_local_orb, n_local_orb]}
        """
        if len(wannier_to_atom_map) != n_wannier_orbitals:
            raise ValueError(
                f"wannier_to_atom_map length ({len(wannier_to_atom_map)}) "
                f"doesn't match n_wannier_orbitals ({n_wannier_orbitals})"
            )

        self.wannier_to_atom_map = wannier_to_atom_map
        n_spin, n_freq, n_orb_per_spin, _ = self.sigma.shape

        # Group Wannier orbitals by atom
        atom_orbitals: dict[int, list[int]] = {}
        for i_wan, i_atom in enumerate(wannier_to_atom_map):
            if i_atom not in atom_orbitals:
                atom_orbitals[i_atom] = []
            atom_orbitals[i_atom].append(i_wan)

        # Build atom-level self-energy map
        self.atom_sigma_map = {}
        for i_atom, wannier_indices in atom_orbitals.items():
            # Check if this atom has correlated orbitals
            if i_atom in self.orbital_map.get("atom_indices", []):
                # Get the self-energy for this atom from the correlated set
                atom_idx_in_corr = self.orbital_map["atom_indices"].index(i_atom)
                sigma_orbital_indices = self.orbital_map["atom_orbital_map"][
                    atom_idx_in_corr
                ]
                spin_channel = self.orbital_map["spin_channels"][atom_idx_in_corr]

                # Extract self-energy block for this atom
                # Map from sigma orbital indices to local orbital indices
                n_local_orb = len(wannier_indices)
                sigma_atom = np.zeros(
                    (n_spin, n_freq, n_local_orb, n_local_orb),
                    dtype=np.complex128,
                )

                for local_idx, wan_idx in enumerate(wannier_indices):
                    # Determine which spin channel this Wannier orbital belongs to
                    # (assuming first half = up, second half = down)
                    if wan_idx < n_wannier_orbitals // 2:
                        wan_spin = 0
                        wan_orb_in_spin = wan_idx
                    else:
                        wan_spin = 1
                        wan_orb_in_spin = wan_idx - n_wannier_orbitals // 2

                    # Only apply self-energy if spin matches
                    if wan_spin == spin_channel:
                        # Get the self-energy value from the correlated orbitals
                        # Map to the appropriate sigma orbital
                        sigma_orb_idx = wan_orb_in_spin % len(sigma_orbital_indices)
                        sigma_val = self.sigma[
                            spin_channel, :, sigma_orb_idx, sigma_orb_idx
                        ]
                        sigma_atom[wan_spin, :, local_idx, local_idx] = sigma_val

                self.atom_sigma_map[i_atom] = sigma_atom
                logger.debug(
                    f"Assigned self-energy to atom {i_atom}: "
                    f"{len(wannier_indices)} orbitals, "
                    f"spin channel {spin_channel}"
                )
            else:
                # Non-correlated atom: zero self-energy
                n_local_orb = len(wannier_indices)
                self.atom_sigma_map[i_atom] = np.zeros(
                    (n_spin, n_freq, n_local_orb, n_local_orb),
                    dtype=np.complex128,
                )

        logger.info(f"Mapped self-energy to {len(self.atom_sigma_map)} atoms")
        return self.atom_sigma_map

    def get_sigma_for_atom(
        self,
        atom_idx: int,
        energy: complex,
        interpolate: bool = True,
    ) -> np.ndarray:
        """
        Get self-energy for a specific atom at a given energy.

        Args:
            atom_idx: Atom index
            energy: Complex energy (Matsubara frequency)
            interpolate: If True, interpolate between mesh points

        Returns:
            sigma_atom: Self-energy matrix [n_spin, n_local_orb, n_local_orb]
        """
        if atom_idx not in self.atom_sigma_map:
            raise KeyError(f"Atom {atom_idx} not in atom_sigma_map")

        sigma_atom_full = self.atom_sigma_map[atom_idx]
        n_spin, n_freq, n_orb, _ = sigma_atom_full.shape

        # Find closest frequency index
        idx = np.argmin(np.abs(self.mesh - energy))

        return sigma_atom_full[:, idx, :, :]

    def _parse_header(self) -> dict:
        """
        Parse header lines from sig.inp file.

        Returns:
            dict with keys:
                - nom: number of Matsubara frequencies (int)
                - ncor_orb: number of correlated orbitals (int)
                - temperature: temperature in eV (float)
                - s_oo_vdc: Σ(∞) - Vdc per orbital (np.ndarray or None)
                - s_oo: Σ(∞) per orbital (np.ndarray or None)
                - vdc: Vdc per orbital (np.ndarray or None)
        """
        header = {
            "nom": None,
            "ncor_orb": None,
            "temperature": None,
            "s_oo_vdc": None,
            "s_oo": None,
            "vdc": None,
        }

        with open(self.sig_file, "r") as f:
            for line in f:
                # Stop at data section
                if not line.startswith("#"):
                    break

                # Parse nom,ncor_orb
                match = re.match(r"#\s*nom,ncor_orb=\s*(\d+)\s+(\d+)", line)
                if match:
                    header["nom"] = int(match.group(1))
                    header["ncor_orb"] = int(match.group(2))
                    continue

                # Parse temperature
                match = re.match(r"#\s*T=\s*([\d.eE+-]+)", line)
                if match:
                    header["temperature"] = float(match.group(1))
                    continue

                # Parse s_oo-Vdc (space-separated values)
                match = re.match(r"#\s*s_oo-Vdc=\s*([0-9.\-eE\s]+)", line)
                if match:
                    header["s_oo_vdc"] = np.array(
                        [float(x) for x in match.group(1).split()]
                    )
                    continue

                # Parse s_oo (bracketed, comma-separated)
                match = re.match(r"#\s*s_oo=\s*\[([0-9.,\s-]+)\]", line)
                if match:
                    header["s_oo"] = np.array(
                        [float(x.strip()) for x in match.group(1).split(",")]
                    )
                    continue

                # Parse Vdc (bracketed, comma-separated)
                match = re.match(r"#\s*Vdc=\s*\[([0-9.,\s-]+)\]", line)
                if match:
                    header["vdc"] = np.array(
                        [float(x.strip()) for x in match.group(1).split(",")]
                    )
                    continue

        # Validate required fields
        if header["nom"] is None or header["ncor_orb"] is None:
            raise ValueError(
                "Missing required header 'nom,ncor_orb' in sig.inp file. "
                "Please check file format."
            )

        return header

    def _parse_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse data section of sig.inp file.

        Returns:
            sigma_diag: Diagonal self-energy elements [n_freq, n_orb]
            mesh: Matsubara frequency array [n_freq]

        Raises:
            ValueError: If frequency mesh is inconsistent
        """
        n_orb = self.header["ncor_orb"]
        nom_expected = self.header["nom"]

        # Pre-allocate arrays
        mesh = np.zeros(nom_expected, dtype=np.float64)
        sigma_diag = np.zeros((nom_expected, n_orb), dtype=np.complex128)

        count = 0
        data_started = False

        with open(self.sig_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Skip header lines
                if line.startswith("#"):
                    continue

                # Skip empty lines
                line = line.strip()
                if not line:
                    continue

                data_started = True
                parts = line.split()

                # Validate column count
                expected_cols = 1 + 2 * n_orb
                if len(parts) != expected_cols:
                    raise ValueError(
                        f"Line {line_num}: expected {expected_cols} columns "
                        f"(1 frequency + {n_orb} orbitals × 2), got {len(parts)}. "
                        f"Check that ncor_orb={n_orb} is correct."
                    )

                # Parse frequency (first column)
                try:
                    mesh[count] = float(parts[0])
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: could not parse frequency '{parts[0]}'"
                    )

                # Parse self-energy (real, imag pairs)
                for i_orb in range(n_orb):
                    re_idx = 1 + 2 * i_orb
                    im_idx = 1 + 2 * i_orb + 1
                    try:
                        re_val = float(parts[re_idx])
                        im_val = float(parts[im_idx])
                        sigma_diag[count, i_orb] = re_val + 1j * im_val
                    except ValueError:
                        raise ValueError(
                            f"Line {line_num}, orbital {i_orb + 1}: "
                            f"could not parse self-energy values "
                            f"'{parts[re_idx]} {parts[im_idx]}'"
                        )

                count += 1

        if not data_started:
            raise ValueError(
                "No data section found in sig.inp file. "
                "File may contain only header or be empty."
            )

        # Verify count
        if count != nom_expected:
            logger.warning(
                f"Expected {nom_expected} frequency points per header, "
                f"found {count}. Using actual count."
            )

        # Truncate to actual count
        mesh = mesh[:count]
        sigma_diag = sigma_diag[:count]

        # Validate Matsubara frequency mesh as positive real omega_n values from file.
        self._validate_matsubara_mesh(mesh)

        return sigma_diag, 1j * mesh

    def _validate_matsubara_mesh(self, mesh: np.ndarray, tolerance: float = 1e-4):
        """
        Validate that the Matsubara frequency mesh is consistent.

        Matsubara frequencies should be: ωₙ = (2n+1)π/β
        This means they should be equally spaced with spacing 2π/β

        Args:
            mesh: Array of Matsubara frequencies
            tolerance: Relative tolerance for checking consistency

        Raises:
            ValueError: If mesh is inconsistent
        """
        if len(mesh) < 2:
            raise ValueError(
                f"Matsubara mesh must have at least 2 frequencies, got {len(mesh)}"
            )

        # Check that all frequencies are positive (for fermionic Matsubara)
        if np.any(mesh < 0):
            raise ValueError(
                f"Matsubara frequencies must be positive, "
                f"found negative values: {mesh[mesh < 0]}"
            )

        # Check that frequencies are strictly increasing
        if not np.all(np.diff(mesh) > 0):
            raise ValueError("Matsubara frequencies must be strictly increasing")

        # Check for equal spacing (characteristic of Matsubara mesh)
        spacings = np.diff(mesh)
        mean_spacing = np.mean(spacings)
        relative_deviation = np.abs(spacings - mean_spacing) / mean_spacing

        if np.any(relative_deviation > tolerance):
            max_deviation = np.max(relative_deviation)
            raise ValueError(
                f"Matsubara frequency mesh is not equally spaced. "
                f"Maximum relative deviation: {max_deviation:.2e} (tolerance: {tolerance}). "
                f"Mean spacing: {mean_spacing:.6f}, "
                f"Min spacing: {np.min(spacings):.6f}, "
                f"Max spacing: {np.max(spacings):.6f}"
            )

        # Calculate beta from spacing
        # For fermionic Matsubara: ωₙ = (2n+1)π/β, so spacing = 2π/β
        beta_calculated = 2 * np.pi / mean_spacing

        # Check consistency with header temperature if available
        if self.header.get("temperature") is not None:
            T = self.header["temperature"]
            beta_from_T = 1.0 / T

            relative_diff = np.abs(beta_calculated - beta_from_T) / beta_from_T
            if relative_diff > tolerance:
                logger.warning(
                    f"Matsubara mesh beta ({beta_calculated:.6f}) inconsistent "
                    f"with header temperature T={T} (beta={beta_from_T:.6f}). "
                    f"Relative difference: {relative_diff:.2e}. "
                    f"Using mesh-derived beta."
                )

        logger.info(
            f"Validated Matsubara mesh: {len(mesh)} frequencies, "
            f"β={beta_calculated:.6f} eV⁻¹, T={1/beta_calculated:.6f} eV"
        )

    def _parse_params(self) -> dict:
        """
        Parse dmft_params.dat file for orbital mapping.

        Returns:
            orbital_map: dict with structure:
                {
                    'n_correlated_atoms': int,
                    'n_orbitals_per_atom': list[int],
                    'atom_orbital_map': list[list[int]],  # sigma indices per atom
                    'spin_channels': list[int],  # 0=up, 1=down per atom
                    'atom_indices': list[int],  # 0-based atom indices
                }
        """
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"DMFT params file not found: {self.params_file}")

        result = {
            "n_correlated_atoms": 0,
            "n_orbitals_per_atom": [],
            "atom_orbital_map": [],
            "spin_channels": [],
            "atom_indices": [],
        }

        with open(self.params_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            original_line = lines[i]
            line = original_line.strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Number of total correlated atoms
            if "Number of total correlated atoms" in original_line:
                # Find next non-empty, non-comment line
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line and not next_line.startswith("#"):
                        result["n_correlated_atoms"] = int(next_line)
                        break
                    i += 1
                i += 1
                continue

            # Number of correlated orbitals per atom
            if "Number of correlated orbitals per atom" in original_line:
                # Find next non-empty, non-comment line
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line and not next_line.startswith("#"):
                        orbitals = [int(x) for x in next_line.split()]
                        result["n_orbitals_per_atom"] = orbitals
                        break
                    i += 1
                i += 1
                continue

            # Orbital index for the self-energy at each atom
            if "Orbital index for the self-energy" in original_line:
                i += 1
                n_atoms = result["n_correlated_atoms"]
                n_orbitals_list = result["n_orbitals_per_atom"]

                if not n_orbitals_list:
                    raise ValueError(
                        "Found orbital indices but no orbital counts. "
                        "Check dmft_params.dat format."
                    )

                for atom_idx in range(n_atoms):
                    # Find next non-empty, non-comment line
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if next_line and not next_line.startswith("#"):
                            break
                        i += 1

                    if i >= len(lines):
                        break

                    indices = [int(x) for x in lines[i].split()]
                    n_orb = (
                        n_orbitals_list[atom_idx]
                        if atom_idx < len(n_orbitals_list)
                        else len(indices)
                    )
                    result["atom_orbital_map"].append(indices[:n_orb])

                    # Determine spin channel from indices
                    # Convention: 1-5 = spin up (0), 6-10 = spin down (1)
                    spin = 0 if indices[0] <= 5 else 1
                    result["spin_channels"].append(spin)
                    result["atom_indices"].append(atom_idx)

                    i += 1
                continue

            i += 1

        # Validate
        if len(result["atom_orbital_map"]) != result["n_correlated_atoms"]:
            raise ValueError(
                f"Mismatch in dmft_params.dat: "
                f"{len(result['atom_orbital_map'])} atoms in orbital map, "
                f"but {result['n_correlated_atoms']} correlated atoms declared"
            )

        return result

    def _create_default_orbital_map(self) -> dict:
        """Create default orbital mapping when no params file is provided."""
        n_orb = self.header["ncor_orb"]

        # Assume all orbitals are correlated and belong to spin-up channel
        return {
            "n_correlated_atoms": 1,
            "n_orbitals_per_atom": [n_orb],
            "atom_orbital_map": [list(range(1, n_orb + 1))],
            "spin_channels": [0],
            "atom_indices": [0],
        }

    def _apply_double_counting(self, sigma_diag: np.ndarray) -> np.ndarray:
        """
        Add Σ(∞) to self-energy if use_total_sigma is True.

        Args:
            sigma_diag: Self-energy diagonal elements [n_freq, n_orb]

        Returns:
            sigma_diag_corrected: Self-energy with Σ(∞) added
        """
        if self.header.get("s_oo") is None:
            logger.warning(
                "use_total_sigma=True but s_oo not found in header. "
                "Using self-energy as-is."
            )
            return sigma_diag

        # s_oo is per-orbital value, broadcast over frequency axis
        s_oo = self.header["s_oo"]

        # Validate dimensions
        if len(s_oo) != sigma_diag.shape[1]:
            raise ValueError(
                f"s_oo has {len(s_oo)} values but self-energy has "
                f"{sigma_diag.shape[1]} orbitals"
            )

        logger.info("Adding Σ(∞) to self-energy (using total self-energy)")
        return sigma_diag + s_oo

    def _expand_to_matrix(self) -> np.ndarray:
        """
        Expand diagonal self-energy to full matrix form.

        Input: sigma_diag[n_freq, n_total_orb]
        Output: sigma_full[n_spin, n_freq, n_orb_per_spin, n_orb_per_spin]

        Convention: first half of orbitals = spin-up, second half = spin-down

        For LaMnO3 example:
        - Input: [4774, 10] (10 = 5 orbitals × 2 spins)
        - Output: [2, 4774, 5, 5]

        Returns:
            sigma_full: Self-energy matrix [n_spin, n_freq, n_orb, n_orb]
        """
        n_freq, n_total_orb = self.sigma_diag.shape

        # Determine orbitals per spin channel (assume equal split)
        if n_total_orb % 2 != 0:
            raise ValueError(
                f"Expected even number of orbitals (for spin up/down), "
                f"got {n_total_orb}"
            )

        n_orb_per_spin = n_total_orb // 2
        n_spin = 2  # collinear (spin-up and spin-down)

        # Initialize full matrix with zeros
        sigma_full = np.zeros(
            (n_spin, n_freq, n_orb_per_spin, n_orb_per_spin),
            dtype=np.complex128,
        )

        # Map diagonal elements to matrix
        # Convention: first half = spin up (channel 0), second half = spin down (channel 1)
        for freq_idx in range(n_freq):
            for orb_idx in range(n_total_orb):
                # Determine spin channel and local orbital index
                if orb_idx < n_orb_per_spin:
                    spin_idx = 0  # up
                    local_orb = orb_idx
                else:
                    spin_idx = 1  # down
                    local_orb = orb_idx - n_orb_per_spin

                # Place on diagonal
                sigma_full[spin_idx, freq_idx, local_orb, local_orb] = self.sigma_diag[
                    freq_idx, orb_idx
                ]

        return sigma_full

    def _read_mu(self) -> float:
        """
        Read chemical potential from DMFT_mu.out file.

        Returns:
            mu: Chemical potential in eV
        """
        if self.mu_file and os.path.exists(self.mu_file):
            with open(self.mu_file, "r") as f:
                content = f.read().strip()
                try:
                    mu = float(content)
                    logger.info(f"Read chemical potential: {mu} eV")
                    return mu
                except ValueError:
                    logger.warning(
                        f"Could not parse mu from {self.mu_file}: '{content}'"
                    )

        # Fallback: try to get from header (non-standard)
        if "mu" in self.header:
            logger.info(f"Using chemical potential from header: {self.header['mu']}")
            return self.header["mu"]

        # Final fallback
        logger.warning(
            "Chemical potential not found, using 0.0 eV. "
            "Provide DMFT_mu.out file or set --mu-file explicitly."
        )
        return 0.0

    def read_self_energy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read self-energy from file.

        Returns:
            sigma: Self-energy array of shape (n_spin, n_freq, n_orb, n_orb)
            mesh: Matsubara frequency mesh (complex array)

        Raises:
            DMFTValidationError: If file format is invalid
            FileNotFoundError: If file doesn't exist

        Example:
            >>> parser = SigInpParser('sig.inp')
            >>> sigma, mesh = parser.read_self_energy()
            >>> print(sigma.shape)
            (2, 4774, 5, 5)
        """
        return self.sigma, self.mesh

    def get_chemical_potential(self) -> float:
        """
        Returns the chemical potential (mu).

        Returns:
            mu: The chemical potential in eV
        """
        return self.mu

    def get_static_sigma(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the static self-energy (frequency-independent part).

        Returns:
            sigma_static: Static self-energy array of shape (n_spin, n_orb, n_orb)
            s_oo: Σ(∞) values per orbital (for reference)

        The static self-energy is:
        - s_oo_vdc (Σ(∞) - Vdc) if available, which is the correlated part
        - or s_oo (Σ(∞)) if s_oo_vdc not available

        Note: The self-energy in sig.inp is typically Σ(iω) - Σ(∞).
        For static calculations, we use Σ(∞) or Σ(∞) - Vdc as a constant shift.
        """
        s_oo_vdc = self.header.get("s_oo_vdc")
        s_oo = self.header.get("s_oo")
        ncor_orb = self.header["ncor_orb"]

        if s_oo_vdc is not None:
            static_values = s_oo_vdc
        elif s_oo is not None:
            static_values = s_oo
        else:
            logger.warning(
                "No Σ(∞) or Σ(∞)-Vdc found in sig.inp header. "
                "Using zero static self-energy."
            )
            static_values = np.zeros(ncor_orb)

        # FIXME: this has nothing to do with 10.
        n_orb_per_spin = ncor_orb // 2 if ncor_orb >= 10 else ncor_orb
        n_spin = 2 if ncor_orb >= 10 else 1

        # Shape is (n_spin, n_orb_per_spin, n_orb_per_spin): each spin block is 5×5
        sigma_static = np.zeros(
            (n_spin, n_orb_per_spin, n_orb_per_spin), dtype=np.complex128
        )

        if n_spin == 2:
            np.fill_diagonal(sigma_static[0], static_values[:n_orb_per_spin])
            np.fill_diagonal(sigma_static[1], static_values[n_orb_per_spin:])
        else:
            np.fill_diagonal(sigma_static[0], static_values)

        s_oo_ref = s_oo if s_oo is not None else static_values

        return sigma_static, s_oo_ref


class DMFTManager:
    """
    Manager for DMFT calculations.
    Combines static Wannier90 model with dynamic self-energy.

    Supports multiple DMFT parser types:
    - w2dynamics: HDF5 format from w2dynamics code
    - siginp: Text-based sig.inp format (TRIQS, EDMFTF, custom solvers)
    - auto: Auto-detect from file extension
    """

    def __init__(
        self,
        path: str,
        prefix: str,
        atoms=None,
        dmft_file: str = None,
        parser_type: str = "auto",
        **kwargs,
    ):
        from ase.io import read

        from TB2J.interfaces.wannier90_interface import WannierHam
        from TB2J.utils import auto_assign_basis_name
        from TB2J.wannier import parse_atoms

        self.path = path
        self.prefix = prefix
        self.dmft_file = dmft_file
        self.parser_type = parser_type
        self.output_path = kwargs.get("output_path", "TB2J_results")
        self.magnetic_elements = kwargs.get("magnetic_elements", None)
        self.spin_channels = kwargs.get(
            "spin_channels", None
        )  # list of +1/-1 per corr atom
        self.use_static_sigma = kwargs.get("static_sigma", False)

        # Select and create parser
        self.parser = self._create_parser(
            dmft_file=dmft_file,
            parser_type=parser_type,
            params_file=kwargs.get("dmft_params_file"),
            mu_file=kwargs.get("mu_file"),
            use_total_sigma=kwargs.get("use_total_sigma", False),
        )

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

    def __call__(self):
        """
        Build the DMFT model and return (dmft_model, basis, description).

        Returns:
            dmft_model: TBModelDMFT instance combining static Wannier model + self-energy
            basis: Orbital basis list
            description: String description for the calculation
        """
        import os

        from TB2J.dmft_model import TBModelDMFT

        os.makedirs(self.output_path, exist_ok=True)

        # Override spin channels in parser's orbital_map if supplied by user
        if self.spin_channels is not None and hasattr(self.parser, "orbital_map"):
            n_atoms = self.parser.orbital_map.get("n_correlated_atoms", 0)
            if len(self.spin_channels) != n_atoms:
                raise ValueError(
                    f"--spin-channels has {len(self.spin_channels)} values "
                    f"but orbital_map has {n_atoms} correlated atoms"
                )
            # Convert +1/-1 convention to 0/1 index convention used internally
            self.parser.orbital_map["spin_channels"] = [
                0 if s > 0 else 1 for s in self.spin_channels
            ]
            logger.info(
                f"Overriding spin channels: {self.spin_channels} → "
                f"{self.parser.orbital_map['spin_channels']}"
            )

        dmft_model = TBModelDMFT(
            static_model=self.static_model,
            dmft_parser=self.parser,
            basis=self.basis,
            atoms=self.atoms,
            magnetic_elements=self.magnetic_elements,
            use_static_sigma=self.use_static_sigma,
        )

        description = (
            f"DMFT exchange from {os.path.basename(self.dmft_file)}, "
            f"μ={dmft_model.mu:.4f} eV"
        )

        return dmft_model, self.basis, description

    def _create_parser(
        self,
        dmft_file: str,
        parser_type: str,
        params_file: Optional[str] = None,
        mu_file: Optional[str] = None,
        use_total_sigma: bool = False,
    ) -> DMFTParser:
        """
        Create appropriate DMFT parser based on type and file extension.

        Args:
            dmft_file: Path to DMFT self-energy file
            parser_type: 'auto', 'w2dynamics', or 'siginp'
            params_file: Path to dmft_params.dat (for siginp)
            mu_file: Path to chemical potential file
            use_total_sigma: Whether to add Σ(∞) in self-energy

        Returns:
            DMFTParser instance

        Raises:
            ValueError: If parser type cannot be determined
            FileNotFoundError: If file doesn't exist
        """
        # Auto-detection from file extension
        if parser_type == "auto":
            if dmft_file.endswith(".h5") or dmft_file.endswith(".hdf5"):
                parser_type = "w2dynamics"
            elif dmft_file.endswith(".inp"):
                parser_type = "siginp"
            else:
                raise ValueError(
                    f"Cannot auto-detect parser for file: {dmft_file}\n"
                    f"Please specify --parser-type explicitly (w2dynamics or siginp)"
                )

        if parser_type == "w2dynamics":
            if h5py is None:
                raise ImportError(
                    "h5py is required for w2dynamics parser. Install with: pip install h5py"
                )
            logger.info(f"Using W2DynamicsParser for {dmft_file}")
            return W2DynamicsParser(dmft_file)

        elif parser_type == "siginp":
            logger.info(f"Using SigInpParser for {dmft_file}")
            return SigInpParser(
                sig_file=dmft_file,
                params_file=params_file,
                mu_file=mu_file,
                use_total_sigma=use_total_sigma,
            )

        else:
            raise ValueError(
                f"Unknown parser type: {parser_type}. "
                f"Supported types: 'w2dynamics', 'siginp', 'auto'"
            )


class DMFTstaticManager:
    """
    Static DMFT exchange manager.

    Implements DMFT exchange as standard collinear TB2J (ExchangeCL2) where
    the static self-energy correction Σ(∞)−Vdc is folded directly into the
    on-site blocks of the Wannier Hamiltonian.

    For each correlated atom the diagonal shift
        Δh[orb, orb] += (s_oo_vdc)_orb
    is added to H(R=0) of the spin model that owns that atom
    (as determined by ``spin_channels``).

    This avoids the embedding/indexing complexity of ``DMFTManager`` and
    reuses the well-tested ``ExchangeCL2`` integration path.

    Parameters
    ----------
    path : str
        Directory containing Wannier90 files.
    prefix : str
        Filename prefix for Wannier90 files (e.g. 'wannier90').
    dmft_file : str
        Path to the sig.inp self-energy file.
    spin_channels : list of int
        ±1 per correlated atom (same order as dmft_params.dat).
        +1 = spin-up model, −1 = spin-down model.
    magnetic_elements : list of str, optional
        Element symbols of magnetic atoms (used for output labelling only).
    posfile : str, optional
        Atomic structure file; falls back to parsing ``.win`` if absent.
    dmft_params_file : str, optional
        Explicit path to dmft_params.dat (auto-detected if None).
    mu_file : str, optional
        Explicit path to DMFT_mu.out (auto-detected if None).
    use_total_sigma : bool
        Add Σ(∞) to the dynamic part (passed through to SigInpParser).
    output_path : str
        Where to write exchange output.
    **exchange_kwargs
        Passed verbatim to ExchangeCL2 (kmesh, Rcut, nz, …).
    """

    def __init__(
        self,
        path: str,
        prefix: str,
        dmft_file: str,
        spin_channels: list,
        magnetic_elements=None,
        posfile: str = "POSCAR",
        dmft_params_file: str = None,
        mu_file: str = None,
        use_total_sigma: bool = False,
        output_path: str = "TB2J_results_static",
        **exchange_kwargs,
    ):
        import copy

        from ase.io import read as ase_read
        from HamiltonIO.wannier import WannierHam

        from TB2J.utils import auto_assign_basis_name, symbol_number
        from TB2J.wannier import parse_atoms

        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        # ── 1. Atoms ──────────────────────────────────────────────────────────
        win_file = os.path.join(path, f"{prefix}.win")
        posfile_full = os.path.join(path, posfile)
        try:
            atoms = ase_read(posfile_full)
            logger.info(f"Read atoms from {posfile_full}")
        except Exception:
            atoms = parse_atoms(win_file)
            logger.info(f"Read atoms from {win_file}")

        # ── 2. DMFT parser (for static sigma values only) ─────────────────────
        parser = SigInpParser(
            sig_file=dmft_file,
            params_file=dmft_params_file,
            mu_file=mu_file,
            use_total_sigma=use_total_sigma,
        )

        # Override spin_channels in orbital_map (+1/-1 → 0/1)
        if spin_channels is not None:
            n_corr = parser.orbital_map.get("n_correlated_atoms", 0)
            if len(spin_channels) != n_corr:
                raise ValueError(
                    f"spin_channels has {len(spin_channels)} entries but "
                    f"orbital_map has {n_corr} correlated atoms"
                )
            parser.orbital_map["spin_channels"] = [
                0 if s > 0 else 1 for s in spin_channels
            ]
            logger.info(
                f"Spin channels: {spin_channels} → "
                f"{parser.orbital_map['spin_channels']}"
            )

        sigma_static, _ = parser.get_static_sigma()
        # sigma_static: (2, n_orb_per_spin, n_orb_per_spin), diagonal matrices
        # Diagonal values: up = sigma_static[0].diagonal(), dn = sigma_static[1].diagonal()
        sigma_up_diag = sigma_static[0].diagonal().real  # shape (n_orb_per_spin,)
        sigma_dn_diag = sigma_static[1].diagonal().real  # shape (n_orb_per_spin,)
        logger.info(f"Static sigma up diagonal: {sigma_up_diag}")
        logger.info(f"Static sigma dn diagonal: {sigma_dn_diag}")

        mu = parser.get_chemical_potential()
        logger.info(f"Chemical potential μ = {mu:.6f} eV")

        # ── 3. Build orbital map: atom_idx → Wannier orbital indices ──────────
        ham_ref = WannierHam.read_from_wannier_dir(
            path=path, prefix=prefix, atoms=atoms, nls=False
        )
        basis, _ = auto_assign_basis_name(
            ham_ref.xred,
            atoms,
            write_basis_file=os.path.join(output_path, "assigned_basis.txt"),
        )
        atom_sym_map = symbol_number(atoms)
        orb_dict = {}
        for orb_idx, base in enumerate(basis):
            parts = base.split("|")
            atom_idx = atom_sym_map[parts[0]]
            orb_dict.setdefault(atom_idx, []).append(orb_idx)

        # ── 4. Deep-copy ham into two spin models and patch H(R=0) ───────────
        ham_up = copy.deepcopy(ham_ref)
        ham_dn = copy.deepcopy(ham_ref)
        ham_up.set_atoms(atoms)
        ham_dn.set_atoms(atoms)

        orbital_map = parser.orbital_map
        atom_symbols = atoms.get_chemical_symbols()

        # Resolve correlated-atom sequence numbers → global atom indices.
        # orbital_map['atom_indices'] are 0-based indices into the list of
        # *correlated* atoms only, NOT into the full atom list.  We recover
        # the global indices by collecting all atoms whose symbol appears in
        # magnetic_elements (in document order).
        if not magnetic_elements:
            raise ValueError(
                "--magnetic-elements is required for --mode=static-hamiltonian"
            )
        mag_set = set(magnetic_elements)
        corr_global_indices = [i for i, s in enumerate(atom_symbols) if s in mag_set]
        logger.info(f"Correlated atom global indices: {corr_global_indices}")

        n_corr = orbital_map.get("n_correlated_atoms", 0)
        if len(corr_global_indices) < n_corr:
            raise ValueError(
                f"Found only {len(corr_global_indices)} magnetic atoms in structure "
                f"but orbital_map expects {n_corr} correlated atoms"
            )

        patched_up, patched_dn = [], []
        for corr_idx in range(n_corr):
            atom_idx = corr_global_indices[corr_idx]  # global index into orb_dict
            if atom_idx not in orb_dict:
                logger.warning(
                    f"Global atom index {atom_idx} ({atom_symbols[atom_idx]}) "
                    f"not found in orb_dict; skipping"
                )
                continue

            wann_orbs = orb_dict[atom_idx]  # Wannier indices for this atom
            n_expected = orbital_map["n_orbitals_per_atom"][corr_idx]
            if len(wann_orbs) != n_expected:
                logger.warning(
                    f"Atom {atom_idx} ({atom_symbols[atom_idx]}): "
                    f"expected {n_expected} orbitals, found {len(wann_orbs)}; skipping"
                )
                continue

            spin_ch = orbital_map["spin_channels"][corr_idx]  # 0=up, 1=dn

            # Correct sigma embedding: add BOTH sigma channels to BOTH Hamiltonians.
            # This matches the DMFT mode (TBModelDMFT._embed_sigma_blocks) and ensures
            # the magnetic splitting Delta = H_up - H_dn = sigma_up - sigma_dn.
            #
            #   Spin-up atom (spin_ch==0):
            #     H_up += sigma_up (majority)
            #     H_dn += sigma_dn (minority)
            #     → Delta = sigma_up - sigma_dn > 0 ✓
            #
            #   Spin-down atom (spin_ch==1):
            #     H_up += sigma_dn (minority)
            #     H_dn += sigma_up (majority)
            #     → Delta = sigma_dn - sigma_up < 0 ✓
            #
            # Reference: TBModelDMFT._embed_sigma_blocks uses the same mapping.
            if spin_ch == 0:
                # Spin-up atom: majority→up, minority→dn
                H0_up = ham_up.data[(0, 0, 0)]
                H0_dn = ham_dn.data[(0, 0, 0)]
                for local_i, wann_i in enumerate(wann_orbs):
                    H0_up[wann_i, wann_i] += sigma_up_diag[local_i]
                    H0_dn[wann_i, wann_i] += sigma_dn_diag[local_i]
                patched_up.append(
                    f"{atom_symbols[atom_idx]}{atom_idx}(up: +{sigma_up_diag.round(2)}, dn: +{sigma_dn_diag.round(2)})"
                )
            else:
                # Spin-down atom: majority→dn, minority→up
                H0_up = ham_up.data[(0, 0, 0)]
                H0_dn = ham_dn.data[(0, 0, 0)]
                for local_i, wann_i in enumerate(wann_orbs):
                    H0_up[wann_i, wann_i] += sigma_dn_diag[local_i]
                    H0_dn[wann_i, wann_i] += sigma_up_diag[local_i]
                patched_dn.append(
                    f"{atom_symbols[atom_idx]}{atom_idx}(up: +{sigma_dn_diag.round(2)}, dn: +{sigma_up_diag.round(2)})"
                )

        logger.info(f"Patched spin-up H(R=0): {patched_up}")
        logger.info(f"Patched spin-dn H(R=0): {patched_dn}")

        # ── 5. Run standard collinear exchange ────────────────────────────────
        description = (
            f"DMFT static exchange: Σ(∞)−Vdc folded into Wannier H(R=0), "
            f"μ={mu:.4f} eV, from {os.path.basename(dmft_file)}"
        )
        if exchange_kwargs.get("efermi") is None:
            exchange_kwargs["efermi"] = mu
        exchange_kwargs["description"] = description

        if magnetic_elements is not None:
            exchange_kwargs.setdefault("magnetic_elements", magnetic_elements)

        from TB2J.exchangeCL2 import ExchangeCL2

        exchange = ExchangeCL2(
            tbmodels=(ham_up, ham_dn),
            atoms=atoms,
            basis=basis,
            output_path=output_path,
            **exchange_kwargs,
        )
        exchange.calculate_all()
        exchange.write_output(path=output_path)
        logger.info(f"Results written to {output_path}")
