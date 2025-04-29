Parameters for Magnetic Interaction Calculations
=============================================

Introduction
-----------
TB2J provides several command-line tools to calculate magnetic interactions from different electronic structure codes. Each tool shares a common set of parameters for controlling the magnetic interaction calculations, while also having some tool-specific parameters.

To view the complete list of parameters for any tool, use the --help option:

::

   wann2J.py --help    # For Wannier90-based calculations
   siesta2J.py --help  # For SIESTA-based calculations
   abacus2J.py --help  # For ABACUS-based calculations

Common Parameters
---------------
These parameters are available across all TB2J tools and control the core aspects of magnetic interaction calculations.

System Definition
^^^^^^^^^^^^^^^
* elements: List of magnetic elements to include in the Heisenberg model. Example: ``--elements Fe Ni``

* index_magnetic_atoms: Explicitly specify magnetic atoms by their indices (starting from 1) in the unit cell. Overrides element-based selection. Example: ``--index_magnetic_atoms 1 2 4``

* exclude_orbs: Zero-based indices of orbitals to exclude from magnetic site calculations. Useful for fine-tuning the magnetic model.

K-space and Real Space Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* kmesh: Three integers defining the Monkhorst-Pack k-point mesh (e.g., ``--kmesh 7 7 7``). This mesh determines:
   - The k-points used for Green's function calculations
   - The real-space supercell size for magnetic interactions
   - For example, a :math:`7 \times 7 \times 7` k-mesh creates a supercell where magnetic interactions are calculated between the central atom and all atoms within this supercell

* rcut: Maximum distance (in Å) for calculating magnetic interactions between atom pairs. If not specified, includes all pairs within the supercell defined by kmesh.

Energy Integration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^
* efermi: Fermi energy in eV. Handling varies by tool:
   - siesta2J.py/abacus2J.py: Automatically read from output
   - wann2J.py: Must be provided from DFT calculation
   - For insulators: Can be within the band gap
   - For metals: Should match the DFT value

* emin, emax: Energy range for integration, relative to efermi:
   - emin: Should be low enough to include all magnetically relevant states
   - Check local density of states: spin up/down should be nearly identical below emin
   - emax: Usually 0.0 (Fermi level). Can be adjusted for charge doping studies
   - Note: Direct DFT doping is preferred over emax adjustment

* nz: Number of energy points for numerical integration of :math:`\int_{emin}^{emax} d\epsilon`

Performance and Output Controls
^^^^^^^^^^^^^^^^^^^^^^^^^^
* np/nproc: Number of CPU cores for parallel processing. Default: 1

* use_cache: Store wavefunctions and Hamiltonian on disk to reduce memory usage. Useful for large systems. Default: False

* cutoff: Minimum magnitude of exchange coupling (J) to write to output (in eV). Helps filter numerical noise.

* output_path: Directory for output files. Default: TB2J_results

Output Customization
^^^^^^^^^^^^^^^^
* description: Add essential calculation details to the XML output:
   - Exchange-correlation functional used
   - Hubbard U values if applicable
   - Magnetic state description
   - Any other relevant parameters

Advanced Options
^^^^^^^^^^^^
* orb_decomposition: Analyze orbital contributions in non-collinear calculations. Default: False

* orth: Apply Löwdin orthogonalization before diagonalization. Usually for testing purposes. Default: False

Tool-Specific Parameters
---------------------

SIESTA Interface (siesta2J.py)
^^^^^^^^^^^^^^^^^^^^^^^^
* fdf_fname: Path to SIESTA's input fdf file. Default: ./
* fname: Output exchange parameters XML filename. Default: exchange.xml
* split_soc: Enable reading of spin-orbit coupling from SIESTA output. Default: False

ABACUS Interface (abacus2J.py)
^^^^^^^^^^^^^^^^^^^^^^^^^
* path: Location of ABACUS calculation files. Default: ./
* suffix: ABACUS calculation label (used in OUT.suffix). Default: abacus
