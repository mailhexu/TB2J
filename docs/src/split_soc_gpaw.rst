GPAW split-SOC: exchange and MAE from one collinear checkpoint
==============================================================

The GPAW split-SOC workflow produces magnetic exchange tensors, DMI and
symmetric anisotropy from a *single* collinear, spin-orbit-free GPAW
calculation.  The frozen density is never recomputed with SOC: the three
magnetization directions are obtained by diagonalizing GPAW's
second-variational spin-orbit operator on that one strength-zero state, and
the PAW exchange vertex stays collinear in the ``psi`` gauge.  Each leg is
rotated into the lattice frame (:math:`T_\chi = O T_\psi O^T` with
:math:`O\mathbf e_z = \mathbf n_\chi`) and the three legs are merged into the
standard rank-six reconstruction with :py:mod:`TB2J.io_merge`.

The same run also produces a magneto-crystalline anisotropy (MAE) comparison:
per-direction band energies from GPAW itself and an independent second-order
contour insertion, reported side by side with an explicit tolerance check.

Strength-zero recipe
--------------------

1. Run a converged, spin-polarized **no-SOC** GPAW ground state with the old
   API and save the PAW density and wavefunctions:

   .. code-block:: python

      from gpaw import GPAW, PW, FermiDirac

      calc = GPAW(
          mode=PW(500),
          xc="PBE",
          kpts=(6, 6, 6),
          symmetry="off",
          spinpol=True,
          nbands=12,
          occupations=FermiDirac(0.05),
          txt="fe_collinear.log",
          legacy_gpaw=True,
      )
      atoms.calc = calc
      atoms.get_potential_energy()
      calc.write("fe_collinear.gpw", mode="all")

2. Run the CLI on that one checkpoint (below).
3. Converge the frozen k mesh, band window, ``--nz`` and ``--smearing``
   before reading exchange or MAE values as material predictions.

The workflow never performs an SOC self-consistency loop.  New-API
checkpoints and checkpoints that already contain projected SOC data are
rejected.

Command line
------------

.. code-block:: bash

   gpaw_split_soc2J.py --input fe_collinear.gpw \
       --output_path TB2J_results_gpaw_split_soc --Rcut 3 \
       --nz 30 --smearing 0.05 --index_magnetic_atoms 1 2

.. list-table:: Options
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Meaning
   * - ``--input``
     - Converged collinear no-SOC legacy ``.gpw`` checkpoint (required).
   * - ``--output_path``
     - Output root; legs are written to ``x/``, ``y/``, ``z/`` and the merged
       tensors to ``merged/`` (default ``TB2J_results_gpaw_split_soc``).
   * - ``--Rcut``, ``--nz``, ``--smearing``
     - Exchange real-space cutoff (Å), continued-fraction poles and CFR
       smearing in eV, as in the collinear projector workflow.
   * - ``--index_magnetic_atoms``
     - **1-based** indices of the magnetic sites carrying exchange vertices.
       Default: all sites with a nonzero frozen local moment.
   * - ``--vertex_component``
     - ``delta_total`` (default) or ``delta_xc``; see :doc:`projector_green`
       for the PAW operator components.  The vertex stays collinear; only
       the KS-band propagator carries SOC.
   * - ``--scale``
     - Frozen SOC operator strength.  It rescales the GPAW SOC band energies
       *and* the KS-band exchange propagator, never the frozen SCF density.
   * - ``--mae-contour-tolerance``
     - Reported bound (eV) for the second-order MAE comparison
       (default ``5e-6``).

The SOC operator covers **every** atom (ligands included), while exchange
vertices occur only on the selected magnetic sites.

Outputs
-------

The CLI writes ``x/``, ``y/``, ``z/`` and ``merged/`` exchange outputs plus
``split_soc_report.json``. The leg angles in degrees are (90,0), (90,90) and
(0,0); the spin moments stored in each leg follow that direction. The PAW
exchange vertex remains collinear in the ``psi`` gauge; no second SOC SCF is
performed.

MAE comparison how-to
---------------------

Each leg writes two independent SOC band-energy estimates into
``split_soc_report.json``:

* ``band_energy_eV`` — the exact
  ``BZWaveFunctions.calculate_band_energy()`` of the second-variational
  states, i.e. GPAW's own band contribution to the total energy;
* ``contour_second_order_shift_eV`` — the independent second-order contour
  insertion :math:`-\mathrm{Im}\int \mathrm{Tr}[(G_0 W_{SO})^2]/(2\pi)`
  evaluated on the same CFR contour, pole count and smearing as the exchange
  calculation.

Both are reported relative to the ``z`` leg (``relative_to_z_eV``,
``contour_relative_to_z_eV``), and their difference
(``contour_residual_eV``) is compared against
``--mae-contour-tolerance`` (``contour_within_tolerance``).  The contour
resolvent references the *strength-zero* Fermi level
(:math:`G_0(z) = [z + \mu - H_0]^{-1}`), so shifting all band energies and
the Fermi level together leaves the comparison unchanged; the GPAW Fermi
level itself cannot be omitted.

A ``contour_within_tolerance: false`` is a real diagnostic failure, not a
fallback: it can reflect higher-order SOC terms, k-mesh error, a too-narrow
band window, occupation mismatches, or contour error.  Investigate before
using the MAE numbers.  Calibration on real systems: bcc Fe reproduces the
GPAW band-energy MAE bitwise (second-order contour residual
:math:`<7\times 10^{-8}` eV at 12 poles), and the 6×6×6 fcc Ni fixture below
closes the comparison well inside the default tolerance.

Each leg also records a **band-window study**: the actual exchange between
the last two paired band prefixes at the production k mesh, R grid, contour
and smearing, with the measured change and an explicit ``converged`` flag.
The study is embedded in the per-leg ``TB2J.pickle``, ``exchange.out`` and
``Multibinit/exchange.xml``.  A false flag is an unresolved error bar, **not**
a convergence certificate: the 6³ fcc Ni 24-band fixture, for example, moves
by :math:`6.65\times 10^{-4}` between 22 and 24 bands at a
:math:`10^{-6}` tolerance and is explicitly *not* window-converged despite
its neV-level MAE residual.

Provenance
----------

Every leg records the strength-zero reference (input path and SHA-256 when
invoked from a file), the PAW operator source, the SOC operator coverage,
the spin frame with the leg angles :math:`(90,0)`, :math:`(90,90)`,
:math:`(0,0)`, the band window with its convergence study, and the merge
mode.  These blocks are embedded in each leg ``TB2J.pickle``,
``exchange.out`` and ``Multibinit/exchange.xml``.  The merged artifacts keep
**all three** leg records as distinct, path-keyed blocks instead of
inheriting the last leg's frame, and ``split_soc_report.json`` mirrors them.
Rerunning the generic ``TB2J_merge.py x y z`` on the same leg directories
retains the same provenance blocks.

Worked example: fcc Ni gate
---------------------------

.. code-block:: bash

   python examples/projector_green/gpaw_fcc_ni_split_soc.py --build \
       --input ni_fcc_pbe_nosoc.gpw --output TB2J_results_ni_split_soc

The example builds one 6×6×6 collinear fcc Ni PBE checkpoint (``--build``),
runs the three SOC legs on it, and asserts

* the MAE band energies equal ``soc_eigenstates(...).calculate_band_energy()``
  bitwise for all three directions and stay within the contour tolerance;
* the merged tensors have rank six;
* inversion-odd DMI and cubic-anisotropy nulls hold on the full-BZ R grid;
* the shell-resolved :math:`J_\mathrm{iso}` spread across legs stays below
  5 µeV (representative run: 1.36 µeV; DMI norms at the 10⁻³² meV level;
  Jani 7.1 µeV).

The gate report is written to ``ni_symmetry_gate.json``.  Serial execution
is required by the legacy GPAW API.
