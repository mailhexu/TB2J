Spin-Spiral Exchange from Frozen Spiral States
===============================================

TB2J can compute exchange parameters from a *frozen spin-spiral state*: a
self-consistent rotating-frame reference produced by TBUpy at a single-q
spiral wavevector, stored as a ``*.spiral.nc`` bundle.  The exchange is
obtained from the magnetic-force-theorem curvature about that reference
and written through the standard TB2J output tree, so all downstream
consumers (Multibinit, Vampire, TomASD, magnon bands) work unchanged.

Overview
--------

The data flow is::

   TBUpy per-q rotating-frame SCF (torque-free reference)
       -> *.spiral.nc frozen-state bundle
       -> TB2J ExchangeSpiral (MFT curvature kernels + tensor mapping)
       -> standard TB2J results directory + spiral_diagnostics.json

The seam follows the spiral frozen-state bundle contract: TBUpy owns the
multi-sublattice generalized-Bloch assembly and the per-q reference;
TB2J owns the Green-function response kernels, the Heisenberg/tensor
mapping, and the output.  The normative conventions are pinned by the
derivation reports in ``docs/sympy/``:

* ``docs/sympy/spiral_state_mft.py`` (``.md`` narrative alongside) —
  two-channel force-theorem curvature about the constrained-planar
  spiral, the Heisenberg mapping, and the q=0 LKAG anchor;
* ``docs/sympy/spiral_green_function_mft.py`` — folded-pencil resolvent
  and unfolding to real-space blocks;
* ``docs/sympy/spin_spiral_generalized_bloch.py`` — folded
  ``Hq(k), Sq(k)`` assembly.

Requirements
------------

The spiral path reuses the TBUpy assembler (single source of truth of
the rebuild rule), so ``import tbupy`` must work in the environment
(see :doc:`tbupy` for the shared environment).  Reading a
``*.spiral.nc`` file uses the tbupy reader lazily; a ``SpiralState``
object can be passed directly instead of a file.

Conventions
-----------

The bundle is spin-interleaved (orb0-up, orb0-down, orb1-up, ...) in eV.
The stored convention puts the spiral rotation axis on +y: the local
exchange field of site ``(a, mu)`` rotates in the x-z plane at the lab
angle ``Theta = 2 pi q.(a + tau_mu) + phi_mu``.  About the
constrained-planar reference, two local transverse perturbation channels
are evaluated:

* out-of-plane tilt ``delta_a``: ``V1 = B sigma_y`` (site independent —
  the tilt direction is the spiral normal), and
* in-plane rotation ``beta_a``: ``V1 = B d_theta field =
  B(-sin Theta sigma_z + cos Theta sigma_x)``,
* with the second-order tip-back ``V2 = -1/2 (delta^2 + beta^2) B_a``.

The pair kernels assemble into the real symmetric curvature matrices
``C^dd``, ``C^bb`` and ``C^db`` over ring sites.  The mapping onto the
standard TB2J tensors is:

.. list-table:: Mapping onto the ExchangeNCL conventions
   :header-rows: 1

   * - Quantity
     - Definition
     - Output
   * - Heisenberg exchange
     - ``J^spiral_ab = -C^dd_ab`` for ``a != b``
     - ``exchange_Jdict[(R, i, j)] = J / sgn(S_i . S_j)``
   * - Diagonal consistency
     - ``C^dd_aa = sum_b J_ab cos(Theta_a - Theta_b)``
     - gate (hard fail unless overridden)
   * - Mixed channel
     - ``C^db = 0`` (exact class of the planar path)
     - checked on every run
   * - Tensor conversion
     - ``A^{00}_{ij}(R) = -i C^dd_ij(R)``, all other components zero
     - ``ExchangeNCL.A_to_Jtensor`` applies verbatim
   * - DMI / anisotropic
     - antisymmetric and symmetric-transverse slots of ``A``
     - zero for the single-axis planar path
   * - Biquadratic
     - ``B = Im(A^{zz}) = 0``, ``J' = Im(A^{00})``
     - ``biquadratic_Jdict`` with ``B = 0``

The four-component tensor mapping places the local-frame out-of-plane
channel on the identity slot of the ``{0, x, y, z}`` Pauli basis, so the
existing conversion formulas produce ``J_iso = Im(A00 - Axx - Ayy -
Azz) = -C^dd_ij`` unchanged while the DMI, Jani and biquadratic slots
evaluate to their symmetry-expected zero.

The stored exchange carries the AFM normalization ``J / sgn(S_i . S_j)``
of the reference local moments (the same convention as the collinear and
q-space TB2J paths): pairs whose reference moments are antiparallel flip
sign, so the stored ``J`` refers to spin magnitudes in the locally
rotated frames.

Because ``A^{00}`` is purely imaginary (the curvature is real
symmetric), the conversion is exact rather than approximate: no
longitudinal or antisymmetric content exists in the two-channel planar
path, and any future DMI extraction requires the three-axis rotation
sets that are out of scope here.

Workflow
--------

Step 1: produce the frozen spiral bundle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the TBUpy per-q rotating-frame SCF to torque tolerance and save the
bundle (story-002 provider, ``tbupy.spiral_state``).  The bundle carries
the collinear hopping channels, the local field magnitudes, the
rotating-frame density matrix and Hubbard potential, the frozen Fermi
level, and the per-orbital torque norms used by the gate.

Step 2: run ExchangeSpiral
^^^^^^^^^^^^^^^^^^^^^^^^^^

From Python:

.. code-block:: python

   from TB2J.exchange_spiral import ExchangeSpiral

   calc = ExchangeSpiral.from_spiral_state(
       "crI3.spiral.nc",
       kernel="contour",       # or "eigenbasis" (degeneracy-safe reference)
       ncell=6,                # 0 (default) derives it from q commensurability
       width=None,             # smearing override in eV (default: bundle value)
       Rcut=None,              # pair distance cutoff in Angstrom
       eq_table=True,          # optional E(q) diagnostic table
   )
   report = calc.run()
   calc.write_output(path="TB2J_spiral_results")

   spinio = calc.get_spinio()  # standard SpinIO without writing

or from the command line (no pyproject entry point yet — use ``python
-m``):

.. code-block:: bash

   python -m TB2J.scripts.tb2j_spiral \
       --spiral-state crI3.spiral.nc \
       --ncell 6 \
       --output TB2J_spiral_results \
       --kernel contour \
       --eq-table \
       --symbols Cr

``--gate-override`` downgrades gate violations to flagged report entries
instead of failing.  ``--cell`` (nine numbers, row-major) and
``--symbols`` set the output structure of the primitive cell; the
default is the fractional ``taus`` in a unit cell with ``X`` placeholders.

Step 3: read the outputs
^^^^^^^^^^^^^^^^^^^^^^^^

The output directory is the standard TB2J tree (``TB2J.pickle``,
``exchange.out``, ``structure.vasp``, Multibinit/Vampire/TomASD/ESPInS
inputs) plus ``spiral_diagnostics.json``.  The pickle loads through the
usual consumers:

.. code-block:: python

   from TB2J.io_exchange import SpinIO
   from TB2J.magnon.magnon3 import Magnon

   exc = SpinIO.load_pickle(path="TB2J_spiral_results")
   magnon = Magnon.load_from_io(exc)

Parameters
----------

``SpiralParameters`` (``TB2J.exchange_spiral``) follows the
``MagnonParameters`` protocol:

.. list-table:: Main parameters
   :header-rows: 1

   * - Parameter
     - Default
     - Meaning
   * - ``ncell``
     - ``0``
     - Ring size; 0 derives the commensurate denominator of ``q``.
   * - ``kernel``
     - ``"contour"``
     - ``"contour"`` (Matsubara resolvent, primary) or
       ``"eigenbasis"`` (degeneracy-safe reference).
   * - ``n_matsubara``
     - ``3000``
     - Matsubara points of the contour kernel.
   * - ``width``
     - ``None``
     - Smearing width override (eV); default is the bundle metadata value.
   * - ``reference_protocol``
     - ``"frozen-bundle"``
     - The required torque-free per-q SCF reference protocol (P-c).
   * - ``spiral_axis``
     - ``(0, 1, 0)``
     - Stored-convention rotation axis; other axes raise.
   * - ``goldstone_tol`` / ``torque_tol``
     - ``1e-7`` / ``1e-6``
     - Gate tolerances for the zero modes.
   * - ``diag_consistency_tol``
     - ``1e-6``
     - Tolerance of ``C^dd_aa = sum_b J_ab cos(Theta_a - Theta_b)``.
   * - ``q0_anchor`` / ``q0_anchor_tol``
     - ``True`` / ``1e-5``
     - Mandatory q=0 LKAG anchor (``C^dd = C^bb = M``, ``C^db = 0``);
       the tolerance is FD-limited.
   * - ``gate_override``
     - ``False``
     - Report gate violations instead of raising.
   * - ``eq_table`` / ``q_set``
     - ``False`` / ``None``
     - Optional frozen band energy ``E(q)`` table (points: 0, +q, -q).
   * - ``Rcut``, ``cell``, ``symbols``
     - ``None``, unit cell, ``X``
     - Output structure of the primitive cell.

Diagnostics and gates
---------------------

Every run writes ``spiral_diagnostics.json`` with:

* ``torque_norms`` — the per-orbital torque residual recorded in the
  bundle;
* ``zero_modes`` — the Goldstone residuals ``C^bb . 1`` and the torque
  residuals ``C^dd . cos(Theta)``, ``C^dd . sin(Theta)`` of the
  extraction;
* ``nonheisenberg_Cbb`` — the norm ``||C^bb + J o cos(Theta)||`` of the
  in-plane block against the Heisenberg prediction built from the
  extracted ``J`` (off-diagonal form, and the full matrix with the
  Heisenberg diagonal ``sum_b J_ab cos(Theta_a - Theta_b)``).  It is
  exactly zero for strictly bilinear energies; a finite value measures
  beyond-pairwise, nonlinear-itinerant and frozen-``V_U`` effects.
* ``cdb_max`` — the ``C^db`` check;
* ``gates`` — per-gate status (``passed``/``flagged``/tolerance), and
  ``eq_table`` when requested.

Gates hard-fail with a diagnostic report (``SpiralGateError``) unless
``gate_override`` is set.  The torque gate flags — instead of fails —
references whose recorded ``torque_norms`` already exceed the tolerance.
The q=0 LKAG anchor runs only at ``q = 0``.

Interpretation bounds
---------------------

``J^spiral`` is the force-theorem curvature about the frozen spiral
reference: it is exact within the linear-response cone of that
reference.  The frozen-``V_U`` freeze-vs-respond tradeoff is documented
in the architecture decision records (ADR-S9 of the spin-spiral MFT
spec); the non-Heisenberg diagnostic quantifies the deviation from the
pairwise form on the same run.  The q=0 anchor and the diagonal
consistency gate are the fastest health checks of a new system.
