ABINIT PAW split-SOC exchange
=============================

The ABINIT PAW split-SOC workflow consumes one collinear, spin-orbit-free
``savetb2j`` export that additionally carries the unit-strength SOC operator
(``savetb2j_soc 1``, schema version 1.1), and reconstructs the same
three-direction merged tensors as the GPAW workflow: all-atom SOC enters the
KS-band propagator, collinear PAW ``delta_total``/``delta_xc`` vertices stay
on the selected magnetic sites, three spinaxis legs are rotated to the
lattice frame (:math:`T_\mathrm{lattice} = O\, T_\mathrm{leg}\, O^T`) and
merged with :py:mod:`TB2J.io_merge`.  Only absolute second-variational
exchange is written; insertion derivatives are never labelled as
:math:`J`.

Strength-zero recipe
--------------------

1. Run a **collinear** PAW ground state (``usepaw 1``, ``nsppol 2``,
   ``nspinor 1``) with an explicit full-Brillouin-zone k-point list
   (``kptopt 0``) and export it with

   .. code-block:: text

      savetb2j 1        # schema 1.0/1.1 projector export (cprj bands, operators)
      savetb2j_soc 1    # schema 1.1: unit-strength soc_pauli at the frozen density

   ABINIT writes ``<prefix>_SAVETB2J.nc`` at the frozen strength-zero
   density.  The ``soc_pauli`` component is the unit-strength
   ``pawdijso`` Pauli operator in Hartree, evaluated *without* any SOC SCF.

2. Run the Python driver on that file (below).
3. Converge the k-point set, band window, ``nz`` and ``smearing_eV``; each
   leg reports its own measured band-window error bar.

File contract and orientation
-----------------------------

The file contract is specified in :doc:`abinit_savetb2j_schema` (section
*Version 1.1: ``soc_pauli``*).  The TB2J loader validates the Hartree units,
the Hermitian dense blocks, the
:math:`\mathbf L\cdot\mathbf S` packing relations
(:math:`dd=-uu`, :math:`du=-\overline{ud}`) and the provenance attributes
(unit strength, lattice-frame ``spinaxis 0 0 1``, all-atom coverage,
SOC-only term class, frozen-density reference), then normalizes the
component to eV with the operator orientation
:math:`O[p,q,s,t] = D[q,p,t,s]` required by the cprj coefficient
convention :math:`c[p]=\langle p|\psi\rangle`.  Transposing projector
indices alone reverses the complex spin-flip matrix elements and fails the
iodine 5p :math:`\mathbf L\cdot\mathbf S` element-level oracle
(residual < 10⁻¹⁰ in the normalized orientation).  Files that violate any
of these rules are rejected at load time.

Driver usage
------------

.. code-block:: python

   from TB2J.interfaces.abinit_paw_split_soc import (
       gen_exchange_abinit_paw_split_soc,
   )

   gen_exchange_abinit_paw_split_soc(
       "fe_k8_soc1o_SAVETB2J.nc",
       output_path="TB2J_results_abinit_split_soc",
       index_magnetic_atoms=[0],   # zero-based Python indices
       Rcut=8.0,
       nz=12,
       smearing_eV=0.05,
   )

.. list-table:: Selected arguments
   :header-rows: 1
   :widths: 26 74

   * - Argument
     - Meaning
   * - ``index_magnetic_atoms`` / ``magnetic_elements``
     - Zero-based magnetic sites carrying the collinear vertices.  One of
       the two (or exported magnetic moments) is **required**: the SOC
       operator covers every atom, so silently selecting all sites would
       treat every ligand as magnetic.
   * - ``vertex_component``
     - ``delta_total`` (default) or ``delta_xc``; the SOC propagator is
       never used as a vertex.
   * - ``lam``
     - SOC strength multiplying the propagator.  ``lam=0`` gives the
       SOC-off leg used as the anchor against the collinear
       schema-1.0 exchange.
   * - ``mode``
     - Only ``"second_variation"`` is accepted: the three-leg SpinIO output
       requires absolute exchange, not insertion derivatives.
   * - ``spinat_magnitude``
     - Norm of the per-site output ``spinat`` vectors, aligned with each
       leg axis (default 1.0).
   * - ``Rcut``, ``Rpts``, ``nz``, ``smearing_eV``, ``legs``
     - Exchange controls as in the collinear workflow; ``legs`` must be the
       ordered triplet ``("x", "y", "z")``.

Magnetic sites and spinat directions
------------------------------------

The exporter stores no moment field.  Signed per-site ``spinat`` directions
use exported ``magnetic_moments`` when available; otherwise they take the
**opposite** of the sign of the frozen :math:`H_\uparrow - H_\downarrow`
PAW potential trace, because the majority-spin potential is lower on a
positively magnetized site.  This keeps AFM sublattices opposite and real
positive moments positive — real bcc Fe has moment :math:`\approx 1.28\,\mu_B`
with a *negative* ``delta_total`` trace.  A site whose vertex trace vanishes
is rejected instead of receiving an arbitrary direction.

Outputs and provenance
----------------------

Each leg is written to ``leg_x/``, ``leg_y/``, ``leg_z/`` as a standard
noncollinear TB2J results directory with ``spinat`` along the leg axis,
plus ``split_soc_provenance.json``; the three legs are merged into
``output_path`` (``TB2J.pickle``, ``exchange.out``,
``Multibinit/exchange.xml``, merged provenance JSON).  Each leg records

* the input checkpoint path and its actual SHA-256;
* the schema version and normalized ``soc_pauli`` operator source;
* the spin frame: exported ``spinaxis`` re-quantized per leg with the
  ABINIT spinaxis SU(2) rotation, and the resulting lattice-frame axis;
* the band window with a real two-prefix paired-window study, its measured
  change, tolerance and ``converged`` flag;
* the magnetic vertex sites, vertex component and ``spinat`` sign source;
* the merge mode ``three_leg_rotate_merge``.

The merge preserves all three leg records; a false ``converged`` flag must
stay visible and calls for a larger ABINIT band window, not for a favourable
error bar.

Physical anchors and gates
--------------------------

* **SOC-off anchor**: with ``lam=0`` every leg reproduces the existing
  collinear schema-1.0 savetb2j exchange shell by shell (real 8-k bcc Fe
  fixture: agreement within :math:`10^{-8}` eV; the export-on/off
  ``OUT.nc`` total energies are bit-identical).
* **Non-vacuous baseline**: a Γ-only Fe export has
  :math:`|J(R{=}1)|\approx 10^{-20}` eV, which makes SOC comparisons
  meaningless.  The 2×2×2 full-BZ Fe fixture has
  :math:`J(R{=}(1,0,0)) = 33.992` meV at ``nz=12``; with SOC on, the x/y/z
  shells are ≈ 33.89/33.86/33.86 meV.  Its 22→24 band study moves by
  :math:`2.25\times 10^{-4}` at a :math:`10^{-6}` tolerance and is *not*
  window-converged.
* **Projector spectrum**: the Fe radial p and d shells reproduce the
  :math:`\mathbf L\cdot\mathbf S` j-manifolds — p: :math:`2+4`,
  d: :math:`4+6` — at machine precision, and the iodine valence 5p block
  pins the orientation and phase conventions.
* **Eigenvalue response**: on the same Fe fixture, the loaded-component
  consumer (``build_paw_split_soc_leg`` → ``paw_split_soc_band_w_soc`` →
  ``second_variation_spectrum``) matches a native fixed-density ABINIT
  spinor response at :math:`\lambda: 0 \to 0.005`
  (8 k-points × 24 states) to within 0.886 meV maximum and 0.065 meV RMS.
  This is a small-strength band-eigenvalue gate, **not** a full-strength
  certificate: :math:`\lambda=1` remains window-unconverged.

Runnable example
----------------

``examples/projector_green/abinit_paw_fe_split_soc.py`` runs the three-leg
driver plus the SOC-off anchor comparison on a real schema-1.1 export:

.. code-block:: bash

   python examples/projector_green/abinit_paw_fe_split_soc.py \
       --input fe_k8_soc1o_SAVETB2J.nc \
       --output TB2J_results_fe_paw_split_soc

The script checks the merged output, prints the first-shell
:math:`J_\mathrm{iso}` per leg and, with ``--anchor``, compares
``lam=0`` legs against the collinear ``delta_total`` exchange on the same
R grid.  The ABINIT inputs that produce the fixture are described in the
strength-zero recipe above.
