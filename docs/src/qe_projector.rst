Quantum ESPRESSO projector-Green exchange
=========================================

TB2J can compute magnetic exchange from Quantum ESPRESSO (QE) ultrasoft (US)
and PAW calculations through the projector-Green workflow. An instrumented QE
build writes all spectral ingredients — projector coefficients
:math:`P_{ni}=\langle\beta_n|\psi_{i}\rangle`, eigenvalues, occupations,
k-points and the spin-split augmentation operator — into a single binary dump
file at the end of a run. TB2J reads that dump into
:class:`~TB2J.projector_green.ProjectorGreenData`, reconstructs
:math:`G(R,E)` at runtime and traces the projector Green function into the
standard ``exchange.out`` output.

The dump format and the instrumented-build workflow are documented in
``TB2J/qe_patch/README.md`` in the TB2J source tree.

Prerequisites
-------------

* **QE fork, branch ``TB2J``**: the exporter lives on branch ``TB2J`` of
  ``git@gitlab.com:mailhexu/q-e.git`` (base: upstream ``develop``). Clone,
  check out the branch and build the plain serial code::

     git clone git@gitlab.com:mailhexu/q-e.git
     cd q-e && git checkout TB2J
     ./configure --disable-parallel
     make pw

* **``npool = 1``**: plane-wave pools are not supported; run ``pw.x`` without
  ``-nk``. A pool-parallel dump is rejected by both QE (at dump time) and the
  TB2J reader (``nkstot != nks``).
* **Collinear LSDA**: ``nspin = 2``. Noncollinear and spin-orbit runs are
  rejected; gamma-only runs are rejected as well.
* **US or PAW pseudopotentials**: pure norm-conserving pseudopotentials
  cannot be exported (see the support table below).

Workflow: scf + nscf with ``TB2J_DUMP``
---------------------------------------

Enable the dump with the environment variable ``TB2J_DUMP``; the QE input
files need no exporter-specific keywords. Run a converged scf and then an
nscf on a denser mesh, each with its own dump file:

.. code-block:: bash

   export TB2J_DUMP=scf_dump.bin
   pw.x -i scf.in > scf.out      # nspin = 2, collinear LSDA

   export TB2J_DUMP=nscf_dump.bin
   pw.x -i nscf.in > nscf.out    # dense mesh for the exchange integral

The nscf input must generate the **full Brillouin zone** and enough bands:

.. code-block:: text

   # nscf.in, &system
   nspin = 2
   nosym = .true.
   noinv = .true.     ! full-BZ k list: the dump stores the run's k-list
                      ! as-is, and G(R,E) needs every k of the mesh
   # &electrons / cards
   nbnd = ...         ! include the empty bands wanted in G(E)
   K_POINTS automatic
   nk1 nk2 nk3 0 0 0

TB2J consumes the **nscf** dump. Keep the scf dump for population
diagnostics: the ``becsum`` record of an nscf dump holds restart occupations
from the scf mesh, not dense-mesh weights (details in
``TB2J/qe_patch/README.md``).

Reading a dump
--------------

The raw parser lives in ``TB2J.interfaces.qe_projector``:

.. code-block:: python

   from TB2J.interfaces.qe_projector import parse_qe_dump

   dump = parse_qe_dump("nscf_dump.bin")   # QEProjectorDump: raw records + metadata

``parse_qe_dump`` validates the magic/version string, ``nspin == 2``,
``nkstot == nks``, the presence of US or PAW species, and every record
shape; violations raise ``ValueError`` with an explicit message. It keeps the
raw records (``deeq``, ``dvan``, ``qq_at``, the beta Gram diagnostic,
``becsum``, ``dbeta_xc``/``ddd_paw`` (v1.2), per-k coefficients, cell/species/ions metadata, ``et``/``wg`` and
the smearing parameters) for inspection.

The normalized view is produced by
:py:func:`TB2J.interfaces.qe_projector.read_qe_dump`, which returns a validated
:class:`~TB2J.projector_green.ProjectorGreenData`:

.. code-block:: python

   from TB2J.interfaces.qe_projector import read_qe_dump

   data = read_qe_dump("nscf_dump.bin")

The loader applies the QE conventions:

* beta channels are concatenated over atoms in QE ion order; atom ``na`` of
  species ``ityp[na]`` owns channels ``ofsbeta(na)+1 … +nh(ityp[na])``;
* the coefficients are **already dual**, so ``overlap_k`` is ``None`` and no
  Gram/overlap dressing is applied anywhere downstream;
* For v1.2 dumps the exchange vertex (``delta_total`` operator component)
  is ``M^-1 <beta|V_xc^up - V_xc^dn|beta> M^-1 + (deeq(up) - deeq(down))``
  with ``M`` the beta Gram: the projected multiplicative xc spin splitting
  (GPAW ``delta_xc`` analog, conjugated by the joint metric transformation)
  plus the augmentation-channel spin vertex (which already contains the PAW
  ``ddd_paw`` one-center splitting). For v1.0/v1.1 dumps ``hij`` falls back
  to the covariant separable operator ``deeq(up) - deeq(down)`` per
  atom block, converted from Ry to eV together with the eigenvalues and Fermi
  energies (factor 13.605693122994); coefficients are dimensionless;
* each k-point is assigned to its spin channel via ``isk``;
* provenance metadata is set to ``hij_definition =
  "qe_dbeta_xc_plus_deeq_spin_difference"`` (v1.2) or
  ``"qe_deeq_spin_difference"`` (v1.0/v1.1), ``coefficient_source = "qe_becp"``,
  ``coefficient_projector = "qe_beta"``, ``channel_interpretation =
  "qe_dual_to_beta"`` and ``operator_basis = "qe_dual_beta_channel"``.

``qq_at`` and the Gram diagnostic are stored for diagnostics only; ``qq_at``
belongs to the wavefunction overlap operator
:math:`S_\psi=I+\beta q_{at}\beta^\dagger`, **not** to the projector channel,
and is never used as a channel metric.

CLI usage
---------

``qe2J.py`` turns a dump into the standard TB2J results directory:

.. code-block:: bash

   qe2J.py nscf_dump.bin --sites Fe -o TB2J_results_fe --nz 30

.. list-table:: Options
   :header-rows: 1
   :widths: 26 74

   * - Argument
     - Meaning
   * - ``dump`` (positional)
     - Path to the nscf dump written with ``TB2J_DUMP``.
   * - ``--sites``
     - Element symbols selecting the magnetic sites that carry the spin
       vertex (same semantics as ``--elements`` in the sibling CLIs).
   * - ``-o``
     - Output directory for the TB2J results (``exchange.out``, ``TB2J.pickle``).
   * - ``--nz``
     - Number of continued-fraction poles for the contour integration.

``--overlap_mode`` and ``--overlap_rcond`` are deliberately **absent**. Those
options exist for exporters whose coefficients are non-orthogonal; the QE
``becp`` coefficients are already dual to the beta basis, so an overlap
correction would be a no-op and offering it would invite silent misuse.

Family support
--------------

.. list-table:: Pseudopotential-family support
   :header-rows: 1
   :widths: 22 18 60

   * - Family
     - Status
     - Notes
   * - US (± KB)
     - gated
     - Supported code path; end-to-end validation is gated on the Fe US
       golden-dump case.
   * - PAW
     - gated
     - Supported code path; end-to-end validation is gated on the Cu PAW
       golden-dump case.
   * - NC (+ KB)
     - excluded
     - Blocker: without augmentation charges the separable operator reduces
       to the spin-independent ``dvan``, so the spin vertex
       :math:`\Delta = deeq_{\uparrow}-deeq_{\downarrow}\equiv 0` and the
       exchange trace is identically zero. Use a Wannier-based interface
       instead.

Limitations
-----------

* Collinear LSDA only (``nspin = 2``); noncollinear and spin-orbit
  calculations are rejected.
* ``npool = 1``: no plane-wave pools, and no gamma-only runs.
* The dump is gfortran sequential unformatted with 4-byte little-endian
  record markers; dumps are not portable across endianness or other
  compilers' unformatted ABI.
* The full Brillouin zone must be present in the nscf run
  (``nosym = .true., noinv = .true.``).
* ``becsum`` in nscf dumps carries scf-mesh restart occupations, not
  dense-mesh weights; occupation-based diagnostics must use scf dumps.
