Quantum ESPRESSO projector-Green exchange
=========================================

TB2J computes magnetic exchange from Quantum ESPRESSO (QE) collinear
calculations using either separable KB beta channels (the default) or UPF
atomic-wavefunction channels (``TB2J_PROJECTORS=atomic``). An instrumented QE
build writes coefficients, eigenvalues, occupations, k-points and the
spin-split projector operator into a versioned binary dump
file at the end of a run. TB2J reads that dump into
:class:`~TB2J.projector_green.ProjectorGreenData`, reconstructs
:math:`G(R,E)` at runtime and traces the projector Green function into the
standard ``exchange.out`` output.

The dump formats and instrumented-build workflow are documented in
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
* **Pseudopotential support depends on projector mode**: KB accepts NC, US,
  and PAW; atomic mode requires ``PP_PSWFC`` atomic wavefunctions in every UPF.

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

Reading a KB dump
-----------------

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

Atomic-wavefunction dumps
-------------------------

Select ``TB2J_PROJECTORS=atomic`` alongside ``TB2J_DUMP`` for both runs.
QE constructs pseudo-atomic orbitals from each UPF's ``PP_PSWFC`` block;
an UPF without atomic wavefunctions (including the tested SG15
``Fe_ONCV_PBE-1.0.upf``) is rejected. The distinct
``TB2JQEATWFC1.0`` format stores primal
:math:`C_{ni}(k)=\langle\phi_n(k)|\psi_i(k)\rangle` and the full
k-dependent Gram :math:`M_{nm}(k)=\langle\phi_n(k)|\phi_m(k)\rangle`,
including intersite blocks. The onsite spin vertex is the full-BZ
weighted :math:`R=0` projection of the smooth XC difference plus
:math:`B(k)(D^\uparrow-D^\downarrow)B(k)^\dagger`, with
:math:`B_{na}(k)=\langle\phi_n(k)|\beta_a(k)\rangle`. Keeping only the
first (often Γ) point would fold neighboring periodic images into the
onsite operator. The reader uses the same atomic basis for the coefficients,
metric and vertex, converting Ry to eV; no KB-channel metric is applied.

Use ``parse_qe_atomic_dump`` for raw records or ``read_qe_atomic_dump`` for
normalized :class:`~TB2J.projector_green.ProjectorGreenData`. The CLI
detects the format by magic. Atomic mode defaults to the full inverse of
:math:`M(k)`; ``--overlap_mode`` selects ``inverse``, ``svd``,
``lowdin``, ``tikhonov`` or ``plain`` and ``--overlap_rcond`` controls
regularized modes. These flags are rejected for KB dumps, whose coefficients
are already dual and need no channel metric.

CLI usage
---------

``qe2J.py`` turns a dump into the standard TB2J results directory:

.. code-block:: bash

   qe2J.py --input nscf_dump.bin --output_path TB2J_results_fe --elements Fe --Rcut 6 --nz 30
   qe2J.py --input atomic_nscf.bin --output_path TB2J_results_atomic --elements Fe --Rcut 6 --overlap_mode inverse

``--index_magnetic_atoms`` accepts 1-based atom indices instead of
``--elements``; ``--smearing`` is the contour smearing in eV (default
0.05). See ``examples/qe/feo/`` for reproducible FeO US and PAW KB
input decks and measured shell exchange.

KB family support
-----------------

.. list-table:: KB pseudopotential-family support
   :header-rows: 1
   :widths: 22 18 60

   * - Family
     - Status
     - Notes
   * - US (rrkjus)
     - validated
     - bccFe J1 = 16.15 meV at 20^3 full BZ, cutoff-insensitive 60-100 Ry
       (GPAW reference 14.73 meV; within the cross-code spread).
   * - PAW (kjpaw)
     - validated
     - bccFe J1 = 14.76 meV vs GPAW 14.73 meV (0.2%).
   * - NC (+ KB, e.g. ONCV)
     - validated
     - bccFe (Fe_ONCV_PBE-1.0, 16 valence electrons) J1 = 15.22 meV.
       The historical ``deeq``-only :math:`\Delta \equiv 0` blocker is lifted
       by the v1.2 ``dbeta_xc`` vertex; ``becsum`` is zero-filled in NC-only
       runs and is not an occupation-parity reference.

The optional atomic basis is **not validated as a quantitative
replacement** by these KB comparisons. In matched bccFe UPF/12³ runs,
atomic versus KB :math:`J_1` is PAW 20.4267 versus 14.7601 meV, US
24.3862 versus 17.6958 meV, and NC with ``PP_PSWFC`` 24.5939 versus
10.0730 meV. The latter UPF differs from the no-atomic-wavefunction
ONCV one used in the KB table. A weighted :math:`R=0` atomic vertex
removes a first-Γ periodic-image artifact but does not make the finite
atomic and KB subspaces interchangeable. See the detailed comparison
in ``TB2J/qe_patch/README.md``.

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
* ``becsum`` in KB nscf dumps carries scf-mesh restart occupations, not
  dense-mesh weights; occupation-based diagnostics must use scf dumps.
