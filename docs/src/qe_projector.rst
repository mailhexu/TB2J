Quantum ESPRESSO projector-Green exchange
=========================================

TB2J computes magnetic exchange from Quantum ESPRESSO (QE) calculations
using separable KB beta channels (the default) or, for collinear runs,
UPF atomic-wavefunction channels (``TB2J_PROJECTORS=atomic``). The KB
exporter also writes a distinct NC1.1 noncollinear spinor schema for magnetic
US/PAW runs; SO-bearing families require native frozen-source/action
checks and a three-reference tensor-consistency gate. Atomic mode
remains collinear.
An instrumented QE
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
* **Magnetic source**: collinear KB/atomic requires ``nspin=2``;
  noncollinear KB requires ``noncolin=.true.``, ``domag=.true.`` and
  ``nspin=4`` internally. A ``lspinorb=.true.`` run requires an actual
  SO-bearing UPF. Gamma-only runs are rejected for every mode.
* **Pseudopotential support depends on projector mode**: collinear KB
  accepts NC, US and PAW; noncollinear KB supports US and PAW after
  native action/source checks. Atomic mode requires ``PP_PSWFC`` atomic
  wavefunctions in every UPF and remains collinear.

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
``nkstot == nks``, family/version compatibility and record shapes;
violations raise ``ValueError`` with an explicit message. It keeps the
raw records (``deeq``, ``dvan``, ``qq_at``, the beta Gram diagnostic,
``becsum``, v1.2 ``dbeta_xc``/``ddd_paw``, per-k coefficients,
cell/species/ions metadata, ``et``/``wg`` and smearing parameters)
for inspection. NC requires v1.2 for a nonzero exchange vertex.

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
  ``ddd_paw`` one-center splitting). Earlier v1.0/v1.1 dumps are readable
  for diagnostics, but their ``deeq``-only vertex failed the bccFe
  exchange gate and ``qe2J.py`` rejects them for exchange. Energies and
  Fermi levels are converted from Ry to eV (factor 13.605693122994);
  coefficients are dimensionless;
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

Noncollinear KB spinor dumps
----------------------------

For ``noncolin=.true.`` with a magnetic source (``domag=.true.``), the
KB exporter writes the separate ``TB2JQEDUMPNC1.0`` or ``NC1.1`` schema.
NC1.0 remains scalar-relativistic-only. NC1.1 retains nine native
magnetic-source responses for SO-bearing US/PAW: each original
Cartesian field component is installed, in turn, into every Pauli
destination slot at frozen density and fixed ionic/SOC Hamiltonian.
PAW one-center magnetic source components are rotated/zeroed in
lockstep. The exporter verifies ``add_vuspsi`` total/action parity,
magnetic subtraction, diagonal response reconstruction, quarter-turn,
tilted rotation and finite-difference source identities. Bare-spin
conjugation fails for SO-bearing QE ``fcoef`` channels and is only an
informational diagnostic, **not** the tangent.

``parse_qe_spinor_dump`` retains native ``becp%nc`` coefficients,
``deeq_nc_total`` diagnostic, separately field-dependent
``deeq_nc_magnetic``, Hermitian projected smooth field and, for NC1.1,
``D_response(nhm,nhm,nat,4,3,3)`` with axes
``(beta_i,beta_j,atom,(uu,ud,du,dd),destination_c,source_j)``.
``read_qe_spinor_dump`` returns ``ProjectorGreenData(nspinor=2)`` with
already-dual coefficients, one full-BZ k set and a magnetic-only spinor
operator. Only the projected smooth field uses a Hermitian beta-Gram
pseudoinverse with relative eigenvalue cutoff ``1e-6``; QE's
fully relativistic Fe beta Gram reaches condition number ``1.1e10``,
which makes a direct inverse unphysical. Neither the dual Green
coefficients nor native ``D_response`` are Gram dressed.

With :math:`A_j=\langle\beta|B_j|\beta\rangle` and native
:math:`D_{c\leftarrow j}`, the magnetic splitting and explicit
site tangent are

.. math::

   \Delta &= 2 E_{\mathrm{Ry}}\left[
      \sum_j(M^+A_jM^+)\otimes\sigma_j+D_{\mathrm{mag}}\right],\\
   V_a &= E_{\mathrm{Ry}}\sum_{cj}
      K^{(a)}_{cj}\left[(M^+A_jM^+)\otimes\sigma_c+
      D_{c\leftarrow j}\right],\qquad
   K^{(a)}_{cj}=\sum_d\epsilon_{cdj}(\mathbf n\times\mathbf e_a)_d .

Here :math:`V_a=\partial H_{\rm mag}/\partial\theta_a
=\frac12\partial\Delta/\partial\theta_a`; fixed ionic/SOC terms in
``deeq_nc_total`` do not rotate. The reader validates Hermiticity,
the longitudinal identity, the native diagonal reconstruction and
the scalar-relativistic commutator limit. ``NC1.0`` SO-bearing dumps
are refused. The final four-component record is deterministic zero:
QE's NSCF ``becsum`` is uninitialized, **not** an occupation check;
band occupations come from ``wg/wk``.

At 12³ full-BZ points, 48 spinor bands, 30 poles and 0.05 eV
smearing, the scalar-relativistic Fe US NC1.1 z transverse leg gives
17.6997098 meV versus matched collinear 17.6957657 meV
(difference 0.0039441 meV); PAW NC1.1 gives 14.7589664 meV
versus matched collinear 14.7600550 meV (difference
0.0010886 meV). Both meet the G3 bound of 0.01 meV. The three
independent PBE fully relativistic Fe US x/y/z legs produce a
nearest-neighbor-only full tensor at ``--Rcut 2.6``: the maximum
repeated-row spread is ``8.367697e-9 eV`` below the ``5e-8 eV``
gate and first-neighbor :math:`J_{\rm iso}=17.7174` meV.

Independent scalar-relativistic US x/y/z references on the
same 12³ mesh also pass nearest-neighbor rank-nine merging
(spread ``1.012117e-8 eV``), but give
:math:`|J^{\rm ani}_{yz}|<2.3\times10^{-15}` meV at
:math:`R=(1,0,0)`, compared with the SO-bearing US
``0.000863377 meV``. This is an independent nonzero
anisotropy sensitivity control; the isotropic difference
between distinct scalar/full-relativistic UPFs is not a
pure SOC shift.

At ``--Rcut 6``, the same family is **rejected** (far-shell
spread ``2.586256e-5 eV``); no six-angstrom full tensor is
validated. The unphysical self pair ``R=0,i=j`` is excluded before
the repeated-row gate because it never enters ``exchange.out``.
Do not raise the consistency tolerance merely to report a
far-shell tensor.

The SO-bearing Fe PAW x/y/z dumps also pass native response/action
gates, but an initial SCF threshold of ``1e-8 Ry`` leaves a
first-shell repeated-row spread of ``4.117504e-6 eV``:
that three-leg output is **correctly refused**. Recomputing
all three PAW SCF densities with ``conv_thr=1e-12`` and their
full-BZ NSCF dumps reduces the first-shell spread to
``2.086799e-8 eV``, below the same ``5e-8 eV`` gate.
The validated nearest-neighbor-only PAW tensor gives
:math:`J_{\rm iso}=14.7775026` meV, offdiagonal symmetric
anisotropy :math:`J^{\rm ani}_{yz}=-0.006662556` meV
for :math:`R=(1,0,0)`, maximum first-shell DMI norm
``7.8915e-7 meV`` and pair reversal below ``4e-18 eV``.
Its 6-Å far shell still fails (first inconsistent row
``1.558e-5 eV``). Thus **both** US and tightly converged
PAW have a validated first-shell tensor, but neither has
a six-angstrom tensor from these reference densities.

A GPAW 26.7 spinor x reference (LDA/PW92, 8³ k mesh,
96 spinor bands, 1100 eV cutoff) yields a first-neighbor
x-leg transverse diagonal ``13.804504 meV`` at 30 poles
and 0.05 eV contour smearing; QE PBE US and the tightly
converged PAW x legs give ``17.717442`` and
``14.777512 meV`` respectively (12³, 48 bands, 80 Ry).
These **single legs** differ in XC, setup, basis, k mesh
and band window; no matched-functional quantitative
comparison follows. GPAW's noncollinear SCF plane-wave
path rejects PBE in this installation and lacks a matched
PZ Fe setup.

One spinor dump measures only a transverse leg; it cannot define a
rank-nine ``exchange.out``. Compute independent x/y/z SCF densities,
each followed by its own full-BZ NSCF with matching geometry, UPF,
k mesh and band window. ``--input_x --input_y --input_z`` requires
all references and reports the worst repeated-row spread;
``--leg_consistency_atol_eV`` defaults to ``5e-8`` for QE and
cannot exceed ``1e-7``. ``--overlap_mode`` and
``--overlap_rcond`` are invalid for dual KB coefficients.
See the six US input decks and official Fe UPF hash in
``examples/qe/bccfe_spinor``.

CLI usage
---------

``qe2J.py`` turns a dump into the standard TB2J results directory:

.. code-block:: bash

   qe2J.py --input nscf_dump.bin --output_path TB2J_results_fe --elements Fe --Rcut 6 --nz 30
   qe2J.py --input atomic_nscf.bin --output_path TB2J_results_atomic --elements Fe --Rcut 6 --overlap_mode inverse
   qe2J.py --input_x fe_x.bin --input_y fe_y.bin --input_z fe_z.bin --output_path TB2J_results_fe_spinor --elements Fe --Rcut 2.6

``--index_magnetic_atoms`` accepts 1-based atom indices instead of
``--elements``; ``--smearing`` is the contour smearing in eV (default
0.05). See ``examples/qe/feo/`` for reproducible FeO US and PAW KB
input decks and measured shell exchange.

Collinear KB family support
---------------------------

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
   * - NC (+ KB, e.g. ONCV; collinear only)
     - validated for collinear v1.2, refused for spinor NC1.1
     - Collinear bccFe (Fe_ONCV_PBE-1.0, 16 valence electrons) J1 = 15.22 meV.
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

* Atomic-wavefunction mode remains collinear; KB has a separate
  noncollinear magnetic US/PAW path whose SO-bearing family is accepted
  only after the native fixed-SOC covariance and action gates pass.
* ``npool = 1``: no plane-wave pools, and no gamma-only runs.
* The dump is gfortran sequential unformatted with 4-byte little-endian
  record markers; dumps are not portable across endianness or other
  compilers' unformatted ABI.
* The full Brillouin zone must be present in the nscf run
  (``nosym = .true., noinv = .true.``).
* ``becsum`` in KB nscf dumps carries scf-mesh restart occupations, not
  dense-mesh weights; occupation-based diagnostics must use scf dumps.
