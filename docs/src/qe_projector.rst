Quantum ESPRESSO projector-Green exchange
=========================================

TB2J computes magnetic exchange from Quantum ESPRESSO (QE) calculations
using separable KB beta channels (the default) or, for collinear runs,
UPF atomic-wavefunction channels (``TB2J_PROJECTORS=atomic``). The
current exporter profiles are collinear KB 1.3, collinear atomic 1.1,
and noncollinear KB NC1.2, including explicit native Hubbard provenance.
Complete Dudarev XC+U sources are supported by the readers and existing
API/CLI; format or source validation alone is not physical exchange
certification. SO-bearing US/PAW tensors require independent x/y/z
references and all existing source/action and tensor-consistency gates.
Fully relativistic NC and mixed NC/augmented production tensors remain
refused. Atomic mode remains collinear. An instrumented QE
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
  accepts NC, US and PAW; noncollinear KB writes NC, US, PAW and mixed
  site families after native action/source checks. The production
  spinor tensor path refuses mixed NC + US/PAW even if a dump passes
  per-site reader checks: the real Fe NC + O US control has an
  induced O source, not the required nonmagnetic ligand. Atomic mode
  requires ``PP_PSWFC`` atomic wavefunctions in every UPF and
  remains collinear.

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
   nbnd = ...         ! include the empty bands wanted in G(E)
   # after the namelists
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
``becsum``, v1.2-and-later ``dbeta_xc``/``ddd_paw``, per-k coefficients,
cell/species/ions metadata, ``et``/``wg`` and smearing parameters)
for inspection. NC requires v1.2 or later for a nonzero XC exchange vertex.

The normalized view is produced by
:py:func:`TB2J.interfaces.qe_projector.read_qe_dump`, which returns a validated
:class:`~TB2J.projector_green.ProjectorGreenData`:

.. code-block:: python

   from TB2J.interfaces.qe_projector import read_qe_dump

   data = read_qe_dump("nscf_dump.bin")

The loader applies the QE conventions:

* beta channels follow QE's **species-major** order: increasing species
  index, then QE ion order within that species. Atom ``na`` of species
  ``ityp[na]`` owns ``nh(ityp[na])`` channels starting at its native
  ``ofsbeta(na)``; this is not a cumulative offset over ion order;
* the coefficients are **already dual**, so ``overlap_k`` is ``None`` and no
  Gram/overlap dressing is applied anywhere downstream;
* For legacy v1.2 no-U dumps the XC exchange vertex
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

.. warning::

   Older TB2J readers assumed atom-major beta offsets and ownership. KB
   calculations whose atoms are not grouped in increasing species-index
   order can therefore have incorrect site-resolved exchange, in both
   collinear and spinor modes and with or without Hubbard U. Recompute
   TB2J results with the corrected reader from the unchanged native dumps;
   no format change or new DFT calculation is needed for this correction.
   Single-atom checks and agreement between two affected readers do not
   establish correct multi-species exchange.

The native coefficient rows retain their species-major order; site index
maps and owner metadata select the matching rows of each atom's operator.
Appended Hubbard channels retain their own native atom identities. Atomic
wavefunction dumps have a different, atom-major ordering and are unchanged.
Correct channel ownership does not waive native-action, source,
convergence or production tensor/family gates.

``qq_at`` and the Gram diagnostic are stored for diagnostics only; ``qq_at``
belongs to the wavefunction overlap operator
:math:`S_\psi=I+\beta q_{at}\beta^\dagger`, **not** to the projector channel,
and is never used as a channel metric.

Atomic-wavefunction dumps
-------------------------

Select ``TB2J_PROJECTORS=atomic`` alongside ``TB2J_DUMP`` for both runs.
QE constructs pseudo-atomic orbitals from each UPF's ``PP_PSWFC`` block.
These are full-cell Bloch sums represented in the plane-wave basis, not
all-electron atomic wavefunctions, PAW partial waves, or functions restricted
to an augmentation sphere. An UPF without atomic wavefunctions (including
the tested SG15 ``Fe_ONCV_PBE-1.0.upf``) is rejected. The atomic
``TB2JQEATWFC1.1`` format extends the legacy ``TB2JQEATWFC1.0`` payload,
which stores primal
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
KB exporter writes the separate ``TB2JQEDUMPNC1.2`` schema, extending
NC1.1 with explicit Hubbard status and source records. Legacy NC1.0 remains
scalar-relativistic-only. A spinor representation does not itself imply
SOC: ``noncolin=.true., lspinorb=.false.`` also uses two-component states.
NC1.1 and later record nine native XC magnetic-source responses for US/PAW
and their measured zeros for NC.
Each original Cartesian field component is placed into every Pauli
destination slot at frozen density and fixed ionic/SOC Hamiltonian.
PAW one-center magnetic source components rotate/zero in lockstep.
The exporter verifies native ``add_vuspsi`` action, magnetic
subtraction, diagonal reconstruction and source rotations/finite
differences. For NC it also checks the **projected smooth Pauli
field's native action and rotations on site-local beta-built
spinors**; mixed cells must not conflate onsite blocks with
nonzero intersite ``<beta_i|B|beta_j>`` terms. Bare-spin
conjugation of SO-bearing QE ``fcoef`` is informational, not
a tangent oracle.

``parse_qe_spinor_dump`` retains native ``becp%nc`` coefficients,
``deeq_nc_total`` diagnostic, separately field-dependent
``deeq_nc_magnetic``, Hermitian projected smooth field and, for NC1.1,
``D_response(nhm,nhm,nat,4,3,3)`` with axes
``(beta_i,beta_j,atom,(uu,ud,du,dd),destination_c,source_j)``.
``read_qe_spinor_dump`` returns ``ProjectorGreenData(nspinor=2)`` with
already-dual coefficients and a magnetic-only spinor operator.
Beta rows use the same species-major ordering as collinear KB dumps;
per-atom operator and response arrays remain indexed in QE ion order.
Only the projected smooth field uses a Hermitian beta-Gram
pseudoinverse (relative cutoff ``1e-6``). No overlap inverse
acts on dual Green coefficients or native ``D_response``. A
first-k numerical eigengap makes the inversion numerically
defined but **does not** bound lost normalized-state weight:
the PseudoDojo FR Fe pilot retains 24 of 34 modes and reaches
``max|M^+ A M^+| = 4535 Ry``. Even an SOC-off run with that
same FR UPF retains separate j-split radials after QE clears
``has_so``; TB2J refuses production exchange when an NC
Gram is truncated, with or without the SOC flag.

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

For a genuine SO-bearing NC site, the fixed ionic/SOC operator
is excluded from both the splitting and its tangent. Pure NC
has exactly measured-zero nonlocal magnetic and response blocks;
the smooth-only tangent reduces to the Pauli commutator. The
optional ``TB2J_NC_GRAM_DIAG`` file (campaign schema
``TB2JNCGRAMDIAG1.1``) records each k/site complex beta Gram
eigenvalue and the **normalized** band-mode weights
:math:`\sum_s|u_m(k)^\dagger\langle\beta(k)|\psi_{nks}\rangle|^2/
\lambda_m(k)`. ``W_OCC_NORM`` includes QE's ``wg`` (k weight and
occupation); ``W_WIN_NORM`` sums the declared
:math:`|E-E_F|\leq0.5` eV band window without k weight. Weight
from discarded modes must be evaluated across the full BZ and
the exchange must be recomputed across conditioning thresholds;
neither the first-k Gram record nor a rank-nine merge alone
certifies FR-NC physics. The sidecar is separate from NC1.1:
its filename must be new and distinct from ``TB2J_DUMP``.

For the official PseudoDojo FR Fe NC UPF (16 valence electrons,
``has_so=T``, SHA-256
``b9b7671a2c667a6f87a7228d30fcc205e339b69f5568fe3f3dba567fdce59a19``),
a real **2³ pilot** has ten discarded Gram modes. They
carry 20.19% of the beta-projected occupied weight and 42.80%
of the stated Fermi-window weight over that pilot's entire
mesh. The 2³ pilot alone cannot establish a 12³ error bound.

Independent x/y/z **12³ full-BZ** SCFs and 48-band NSCFs
for this same UPF provide a stronger negative gate: all
1728 k points in each reference discard ten of 34 beta
modes at ``rcond=1e-6``, carrying 20.03% of the
beta-projected occupied weight and 43.14% within
:math:`|E-E_F|\leq0.5` eV. At ``1e-5`` and ``1e-3``
13 and 16 modes are discarded, carrying 24.22% and
32.42% of occupied weight. Research-only x-leg
:math:`J_{yy}(R=(1,0,0))` is 9.6503205, 9.6675615
and 9.6776549 meV at these three cutoffs. Across all
independent x/y/z legs, six first-neighbor translations
and components, the largest full-bracket raw shift is
0.0273347 meV. This does **not** bound the error from
discarded normalized states. The real three-reference
``qe2J.py`` invocation refused before rank-nine output
and created no tensor directory.

The below-gap control ``rcond=1e-10`` retains all 34 modes
at every k. Its untruncated smooth operator reaches
``3.4746e8 eV`` rather than ``8.6952e4 eV`` at
``1e-6``; x/y/z fail the existing Hermiticity gate
with absolute defects ``13.7–16.9 eV``. There is no
valid below-gap raw exchange component to compare.
The ``1e-6``–``1e-3`` bracket is restricted to
truncated-mode diagnostics; symmetrizing the failed
below-gap operator would mask the numerical failure
and would not recover the substantial lost weight.

A matched **same-FR-UPF SOC-off control** uses independent
x/y/z 12³ SCFs and 48-band full-BZ NSCFs with the same
80/320 Ry cutoffs, smearing and contour. QE's scalar
``dion`` weighting clears ``has_so`` without merging
the 34 j-split beta channels. At ``rcond=1e-6``,
all three SOC-off legs still discard ten modes at every
k, carrying 20.0307% occupied and 43.1286% Fermi-window
beta-projected normalized weight. The actual three-leg
SOC-off CLI refuses without output. At
:math:`R=(1,0,0)`, the research-only symmetric
transverse raw-leg off-diagonals (x:yz, y:xz, z:xy)
are ``0.0001784051``, ``0.0001779155``,
``0.0001776593`` meV with SOC on, versus
``+2.539e-9``, ``-1.777e-9``, ``+0.904e-9``
meV with SOC off. Neither side passes the physical
mode-weight gate; these are **not** certified
rank-nine anisotropy or a quantitative SOC-only shift.

``qe2J.py`` refuses a tensor while the
FR-NC conditioning error budget is unestablished; use
``read_qe_spinor_dump`` and the raw single-leg API for
research diagnostics only. Scalar ONCV Fe NC spinor dumps pass
the real source/action/rotation checks. With matched same-UPF
12³ full-BZ data (48 total spinor bands versus 24 bands per
collinear spin, 30 CFR poles and 0.05 eV contour smearing),
the nearest-neighbor z transverse leg gives
:math:`J_{xx}=J_{yy}=14.7910404` meV versus collinear v1.2
:math:`14.7932825` meV: maximum difference :math:`0.0023220`
meV over all six nearest-neighbor vectors, below the
:math:`0.01` meV G3 gate. The earlier 24-spinor-band smoke
was *not* a matched-band comparison (0.300 meV difference).

The following NC1.1 exchange benchmarks are historical **no-U** results;
they do not certify a new active-U source or tensor.

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

An independent GPAW 26.7 LDA/PW92 plane-wave spinor
x/y/z reference family was also calculated (8³ k mesh,
96 spinor bands, 1100 eV cutoff). Its initial three-leg
nearest-shell merge fails the GPAW default repeated-row
bound ``1e-8 eV`` (first inconsistent row
``1.592e-7 eV``). Tightening all three GPAW SCFs to
energy/density thresholds ``1e-8`` still fails (first
row ``4.751e-8 eV``; measured **worst** first-shell
spread ``5.241305e-6 eV`` and transverse-mask
residual ``1.750664e-6 eV``). The GPAW rank-nine
output is therefore **refused**, not compared as a
certified tensor with the QE US/PAW outputs.
At 30 poles and 0.05 eV contour smearing, the tight
GPAW **single x leg** has first-neighbor transverse
diagonal ``13.803708 meV``; QE PBE US and tight PAW x
legs give ``17.717442`` and ``14.777512 meV``
respectively (12³, 48 bands, 80 Ry). These diagnostic
legs differ in XC, setup, beta basis, k mesh and band
window; no matched-functional quantitative inference
follows. GPAW's noncollinear SCF plane-wave path in
this installation rejects PBE and lacks a matched
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
     - collinear v1.2 and scalar NC1.1 z transverse leg validated; FR-NC tensor refused
     - Collinear bccFe (Fe_ONCV_PBE-1.0, 16 valence electrons)
       :math:`J_1=14.7932825` meV versus matched NC1.1 z
       :math:`J_{xx}=J_{yy}=14.7910404` meV at 12³/48-spinor bands
       (absolute difference 0.0023220 meV). The v1.2
       ``dbeta_xc`` vertex removes the historical ``deeq``-only
       :math:`\Delta\equiv0` blocker. FR-NC still requires
       per-k physical conditioning evidence. NC ``becsum`` is
       not an NSCF occupation reference.
   * - Mixed NC + US/PAW (spinor)
     - production refused
     - A real 2³ Fe NC + O US dump passes per-site writer/reader
       oracles, but the O ligand is magnetic (nonlocal source
       ``DMAX=0.1074555 Ry``). Two compensated-Fe trial cells
       did not converge to a validated zero-source US ligand.
       The public three-leg CLI rejects this family until that
       physical gate passes; research reading is not a tensor
       certification.

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

* Atomic-wavefunction mode remains collinear. KB has NC, US and PAW
  noncollinear exporter/reader paths; only the previously validated
  US/PAW SO-bearing families currently publish rank-nine tensors.
* ``npool = 1``: no plane-wave pools, and no gamma-only runs.
* The dump is gfortran sequential unformatted with 4-byte little-endian
  record markers; dumps are not portable across endianness or other
  compilers' unformatted ABI.
* The full Brillouin zone must be present in the nscf run
  (``nosym = .true., noinv = .true.``).
* Legacy ``becsum`` in KB nscf dumps carries scf-mesh restart occupations,
  not dense-mesh weights. Modern Hubbard records separately retain frozen
  SCF occupations and fresh NSCF contraction checks; these are not equated.

DFT+U usage and source contracts
----------------------------------------

Two independent projector choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QE ``HUBBARD`` card defines the correlated subspace and native
Hamiltonian. For example, the measured bcc-Fe PBE case uses:

.. code-block:: text

   HUBBARD (ortho-atomic)
   U Fe-3d 4.0

``U`` is in eV and ``Fe`` must match the species label. ``HUBBARD (atomic)``
and ``HUBBARD (ortho-atomic)`` are admitted for scalar Dudarev U; the latter
uses QE's native full-cell orthogonalization, including the full-spinor
construction for noncollinear input. The actual S-applied QE ``wfcU``
columns and acted potential are exported. PAW does not change this choice
into PAW augmentation partial waves or a sphere-only Hubbard construction.

Separately, ``TB2J_PROJECTORS=kb`` (default) or ``atomic`` selects the
**exchange representation** for the predecessor XC source. It does not
change QE's Hubbard subspace, remove U, or select a second Hubbard model.
In atomic exchange mode, the original XC rows use the full per-k atomic
Gram once; appended native Hubbard rows are not filtered again.

.. list-table:: Native producer/source coverage, not tensor certification
   :header-rows: 1
   :widths: 30 24 46

   * - State
     - Exchange mode
     - Native Hubbard
   * - Collinear NC, US, PAW
     - KB or atomic
     - Atomic / ortho-atomic

       Main U; US backgrounds/backall
   * - Scalar-relativistic

       spinor NC, US, PAW
     - KB
     - Atomic / ortho-atomic

       Main U; native tangents
   * - SOC spinor US, PAW
     - KB
     - Atomic / ortho-atomic

       Main U; native tangents
   * - Spinor
     - Atomic: unsupported
     - Refused

The admitted branch is scalar Dudarev ``lda_plus_u_kind=0`` with
``atomic`` or ``ortho-atomic`` projectors. Other projector choices,
including ``norm-atomic`` and ``pseudo``, are not enabled by this interface.
Nonzero J/J0, alpha/alpha_back/beta, orbital-resolved parameters, Hubbard V
and other Hubbard functionals are excluded. Collinear main/background
manifolds are retained as actually acted native blocks; spinor background U
is unsupported and refused. Native scalar-U shell angular momenta are
limited to l=0..3. These restrictions are narrower than QE's general
DFT+U capabilities.

For a no-U or U=0 control, **omit the entire HUBBARD card**. The pinned QE
parser rejects an all-zero card; an epsilon U is not a no-U control.
Retain the converged SCF parent, full-BZ final NSCF input/log, executable,
UPF hashes and export manifest. An NSCF export is not itself an SCF
convergence certificate. For matched collinear/spinor comparisons,
``nbnd`` counts bands per spin in the collinear case and total spinor
bands in the noncollinear case.

Versioned source contract
~~~~~~~~~~~~~~~~~~~~~~~~~

The new executable derivations ``docs/sympy/qe_hubbard_native_contract.py``
and ``docs/sympy/qe_hubbard_joint_frame.py`` have paired reports in the same
directory. They pin native Hubbard column/action conventions, frozen-source
rotations, the redundant joint XC+U trace with all mixed contractions, and
one-time filtering of the original atomic exchange frame. Run them with the
``mydev`` environment. They are algebraic checks, not native action or
exchange certification.

The typed native extension profiles add versioned Hubbard provenance to the
existing QE raw readers:

* KB ``TB2JQEDUMPV1.3`` extends the preceding collinear KB payload.
* Atomic ``TB2JQEATWFC1.1`` extends the preceding collinear pseudo-atomic
  payload.
* Spinor ``TB2JQEDUMPNC1.2`` extends the preceding NC KB payload.

The existing ``parse_qe_dump``, ``parse_qe_atomic_dump`` and
``parse_qe_spinor_dump`` interfaces retain their legacy versions and read
these versions as well. Each returned raw record object exposes
``.hubbard``. Its ``state`` distinguishes ``"active"``, explicit
``"absent"`` in a new-format extension, and ``"unknown"`` for legacy
payloads that contain no reliable Hubbard-status evidence. New-format
``absent`` is explicit no-U provenance; it requires no legacy assertion and
retains the inherited raw records. ``unknown`` is not evidence of no-U.

For active extensions, raw parsing strictly validates the typed records,
dimensions, finite and Hermitian matrix data, shell/site/column maps, exact
stream termination and provenance. The raw ``.hubbard`` data retains the
native parameter inventory, maps, occupation and potential matrices,
per-k contractions, validation/action evidence and source provenance for
inspection; it is not reduced to a summary or silently discarded. The
``complete`` flag is false for legacy ``unknown`` state and true only after
strict validation of a complete new-format extension, whether active or
explicitly absent. Malformed, truncated, inconsistent or unsupported active
records are refused rather than interpreted as zero-U. Raw validation and
the normalized joint-frame construction below are separate checks; neither
alone certifies a physical exchange result.

Extension species labels must match predecessor species identities. Fresh
occupation evidence is cross-checked against the per-k native contractions
through species/shell-kind total traces, which survive native orbital/spin
rotations and equivalent-site permutations. This does not equate fresh NSCF
occupations with the separately retained frozen SCF state or replay omitted
symmetry matrices. The KB carrier scale is reconstructed from populated
``deeq`` blocks before applying the native U-free-carrier tolerance. Spinor
NC1.2 retains only the complex carrier bank: it supplies a lower bound on
QE's combined real/complex scale, not an upper bound or full reconstruction.
Atomic predecessors retain no carrier bank, so their source difference and
scale remain native measurements. ``hubbard.provenance["carrier_scale_evidence"]``
records these distinct evidence limits; none substitutes band energies for
source validation.

The normalized ``read_qe_dump``, ``read_qe_atomic_dump`` and
``read_qe_spinor_dump`` interfaces accept the keyword-only
``legacy_no_u=True`` assertion for a legacy ``unknown`` record. This is a
user attestation, not measured provenance, and normalized metadata labels it
``hubbard_state="user_attested_no_u"``. Legacy records remain available for
raw inspection without the assertion, but production normalization requires
it. The assertion cannot override declared U, missing or invalid source
evidence, unsupported branches, or failed source/tangent validation. New
explicitly absent records need no assertion and are labeled
``hubbard_state="absent"``.

Validated active records normalize to one joint separable XC+U frame. Native
``wfcU`` contractions are already S-applied and natively orthogonalized where
requested; neither S nor a second spin expansion is applied. KB coefficients
stay unfiltered. For active UPF-atomic input, ``read_qe_atomic_dump`` accepts
``overlap_mode`` and ``overlap_rcond`` and applies the existing full per-k
atomic filter exactly once to the original XC rows before appending U rows.
The resulting ``overlap_k`` is ``None``: the retained ``original_overlap_k``
is provenance only, not a metric used by the runtime. Effective overlap
settings and the source inventory remain in metadata.

``operator_components`` contains complete ``delta_total`` plus the
``delta_xc`` and ``delta_u`` source blocks for diagnostics. Site-local matrices
use ``site_projector_indices`` to address global coefficient columns, including
noncontiguous appended Hubbard channels. ``qe_channel_groups`` identifies the
original XC and native U global columns. The single spectral Green function
retains all XC--U cross blocks; total exchange is not the sum of independently
computed XC-only and U-only exchanges. No joint Gram is inverted or truncated,
even when source channels are redundant.

New spinor records retain version1 ``qe_source_frame`` metadata with definition
``qe_original_xc_or_mapped_native_u_v1``. Its ``site_map`` contains magnetic
data-site indices; unit ``directions`` and ``direction_provenance`` align with
that map. Nonmagnetic origin0 sites have no invented direction. Original XC
fixes the signed reference; only zero XC may use the validated mapped native
Hubbard-occupation fallback. Total and diagnostic sources share that frame,
including oppositely signed or canceled traces. The matching explicit
``qe_native_xc_u_frozen_source_tangent_v1`` vertices are physical frozen-source
Hamiltonian derivatives in eV. Their consumer bypasses operator-trace
direction inference and commutator construction. Unrelated older producer
contracts retain their existing path. Missing or unresolved frames fail before
exchange output. Frame metadata and the original atomic Gram round-trip in
the existing projector-Green NetCDF container.

Complete sources and explicit diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The existing readers and ``gen_exchange_qe`` accept
``operator_component=None`` or ``"delta_total"`` for complete exchange.
For an active-U record, ``"delta_xc"`` and ``"delta_u"`` select diagnostic
sources, not complete XC+U results. Reader selection changes the default
collinear ``hij`` spin difference or the spinor operator and matching
explicit tangent; it does not filter the joint coefficients again or change
``qe_source_frame``. Unavailable components are refused rather than replaced
with a fabricated zero source.
The default ``ProjectorGreen`` collinear trace and exchange consumers honor
this declared selection while retaining the full component inventory.
Older data without a declared selection keeps its historical XC-first default.

.. code-block:: python

   from TB2J.interfaces.qe_projector import gen_exchange_qe, read_qe_spinor_dump

   gen_exchange_qe("native-u.bin")   # complete delta_total
   gen_exchange_qe("native-u.bin", operator_component="delta_xc",
                   output_path="xc_diagnostic")
   u_leg_data = read_qe_spinor_dump("native-u-z.bin",
                                  operator_component="delta_u")

The same options extend the existing CLI; no separate U-only runtime or
command is needed:

.. code-block:: bash

   qe2J.py --input native-u.bin --operator_component delta_u \
           --output_path u_diagnostic
   qe2J.py --input_x native-u-x.bin --input_y native-u-y.bin \
           --input_z native-u-z.bin --operator_component delta_total
   qe2J.py --input historical-no-u.bin --legacy_no_u

``legacy_no_u=True`` / ``--legacy_no_u`` is only a user assertion for a
legacy Hubbard-unknown input. Passing it on native active-U input does not
remove U or relabel the input as absent; malformed/incomplete source,
unsupported family and tangent failures remain refusals.

Before standard tensor files are written, x/y/z inputs must agree in
functional, U/DC/projector parameters, shell and native-column maps, UPF
digest/family identity, selected component and coefficient/frame/tangent
conventions. Byte-identical UPFs may reside at different paths. New-format
references with an unresolved ``NOT SET`` UPF digest remain inspectable but
cannot establish production reference identity. Independently
converged SOC occupations and actual x/y/z direction vectors are not required
to be identical. A valid declared frame, not the selected operator trace,
controls magnetic-site selection and output moment signs, including zero or
canceling diagnostics. Existing family, rank-nine, conditioning, repeated-row
and pair-reversal gates remain in force.
Output moments oppose ``original_xc_pauli_trace`` directions but follow
``mapped_native_hubbard_occupation`` directions, which already represent
physical occupation moments. This per-site rule also applies to mixed-origin
frames and zero diagnostic sources.
For NC1.2 scalar-relativistic inputs, the explicit-versus-commutator
parity check also uses that declared frame, not a second inferred direction.
The inherited NC1.1 frame behavior and all parity tolerances are unchanged.

Native finite-difference metrics describe the retained native arrays. The
reader verifies those metrics against those arrays and independently
reconstructs the derivative. Both retained and reconstructed derivatives
must meet the existing relative or analytic-zero absolute budgets; native
roundoff is not permission to bypass a failed physical tangent gate.

The standard output description includes source provenance:
``operator_component``, ``source_completeness``, native Hubbard state and
definition versus user attestation, coefficient groups, explicit spinor frame
and effective original-atomic overlap controls. Per-component validity is
distinct from physical completeness: a valid U-only block is still labeled
``diagnostic_u_only``. Active atomic output identifies the one-time XC filter
and unchanged native U rows even though the final runtime overlap is absent.


This source-normalization contract does not certify +U exchange tensors,
physical/material convergence or atomic-wavefunction J accuracy. Existing
fully relativistic NC and mixed NC/augmented production restrictions remain.
API/CLI source integration does not waive independent scientific validation
or any tensor gate. U=0 native controls omit the ``HUBBARD`` card. Executable
``docs/sympy/qe_hubbard_native_contract.py``,
``docs/sympy/qe_hubbard_joint_frame.py`` and
``docs/sympy/qe_hubbard_cartesian_tangents.py`` have paired reports; symbolic
checks do not replace native action, rotation or production certification.
