VASP split-SOC exchange (patched native PAW)
============================================

The VASP split-SOC workflow reconstructs magnetic exchange from a *single*
collinear, spin-orbit-free **strength-zero** VASP run whose PAW projectors are
the native ``W%CPROJ`` augmentation maps.  It consumes **two** artifacts of
that one run, produced by a VASP build patched with the ``VASP_TB2J_patch``
story-010 dump hooks (branch ``story010-cso-provenance``):

1. a ``tb2j_native.bin`` **v5/v6 collinear native export** (complex
   ``W%CPROJ``, band eigenvalues/occupations, ``CDIJ`` spin difference),
   read by :func:`TB2J.interfaces.vasp_native.read_vasp_native`;
2. a ``tb2j_cso.bin`` **CSO dump** written by the patched Fortran module
   ``tb2j_cso.F``: the per-ion one-center spin-orbit operator ``CSO``, the
   augmentation occupations ``COCC`` and the spherical augmentation-sphere
   potential ``POTAE`` exactly as consumed by VASP's ``SPINORB_STRENGTH``,
   plus SAXIS Euler metadata.  It is read by
   :mod:`TB2J.interfaces.vasp_cso_dump`.

The production chain is therefore::

   patched collinear VASP strength-0 run (ISPIN=2, LSORBIT off, SAXIS axis)
        → tb2j_native.bin (v5/v6 CPROJ export)
        → tb2j_cso.bin (CSO + COCC + POTAE at the SAXIS Euler angles)
        → TB2J identity/pairing gates (native header, k-points, weights, COCC)
        → W^K_SO(k) = Σ_a B_a† CSO_a B_a   (all-atom, state-space)
        → three x/y/z legs (frame re-expressions, not re-runs)
        → lattice-frame rotation → three-leg merge

Only the collinear strength-0 branch is a production source: the consumer
refuses dumps with ``lsorbit=1`` or ``ncdij≠2`` (fully-relativistic runs are
comparison legs only), and the patched dump must be produced by ``vasp_ncl``
— ``vasp_std`` is not supported for CSO dumps.

Strength-zero recipe
--------------------

1. Build VASP from the ``VASP_TB2J_patch`` ``story010-cso-provenance``
   branch (patch commit ``0f9f073`` or newer; dump **v3** is the current
   format — reference-potential provenance included, see below; v2 dumps
   remain readable).
2. Run a **collinear** ground state with ``ISPIN=2``, no ``LSORBIT``, an
   explicit ``MAGMOM``, and ``SAXIS`` along the chosen polarisation axis.
   The retained FeO campaign runs one leg per axis — ``SAXIS = 1 0 0``,
   ``0 1 0``, ``0 0 1`` — but any single leg suffices: the consumer
   re-expresses the run in all three leg frames (see the frame section).
   The dump records the SAXIS Cartesian vector and its EULER angles
   :math:`(\alpha, \beta)`, so the spin frame of the operator is
   unambiguous.  The strength-zero reference may also be an ``ICHARG=11``
   frozen-potential run.
3. Pass the two dumps to the driver (below); converge the k-point set,
   ``Rcut``, ``nz`` and ``smearing``.

No licensed VASP files (POTCAR, WAVECAR, CHGCAR, …) are needed by, or
committed with, the TB2J example: the fixtures live outside the repository
and are identified by input SHA-256 lists kept with the campaign
(``feo/retained_inputs.sha256``).

``tb2j_cso.bin`` format contract
--------------------------------

The stream is little-endian raw Fortran (``ACCESS='STREAM'``,
``CONVERT='LITTLE_ENDIAN'``); a magic number (``20260927``) plus an
explicit version dispatch make accidental host-endian or format-drift
reads fail closed.  Versions 1–3 are supported; v1 is frozen upstream and
each newer version appends fields at unchanged offsets:

* **v2 — band/k provenance.**  After the header prefix the dump stores
  ``int32 ispin, nkpts, nbands, nb_tot``, ``f64 efermi``, and the writer
  identity of the companion native export, ``int32 native_magic
  (20260812), native_version (5 or 6)``; after the per-ion metadata it
  stores ``f64 vkpt(3, nkpts)`` (Fortran ``VKPT_PROV``, cartesian index
  fastest, IBZ form) and ``f64 wtkpt(nkpts)``.  These tie the dumped
  ``COCC`` to the same run's CPROJ export
  (``COCC = Σ_k w_k f_nk C* C`` per ``fast_aug.F``).  The loader
  validates the ``ispin`` value, rejects degenerate band/k provenance and
  requires the k weights to sum to :math:`1` within :math:`10^{-8}`.
* **v3 — reference-potential provenance (current).**  Appends the
  constants entering ``APOT(r)`` and
  ``xi(r) = invmc2 · dAPOT/dr`` in ``SPINORB_STRENGTH``
  (``felect``, ``invmc2 = 7.45596·10⁻⁶ Å²``, ``autoa``) and the per-type
  ``PP%POTAE_XCUPDATED`` radial reference potential ``potae_xcr``
  broadcast per ion.  v3 files are only written when the companion native
  export committed, so the recorded native identity is defined.  Landed
  in the patch binary at commit ``0f9f073``: the direct patch matches the
  installer files bit-identically on a pristine clone, and a native
  ``OPEN`` failure leaves no sidecar behind in a fresh run.  v2 remains a
  supported read path; v3 is what current patched builds write.
* **POTAE** carries the raw VASP storage convention (a factor
  :math:`2\sqrt{\pi}`; divide it out before building
  :math:`\xi(r)`); only ``potae[:nmax_ion[i], i]`` is valid — the file
  zero-pads every ion to ``nmax_max``.  For strength-zero ISPIN=2 runs
  the stored spherical potential is the spin **average**
  :math:`(\mathrm{POTAE}_{\uparrow} + \mathrm{POTAE}_{\downarrow})/2`,
  matching the total potential the stock noncollinear path consumes.
* **CSO** is returned as ``(nions, 4, lmdim, lmdim)`` complex blocks in
  the ``(uu, ud, du, dd)`` spinor representation, in the SAXIS spin
  frame.  The operator structure is Hermitian per spin channel
  (:math:`uu` and :math:`dd` Hermitian, :math:`du = ud^\dagger`).

The :math:`E_{soc}` identity oracle
-----------------------------------

The dump supports a direct numeric oracle against VASP itself: per ion,

.. math::

   E_{soc} = \mathrm{Re} \sum_{\text{same-}l} CSO \cdot \overline{COCC}

mirrors ``CALC_SPINORB_MATRIX_ELEMENTS`` (VASP ``relativistic.F``) with
the identical LM/LMP window accumulation, so it is directly comparable
with the ``Spin-Orbit-Coupling matrix elements`` block of OUTCAR.
Measured on the retained story-010 fixtures (identity check tolerance
:math:`10^{-6}` eV):

.. list-table:: :math:`E_{soc}` oracle (eV)
   :header-rows: 1
   :widths: 34 22 22 22

   * - leg
     - ion
     - dump
     - OUTCAR
   * - FeO ``lsorbit`` (full SCF SOC)
     - Fe
     - :math:`-0.0227451`
     - :math:`-0.0227451` (dev :math:`4.6·10⁻⁸`)
   * - FeO ``lsorbit``
     - O
     - :math:`-0.0001616`
     - :math:`-0.0001616` (dev :math:`4.6·10⁻⁸`)
   * - FeO ``lsorbit11`` (ICHARG=11)
     - Fe
     - :math:`-0.0080850`
     - :math:`-0.0080850` (dev :math:`3.6·10⁻⁸`)
   * - FeO ``lsorbit11``
     - O
     - :math:`-0.0001801`
     - :math:`-0.0001801` (dev :math:`1.9·10⁻⁸`)
   * - FeO collinear strength-0 (any SAXIS)
     - Fe, O
     - :math:`0` (exact, :math:`<10^{-20}`)
     — orbital quenching
   * - Ni collinear strength-0
     - Ni
     - :math:`0` (exact, :math:`<10^{-20}`)
     — orbital quenching
   * - Ni fully-relativistic leg (v7 native)
     - Ni
     - :math:`-0.08188974`
     - :math:`-0.0818897` (tol :math:`10^{-6}`)

The strength-zero legs certify the *quenching* half of the identity: a
collinear ``CSO`` dump must give exactly zero :math:`E_{soc}`, and does.
The fully-relativistic legs certify the *nonzero* half against OUTCAR.
Both halves run in the test suite and in the bundled example.

Pairing gates: native header, k-points, weights, occupations
------------------------------------------------------------

The consumer joins the two artifacts before any physics runs
(:func:`TB2J.interfaces.vasp_split_soc._check_consistency` and the COCC
reconstruction gate).  It refuses:

* a dump whose recorded ``ispin`` differs from the native export's spin
  count, whose ``nbands`` differs from the export band count, whose
  ``nkpts`` differs from the export IBZ k-point count (v2+ provenance),
  or whose ``efermi`` differs from the export Fermi level by more than
  :math:`10^{-8}` eV — these are run-identity checks, and a mismatch
  means the two files come from *different runs*;
* a dump whose ion count, per-ion projector counts (``lmmax``) or
  branch tags do not match the export, or whose header is internally
  inconsistent (degenerate dimensions, ``lps``/``lmmax`` contradictions,
  out-of-range ``nmax_ion``, trailing bytes);
* a pair that fails **COCC reconstruction**: the collinear augmentation
  occupations are rebuilt from the export's ``CPROJ`` as
  ``Σ_k w_k f_{nk} \overline{CPROJ(LP)} CPROJ(L)`` (``fast_aug`` order)
  and compared slot by slot; a residual at or above
  :math:`10^{-4}\times` the slot scale aborts the driver.  On the real
  FeO dump the observed residual is :math:`\le 1.5·10^{-5}\times` scale
  (largest on ``dd``) — the story-010 patch dumps ``COCC`` one
  occupation update away from the exported band occupations; a
  convention or weight error would sit at :math:`O(1)`;
* an empty magnetic pair set (degenerate ``Rcut``): the driver fails
  fast instead of writing empty results.

The k-point pairing is gauge-checked twice: the native v6 reader refuses
unsupported ``kpoint_storage_mode`` values, and the dump's ``vkpt`` is
stored in IBZ form to be compared against the export's IBZ count.  The
fully-relativistic v7 spinor export (``ibz_with_expansion_plan``) is a
separate comparison artifact, never a split-SOC input.

State-space SOC operator and the leg frames
-------------------------------------------

The band-basis operator follows the ``CALC_PAW_OVERLAP`` contraction,

.. math::

   W^{k}_{SO} = \sum_a B_a(k)^\dagger\, CSO_a\, B_a(k),

with :math:`B_a` the rectangular projector map of atom :math:`a` and the
``(uu, ud, du, dd)`` blocks of :math:`CSO_a`.  Both the collinear CPROJ
spin channels and ``CSO`` are SAXIS-frame objects, so the contraction is
frame-independent and yields the **state-space** matrix
:math:`\langle \psi_\nu | W_{SO} | \psi_{\nu'} \rangle` of shape
:math:`(n_k, 2n_{band}, 2n_{band})` (spin-major stacking, eV).  Each leg
:math:`d \in \{x, y, z\}` re-expresses the *whole* strength-zero problem
in the spin frame quantized along :math:`d` through the frame map

.. math::

   M = U_d^\dagger\, U_{SAXIS},

where :math:`U` are VASP ``SETUP_LS`` ``ROTMAT`` unitaries built from the
EULER angles.  The band spinor components and the magnetic vertices
(:math:`\Delta_a M \sigma_z M^\dagger` — the physical SAXIS-axis
splitting field in leg-frame components) are conjugated by :math:`M`;
``W_SO`` **enters unchanged in every leg** — it is never re-rotated,
because a state-space matrix needs no frame.  The kernel extracts the
exchange tensor in that leg frame and the output is rotated back with
:math:`T_{lattice} = O\, T_{leg}\, O^T`, :math:`O = SO(3)(U_d)`,
:math:`O\,e_z = d` (the story-001 corrected O map; ``det O = 1`` is
pinned in tests).  The site-magnetization signs are physical SAXIS-frame
quantities and are probed once from the un-conjugated vertex — the
leg-frame vertex's :math:`z`-trace vanishes for transverse legs and must
not be used as a sign probe.

Consequences measured on the real FeO fixture:

* **SAXIS covariance**: the strength-0 reference is spin-rotation
  invariant, so legs built from dumps recorded at ``SAXIS`` :math:`x`,
  :math:`y` or :math:`z` agree in the lattice frame (the ``lam=0``
  anchor is cross-SAXIS covariant by test);
* **leg consistency**: per-leg nearest-neighbour
  :math:`J_{iso}` agree across the :math:`x/y/z` legs to
  :math:`\sim 10^{-13}` meV (e.g. pair
  ``((−1,0,0), 0, 0)``: 8.035245150253187 / …3743 / …3736 meV).

Ligand SOC versus magnetic vertices
-----------------------------------

The two operators have deliberately different coverage:

* **W_SO is all-atom.  Ligand SOC enters the propagator.**  On FeO the
  oxygen one-center ``CSO`` enters :math:`W^{k}_{SO}` at the ~10% level
  of its scale (operator-norm shift 0.0101 eV against 0.098 eV), and it
  measurably shifts the symmetry-allowed :math:`J_{iso}` channel
  (observed :math:`2.0·10^{-7}` eV on :math:`J_{iso}\approx 7.4·10^{-3}`
  eV; numerical noise floor :math:`\sim 10^{-15}`).  For ligand studies
  the assembly function takes ``skip_atoms`` to zero selected ions'
  one-center operators.
* **The magnetic rotation vertices are site-local and magnetic-only.**
  ``--elements Fe`` / ``--index_magnetic_atoms`` select the sites
  carrying the collinear splitting
  :math:`\Delta_a = (D_{up} - D_{down})\,\sigma_z`; one of the two
  selectors is required whenever the cell has nonmagnetic ligands, so
  ligands are never silently treated as magnetic.  The vertex splitting
  is untouched by the ligand's :math:`W_{SO}` (same-axis collinear
  reduction to second order).

Comparison with the fully-relativistic (FR) leg
-----------------------------------------------

The retained campaign keeps a fully-relativistic LSORBIT FeO run (native
**v7 spinor export**) as the comparison leg.  At matched contour
parameters (``Rcut`` 10 Å, ``nz`` 60, smearing 0.05 eV), nearest-neighbour
Fe–Fe pair ``((−1,0,0), 0, 0)``:

.. list-table:: FeO nn :math:`J_{iso}` (meV)
   :header-rows: 1
   :widths: 60 40

   * - source
     - :math:`J_{iso}`
   * - FR (LSORBIT native, full-SCF SOC)
     - :math:`+5.138676`
   * - split (collinear bands + CSO :math:`W_{SO}`)
     - :math:`+8.035245`
   * - collinear anchor (same native, no SOC)
     - :math:`+8.0402`
   * - split at ``lam=0`` (calibration anchor)
     - :math:`+8.040157525` — reproduces the collinear extraction
   * - ``lam=1`` perturbative CSO correction
     - :math:`-4.912·10^{-3}` meV

The raw FR−split gap (2.897 meV, 35% of :math:`J`) is an **upper bound**
on the method difference: the FR leg's :math:`J_{iso}` flows through the
shared projector channel primitives that the rank-9 cutover reworked, so
this number must be re-compared with the cutover core before it is
quoted (Story 011).  What *is* certified today: the ``lam=0`` anchor reproduces
the stock collinear extraction, the :math:`lam=1` perturbative
correction is physically small, and the three legs are frame copies of
one another.

Gates, safeguards and the nonconverged window
---------------------------------------------

* **SOC-off anchor**: ``--lam 0`` replays the collinear limit; on the
  real FeO pair it matches the stock collinear native extraction to the
  printed precision (table above) and is cross-SAXIS covariant.
* **Run-identity + COCC gates**: always on (pairing section); they fail
  closed before any exchange is computed.
* **Gauge safeguards**: the dump/native join is IBZ-gauge checked
  (k-count + per-k arrays); unsupported native k-point storage modes are
  refused at read time; fully-relativistic dumps are refused as
  production inputs.
* **Band window (FR-050)**: the kernel metadata starts with
  ``convergence_study: null`` and the driver does **not** silently fill
  it.  The window-enlargement study
  (:func:`TB2J.split_soc_kernel.band_window_convergence_report`) is
  available as an explicit diagnostic and reports per-window
  :math:`J_{iso}`/DMI/Jani norms, relative changes and a ``converged``
  flag with ``converged_from``.  A ``converged: false`` (or a missing
  study) is an unresolved error bar that calls for a larger native band
  window / k mesh — never for a favourable error bar.  On the ABINIT
  siblings this study is mandatory; on the VASP side it is currently an
  opt-in diagnostic and its absence is visible in the provenance.
* **Merged DMI/Jani status**: the shared-core cutover has landed — the
  invalid common A-channel tensor is retired and
  ``TB2J.split_soc_kernel.merge_transverse_legs`` is the only merge, so
  merged DMI/Jani now come from the raw rank-nine solve of three
  transverse legs (no legacy ``io_merge`` stage anywhere in this
  workflow).  What remains open is *physical cross-validation*, not
  plumbing: until the Story-011 full-exchange comparison against the
  fully-relativistic leg lands, merged DMI/Jani from this workflow are
  raw-tensor decompositions, not certified predictions.  FeO is
  centrosymmetric (rocksalt, translation self-pairs
  inversion-symmetric), so its DMI is symmetry-forbidden regardless; the
  observed :math:`|D| \le 10^{-7}` meV nulls are symmetry-consistent and
  are **not** independent pass evidence.

Absolute scale: what is certified, what is only monitored
---------------------------------------------------------

The absolute :math:`J` scale of any projector-basis split-SOC workflow is
an empirical quantity.  The ABINIT NC path (pypao pseudo-atomic orbital
basis) is explicitly **monitor-only** here: its identities and anchors
are certified, but the absolute scale rides on the PAO basis choice and
monitor empirically (see :doc:`split_soc_abinit_nc`).  The
VASP path improves on this by using the **native PAW projector basis**
and by anchoring the scale twice — the ``lam=0`` anchor reproduces the
stock collinear native extraction, and the FR leg provides an external
monitor — but the certified statements are the *identities* (anchor,
leg covariance, :math:`E_{soc}` oracle), not the absolute FR−split
agreement, which is an upper bound until the Story-011 full-exchange
cross-validation lands.

Driver usage
------------

.. code-block:: python

   from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

   gen_exchange_vasp_split_soc(
       "tb2j_native.bin",         # collinear v5/v6 native export
       "tb2j_cso.bin",            # story-010 CSO dump from the same run
       output_path="TB2J_results_vasp_split_soc",
       magnetic_elements=("Fe",),     # or index_magnetic_atoms (1-based)
       rcut=10.0,
       nz=60,
       smearing_eV=0.05,
       lam=1.0,                   # 0.0 = SOC-off calibration anchor
       mode="second_variation",
       legs=((1.,0.,0.), (0.,1.,0.), (0.,0.,1.)),
   )

The same run from the command line (console script ``vasp_split_soc2J.py``,
registered by the VASP adapter branch; it becomes importable from the
SPLIT_SOC environment after the adapter merge — the example below detects
this and reports it):

.. code-block:: bash

   vasp_split_soc2J.py --native-input tb2j_native.bin \
       --cso-dump tb2j_cso.bin --elements Fe \
       --output_path TB2J_results_vasp_split_soc \
       --Rcut 10.0 --nz 60 --smearing 0.05 --lam 1.0 --legs xyz

.. list-table:: Selected options
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``--native-input`` / ``--cso-dump``
     - The two required artifacts of the same strength-0 run; joined by
       the run-identity and COCC gates before any physics.
   * - ``--elements`` / ``--index_magnetic_atoms``
     - Magnetic sites carrying the collinear vertices (1-based CLI
       indices).  One of the two is required when the cell has ligands;
       the all-atom :math:`W_{SO}` keeps the ligand SOC regardless.
   * - ``--lam``
     - Dimensionless SOC scaling.  ``1.0`` physical, ``0.0`` SOC-off
       calibration anchor.
   * - ``--mode``
     - ``second_variation`` (production) or ``first_order_insertion``.
   * - ``--legs``
     - Leg axes as a string of ``x/y/z`` (default ``xyz``).
   * - ``--Rcut``, ``--nz``, ``--smearing``, ``--output_path``
     - Exchange controls as in the collinear workflow; the driver fails
       fast if the chosen ``Rcut`` leaves no magnetic pair.

Outputs and provenance
----------------------

Each leg is written to ``leg_x/``, ``leg_y/``, ``leg_z/`` as a
noncollinear TB2J results directory with ``spinat`` along the leg axis
(signed by the SAXIS-frame magnetization probe), plus a merged
``output_path`` and ``split_soc_provenance.json``.  The provenance
records the schema (``tb2j.vasp_split_soc_provenance/1.0``), backend,
mode, ``lambda``, both input paths, the native export version, the SAXIS
frame (vector + EULER angles), the magnetic sites with species and sign
probes, the :math:`R` grid, ``nz``, smearing, and per-leg the leg
direction, the full :math:`O` map (``O e_z = leg axis``), the kernel
metadata (operator source, strength-zero reference, frame/conjugation
description, band window with its — possibly absent — convergence study)
and the merge record.  Every leg description embeds the frame map and the
``T_lattice = O T_leg O^T`` rotation used.

Runnable example
----------------

``examples/projector_green/vasp_feo_split_soc.py`` exercises the real
retained FeO v2 fixture pair (rocksalt FM FeO primitive,
:math:`a = 4.332` Å, PBE+U with :math:`U_{eff} = 5.0` eV on Fe :math:`d`,
7×7×7 Γ mesh, 32 bands/spin, one leg per SAXIS :math:`x/y/z`) — or the Ni
strength-0 pair — through the schema/loader smoke: dump format
dispatch, provenance fields, k-weight normalization, the exact
strength-0 :math:`E_{soc} = 0` oracle, CSO Hermiticity, the run-identity
pairing, the COCC reconstruction gate and the all-atom :math:`W_{SO}`
Hermiticity.  With ``--run-driver`` it additionally runs the three-leg
driver on the raw rank-nine core (``merge_transverse_legs``); merged
tensors are raw rank-nine reconstructions whose DMI/Jani decompositions
are **not** certified as physical predictions until the Story-011
full-exchange cross-validation lands.

.. code-block:: bash

   # loader/oracle smoke against the retained FeO v2 pair (default paths)
   python examples/projector_green/vasp_feo_split_soc.py

   # Ni strength-0 pair
   python examples/projector_green/vasp_feo_split_soc.py --fixture ni

   # full three-leg driver (heavy)
   python examples/projector_green/vasp_feo_split_soc.py \
       --native-input collinear_z/tb2j_native.bin \
       --cso-dump collinear_z/tb2j_cso.bin --run-driver

The fixture files are **not** committed (licensed VASP PAW data): point
``--native-input``/``--cso-dump`` or ``TB2J_FEO_SPLITSOC_DIR`` /
``TB2J_NI_SPLITSOC_DIR`` at a local retained-campaign copy.  The script
skips cleanly with the exact missing path when the fixture is absent, and
refuses with a merge pointer when ``TB2J.interfaces.vasp_split_soc`` is
not importable yet.

.. warning::

   Do **not** quote merged DMI/Jani values from this workflow as
   certified physical results: the raw rank-nine merge is consistent by
   its own gates, but the full-exchange cross-validation against the
   fully-relativistic leg (Story 011) is pending, and the FR comparison
   above is an upper bound.  :math:`J_{iso}` per leg, the anchors and the
   :math:`E_{soc}` oracle are the certified observables.
