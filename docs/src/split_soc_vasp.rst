VASP split-SOC exchange (patched native PAW)
============================================

The VASP split-SOC workflow reconstructs magnetic exchange from collinear,
spin-orbit-free **strength-zero** VASP runs whose PAW projectors are the
native ``W%CPROJ`` augmentation maps.  **Each x/y/z leg is one independent
strength-0 run** quantized along its own ``SAXIS`` axis; the rank-nine
merge needs all three SAXIS references, because a single run determines
only the transverse plane of its own spin frame.  Each run produces **two**
artifacts, via a VASP build patched with the ``VASP_TB2J_patch``
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

   per leg d ∈ {x, y, z}:
     patched collinear VASP strength-0 run (ISPIN=2, LSORBIT off, SAXIS = d)
          → tb2j_native.bin (v5/v6 CPROJ export)  +  tb2j_cso.bin (CSO + COCC + POTAE)
          → per-leg identity/pairing gates (native header, k-points, weights, COCC)
          → W^K_SO,d(k) = Σ_a B_a† CSO_a B_a   (all-atom, state-space, psi gauge)
          → measured transverse block of leg d  →  split_soc_leg.npz + provenance
   lattice-frame rotation of the three legs → rank-nine three-leg merge

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
2. Run **one collinear ground state per leg** — three independent runs —
   with ``ISPIN=2``, no ``LSORBIT``, an explicit ``MAGMOM``, and ``SAXIS``
   along that leg's polarisation axis: ``SAXIS = 1 0 0``, ``0 1 0``,
   ``0 0 1`` for the x/y/z legs of the retained FeO campaign.  All three
   runs share the same frozen physics; only the spin quantization axis
   differs.  One run alone determines only its own transverse plane — the
   rank-nine merge refuses anything less than the three SAXIS references.
   Each dump records the run's SAXIS Cartesian vector and its EULER angles
   :math:`(\alpha, \beta)`, so the spin frame of the operator is
   unambiguous.  The strength-zero reference may also be an ``ICHARG=11``
   frozen-potential run.
3. Pass the three leg artifact directories to the driver (below); converge
   the k-point set, ``Rcut``, ``nz`` and ``smearing``.

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
     - orbital quenching (expected)
   * - Ni collinear strength-0
     - Ni
     - :math:`0` (exact, :math:`<10^{-20}`)
     - orbital quenching (expected)
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

For every leg the consumer joins that leg's two artifacts before any
physics runs
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

Each leg is one independent strength-0 run, kept natively in its own
psi-gauge ``SAXIS`` frame (``leg_artifacts`` x/y/z).  The band-basis
operator of that run follows the ``CALC_PAW_OVERLAP`` contraction,

.. math::

   W^{k}_{SO,d} = \sum_a B_{a,d}(k)^\dagger\, CSO_{a,d}\, B_{a,d}(k),

with :math:`B_{a,d}` the rectangular projector map of atom :math:`a` in
leg :math:`d`'s export and the ``(uu, ud, du, dd)`` blocks of that run's
:math:`CSO_{a,d}`.  Both the collinear CPROJ spin channels and ``CSO``
are SAXIS-frame objects of the *same* run, so the contraction yields the
**state-space** matrix
:math:`\langle \psi^d_\nu | W_{SO,d} | \psi^d_{\nu'} \rangle` of shape
:math:`(n_k, 2n_{band}, 2n_{band})` (spin-major stacking, eV), and the
magnetic rotation vertices are the run's own collinear splittings
:math:`\Delta_{a,d}\,\sigma_z` in the same psi gauge.  The kernel
extracts the exchange tensor in that leg frame and the leg's measured
transverse block is rotated to the lattice with
:math:`T_{lattice} = O_d\, T_{leg,d}\, O_d^T`, :math:`O_d = SO(3)(U_d)`,
:math:`O_d\,e_z = d` (the story-001 corrected O map; ``det O = 1`` is
pinned in tests).  Per-leg tensors are stored as raw
``split_soc_leg.npz`` blocks with schema-2.0 provenance; the full lattice
tensor exists only after the rank-nine merge of the three legs — one run
determines only its own transverse plane, never the full tensor.

Consequences measured on the real FeO fixture:

* **SAXIS covariance**: the strength-0 reference is spin-rotation
  invariant, so the three independently run ``SAXIS`` legs agree in the
  lattice frame — the merged repeated-diagonal rows (each diagonal is
  measured in two different legs) close to :math:`2.9\times10^{-5}` eV
  against the driver's consistency gate;
* **rank-nine gate**: the merged FeO design matrix has rank 9 on every
  pair, i.e. the three SAXIS references are genuinely independent;
* **anchor**: the merged nearest-neighbour :math:`J_{iso}` is 7.3836 meV
  against the SOC-off extraction's 7.3856 meV on the same retained data.

Ligand SOC versus magnetic vertices
-----------------------------------

.. _vasp-ligand-vertex:

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
Fe–Fe pair:

.. list-table:: FeO nn :math:`J_{iso}` (meV)
   :header-rows: 1
   :widths: 60 40

   * - source
     - :math:`J_{iso}`
   * - FR (LSORBIT native, full-SCF SOC)
     - :math:`+5.138676`
   * - split, merged rank-nine (three SAXIS runs, ``lam=1``)
     - :math:`+7.3836`
   * - SOC-off anchor (same native, no SOC)
     - :math:`+7.3856`
   * - merged repeated-diagonal invariance (rank-9 gate)
     - spread :math:`2.9\times10^{-5}` eV

The raw FR−split gap is an **upper bound** on the method difference: the
FR leg's :math:`J_{iso}` flows through the shared projector channel
primitives that the rank-9 cutover reworked, so the FR number above must
be re-produced on the cutover core before it is quoted (story-013
cross-code gate).  What *is* certified today: the merged rank-nine
tensor is internally consistent (repeated-row gate), the SOC-off anchor
reproduces the stock collinear extraction, and the three legs are
independent measurements that agree within the merge tolerance.

Gates, safeguards and the nonconverged window
---------------------------------------------

* **SOC-off anchor**: ``--lam 0`` replays the collinear limit; on the
  retained FeO legs the merged nearest-neighbour :math:`J_{iso}` is
  7.3836 meV against the SOC-off extraction's 7.3856 meV (table above),
  with the merged repeated-diagonal rows closing to
  :math:`2.9\times10^{-5}` eV.
* **Run-identity + COCC gates**: always on (pairing section); they fail
  closed before any exchange is computed.
* **Gauge safeguards**: the dump/native join is IBZ-gauge checked
  (k-count + per-k arrays); unsupported native k-point storage modes are
  refused at read time; fully-relativistic dumps are refused as
  production inputs.
* **Band window (FR-050)**: the driver runs the window-enlargement study
  (:func:`TB2J.split_soc_kernel.band_window_convergence_report`) by
  default and records per-window :math:`J_{iso}`/DMI/Jani norms, relative
  changes and a ``converged`` flag with ``converged_from`` in each leg's
  provenance; ``--no-band-window-study`` skips it, and a skipped study is
  visible as an absent entry.  A ``converged: false`` (or a missing
  study) is an unresolved error bar that calls for a larger native band
  window / k mesh — never for a favourable error bar.  On the ABINIT
  siblings this study is mandatory as well.
* **Merged DMI/Jani status**: the shared-core cutover has landed — the
  invalid common A-channel tensor is retired and
  ``TB2J.split_soc_kernel.merge_transverse_legs`` is the only merge, so
  merged DMI/Jani now come from the raw rank-nine solve of three
  transverse legs (no legacy ``io_merge`` stage anywhere in this
  workflow).  What remains open is *physical cross-validation*, not
  plumbing: until the FR full-exchange comparison against the
  fully-relativistic leg lands (story-013 cross-code gate), merged
  DMI/Jani from this workflow are raw-tensor decompositions, not
  certified predictions.  FeO is
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
agreement, which is an upper bound until that FR full-exchange
cross-validation lands.

Driver usage
------------

The driver consumes the three leg artifact directories, one per SAXIS run
(console script ``vasp_split_soc2J.py`` from the migrated VASP adapter):

.. code-block:: bash

   vasp_split_soc2J.py --leg x=collinear_x --leg y=collinear_y --leg z=collinear_z \
       --output_path TB2J_results_vasp_split_soc \
       --Rcut 10.0 --nz 60 --smearing 0.05 --lam 1.0

.. list-table:: Selected options
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Meaning
   * - ``--leg <axis>=<RUN_DIR>``
     - One strength-0 run directory (native export + CSO dump) per
       ``x``/``y``/``z``; give all three — the rank-nine merge needs the
       three SAXIS references.  Each leg's pair is joined by the
       run-identity and COCC gates before any physics.
   * - ``--lam``
     - Dimensionless SOC scaling.  ``1.0`` physical, ``0.0`` SOC-off
       calibration anchor.
   * - ``--mode``
     - ``second_variation`` (production) or ``first_order_insertion``.
   * - ``--merge_consistency_atol``
     - Repeated-row invariance tolerance of the rank-nine merge in eV;
       the merge fails closed when the twice-measured diagonals of the
       three legs disagree beyond it.
   * - ``--no-band-window-study``
     - Skip the explicit window-enlargement diagnostic (not
       recommended; see the gates section).
   * - ``--Rcut``, ``--nz``, ``--smearing``, ``--output_path``
     - Exchange controls as in the collinear workflow; the driver fails
       fast if the chosen ``Rcut`` leaves no magnetic pair.

Magnetic-site selection for the collinear vertices follows the
``--elements`` / ``--index_magnetic_atoms`` rules of the adapter
interface (one of the two is required when the cell has nonmagnetic
ligands; see :ref:`the ligand/vertex split below <vasp-ligand-vertex>`).

Outputs and provenance
----------------------

Each leg writes the raw rotated transverse ``J_leg`` blocks to
``leg_x/split_soc_leg.npz`` (``leg_y``, ``leg_z`` likewise) plus its
provenance JSON; the rank-nine merged results are written to
``output_path`` with the merged ``split_soc_provenance.json``
(schema **2.0**), which records the backend, mode, ``lambda``, the three
leg input directories, the native export version, per-leg the SAXIS
frame (vector + EULER angles) and the full :math:`O_d` map
(``O_d e_z = d``), the kernel metadata (operator source, strength-zero
reference, band window with its — possibly absent — convergence study),
the magnetic sites with species and sign probes, and the
``raw_rank_nine`` merge diagnostics.  The merged provenance keeps the
three leg records as distinct entries.

Runnable gates
--------------

The retained FeO campaign fixtures (rocksalt FM FeO primitive,
:math:`a = 4.332` Å, PBE+U with :math:`U_{eff} = 5.0` eV on Fe :math:`d`,
7×7×7 Γ mesh, 32 bands/spin, one run per SAXIS :math:`x/y/z`) and the
Ni strength-0 pair back the env-gated test suite on the adapter branch:
dump format dispatch, provenance fields, k-weight normalization, the
exact strength-0 :math:`E_{soc} = 0` oracle, CSO Hermiticity, the
run-identity pairing, the COCC reconstruction gate, the all-atom
:math:`W_{SO}` Hermiticity, and — on the real FeO three-run set — the
merged nn :math:`J_{iso}` = 7.3836 meV anchor of the frame section.
Merged DMI/Jani decompositions are **not** certified as physical
predictions until the FR full-exchange cross-validation (story-013
cross-code gate) lands.

.. code-block:: bash

   # full three-leg rank-nine driver (heavy): three SAXIS run directories
   vasp_split_soc2J.py --leg x=collinear_x --leg y=collinear_y --leg z=collinear_z \
       --output_path TB2J_results_vasp_split_soc

The driver CLI and the env-gated real-fixture gates live on the VASP
split-SOC adapter branch (``wt-tb2j-vasp``; adapter commits ``7c59606``
and ``37a2a83`` on top of this kernel cutover) until that branch merges
into ``SPLIT_SOC``; ``tests/tests/test_vasp_split_soc.py`` there runs
the loader/oracle smoke (dump format dispatch, provenance, k-weight
normalization, the exact strength-0 :math:`E_{soc} = 0` oracle, CSO
Hermiticity, run-identity pairing, COCC reconstruction, all-atom
:math:`W_{SO}` Hermiticity) and the three-run FeO merge gates.  The
licensed fixture files (VASP PAW data) are never committed: point the
tests at the retained campaign directories via their environment
variables.

.. warning::

   Do **not** quote merged DMI/Jani values from this workflow as
   certified physical results: the raw rank-nine merge is consistent by
   its own gates, but the full-exchange cross-validation against the
   fully-relativistic leg (story-013 cross-code gate) is pending, and
   the FR comparison above is an upper bound.  :math:`J_{iso}` per leg,
   the anchors and the :math:`E_{soc}` oracle are the certified
   observables.
