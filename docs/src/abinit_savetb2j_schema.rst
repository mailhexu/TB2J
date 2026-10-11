ABINIT savetb2j NetCDF Schema
=============================

This document defines the version 1 contract for ABINIT PAW projector data
exported with the ``savetb2j`` input variable and consumed by TB2J.  The schema
stores spectral ingredients and local PAW operators; TB2J reconstructs projector
Green functions at runtime.

Scope
-----

Version 1 supports collinear PAW data only:

* ABINIT ``usepaw == 1``;
* ABINIT ``nspinor == 1``;
* spin-resolved collinear data suitable for ``delta_total`` exchange;
* full-Brillouin-zone k-points and projector coefficients.

Noncollinear, spin-orbit, spinor, norm-conserving, and IBZ-only exports are not
part of version 1 unless a later schema version explicitly extends the contract.

User Workflow
-------------

Set ``savetb2j 1`` in a supported ABINIT ground-state PAW input.  ABINIT writes
one NetCDF file whose name is the dataset output prefix followed by
``_SAVETB2J.nc``.  The file can be passed directly to TB2J:

.. code-block:: bash

   abinit2J.py savetb2j --input run_SAVETB2J.nc \
       --output_path TB2J_results_abinit --elements Fe --Rcut 10.0 --nz 30 \
       --smearing 0.05

The ``savetb2j`` subcommand auto-detects the backend from the file's
``schema_name`` attribute: ``abinit.savetb2j.projector`` files (this schema)
take the PAW option set, while norm-conserving PAO and spherical-window
exports are routed to the NC backend with its shell-filter and overlap
options.  Options that do not apply to the detected backend are rejected.

TB2J uses ``operator_components/delta_total`` by default.  In version 1 this is
the exchange-ready spin-up minus spin-down PAW onsite operator in the native
ABINIT PAW projector basis.

Root Attributes
---------------

The NetCDF root group must define these attributes:

.. list-table:: Required root attributes
   :header-rows: 1

   * - Attribute
     - Version 1 value or meaning
   * - ``schema_name``
     - ``abinit.savetb2j.projector``
   * - ``schema_version``
     - ``1.0``
   * - ``source_code``
     - ``abinit``
   * - ``abinit_version``
     - ABINIT version string used for the export
   * - ``spin_mode``
     - ``collinear``
   * - ``spin_channel_order``
     - ``up,down``; spin index 0 is spin-up and spin index 1 is spin-down
   * - ``full_bz``
     - true
   * - ``kpoint_convention``
     - ``fractional_reciprocal``
   * - ``phase_convention``
     - convention used by TB2J after conversion, normally ``exp(-2*pi*i*k.R)``
   * - ``coefficient_source``
     - ``abinit.cprj``
   * - ``operator_basis``
     - ``abinit_native_paw_projector``
   * - ``units_json``
     - JSON mapping for length, energy, positions, eigenvalues, and operators

Files missing any required root attribute must be rejected by the TB2J ABINIT
loader.

Dimensions
----------

Version 1 uses the following common dimensions:

.. list-table:: Dimensions
   :header-rows: 1

   * - Name
     - Meaning
   * - ``nspin``
     - Number of collinear spin channels; must be 2 for exchange-ready data
   * - ``nkpt``
     - Number of full-BZ k-points
   * - ``nband``
     - Number of exported bands
   * - ``nproj``
     - Number of global PAW projector channels
   * - ``nsite``
     - Number of atomic sites with projector blocks
   * - ``nproj_site_max``
     - Maximum projector count on any site
   * - ``natom``
     - Number of atoms
   * - ``three``
     - Cartesian/reduced-vector length, always 3
   * - ``complex``
     - Complex encoding length, always 2: real then imaginary

Groups and Variables
--------------------

``/structure``
~~~~~~~~~~~~~~

``cell(three, three)``
    Lattice vectors in Angstrom.

``positions(natom, three)``
    Atomic Cartesian positions in Angstrom.

``atomic_numbers(natom)``
    Atomic numbers.  This array is required in version 1.  ABINIT-side export
    code must resolve species to atomic numbers before writing the file rather
    than relying on a loader-specific species-name fallback.

``/kpoints``
~~~~~~~~~~~~

``kpoints(nkpt, three)``
    Full-BZ k-points in fractional reciprocal coordinates.

``weights(nkpt)``
    Full-BZ k-point weights.  The weights must sum to one within numerical
    tolerance.

``/bands``
~~~~~~~~~~

``eigenvalues(nspin, nkpt, nband)``
    Eigenvalues in eV.  For version 1, spin index 0 is spin-up and spin index 1
    is spin-down, matching root attribute ``spin_channel_order = "up,down"``.

``occupations(nspin, nkpt, nband)``
    Optional occupations in the same spin/k/band order.

``efermi``
    Fermi energy in eV, stored as a group attribute.

``/projectors``
~~~~~~~~~~~~~~~

The projector group must carry these attributes:

* ``coefficient_source = "abinit.cprj"``;
* ``coefficient_projector = "paw_nonlocal_projector"``;
* ``channel_interpretation = "abinit_paw_lmn_channel"``;
* ``operator_basis = "abinit_native_paw_projector"``.
* ``index_base = 0``.

``coefficients(nspin, nkpt, nband, nproj, complex)``
    Complex ``<p_lmn|Cnk>`` coefficients from ABINIT ``pawcprj_type%cp``.

``projector_atom(nproj)`` and ``projector_site(nproj)``
    Zero-based atom and site indices.  Version 1 files must store zero-based
    indices and set ``/projectors:index_base = 0``.  One-based ABINIT internals
    must be converted before writing.

``projector_l(nproj)``, ``projector_m(nproj)``, ``projector_radial(nproj)``
    PAW channel metadata derived from ABINIT PAW tables.

``site_nproj(nsite)`` and ``site_projector_indices(nsite, nproj_site_max)``
    Per-site projector block metadata.  Padding entries must be ``-1``.

``overlap_metric(nproj, nproj, complex)``
    Optional global projector overlap metric.  Version 1 should populate onsite
    blocks from ABINIT PAW overlap information when available and set intersite
    augmentation blocks to zero.

``/operators``
~~~~~~~~~~~~~~

``hij(nspin, nsite, nproj_site_max, nproj_site_max, complex)``
    Optional spin-resolved total onsite PAW operator in eV.  If present, its
    attributes must define ``definition``, ``units``, ``source``, ``projection``,
    and ``operator_basis``.  The spin dimension must follow
    ``spin_channel_order = "up,down"``.

``operator_components/delta_total(nsite, nproj_site_max, nproj_site_max, complex)``
    Exchange-ready local operator in eV.  This is the default operator used by
    TB2J and should equal the spin-up total onsite operator minus the spin-down
    total onsite operator in the exported projector basis.

``operator_components/dijxc``
    Optional XC onsite contribution or spin splitting in eV.  Metadata must
    declare whether it is spin-resolved, already spin-differenced, and complete.

``operator_components/dijU``
    Optional PAW+U onsite contribution or spin splitting in eV.  Metadata must
    declare whether the calculation used PAW+U and whether the array is absent,
    present zero, or present nonzero.

``operator_components/dijso``
    Optional spin-orbit onsite contribution in eV.  For version 1 collinear
    exports this is usually absent or zero; if nonzero, TB2J must not use it for
    exchange unless a later story validates the convention.

Operator component arrays with ``spin_treatment = "spin_difference"`` must use
shape ``(nsite, nproj_site_max, nproj_site_max, complex)``.  Operator component
arrays with ``spin_treatment = "spin_resolved"`` must use shape
``(nspin, nsite, nproj_site_max, nproj_site_max, complex)`` and the same
``spin_channel_order`` as ``hij``.

Component Metadata
------------------

Each operator component must carry attributes describing:

* ``source``: ABINIT source array, for example ``paw_ij%dijU``;
* ``units``: eV;
* ``operator_basis``: ``abinit_native_paw_projector``;
* ``spin_treatment``: ``spin_resolved`` or ``spin_difference``;
* ``completeness``: ``complete``, ``not_present``, ``zero_by_symmetry``, or a
  documented incomplete status.

TB2J must reject files that request exchange from a component whose
``completeness`` metadata is absent or incompatible with the requested use.

Validation Rules
----------------

The TB2J ABINIT loader must reject files when:

* ``schema_name`` or ``schema_version`` is unsupported;
* ``full_bz`` is false or absent;
* ``spin_channel_order`` is absent or differs from ``up,down``;
* required groups or arrays are missing;
* ``/structure/atomic_numbers`` is missing;
* ``/projectors:index_base`` is absent or not zero;
* complex arrays do not use the final ``complex`` dimension;
* k-point weights are malformed;
* coefficient and eigenvalue leading dimensions disagree;
* projector metadata length does not match ``nproj``;
* site block metadata references out-of-range projectors;
* operator basis metadata is absent or differs from the coefficient basis;
* exchange is requested but neither ``operator_components/delta_total`` nor
  sufficient spin-resolved ``hij`` data are present.

Version 1.1: ``soc_pauli`` (additive)
-------------------------------------

Schema version ``1.1`` extends the version 1 collinear export additively with
one optional operator component, written by ABINIT when ``savetb2j_soc`` is
enabled (the unit-strength ``pawdijso`` operator evaluated at the frozen
strength-0 density).  Version 1.0 files are unchanged, and version 1.1 files
without the component load exactly as before.

``operators/operator_components/soc_pauli``
    Shape ``(nsite, nproj_site_max, nproj_site_max, nspinor, nspinor,
    complex)`` with ``nspinor = 2``, on-disk units **Hartree** (also declared
    in ``units_json``).  Required provenance attributes: ``source``
    (``pawdijso(frozen_density)``), ``spin_treatment = "pauli_2x2"``,
    ``completeness = "complete"``, ``soc_strength = "1.0"``,
    ``spinaxis = "0 0 1"``, ``zora_term_class = "soc_only"``,
    ``quantization = "lattice_frame"``, ``covers = "all_atoms"``,
    ``reference = "strength_zero_frozen_density"``, and a non-empty
    ``pauli_component_order``.

On-disk orientation and TB2J contraction
    ABINIT packs one value per projector pair and Pauli component with
    ``stored(i <= j) = <p_j|W_SO|p_i>`` (``m_opernlc_ylm_allwf`` applies
    ``gxfac(jlmn) += enl * gxi(ilmn)``, ``gxfac(ilmn) += conj(enl) * gxj(jlmn)``
    per Pauli component); the packer writes these packed values directly into
    the ``[i, j]`` slots.  The dense blocks are Hermitian with the L\ \cdot S
    packing relations ``dd = -uu`` and ``du = -conjg(ud)``.  With the cprj
    coefficient convention ``c[p] = <p|psi>``, the band-space SOC operator is
    ``W_nm(k) = c_n^dag O c_m`` with a **full composite transpose**
    ``O[p,q,s,t] = D[q,p,t,s]`` (both projector and spin index pairs,
    flat index ``2*p+s``).  Transposing projector indices alone reverses
    the complex spin-flip matrix elements and fails the iodine 5p L·S
    element-level oracle.  Since the dense block is Hermitian, the full
    composite transpose equals elementwise ``conjg(D)`` numerically, but
    conjugating the wrong coefficient side does not implement the same
    band-space contraction. The TB2J loader normalizes ``soc_pauli`` to
    the operator orientation ``O[p,q,s,t]`` in eV at load time and records
    ``orientation = "operator"``, ``source_units``, and the
    ``band_space_contraction`` string in the component metadata.

Loader validation for ``soc_pauli``
    The loader rejects the component when the units are not Hartree,
    ``units_json`` contradicts Hartree, the shape is not
    ``(nsite, nproj_site_max, nproj_site_max, 2, 2)``, the dense blocks are
    not Hermitian, the packing relations ``dd = -uu`` or
    ``du = -conjg(ud)`` are violated, the provenance attributes are missing
    or contradict unit strength / lattice-frame quantization / all-atom
    coverage / SOC-only term class, or the component appears in a
    ``schema_version`` ``1.0`` file.

Split-SOC workflow
    ``TB2J/interfaces/abinit_paw_split_soc.py`` consumes the normalized
    component: collinear up/down bands are **interleaved** so each even
    window prefix contains equal numbers of both channels.
    ``W_SO^K(k) = sum_a c_a^dag O_a^{(leg)} c_a`` uses SOC on ALL atoms
    (ligands included); exchange vertices use ``delta_total`` or ``delta_xc``
    on explicitly selected magnetic sites only. If the export lacks magnetic
    moments, pass ``index_magnetic_atoms`` or ``magnetic_elements``: never
    infer that every SOC-bearing ligand is magnetic. Signed per-site
    ``spinat`` directions use exported moments when available; otherwise
    they are **opposite** the sign of the frozen ``H_up-H_down`` PAW
    potential trace (majority-spin potential is lower for positive moment).
    This preserves AFM sublattices and the positive Fe moment. Three
    spinaxis SU(2) legs rotate their measured transverse blocks to the
    lattice frame and are merged by the raw rank-nine solve
    (``TB2J.split_soc_kernel.merge_transverse_legs``, repeated-row
    invariance gate at ``merge_consistency_atol`` = 1e-4 eV in this
    driver); the legacy scalar ``TB2J.io_merge`` averaging is not used.

    The driver writes only absolute second-variational exchange; insertion
    derivatives are not mislabeled as J. A real two-prefix band-window
    study, its measured change, tolerance, and convergence flag accompany
    each leg in ``leg_<x|y|z>/split_soc_provenance.json`` (per-leg tensors
    are raw ``split_soc_leg.npz`` blocks); the rank-nine merged result is
    written to the output root with the three distinct leg records in
    ``split_soc_provenance.json``. A false convergence flag
    requires a larger ABINIT band window, not a favorable error bar.

    The real eight-k Fe fixture has a nonzero 33.992-meV SOC-off first
    exchange shell. Its physical Fe moment is positive although the PAW
    splitting trace is negative. A native fixed-density spinor
    ``iscf=-2`` response at lambda 0→0.005 (8 k-points × 24 states)
    matches the *loaded-component consumer* band eigenvalue changes to
    within 0.886 meV maximum / 0.065 meV RMS. This is a small-strength
    eigenvalue check, not a full-strength ABINIT SOC validation: the same
    Fe 22→24 band-window study reports ``converged: false`` at 1e-6.

    A real fcc Ni PAW fixture (``a = 3.52`` Å, one-atom primitive cell,
    schema 1.1, 28-band window) passes the full rank-nine three-leg
    merge: design-matrix rank 9, repeated-diagonal agreement
    9.0e-8 eV against the 1e-4 eV gate, transverse-mask and reciprocity
    residuals below 1e-18 eV, merged nearest-neighbour ``J_iso`` 0.209 meV.
    Its own 26→28 band study is not converged (4.7e-5 change at 1e-6) and
    stays flagged.

For example, using zero-based Python atom indices::

    from TB2J.interfaces.abinit_paw_split_soc import gen_exchange_abinit_paw_split_soc
    gen_exchange_abinit_paw_split_soc(
        "fe_soc1o_SAVETB2J.nc", index_magnetic_atoms=[0], Rcut=8.0
    )

Synthetic Fixture Requirements
------------------------------

TB2J tests should include a tiny synthetic ABINIT-like NetCDF fixture with:

* two spin channels;
* two full-BZ k-points with weights summing to one;
* two bands;
* one atom/site with two PAW channels;
* complex coefficients with nonzero imaginary parts;
* one ``hij`` block and matching ``delta_total``;
* ``dijxc``, ``dijU``, and ``dijso`` component groups with explicit metadata;
* at least one negative fixture missing ``full_bz`` or operator-basis metadata.

The fixture must be small enough to create inside unit tests and must not depend
on an ABINIT executable.  ABINIT-generated fixtures are covered by later
end-to-end validation stories.


ABINIT NC split-SOC sidecar (``abinao.nc_soc_ks`` v1)
-----------------------------------------------------

This section documents the SOC sidecar consumed together with the
norm-conserving PAO projection file (``abinit.nc_pao_hs`` v2, see
:doc:`projector_green`) by the ABINIT NC split-SOC workflow of
:doc:`split_soc_abinit_nc`.  The sidecar is written by the *abinao*
``soc_kernel`` writer, not by ABINIT's ``savetb2j``; it is a separate
contract that exists only as the SOC partner of the NC PAO file, and the
TB2J consumer joins the two artifacts by SHA-256 before any physics.

Root attributes
~~~~~~~~~~~~~~~

* ``schema_name = "abinao.nc_soc_ks"``, ``schema_version = "1"``;
* ``energy_unit = "eV"`` and ``hartree_to_ev`` (validated against the
  TB2J constant to :math:`10^{-12}` relative tolerance);
* ``band_window_lo`` / ``band_window_hi`` (half-open composite window,
  ``-1 -1`` = full window);
* ``spnorbscl`` (must be exactly ``1.0``: the sidecar stores the
  :math:`\lambda=1` kernel; scaling belongs to the consumer's ``lam``);
* ``all_atoms_covered`` (must be ``1``: ligand SOC enters the propagator);
* ``fr050_metadata`` (JSON: ``strength0_provenance`` including
  ``nsppol``/``nspinor``, ``operator`` source, ``frame``/``spinaxis``);
* ``source_wfk`` and ``source_wfk_sha256``;
* optional ``pao_hs`` and ``pao_hs_sha256`` — the TB2J consumer *requires*
  the hash and refuses an unverifiable pairing without it.

Variables
~~~~~~~~~

* ``leg(nleg)`` = ``x, y, z`` and ``spinaxis(nleg, 3)`` (one unit axis per
  leg, matching the ABINIT spinaxis rotation convention);
* ``kpts(nkpt, 3)`` and ``kweights(nkpt)`` in full-BZ WFK order, weights
  non-negative;
* ``w_so_real`` / ``w_so_imag`` of shape ``(nleg, nkpt, 2n, 2n)``: the
  Hermitian band-window SOC operator in the composite basis
  :math:`i = 2n + \sigma`, all atoms, in eV;
* optional ``w_so_site_real`` / ``w_so_site_imag`` (site-resolved blocks
  summing to the band matrix);
* optional ``eigenvalues_ev(nleg, nkpt, 2n)`` used as a cross-check
  against the PAO_HS band energies (tolerance :math:`10^{-6}` eV).

Loader and pairing refusals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TB2J loader refuses: any schema name/version mismatch; a non-eV
``energy_unit``; a ``hartree_to_ev`` mismatch; ``spnorbscl != 1``; missing
all-atom coverage; non-Hermitian :math:`W^k_{SO}`; negative k-weights;
spinaxis vectors that disagree with the leg axes; non-finite or
all-zero leg matrices; inconsistent shapes; and composite eigenvalue
arrays of the wrong length.  The pairing gate additionally refuses: a
PAO_HS/WFK SHA-256 mismatch, a missing ``pao_hs_sha256``, k-point
order/gauge or k-weight mismatches against the PAO_HS file, an
eigenvalue cross-check failure, a composite band count different from
:math:`2 \times` the PAO_HS ``nband``, and a *spinor-flavor* sidecar
(``nsppol=1``, ``nspinor=2``), which must be routed through a
spinor-flavor consumer instead.
