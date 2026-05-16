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

   abinit_projector2J.py --input run_SAVETB2J.nc \
       --output_path TB2J_results_abinit --elements Fe --Rmax 1 --nz 30 \
       --smearing 0.05

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
