Projector Green Data
====================

TB2J can store and read code-agnostic projector-space spectral data for
Green-function reconstruction.  The version 1 schema stores spectral ingredients
only; it does not store precomputed Green functions.  TB2J reconstructs
``G(k, E)`` and ``G(R, E)`` at runtime for the requested energy points.

The Python API lives in ``TB2J.projector_green``.  The core classes are
``ProjectorGreenData`` for validated data storage and ``ProjectorGreen`` for
runtime Green-function reconstruction.

Required Data
-------------

A valid projector Green dataset contains:

* full-Brillouin-zone k-points, ``kpoints[nkpt, 3]``;
* k-point weights, ``weights[nkpt]``;
* eigenvalues, ``eigenvalues[nspin, nkpt, nband]``;
* projector coefficients, ``coefficients[nspin, nkpt, nband, nproj]``;
* Fermi energy, ``efermi``;
* projector-to-site and projector-to-atom maps;
* optional structure data: cell, positions, and atomic numbers;
* optional occupations with the same shape as eigenvalues.

The coefficient convention is

.. math::

   C^p_{n k \sigma}=\langle p | \psi_{n k \sigma}\rangle.

TB2J reconstructs

.. math::

   G^{pq}_{\sigma}(k,E)=\sum_n
   \frac{C^p_{n k \sigma} C^{q *}_{n k \sigma}}
   {E + E_F - \epsilon_{n k \sigma}}.

The real-space transform uses the TB2J phase convention
``exp(-2*pi*i*k.R)`` and the stored full-BZ k-point weights.

Projector files can record optional channel metadata:

* ``coefficient_source``: source-code object or formula for the stored
  coefficients;
* ``coefficient_projector``: projector used in the coefficient overlap;
* ``channel_interpretation``: local channel interpretation for matrices and
  reporting;
* ``operator_basis``: basis/representation of the site-local ``hij`` operator;
* ``overlap_metric_definition`` and ``population_metric``: provenance for
  overlap and population contractions.

For GPAW PAW export, the stored coefficients are ``gpaw.P_ani`` overlaps with
dual PAW projectors.  The local channel is interpreted as the PAW partial-wave
channel, and the current ``hij`` operator basis is the native GPAW PAW projector
Hamiltonian matrix.  Pseudo-partial-wave or all-electron partial-wave variants
should export their final local operator matrices and identify the corresponding
``operator_basis`` explicitly.

ABINIT PAW exports produced by the ``savetb2j`` input variable use a
source-specific NetCDF contract documented in
``docs/src/abinit_savetb2j_schema.rst``.  TB2J normalizes that schema into
``ProjectorGreenData`` only after validating the full-BZ flag, projector basis
metadata, and local-operator basis metadata.

NetCDF Layout
-------------

NetCDF export is optional and requires ``netCDF4``.  Complex arrays are encoded
with a final dimension ``complex=2`` where index 0 is real and index 1 is
imaginary.

Version 1 files use these groups:

* ``/structure``: optional cell, positions, and atomic numbers;
* ``/kpoints``: full-BZ k-points and weights;
* ``/bands``: eigenvalues, occupations, and Fermi energy;
* ``/projectors``: coefficients, projector metadata, and optional
  ``overlap_metric``;
* ``/operators``: optional site-local operators such as ``hij``.

Version 1 files must not contain stored ``/greens_k`` or ``/greens_R`` groups.

For GPAW PAW files, ``/projectors/overlap_metric`` stores the block-diagonal
onsite PAW overlap correction matrix ``dO_ii`` in the global projector order.
Intersite blocks are zero because GPAW's PAW augmentation overlap is onsite.
The matrix is independent of k; k dependence enters through the projector
coefficients ``P_ni(k)`` and eigenvalues.

GPAW PAW Export
---------------

The reusable GPAW exporter is
``TB2J.interfaces.gpaw_projector.save_gpaw_projector_netcdf``.  It accepts a
converged GPAW calculator and writes the TB2J projector Green NetCDF file:

.. code-block:: python

   from ase.build import bulk
   from gpaw import GPAW, PW, FermiDirac
   from TB2J.interfaces.gpaw_projector import save_gpaw_projector_netcdf

   atoms = bulk("Fe", "bcc", a=2.86)
   atoms.set_initial_magnetic_moments([2.2])
   calc = GPAW(
       mode=PW(400),
       xc="PBE",
       kpts=(9, 9, 9),
       spinpol=True,
       occupations=FermiDirac(0.05),
       nbands=16,
       symmetry="off",
       txt=None,
   )
   atoms.calc = calc
   atoms.get_potential_energy()

   data = save_gpaw_projector_netcdf(
       calc,
       "gpaw_bcc_fe_projector_green.nc",
       atoms=atoms,
       metadata={"kmesh": [9, 9, 9], "pw_cutoff_eV": 400},
   )

The exporter stores full-BZ projector coefficients ``P_ni(k)``, eigenvalues,
occupations, native GPAW ``dH_asp`` as ``hij``, PAW onsite ``dO_ii`` as
``overlap_metric``, and GPAW PAW population matrices ``N0_p`` in metadata for
``w_charge``/``w_magmom`` reporting.

CLI Use
-------

After a NetCDF file has been written, run the projector exchange interface with
the CLI:

.. code-block:: bash

   gpaw_projector2J.py --input gpaw_bcc_fe_projector_green.nc \
       --output_path TB2J_results --elements Fe --Rmax 1 --nz 30 --smearing 0.05

For cubic SrMnO3 with Mn as the only magnetic site:

.. code-block:: bash

   gpaw_projector2J.py --input gpaw_cubic_srmno3_projector_green.nc \
       --output_path TB2J_results_srmno3 --index_magnetic_atoms 2 \
       --Rmax 1 --nz 30 --smearing 0.05

The CLI reads the NetCDF file, reconstructs ``G(R,E)`` at the continued-fraction
energy points, contracts the controlled projector trace, and writes the standard
TB2J text output ``exchange.out``.

ABINIT PAW Export
-----------------

ABINIT writes TB2J projector data when a ground-state input sets
``savetb2j 1``.  The output file name is the ABINIT dataset output prefix with
``_SAVETB2J.nc`` appended.  Version 1 is intentionally narrow: it supports only
collinear PAW calculations with ``usepaw=1``, ``nsppol=2``, ``nspinor=1``, and
an explicit full-Brillouin-zone k-point list via ``kptopt=0``.  Symmetry-reduced
IBZ exports, noncollinear spinors, spin-orbit workflows, and norm-conserving
pseudopotentials are not part of this schema version.

After ABINIT produces the file, run:

.. code-block:: bash

   abinit_projector2J.py --input run_SAVETB2J.nc \
       --output_path TB2J_results_abinit --elements Fe --Rmax 1 --nz 30 \
       --smearing 0.05

The ABINIT interface uses the exported ``delta_total`` operator component by
default.  That component is the spin-up minus spin-down onsite PAW operator in
the ABINIT native PAW projector basis.  Advanced users may select another
component with ``--operator_component`` only when the file marks that component
as complete and exchange-ready.

Projector Hamiltonian ``H_ij``
-----------------------------

Exchange-like projector traces require a site-local spin-dependent operator in
the same projector channel space as the Green function.  TB2J stores this as
``hij`` with dimensions

``hij[nspin, nsite, nproj_site_max, nproj_site_max]``.

The intended meaning is a projector Hamiltonian or potential matrix,

.. math::

   H^a_{ij,\sigma}=\langle p_i^a | H_\sigma | p_j^a\rangle

or

.. math::

   \langle p_i^a | V_\sigma | p_j^a\rangle.

It is not a density or density matrix.  For the first collinear trace path TB2J
uses only the spin-dependent part,

.. math::

   \Delta^a_{ij}=H^{a,\uparrow}_{ij}-H^{a,\downarrow}_{ij}.

For PAW codes this matrix may already exist in the DFT code.  For example, GPAW
has native PAW projector Hamiltonian matrices such as ``dH_asp`` or
``dH_asii``.  Exporters should write these matrices as normalized ``hij`` and
record provenance in attributes such as ``hij.source`` and ``hij.projection``.

For non-PAW projectors, the DFT-code exporter must compute the matrix by
projecting the relevant potential or Hamiltonian onto the chosen projectors.
TB2J provides helper functions for simple discrete grids:

.. code-block:: python

   from TB2J.projector_green import (
       build_site_projector_indices,
       pack_site_hij,
       project_potential_to_hij,
   )

   hij_global = project_potential_to_hij(projectors, potential, weights)
   site_nproj, site_indices = build_site_projector_indices(projector_site)
   hij = pack_site_hij(hij_global, site_indices, site_nproj)

These helpers assume the projectors and potential are already represented on a
compatible grid or quadrature.  Real DFT exporters remain responsible for using
the source code's exact projector definitions and integration conventions.

Minimal Runtime Use
-------------------

.. code-block:: python

   from TB2J.projector_green import ProjectorGreenData, ProjectorGreen

   data = ProjectorGreenData.load_netcdf("projector_green.nc")
   green = ProjectorGreen(data)

   Gk = green.get_Gk_all(energy=0.2 + 0.01j, ispin=0)
   GR = green.get_GR([(0, 0, 0), (1, 0, 0)], energy=0.2 + 0.01j, ispin=0)

Examples
--------

``examples/projector_green/build_synthetic_projector_green.py`` is a small
non-PAW synthetic example that needs no external DFT installation.

``examples/projector_green/gpaw_bcc_fe_projector_green.py`` runs a bcc Fe GPAW
PAW calculation, calls ``save_gpaw_projector_netcdf()``, reloads the NetCDF file,
reconstructs ``G(R,E)``, and writes ``TB2J_results/exchange.out``.  The example
uses a ``9x9x9`` k mesh and ``400 eV`` plane-wave cutoff by default to stabilize
the ferromagnetic bcc Fe state.  A representative run gave 729 k-points, 16
bands, 18 projectors, ``w_magmom=2.1623 mu_B``, and nearest-neighbor
``J_iso=7.903 meV`` at ``2.477 Angstrom``.

Representative bcc Fe ``exchange.out`` excerpt:

.. code-block:: text

   Atom number        x             y             z       w_charge  w_magmom
   Fe1             0.00000000    0.00000000    0.00000000    6.1692    2.1623
   Total                                                     6.1692    2.1623

      Fe1   Fe1   ( -1,  -1,  -1)  7.9029   (-1.430, -1.430, -1.430)  2.477

``examples/projector_green/gpaw_cubic_srmno3_projector_green.py`` builds a
five-atom cubic perovskite SrMnO3 cell with ``a=3.8 Angstrom``, PBE,
``5x5x5`` k mesh, and ferromagnetic Mn initialization.  The five-atom cubic
cell cannot represent G-type AFM, so the example is a projector-workflow test.
A representative run gave 125 k-points, 64 bands, 70 projectors,
``w_magmom(Mn)=2.5938 mu_B``, and nearest Mn-Mn ``J_iso=-17.143 meV`` at
``3.8 Angstrom``.

Representative SrMnO3 ``exchange.out`` excerpt:

.. code-block:: text

   Atom number        x             y             z       w_charge  w_magmom
   Sr1             0.00000000    0.00000000    0.00000000    7.7405    0.0089
   Mn1             1.90000000    1.90000000    1.90000000   13.4924    2.5938
   O1              1.90000000    1.90000000    0.00000000    3.6156    0.0607

      Mn1   Mn1   (  0,   0,   1) -17.1433   ( 0.000,  0.000,  3.800)  3.800

These GPAW examples require ``gpaw`` and ``gpaw-data``.
