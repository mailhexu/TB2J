Use TB2J with Siesta
====================

In this tutorial we will learn how to use TB2J with Siesta. First we calculate the isotropic exchange in bcc Fe. Then we calculate the isotropic exhange, anisotropic exchange and Dzyanoshinskii-Moriya interaction (DMI) parameters from a calculation with spin-orbit coupling enabled.
Before running this tutorial, please make sure that the sisl package is installed.

For the separate certified finite-pitch spiral workflow, see
:doc:`siesta_spiral_cli`. It runs digest-pinned producer inputs through the
required response/mapping gates and writes a validated SpinIO pickle to a
new directory; it does not replace the collinear tutorial below.

For a CrI3 material-response campaign with independent basis, grid, k-mesh,
contour-integration and derivative evidence, see
:doc:`siesta_spiral_material_campaign`.

.. toctree::
   :maxdepth: 1

   siesta_spiral_material_campaign.md


Complete WFSX spectral preparation (Python)
------------------------------------------

``TB2J.green.TBGreen`` accepts a metadata-only model capability named
``eigenstate_source`` containing a HamiltonIO ``SiestaWFSXParser``. This
low-level preparation interface reads complete WFSX records and explicit
metadata; it never loads Hamiltonian/overlap input or solves an electronic
eigenproblem. It is separate from the ordinary exchange tutorial below.

Supply HamiltonIO's flat metadata mapping, including ``schema_version=1``,
ordered lattice/atom/AO/spin data, absolute Fermi energy and its provenance,
electron/occupation provenance, raw gauge and weight semantics. A fixture's
provenance ``fixture.json`` is not this flat consumer mapping. TB2J additionally
requires ``scf_mesh`` for every export mode, including selected lists: supply the
original full SCF coordinates and uniform weights, not only representatives.
The full-domain all-channel electron count is checked at the declared occupation
temperature before any contour work. Missing provenance mesh raises
``MissingMetadata`` and inconsistent electron/Fermi provenance raises
``InconsistentMetadata``. Requested subsets retain their separate explicit
uniform quadrature; their populations do not substitute for the global count.

For example, for a complete export covering a Gamma-centered 1x1x5 grid::

   import json
   from pathlib import Path
   from types import SimpleNamespace
   from HamiltonIO.siesta.wfsx import SiestaWFSXParser
   from TB2J.green import TBGreen

   metadata = json.loads(Path("wfsx_metadata.json").read_text())
   model = SimpleNamespace(
       eigenstate_source=SiestaWFSXParser("states.WFSX", metadata),
       spin_channel=0,
       reduce_scalar=False,
   )
   green = TBGreen(
       model, kmesh=(1, 1, 5), gamma=True, efermi=0,
       cache_path="spectral_cache",
   )
   try:
       gk = green.get_Gk(0, -5 + 1j)
       gr = green.get_GR([[0, 0, 1]], -5 + 1j)
       population = green.get_density_matrix()
       onsite = green.H0
   finally:
       green.clean_cache()

``spin_channel`` is 0 or 1 for independent collinear channels, and 0 for
scalar/full-spinor input. ``reduce_scalar=True`` requests optional internal
scalar folding only for a fully inversion-closed requested domain. It needs
all four true symmetry declarations (real scalar Hamiltonian, no SOC, no
orbital field and no spin spiral). Supplied partners must pass operator and
normalization validation before contour evaluation. Missing scalar partners
retain declaration-only provenance. Spinors require direct full-grid records.

One coefficient LU factorization per representative/channel is reused across
contour energies. Reciprocal condition below 1e-10 raises HamiltonIO's
``SingularEigenbasis``; no regularization or H/S fallback is attempted. The
cache holds coefficient/eigenvalue/dual factors, never all-k H/S. Only one
representative is memory-mapped at a time and internal mappings close before
a getter returns. Full energy arrays and explicitly requested output arrays
still have their normal sizes.

Spectral disk staging is mandatory even if ordinary ``use_cache=False``.
Choose a real disk cache directory with adequate capacity; the default uses
the system temporary directory, not the ordinary ``/dev/shm/TB2J_cache``
default. A user-configured temporary directory may itself be RAM-backed.
Cleanup removes only the owned temporary child and preserves its parent and
user inputs. ``get_Hk``/``get_Sk`` reconstruct one requested dense diagnostic
operator. Both ordinary ``TB2J_SIESTA_GREEN_MODE`` settings are preserved;
the environment's metric dressing applies at ``get_GR`` as in the ordinary
path.

WFSX preparation always retains the complete spectrum. Ordinary H/S
``TBGreen`` keeps its legacy contour band window by default. Use its explicit
``complete_spectrum=True`` option for matched full-spectrum H/S comparisons;
this bypasses only that window, not the ordinary eigensolve. Measured
factorization, import/preparation and disk-size fields are available in
``green.spectral_stats``. No net speed or memory improvement is implied
without a matched measurement.


Exchange from complete WFSX states (Python)
------------------------------------------

``prepare_exchange_siesta_wfsx`` prepares the existing collinear or full
noncollinear exchange kernel directly from a validated HamiltonIO parser.
No H/S files or electronic eigensolver are used:

.. code-block:: python

   from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

   exchange = prepare_exchange_siesta_wfsx(
       model.eigenstate_source, magnetic_elements=["O"],
       kmesh=[1, 1, 5], nz=100, Rcut=4.3,
       reduce_scalar=True, cache_path="spectral_cache",
   )
   try:
       exchange.calculate_all()
       exchange.write_output(path="TB2J_results")
   finally:
       exchange.finalize()

For collinear states each spin channel is reduced independently. Supplied
and reconstructed scalar partners retain their distinct full-grid Fourier
phases and weights; the same-complex-energy partner Green is transposed,
not conjugated. Mesh H0 includes the partner operator before averaging.
Both existing Green metric modes retain their ordinary conventions.

Explicit ``kpts``/``kweights`` select the actual integration quadrature.
``kmesh`` retains the ordinary real-space R-list construction, including
its existing odd-mesh convention; it does not replace explicit points.
Non-inversion-closed requested meshes gain no extra points. Full spinor
states require direct records at every requested point and
``reduce_scalar=False``.

The supported operation is single-process CPU full-operator exchange.
Scalar nspin=1 is reader/Green-only. Spinor reduction raises
``UnsupportedSpinMode``. Split-SOC/MAE decomposition, operator rescaling,
spin-spiral/GBT, orthogonalization, GPU, IBZ interpolation and DMFT requests
raise ``UnsupportedOperatorMode`` rather than falling back to H/S.
Missing scalar symmetry declarations and invalid meshes retain HamiltonIO's
named errors.

The default occupation/integration width is the metadata temperature times
Boltzmann's constant. Exactly zero temperature retains half occupation at
the Fermi pole. CFR/CFR2 contours require positive integration broadening:
explicitly supply positive ``smearing`` or choose a non-CFR ``method``
such as ``"semicircle"``. No width or contour is changed automatically.
The global electron provenance check still uses the metadata temperature,
independently of an explicitly requested integration width.

Same-run magnetic collinear full/reduced and NC4 full-grid fixtures were
compared against complete-spectrum H/S exchange, including the full spinor
tensor, nonzero R, and both metric modes. This does not establish SOC8
producer coverage or a performance improvement.


Collinear calculation without SOC
--------------------------------------
Let's start from the example of BCC Fe. The input files used can be found in the examples/Siesta/bccFe directory. 

First, we do a siesta self consistent calculation with the BCC Fe primitive cell of bcc Fe has only one Fe atom. In the example, we use the pseudopotential from the PseudoDojo dataset in the psml format. Note that at the moment the psml support is implemented in the master development and "Max" branches of siesta (see https://gitlab.com/siesta-project/siesta/-/wikis/Guide-to-Siesta-versions for the different versions of siesta, and https://docs.siesta-project.org/projects/siesta/en/latest/installation/build-manually.html for how to build it.). We need to save the electronic Kohn-Sham Hamiltonian in the atomic orbital basis set with the options:

::

   SaveHS True

and this option to  use the netcdf format (if netcdf is enabled within the siesta version being used). 

::
   CDF.Save True

After that, we will have the files siesta.nc and DMHS.nc file, which contains the Hamiltonian and overlap matrix information.

Now we can run the siesta2J.py command to calculate the exchange parameters:

::

   siesta2J.py --fdf_fname siesta.fdf --elements Fe --kmesh 7 7 7

This first read the siesta.fdf, the input file for Siesta. It then read the Hamiltonian and the overlap matrices, calculate the J with a :math:`7\times 7 \times 7` k-point grid. This allows for the calculation of exchange between spin pairs between :math:`i` and :math:`j` in a :math:`7\times 7 \times 7` supercell, where :math:`i` is fixed in the center cell. Note: the kmesh is not dense enough for a practical calculation. Please check the convergence. 



Non-collinear calculation
-----------------------------------------

The anisotropic exchange and the DMI parameters can be calculated with non-collinear DFT calculation. The procedure is almost the same as in the collinear calculation except that the parameters for non-collinear calculation must be set in the Siesta input (Spin should be set to non-colinear or spin-orbit). 


Frozen finite-pitch response and Heisenberg compatibility (experimental)
-----------------------------------------------------------------------

The ``TB2J.interfaces.siesta_spiral_*`` Python APIs consume a validated
native-spiral HSX/EIG/KP export, a certified site partition, and two
transverse rotation axes per site. This is a frozen one-body response,
not a self-consistent total-energy exchange calculation. The default
rotating operator is ``onsite_delta`` (all R=0 same-site orbital blocks);
``weighted_total_h`` is a separately labeled diagnostic variant.

For a gapped T=0 reference use ``evaluate_response_set`` with a declared
occupied-pole ``EnergyContour``. For a metallic reference with matched
Fermi-Dirac occupations use ``Ensemble(kind=ENSEMBLE_METALLIC_FD, ...)``
and ``evaluate_metallic_response_set`` from
``TB2J.interfaces.siesta_spiral_metallic``. The latter solves **one global
chemical potential** at fixed electron count, includes the diagonal
Fermi-surface derivative and the fixed-N Schur correction, and certifies
analytic gradients/curvature against free-energy finite rotations with
the chemical potential re-solved at each displacement. Its
``contour_vs_band_max_abs`` is ``None`` (no contour was evaluated). The
declared producer Fermi energy must reproduce the count within 1e-4
electrons; otherwise it is refused. The source and integration provenance,
actual sampled gap, occupations per k point, chemical potential, and
finite-rotation errors are retained in the ``ResponseSet``.

Pass the raw response to ``map_response_set`` in
``TB2J.interfaces.siesta_spiral_mapping`` with **certified** producer
``reference_directions_by_site``. The output retains the raw response
unchanged, reports a fitted pair-once J and stored ordered-pair J=J/2
only where identifiable, and labels compatibility from Hessian and
measured-torque residuals (chi2 <= 1e-4 compatible, >= 2.5e-3
incompatible, otherwise marginal). ``j_hat_scalar_eV`` is the separate
zero-pitch common-axis diagnostic matched to the collinear LKAG anchor;
it is not generally the fitted J when transverse anisotropy is present.
The measured Fe4 producer-gate response at q=(0,0,1/4), 300 K and
32 electrons has sampled indirect gap -0.24038 eV and is labeled
``incompatible`` by this mapping (chi2_hessian ~0.1865); its individual
pair J values are **undetermined** by the joint four-site fit, never
published as converged exchange constants.

To compare two orientations of the **same** onsite-Delta frozen operator,
call ``recenter_metallic_reference(problem, angles_by_site=...,
reference_directions_by_site=...)`` and evaluate its returned problem with
the original producer ensemble and its returned rotated normals. This
rotates site-local magnetic blocks without changing the root HSX/vertex
certificate; the fixed-N chemical potential and occupations are solved
anew. It does not re-run SCF and does not support weighted-total-H rotation.
Both references must pass a stable two-step free-energy finite-rotation
ladder before a numeric fitted-J difference is reported.

``compare_mapping_outputs(left, right)`` reports fitted-J reference
dependence only when source bundle, pitch, vertex/operator partition,
site identity, **declared** ensemble settings and integration all match.
A matched but unvalidated ladder yields ``uncertified_reference`` with
no numeric pair differences. Different SCF runs or pitches are explicitly
``unmatched_cross_scf``: both records and
provenance are kept, but no numeric exchange difference is emitted.
Chart-local D and Gamma are not compared across orientations. The
story-004 independent Fe4 primitive/supercell SCFs differ by +0.115644
eV per primitive cell in free energy; that mismatch is not an exchange
convergence datum. The SymPy authorities are
``docs/sympy/siesta_spiral_metallic_response.py`` and
``docs/sympy/spiral_reference_dependence.py``.
