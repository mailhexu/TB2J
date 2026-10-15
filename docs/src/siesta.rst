Use TB2J with Siesta
====================

In this tutorial we will learn how to use TB2J with Siesta. First we calculate the isotropic exchange in bcc Fe. Then we calculate the isotropic exhange, anisotropic exchange and Dzyanoshinskii-Moriya interaction (DMI) parameters from a calculation with spin-orbit coupling enabled.
Before running this tutorial, please make sure that the sisl package is installed.

Collinear calculation without SOC
--------------------------------------
Let's start from the example of BCC Fe. The input files used can be found in the examples/Siesta/bccFe directory. 

First, we do a siesta self consistent calculation with the BCC Fe primitive cell of bcc Fe has only one Fe atom. In the example, we use the pseudopotential from the PseudoDojo dataset in the psml format. Note that at the moment the psml support is implemented in the master development and "Max" branches of siesta (see https://gitlab.com/siesta-project/siesta/-/wikis/Guide-to-Siesta-versions for the different versions of siesta, and https://gitlab.com/siesta-project/siesta/-/wikis/How-to-build-the-master-version-of-Siesta for how to build it.). We need to save the electronic Kohn-Sham Hamiltonian in the atomic orbital basis set with the options:

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
