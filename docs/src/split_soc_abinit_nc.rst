ABINIT NC split-SOC exchange
============================

The ABINIT norm-conserving (NC) split-SOC workflow reconstructs magnetic
exchange from a *single* collinear, spin-orbit-free strength-zero reference
whose PAO basis comes from the pseudopotential's pseudo-atomic orbitals
(via :py:mod:`pypao`).  It consumes **two** stored artifacts and joins them
by content hash before any physics runs:

1. an ``abinit.nc_pao_hs`` **v2** projection file written by
   ``abinao project-pao`` from the strength-0 WFK: PAO projections
   :math:`C[s,k,n,p]=\langle\phi_p|\psi_{kns}\rangle`, the k-dependent PAO
   overlap :math:`S(k)`, and the collinear spin-splitting operators
   (``delta_total``, ``delta_xc_smooth``, and the NC DFT+U splitting) in
   the PAO basis.  The file stores Hartree on disk and is normalized to eV
   in memory.
2. an ``abinao.nc_soc_ks`` **v1** SOC sidecar written by
   :py:mod:`abinao.soc_kernel`: the all-atom band-window SOC operator
   :math:`W^k_{SO}` for the x/y/z legs in the composite spinor basis
   :math:`i = 2n + \sigma`, in eV, Hermitian, in full-BZ WFK order, with
   the SHA-256 provenance of its sources.

The production chain is therefore::

   ABINIT strength-0 collinear WFK (+VXC)
        → abinao project-pao (pypao pseudo-atomic orbitals)  → abinit.nc_pao_hs v2
        → abinao soc_kernel (pseudo SO channels, three spinaxis legs)
             → abinao.nc_soc_ks v1 sidecar
        → TB2J hash join → PAO dualization → three x/y/z legs
        → lattice-frame rotation → three-leg merge

Each leg applies one spinaxis direction of the unit-strength SOC operator
to the same strength-zero bands and the same frozen density; the collinear
PAO spin-splitting vertex stays on the selected magnetic sites.  Legs are
rotated to the lattice frame with
:math:`T_\mathrm{lattice} = O\, T_\mathrm{leg}\, O^T`
(:math:`O\mathbf e_z` = leg ``spinaxis``) and merged.  The SOC operator
enters only the band propagator — it is never used as a vertex.

Strength-zero recipe
--------------------

1. Run a **collinear, norm-conserving** ground state and NSCF
   (``so_psp 0``, ``nspinor 1``, ``nsppol 2``) with an explicit
   full-Brillouin-zone k-point list (``kptopt 0``), writing the wavefunction
   and the XC potential (``prtwf 1``, ``prtvxc 1``, ``iomode 3``).  The
   consumer refuses anything but the full-BZ gauge: an IBZ-symmetry-reduced
   projection set cannot be paired with the sidecar's full-BZ operator
   order.  (A molecule-in-box calculation with a single Γ point already
   satisfies this — Γ *is* the full mesh of a 1×1×1 grid.)

2. Export the PAO projections and spin-splitting operators:

   .. code-block:: bash

      abinao project-pao --wfk runo_WFK.nc --vxc runo_VXC.nc \
          --orb I.upf --out PAO_HS.nc --ibz-to-bz

3. Compute the three-leg SOC kernel sidecar with the abinao Python API
   (one independent ``SocKernelResult`` per spinaxis leg from the *same*
   WFK, then the versioned writer):

   .. code-block:: python

      from abinao.soc_kernel import compute_wfk_soc_kernel, write_nc_soc_kernel
      from pypao.libpsp import read_orbitals

      orbitals = read_orbitals(["I.upf"])
      legs = {}
      for axis in ("x", "y", "z"):
          spinaxis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
          legs[axis] = compute_wfk_soc_kernel(
              wfk, pseudo_by_species={"I": orbitals[0].pseudo},
              spinaxis=spinaxis, spnorbscl=1.0,
          )
      write_nc_soc_kernel(
          "nc_soc_ks.nc", legs, wfk_path="runo_WFK.nc",
          pao_hs_path="PAO_HS.nc",
      )

   The writer refuses a non-unit ``spnorbscl`` (the sidecar stores the
   :math:`\lambda=1` kernel; scaling happens with ``lam`` at the consumer,
   never inside the stored operator), requires all-atom coverage (ligand
   SOC enters the propagator), and binds the SHA-256 of the WFK and — when
   given — the PAO_HS file into the provenance.

4. Run the TB2J consumer (below) and converge the k-point set, band
   window, ``nz`` and ``smearing``; each leg reports its own measured
   even-prefix band-window study.

File contract and hash join
---------------------------

The sidecar on-disk contract is summarized in
:doc:`abinit_savetb2j_schema` (section *ABINIT NC split-SOC sidecar*).
The TB2J loader validates the schema name/version, the ``eV``
``energy_unit`` (with a :math:`10^{-12}` relative check of the stored
``hartree_to_ev`` factor), the Hermiticity of every :math:`W^k_{SO}`
matrix, the non-negative full-BZ k-weights, one ``spinaxis`` per leg, and
that every leg matrix is finite and nonzero.  Files violating any of these
rules are rejected at load time.

The pairing gate then refuses:

* a PAO_HS file whose SHA-256 differs from the sidecar's recorded
  ``pao_hs_sha256`` — the hash join is mandatory; a sidecar without that
  attribute is refused as an unverifiable pairing;
* a strength-0 WFK whose SHA-256 differs from the sidecar provenance,
  when ``--wfk`` is given (recommended);
* any k-point order/gauge mismatch between the two artifacts, or
  k-weight mismatch (one shared BZ gauge is required);
* sidecar eigenvalues disagreeing with the PAO_HS band energies by more
  than :math:`10^{-6}` eV;
* a composite band count different from :math:`2 \times` the PAO_HS
  ``nband``;
* a **spinor-flavor sidecar** (``nsppol=1``, ``nspinor=2``, contracted
  spinor bands): its :math:`W` is not elementwise comparable to a
  collinear PAO_HS-side matrix, and the error message routes it to a
  spinor-flavor consumer.  This consumer is collinear-only
  (``nsppol=2``, ``nspinor=1``, composite :math:`2n+\sigma` basis).

PAO dualization
---------------

NC PAO projectors are nonorthogonal, so the consumer dualizes the PAO
maps :math:`B(k) = S(k)^{-1} C(k)` — the overlap is *used*, never dropped
by fiat — and feeds the normalized spinor band window to the split-SOC
kernel with ``overlap_k=None`` (the dual basis is already covariant).
Dropping :math:`S` silently changes the projected exchange; the dualized
path replays the inverse-overlap identity exactly in the test suite.

Units
-----

The stored sidecar carries ``energy_unit = "eV"`` and a ``hartree_to_ev``
factor that is validated against the internal constant; the PAO_HS file
stores Hartree on disk and is normalized to eV in memory.  All consumer
outputs, gates and tolerances are eV; ``lam`` (CLI ``--scale``) is a
dimensionless multiplier applied at the kernel level.

Driver usage
------------

.. code-block:: python

   from TB2J.interfaces.abinit_nc_split_soc import (
       gen_exchange_abinit_nc_split_soc,
   )

   gen_exchange_abinit_nc_split_soc(
       "PAO_HS.nc",              # abinit.nc_pao_hs v2
       "nc_soc_ks.nc",           # abinao.nc_soc_ks v1 sidecar
       wfk="runo_WFK.nc",        # optional but recommended (hash check)
       output_path="TB2J_results_nc_split_soc",
       index_magnetic_atoms=[0, 1],   # zero-based Python indices
       Rcut=10.0,
       nz=30,
       smearing_eV=0.05,
   )

The same run from the command line:

.. code-block:: bash

   abinit_nc_split_soc2J.py --pao_hs PAO_HS.nc --soc_kernel nc_soc_ks.nc \
       --wfk runo_WFK.nc --output_path TB2J_results_nc_split_soc \
       --index_magnetic_atoms 1 2 --Rcut 10.0 --nz 30 --smearing 0.05

.. list-table:: Selected options (CLI indices are 1-based; the Python API is 0-based)
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``--pao_hs`` / ``--soc_kernel``
     - The two required artifacts; joined by SHA-256 before any physics.
   * - ``--wfk``
     - Strength-0 WFK path; its SHA-256 is checked against the sidecar.
   * - ``--index_magnetic_atoms``
     - 1-based magnetic sites carrying the collinear vertices; required
       whenever the cell has nonmagnetic ligands.  One of
       ``--index_magnetic_atoms`` / ``magnetic_elements`` is mandatory —
       the sidecar SOC covers every atom, so silently selecting all sites
       would treat every ligand as magnetic.
   * - ``--vertex_component``
     - ``delta_total`` (default), ``delta_xc_smooth`` or
       ``spectral_spin_split``; the SOC propagator is never a vertex.
   * - ``--scale``
     - Dimensionless SOC scaling ``lam`` of the second-variation spectrum.
   * - ``--window_prefixes``
     - Even composite-band prefixes for the per-leg window convergence
       study (default: :math:`2b-2` and :math:`2b`).
   * - ``--tangent_tol`` / ``--no-verify-tangent-projection``
     - FR-032 tangent-block tolerance in eV; the default (``None``) is the
       :math:`10^{-2}` eV specification value on both the CLI and the
       Python API, and ``tol_source`` records which bound a report used.
       It is never derived from the merge's own spread.  Opt-out (not
       recommended).
   * - ``merge_consistency_atol`` (Python API)
     - Bound on the repeated-diagonal spread across references; strict
       default :math:`10^{-6}` eV (refusal).  A larger value is the
       explicit option-b that publishes the tensor with its measured
       reference-anisotropy spread.
   * - ``--anchor_rtol`` / ``--no-soc-off-anchor``
     - SOC-off anchor relative-Jiso tolerance (default :math:`10^{-7}`),
       and its opt-out (not recommended).

Outputs and provenance
----------------------

Each leg is written to ``leg_x/``, ``leg_y/``, ``leg_z/`` as the raw
rotated transverse blocks ``split_soc_leg.npz`` (``J_leg`` per
``(R, i, j)`` pair — never per-leg scalar :math:`J_\mathrm{iso}`/DMI/Jani,
which a single reference cannot determine) plus
``split_soc_provenance.json``; a successful three-leg merge writes the
rank-nine merged results to ``output_path``.  Every leg records the
FR-050/ADR-8 provenance block:
strength-0 reference (PAO_HS path + SHA-256, WFK name + SHA-256), sidecar
schema and operator source with explicit ``units: "eV"``, the spin frame
(``spinaxis`` and the resulting lattice-frame axis), the magnetic vertex
sites and component, the even-prefix band-window study with its measured
change, tolerance and ``converged`` flag, the merge mode
``three_leg_rotate_merge``, and the full pairing report.  The merged
provenance keeps the three leg records (each with its frame report and
rotation residual) and adds the ``merge_diagnostics`` block (design-matrix
rank, worst repeated-row deviation, transverse mask and reciprocity
residuals).

.. warning::

   Per-leg scalar ``Jiso``/``DMI``/``Jani`` decompositions of a
   single-reference leg are **not** physical observables; see the next
   section.  Per-leg artifacts store only the raw transverse ``J_leg``
   blocks; consume the driver's return value and the provenance JSON, not
   the per-leg ``npz`` internals.

Single-reference scope: what one leg can and cannot determine
-------------------------------------------------------------

A single collinear magnetic reference axis determines only the raw
transverse :math:`2\times2` block :math:`(J_{xx}, J_{xy}, J_{yx}, J_{yy})`
and :math:`D_z` of that leg.  Each leg's provenance carries this as an
explicit ``quantity_scope`` statement, and the merged provenance records
that the full rank-9 exchange tensor is final **only** after merging three
SU(2)-rotated magnetic references (x, y and z reference axes).  That merge
exists (``TB2J.split_soc_kernel.merge_transverse_legs``) and it *certifies
itself per fixture*: its repeated-row invariance gate can refuse a given
three-leg set, and a refused merge means no exchange number may be quoted
at all (the iodine dimer below is the standing example).

Gates
-----

Three gates run by default and are recorded in the provenance; the
opt-outs below are explicit and not recommended.

* **SOC-off collinear anchor**: with ``lam=0`` the consumer's spinor replay
  must reproduce the existing collinear NC PAO exchange kernel
  (``compute_projector_exchange_jdict``) shell for shell on the same
  strength-0 data (default relative tolerance :math:`10^{-7}`, plus a
  near-zero DMI norm check).  Disable with ``--no-soc-off-anchor`` /
  ``soc_off_anchor=False``; retune with ``--anchor_rtol``.
* **Rank-9 merge invariance gate** (always on, not optional): the three
  rotated legs must jointly determine the raw lattice tensor —
  ``merge_transverse_legs`` requires design-matrix rank 9 (three
  independent reference axes) and that every *repeated* diagonal row
  (each diagonal entry is measured in two different legs) agrees within
  ``merge_consistency_atol`` (:math:`10^{-6}` eV in the NC driver).
  A refusal aborts the driver before any merged tensor exists; see the
  iodine dimer below.  A larger ``merge_consistency_atol`` is a
  deliberate, diagnostics-visible choice that publishes the merged tensor
  together with the measured spread as an explicit reference-anisotropy
  error bar — never a silent default.
* **FR-032 tangent projection gate**: after a successful merge, the merged
  raw transverse block ``[:2, :2]`` must match the z one-shot leg (whose
  lattice-frame rotation is the identity) within the documented tolerance.
  The default is the :math:`10^{-2}` eV specification value (CLI and
  Python API alike; ``tol_source`` records whether a report used the spec
  default or an explicit value).  The tolerance is deliberately *not*
  derived from the merge's own repeat spread — a bound taken from the
  very inconsistency under test cannot fail.  This is a *projection-only*
  consistency gate — it is never a full-tensor certificate.  Disable with
  ``--no-verify-tangent-projection``; retune with ``--tangent_tol`` /
  ``tangent_tol_eV``.

The band-window study is the fourth, always-on diagnostic: even composite
prefixes (:math:`2b-2`, :math:`2b` by default) are compared per leg, and a
``converged: false`` flag is an unresolved error bar that calls for a
larger sidecar band window, never for a favourable error bar.

Real iodine dimer (I₂): anchor certified, tensor **not** certified
------------------------------------------------------------------

The retained stretched iodine dimer fixture
(bond :math:`7.5\,a_0` in an :math:`18\,a_0` box, Hund-state ``spinat``
:math:`(0,0,1)` on both atoms, 14 bands/spin → 28 composite states,
Γ-point :math:`1\times1\times1` mesh) is **not a certified exchange
fixture** under the rank-nine core.  Its verified status:

* **SOC-off anchor — passes.**  Max relative Jiso deviation
  :math:`4.0\times10^{-15}` over 12 R-pairs (a shorter-cutoff recording:
  :math:`1.7\times10^{-15}`); DMI norms at the :math:`10^{-18}` eV level.
  The strength-zero replay itself is exact.
* **Rank-9 merge — refuses.**  The three legs measure each diagonal entry
  twice (repeated rows of the design matrix).  On this fixture the two
  measurements disagree by up to :math:`2.495\times10^{-2}` eV (worst
  pair ``((-1,0,0), 0, 0)``) against the :math:`10^{-6}` eV
  ``merge_consistency_atol``, so ``merge_transverse_legs`` raises and the
  driver aborts: **no merged tensor, no tangent-gate outcome, no
  :math:`J_\mathrm{iso}`/DMI/Jani exists for this fixture.**  The
  refusal is uniform (24.95–25.99 meV across all 24 non-onsite pairs).
  Root cause (characterized 2026-09-28, not a defect): the legs are
  individually window-converged (:math:`26\to28`-band changes
  :math:`\le 2\times10^{-5}` eV) and the sidecar legs are faithful
  producer outputs (bit-exact recompute, Hermitian to
  :math:`10^{-16}`), but the strongly split collinear reference
  (:math:`|\Delta|` up to 0.72 eV/band) makes the three magnetization
  axes physically inequivalent references for this anisotropic single
  dimer — substituting the *exact* SU(2) projected images of the z-leg
  still refuses (spread :math:`3.03\times10^{-2}` eV, worse).  The
  three-reference rank-nine merge is certified only where the three
  references are symmetry-equivalent (fcc Ni: passes at
  :math:`9\times10^{-8}` eV); for dimer oracles use a single-reference
  full-tensor method, or pass a larger ``merge_consistency_atol`` as the
  explicit, reported option-b that publishes the tensor with its measured
  reference-anisotropy spread.  Do not raise the default.
* **Band window (FR-050)**: the default :math:`2b-2 \to 2b` study moves by
  :math:`1.7\times10^{-5}` eV at a :math:`10^{-6}` eV tolerance, i.e.
  ``converged: false`` — production use of this fixture needs a larger
  sidecar band window, and the flag stays visible in the per-leg
  provenance.

An earlier, pre-cutover recording quoted a passing FR-032 tangent residual
(:math:`7.1` meV against a 10 meV tolerance) computed against the legacy
scalar merge; that path is retired.  Under the rank-9 code the
forced-lstsq projection deviation on this fixture is **13.0 meV — above
the 10 meV tolerance**, so the projection gate fails too.  An interim
policy (c349a10) derived the gate bound from the merge's own repeat
spread, which let this gate pass by construction; it was reverted the
same day.  Every exchange-level statement about this dimer stays
uncertified; only the anchor and the per-leg diagnostics stand.

Runnable example
----------------

``examples/projector_green/abinit_nc_i2_split_soc.py`` runs the three-leg
driver on a real ``abinit.nc_pao_hs`` v2 + ``abinao.nc_soc_ks`` v1 pair
and prints the pairing, the anchor and merge gate reports and the output
paths.  On the current retained fixture the rank-nine merge refuses
(repeated-row spread ≈ 25 meV against the :math:`10^{-6}` eV default),
so the script prints the refusal, points at the per-leg artifacts and
exits nonzero — that refusal is the documented behaviour, not a crash:

.. code-block:: text

   REFUSED: leg diagonals disagree for pair ((-1, 0, 0), 0, 0): max spread 2.495e-02 exceeds consistency_atol 1.0e-06

Run it against the TB2J checkout (the interface module is not part of a
released TB2J yet):

.. code-block:: bash

   PYTHONPATH=/path/to/TB2J-checkout python examples/projector_green/abinit_nc_i2_split_soc.py \
       --pao_hs PAO_HS_i2.nc \
       --soc_kernel nc_soc_ks_i2.nc \
       --wfk iodine_i2_upf_so0o_WFK.nc \
       --index-magnetic-atoms 1 2 \
       --output TB2J_results_nc_split_soc

The script exits nonzero if an enabled gate fails.  It prints only gate
reports and provenance summaries — never per-leg Jiso/DMI/Jani as final
observables.  The ABINIT and abinao inputs that produce the fixture are
described in the strength-zero recipe above.
