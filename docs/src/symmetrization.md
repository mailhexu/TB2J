# Symmetrization of Exchange Parameters

```{note}
This page documents the symmetry-based symmetrization of the magnetic
interaction parameters computed by TB2J: the isotropic exchange $J^{iso}$, the
symmetric anisotropic exchange $\mathbf{J}^{ani}$, the Dzyaloshinskii–Moriya
interaction (DMI) vector $\mathbf{D}$, and, when present, the single-ion
anisotropy (SIA) tensors $\mathbf{K}_i$.  Every rule quoted here is derived and
verified symbolically with [SymPy](https://www.sympy.org) in the derivation
report `docs/sympy/04_spacegroup_symmetrization.md` shipped in the TB2J source
tree.
```

## Overview

The exchange parameters obtained from a TB2J run are rarely *exactly*
symmetric.  The magnetic ground state breaks part of the crystal symmetry, and
numerical noise accumulated during the DFT calculation, the Wannierization (or
LOAO) procedure, and the post-processing leaves its trace in the parameters:
pairs of bonds that are related by a symmetry operation of the crystal come out
with slightly different values, and components that symmetry forbids come out
as small non-zero numbers instead of exact zeros.

The TB2J symmetrizer removes this noise by *projection*: it detects the
symmetry operations of the structure with
[spglib](https://spglib.readthedocs.io/), maps every stored bond onto its
symmetry images, and replaces the parameters of each orbit of symmetry-equivalent
bonds by their average.  Averaging over an orbit is a linear projection onto the
subspace of parameters invariant under the symmetry group, so the result is
exactly symmetric — not approximately.

Use the symmetrizer when the parameters feed downstream tools that assume the
symmetry (spin-dynamics codes, model construction, fitting), or whenever
symmetry-forbidden components should be exactly zero rather than "small".

Compared to the historical implementation (which grouped pairs by distance and
atomic tags, and symmetrized only the isotropic exchange), the current
symmetrizer:

1. uses the **full space group** of the cell, including the *translational*
   parts of the operations (lattice centering, non-symmorphic glide/screw
   translations), instead of a distance+tag heuristic;
2. optionally uses the **magnetic space group** (`--magnetic`), i.e. operations
   combined with time reversal, which is essential for antiferromagnets whose
   magnetic cell is a supercell of the chemical cell;
3. symmetrizes the interaction at the level of the **full pair tensor**, so
   $J^{iso}$, $\mathbf{J}^{ani}$ and $\mathbf{D}$ are treated consistently and
   all symmetry-forbidden components are **zero by construction**;
4. is backed by a SymPy-verified derivation (`docs/sympy/04_spacegroup_symmetrization.md`).

## Hamiltonian and conventions

TB2J uses the convention described in `docs/src/convention.rst`
("Conventions of Heisenberg Model"):

$$
E = -\sum_i \mathbf{S}_i^T \mathbf{K}_i \mathbf{S}_i
  - \sum_{i \neq j} \Big[
      J^{iso}_{ij}\, \mathbf{S}_i \cdot \mathbf{S}_j
      + \mathbf{S}_i \mathbf{J}^{ani}_{ij} \mathbf{S}_j
      + \mathbf{D}_{ij} \cdot \left( \mathbf{S}_i \times \mathbf{S}_j \right)
    \Big],
$$

with $\mathbf{J}^{ani}_{ij} = (\mathbf{J}^{ani}_{ij})^T$.  Positive $J^{iso}$
favours ferromagnetic alignment, the spins $\mathbf{S}_i$ are unit vectors, and
**every ordered pair is stored**: both $(ij)$ and $(ji)$ appear in the
parameter dictionaries.

It is convenient to collect the three pair channels into a single
$3\times 3$ **pair tensor**

$$
\mathbf{\Gamma}_{ij}(\mathbf{R}) = J^{iso}_{ij}\, \mathbf{I}
  + \mathbf{J}^{ani}_{ij} + \mathbf{A}(\mathbf{D}_{ij}),
  \qquad
  A^{\alpha\beta}(\mathbf{D}) = \sum_\gamma \epsilon^{\alpha\beta\gamma} D^\gamma,
$$

so that $\mathbf{S}_i^T \mathbf{\Gamma}_{ij}(\mathbf{R}) \mathbf{S}_j$
reproduces the three terms of the Hamiltonian for the pair $i \to j+\mathbf{R}$
($\mathbf{R}$ is a lattice vector of the cell used in the calculation).  The
stored data obeys the exact **reversal identity**

$$
\mathbf{\Gamma}_{ji}(-\mathbf{R}) = \mathbf{\Gamma}_{ij}(\mathbf{R})^T,
$$

which says that $J^{iso}$ and the symmetric part of $\mathbf{J}^{ani}$ are even
under bond reversal while $\mathbf{D}$ is odd.

Symmetrization acts on $\mathbf{\Gamma}$ and afterwards the three physical
channels are read off uniquely:

$$
J^{iso} = \frac{1}{3}\,\mathrm{tr}\,\bar{\mathbf{\Gamma}},
\qquad
\mathbf{J}^{ani} = \mathrm{sym}\,\bar{\mathbf{\Gamma}}
  - \frac{\mathrm{tr}\,\bar{\mathbf{\Gamma}}}{3}\mathbf{I},
\qquad
D^\gamma = \frac{1}{2}\sum_{\alpha\beta}
  \epsilon^{\gamma\alpha\beta}\,
  \mathrm{skew}(\bar{\mathbf{\Gamma}})^{\alpha\beta},
$$

where $\bar{\mathbf{\Gamma}}$ is the symmetrized tensor,
$\mathrm{sym}\,\mathbf{\Gamma} = (\mathbf{\Gamma}+\mathbf{\Gamma}^T)/2$ and
$\mathrm{skew}\,\mathbf{\Gamma} = (\mathbf{\Gamma}-\mathbf{\Gamma}^T)/2$.  The
decomposition is exact: applied to any tensor it reproduces that tensor
identically (asserted in the SymPy derivation and in the test suite).

## Symmetry operations and orbits

### Operations

The symmetrizer obtains the space-group operations
$g = \{W | \mathbf{t}\}$ of the **actual cell** stored in the TB2J results from
spglib (fractional/integer rotation $W$ and fractional translation
$\mathbf{t}$, in the basis of that cell).  Because the operations are those of
the real crystal — not a heuristic — this fixes the main weakness of the old
distance+tag grouping:

- pairs related **only through a translational part** (face/ body centering,
  non-symmorphic glides and screws) are correctly identified as equivalent;
- pairs that merely have the **same distance and tags** are *not* merged
  unless a true symmetry operation relates them — accidental degeneracies are
  not averaged into each other;
- the standard/default setting of the group is handled by spglib, so the
  operations are consistent for any cell choice.

### Site map and bond map

An operation $g = \{W|\mathbf{t}\}$ maps the atom at fractional position
$\mathbf{x}$ onto the (equivalent) atom containing $W\mathbf{x} + \mathbf{t}$
modulo 1 (within `symprec`).  A stored bond $(i, j, \mathbf{R})$, whose
fractional **bond vector** is

$$
\mathbf{d}_{ij}(\mathbf{R}) = \mathbf{x}_j + \mathbf{R} - \mathbf{x}_i,
$$

is mapped by $g$ to the bond $(i', j', \mathbf{R}')$ between the image atoms,

$$
\mathbf{d}' = W\mathbf{d},
\qquad
\mathbf{R}' = \mathbf{d}' - (\mathbf{x}_{j'} - \mathbf{x}_{i'})
\quad \text{(rounded to integers)}.
$$

The cartesian rotation acting on vectors is

$$
W_c = \mathbf{A}\, W\, \mathbf{A}^{-1},
$$

with $\mathbf{A}$ the $3\times 3$ cell matrix (lattice vectors as columns);
$\det W_c = \pm 1$.

### Transformation rules

| Quantity | Transformation under $g=\{W|\mathbf{t}\}$ |
| --- | --- |
| ordered bond $(i,j,\mathbf{R})$ | $\mapsto (i', j', \mathbf{R}')$ via the site/bond map |
| isotropic exchange $J^{iso}$ | invariant (scalar) |
| anisotropic exchange $\mathbf{J}^{ani}$ | $\mapsto W_c\, \mathbf{J}^{ani}\, W_c^T$ |
| DMI vector $\mathbf{D}$ | $\mapsto \det(W_c)\, W_c\, \mathbf{D}$ (axial vector) |
| pair tensor $\mathbf{\Gamma}$ | $\mapsto W_c\, \mathbf{\Gamma}\, W_c^T$ |
| SIA tensor $\mathbf{K}_i$ | $\mapsto W_c\, \mathbf{K}_i\, W_c^T$ (site mapped by $g$) |

In addition, the symmetrizer always applies **exchange reversal**, the exact
identity $\mathbf{\Gamma}_{ji}(-\mathbf{R}) = \mathbf{\Gamma}_{ij}(\mathbf{R})^T$
recalled above, so reversed images of a bond are related to it as well.

### Orbit averaging as projection

The set of stored keys $(\mathbf{R}, i, j)$ is partitioned into **orbits** under
the group generated by all operations (plus reversal).  Symmetrization replaces
every member of an orbit $O$ by the orbit average of the transformed tensors,

$$
\bar{\mathbf{\Gamma}}_{O} = \frac{1}{|O|}
  \sum_{(\mathbf{R},i,j)\,\in\, O}
  W_c\, \mathbf{\Gamma}_{ij}(\mathbf{R})\, W_c^{T}.
$$

This is a linear projection onto the invariant subspace of the group: applying
it twice changes nothing, and every orbit yields parameters that transform
exactly correctly under every operation.  If the stored bond list was truncated
by a cutoff (`Rcut`) some images of a key may be missing from the data; each
orbit is then averaged under exactly the operations that map it onto itself
(see [Caveats](#caveats-and-limitations)).

## Magnetic space group

### Primed operations

A magnetic operation $\{W|\mathbf{t}\}'$ combines the spatial operation with
**time reversal**: spins transform as $\mathbf{S} \rightarrow -W_c\mathbf{S}$.
spglib detects such operations when it is given the magnetic moments of the
sites (see below).

For **all quantities stored by TB2J** the spin flips cancel.  For a bilinear
term,

$$
\mathbf{S}_i^T \mathbf{\Gamma}' \mathbf{S}_j
\;\leftarrow\;
(-\mathbf{S}_i)^T W_c^T\, \mathbf{\Gamma}'\, W_c\, (-\mathbf{S}_j)
= \mathbf{S}_i^T\, W_c^T \mathbf{\Gamma}' W_c\, \mathbf{S}_j,
$$

and likewise for the rank-2 SIA tensor $\mathbf{K}_i$ — the two minus signs
cancel in both.  Invariance therefore requires exactly the *same* tensor
transformation as for the unprimed operation,
$\mathbf{\Gamma}' = W_c\,\mathbf{\Gamma}\,W_c^T$.  **Priming never changes the
tensor action; it only enlarges the set of valid site maps** — a site whose
spin is reversed by the operation can be mapped by the primed operation even
when the unprimed one is not a symmetry of the spin pattern.  This is proven
symbolically for the full Hamiltonian in
`docs/sympy/04_spacegroup_symmetrization.md`.

### Example: a primed half-translation kills the DMI

Consider a collinear antiferromagnet with sublattices A (spin $\uparrow$) at
$\mathbf{x}_A$ and B (spin $\downarrow$) at $\mathbf{x}_A + \mathbf{t}/2$, and
the magnetic operation $g = \{E|\mathbf{t}/2\}'$ (half translation plus time
reversal).  For the A–B bond stored as $(A, B, \mathbf{R}=0)$ with bond vector
$\mathbf{d} = \mathbf{t}/2$:

1. $g$ maps $(A,B,\mathbf{0})$ onto the parallel bond $(B, A, \mathbf{R}
   =\mathbf{t})$ with $W_c = I$, hence
   $\mathbf{\Gamma}_{BA}(\mathbf{t}) = \mathbf{\Gamma}_{AB}(\mathbf{0})$;
2. reversal contributes the transposed tensors to the same orbit:
   $\mathbf{\Gamma}_{BA}(\mathbf{0}) = \mathbf{\Gamma}_{AB}(\mathbf{0})^T$ and
   $\mathbf{\Gamma}_{AB}(-\mathbf{t}) = \mathbf{\Gamma}_{AB}(\mathbf{0})^T$;
3. the orbit average is therefore
   $\tfrac14\,(\mathbf{\Gamma} + \mathbf{\Gamma} + \mathbf{\Gamma}^T + \mathbf{\Gamma}^T)
   = \mathrm{sym}(\mathbf{\Gamma})$.

The antisymmetric part cancels **exactly**: $D_{AB} = 0$ by construction, while
$J^{iso}$ and $\mathbf{J}^{ani}$ survive.  Without the primed operation the
pairs $(A,B,\mathbf{0})$ and $(B,A,\mathbf{t})$ belong to different orbits and
the DMI on that bond is allowed.  This mechanism — primed half-translations
forcing DMI to vanish — is common in antiferromagnetic supercells.

### Ferromagnets and the gray group

For a ferromagnet one may include pure time reversal $\{E|\mathbf{0}\}'$ (the
"gray" operation).  Its site map is the identity and its tensor action is the
identity as shown above, so it generates no new orbits and changes no
parameter: symmetrizing with or without it gives **identical** results.  In
practice `--magnetic` is only *useful* when the magnetic cell contains
inequivalent spin orientations (typically antiferromagnets).

### Requirements

`--magnetic` needs the magnetic moments in the TB2J results (the `spinat`
array, exposed as `magmoms`), which standard TB2J workflows write together with
the exchange parameters:

- **collinear** results carry scalar moments (the sign of the $z$ component
  distinguishes the sublattices); they are passed to spglib as collinear
  magnetic moments;
- **noncollinear** results carry the full moment directions, which are passed
  to spglib as axial magnetic moments, and the detected primed operations are
  those compatible with the moment pattern.

If the results contain no spins, the magnetic path cannot be used; run the
crystallographic (space-group) symmetrization instead.

## What is forced to zero

### Master rules

For an operation that maps a bond **onto itself**, the invariance condition
$\mathbf{\Gamma}' = W_c\mathbf{\Gamma}W_c^T$ combined with the reversal identity
fixes the orientation channels of the bond:

- bond mapped with **reversed** orientation (endpoints exchanged,
  $\mathbf{d} \to -\mathbf{d}$):
  $$\det(W_c)\, W_c\, \mathbf{D} = -\mathbf{D};$$
- bond mapped with **preserved** orientation:
  $$\det(W_c)\, W_c\, \mathbf{D} = +\mathbf{D};$$
- in both cases $\mathbf{J}^{ani}$ must satisfy
  $W_c\,\mathbf{J}^{ani}\,W_c^T = \mathbf{J}^{ani}$.

These two lines generate Moriya's rules.  Note that for inversion
$\det(W_c)W_c = \mathbf{I}$, so the reversed-orientation line reads
$\mathbf{D} = -\mathbf{D}$: inversion at the bond midpoint forces
$\mathbf{D}=0$ (and, through $W_c\mathbf{\Gamma}W_c^T = \mathbf{\Gamma}^T$,
a symmetric pair tensor).

### Moriya rules for a bond

Let the bond point along $\hat{u}_{ij}$.  Each row is a direct corollary of the
master rules; all are derived symbolically in
`docs/sympy/04_spacegroup_symmetrization.md`.

| Operation leaving the bond invariant | Constraint on $\mathbf{D}_{ij}$ | Constraint on $\mathbf{J}^{ani}_{ij}$ |
| --- | --- | --- |
| inversion at the bond midpoint (primed or unprimed) | $\mathbf{D}_{ij} = 0$ | none beyond symmetry |
| mirror plane **containing** the bond | $\mathbf{D}_{ij} \perp$ mirror plane | tensor respects the mirror |
| mirror plane **perpendicular** to the bond, through the midpoint | $\mathbf{D}_{ij}$ lies in the mirror plane, i.e. $\mathbf{D}_{ij} \perp \hat{u}_{ij}$ | tensor respects the mirror |
| 2-fold axis **perpendicular** to the bond, through the midpoint | $\mathbf{D}_{ij} \perp$ that $C_2$ axis | tensor respects the rotation |
| 2-fold axis **along** the bond | $\mathbf{D}_{ij} \parallel \hat{u}_{ij}$ | diagonal in the bond frame |
| $n$-fold axis ($n \ge 3$) along the bond | $\mathbf{D}_{ij} \parallel \hat{u}_{ij}$ | **uniaxial**: $\mathrm{diag}(J_\perp, J_\perp, J_\parallel)$ about the bond |
| $C_4$ (or $S_4$) axis perpendicular to the bond | $\mathbf{D}_{ij} = 0$ | tensor respects the rotation |
| primed half-translation exchanging the two sublattices | $\mathbf{D}_{ij} = 0$ | none beyond symmetry |

DMI is therefore a fingerprint of **broken inversion symmetry on the bond**;
whenever an orbit is constrained to $\mathbf{D}=0$ (or to a specific
direction), the symmetrizer enforces it exactly.

### Zero snapping and the per-orbit report

After the orbit projection the symmetrizer **snaps** every component with
$|{\cdot}| <$ `zero_tol` (default $10^{-8}$, in the energy units of the stored
parameters — meV, as used in TB2J output) to exactly $0.0$.  Numerical dust
below the tolerance never survives into the output.

With `verbose=True` (the default) the symmetrizer prints a per-orbit report
listing, for each orbit, the averaged channels and **which components are
identically zero by symmetry**.  An illustrative example of the report:

```
======================================================================
Symmetrizing exchange parameters
  space group: Pnma (No. 62),  symprec = 1e-05 Ang
  magnetic operations: off
  96 bonds in 14 orbits
----------------------------------------------------------------------
orbit  3/14  (Fe1, Fe2)  R = (0, 0, 0)   multiplicity 8
  J_iso = -21.437118 meV
  D     = ( 0.000000,  0.000000, -0.412705) meV
    zero by symmetry: D_x, D_y
  J_ani =
    [  0.000000   0.013402   0.000000 ]
    [  0.013402  -0.227117   0.000000 ]
    [  0.000000   0.000000   0.227117 ]  meV
    zero by symmetry: ani_xx, ani_xz, ani_yz
----------------------------------------------------------------------
snapped 23 components below zero_tol = 1e-08 meV
======================================================================
```

(The exact layout of the numbers may differ slightly between versions; the
content — space group, orbit multiplicities, per-channel values and the
"zero by symmetry" lists — is what to look at.)

## Usage

### Command line

The script `TB2J_symmetrize.py` reads a TB2J results directory and writes the
symmetrized parameters to a new directory:

```bash
TB2J_symmetrize.py -i TB2J_results -o TB2J_results_symmetrized [-s symprec] [--Jonly] [--magnetic] [--zero-tol 1e-8]
```

| Flag | Meaning |
| --- | --- |
| `-i, --inpath` | input path to the TB2J results |
| `-o, --outpath` | output path for the symmetrized results (default `TB2J_results_symmetrized`) |
| `-s, --symprec` | precision for symmetry detection, in Å (default `1e-5`) |
| `--Jonly` | symmetrize only the isotropic exchange; DMI and anisotropic exchange are not symmetrized and are dropped from the output (legacy behaviour) |
| `--magnetic` | use the magnetic space group (time-reversed operations); requires the spins to be stored in the results |
| `--zero-tol` | snap threshold for symmetry-forbidden components, meV (default `1e-8`) |

Worked examples.

Ferromagnet — crystallographic space group of the cell, all channels:

```bash
TB2J_symmetrize.py -i TB2J_results -o TB2J_results_symmetrized
```

Collinear antiferromagnet whose magnetic cell is a supercell — use the
magnetic space group, with a slightly looser tolerance to absorb relaxation
noise:

```bash
TB2J_symmetrize.py -i TB2J_results_AFM -o TB2J_results_AFM_sym -s 1e-4 --magnetic
```

### Python API

The same functionality is available programmatically:

```python
from TB2J.symmetrize_J import TB2JSymmetrizer, symmetrize_J
from TB2J.io_exchange import SpinIO

# one-shot, from a results directory:
symmetrize_J(path="TB2J_results",
             output_path="TB2J_results_symmetrized",
             symprec=1e-5,
             Jonly=False,
             magnetic=False,
             zero_tol=1e-8)

# or, with full control:
exc = SpinIO.load_pickle(path="TB2J_results", fname="TB2J.pickle")
symmetrizer = TB2JSymmetrizer(exc,
                              symprec=1e-5,
                              verbose=True,
                              Jonly=False,
                              magnetic=False,
                              zero_tol=1e-8)
symmetrizer.symmetrize_J()          # symmetrize (in place on an internal copy)
symmetrizer.output("TB2J_symmetrized")  # write results
# symmetrizer.run("TB2J_symmetrized") does both steps at once.
```

Signatures:

```python
TB2JSymmetrizer(exc, symprec=1e-5, verbose=True, Jonly=False,
                magnetic=False, zero_tol=1e-8)

symmetrize_J(exc=None, path=None, fname="TB2J.pickle", symprec=1e-5,
             output_path="TB2J_symmetrized", Jonly=False, magnetic=False,
             zero_tol=1e-8)
```

### Symmetrizing to a target structure

To force the parameters to obey the symmetry of a chosen (possibly idealized)
structure rather than the symmetry found in the results, use

```python
from ase.io import read
from TB2J.symmetrize_J import symmetrize_exchange

symmetrize_exchange(spinio, atoms, symprec)
```

where `atoms` is an `ase.Atoms` object defining the target symmetry: pairs
equivalent under the symmetry detected in `atoms` are averaged.  For example,
reading a cubic CIF symmetrizes the isotropic exchange to cubic symmetry, while
passing `spinio.atoms` averages within the groups found for the original
structure.  Only the isotropic channel is modified by this helper; DMI and
anisotropic exchange are left untouched, and the structure stored in `spinio`
is not modified.  The `TB2J_edit.py symmetrize` subcommand wraps the same
function.

### Reading the verbose report

- The **space group line** tells you which symmetry spglib detected at the
  requested `symprec` — check that it matches the space group you expect for
  the structure.
- The **orbit list** shows how the stored bonds were grouped; multiplicities
  smaller than expected usually mean the bond list was truncated by a cutoff
  or `symprec` is too tight.
- The **zero-by-symmetry lists** document which DMI / anisotropic components
  the symmetry forbids; after symmetrization they are exactly zero.
- The **snapped count** reports how many components were below `zero_tol` and
  set to exactly zero.

## Caveats and limitations

- **`symprec` choice.** Too small a value misses symmetry (under-symmetrized
  results); too large a value merges genuinely inequivalent bonds.  The
  default $10^{-5}$ Å suits idealized cells; relaxed cells often need
  $10^{-3}$–$10^{-2}$ Å.  Always check the reported space group.
- **Truncated orbits.** If the stored interactions were cut off at `Rcut`, an
  operation may map some bonds outside the stored set.  Orbits are then the
  connected components of the "bond -> present image" graph, and each orbit is
  averaged under exactly the operations that map it onto itself (its set-wise
  stabilizer): partially defined operations are excluded per orbit instead of
  being applied unevenly, so the result remains exactly invariant under every
  retained operation.  A bond-distance cutoff dataset as produced by TB2J is
  closed under all operations up to boundary rounding, so in practice no
  operation is lost.
- **Averaging over available data.** Relatedly, tensors are never *extrapolated*;
  only bonds actually stored in the results are averaged.
- **Magnetic group for noncollinear systems.** With `--magnetic`, the stored
  spin vectors are passed to spglib as axial magnetic moments and the detected
  operations are those compatible with the spin pattern; as everywhere, the
  primed operations only enlarge the site maps and never alter the tensor
  transformation rules.
- **SIA.** Single-ion anisotropy tensors are symmetrized only when present in
  the results (`has_sia_tensor`); they are not produced by Wannier90-based
  workflows (they require the SOC part of the Hamiltonian, e.g. from
  constrained DFT).
- **`--Jonly`.** Kept for backward compatibility with the historical
  behaviour: it restricts the symmetrization to the isotropic channel and
  drops DMI/anisotropic exchange from the output.  The default is now to
  symmetrize and retain **all** channels.

## References

1. T. Moriya, *Anisotropic Superexchange Interaction and Weak Ferromagnetism*,
   Phys. Rev. **120**, 91 (1960),
   <https://doi.org/10.1103/PhysRev.120.91> — the original statement of the
   symmetry rules for $\mathbf{D}_{ij}$ used above.
2. A. Togo, K. Shinohara and I. Tanaka, *Spglib: a software library for crystal
   symmetry search*, arXiv:2407.10725 (2024),
   <https://arxiv.org/abs/2407.10725>; see also the spglib documentation at
   <https://spglib.readthedocs.io/> — space-group and magnetic-space-group
   detection.
3. I. Dzyaloshinsky, *A thermodynamic theory of "weak" ferromagnetism of
   antiferromagnetics*, J. Phys. Chem. Solids **4**, 241 (1958),
   <https://doi.org/10.1016/0022-3697(58)90076-3>.
4. Full symbolic derivation of every rule on this page:
   `docs/sympy/04_spacegroup_symmetrization.md` in the TB2J source tree —
   space-group orbit symmetrization of the pair tensor, the cancellation of the
   time-reversal spin flips, the primed half-translation example, and the
   Moriya-rule table.
