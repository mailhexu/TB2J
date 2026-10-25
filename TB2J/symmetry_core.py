"""Shared space-group and magnetic-space-group symmetry engine for TB2J.

MathDoc: tb2j/spacegroup-exchange-symmetrization (adjacent math authority).

This module owns the symmetry operations of a structure and the geometric
seams built on them:

* ``SymmetryOperations`` holds the operations in fractional coordinates:
  integer rotations ``W``, fractional translations ``t`` and the optional
  time-reversal (priming) flags.
* ``crystal_symmetry_ops`` / ``magnetic_symmetry_ops`` detect the
  crystallographic and the magnetic space group with spglib.
* ``_build_site_maps`` maps every atom to its image atom under every
  operation; ``_bond_image`` maps a stored bond key to its image key;
  ``pair_images`` bundles the per-operation bond images with the matching
  Cartesian rotations.
* ``_collect_atom_keys`` / ``_atom_tensor`` / ``_impose_reversal_identity``
  build the working pair tensors of the stored data.
* ``_find_orbits`` groups the stored keys into orbits under the forward
  operation images.
* ``combine_tensor`` / ``decompose_tensor`` / ``d_to_skew`` / ``skew_to_d``
  convert between the full pair tensor and its (J^iso, D, J^ani) channels.

Conventions (docs/src/convention.rst): the energy is written as

    E = -sum_i S_i^T K_i S_i - sum_{i!=j} [ J^iso_ij S_i.S_j
                                           + S_i J^ani_ij S_j
                                           + D_ij . (S_i x S_j) ]

Every ordered pair (ij and ji) is stored separately.  For each stored key
(R, i, j) the full pair tensor is

    Gamma_ij(R) = J^iso I + J^ani + A(D),   A^{ab} = eps^{abg} D^g ,

with the reversal identity Gamma_ji(-R) = Gamma_ij(R)^T.

Symmetry detection uses the exact operations of the structure stored in the
SpinIO object (rotations AND translations), obtained from spglib:

* crystallographic space group (default): spglib.get_symmetry_dataset on
  (lattice, scaled_positions, numbers).  This includes pure lattice and
  centering translations, non-symmorphic parts, and the true setting of the
  input cell.
* magnetic space group (magnetic=True): spglib.get_magnetic_symmetry_dataset
  on (lattice, positions, numbers, magmoms), which additionally returns
  time_reversals.  For all terms stored in TB2J (bilinear pair tensors and
  rank-2 single-ion tensors) the two spin flips of a primed operation cancel,
  so the tensor action is Gamma' = W_c Gamma W_c^T regardless of priming;
  priming only changes which site maps are admitted (spin-consistent maps).

The projection of these operations onto the stored exchange data lives in
``TB2J.symmetrize_J``.
"""

import numpy as np


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_skew_form
def _levi_civita():
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return eps


_EPS = _levi_civita()


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_skew_form
def d_to_skew(D):
    """Skew-symmetric matrix A with A^{ab} = eps^{abg} D^g (so S_i^T A S_j = D.(S_i x S_j))."""
    D = np.asarray(D, dtype=float)
    return np.einsum("abg,g->ab", _EPS, D)


# MathDoc: tb2j/spacegroup-exchange-symmetrization#reversal_dmi_axial_vector_negated
def skew_to_d(S):
    """DMI vector D^g = 1/2 eps^{gab} S^{ab} from a skew-symmetric matrix."""
    S = np.asarray(S, dtype=float)
    return 0.5 * np.einsum("gab,ab->g", _EPS, S)


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
def combine_tensor(Jiso=None, D=None, Jani=None):
    """Gamma = J^iso I + Jani + A(D); missing channels contribute zero."""
    G = np.zeros((3, 3), dtype=float)
    if Jiso is not None:
        G += np.eye(3) * float(Jiso)
    if Jani is not None:
        G += np.asarray(Jani, dtype=float)
    if D is not None:
        G += d_to_skew(D)
    return G


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
def decompose_tensor(G):
    """Decompose a pair tensor into (J^iso, D, J^ani); combine_tensor inverts it exactly."""
    G = np.asarray(G, dtype=float)
    Jiso = np.trace(G) / 3.0
    D = skew_to_d(0.5 * (G - G.T))
    Jani = 0.5 * (G + G.T) - np.eye(3) * Jiso
    return Jiso, D, Jani


def _snap_zero(array, zero_tol):
    array = np.asarray(array, dtype=float)
    array[np.abs(array) < zero_tol] = 0.0
    return array


class SymmetryOperations:
    """Container for symmetry operations in fractional coordinates.

    rotations: (nops, 3, 3) integer matrices W (fractional basis).
    translations: (nops, 3) fractional translations t.
    time_reversals: (nops,) bool array, or None for a purely crystallographic
        group. Per the pinned contract, priming does not change the tensor
        action on any term stored in TB2J; it only certifies that the site
        map is spin-consistent.
    description: human readable group identification for verbose output.
    """

    def __init__(self, rotations, translations, time_reversals=None, description=""):
        self.rotations = np.asarray(rotations, dtype=int)
        self.translations = np.asarray(translations, dtype=float) % 1.0
        self.time_reversals = (
            None if time_reversals is None else np.asarray(time_reversals, dtype=bool)
        )
        self.description = description

    def __len__(self):
        return len(self.rotations)


def _get_spglib():
    try:
        import spglib
    except ImportError as e:
        raise ImportError(
            "spglib is required for symmetrization. Install it with: pip install spglib"
        ) from e
    return spglib


def crystal_symmetry_ops(atoms, symprec=1e-5):
    """Detect the crystallographic space group of an ASE atoms object."""
    spglib = _get_spglib()
    cell = (
        np.asarray(atoms.get_cell().array, dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise ValueError(
            "spglib could not detect the space group of the structure. "
            "Check that the structure is valid and try adjusting symprec."
        )
    description = (
        f"crystal space group {dataset.international} (No. {dataset.number}), "
        f"{len(dataset.rotations)} symmetry operations"
    )
    return SymmetryOperations(
        dataset.rotations, dataset.translations, None, description
    )


# MathDoc: tb2j/spacegroup-exchange-symmetrization#primed_bilinear_energy_difference
def magnetic_symmetry_ops(atoms, magmoms, symprec=1e-5, *, magnetic_tol=None):
    """Detect the magnetic space group of an ASE atoms object with moments.

    magmoms: (natom,) for collinear moments (taken along z) or (natom, 3)
        for non-collinear moments, aligned with the atoms order.
    magnetic_tol: optional magnetic-moment tolerance in Bohr magnetons.
        None preserves spglib's default (the structural symprec value).
    """
    spglib = _get_spglib()
    cell = (
        np.asarray(atoms.get_cell().array, dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
        np.asarray(magmoms, dtype=float),
    )
    dataset = spglib.get_magnetic_symmetry_dataset(
        cell,
        symprec=symprec,
        mag_symprec=-1.0 if magnetic_tol is None else magnetic_tol,
    )
    if dataset is None:
        raise ValueError(
            "spglib could not detect the magnetic space group of the structure. "
            "Check the structure and the magnetic moments, or adjust symprec."
        )
    msgtype = spglib.get_magnetic_spacegroup_type(dataset.uni_number)
    nprimed = (
        int(np.sum(dataset.time_reversals)) if dataset.time_reversals is not None else 0
    )
    description = (
        f"magnetic space group {msgtype.bns_number} (BNS, No. {msgtype.number}, "
        f"type {msgtype.type}), {len(dataset.rotations)} operations "
        f"({nprimed} with time reversal)"
    )
    return SymmetryOperations(
        dataset.rotations, dataset.translations, dataset.time_reversals, description
    )


def _build_site_maps(xfrac, cell, ops, symprec):
    """Map each atom to its image atom under every operation.

    The image position W x + t (mod 1) must coincide with an actual atom
    position within symprec (Cartesian, Angstrom). Returns a list of integer
    arrays (natom,), one site map per operation.
    """
    natom = len(xfrac)
    site_maps = []
    for W, t in zip(ops.rotations, ops.translations):
        images = (xfrac @ np.asarray(W).T + t) % 1.0
        smap = np.empty(natom, dtype=int)
        for a in range(natom):
            dfrac = images[a] - xfrac
            dfrac -= np.round(dfrac)
            dist = np.linalg.norm(dfrac @ cell, axis=1)
            b = int(np.argmin(dist))
            if dist[b] > symprec:
                raise ValueError(
                    f"Symmetry operation maps atom {a} to no partner: nearest image "
                    f"atom is {dist[b]:.3e} Ang away (symprec={symprec})."
                )
            smap[a] = b
        if len(set(smap.tolist())) != natom:
            raise ValueError(
                "Symmetry operation gives a non one-to-one site map; "
                "the structure and symprec are inconsistent."
            )
        site_maps.append(smap)
    return site_maps


# MathDoc: tb2j/spacegroup-exchange-symmetrization#bond_map_image_shift
def _bond_image(W, smap, xfrac, ia, ja, R):
    """Image of the bond key (ia, ja, R) under the operation (W, t, site map).

    The fractional bond vector d = x_j + R - x_i transforms as d' = W d and
    the cell index follows R' = d' - (x_j' - x_i'), rounded to integers.
    """
    d = xfrac[ja] + np.asarray(R, dtype=float) - xfrac[ia]
    d2 = np.asarray(W, dtype=float) @ d
    ia2, ja2 = int(smap[ia]), int(smap[ja])
    diff = d2 - (xfrac[ja2] - xfrac[ia2])
    R2 = np.round(diff)
    if np.max(np.abs(diff - R2)) > 0.1:
        raise ValueError(
            f"Bond ({ia}, {ja}, {R}) has a non-integer image cell under a "
            f"symmetry operation; check the structure and symprec."
        )
    return (ia2, ja2, tuple(int(x) for x in R2))


def _collect_atom_keys(exc):
    """Union of atom-level keys (ia, ja, R) over all stored pair channels."""
    if exc.exchange_Jdict is None:
        raise ValueError(
            "The SpinIO object has no exchange_Jdict; there is nothing to symmetrize."
        )
    keys = set()
    for R, i, j in exc.exchange_Jdict:
        keys.add((exc.iatom(i), exc.iatom(j), tuple(R)))
    for dct in (exc.dmi_ddict, exc.Jani_dict):
        if dct:
            for R, i, j in dct:
                keys.add((exc.iatom(i), exc.iatom(j), tuple(R)))
    return sorted(keys)


def _atom_tensor(exc, ia, ja, R):
    """Full pair tensor of a key built from its channels (missing channels = 0)."""
    i, j = exc.index_spin[ia], exc.index_spin[ja]
    key = (tuple(R), i, j)
    Jiso = D = Jani = None
    if exc.exchange_Jdict is not None and key in exc.exchange_Jdict:
        Jiso = float(np.real(exc.exchange_Jdict[key]))
    if exc.dmi_ddict is not None and key in exc.dmi_ddict:
        D = np.asarray(exc.dmi_ddict[key], dtype=float)
    if exc.Jani_dict is not None and key in exc.Jani_dict:
        Jani = np.asarray(exc.Jani_dict[key], dtype=float)
    return combine_tensor(Jiso, D, Jani)


# MathDoc: tb2j/spacegroup-exchange-symmetrization#reversal_transpose_tensor
def _impose_reversal_identity(tensors):
    """Enforce Gamma_ji(-R) = Gamma_ij(R)^T on the working tensor dict.

    For each key pair (K, rev(K)) the stored tensors are replaced by their
    reversal-symmetrized combinations. On data that already satisfies the
    identity this is a no-op (in particular DMI is untouched).
    """
    for (ia, ja, R), _ in list(tensors.items()):
        rK = (ja, ia, tuple(-np.asarray(R, dtype=int)))
        if rK in tensors:
            S = 0.5 * (tensors[(ia, ja, R)] + tensors[rK].T)
            tensors[(ia, ja, R)] = S
            tensors[rK] = S.T


def _find_orbits(keys, key_images):
    """Group keys into orbits under the operation images (absent targets skipped)."""
    keyset = set(keys)
    remaining = set(keys)
    orbits = []
    while remaining:
        seed = min(remaining)
        orbit = {seed}
        stack = [seed]
        while stack:
            K = stack.pop()
            for im in key_images:
                tgt = im.get(K)
                if tgt is not None and tgt in keyset and tgt not in orbit:
                    orbit.add(tgt)
                    stack.append(tgt)
        remaining -= orbit
        orbits.append(sorted(orbit))
    return orbits


# MathDoc: tb2j/spacegroup-exchange-symmetrization#bond_map_image_shift
# MathDoc: tb2j/spacegroup-exchange-symmetrization#cartesian_rotation_orthogonality
def pair_images(keys, ops, site_maps, xfrac, cell):
    """Bond images and Cartesian rotations of every operation.

    Parameters
    ----------
    keys : sequence of (ia, ja, R)
        The stored atom-level bond keys.
    ops : SymmetryOperations
        The operations the images are taken under.
    site_maps : sequence of site maps
        One site map per operation; either an integer array (atom -> image
        atom) or a dict with the same content.
    xfrac : (natom, 3) array
        Fractional coordinates of the atoms the keys refer to.
    cell : (3, 3) array
        The Cartesian cell of those fractional coordinates.

    Returns
    -------
    (image_maps, cartesian_rotations) : tuple of two lists
        Both lists have one entry per operation, in the order of
        ``ops.rotations``. ``image_maps[i]`` is a dict mapping every key to
        its image key ``(ia, ja, R)`` under operation ``i``;
        ``cartesian_rotations[i]`` is the (3, 3) Cartesian rotation
        ``cell @ W @ cell^-1`` of operation ``i``.

    Raises
    ------
    ValueError
        If a bond has no integer image cell under an operation.
    """
    cell_inv = np.linalg.inv(cell)
    image_maps = []
    cartesian_rotations = []
    for W, smap in zip(ops.rotations, site_maps):
        cartesian_rotations.append(
            np.asarray(cell, dtype=float) @ np.asarray(W, dtype=float) @ cell_inv
        )
        image_maps.append({K: _bond_image(W, smap, xfrac, *K) for K in keys})
    return image_maps, cartesian_rotations
