"""
Symmetrization of exchange parameters using exact space-group operations.

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

Symmetrization is orbit-wise projection: keys are grouped into orbits under
the symmetry operations (plus the reversal identity, imposed as a
pre/post-condition), the transformed tensors are averaged, and the average is
decomposed back into J^iso, J^ani and D.  Components smaller than
``zero_tol`` after the projection are snapped to exactly zero and reported as
zero by symmetry.

If the stored data are truncated (e.g. by a finite Rcut), a symmetry image of
a key may be absent from the dataset.  Orbits are then the connected
components of the "key -> present image" graph, and each orbit is averaged
under exactly the operations that map it onto itself (its set-wise
stabilizer); partially defined operations are excluded per orbit rather than
applied unevenly, so the result stays exactly invariant under every retained
operation.
"""

import copy

import numpy as np

from TB2J.io_exchange import SpinIO
from TB2J.versioninfo import print_license

DEFAULT_ZERO_TOL = 1e-8


def _levi_civita():
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return eps


_EPS = _levi_civita()


def d_to_skew(D):
    """Skew-symmetric matrix A with A^{ab} = eps^{abg} D^g (so S_i^T A S_j = D.(S_i x S_j))."""
    D = np.asarray(D, dtype=float)
    return np.einsum("abg,g->ab", _EPS, D)


def skew_to_d(S):
    """DMI vector D^g = 1/2 eps^{gab} S^{ab} from a skew-symmetric matrix."""
    S = np.asarray(S, dtype=float)
    return 0.5 * np.einsum("gab,ab->g", _EPS, S)


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


def magnetic_symmetry_ops(atoms, magmoms, symprec=1e-5):
    """Detect the magnetic space group of an ASE atoms object with moments.

    magmoms: (natom,) for collinear moments (taken along z) or (natom, 3)
        for non-collinear moments, aligned with the atoms order.
    """
    spglib = _get_spglib()
    cell = (
        np.asarray(atoms.get_cell().array, dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
        np.asarray(magmoms, dtype=float),
    )
    dataset = spglib.get_magnetic_symmetry_dataset(cell, symprec=symprec)
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


def _report_zero_channels(Jiso, D, Jani, zero_tol):
    """Names of channels/components that are identically zero after projection."""
    names = []
    if abs(Jiso) < zero_tol:
        names.append("J^iso = 0 (by symmetry)")
    dzero = np.abs(D) < zero_tol
    if np.all(dzero):
        names.append("DMI = 0 (by symmetry)")
    else:
        names += [
            f"D_{c} = 0 (by symmetry)" for c, z in zip(("x", "y", "z"), dzero) if z
        ]
    jzero = np.abs(Jani) < zero_tol
    if np.all(jzero):
        names.append("J^ani = 0 (by symmetry)")
    else:
        comps = [("x", "x"), ("y", "y"), ("z", "z"), ("x", "y"), ("y", "z"), ("z", "x")]
        names += [
            f"J^ani_{a}{b} = 0 (by symmetry)"
            for (a, b), z in zip(
                comps,
                [
                    jzero[0, 0],
                    jzero[1, 1],
                    jzero[2, 2],
                    jzero[0, 1],
                    jzero[1, 2],
                    jzero[2, 0],
                ],
            )
            if z
        ]
    return names


def _symmetrize_pairs(exc, ops, site_maps, xfrac, cell, zero_tol, verbose):
    """Orbit-wise symmetrization of the pair channels of exc, in place."""
    keys = _collect_atom_keys(exc)
    tensors = {K: _atom_tensor(exc, *K) for K in keys}
    _impose_reversal_identity(tensors)

    # Per-operation bond images and Cartesian rotations.
    cell_inv = np.linalg.inv(cell)
    key_images = []
    cart_rots = []
    for W, smap in zip(ops.rotations, site_maps):
        cart_rots.append(
            np.asarray(cell, dtype=float) @ np.asarray(W, dtype=float) @ cell_inv
        )
        key_images.append({_K: _bond_image(W, smap, xfrac, *_K) for _K in keys})

    # Orbits: connected components of the graph "key -> present image" over
    # ALL operations.  Each orbit is then averaged under its own set-wise
    # stabilizer: the operations that map every member of the orbit onto a
    # member of the orbit.  The stabilizer is a group, so the orbit average is
    # exactly invariant under it; operations that would map part of the orbit
    # outside the stored data (truncated R shells) are excluded only for that
    # orbit, never applied unevenly.
    orbits = _find_orbits(keys, key_images)

    if verbose:
        print(f"Number of symmetry-equivalent exchange-pair orbits: {len(orbits)}")
        print("Per-orbit report (channels forced to zero by symmetry are listed):")

    for iorb, orbit in enumerate(orbits):
        oset = set(orbit)
        subgroup = [
            idx
            for idx, im in enumerate(key_images)
            if all(im[K] in oset for K in orbit)
        ]
        contribs = []
        for K in orbit:
            G = tensors[K]
            for idx in subgroup:
                contribs.append(cart_rots[idx] @ G @ cart_rots[idx].T)
        Gbar = np.mean(contribs, axis=0)
        Jiso, D, Jani = decompose_tensor(Gbar)
        Jiso = float(_snap_zero(np.array([Jiso]), zero_tol)[0])
        D = _snap_zero(D, zero_tol)
        Jani = _snap_zero(Jani, zero_tol)

        for ia, ja, R in orbit:
            i, j = exc.index_spin[ia], exc.index_spin[ja]
            key = (tuple(R), i, j)
            exc.exchange_Jdict[key] = Jiso
            if exc.dmi_ddict is not None:
                exc.dmi_ddict[key] = D
            if exc.Jani_dict is not None:
                exc.Jani_dict[key] = Jani

        if verbose:
            members = "; ".join(
                f"{exc.atoms[ia].symbol}{ia}-{exc.atoms[ja].symbol}{ja} R={tuple(R)}"
                for ia, ja, R in orbit
            )
            zeros = _report_zero_channels(Jiso, D, Jani, zero_tol)
            line = f"  Orbit {iorb}: {members}"
            if zeros:
                line += ": " + ", ".join(zeros)
            print(line)

    _symmetrize_sia(exc, ops, site_maps, cell, zero_tol, verbose)


def _symmetrize_sia(exc, ops, site_maps, cell, zero_tol, verbose):
    """Orbit-wise symmetrization of the single-ion anisotropy tensors, in place."""
    if not getattr(exc, "has_sia_tensor", False) or not exc.sia_tensor:
        return
    spin_of_atom = {exc.iatom(i): i for i in exc.sia_tensor}
    atoms = sorted(spin_of_atom)
    K = {a: np.asarray(exc.sia_tensor[spin_of_atom[a]], dtype=float) for a in atoms}
    site_images = [dict(zip(atoms, smap[atoms].tolist())) for smap in site_maps]
    orbits = _find_orbits(atoms, site_images)
    cart_rots = [
        np.asarray(cell, dtype=float) @ np.asarray(W, dtype=float) @ np.linalg.inv(cell)
        for W in ops.rotations
    ]

    if verbose:
        print(f"Number of single-ion anisotropy site orbits: {len(orbits)}")
    for iorb, orbit in enumerate(orbits):
        oset = set(orbit)
        contribs = []
        for a in orbit:
            for images, Wc in zip(site_images, cart_rots):
                tgt = images.get(a)
                if tgt in oset:
                    contribs.append(Wc @ K[a] @ Wc.T)
        Kbar = _snap_zero(np.mean(contribs, axis=0), zero_tol)
        for a in orbit:
            exc.sia_tensor[spin_of_atom[a]] = Kbar
        if verbose:
            zeros = np.all(Kbar == 0.0)
            label = ", ".join(f"{exc.atoms[a].symbol}{a}" for a in orbit)
            line = f"  SIA orbit {iorb}: {label}"
            if zeros:
                line += ": K = 0 (by symmetry)"
            print(line)


def symmetrize_spinio(exc, ops, symprec=1e-5, zero_tol=DEFAULT_ZERO_TOL, verbose=False):
    """Symmetrize all pair channels (and SIA) of exc in place with the given operations.

    The operations must be given in the fractional basis of exc.atoms.
    """
    xfrac = np.asarray(exc.atoms.get_scaled_positions(), dtype=float)
    cell = np.asarray(exc.atoms.get_cell().array, dtype=float)
    site_maps = _build_site_maps(xfrac, cell, ops, symprec)
    _symmetrize_pairs(exc, ops, site_maps, xfrac, cell, zero_tol, verbose)


class TB2JSymmetrizer:
    """Symmetrizer of the exchange parameters of a SpinIO object.

    Parameters
    ----------
    exc: SpinIO
        The exchange parameters to symmetrize. They are not modified; the
        symmetrized copy is ``.new_exc``.
    symprec: float
        Symmetry precision in Angstrom, used both for spglib and for the
        site maps.
    verbose: bool
        Print the detected group, the number of orbits and a per-orbit
        report of the channels that are zero by symmetry.
    Jonly: bool
        Discard DMI, anisotropic exchange and single-ion anisotropy after
        symmetrizing the isotropic exchange.
    magnetic: bool
        Use the magnetic space group (requires spinat on the SpinIO object)
        instead of the crystallographic space group.
    zero_tol: float
        Components of the symmetrized channels smaller than this are snapped
        to exactly zero.
    """

    def __init__(
        self,
        exc,
        symprec=1e-5,
        verbose=True,
        Jonly=False,
        magnetic=False,
        zero_tol=DEFAULT_ZERO_TOL,
    ):
        self.exc = exc
        self.symprec = symprec
        self.verbose = verbose
        self.Jonly = Jonly
        self.magnetic = magnetic
        self.zero_tol = zero_tol
        self.new_exc = copy.deepcopy(exc)
        if self.verbose:
            print("=" * 30)
            print_license()
            print("-" * 30)

    def print_license(self):
        print_license()

    def _get_magmoms(self):
        exc = self.exc
        spinat = getattr(exc, "spinat", None)
        if spinat is None:
            raise ValueError(
                "magnetic=True requires the magnetic moments (spinat) on the "
                "SpinIO object, but spinat is missing. Please provide magmoms."
            )
        spinat = np.asarray(spinat, dtype=float)
        natom = len(exc.atoms)
        if exc.colinear:
            if spinat.ndim == 2:
                return spinat[:, 2]
            return spinat.reshape(natom)
        return spinat.reshape(natom, 3)

    def _detect_ops(self, magnetic):
        atoms = self.exc.atoms
        if magnetic:
            ops = magnetic_symmetry_ops(
                atoms, self._get_magmoms(), symprec=self.symprec
            )
        else:
            ops = crystal_symmetry_ops(atoms, symprec=self.symprec)
        if self.verbose:
            print(f"Symmetry detection precision (symprec): {self.symprec} Angstrom.")
            if magnetic:
                print("Detected magnetic group:")
            else:
                print("Detected group:")
            print(f"  {ops.description}")
            if ops.time_reversals is not None:
                print(
                    "  Time reversal: primed operations use the same tensor "
                    "action as unprimed ones; they only enlarge the set of "
                    "spin-consistent site maps."
                )
        return ops

    def symmetrize_J(self, magnetic=None, zero_tol=None):
        """Symmetrize the exchange parameters (J^iso, DMI, J^ani and SIA).

        magnetic, zero_tol: default to the values given at construction.
        """
        magnetic = self.magnetic if magnetic is None else magnetic
        zero_tol = self.zero_tol if zero_tol is None else zero_tol
        ops = self._detect_ops(magnetic)
        symmetrize_spinio(
            self.new_exc,
            ops,
            symprec=self.symprec,
            zero_tol=zero_tol,
            verbose=self.verbose,
        )
        if self.Jonly:
            ne = self.new_exc
            ne.dmi_ddict = None
            ne.Jani_dict = None
            ne.sia_tensor = None
            ne.has_sia_tensor = False
            ne.has_uniaxial_anistropy = False
            ne.k1 = None
            ne.k1dir = None

    def output(self, path="TB2J_symmetrized"):
        if path is None:
            path = "TB2J_symmetrized"
        self.new_exc.write_all(path=path)

    def run(self, path="TB2J_symmetrized"):
        print("** Symmetrizing exchange parameters.")
        self.symmetrize_J()
        print("** Outputing the symmetrized exchange parameters.")
        print(f"** Output path: {path} .")
        self.output(path=path)
        print("** Finished.")


def symmetrize_J(
    exc=None,
    path=None,
    fname="TB2J.pickle",
    symprec=1e-5,
    output_path="TB2J_symmetrized",
    Jonly=False,
    magnetic=False,
    zero_tol=DEFAULT_ZERO_TOL,
):
    """
    Symmetrize the exchange parameters.

    :param exc: SpinIO exchange object. If not given, it is loaded from path.
    :param path: path to the TB2J pickle output, used if exc is None.
    :param fname: name of the pickle file.
    :param symprec: symmetry precision in Angstrom.
    :param output_path: output path of the symmetrized results.
    :param Jonly: discard DMI, anisotropic exchange and SIA after symmetrization.
    :param magnetic: use the magnetic space group built from the stored spinat.
    :param zero_tol: snap symmetrized components smaller than this to zero.
    """
    if exc is None:
        if path is None:
            raise ValueError("Please provide the path to the exchange parameters.")
        exc = SpinIO.load_pickle(path=path, fname=fname)
    symmetrizer = TB2JSymmetrizer(
        exc,
        symprec=symprec,
        Jonly=Jonly,
        magnetic=magnetic,
        zero_tol=zero_tol,
    )
    symmetrizer.run(path=output_path)


def _map_atoms_to_spinio(atoms, spinio, symprec=1e-3):
    """
    Map atoms from input structure to SpinIO structure.

    Uses species and position matching within symprec tolerance.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic structure.
    spinio : SpinIO
        The SpinIO object containing the reference structure.
    symprec : float, optional
        Position tolerance in Angstrom. Default is 1e-3.

    Returns
    -------
    dict
        Mapping from SpinIO atom index to input structure atom index.
    """
    mapping = {}
    symbols_in = atoms.get_chemical_symbols()
    pos_in = atoms.get_positions()
    symbols_s = spinio.atoms.get_chemical_symbols()
    pos_s_array = spinio.atoms.get_positions()

    for i_in, (sym, pos) in enumerate(zip(symbols_in, pos_in)):
        for i_s, (sym_s, pos_s) in enumerate(zip(symbols_s, pos_s_array)):
            if sym == sym_s:
                if np.linalg.norm(pos - pos_s) < symprec:
                    mapping[i_s] = i_in
                    break
    return mapping


def symmetrize_exchange(spinio, atoms, symprec=1e-3):
    """
    Symmetrize the exchange parameters of spinio using the symmetry of a
    provided atomic structure.

    The space group is detected from the provided structure with spglib
    (rotations and translations) and the full pair tensors (isotropic
    exchange, DMI, anisotropic exchange) and single-ion anisotropy are
    symmetrized orbit-wise, in place.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    atoms : ase.Atoms
        Atomic structure that defines the target symmetry. It must share the
        cell of spinio.atoms; positions may be idealized. For example,
        provide a cubic structure to symmetrize to cubic symmetry.
    symprec : float, optional
        Symmetry precision in Angstrom. Default is 1e-3.

    Notes
    -----
    - The spinio.atoms structure is NOT modified; only the exchange values
      change.
    - Atoms are mapped between the input structure and the SpinIO structure
      by species and position; operations for which some involved atom has
      no counterpart are skipped.

    Examples
    --------
    >>> from ase.io import read
    >>> # Symmetrize to cubic symmetry
    >>> cubic_structure = read('cubic_smfeo3.cif')
    >>> symmetrize_exchange(spinio, atoms=cubic_structure)

    >>> # Symmetrize to the original Pnma symmetry
    >>> symmetrize_exchange(spinio, atoms=spinio.atoms)
    """
    ops_atoms = crystal_symmetry_ops(atoms, symprec=symprec)
    mapping = _map_atoms_to_spinio(atoms, spinio, symprec=symprec)
    if not mapping:
        raise ValueError(
            "No atoms of the SpinIO structure could be mapped to the provided "
            "structure; cannot transfer symmetry."
        )
    inv_map = {}
    for i_s, i_in in mapping.items():
        if i_in in inv_map and inv_map[i_in] != i_s:
            raise ValueError(
                "Ambiguous atom mapping between the provided structure and the "
                "SpinIO structure."
            )
        inv_map[i_in] = i_s

    xfrac = np.asarray(spinio.atoms.get_scaled_positions(), dtype=float)
    cell = np.asarray(spinio.atoms.get_cell().array, dtype=float)
    pos_in = atoms.get_positions()
    pos_s = spinio.atoms.get_positions()

    keys = _collect_atom_keys(spinio)
    needed_atoms = sorted({a for ia, ja, _ in keys for a in (ia, ja)})
    tensors = {K: _atom_tensor(spinio, *K) for K in keys}
    _impose_reversal_identity(tensors)

    # Site maps in the SpinIO basis: spinio atom a -> image in atoms -> spinio atom.
    xfrac_in = np.asarray(atoms.get_scaled_positions(), dtype=float)
    cell_in = np.asarray(atoms.get_cell().array, dtype=float)
    site_maps = []
    cart_rots = []
    key_images = []
    cell_inv = np.linalg.inv(cell)
    for W, t, smap_atoms in zip(
        ops_atoms.rotations,
        ops_atoms.translations,
        _build_site_maps(xfrac_in, cell_in, ops_atoms, symprec),
    ):
        smap = {}
        ok = True
        for a in needed_atoms:
            if a not in mapping:
                ok = False
                break
            i_img = int(smap_atoms[mapping[a]])
            dist = np.linalg.norm(pos_in[i_img] - pos_s, axis=1)
            b = int(np.argmin(dist))
            if dist[b] > symprec:
                ok = False
                break
            smap[a] = b
        if not ok:
            continue
        site_maps.append(smap)
        cart_rots.append(np.asarray(cell) @ np.asarray(W, dtype=float) @ cell_inv)
        images = {}
        for ia, ja, R in keys:
            d = xfrac[ja] + np.asarray(R, dtype=float) - xfrac[ia]
            d2 = np.asarray(W, dtype=float) @ d
            ia2, ja2 = smap[ia], smap[ja]
            diff = d2 - (xfrac[ja2] - xfrac[ia2])
            R2 = np.round(diff)
            if np.max(np.abs(diff - R2)) > 0.1:
                continue
            images[(ia, ja, R)] = (ia2, ja2, tuple(int(x) for x in R2))
        key_images.append(images)
    if not site_maps:
        raise ValueError(
            "None of the symmetry operations of the provided structure could "
            "be transferred to the SpinIO structure."
        )

    orbits = _find_orbits(keys, key_images)
    for orbit in orbits:
        oset = set(orbit)
        # average under the orbit's set-wise stabilizer: the transferred
        # operations that map every member onto a stored member of the orbit
        subgroup = [
            idx
            for idx, im in enumerate(key_images)
            if all(im.get(K) in oset for K in orbit)
        ]
        contribs = []
        for K in orbit:
            G = tensors[K]
            for idx in subgroup:
                Wc = cart_rots[idx]
                contribs.append(Wc @ G @ Wc.T)
        Gbar = np.mean(contribs, axis=0)
        Jiso, D, Jani = decompose_tensor(Gbar)
        Jiso = float(_snap_zero(np.array([Jiso]), 0.0)[0])
        D = _snap_zero(D, 0.0)
        Jani = _snap_zero(Jani, 0.0)
        for ia, ja, R in orbit:
            i, j = spinio.index_spin[ia], spinio.index_spin[ja]
            key = (tuple(R), i, j)
            spinio.exchange_Jdict[key] = Jiso
            if spinio.dmi_ddict is not None:
                spinio.dmi_ddict[key] = D
            if spinio.Jani_dict is not None:
                spinio.Jani_dict[key] = Jani


def symmetrize_J_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=(
            "Symmetrize exchange parameters (isotropic exchange, DMI, "
            "anisotropic exchange and single-ion anisotropy) with the exact "
            "space-group operations of the structure (rotations and "
            "translations). With --magnetic, the magnetic space group built "
            "from the stored magnetic moments (spinat) is used instead."
        )
    )
    parser.add_argument(
        "-i",
        "--inpath",
        default=None,
        help="input path to the exchange parameters",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        default="TB2J_results_symmetrized",
        help="output path to the symmetrized exchange parameters",
    )
    parser.add_argument(
        "-s",
        "--symprec",
        type=float,
        default=1e-5,
        help="precision for symmetry detection. default is 1e-5 Angstrom",
    )
    parser.add_argument(
        "--Jonly",
        action="store_true",
        help=(
            "symmetrize only the exchange parameters and discard the DMI, "
            "anisotropic exchange and single-ion anisotropy"
        ),
        default=False,
    )
    parser.add_argument(
        "--magnetic",
        action="store_true",
        help=(
            "use the magnetic space group (from the stored spinat) instead "
            "of the crystallographic space group"
        ),
        default=False,
    )
    parser.add_argument(
        "--zero-tol",
        type=float,
        default=DEFAULT_ZERO_TOL,
        help=(
            "components of the symmetrized channels smaller than this are "
            "snapped to exactly zero. default is 1e-8"
        ),
    )

    args = parser.parse_args()
    if args.inpath is None:
        parser.print_help()
        raise ValueError("Please provide the input path to the exchange.")
    symmetrize_J(
        path=args.inpath,
        output_path=args.outpath,
        symprec=args.symprec,
        Jonly=args.Jonly,
        magnetic=args.magnetic,
        zero_tol=args.zero_tol,
    )


if __name__ == "__main__":
    symmetrize_J_cli()
