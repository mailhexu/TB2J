"""Topology-ready model preparation from TB2J SpinIO results.

This module owns input fidelity for topological magnon calculations
(architecture ADR-001). It converts a :class:`TB2J.io_exchange.SpinIO`
instance into an immutable canonical snapshot of the selected effective
real-space interactions, geometry, moments and mappings, and validates the
source data before any information is discarded.

It deliberately reuses the existing exchange conventions and does not add a
second parser: the canonical tensors are assembled from the same channel
dictionaries that ``SpinIO.get_J_tensor`` consumes, with the DMI tensor
convention of :func:`TB2J.Jtensor.combine_J_tensor`.

Legacy band/DOS callers are unaffected: nothing here mutates the input
``SpinIO``, and no strict topology-only rejection is installed in the legacy
loader path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from TB2J.Jtensor import combine_J_tensor

__all__ = [
    "CANONICALIZATION_VERSION",
    "SUPPORTED_CHANNELS",
    "UNSUPPORTED_CHANNELS",
    "TopologyModel",
    "TopologyPreparationError",
    "prepare_topology_model",
]

#: Version of the canonicalization scheme encoded in the model digest.
CANONICALIZATION_VERSION = "tb2j-topology-model/1"

#: Channels that enter the topology tensor model.
SUPPORTED_CHANNELS = ("Jiso", "Jani", "DMI", "SIA")

#: Interaction channels that would alter the spin Hamiltonian but are not
#: supported by the topology tensor model.
UNSUPPORTED_CHANNELS = ("biquadratic", "NJT_exchange", "NJT_DMI")

#: Auxiliary records stored on SpinIO that do not alter the static
#: spin Hamiltonian consumed by the magnon model.
_AUXILIARY_CHANNELS = (
    "dJdx",
    "dJdx2",
    "dJdx_orb",
    "Jiso_orb",
    "DMI_orb",
    "Jani_orb",
    "distance_dict",
)


class TopologyPreparationError(ValueError):
    """Raised when source data cannot be prepared topology-ready."""


def _as_int_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    return arr


def _validate_index_spin(index_spin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the atom->spin map and return ``(index_spin, spin_atoms)``."""
    if index_spin.ndim != 1:
        raise TopologyPreparationError(
            f"index_spin must be one-dimensional, got shape {index_spin.shape}."
        )
    if not np.all(np.isfinite(index_spin)):
        raise TopologyPreparationError("index_spin must contain only finite integers.")
    spins = index_spin[index_spin >= 0]
    if spins.size == 0:
        raise TopologyPreparationError(
            "No magnetic atoms found: index_spin contains no non-negative entry."
        )
    if not np.all(np.sort(spins) == np.arange(spins.size)):
        raise TopologyPreparationError(
            "index_spin must assign each spin index 0..nspin-1 exactly once "
            "(no duplicates or gaps in the magnetic atoms). "
            f"Got magnetic spin indices {sorted(spins.tolist())}."
        )
    spin_atoms = np.array(
        [int(np.flatnonzero(index_spin == s)[0]) for s in range(spins.size)],
        dtype=np.int64,
    )
    return index_spin.astype(np.int64), spin_atoms


def _dict_nonzero_count(ddict, tol: float) -> int:
    """Count entries of a channel dictionary with content above ``tol``.

    The magnitude is used so imaginary-only complex content counts as
    nonzero; casting to float would silently discard it.
    """
    if not ddict:
        return 0
    count = 0
    for value in ddict.values():
        arr = np.abs(np.asarray(value))
        if np.max(arr) > tol:
            count += 1
    return count


def _real_channel_array(
    value,
    channel: str,
    location: str,
    shape: Optional[Tuple[int, ...]],
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Return a real channel array, rejecting material imaginary content.

    Imaginary exchange cannot enter the real tensor model. Roundoff-sized
    imaginary noise is tolerated only below the pair-reversal tolerances
    (``max(atol, rtol * max|real|)`` for the same entry), so nothing
    meaningful is discarded silently.
    """
    arr = np.asarray(value)
    if not np.iscomplexobj(arr):
        arr = np.asarray(arr, dtype=float)
    if shape is not None and arr.shape != shape:
        raise TopologyPreparationError(
            f"{channel} entry at {location} must have shape {shape}, "
            f"got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise TopologyPreparationError(
            f"{channel} entry at {location} contains non-finite values."
        )
    if not np.iscomplexobj(arr):
        return arr
    real = np.asarray(arr.real, dtype=float)
    imag = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
    scale = float(np.max(np.abs(real))) if arr.size else 0.0
    tol = max(atol, rtol * scale)
    if imag > tol:
        raise TopologyPreparationError(
            f"{channel} entry at {location} has material imaginary content "
            f"(max |Im| = {imag:.3e} exceeds tolerance {tol:.3e}); the "
            "topology model requires real exchange tensors and cannot "
            "represent or discard imaginary exchange."
        )
    return real


def _sia_contribution(exc, i: int, rtol: float, atol: float) -> np.ndarray:
    """Effective single-ion anisotropy tensor of spin ``i``.

    The SpinIO constructor merges uniaxial ``k1``/``k1dir`` into
    ``sia_tensor`` and clears the former. Uniaxial terms found on a fully
    constructed instance (both ``k1`` and ``k1dir`` set, e.g. via
    ``TB2J.io_exchange.edit.set_anisotropy``) were therefore added after
    construction and are independent of the tensor entries; both are summed
    exactly once. A uniaxial term with only one of ``k1``/``k1dir`` set is
    ambiguous and rejected.
    """
    tensor = np.zeros((3, 3), dtype=float)
    if getattr(exc, "has_sia_tensor", False) and exc.sia_tensor is not None:
        value = exc.sia_tensor.get(i)
        if value is not None:
            tensor += _real_channel_array(
                value, "SIA (sia_tensor)", f"spin {i}", (3, 3), rtol, atol
            )
    k1 = getattr(exc, "k1", None)
    k1dir = getattr(exc, "k1dir", None)
    has_k1 = k1 is not None
    has_dir = k1dir is not None
    if has_k1 != has_dir:
        raise TopologyPreparationError(
            "Ambiguous single-ion anisotropy representation: exactly one of "
            "k1/k1dir is set on an already-constructed SpinIO. The "
            "constructor merges both into sia_tensor; set both or neither."
        )
    if has_k1 and getattr(exc, "has_uniaxial_anistropy", False):
        k1_values = np.asarray(k1)
        k1dir_values = np.asarray(k1dir)
        if i < k1_values.size:
            K = _real_channel_array(
                k1_values[i], "SIA (k1)", f"spin {i}", (), rtol, atol
            )
            K = float(K.reshape(-1)[0])
            if abs(K) > 1e-12:
                axis = _real_channel_array(
                    k1dir_values[i], "SIA (k1dir)", f"spin {i}", (3,), rtol, atol
                )
                norm = np.linalg.norm(axis)
                if norm <= 1e-10:
                    raise TopologyPreparationError(
                        f"k1dir for spin {i} has zero norm with nonzero k1."
                    )
                tensor += K * np.outer(axis / norm, axis / norm)
    return tensor


def _channel_R_keys(ddict) -> set:
    keys = set()
    if ddict:
        for R, _i, _j in ddict.keys():
            keys.add(tuple(int(x) for x in R))
    return keys


@dataclass(frozen=True)
class TopologyModel:
    """Immutable canonical snapshot of a topology-ready magnetic model.

    The content digest covers the effective physical content only (cell,
    geometry, mappings, moments, translations, canonical tensors and the
    canonicalization version). Source selection, exclusions and paths are
    provenance: they are recorded on the model but do not change identity.
    """

    cell: np.ndarray  # (3, 3) row i = lattice vector a_i (Angstrom)
    positions: np.ndarray  # (nspin, 3) cartesian site positions (Angstrom)
    positions_fractional: np.ndarray  # (nspin, 3) fractional coordinates
    index_spin: np.ndarray  # (natom,) atom -> spin (-1 nonmagnetic)
    spin_atoms: np.ndarray  # (nspin,) spin -> atom index
    magmoms: np.ndarray  # (nspin, 3) moments (muB), spin-ordered
    Rlist: np.ndarray  # (nR, 3) integer translations, includes origin
    JR: np.ndarray  # (nR, nspin, nspin, 3, 3) effective exchange tensors
    digest: str
    selection: Dict[str, bool]
    exclusions: Tuple[str, ...]
    unsupported_nonzero: Dict[str, int]
    auxiliary_present: Tuple[str, ...]
    source_path: Optional[str] = None
    canonicalization_version: str = CANONICALIZATION_VERSION
    magnetic_cell_transform: Optional[Dict[str, Any]] = None

    @property
    def nspin(self) -> int:
        return int(self.magmoms.shape[0])

    @property
    def natom(self) -> int:
        return int(self.index_spin.shape[0])

    @property
    def Snorm(self) -> np.ndarray:
        """|mu| / 2 for each spin, matching ``Magnon.Snorm``."""
        return np.linalg.norm(self.magmoms, axis=1) / 2.0

    def provenance(self) -> Dict[str, Any]:
        """Provenance record for serialization (identity is the digest)."""
        return {
            "digest": self.digest,
            "canonicalization_version": self.canonicalization_version,
            "source_path": self.source_path,
            "selection": dict(self.selection),
            "exclusions": list(self.exclusions),
            "unsupported_nonzero": dict(self.unsupported_nonzero),
            "auxiliary_present": list(self.auxiliary_present),
            "natom": self.natom,
            "magnetic_cell_transform": self.magnetic_cell_transform,
            "nspin": self.nspin,
        }

    def magnon(self, Q=(0.0, 0.0, 0.0), uz=None, n=(1.0, 0.0, 0.0), magmoms=None):
        """Build a legacy :class:`~TB2J.magnon.magnon3.Magnon` from this model.

        The canonical ``JR`` is passed through unchanged, exactly as
        ``Magnon.load_from_io`` consumes ``SpinIO`` tensors, so band/DOS
        energies are identical for identical sources.
        """
        from ase.cell import Cell as AseCell

        from TB2J.magnon.magnon3 import Magnon

        if uz is None:
            uz = np.array([[0.0, 0.0, 1.0]])
        moments = self.magmoms if magmoms is None else np.asarray(magmoms, float)
        magnon = Magnon(
            nspin=self.nspin,
            magmom=moments,
            Rlist=self.Rlist,
            JR=self.JR,
            cell=AseCell(np.array(self.cell, dtype=float)),
            _Q=np.asarray(Q, dtype=float),
            _uz=np.asarray(uz, dtype=float),
            _n=np.asarray(n, dtype=float),
            pbc=(True, True, True),
        )
        magnon.set_reference(Q, uz, n, magmoms=moments)
        return magnon


def _digest_model(cell, positions, index_spin, spin_atoms, magmoms, Rlist, JR) -> str:
    hasher = hashlib.sha256()
    hasher.update(CANONICALIZATION_VERSION.encode("utf-8"))

    def feed_array(arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(arr.tobytes())

    feed_array(cell.astype(np.float64))
    feed_array(positions.astype(np.float64))
    feed_array(index_spin.astype(np.int64))
    feed_array(spin_atoms.astype(np.int64))
    feed_array(magmoms.astype(np.float64))
    feed_array(Rlist.astype(np.int64))
    feed_array(JR.astype(np.float64))
    return hasher.hexdigest()


def prepare_topology_model(
    exc,
    *,
    include: Optional[Mapping[str, bool]] = None,
    exclude: Sequence[str] = (),
    hermiticity_rtol: float = 1e-6,
    hermiticity_atol: float = 1e-12,
    source_path: Optional[str] = None,
) -> TopologyModel:
    """Prepare an immutable topology-ready model from a ``SpinIO`` source.

    Parameters
    ----------
    exc : TB2J.io_exchange.SpinIO
        Source interactions. Not mutated.
    include : mapping, optional
        Channel selection with keys ``Jiso``, ``Jani``, ``DMI``, ``SIA``;
        default selects all present supported channels.
    exclude : sequence of str
        Unsupported channels (``biquadratic``, ``NJT_exchange``,
        ``NJT_DMI``) to drop explicitly. Each named channel must actually
        be present with nonzero content; clean exclusions are recorded and
        the resulting model is labeled a selected approximation.
    hermiticity_rtol, hermiticity_atol : float
        Scale-aware tolerance for pair-reversal validation of the assembled
        tensors. The same tolerance bounds the numerically negligible
        imaginary part accepted in selected exchange channels; larger
        imaginary content is rejected rather than discarded.
    source_path : str, optional
        Provenance only; excluded from the content digest.

    Returns
    -------
    TopologyModel

    Raises
    ------
    TopologyPreparationError
        On inconsistent maps, zero moments, nonzero undisclosed unsupported
        interactions, ambiguous SIA representations, material imaginary
        content in selected exchange channels, or pair tensors that violate
        reversal symmetry.
    """
    if include is None:
        include = {name: True for name in SUPPORTED_CHANNELS}
    missing = set(include) - set(SUPPORTED_CHANNELS)
    if missing:
        raise TopologyPreparationError(
            f"Unknown include channels {sorted(missing)}; "
            f"supported: {sorted(SUPPORTED_CHANNELS)}."
        )
    unknown_exclude = set(exclude) - set(UNSUPPORTED_CHANNELS)
    if unknown_exclude:
        raise TopologyPreparationError(
            f"Unknown exclude channels {sorted(unknown_exclude)}; "
            f"excludable: {sorted(UNSUPPORTED_CHANNELS)}."
        )
    exclude = tuple(sorted(exclude))

    index_spin, spin_atoms = _validate_index_spin(
        _as_int_array(exc.index_spin, "index_spin")
    )
    nspin = int(spin_atoms.size)
    natom = int(index_spin.size)

    spinat = np.asarray(exc.spinat, dtype=float)
    if spinat.ndim == 1:
        moments = np.zeros((natom, 3))
        moments[:, 2] = spinat
    else:
        if spinat.shape != (natom, 3):
            raise TopologyPreparationError(
                f"spinat must have shape ({natom}, 3) or ({natom},), "
                f"got {spinat.shape}."
            )
        moments = spinat
    magmoms = moments[spin_atoms]
    if not np.all(np.isfinite(magmoms)):
        raise TopologyPreparationError("Magnetic moments contain non-finite values.")
    if np.any(np.linalg.norm(magmoms, axis=1) <= 0.0):
        bad = np.flatnonzero(np.linalg.norm(magmoms, axis=1) <= 0.0).tolist()
        raise TopologyPreparationError(
            f"Magnetic spins with zero moment cannot enter LSWT: spins {bad}."
        )

    atoms = exc.atoms
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise TopologyPreparationError("Cell must be a finite 3x3 array.")
    if abs(np.linalg.det(cell)) <= 1e-12:
        raise TopologyPreparationError("Cell is singular.")
    all_positions = np.asarray(atoms.get_positions(), dtype=float)
    if all_positions.shape != (natom, 3) or not np.all(np.isfinite(all_positions)):
        raise TopologyPreparationError("Atomic positions are non-finite or malformed.")
    positions = all_positions[spin_atoms]
    positions_fractional = positions @ np.linalg.inv(cell)

    # --- channel inventory -------------------------------------------------
    scale_tol = 1e-10  # absolute threshold for "nonzero" content detection
    unsupported_nonzero: Dict[str, int] = {}
    for name, ddict in (
        ("biquadratic", getattr(exc, "biquadratic_Jdict", None)),
        ("NJT_exchange", getattr(exc, "NJT_Jdict", None)),
        ("NJT_DMI", getattr(exc, "NJT_ddict", None)),
    ):
        count = _dict_nonzero_count(ddict, scale_tol)
        unsupported_nonzero[name] = count
    nonzero_unsupported = {
        name: count for name, count in unsupported_nonzero.items() if count > 0
    }
    undisclosed = set(nonzero_unsupported) - set(exclude)
    if undisclosed:
        raise TopologyPreparationError(
            "Nonzero unsupported interaction channels present in the source: "
            f"{ {name: nonzero_unsupported[name] for name in sorted(undisclosed)} }. "
            "Topology-ready preparation refuses undisclosed terms; pass "
            "`exclude=(...)` to record an explicit selection."
        )
    excluded_but_clean = [
        name for name in exclude if nonzero_unsupported.get(name, 0) == 0
    ]
    if excluded_but_clean:
        raise TopologyPreparationError(
            f"Channels {excluded_but_clean} were selected for exclusion but "
            "have no nonzero content in the source; remove them from "
            "`exclude` so the selection stays truthful."
        )

    auxiliary_present = tuple(
        name for name in _AUXILIARY_CHANNELS if getattr(exc, name, None) is not None
    )

    # --- translations: union over selected channels ------------------------
    R_keys: set = {(0, 0, 0)}
    if include["Jiso"] and exc.exchange_Jdict:
        R_keys |= _channel_R_keys(exc.exchange_Jdict)
    if include["DMI"] and exc.dmi_ddict:
        R_keys |= _channel_R_keys(exc.dmi_ddict)
    if include["Jani"] and exc.Jani_dict:
        R_keys |= _channel_R_keys(exc.Jani_dict)
    Rlist = np.array(sorted(R_keys), dtype=np.int64)
    nR = Rlist.shape[0]
    R_index = {tuple(R): idx for idx, R in enumerate(Rlist)}

    # --- canonical tensor assembly -----------------------------------------
    JR = np.zeros((nR, nspin, nspin, 3, 3), dtype=float)

    def _pair_value(ddict, R, i, j):
        if not ddict:
            return None
        return ddict.get((tuple(int(x) for x in R), int(i), int(j)))

    for iR, R in enumerate(Rlist):
        for i in range(nspin):
            for j in range(nspin):
                tensor = np.zeros((3, 3), dtype=float)
                location = f"R={tuple(int(x) for x in R)}, spins ({i},{j})"
                if include["Jiso"]:
                    Jiso = _pair_value(exc.exchange_Jdict, R, i, j)
                    if Jiso is not None:
                        Jiso = _real_channel_array(
                            Jiso,
                            "Jiso",
                            location,
                            None,
                            hermiticity_rtol,
                            hermiticity_atol,
                        )
                        if Jiso.size != 1:
                            raise TopologyPreparationError(
                                f"Jiso entry at {location} must be a scalar, "
                                f"got shape {Jiso.shape}."
                            )
                        tensor += float(Jiso.reshape(-1)[0]) * np.eye(3)
                if include["DMI"]:
                    D = _pair_value(exc.dmi_ddict, R, i, j)
                    if D is not None:
                        D = _real_channel_array(
                            D,
                            "DMI",
                            location,
                            (3,),
                            hermiticity_rtol,
                            hermiticity_atol,
                        )
                        tensor += combine_J_tensor(D=D)
                if include["Jani"]:
                    Jani = _pair_value(exc.Jani_dict, R, i, j)
                    if Jani is not None:
                        Jani = _real_channel_array(
                            Jani,
                            "Jani",
                            location,
                            (3, 3),
                            hermiticity_rtol,
                            hermiticity_atol,
                        )
                        if np.max(np.abs(Jani - Jani.T)) > max(
                            hermiticity_atol,
                            hermiticity_rtol * float(np.max(np.abs(Jani))),
                        ):
                            raise TopologyPreparationError(
                                f"Jani entry at R={R}, spins ({i},{j}) is not "
                                "symmetric; the topology model requires the "
                                "symmetric anisotropy convention."
                            )
                        tensor += Jani
                if include["SIA"] and i == j and np.all(R == 0):
                    tensor += _sia_contribution(
                        exc, i, hermiticity_rtol, hermiticity_atol
                    )
                JR[iR, i, j] = tensor

    # --- pair-reversal (Hermiticity) validation ---------------------------
    max_abs = float(np.max(np.abs(JR))) if nR else 0.0
    tol = max(hermiticity_atol, hermiticity_rtol * max_abs)
    violations = []
    for iR, R in enumerate(Rlist):
        Rneg = tuple(-int(x) for x in R)
        iRneg = R_index.get(Rneg)
        if iRneg is None:
            violations.append(f"translation {R} present without its reverse {Rneg}")
            continue
        # JR[-R][i,j] must equal JR[R][j,i]^T elementwise:
        diff = JR[iRneg] - np.transpose(JR[iR], (1, 0, 3, 2))
        if np.max(np.abs(diff)) > tol:
            i_bond, j_bond = np.argwhere(np.max(np.abs(diff), axis=(2, 3)) > tol)[0]
            violations.append(
                f"J({R})[{i_bond},{j_bond}] != J({Rneg})[{j_bond},{i_bond}]^T "
                f"(max |diff| = {np.max(np.abs(diff)):.3e}, tol = {tol:.3e})"
            )
        del diff
    if violations:
        raise TopologyPreparationError(
            "Pair-reversal validation failed; the topology model requires "
            "J_ij(R) = J_ji(-R)^T without fabricating missing reverse bonds. "
            "Offenses:\n  "
            + "\n  ".join(violations[:10])
            + (
                f"\n  ... and {len(violations) - 10} more"
                if len(violations) > 10
                else ""
            )
        )

    digest = _digest_model(cell, positions, index_spin, spin_atoms, magmoms, Rlist, JR)

    selection = {name: bool(include[name]) for name in SUPPORTED_CHANNELS}
    return TopologyModel(
        cell=cell,
        positions=positions,
        positions_fractional=positions_fractional,
        index_spin=index_spin,
        spin_atoms=spin_atoms,
        magmoms=magmoms,
        Rlist=Rlist,
        JR=JR,
        digest=digest,
        selection=selection,
        exclusions=exclude,
        unsupported_nonzero=dict(unsupported_nonzero),
        auxiliary_present=auxiliary_present,
        source_path=source_path,
    )
