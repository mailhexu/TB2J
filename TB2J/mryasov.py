"""Isotropic, FM-normalized linear-response elimination of soft moments."""

from collections import defaultdict
from math import fsum, isclose, isfinite
from numbers import Integral, Real


def mryasov_downfold(exchange, metals, ligands, *, dressing=1.0):
    """Return the metal-only ordered-pair exchange dictionary in eV.

    Keys are ``(R, i, j)``, where ``R`` is the integer cell displacement
    from spin ``i`` to spin ``j``. Both directions of every metal-ligand
    bond must be present with the same physical signed coupling. Ligand-only
    bonds are ignored; the model treats each ligand as an independent soft
    moment normalized to its collinear FM-reference magnitude. ``dressing``
    scales the induced interaction and must be finite and positive.
    """

    def indices(values, name):
        values = list(values)
        if (
            not values
            or any(
                isinstance(i, bool) or not isinstance(i, Integral) or i < 0
                for i in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain distinct nonnegative spin indices")
        return set(values)

    metals = indices(metals, "metals")
    ligands = indices(ligands, "ligands")
    if metals & ligands:
        raise ValueError("metal and ligand indices must not overlap")
    if (
        isinstance(dressing, bool)
        or not isinstance(dressing, Real)
        or not isfinite(dressing)
        or dressing <= 0
    ):
        raise ValueError("dressing must be a finite positive scalar")

    result = defaultdict(float)
    cross = {}
    zero = (0, 0, 0)
    for key, value in exchange.items():
        try:
            R, i, j = key
            R = tuple(R)
        except (TypeError, ValueError):
            raise ValueError("exchange keys must be (R, i, j)") from None
        if (
            len(R) != 3
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in R)
            or any(
                isinstance(x, bool) or not isinstance(x, Integral) or x < 0
                for x in (i, j)
            )
        ):
            raise ValueError(
                "exchange keys require a 3D integer translation and nonnegative spin indices"
            )
        R = tuple(int(x) for x in R)
        direct = i in metals and j in metals
        relevant = (
            direct or (i in metals and j in ligands) or (i in ligands and j in metals)
        )
        if not relevant:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
        ):
            raise ValueError("relevant exchange couplings must be finite real scalars")
        if direct:
            if R != zero or i != j:
                result[(R, i, j)] += value
        else:
            cross[(R, i, j)] = value

    ligand_bonds = defaultdict(list)
    for (R, i, j), value in cross.items():
        reverse = (tuple(-x for x in R), j, i)
        if reverse not in cross or not isclose(
            value, cross[reverse], rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"missing or inconsistent reciprocal metal-ligand bond: {(R, i, j)}"
            )
        if i in ligands:
            ligand_bonds[i].append((R, j, value))

    for ligand in ligands:
        bonds = ligand_bonds[ligand]
        try:
            field = fsum(value for _, _, value in bonds)
        except OverflowError:
            raise ValueError(
                f"non-finite FM exchange field for ligand {ligand}"
            ) from None
        if not isfinite(field) or field == 0:
            raise ValueError(
                f"missing or vanishing FM exchange field for ligand {ligand}"
            )
        for Ra, a, Ja in bonds:
            for Rb, b, Jb in bonds:
                R = tuple(y - x for x, y in zip(Ra, Rb))
                if R != zero or a != b:
                    result[(R, a, b)] += dressing * (Ja / abs(field)) * Jb
    if not all(isfinite(value) for value in result.values()):
        raise ValueError("Mryasov reduction produced non-finite effective exchange")
    return {key: value for key, value in result.items() if value != 0}
