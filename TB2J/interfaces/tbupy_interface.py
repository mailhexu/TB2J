"""Optional TBUpy interface for TB2J exchange calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manager import Manager


def _load_tbupy_result(path):
    try:
        from tbupy.io import load_scf_result
    except ImportError as exc:
        raise ImportError(
            "Reading .tbupy.nc files requires TBUpy. Install TBUpy or pass "
            "preconstructed HamiltonIO-compatible collinear models."
        ) from exc
    return load_scf_result(path)


def _basis_from_model(model):
    return list(model.orbs)


def _prepare_tbupy_spinor_inputs(tbupy_result=None, basis=None):
    """Certified spinor handoff: the full spinor effective model.

    The spinor path forwards the TBUpy effective Hamiltonian
    (spin-interleaved) directly to the TB2J noncollinear machinery; no
    collinear channel split is performed. Only energy-certified,
    digest-verified results are accepted.
    """
    from tbupy.io import _interaction_tensor_digest

    if tbupy_result is None:
        raise ValueError(
            "the spinor handoff requires a certified TBUpy result object "
            "(pass tbupy_result=...); pre-split tbmodels are collinear-only"
        )
    meta = getattr(tbupy_result, "metadata", {}) or {}
    certified = bool(meta.get("energy_certified")) and bool(
        (meta.get("certification") or {}).get("certified", True)
    )
    if not certified:
        raise ValueError(
            "refusing spinor handoff: the TBUpy result is not energy-certified"
        )
    manifest = meta.get("interaction_tensors")
    digest = meta.get("interaction_tensor_digest")
    if manifest is None or digest is None:
        raise ValueError(
            "refusing spinor handoff: missing interaction tensor manifest/digest"
        )
    if _interaction_tensor_digest(manifest) != digest:
        raise ValueError(
            "refusing spinor handoff: interaction tensor digest mismatch "
            "(the result was tampered with after certification)"
        )
    orbs = list(tbupy_result.hamiltonian.orbs)
    if basis is None:
        basis = orbs
    atoms = tbupy_result.atoms
    owners = {orb.iatom for orb in basis}
    if not basis or owners != set(range(len(atoms))):
        raise ValueError(
            "basis ownership violation: every atom must own at least one "
            "orbital of the handed-off basis"
        )
    return atoms, tbupy_result.hamiltonian, list(basis), tbupy_result.efermi


def prepare_tbupy_inputs(
    tbupy_result_file=None,
    tbupy_result=None,
    tbmodels=None,
    atoms=None,
    basis=None,
    efermi=None,
    spinflip_tol: float = 1e-10,
    colinear: bool = True,
):
    """Prepare ``Manager`` inputs from TBUpy result inputs."""
    if not colinear:
        if tbmodels is not None:
            raise ValueError(
                "pre-split tbmodels cannot be used with colinear=False: the "
                "spinor path requires a certified TBUpy result"
            )
        return _prepare_tbupy_spinor_inputs(tbupy_result=tbupy_result, basis=basis)
    if tbmodels is None:
        if tbupy_result is None:
            if tbupy_result_file is None:
                raise ValueError(
                    "Provide tbupy_result_file, tbupy_result, or pre-split tbmodels"
                )
            tbupy_result = _load_tbupy_result(tbupy_result_file)
        tbmodels = tbupy_result.to_collinear_models(spinflip_tol=spinflip_tol)
        atoms = tbupy_result.atoms if atoms is None else atoms
        efermi = tbupy_result.efermi if efermi is None else efermi

    if atoms is None:
        atoms = tbmodels[0].atoms
    if basis is None:
        basis = _basis_from_model(tbmodels[0])
    return atoms, tbmodels, basis, efermi


class TBUpyManager(Manager):
    """TB2J manager for TBUpy-converged Hamiltonians."""

    def __init__(
        self,
        tbupy_result_file=None,
        tbupy_result=None,
        tbmodels=None,
        atoms=None,
        basis=None,
        colinear: bool = True,
        spinflip_tol: float = 1e-10,
        **kwargs: Any,
    ):
        if not colinear:
            raise NotImplementedError(
                "TBUpy interface currently supports collinear runs"
            )
        atoms, tbmodels, basis, efermi = prepare_tbupy_inputs(
            tbupy_result_file=tbupy_result_file,
            tbupy_result=tbupy_result,
            tbmodels=tbmodels,
            atoms=atoms,
            basis=basis,
            efermi=kwargs.get("efermi"),
            spinflip_tol=spinflip_tol,
        )
        if efermi is not None:
            kwargs["efermi"] = efermi
        if tbupy_result_file is not None:
            kwargs.setdefault(
                "description",
                f"Input from TBUpy converged result: {Path(tbupy_result_file)}",
            )
        super().__init__(
            atoms=atoms, models=tbmodels, basis=basis, colinear=True, **kwargs
        )


def gen_exchange_tbupy(**kwargs):
    """Run TB2J exchange from a TBUpy result file or live result."""
    return TBUpyManager(**kwargs)


gen_exchange = gen_exchange_tbupy
