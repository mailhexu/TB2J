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


def prepare_tbupy_inputs(
    tbupy_result_file=None,
    tbupy_result=None,
    tbmodels=None,
    atoms=None,
    basis=None,
    efermi=None,
    spinflip_tol: float = 1e-10,
):
    """Prepare ``Manager`` inputs from TBUpy result inputs."""
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
