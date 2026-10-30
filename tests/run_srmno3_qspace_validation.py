"""Private SrMnO3 reciprocal-space exchange validation runner (Story 005).

This is the private study entry point of subproject
``tb2j/srmno3-reciprocal-space-exchange-validation``.  It is not a public TB2J
interface: no CLI, no plugin configuration language, no general artifact
framework.  It compares, on one fixed spin-resolved Hamiltonian,

  * the direct unrenormalized reciprocal-space response ``Jq`` obtained with
    the user-approved raw route ``ExchangeCLQspace.calculate_raw()`` (no Bruno
    renormalization), against
  * the complete finite-grid Fourier transform of the self-inclusive
    real-space response collected with ``ExchangeCL2`` *before* the public
    ``exchange_Jdict`` self-term exclusion,

for every q point of the declared mesh including Gamma and the onsite self
term, using the conventions certified by the declared shared MathDoc
authorities:

  - ``qspace-unrenormalized-kernel`` (spin/ordered-pair contract, reciprocal
    evenness under real-gauge reciprocity),
  - ``qspace-finite-grid-transform`` (plus-phase forward transform carrying
    no extra factor, minus-phase production inverse with the 1/nqpt factor,
    retained onsite self).

Identities are verified, not just declared: the consumed TB2J checkout must
be exactly at ``source_commit`` with a clean package tree, and each declared
math authority file must hash to its declared SHA-256, both before any
calculation.

The declared mathematical authority is valid for the one-magnetic-site scalar
ferromagnetic case on a complete uniform Gamma-centered mesh with a real
orbital gauge.  Production ``integrate`` allocates real q arrays, so the direct
route yields real values; the runner's forward transform of the real-space
response is retained as a complex array and its imaginary components are
quantified and reported, never silently discarded.  Cases outside the declared
domain (several magnetic sites, non-orthogonal or complex-gauge Hamiltonians,
odd conventions) are refused before any numerical comparison.

Usage (private study / smoke, not a CLI)::

    from tests.run_srmno3_qspace_validation import run_case

    result = run_case(settings, output_root)   # -> summary dict

Outcomes are distinct:

  * ``pass``    -- all gates passed on the primary (last) refinement;
  * ``failed``  -- the numerical comparison completed and a gate failed; the
    full evidence bundle is written and labelled ``failed``;
  * ``PreflightRefusal`` -- invalid inputs/conventions/authority are refused
    *before* calculation; a labelled refusal record is written to
    ``output_root`` and the exception carries the reasons.  A refusal is
    neither a pass nor a numerical failure.

Runtime exceptions from the library (for example a pole on the contour)
propagate unchanged; they are never converted into a passing result.

Evidence bundle (one run identity binds all files, written into
``output_root/<run_id>/``; existing evidence is never overwritten implicitly):

  * ``manifest.json``  -- resolved settings, input/source/math identities,
    ensemble and numerical conventions, q/R/site order, units, environment;
  * ``raw.npz``        -- ``qpts[nq,3]``, ``R[nR,3]``, per-refinement raw
    self-inclusive real-space response (complex), spin-sign-converted real
    response, direct ``Jq``, complex transformed values and pointwise
    residuals, all on the shared q/R/site order;
  * ``summary.json``   -- per-gate outcomes, residual metrics, worst q/site
    indices, near-zero treatment, refinement ledger (traceable to raw.npz);
  * ``report.md``      -- concise human-readable report of the same data.

Example smoke settings (small real five-site insulating Fe chain; the model is
real input, both routes run the actual library code)::

    settings = {
        "run_id": "smoke-fe-chain",
        "model": {"kind": "chain", "npoints": 5, "onsite_up_eV": -2.0,
                  "onsite_dn_eV": 2.0, "hopping_eV": -0.5,
                  "basis": ["Fe1|d"], "magnetic_site": 0},
        "physical": {"efermi_eV": 0.0, "response_temperature_K": 0.0,
                     "density_smearing_eV": 0.01, "emin_eV": -15.0, "emax_eV": 0.0},
        "grid": {"kmesh": [5, 1, 1], "qmesh": [5, 1, 1],
                 "nz_refinements": [129, 257]},
        "integration": {"method": "semicircle", "nproc": 1},
        "conventions": {...see REQUIRED_CONVENTIONS...},
        "acceptance": {"atol_eV": 1e-8, "rtol": 1e-6,
                       "near_zero_floor_eV": 1e-8, "real_gauge_tol_eV": 1e-8},
        "math_authorities": {
            "qspace-unrenormalized-kernel":
                {"path": "<.../mathdocs/qspace-unrenormalized-kernel.py>",
                 "sha256": "<64-hex of that authority file>",
                 "verified_at": "<utc>"},
            "qspace-finite-grid-transform":
                {"path": "<.../mathdocs/qspace-finite-grid-transform.py>",
                 "sha256": "<64-hex of that authority file>",
                 "verified_at": "<utc>"}},
        "source_commit": "<40-hex TB2J revision; the consumed checkout must be "
        "at this commit with a clean TB2J package tree>",
    }

For the actual SrMnO3 material study use ``"kind": "wannier"`` with the
durable data-repo fixture root, ``posfile="abinit.in"``, prefixes
``abinito_w90_up`` / ``abinito_w90_down``, ``magnetic_elements=["Mn"]``, the
seven declared input hashes, explicit ``efermi_eV`` (historical candidate
6.15) and the full ``[5, 5, 5]`` mesh.  Material calibration and the full
study belong to Story 006; this runner does not broaden the smoke into a
campaign.
"""

import hashlib
import inspect
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ase
import scipy
import HamiltonIO
import yaml
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack
from ase.io import read as ase_read
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from HamiltonIO.wannier import WannierHam

from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.utils import auto_assign_basis_name
from TB2J.wannier import parse_atoms
from TB2J import __version__ as _TB2J_VERSION, __file__ as _TB2J_FILE

__all__ = ["run_case", "PreflightRefusal"]

# Durable SrMnO3 fixture files whose identity must be declared and verified.
CORE_WANNIER_FILES = (
    "abinit.in",
    "abinito_w90_up_hr.dat",
    "abinito_w90_down_hr.dat",
    "abinito_w90_up.win",
    "abinito_w90_down.win",
    "abinito_w90_up_centres.xyz",
    "abinito_w90_down_centres.xyz",
)

# The single supported convention set, fixed by the declared MathDoc
# authorities.  Anything else is refused before calculation.
REQUIRED_CONVENTIONS = {
    "units": "eV",
    "ensemble": "fixed_chemical_potential",
    "self_term": "included",
    "q_to_r_phase": "minus",
    "forward_transform_phase": "plus",
    "forward_transform_normalization": "unity",
    "spin_sign_conversion": "divide_by_sign_spin_dot_spin",
}

REQUIRED_MATH_AUTHORITIES = (
    "qspace-unrenormalized-kernel",
    "qspace-finite-grid-transform",
)

SUPPORTED_INTEGRATION_METHODS = ("semicircle", "legendre")

ARTIFACT_NAMES = ("manifest.json", "raw.npz", "summary.json", "report.md")

_TOP_KEYS = {
    "run_id",
    "model",
    "physical",
    "grid",
    "integration",
    "conventions",
    "acceptance",
    "math_authorities",
    "source_commit",
    "description",
    "overwrite",
}
_MODEL_KEYS = {
    "wannier": {
        "kind",
        "input_root",
        "posfile",
        "prefix_up",
        "prefix_dn",
        "magnetic_elements",
        "input_hashes",
        "basis",
        "magnetic_site",
    },
    "chain": {
        "kind",
        "npoints",
        "onsite_up_eV",
        "onsite_dn_eV",
        "hopping_eV",
        "basis",
        "magnetic_site",
    },
}
_PHYSICAL_KEYS = {
    "efermi_eV",
    "response_temperature_K",
    "density_smearing_eV",
    "emin_eV",
    "emax_eV",
}
_GRID_KEYS = {"kmesh", "qmesh", "nz_refinements"}
_INTEGRATION_KEYS = {"method", "nproc"}
_ACCEPTANCE_KEYS = {"atol_eV", "rtol", "near_zero_floor_eV", "real_gauge_tol_eV"}


class PreflightRefusal(RuntimeError):
    """Invalid inputs, conventions or authority, refused before calculation."""

    def __init__(self, run_id, stage, reasons):
        self.run_id = run_id
        self.stage = stage
        self.reasons = list(reasons)
        super().__init__(
            f"preflight refusal for run '{run_id}' at stage '{stage}': "
            + "; ".join(self.reasons)
        )


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_num(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _is_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def _validate_settings(settings):
    """Static preflight: completeness, types, identities, conventions, grids."""
    run_id = settings.get("run_id")
    reasons = []
    if (
        not isinstance(run_id, str)
        or not run_id
        or set(run_id) & {"/", "\\"}
        or run_id in {".", ".."}
    ):
        reasons.append("run_id must be a non-empty path-safe string")
        run_id = None
    unknown = sorted(set(settings) - _TOP_KEYS)
    if unknown:
        reasons.append(f"unknown settings keys: {unknown}")
    for key in _TOP_KEYS - {"description", "overwrite"}:
        if key not in settings:
            reasons.append(f"missing required settings key '{key}'")
    if reasons:
        raise PreflightRefusal(run_id or "<unset>", "settings", reasons)

    for key in (
        "model",
        "physical",
        "grid",
        "integration",
        "conventions",
        "acceptance",
        "math_authorities",
    ):
        if not isinstance(settings[key], dict):
            reasons.append(f"settings['{key}'] must be a dict")
    if reasons:
        raise PreflightRefusal(run_id or "<unset>", "settings", reasons)

    if not isinstance(settings.get("overwrite", False), bool):
        reasons.append("overwrite must be a boolean")
    if not _is_hex(settings["source_commit"], 40):
        reasons.append("source_commit must be the 40-hex TB2J source revision")

    mcfg = settings["model"]
    kind = mcfg.get("kind")
    if kind not in _MODEL_KEYS:
        reasons.append(f"model.kind must be one of {sorted(_MODEL_KEYS)}")
    else:
        unknown = sorted(set(mcfg) - _MODEL_KEYS[kind])
        missing = sorted(_MODEL_KEYS[kind] - set(mcfg))
        if unknown:
            reasons.append(f"unknown model keys for kind '{kind}': {unknown}")
        if missing:
            reasons.append(f"missing model keys for kind '{kind}': {missing}")
        if kind == "wannier":
            if not isinstance(mcfg.get("input_root"), str) or not mcfg["input_root"]:
                reasons.append("model.input_root must be a non-empty string")
            for key in ("posfile", "prefix_up", "prefix_dn"):
                if not isinstance(mcfg.get(key), str) or not mcfg[key]:
                    reasons.append(f"model.{key} must be a non-empty string")
            elements = mcfg.get("magnetic_elements")
            if not (
                isinstance(elements, list)
                and len(elements) == 1
                and all(isinstance(e, str) for e in elements)
            ):
                reasons.append(
                    "model.magnetic_elements must declare exactly one element "
                    "(the declared derivations are one-magnetic-site scalar)"
                )
            hashes = mcfg.get("input_hashes")
            if not isinstance(hashes, dict):
                reasons.append("model.input_hashes must be a {relpath: sha256} dict")
            else:
                for name in CORE_WANNIER_FILES:
                    if name not in hashes:
                        reasons.append(f"model.input_hashes misses core file '{name}'")
                for name, digest in hashes.items():
                    if not _is_hex(digest, 64):
                        reasons.append(f"input hash for '{name}' must be 64-hex")
        else:
            for key in ("npoints",):
                if (
                    not isinstance(mcfg.get(key), int)
                    or isinstance(mcfg.get(key), bool)
                    or mcfg[key] < 1
                ):
                    reasons.append(f"model.{key} must be a positive integer")
            for key in ("onsite_up_eV", "onsite_dn_eV", "hopping_eV"):
                if not _is_num(mcfg.get(key)):
                    reasons.append(f"model.{key} must be a number in eV")

    phys = settings["physical"]
    unknown = sorted(set(phys) - _PHYSICAL_KEYS)
    missing = sorted(_PHYSICAL_KEYS - set(phys))
    if unknown:
        reasons.append(f"unknown physical keys: {unknown}")
    if missing:
        reasons.append(f"missing physical keys: {missing}")
    for key in _PHYSICAL_KEYS:
        if key in phys and not _is_num(phys[key]):
            reasons.append(f"physical.{key} must be a number")
    if (
        _is_num(phys.get("response_temperature_K"))
        and phys["response_temperature_K"] != 0
    ):
        reasons.append("geometric contour response_temperature_K must be zero")
    if _is_num(phys.get("density_smearing_eV")) and phys["density_smearing_eV"] <= 0:
        reasons.append("physical.density_smearing_eV must be positive")
    if _is_num(phys.get("emax_eV")) and phys["emax_eV"] != 0:
        reasons.append("physical.emax_eV must be zero relative to the Fermi level")
    if (
        _is_num(phys.get("emin_eV"))
        and _is_num(phys.get("emax_eV"))
        and phys["emin_eV"] >= phys["emax_eV"]
    ):
        reasons.append("physical.emin_eV must be below physical.emax_eV")
    if "description" in settings and not isinstance(settings["description"], str):
        reasons.append("description must be a string")

    grid = settings["grid"]
    unknown = sorted(set(grid) - _GRID_KEYS)
    missing = sorted(_GRID_KEYS - set(grid))
    if unknown:
        reasons.append(f"unknown grid keys: {unknown}")
    if missing:
        reasons.append(f"missing grid keys: {missing}")
    kmesh, qmesh = grid.get("kmesh"), grid.get("qmesh")
    for name, mesh in (("kmesh", kmesh), ("qmesh", qmesh)):
        if not (
            isinstance(mesh, list)
            and len(mesh) == 3
            and all(
                isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in mesh
            )
        ):
            reasons.append(f"grid.{name} must be three positive integers")
    if (
        isinstance(kmesh, list)
        and len(kmesh) == 3
        and all(isinstance(n, int) for n in kmesh)
    ):
        if any(n % 2 == 0 for n in kmesh):
            # The library silently forces odd meshes; a declared even mesh would
            # silently mismatch the effective grid.
            reasons.append(
                "grid.kmesh must be odd in each direction "
                "(the library forces odd meshes; even meshes are not declared here)"
            )
    if isinstance(kmesh, list) and kmesh == qmesh and isinstance(qmesh, list):
        pass
    elif (
        isinstance(kmesh, list)
        and isinstance(qmesh, list)
        and len(kmesh) == len(qmesh) == 3
    ):
        reasons.append(
            "grid.qmesh must equal grid.kmesh (k+q must map onto the same mesh)"
        )
    nzs = grid.get("nz_refinements")
    if not (
        isinstance(nzs, list)
        and nzs
        and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in nzs)
    ):
        reasons.append(
            "grid.nz_refinements must be a non-empty list of positive integers"
        )

    integ = settings["integration"]
    unknown = sorted(set(integ) - _INTEGRATION_KEYS)
    if unknown:
        reasons.append(f"unknown integration keys: {unknown}")
    if "nproc" not in integ:
        reasons.append("integration.nproc must be explicitly specified")
    if integ.get("method") not in SUPPORTED_INTEGRATION_METHODS:
        reasons.append(
            f"integration.method must be one of {list(SUPPORTED_INTEGRATION_METHODS)} "
            "(the raw q-space route requires a geometric contour)"
        )
    if "nproc" in integ and (
        not isinstance(integ["nproc"], int)
        or isinstance(integ["nproc"], bool)
        or integ["nproc"] < 1
    ):
        reasons.append("integration.nproc must be a positive integer")

    if settings["conventions"] != REQUIRED_CONVENTIONS:
        got = settings["conventions"]
        if not isinstance(got, dict):
            reasons.append("conventions must be a dict equal to REQUIRED_CONVENTIONS")
        else:
            for key in sorted(set(REQUIRED_CONVENTIONS) | set(got)):
                if REQUIRED_CONVENTIONS.get(key) != got.get(key):
                    reasons.append(
                        f"conventions['{key}'] must be "
                        f"{REQUIRED_CONVENTIONS.get(key)!r}, got {got.get(key)!r}"
                    )

    acc = settings["acceptance"]
    unknown = sorted(set(acc) - _ACCEPTANCE_KEYS)
    missing = sorted(_ACCEPTANCE_KEYS - set(acc))
    if unknown:
        reasons.append(f"unknown acceptance keys: {unknown}")
    if missing:
        reasons.append(f"missing acceptance keys: {missing}")
    for key in ("atol_eV", "near_zero_floor_eV", "real_gauge_tol_eV"):
        if _is_num(acc.get(key)) and acc[key] <= 0:
            reasons.append(f"acceptance.{key} must be positive")
        elif key in acc and not _is_num(acc[key]):
            reasons.append(f"acceptance.{key} must be a number")
    if "rtol" in acc and not _is_num(acc["rtol"]):
        reasons.append("acceptance.rtol must be a number")
    elif _is_num(acc.get("rtol")) and acc["rtol"] < 0:
        reasons.append("acceptance.rtol must be non-negative")

    authorities = settings["math_authorities"]
    if not isinstance(authorities, dict):
        reasons.append("math_authorities must map derivation names to their identities")
    else:
        unknown = sorted(set(authorities) - set(REQUIRED_MATH_AUTHORITIES))
        missing = sorted(set(REQUIRED_MATH_AUTHORITIES) - set(authorities))
        if unknown:
            reasons.append(f"unexpected math authorities: {unknown}")
        if missing:
            reasons.append(f"missing math authorities: {missing}")
        for name in REQUIRED_MATH_AUTHORITIES:
            ident = authorities.get(name)
            if isinstance(ident, dict):
                if not isinstance(ident.get("path"), str) or not ident["path"]:
                    reasons.append(
                        f"math_authorities['{name}'].path must be a non-empty string"
                    )
                if not _is_hex(ident.get("sha256"), 64):
                    reasons.append(f"math_authorities['{name}'].sha256 must be 64-hex")
                if (
                    not isinstance(ident.get("verified_at"), str)
                    or not ident["verified_at"]
                ):
                    reasons.append(
                        f"math_authorities['{name}'].verified_at must be a string"
                    )
            elif name in authorities:
                reasons.append(f"math_authorities['{name}'] must be a mapping")
    if reasons:
        raise PreflightRefusal(settings["run_id"], "settings", reasons)

    if kind == "chain":
        if grid["kmesh"] != [mcfg["npoints"], 1, 1]:
            raise PreflightRefusal(
                settings["run_id"],
                "settings",
                [
                    f"chain model requires grid.kmesh == [{mcfg['npoints']}, 1, 1], "
                    f"got {grid['kmesh']}"
                ],
            )
    return settings


def _check_evidence_slot(settings, output_root):
    """Refuse any implicit overwrite of previously written evidence."""
    run_dir = Path(output_root) / settings["run_id"]
    existing = [name for name in ARTIFACT_NAMES if (run_dir / name).exists()]
    if existing and not settings.get("overwrite", False):
        raise PreflightRefusal(
            settings["run_id"],
            "evidence-slot",
            [
                f"evidence already exists in '{run_dir}': {existing}; "
                f"set overwrite=true to replace it explicitly"
            ],
        )
    return run_dir


def _verify_source_identity(settings):
    """Verify the consumed TB2J checkout: exact declared commit, clean package
    tree.  Refuses when the identity cannot be established."""
    package_dir = Path(_TB2J_FILE).resolve().parent
    repo = package_dir.parent
    declared = settings["source_commit"].lower()
    try:
        head = (
            subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .lower()
        )
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", package_dir.name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightRefusal(
            settings["run_id"],
            "source-identity",
            [f"cannot verify consumed source identity of '{repo}': {exc}"],
        )
    reasons = []
    if head != declared:
        reasons.append(
            f"source_commit mismatch: declared {declared}, consumed checkout HEAD {head}"
        )
    dirty = [line for line in status if not line.startswith("??")]
    if dirty:
        reasons.append(
            f"consumed source '{package_dir.name}/' is dirty: {dirty[:5]}"
            + (f" (+{len(dirty) - 5} more)" if len(dirty) > 5 else "")
        )
    if reasons:
        raise PreflightRefusal(settings["run_id"], "source-identity", reasons)
    return {
        "observed_commit": head,
        "package_dir": str(package_dir),
        "dirty_consumed_source": [],
        "untracked": [line[3:] for line in status if line.startswith("??")],
    }


def _verify_math_authorities(settings):
    """Verify each declared math authority file against its declared SHA-256."""
    reasons = []
    verified = {}
    for name, ident in settings["math_authorities"].items():
        path = Path(ident["path"])
        canonical = (
            Path("/home/hexu/projects/memnotes/memnotes/Projects/TB2J/mathdocs")
            / f"{name}.py"
        )
        if path.resolve() != canonical.resolve():
            reasons.append(
                f"math authority '{name}' is not the canonical Project authority"
            )
            continue
        if not path.is_file():
            reasons.append(f"math authority '{name}' file '{path}' does not exist")
            continue
        actual = _sha256(path)
        try:
            metadata = yaml.safe_load(
                path.with_suffix(".md").read_text().split("---", 2)[1]
            )
        except (OSError, IndexError, yaml.YAMLError) as exc:
            reasons.append(f"math authority '{name}' metadata unavailable: {exc}")
            continue
        if (
            not isinstance(metadata, dict)
            or metadata.get("status") != "stable"
            or metadata.get("human_review") != "approved"
            or metadata.get("source_sha256") != actual
        ):
            reasons.append(
                f"math authority '{name}' is not approved, stable, and pinned to its source"
            )
            continue
        if actual != ident["sha256"].lower():
            reasons.append(
                f"math authority '{name}' hash mismatch for '{path}': declared "
                f"{ident['sha256']}, observed {actual}"
            )
            continue
        verified[name] = {
            "path": str(path),
            "sha256": actual,
            "verified_at": ident["verified_at"],
        }
    if reasons:
        raise PreflightRefusal(settings["run_id"], "math-authorities", reasons)
    return verified


def _check_inputs(settings):
    """Verify declared input identities by SHA-256 before any model load."""
    mcfg = settings["model"]
    if mcfg["kind"] != "wannier":
        return None
    root = Path(mcfg["input_root"])
    reasons = []
    if (mcfg["posfile"], mcfg["prefix_up"], mcfg["prefix_dn"]) != (
        "abinit.in",
        "abinito_w90_up",
        "abinito_w90_down",
    ):
        reasons.append("this study requires the pinned SrMnO3 structure and prefixes")
    for prefix in ("abinito_w90_up", "abinito_w90_down"):
        for suffix in ("_tb.dat", "_wsvec.dat"):
            name = prefix + suffix
            if (root / name).exists() and name not in mcfg["input_hashes"]:
                reasons.append(f"loader input '{name}' exists but is not hashed")
    if not root.is_dir():
        reasons.append(f"model.input_root '{root}' is not a directory")
    else:
        for name in sorted(mcfg["input_hashes"]):
            path = root / name
            if not path.is_file():
                reasons.append(f"declared input file '{name}' is missing from '{root}'")
                continue
            actual = _sha256(path)
            if actual != mcfg["input_hashes"][name].lower():
                reasons.append(
                    f"input hash mismatch for '{name}': declared "
                    f"{mcfg['input_hashes'][name]}, observed {actual}"
                )
    if reasons:
        raise PreflightRefusal(settings["run_id"], "inputs", reasons)
    return root


def _load_models_wannier(mcfg, run_id):
    """Load the spin-up/down Wannier Hamiltonians and basis exactly as the
    production WannierManager does, for the durable pinned fixture."""
    root = Path(mcfg["input_root"])
    try:
        atoms = ase_read(root / mcfg["posfile"])
    except Exception:
        atoms = parse_atoms(str(root / f"{mcfg['prefix_up']}.win"))
    symbols = atoms.get_chemical_symbols()
    mag_sites = [i for i, s in enumerate(symbols) if s in mcfg["magnetic_elements"]]
    if len(mag_sites) != 1:
        raise PreflightRefusal(
            run_id,
            "inputs",
            [
                f"declared magnetic elements {mcfg['magnetic_elements']} resolve to "
                f"{len(mag_sites)} sites {mag_sites}; the declared derivations cover "
                "exactly one magnetic site"
            ],
        )
    model_up = WannierHam.read_from_wannier_dir(
        path=str(root), prefix=mcfg["prefix_up"], atoms=atoms, nls=False
    )
    model_dn = WannierHam.read_from_wannier_dir(
        path=str(root), prefix=mcfg["prefix_dn"], atoms=atoms, nls=False
    )
    basis, _ = auto_assign_basis_name(model_up.xred, atoms)
    basis_dn, _ = auto_assign_basis_name(model_dn.xred, atoms)
    if len(model_up.xred) != len(model_dn.xred) or list(basis) != list(basis_dn):
        raise PreflightRefusal(
            run_id,
            "mapping",
            ["spin channels have incompatible ordered orbital/site mappings"],
        )
    return (model_up, model_dn), atoms, basis


def _load_models_chain(mcfg):
    """Build the small real insulating-chain model: one orthogonal orbital per
    site, eV, nearest-neighbour hopping, up occupied / down empty."""
    atoms = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 10.0, 10.0])
    rlist = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])

    def model(onsite):
        h = mcfg["hopping_eV"]
        return LCAOHamiltonian(
            HR=np.array([[[h]], [[onsite]], [[h]]]),
            SR=np.array([[[0.0]], [[1.0]], [[0.0]]]),
            Rlist=rlist,
            nbasis=1,
            atoms=atoms,
            nspin=1,
            orth=True,
        )

    models = (model(mcfg["onsite_up_eV"]), model(mcfg["onsite_dn_eV"]))
    return models, atoms, ["Fe1|d"]


def _model_identity(mcfg):
    if mcfg["kind"] == "chain":
        block = {
            k: mcfg[k]
            for k in ("kind", "npoints", "onsite_up_eV", "onsite_dn_eV", "hopping_eV")
        }
        canonical = json.dumps(block, sort_keys=True, separators=(",", ":"))
        return {"kind": "chain", "canonical_json_sha256": _sha256_of_text(canonical)}
    return {
        "kind": "wannier",
        "input_root": str(Path(mcfg["input_root"]).resolve()),
        "posfile": mcfg["posfile"],
        "prefix_up": mcfg["prefix_up"],
        "prefix_dn": mcfg["prefix_dn"],
        "magnetic_elements": list(mcfg["magnetic_elements"]),
        "input_hashes": dict(mcfg["input_hashes"]),
    }


def _sha256_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_calculator(cls, models, atoms, basis, settings, run_dir, nz):
    phys = settings["physical"]
    integ = settings["integration"]
    mcfg = settings["model"]
    return cls(
        tbmodels=models,
        atoms=atoms,
        basis=basis,
        # The chain fixture is a single Fe site; wannier declares its element.
        magnetic_elements=list(mcfg.get("magnetic_elements") or ["Fe"]),
        kmesh=list(settings["grid"]["kmesh"]),
        efermi=phys["efermi_eV"],
        # Geometric contour response is zero-temperature; this width only affects density.
        smearing=phys["density_smearing_eV"],
        emin=phys["emin_eV"],
        emax=phys["emax_eV"],
        nz=nz,
        nproc=integ.get("nproc", 1),
        # No physical R cutoff in the primary discrete comparison (ADR-4).
        Rcut=None,
        method=integ["method"],
        output_path=str(run_dir),
        description=settings.get("description", "private SrMnO3 validation run"),
    )


def _model_hoppings(model):
    """Return [(R tuple, matrix)] for every stored R of either supported model
    type (WannierHam keeps a per-R dict, LCAO a dense HR array)."""
    data = getattr(model, "data", None)
    if data is not None:
        return [(tuple(int(x) for x in R), np.asarray(mat)) for R, mat in data.items()]
    hr = np.asarray(model.HR)
    return [
        (tuple(int(x) for x in r), hr[i]) for i, r in enumerate(np.asarray(model.Rlist))
    ]


def _real_gauge_reasons(models, tol):
    """The declared reciprocity premise concerns every R, not only R=0:
    both spin channels must have real hoppings at all R and obey transpose
    reciprocity H(R) = H(-R)^T within the declared tolerance."""
    reasons = []
    for channel, model in (("up", models[0]), ("down", models[1])):
        hoppings = _model_hoppings(model)
        imaginary = max(
            (float(np.max(np.abs(mat.imag))) for _, mat in hoppings),
            default=0.0,
        )
        if imaginary > tol:
            reasons.append(
                f"{channel} channel has max |Im H(R)| {imaginary:.3e} eV over all "
                f"R, above real_gauge_tol_eV {tol}; complex-gauge reciprocity is "
                "outside the declared authority"
            )
        index = {r: mat for r, mat in hoppings}
        for r, mat in hoppings:
            partner = index.get(tuple(-x for x in r))
            if partner is None:
                reasons.append(
                    f"{channel} channel has no -R partner for R={r}; transpose "
                    "reciprocity cannot be established"
                )
                break
            deviation = float(np.max(np.abs(mat - partner.T)))
            if deviation > tol:
                reasons.append(
                    f"{channel} channel violates transpose reciprocity at R={r}: "
                    f"max |H(R) - H(-R)^T| {deviation:.3e} eV above "
                    f"real_gauge_tol_eV {tol}"
                )
                break
    return reasons


def _preflight_model_contract(models, calc_r, calc_q, settings):
    """Refuse cases outside the declared derivation domain, before any
    response calculation.  The declared authorities cover the one-magnetic-site
    scalar ferromagnetic case with a real, orthogonal orbital gauge."""
    run_id = settings["run_id"]
    tol = settings["acceptance"]["real_gauge_tol_eV"]
    reasons = []
    for name, calc in (("real-space", calc_r), ("q-space", calc_q)):
        if len(calc.ind_mag_atoms) != 1:
            reasons.append(
                f"{name} route resolved {len(calc.ind_mag_atoms)} magnetic sites "
                f"{list(calc.ind_mag_atoms)}; the declared derivations are "
                "one-magnetic-site scalar"
            )
        if not calc.is_orthogonal:
            reasons.append(
                f"{name} route Hamiltonian is not orthogonal; the declared "
                "real-gauge reciprocity requires a common orthonormal basis"
            )
    reasons.extend(_real_gauge_reasons(models, tol))
    if not reasons:
        delta = calc_r.Delta
        if np.max(np.abs(delta - delta.T)) > tol:
            reasons.append(
                "local splitting Delta is not symmetric within real_gauge_tol_eV; "
                "the declared even-q reciprocity requires symmetric splittings "
                "in the real gauge"
            )
        # The moment enters the spin-sign conversion; a momentless reference is
        # outside the ferromagnetic even-q contract.
        m = calc_r.ind_mag_atoms[0]
        orb = calc_r.iorb(m)
        moment = float(
            np.real(
                np.trace(calc_r.rho_up[np.ix_(orb, orb)])
                - np.trace(calc_r.rho_dn[np.ix_(orb, orb)])
            )
        )
        if abs(moment) <= 1e-8:
            reasons.append(
                f"magnetic site {m} has near-zero moment {moment:.3e}; the "
                "spin-sign conversion and the even-q contract require a "
                "ferromagnetic reference"
            )
        # The q-space grid attributes exist only after _prepare(); preparing
        # here inspects the declared grid before any response integration.
        calc_q._prepare()
        if not np.array_equal(
            calc_q.qpts, monkhorst_pack(size=list(settings["grid"]["kmesh"]))
        ):
            reasons.append("q-space route q-points do not match the declared mesh")
        r_r = sorted(calc_r.Rlist)
        r_q = sorted(calc_q.Rlist)
        if r_r != r_q:
            reasons.append(
                "real-space and q-space routes resolved different R sets; "
                f"{len(r_r)} vs {len(r_q)} representatives"
            )
        qpts = np.asarray(calc_q.qpts, dtype=float)
        for iq, q in enumerate(qpts):
            if not any(np.allclose(nq, -q, atol=1e-9) for nq in qpts):
                reasons.append(
                    f"q point {q.tolist()} has no -q partner; the mesh is not "
                    "closed under negation"
                )
                break
    if reasons:
        raise PreflightRefusal(run_id, "model-contract", reasons)


def _collect_real_space(calc):
    """Self-inclusive real-space response with the production spin-sign
    conversion, retained raw (complex) before conversion."""
    ind = list(calc.ind_mag_atoms)
    rlist = sorted(calc.Rlist)
    ns = len(ind)
    jjr = np.zeros((len(rlist), ns, ns), dtype=complex)
    jr = np.zeros((len(rlist), ns, ns), dtype=float)
    spinat = np.asarray(calc.spinat, dtype=float)
    signs = {}
    for i, ia in enumerate(ind):
        for j, ja in enumerate(ind):
            key = float(np.dot(spinat[ia], spinat[ja]))
            if key == 0.0:
                raise PreflightRefusal(
                    "<real-space>",
                    "model-contract",
                    [
                        f"spin dot product of sites {ia},{ja} is zero; spin-sign "
                        "conversion undefined"
                    ],
                )
            signs[(i, j)] = np.sign(key)
    for ir, rvec in enumerate(rlist):
        for i, ia in enumerate(ind):
            for j, ja in enumerate(ind):
                key = (rvec, ia, ja)
                if key not in calc.JJ:
                    raise PreflightRefusal(
                        "<real-space>",
                        "model-contract",
                        [
                            f"incomplete real-space response: missing {key}; the "
                            "complete-grid comparison requires every representative"
                        ],
                    )
                val = complex(calc.JJ[key])
                jjr[ir, i, j] = val
                jr[ir, i, j] = np.imag(val) / signs[(i, j)]
    return rlist, ind, jjr, jr


def _forward_transform(jr, rlist, qpts):
    """Complete finite-grid forward transform: the exact inverse of the
    production minus-phase q_to_r.  The forward transform carries no extra
    factor; the 1/nqpt normalization belongs to the inverse only.  The result
    is kept complex; imaginary components are quantified, never discarded."""
    # MathDoc: tb2j/qspace-finite-grid-transform
    phase = np.exp(2.0j * np.pi * qpts @ np.asarray(rlist, dtype=float).T)
    return np.einsum("qr,rij->qij", phase, jr)


def _pair_negation_indices(qpts):
    pairs = []
    for iq, q in enumerate(qpts):
        for jq in range(iq + 1, len(qpts)):
            if np.allclose(qpts[jq], -q, atol=1e-9):
                pairs.append((iq, jq))
    return pairs


def _evaluate_refinement(models, atoms, basis, settings, run_dir, nz):
    """Run both actual library routes at one contour resolution and compare."""
    calc_r = _make_calculator(ExchangeCL2, models, atoms, basis, settings, run_dir, nz)
    calc_q = _make_calculator(
        ExchangeCLQspace, models, atoms, basis, settings, run_dir, nz
    )
    _preflight_model_contract(models, calc_r, calc_q, settings)

    calc_r.calculate_all()
    rlist, ind, jjr, jr = _collect_real_space(calc_r)

    # MathDoc: tb2j/qspace-unrenormalized-kernel
    # Direct raw reciprocal-space response, unrenormalized, no Bruno term.
    jq_direct = np.asarray(calc_q.calculate_raw(), dtype=float)

    qpts = np.asarray(calc_q.qpts, dtype=float)
    jq_trans = _forward_transform(jr, rlist, qpts)
    # Complex residual: an unexpected imaginary component enters the gate
    # magnitude instead of being silently discarded by a real-part projection.
    residual = jq_trans - jq_direct.astype(complex)

    acc = settings["acceptance"]
    atol, rtol, floor = acc["atol_eV"], acc["rtol"], acc["near_zero_floor_eV"]
    mag = np.abs(jq_direct)
    absres = np.abs(residual)
    allowed = np.where(mag > floor, atol + rtol * mag, atol)
    route_ok = bool(np.all(absres <= allowed))
    near_zero = int(np.count_nonzero(mag <= floor))
    above = mag > floor
    max_rel = float(np.max(absres[above] / mag[above])) if np.any(above) else 0.0
    imax = int(np.argmax(absres))
    iq_w, i_w, j_w = np.unravel_index(imax, absres.shape)
    even_pairs = _pair_negation_indices(qpts)
    even_dev = [
        float(np.max(np.abs(jq_direct[a] - jq_direct[b]))) for a, b in even_pairs
    ]
    even_allowed = [
        atol + rtol * float(np.max(np.maximum(mag[a], mag[b]))) for a, b in even_pairs
    ]
    even_ok = all(d <= g for d, g in zip(even_dev, even_allowed))
    i_r0 = int(np.argmin(np.linalg.norm(np.asarray(rlist, dtype=float), axis=1)))
    i_gamma = int(np.argmin(np.linalg.norm(qpts, axis=1)))

    ledger = {
        "nz": nz,
        "resolved_emin_eV": float(calc_r.contour.emin),
        "resolved_emax_eV": float(calc_r.contour.emax),
        "route_pass": route_ok,
        "reciprocity_even_q_pass": even_ok,
        "max_abs_residual_eV": float(absres.max()),
        "rms_residual_eV": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "max_relative_residual_above_floor": max_rel,
        "near_zero_count": near_zero,
        "transformed_imag_max_eV": float(np.max(np.abs(jq_trans.imag))),
        "even_q_max_deviation_eV": float(max(even_dev)) if even_dev else 0.0,
        "self_term_real_space_eV": float(jr[i_r0, 0, 0]),
        "gamma_direct_eV": float(jq_direct[i_gamma, 0, 0]),
        "gamma_transformed_eV": float(jq_trans[i_gamma, 0, 0].real),
    }
    arrays = {
        "JJR_raw": jjr,
        "JR_converted": jr,
        "Jq_direct": jq_direct,
        "Jq_transformed": jq_trans,
        "residual": residual,
    }
    worst = {
        "iq": int(iq_w),
        "q": qpts[iq_w].tolist(),
        "i": int(i_w),
        "j": int(j_w),
        "site_atoms": [int(ind[i_w]), int(ind[j_w])],
        "direct_eV": float(jq_direct[iq_w, i_w, j_w]),
        "transformed_eV": float(jq_trans[iq_w, i_w, j_w].real),
        "transformed_imag_eV": float(jq_trans[iq_w, i_w, j_w].imag),
        "residual_eV": float(absres[iq_w, i_w, j_w]),
    }
    return (
        ledger,
        arrays,
        worst,
        {
            "qpts": qpts,
            "rlist": rlist,
            "ind": ind,
            "spinat": np.asarray(calc_r.spinat, dtype=float),
            "labels": {int(a): list(calc_r.labels[a]) for a in ind},
        },
    )


def _environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "ase": ase.__version__,
        "scipy": scipy.__version__,
        "hamiltonio_version": importlib.metadata.version("HamiltonIO"),
        "hamiltonio_package_root": str(Path(HamiltonIO.__file__).resolve().parent),
        "hamiltonio_python_files": {
            str(path.relative_to(Path(HamiltonIO.__file__).resolve().parent)): _sha256(
                path
            )
            for path in sorted(Path(HamiltonIO.__file__).resolve().parent.rglob("*.py"))
        },
        "hamiltonio_sources": {
            cls.__name__: {
                "path": inspect.getfile(cls),
                "sha256": _sha256(inspect.getfile(cls)),
            }
            for cls in (WannierHam, LCAOHamiltonian)
        },
        "tb2j_version": _TB2J_VERSION,
        "tb2j_file": str(_TB2J_FILE),
    }


def _write_json(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")


def _write_refusal_record(output_root, run_id, stage, reasons, raw_settings):
    run_id = run_id or "<unset>"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_root) / f"refusal-{stamp}.json"
    payload = {
        "status": "refused",
        "run_id": run_id,
        "stage": stage,
        "reasons": list(reasons),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings_as_given": raw_settings,
        "environment": _environment(),
    }
    try:
        Path(output_root).mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
        return str(path)
    except OSError as exc:
        return f"<refusal record could not be written: {exc}>"


def _report_text(run_id, status, manifest, summary):
    lines = [
        "# SrMnO3 reciprocal-space validation run report",
        "",
        f"- Run: `{run_id}`",
        f"- Outcome: **{status}**",
        f"- Model: {manifest['model']['kind']}",
        f"- Primary contour resolution: nz={summary['primary_nz']}",
        "- Units: eV for all saved values (meV quoted in parentheses where noted)",
        "",
        "## Gates (primary refinement)",
    ]
    for gate, ok in summary["gates"].items():
        lines.append(f"- {gate}: {'PASS' if ok else 'FAIL'}")
    lines += [
        "",
        "## Residual metrics (primary refinement, pointwise complex magnitude)",
        f"- max |residual|: {summary['max_abs_residual_eV']:.6e} eV "
        f"({summary['max_abs_residual_eV'] * 1e3:.6e} meV)",
        f"- RMS residual: {summary['rms_residual_eV']:.6e} eV",
        f"- max relative residual (above floor {summary['near_zero_floor_eV']:.3e} eV): "
        f"{summary['max_relative_residual_above_floor']:.3e}",
        f"- near-zero |Jq| entries at or below the floor: {summary['near_zero_count']}",
        "",
        "## Worst q/site entry (primary refinement)",
        f"- q index {summary['worst']['iq']} at q={summary['worst']['q']}, "
        f"site pair ({summary['worst']['site_atoms'][0]}, {summary['worst']['site_atoms'][1]})",
        f"- direct {summary['worst']['direct_eV']:.9e} eV, transformed "
        f"{summary['worst']['transformed_eV']:.9e} eV (imag "
        f"{summary['worst']['transformed_imag_eV']:.3e} eV), residual magnitude "
        f"{summary['worst']['residual_eV']:.3e} eV",
        "",
        "## Diagnostics",
        f"- max |Im(transformed)|: {summary['transformed_imag_max_eV']:.3e} eV "
        "(complex transform retained in raw.npz; not silently discarded)",
        f"- max |Jq(q) - Jq(-q)|: {summary['even_q_max_deviation_eV']:.3e} eV",
        f"- onsite self term (real space, R=0): {summary['self_term_real_space_eV']:.9e} eV",
        f"- Gamma direct: {summary['gamma_direct_eV']:.9e} eV, Gamma transformed: "
        f"{summary['gamma_transformed_eV']:.9e} eV (self term retained; no Gamma subtraction)",
        "",
        "## Refinement ledger",
    ]
    for entry in summary["refinement_ledger"]:
        lines.append(
            f"- nz={entry['nz']}: max |res| {entry['max_abs_residual_eV']:.3e} eV, "
            f"route {'PASS' if entry['route_pass'] else 'FAIL'}, even-q "
            f"{'PASS' if entry['reciprocity_even_q_pass'] else 'FAIL'}"
        )
    lines += [
        "",
        "## Limitations",
        "- Mesh representatives describe periodic-image alias classes, not "
        "infinite-range convergence; no physical-R cutoff is applied here.",
        "- One-magnetic-site scalar scope under the declared MathDoc authorities; "
        "no multi-sublattice, SOC, or AFM-reference claim.",
        "- Route consistency is not independent physical validation of SrMnO3 "
        "exchange, DFT/Wannier convergence, or a historical reproduction.",
        "",
        "## Provenance",
        f"- source_commit (declared): {manifest['source']['declared_commit']}",
        f"- TB2J {manifest['environment']['tb2j_version']} at "
        f"{manifest['environment']['tb2j_file']}",
        f"- created: {manifest['created_utc']}",
        "",
    ]
    return "\n".join(lines)


def run_case(settings, output_root):
    """Execute one private validation case and write its evidence bundle.

    Returns a compact result summary dict with ``status`` ``"pass"`` or
    ``"failed"``.  Refusals raise :class:`PreflightRefusal` after writing a
    labelled refusal record under ``output_root``.  Library runtime errors
    (for example a pole on the contour) propagate unchanged.
    """
    raw_settings = settings
    output_root = Path(output_root)
    try:
        settings = _validate_settings(dict(settings))
        run_id = settings["run_id"]
        run_dir = _check_evidence_slot(settings, output_root)
        source_identity = _verify_source_identity(settings)
        authorities = _verify_math_authorities(settings)
        _check_inputs(settings)
        mcfg = settings["model"]
        if mcfg["kind"] == "wannier":
            models, atoms, basis = _load_models_wannier(mcfg, run_id)
        else:
            models, atoms, basis = _load_models_chain(mcfg)
        if list(basis) != mcfg["basis"] or [
            i
            for i, atom in enumerate(atoms)
            if atom.symbol in mcfg.get("magnetic_elements", ["Fe"])
        ] != [mcfg["magnetic_site"]]:
            raise PreflightRefusal(
                run_id,
                "mapping",
                ["resolved basis or magnetic site differs from the explicit mapping"],
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        nzs = list(settings["grid"]["nz_refinements"])
        ledgers, stack = (
            [],
            {
                name: []
                for name in (
                    "JJR_raw",
                    "JR_converted",
                    "Jq_direct",
                    "Jq_transformed",
                    "residual",
                )
            },
        )
        grid_info = None
        worst = None
        for nz in nzs:
            ledger, arrays, worst_nz, grid_info = _evaluate_refinement(
                models, atoms, basis, settings, run_dir, nz
            )
            ledgers.append(ledger)
            for name, arr in arrays.items():
                stack[name].append(arr)
            worst = worst_nz
    except PreflightRefusal as refusal:
        known = refusal.run_id and not str(refusal.run_id).startswith("<")
        if not known and isinstance(raw_settings, dict):
            candidate = raw_settings.get("run_id")
            if isinstance(candidate, str) and candidate:
                refusal.run_id = candidate
        refusal.record = _write_refusal_record(
            output_root, refusal.run_id, refusal.stage, refusal.reasons, raw_settings
        )
        raise
    primary = ledgers[-1]
    gates = {
        "route_agreement": bool(primary["route_pass"]),
        "reciprocity_even_q": bool(primary["reciprocity_even_q_pass"]),
    }
    status = "pass" if all(gates.values()) else "failed"

    qpts = grid_info["qpts"]
    rlist = grid_info["rlist"]
    ind = grid_info["ind"]
    np.savez(
        run_dir / "raw.npz",
        qpts=qpts,
        R=np.asarray(rlist, dtype=int),
        site_indices=np.asarray(ind, dtype=int),
        nz_values=np.asarray(nzs, dtype=int),
        **{name: np.asarray(values) for name, values in stack.items()},
    )

    summary = {
        "status": status,
        "run_id": run_id,
        "output_dir": str(run_dir),
        "primary_nz": int(nzs[-1]),
        "gates": gates,
        "refinement_ledger": ledgers,
        "max_abs_residual_eV": primary["max_abs_residual_eV"],
        "rms_residual_eV": primary["rms_residual_eV"],
        "max_relative_residual_above_floor": primary[
            "max_relative_residual_above_floor"
        ],
        "near_zero_count": primary["near_zero_count"],
        "near_zero_floor_eV": settings["acceptance"]["near_zero_floor_eV"],
        "atol_eV": settings["acceptance"]["atol_eV"],
        "rtol": settings["acceptance"]["rtol"],
        "worst": worst,
        "transformed_imag_max_eV": primary["transformed_imag_max_eV"],
        "even_q_max_deviation_eV": primary["even_q_max_deviation_eV"],
        "self_term_real_space_eV": primary["self_term_real_space_eV"],
        "gamma_direct_eV": primary["gamma_direct_eV"],
        "gamma_transformed_eV": primary["gamma_transformed_eV"],
    }

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "settings": settings,
        "model": _model_identity(mcfg),
        "source": {
            "declared_commit": settings["source_commit"],
            "observed_commit": source_identity["observed_commit"],
            "package_dir": source_identity["package_dir"],
            "dirty_consumed_source": source_identity["dirty_consumed_source"],
            "untracked": source_identity["untracked"],
            "tb2j_version": _TB2J_VERSION,
            "tb2j_file": str(_TB2J_FILE),
            "runner_sha256": _sha256(Path(__file__)),
        },
        "math_authorities": authorities,
        "conventions": dict(REQUIRED_CONVENTIONS),
        "grid": {
            "kmesh": list(settings["grid"]["kmesh"]),
            "qmesh": list(settings["grid"]["qmesh"]),
            "nz_refinements": nzs,
            "nqpt": int(len(qpts)),
            "nR": int(len(rlist)),
            "qpts_order": "ase monkhorst_pack(size=kmesh), unshifted",
            "R_order": "sorted kmesh_to_R integer tuples, lexicographic",
            "rcut": None,
        },
        "sites": {
            "indices": [int(a) for a in ind],
            "orbital_labels": grid_info["labels"],
            "spinat_z": [float(grid_info["spinat"][a, 2]) for a in ind],
        },
        "units": {"exchange": "eV", "energy": "eV", "report_also_quotes": "meV"},
        "ensemble": REQUIRED_CONVENTIONS["ensemble"],
        "array_shapes": {
            "qpts": [len(qpts), 3],
            "R": [len(rlist), 3],
            "JJR_raw": [len(nzs), len(rlist), len(ind), len(ind)],
            "JR_converted": [len(nzs), len(rlist), len(ind), len(ind)],
            "Jq_direct": [len(nzs), len(qpts), len(ind), len(ind)],
            "Jq_transformed": [len(nzs), len(qpts), len(ind), len(ind)],
            "residual": [len(nzs), len(qpts), len(ind), len(ind)],
        },
        "dtypes": {
            "JJR_raw": "complex128",
            "JR_converted": "float64",
            "Jq_direct": "float64",
            "Jq_transformed": "complex128",
            "residual": "complex128",
        },
        "artifacts": list(ARTIFACT_NAMES),
        "environment": _environment(),
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(_report_text(run_id, status, manifest, summary))
    _write_json(run_dir / "manifest.json", manifest)
    return summary
