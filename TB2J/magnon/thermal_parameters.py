"""Parameters for temperature-dependent magnon calculations.

Implements the flat ``thermal_*`` TOML surface agreed in the thermal-magnon
architecture: method, spin regime, order mode, physical dimensionality,
explicit temperatures, q-mesh sequence, and nonlinear-solver controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import tomli
import tomli_w

THERMAL_METHODS = ("rpa", "callen", "hp", "rpa_callen", "mfa")
THERMAL_METHOD_LABELS = {
    "rpa": "RPA (Tyablikov)",
    "callen": "Callen decoupling",
    "hp": "HP mean field",
    "rpa_callen": "RPA+CD (RPA with Callen SIA)",
    "mfa": "Weiss mean-field approximation (MFA)",
}
SPIN_REGIMES = ("quantum", "classical")
ORDER_MODES = ("ferromagnetic", "bipartite_afm")


def _validate_qmeshes(meshes, dimensionality: int) -> List[List[int]]:
    cleaned = []
    previous = None
    for mesh in meshes:
        if len(mesh) != 3 or not all(int(m) >= 1 for m in mesh):
            raise ValueError(
                f"each entry of thermal_qmeshes must be 3 integers >= 1, got {mesh}"
            )
        current = [int(m) for m in mesh]
        if previous is not None and all(c <= p for c, p in zip(current, previous)):
            raise ValueError(
                "thermal_qmeshes must strictly increase (at least one "
                f"direction per step); got {previous} then {current}"
            )
        cleaned.append(current)
        previous = current
    if dimensionality is not None:
        for mesh in cleaned:
            inactive = mesh[dimensionality:]
            if dimensionality < 3 and any(m != 1 for m in inactive):
                raise ValueError(
                    f"q-mesh {mesh} has divisions outside the declared "
                    f"{dimensionality}D periodicity; use 1 in the aperiodic "
                    "directions"
                )
    return cleaned


@dataclass
class ThermalMagnonParameters:
    """Configuration of a thermal-magnon calculation.

    All fields serialize flat into TOML alongside the existing
    ``MagnonParameters`` fields (``thermal_`` prefix).
    """

    thermal_method: str = "rpa"
    thermal_spin_regime: str = "quantum"
    thermal_spin: Optional[List[float]] = None
    thermal_order_mode: str = "ferromagnetic"
    thermal_dimensionality: int = 3
    thermal_temperatures: List[float] = field(default_factory=list)
    thermal_qmeshes: List[List[int]] = field(
        default_factory=lambda: [[8, 8, 8], [12, 12, 12]]
    )
    thermal_mesh_tolerance: float = 1.0
    thermal_strict: bool = False

    thermal_max_iterations: int = 2000
    thermal_mixing: float = 0.5
    thermal_tolerance: float = 1e-10

    def __post_init__(self):
        if self.thermal_method not in THERMAL_METHODS:
            raise ValueError(
                f"thermal_method must be one of {THERMAL_METHODS}, "
                f"got {self.thermal_method!r}"
            )
        if self.thermal_spin_regime not in SPIN_REGIMES:
            raise ValueError(
                f"thermal_spin_regime must be one of {SPIN_REGIMES}, "
                f"got {self.thermal_spin_regime!r}"
            )
        if self.thermal_order_mode not in ORDER_MODES:
            raise ValueError(
                f"thermal_order_mode must be one of {ORDER_MODES}, "
                f"got {self.thermal_order_mode!r}"
            )
        if self.thermal_dimensionality not in (1, 2, 3):
            raise ValueError(
                "thermal_dimensionality must be 1, 2, or 3, got "
                f"{self.thermal_dimensionality}"
            )
        if self.thermal_spin is not None:
            if len(self.thermal_spin) == 0:
                raise ValueError("thermal_spin must be non-empty when provided")
            if any(s <= 0 or not np.isfinite(s) for s in self.thermal_spin):
                raise ValueError(
                    "thermal_spin entries must be positive finite spin lengths"
                )
        if self.thermal_temperatures is not None:
            if any(t < 0 or not np.isfinite(t) for t in self.thermal_temperatures):
                raise ValueError(
                    "thermal_temperatures entries must be finite and >= 0 K"
                )
        self.thermal_qmeshes = _validate_qmeshes(
            self.thermal_qmeshes, self.thermal_dimensionality
        )
        if not np.isfinite(self.thermal_mesh_tolerance) or (
            self.thermal_mesh_tolerance <= 0
        ):
            raise ValueError("thermal_mesh_tolerance must be a positive number (K)")
        if self.thermal_max_iterations < 10:
            raise ValueError("thermal_max_iterations must be >= 10")
        if not (0.0 < self.thermal_mixing <= 1.0):
            raise ValueError("thermal_mixing must lie in (0, 1]")
        if self.thermal_tolerance <= 0:
            raise ValueError("thermal_tolerance must be positive")

    @classmethod
    def from_toml(cls, filename: str) -> "ThermalMagnonParameters":
        """Load thermal parameters from a TOML file."""
        with open(filename, "rb") as f:
            data = tomli.load(f)
        return cls(**data)

    def to_toml(self, filename: str) -> None:
        """Save thermal parameters to a TOML file."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        with open(filename, "wb") as f:
            tomli_w.dump(data, f)


def _parse_mesh(token: str) -> List[int]:
    parts = token.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"q-mesh {token!r} must look like NxNxN (e.g. 8x8x8)"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"q-mesh {token!r} must contain integers"
        ) from exc
    return values


def add_thermal_args(parser: argparse.ArgumentParser) -> None:
    """Add thermal-magnon arguments to a CLI parser."""
    group = parser.add_argument_group("thermal magnon options")
    group.add_argument(
        "--thermal-method",
        choices=list(THERMAL_METHODS),
        default=None,
        help="finite-temperature method: RPA/Callen/HP decouplings or the "
        "bandless Weiss mean-field (MFA) baseline (default: rpa)",
    )
    group.add_argument(
        "--thermal-spin-regime",
        choices=list(SPIN_REGIMES),
        default=None,
        help="spin treatment: quantum (Callen closure) or classical "
        "(S(S+1) -> S^2 prescription)",
    )
    group.add_argument(
        "--thermal-spin",
        default=None,
        help="comma-separated per-site spin lengths S_i overriding the "
        "moment-derived defaults |mu_i| / (2 mu_B)",
    )
    group.add_argument(
        "--thermal-order-mode",
        choices=list(ORDER_MODES),
        default=None,
        help="ordered reference: uniform ferromagnet or collinear "
        "bipartite antiferromagnet (default: ferromagnetic)",
    )
    group.add_argument(
        "--thermal-dimensionality",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="physical periodicity of the magnetic model (1D/2D/3D)",
    )
    group.add_argument(
        "--thermal-temperatures",
        default=None,
        help="comma-separated temperatures in K at which bands are "
        "evaluated after thermal self-consistency",
    )
    group.add_argument(
        "--thermal-qmeshes",
        default=None,
        help="comma-separated increasing Gamma-centered q meshes, each "
        "NxNxN (e.g. 6x6x6,8x8x8) used to converge the transition",
    )
    group.add_argument(
        "--thermal-mesh-tolerance",
        type=float,
        default=None,
        help="transition convergence tolerance between successive meshes (K)",
    )
    group.add_argument(
        "--thermal-strict",
        action="store_true",
        default=None,
        help="raise an error when the q-mesh sequence fails its tolerance "
        "instead of returning a flagged estimate",
    )
    group.add_argument(
        "--thermal-max-iterations",
        type=int,
        default=None,
        help="maximum nonlinear self-consistency iterations",
    )
    group.add_argument(
        "--thermal-mixing",
        type=float,
        default=None,
        help="adaptive mixing parameter in (0, 1] for the magnetization "
        "fixed-point iteration",
    )
    group.add_argument(
        "--thermal-tolerance",
        type=float,
        default=None,
        help="magnetization fixed-point tolerance",
    )


def _parse_float_list(token: str) -> Optional[List[float]]:
    values = [float(t) for t in token.split(",") if t.strip()]
    return values or None


def thermal_parameters_from_args(args) -> ThermalMagnonParameters:
    """Build ThermalMagnonParameters from parsed CLI arguments."""
    kwargs = {}
    if getattr(args, "thermal_method", None) is not None:
        kwargs["thermal_method"] = args.thermal_method
    if getattr(args, "thermal_spin_regime", None) is not None:
        kwargs["thermal_spin_regime"] = args.thermal_spin_regime
    if getattr(args, "thermal_order_mode", None) is not None:
        kwargs["thermal_order_mode"] = args.thermal_order_mode
    if getattr(args, "thermal_dimensionality", None) is not None:
        kwargs["thermal_dimensionality"] = args.thermal_dimensionality
    if getattr(args, "thermal_spin", None):
        kwargs["thermal_spin"] = _parse_float_list(args.thermal_spin)
    if getattr(args, "thermal_temperatures", None):
        kwargs["thermal_temperatures"] = _parse_float_list(args.thermal_temperatures)
    if getattr(args, "thermal_qmeshes", None):
        kwargs["thermal_qmeshes"] = [
            _parse_mesh(t) for t in args.thermal_qmeshes.split(",") if t.strip()
        ]
    if getattr(args, "thermal_mesh_tolerance", None) is not None:
        kwargs["thermal_mesh_tolerance"] = args.thermal_mesh_tolerance
    if getattr(args, "thermal_strict", None) is not None:
        kwargs["thermal_strict"] = args.thermal_strict
    if getattr(args, "thermal_max_iterations", None) is not None:
        kwargs["thermal_max_iterations"] = args.thermal_max_iterations
    if getattr(args, "thermal_mixing", None) is not None:
        kwargs["thermal_mixing"] = args.thermal_mixing
    if getattr(args, "thermal_tolerance", None) is not None:
        kwargs["thermal_tolerance"] = args.thermal_tolerance
    return ThermalMagnonParameters(**kwargs)
