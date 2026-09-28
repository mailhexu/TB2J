"""Three-leg GPAW split-SOC exchange and magnetic-anisotropy workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import kB

from TB2J.interfaces.gpaw_projector import _R_grid_for_cutoff
from TB2J.interfaces.gpaw_spinor_split_soc import (
    LEG_AXES,
    apply_frame_rotation,
    collect_soc_leg,
    soc_leg_to_projector_green_data,
)
from TB2J.io_exchange.io_exchange import SpinIO
from TB2J.io_merge import merge
from TB2J.mycfr import CFR
from TB2J.split_soc_kernel import (
    band_window_convergence_report,
    compute_ks_split_soc_exchange,
    json_safe_provenance,
)


def _magnetic_sites(calc, indices):
    moments = np.asarray(calc.get_magnetic_moments(), dtype=float)
    if moments.shape != (len(calc.get_atoms()),):
        raise ValueError("collinear no-SOC reference must have one moment per atom")
    if indices is None:
        sites = [i for i, moment in enumerate(moments) if abs(moment) > 1e-3]
    else:
        sites = [int(i) for i in indices]
    if (
        not sites
        or len(set(sites)) != len(sites)
        or any(i < 0 or i >= len(moments) or abs(moments[i]) <= 1e-3 for i in sites)
    ):
        raise ValueError(
            "magnetic_sites must be distinct atoms with nonzero frozen moments"
        )
    return sites, moments


def _write_leg(
    path, data, exchange, sites, moments, axis, rotation, rcut, metadata=None
):
    """Write the rotated physical tensor and per-leg spin axes for io_merge."""
    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
        pbc=True,
    )
    site_to_spin = {site: n for n, site in enumerate(sites)}
    index_spin = [site_to_spin.get(i, -1) for i in range(len(atoms))]
    spinat = np.zeros((len(atoms), 3))
    for site in sites:
        spinat[site] = moments[site] * axis
    distances = {}
    jiso = {}
    dmi = {}
    jani = {}
    for (r, i, j), entry in exchange.items():
        vector = np.asarray(r) @ data.cell + data.positions[j] - data.positions[i]
        distance = float(np.linalg.norm(vector))
        if not (1e-6 < distance < rcut):
            continue
        key = (tuple(int(v) for v in r), site_to_spin[i], site_to_spin[j])
        distances[key] = (vector, distance)
        jiso[key] = float(entry["Jiso"])
        dmi[key] = rotation @ np.asarray(entry["dmi"], dtype=float)
        jani[key] = apply_frame_rotation(entry["jani"], rotation)
    if not jiso:
        raise ValueError("no magnetic pairs lie within Rcut; increase the cutoff")
    spinio = SpinIO(
        atoms=atoms,
        charges=np.zeros(len(atoms)),
        spinat=spinat,
        index_spin=index_spin,
        colinear=False,
        distance_dict=distances,
        exchange_Jdict=jiso,
        dmi_ddict=dmi,
        Jani_dict=jani,
        description=(
            "GPAW split-SOC second-variational leg, strength-0 collinear frozen "
            "density; all-atom W_SO, magnetic-only PAW XC/+U vertices; "
            "tensor rotated psi->lattice before three-leg merge."
            + (
                "\nsplit_soc_provenance: " + json.dumps(metadata, sort_keys=True)
                if metadata is not None
                else ""
            )
        ),
    )
    if metadata is not None:
        spinio.split_soc_provenance = metadata
    spinio.write_all(path=str(path))


def _contour_second_order_shift(leg, nz, smearing_eV):
    """Band-space second-order trace, -Im int Tr[(G0 W)^2]/(2 pi).

    The strength-0 resolvent is diagonal in the paired collinear KS basis.
    The CFR contour and occupation width match the exchange calculation;
    this is a perturbative comparison, not a replacement for GPAW band MAE.
    """
    contour = CFR(nz=nz, T=smearing_eV / kB)
    weighted_w2 = np.abs(leg.w_soc) ** 2 * leg.weights[:, None, None]
    values = []
    for energy in contour.path:
        g = 1.0 / (energy + leg.efermi - leg.eigenvalues_strength0)
        values.append(np.einsum("knm,kn,km->", weighted_w2, g, g, optimize=True))
    return -float(np.imag(contour.integrate_values(np.asarray(values)))) / (2.0 * np.pi)


def gen_exchange_gpaw_split_soc(
    calc_or_gpw,
    output_path="TB2J_results_gpaw_split_soc",
    Rcut=10.0,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_sites=None,
    scale=1.0,
    mae_contour_tolerance_eV=5e-6,
    vertex_component="delta_total",
):
    """Run x/y/z SOC legs from one frozen no-SOC GPAW calculation and merge.

    ``calc_or_gpw`` is an old-API collinear calculator or its legacy checkpoint.
    The per-leg Fermi level/occupations from GPAW's second-variational solver
    define the band-energy MAE; exchange uses the same strength-0 band window
    with the shared KS-band kernel.  No SOC or noncollinear SCF is performed.
    """
    if isinstance(calc_or_gpw, (str, Path)):
        from gpaw import GPAW

        checkpoint = Path(calc_or_gpw).resolve()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        strength0_id = {"checkpoint": str(checkpoint), "sha256": digest.hexdigest()}
        calc = GPAW(str(checkpoint), legacy_gpaw=True)
    else:
        calc = calc_or_gpw
        strength0_id = {"source": "in-memory old-API GPAW calculator"}
    if (
        not np.isfinite(scale)
        or not np.isfinite(mae_contour_tolerance_eV)
        or mae_contour_tolerance_eV <= 0
    ):
        raise ValueError(
            "SOC scale and MAE tolerance must be finite; tolerance positive"
        )
    if Rcut <= 0:
        raise ValueError("Rcut must be positive")
    sites, moments = _magnetic_sites(calc, magnetic_sites)
    output = Path(output_path)
    leg_paths = {}
    mae = {}
    metadata = {}
    rpts = None if Rpts is None else np.asarray(Rpts, dtype=int)
    for direction, (theta, phi) in LEG_AXES.items():
        leg = collect_soc_leg(
            calc, theta=theta, phi=phi, scale=scale, delta_kind=vertex_component
        )
        data = soc_leg_to_projector_green_data(leg)
        if rpts is None:
            rpts = _R_grid_for_cutoff(data, sites, Rcut)
        result = compute_ks_split_soc_exchange(
            data,
            leg.w_soc,
            lam=1.0,
            Rpts=rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=sites,
            metadata={
                "strength0_reference": {
                    **leg.metadata["strength0_reference"],
                    **strength0_id,
                },
                "soc_operator_source": leg.metadata["soc_operator_source"],
                "frame": data.metadata["frame"],
                "merge_mode": "three_legs",
            },
        )
        if data.nband < 4 or data.nband % 2:
            raise ValueError(
                "window convergence requires at least two paired spinor band windows"
            )
        study = band_window_convergence_report(
            data,
            leg.w_soc,
            [data.nband - 2, data.nband],
            pair=(sites[0], sites[1] if len(sites) > 1 else sites[0]),
            Rpts=rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=sites,
        )
        result["metadata"]["band_window"]["convergence_study"] = json_safe_provenance(
            study
        )
        leg_metadata = json_safe_provenance(result["metadata"])
        leg_path = output / direction
        _write_leg(
            leg_path,
            data,
            result["exchange"],
            sites,
            moments,
            leg.axis,
            leg.rotation,
            Rcut,
            metadata=leg_metadata,
        )
        leg_paths[direction] = leg_path
        mae[direction] = {
            "band_energy_eV": leg.band_energy_soc,
            "fermi_eV": leg.efermi_soc,
            "axis": leg.axis.tolist(),
            "contour_second_order_shift_eV": _contour_second_order_shift(
                leg, nz, smearing_eV
            ),
        }
        metadata[direction] = leg_metadata
    reference = mae["z"]["band_energy_eV"]
    contour_reference = mae["z"]["contour_second_order_shift_eV"]
    for row in mae.values():
        row["relative_to_z_eV"] = row["band_energy_eV"] - reference
        row["contour_relative_to_z_eV"] = (
            row["contour_second_order_shift_eV"] - contour_reference
        )
        row["contour_residual_eV"] = (
            row["relative_to_z_eV"] - row["contour_relative_to_z_eV"]
        )
        row["contour_tolerance_eV"] = mae_contour_tolerance_eV
        row["contour_within_tolerance"] = (
            abs(row["contour_residual_eV"]) <= mae_contour_tolerance_eV
        )
    merged_path = output / "merged"
    merge(
        *(str(leg_paths[name]) for name in LEG_AXES),
        save=True,
        write_path=str(merged_path),
        merged_provenance={"merge_mode": "three_legs", "legs": metadata},
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_soc_report.json").write_text(
        json.dumps(
            {
                "mae": mae,
                "legs": {name: str(path) for name, path in leg_paths.items()},
                "merged": str(merged_path),
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "leg_paths": leg_paths,
        "merged_path": merged_path,
        "mae": mae,
        "metadata": metadata,
    }
