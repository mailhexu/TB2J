"""Build and use a cubic SrMnO3 GPAW PAW-projector Green NetCDF file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from TB2J.interfaces.gpaw_projector import (
    save_gpaw_projector_netcdf,
    write_projector_exchange_out,
)
from TB2J.projector_green import ProjectorGreen, ProjectorGreenData


def build_cubic_srmno3(a=3.8):
    """Return the five-atom cubic perovskite SrMnO3 cell."""
    return Atoms(
        symbols=["Sr", "Mn", "O", "O", "O"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        cell=np.eye(3) * a,
        pbc=True,
    )


def build_gpaw_cubic_srmno3_projector_green_data(
    a=3.8,
    kpts=(5, 5, 5),
    pw_cutoff=400,
    width=0.05,
    nbands=64,
):
    """Run GPAW cubic SrMnO3 and return ProjectorGreenData."""
    from gpaw import GPAW, PW, FermiDirac

    atoms = build_cubic_srmno3(a=a)
    atoms.set_initial_magnetic_moments([0.0, 3.0, 0.0, 0.0, 0.0])
    calc = GPAW(
        mode=PW(pw_cutoff),
        xc="PBE",
        kpts=kpts,
        spinpol=True,
        occupations=FermiDirac(width),
        nbands=nbands,
        symmetry="off",
        convergence={"energy": 1e-6, "density": 1e-5, "eigenstates": 1e-6},
        txt=None,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return save_gpaw_projector_netcdf(
        calc,
        Path("gpaw_cubic_srmno3_projector_green.nc"),
        atoms=atoms,
        metadata={
            "source": "GPAW cubic SrMnO3 PAW workflow",
            "xc": "PBE",
            "pw_cutoff_eV": pw_cutoff,
            "kmesh": list(kpts),
            "nbands": nbands,
            "symmetry": "off",
            "lattice_constant_A": a,
            "magnetic_site": 1,
        },
    )


def run_gpaw_cubic_srmno3_projector_green_workflow(filename=None):
    """Run GPAW, optionally roundtrip NetCDF, and compute smoke outputs."""
    data = build_gpaw_cubic_srmno3_projector_green_data()
    if filename is not None:
        data.save_netcdf(filename)
        data = ProjectorGreenData.load_netcdf(filename)
    green = ProjectorGreen(data)
    energy = 0.1 + 0.02j
    GR = green.get_GR([(0, 0, 0)], energy=energy, ispin=0)
    return data, GR


def write_mn_projector_exchange_out(data, path="TB2J_results_srmno3", Rpts=None):
    """Write a TB2J-style exchange.out for the Mn projector trace result."""
    return write_projector_exchange_out(
        data,
        path=path,
        Rpts=Rpts,
        index_magnetic_atoms=[int(data.metadata.get("magnetic_site", 1))],
    )


def main():
    outfile = Path("gpaw_cubic_srmno3_projector_green.nc")
    data, GR = run_gpaw_cubic_srmno3_projector_green_workflow(outfile)
    exchange_out, exchange_Jdict = write_mn_projector_exchange_out(data)
    print(f"Wrote {outfile}")
    print(f"Wrote {exchange_out}")
    print(f"k-points: {data.nkpt}, bands: {data.nband}, projectors: {data.nproj}")
    print(
        f"moments: total={data.metadata['magnetic_moment_total']:.6f} mu_B, "
        f"local={data.metadata['magnetic_moments']}"
    )
    print(
        f"Mn |Delta H_ij|_F: "
        f"{np.linalg.norm(data.get_hij_spin_difference(site=1)):.6f} eV"
    )
    print(f"GR shape: {GR.shape}")
    first_shell = [
        v
        for (R, _, _), v in exchange_Jdict.items()
        if np.isclose(np.linalg.norm(np.asarray(R) @ data.cell), 3.8)
    ]
    if first_shell:
        print(f"first-shell Mn-Mn J average: {np.mean(first_shell) * 1000:.6f} meV")


if __name__ == "__main__":
    main()
