"""Build and use a bcc Fe GPAW PAW-projector Green NetCDF file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import bulk

from TB2J.interfaces.gpaw_projector import (
    save_gpaw_projector_netcdf,
    write_projector_exchange_out,
)
from TB2J.projector_green import ProjectorGreen, ProjectorGreenData


def build_gpaw_bcc_fe_projector_green_data(
    a=2.86,
    kpts=(9, 9, 9),
    pw_cutoff=400,
    width=0.05,
    nbands=16,
):
    """Run GPAW bcc Fe and return ProjectorGreenData."""
    from gpaw import GPAW, PW, FermiDirac

    atoms = bulk("Fe", "bcc", a=a)
    atoms.set_initial_magnetic_moments([2.2])
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
        Path("gpaw_bcc_fe_projector_green.nc"),
        atoms=atoms,
        metadata={
            "source": "GPAW bcc Fe PAW workflow",
            "xc": "PBE",
            "pw_cutoff_eV": pw_cutoff,
            "kmesh": list(kpts),
            "nbands": nbands,
            "symmetry": "off",
        },
    )


def run_gpaw_bcc_fe_projector_green_workflow(filename=None):
    """Run GPAW, save/load NetCDF when requested, and compute smoke outputs."""
    data = build_gpaw_bcc_fe_projector_green_data()
    if filename is not None:
        data.save_netcdf(filename)
        data = ProjectorGreenData.load_netcdf(filename)
    green = ProjectorGreen(data)
    energy = 0.1 + 0.02j
    GR = green.get_GR([(0, 0, 0)], energy=energy, ispin=0)
    trace = None
    if data.nspin == 2:
        from TB2J.projector_green import projector_exchange_trace

        trace = projector_exchange_trace(green, [(0, 0, 0)], energy=energy)
    return data, GR, trace


def main():
    outfile = Path("gpaw_bcc_fe_projector_green.nc")
    data, GR, trace = run_gpaw_bcc_fe_projector_green_workflow(outfile)
    exchange_out, exchange_Jdict = write_projector_exchange_out(data)
    print(f"Wrote {outfile}")
    print(f"Wrote {exchange_out}")
    print(f"k-points: {data.nkpt}, bands: {data.nband}, projectors: {data.nproj}")
    print(f"GR shape: {GR.shape}")
    if trace is not None:
        print(f"trace keys: {list(trace['trace'])[:3]}")
    first_shell = [
        v
        for (R, _, _), v in exchange_Jdict.items()
        if np.isclose(np.linalg.norm(np.asarray(R) @ data.cell), 2.476832985)
    ]
    if first_shell:
        print(f"first-shell J average: {np.mean(first_shell) * 1000:.6f} meV")


if __name__ == "__main__":
    main()
