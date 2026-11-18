"""Build a small synthetic projector Green dataset.

This example has no external DFT dependency.  It demonstrates the non-PAW path
where a converter projects a spin-dependent potential onto selected projectors
to produce the normalized TB2J ``hij`` operator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    build_site_projector_indices,
    pack_site_hij,
    project_potential_to_hij,
)


def build_synthetic_projector_green_data():
    """Return a tiny non-PAW projector Green dataset for examples/tests."""
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    weights = np.array([0.5, 0.5])
    eigenvalues = np.array(
        [
            [[-0.5, 1.0], [-0.25, 1.25]],
            [[-0.3, 1.2], [-0.05, 1.45]],
        ]
    )
    coefficients = np.zeros((2, 2, 2, 3), dtype=complex)
    coefficients[:, :, 0, 0] = 1.0
    coefficients[:, :, 1, 1] = 0.8
    coefficients[:, :, 1, 2] = 0.6j

    projector_site = np.array([0, 0, 0])
    site_nproj, site_projector_indices = build_site_projector_indices(projector_site)

    projectors_on_grid = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5j],
            [0.5, 0.5, 1.0],
        ],
        dtype=complex,
    )
    spin_potential = np.array(
        [
            [1.0, 1.2, 1.4],
            [0.6, 0.8, 1.0],
        ]
    )
    grid_weights = np.array([0.25, 0.25, 0.5])
    hij_global = project_potential_to_hij(
        projectors_on_grid, spin_potential, weights=grid_weights
    )
    hij = pack_site_hij(hij_global, site_projector_indices, site_nproj)

    return ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.0,
        projector_site=projector_site,
        projector_atom=np.array([0, 0, 0]),
        cell=np.eye(3),
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([26]),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        hij=hij,
        hij_definition="projected_spin_dependent_potential",
        hij_units="eV",
        hij_source="synthetic non-PAW projected potential",
        hij_projection="<p_i|V_up/down|p_j> on a three-point grid",
        coefficient_source="synthetic orthonormal projector coefficients",
        coefficient_projector="custom_discrete_grid_projector",
        channel_interpretation="custom_discrete_grid_projector",
        operator_basis="projected_spin_dependent_potential",
        metadata={
            "source": "synthetic non-PAW projector example",
            "projector_basis_type": "custom_discrete_grid",
        },
    )


def main():
    data = build_synthetic_projector_green_data()
    green = ProjectorGreen(data)
    energy = 0.1 + 0.02j
    GR = green.get_GR([(0, 0, 0)], energy=energy, ispin=0)
    print(f"Built synthetic projector Green data with GR shape {GR.shape}")

    try:
        import netCDF4  # noqa: F401
    except ImportError:
        print("netCDF4 is not installed; skipping NetCDF write")
        return

    outfile = Path("synthetic_projector_green.nc")
    data.save_netcdf(outfile)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
