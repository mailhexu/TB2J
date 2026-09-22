"""GPAW noncollinear+SOC spinor projector export (story 004).

Converts a converged GPAW noncollinear (``soc=True``/``magmoms=[...]``)
calculation into the TB2J spinor-native :class:`ProjectorGreenData`
contract (``nspinor=2``):

- coefficients ``P_ani[n, s, i]`` (spinor axis from GPAW's noncollinear
  projections, shape ``(nband, 2, nproj)``) -> ``(1, nkpt, nband, 2, nproj)``;
- site operator from ``dH_asii`` whose 4 components are the Pauli vector
  ``(v, x, y, z)`` (gpaw/new/potential.py:45, stored transposed), with the
  spin-dependent part packed as the 2x2 matrix
  ``Delta = x*sigma_x + y*sigma_y + z*sigma_z``.

Symmetry-unfolding (story 005) is not applied here: the k-point set is
used as stored (run with ``symmetry='off'`` or via the unfolding adapter).
"""

from __future__ import annotations

import numpy as np

from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreenData,
)

SIGMA = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)


def _require_new_api(calc):
    dft = getattr(calc, "dft", None)
    if dft is None:
        raise TypeError(
            "spinor export requires a new-API GPAW calculation "
            "(gpaw >= 22 with calc.dft); old-API calculators are not supported"
        )
    return dft


def _iter_kpt_wfs(wfs):
    """Yield (ibz_index, wfs) sorted by the IBZ k-point index."""
    entries = []
    for w in wfs._wfs_u:
        k = int(getattr(w, "k"))
        entries.append((k, w))
    entries.sort(key=lambda item: item[0])
    return entries


def gpaw_spinor_calc_to_projector_green_data(calc) -> ProjectorGreenData:
    """Convert a converged noncollinear+SOC GPAW calculation to spinor data."""
    dft = _require_new_api(calc)
    wfs = dft.ibzwfs
    if wfs.collinear or wfs.ncomponents != 4:
        raise ValueError(
            "spinor export requires a noncollinear calculation "
            f"(got collinear={wfs.collinear}, ncomponents={wfs.ncomponents})"
        )
    ibz = wfs.ibz
    kpt_kc = np.asarray(ibz.kpt_kc, dtype=float)
    weight_k = np.asarray(ibz.weight_k, dtype=float)
    nkpt = len(kpt_kc)

    # GPAW forces symmetry off for SC noncollinear PW runs
    # (gpaw/new/builder.py asserts identity-only symmetry without time
    # reversal), so the stored k-set must be the complete BZ grid. Guard
    # against partially symmetrized data from other modes: every weight
    # must be a whole multiple of 1/nkpt for a complete unsymmetrized grid.
    grid_weight = 1.0 / nkpt
    if not np.allclose(weight_k / grid_weight, np.round(weight_k / grid_weight)):
        raise ValueError(
            "spinor export requires a complete (unsymmetrized) BZ k-point "
            "set; GPAW SC noncollinear calculations are symmetry-forced "
            "off, so symmetrized k-weights indicate an unsupported mode"
        )
    nbands = wfs.nbands

    setups = dft.setups
    natoms = len(setups)
    site_nproj = np.array([setup.ni for setup in setups], dtype=int)
    nproj = int(site_nproj.sum())
    site_projector_indices = np.full((natoms, int(site_nproj.max())), -1, dtype=int)
    start = 0
    for atom, ni in enumerate(site_nproj):
        site_projector_indices[atom, :ni] = np.arange(start, start + ni)
        start += ni
    projector_site = np.repeat(np.arange(natoms), site_nproj)

    entries = _iter_kpt_wfs(wfs)
    if [k for k, _ in entries] != list(range(nkpt)):
        raise ValueError(
            "could not enumerate all IBZ k-point wavefunctions "
            f"(found {len(entries)} of {nkpt})"
        )
    coefficients = np.empty((1, nkpt, nbands, 2, nproj), dtype=complex)
    eigenvalues = np.empty((1, nkpt, nbands), dtype=float)
    occupations = np.empty((1, nkpt, nbands), dtype=float)
    for ik, w in entries:
        eig_n = np.asarray(w.myeig_n, dtype=float)
        occ_n = np.asarray(w.myocc_n, dtype=float)
        if eig_n.shape != (nbands,):
            raise ValueError(f"k-point {ik}: eigenvalue shape {eig_n.shape}")
        blocks = [
            np.asarray(w.P_ani[atom])  # (nband, 2, ni)
            for atom in range(natoms)
        ]
        for atom, block in enumerate(blocks):
            ni = int(site_nproj[atom])
            if block.shape != (nbands, 2, ni):
                raise ValueError(
                    f"k-point {ik}, atom {atom}: projections {block.shape} "
                    f"!= ({nbands}, 2, {ni})"
                )
        coefficients[0, ik] = np.concatenate(blocks, axis=2)
        eigenvalues[0, ik] = eig_n
        occupations[0, ik] = occ_n

    # Site operator: dH_asii components (v, x, y, z), stored transposed
    # (gpaw/new/potential.py unpacks with .T).
    dH_asii = dft.potential.dH_asii
    nproj_max = int(site_nproj.max())
    spinor_operator = np.zeros((natoms, nproj_max, nproj_max, 2, 2), dtype=complex)
    for atom in range(natoms):
        ni = int(site_nproj[atom])
        comps = [np.asarray(dH_asii[atom][i]).T for i in range(4)]
        block = np.einsum("aij,ast->ijst", np.asarray(comps[1:]), SIGMA)
        spinor_operator[atom, :ni, :ni] = block

    atoms = dft.atoms
    data = ProjectorGreenData(
        kpoints=kpt_kc,
        weights=weight_k,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=float(calc.get_fermi_level()),
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=np.asarray(atoms.cell.array, dtype=float),
        positions=atoms.get_positions(),
        atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        occupations=occupations,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        nspinor=2,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source="gpaw.noncollinear_P_ani",
        coefficient_projector="native_paw_projector",
        channel_interpretation="paw_projector_channel",
        operator_basis="gpaw.dH_asii pauli (v,x,y,z)",
        metadata={
            "nspinor": 2,
            "code": "gpaw",
            "spinor_export": "gpaw_spinor_projector",
            "units": {
                "cell": "Angstrom",
                "positions": "Angstrom",
                "eigenvalues": "eV",
                "efermi": "eV",
                "spinor_operator": "eV",
            },
        },
    )
    data.validate(exchange_ready=True)
    return data


def save_gpaw_spinor_projector_netcdf(calc, filename, metadata=None):
    """Save a converged noncollinear+SOC GPAW calculation as spinor NetCDF."""
    data = gpaw_spinor_calc_to_projector_green_data(calc)
    if metadata:
        data.metadata.update(metadata)
    data.save_netcdf(filename)
    return data
