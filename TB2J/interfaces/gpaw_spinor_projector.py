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

from pathlib import Path

import numpy as np
from ase.units import kB

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


def _site_magnetization_sign(operator_block):
    """Sign of the z-projector trace (majority-spin direction)."""
    block = np.asarray(operator_block)
    ztrace = float(np.real(np.trace(block[:, :, 0, 0] - block[:, :, 1, 1])))
    return 1.0 if ztrace >= 0.0 else -1.0


def compute_spinor_projector_exchange(
    data,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    sites=None,
    overlap_mode=None,
    overlap_rcond=None,
):
    """Contour-integrated spinor exchange tensor per (R, i, j).

    Integrates the sympy-pinned J^{ab}(E) object over the fermion contour;
    the imaginary-part prescription removes the same-spin-channel piece,
    so the collinear reduction J_iso = (J_xx+J_yy)/2 holds. Returns
    {(R, i, j): {"Jiso", "dmi", "jani", "tensor"}} with the ExchangeNCL
    (TB2J.Jtensor) decomposition.
    """
    from TB2J.mycfr import CFR
    from TB2J.projector_green import ProjectorGreen, spinor_projector_exchange_trace

    if data.nspinor != 2:
        raise ValueError("spinor exchange requires nspinor=2 data")
    if Rpts is None:
        from TB2J.interfaces.gpaw_projector import _R_grid

        Rpts = _R_grid(nmax=1)
    Rpts = np.asarray(Rpts, dtype=int)
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    sites = [int(site) for site in sites]
    green = ProjectorGreen(data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond)
    local_operators = green.get_local_operators_spinor(sites=sites)
    signs = {site: _site_magnetization_sign(op) for site, op in local_operators.items()}

    contour = CFR(nz=nz, T=smearing_eV / kB)
    values = {
        (tuple(int(x) for x in R), i, j): [] for R in Rpts for i in sites for j in sites
    }
    for energy in contour.path:
        trace = spinor_projector_exchange_trace(
            green, Rpts, energy=energy, local_operators=local_operators, sites=sites
        )
        for key in values:
            values[key].append(trace["tensor_complex"][key])

    from TB2J.Jtensor import decompose_J_tensor

    result = {}
    for key, vals in values.items():
        R, i, j = key
        integrated = np.asarray(
            [
                [
                    contour.integrate_values(np.asarray([v[a, b] for v in vals]))
                    for b in range(3)
                ]
                for a in range(3)
            ]
        )
        Jtens = np.imag(integrated) * signs[i] * signs[j]
        Jiso, D, Jani = decompose_J_tensor(Jtens)
        result[key] = {"Jiso": Jiso, "dmi": D, "jani": Jani, "tensor": Jtens}
    return result


def write_spinor_projector_exchange_out(
    data,
    path="TB2J_results",
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    description=None,
    charges=None,
    spinat=None,
    Rcut=None,
    overlap_mode=None,
    overlap_rcond=None,
):
    """Write standard TB2J outputs (noncollinear SpinIO) from spinor data."""
    from ase import Atoms

    from TB2J.interfaces.gpaw_projector import _magnetic_sites, _R_grid_for_cutoff
    from TB2J.io_exchange.io_exchange import SpinIO

    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
        pbc=True,
    )
    sites = _magnetic_sites(
        data,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
    )
    if Rpts is None:
        Rpts = _R_grid_for_cutoff(data, sites, Rcut if Rcut is not None else 10.0)
    exchange = compute_spinor_projector_exchange(
        data,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )
    if charges is None:
        charges = np.zeros(len(atoms), dtype=float)
    if spinat is None:
        spinat = np.zeros((len(atoms), 3), dtype=float)
    index_spin = [-1] * len(atoms)
    site_to_spin = {}
    for ispin, site in enumerate(sites):
        index_spin[site] = ispin
        site_to_spin[site] = ispin
    exchange_Jdict = {}
    dmi_ddict = {}
    Jani_dict = {}
    distance_dict = {}
    for (R, i, j), entry in exchange.items():
        vector = np.asarray(R) @ data.cell + atoms.positions[j] - atoms.positions[i]
        distance = float(np.linalg.norm(vector))
        if Rcut is not None and distance >= float(Rcut):
            continue
        key = (tuple(int(x) for x in R), site_to_spin[i], site_to_spin[j])
        distance_dict[key] = (vector, distance)
        exchange_Jdict[key] = float(entry["Jiso"])
        dmi_ddict[key] = np.asarray(entry["dmi"], dtype=float)
        Jani_dict[key] = np.asarray(entry["jani"], dtype=float)
    if description is None:
        description = (
            "Spinor projector Green workflow using GPAW noncollinear+SOC "
            "projections and the dH_asii Pauli-decomposed 2x2 site operator. "
            "J_iso, DMI, and anisotropic exchange from the sympy-pinned "
            "spinor exchange tensor (docs/sympy/spinor_projector_green.md).\n"
        )
    output = SpinIO(
        atoms=atoms,
        charges=charges,
        spinat=spinat,
        index_spin=index_spin,
        colinear=False,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_Jdict,
        dmi_ddict=dmi_ddict,
        Jani_dict=Jani_dict,
        description=description,
    )
    output.write_all(path=path)
    return Path(path) / "exchange.out", exchange_Jdict


def gen_exchange_gpaw_spinor_netcdf(
    filename,
    output_path="TB2J_results",
    Rcut=10.0,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
):
    """Python interface for spinor projector-NetCDF exchange calculation."""
    from TB2J.projector_green import ProjectorGreenData

    data = ProjectorGreenData.load_netcdf(filename)
    return write_spinor_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        Rcut=Rcut,
    )
