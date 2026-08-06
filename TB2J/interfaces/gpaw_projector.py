"""GPAW PAW-projector NetCDF export and projector exchange interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Ha, kB

from TB2J.io_exchange import SpinIO
from TB2J.mycfr import CFR
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    projector_exchange_trace,
)


def _pack_density(matrix):
    matrix = np.asarray(matrix)
    packed = []
    for row in range(matrix.shape[0]):
        packed.append(matrix[row, row])
        for col in range(row + 1, matrix.shape[1]):
            packed.append(matrix[row, col] + matrix[col, row])
    return np.asarray(packed)


def _setup_projector_metadata(setups):
    projector_l = []
    projector_m = []
    projector_radial = []
    projector_atom = []
    projector_site = []
    site_nproj = []
    site_projector_indices = []
    paw_N0_p = []
    overlap_blocks = []
    offset = 0
    for atom, setup in enumerate(setups):
        atom_indices = []
        radial_count = {}
        for l in setup.l_j:
            radial = radial_count.get(l, 0)
            radial_count[l] = radial + 1
            for m in range(-l, l + 1):
                projector_l.append(l)
                projector_m.append(m)
                projector_radial.append(radial)
                projector_atom.append(atom)
                projector_site.append(atom)
                atom_indices.append(offset)
                offset += 1
        site_nproj.append(len(atom_indices))
        site_projector_indices.append(atom_indices)
        paw_N0_p.append(np.asarray(setup.N0_p, dtype=float).tolist())
        overlap_blocks.append(np.asarray(setup.dO_ii, dtype=complex))

    nmax = max(site_nproj)
    nproj = offset
    overlap_metric = np.zeros((nproj, nproj), dtype=complex)
    padded = -np.ones((len(site_nproj), nmax), dtype=int)
    for site, indices in enumerate(site_projector_indices):
        padded[site, : len(indices)] = indices
        overlap_metric[np.ix_(indices, indices)] = overlap_blocks[site]

    return {
        "projector_l": np.array(projector_l, dtype=int),
        "projector_m": np.array(projector_m, dtype=int),
        "projector_radial": np.array(projector_radial, dtype=int),
        "projector_atom": np.array(projector_atom, dtype=int),
        "projector_site": np.array(projector_site, dtype=int),
        "site_nproj": np.array(site_nproj, dtype=int),
        "site_projector_indices": padded,
        "paw_N0_p": paw_N0_p,
        "overlap_metric": overlap_metric,
    }


def _collect_kpoint_data(calc):
    wfs = calc.wfs
    kd = wfs.kd
    nspin = wfs.nspins
    bz_kpoints = np.asarray(getattr(kd, "bzk_kc", calc.get_bz_k_points()), dtype=float)
    nkpt = len(bz_kpoints)
    ibz2bz = None
    if getattr(kd, "nibzkpts", nkpt) != nkpt:
        ibz2bz = _make_ibz2bz_maps(calc)
    first = _source_kpoint(wfs, spin=0, ibz_index=0)
    nband = len(first.eps_n)
    nproj = sum(P_ni.shape[1] for P_ni in _projection_blocks(first.P_ani))

    eigenvalues = np.empty((nspin, nkpt, nband), dtype=float)
    occupations = np.empty((nspin, nkpt, nband), dtype=float)
    coefficients = np.empty((nspin, nkpt, nband, nproj), dtype=complex)
    output_weight = 1.0 / nkpt
    for spin in range(nspin):
        for bz_index in range(nkpt):
            ibz_index = int(kd.bz2ibz_k[bz_index])
            kpt = _source_kpoint(wfs, spin=spin, ibz_index=ibz_index)
            eigenvalues[spin, bz_index] = kpt.eps_n * Ha
            kpt_weight = float(getattr(kpt, "weight", output_weight))
            occupations[spin, bz_index] = kpt.f_n * output_weight / kpt_weight
            coefficients[spin, bz_index] = np.concatenate(
                _full_bz_projection_blocks(wfs, bz_index, spin, ibz2bz, kpt), axis=1
            )

    weights = np.full(nkpt, 1.0 / nkpt)
    return bz_kpoints, weights, eigenvalues, occupations, coefficients


def _collect_fermi_levels(calc):
    if hasattr(calc, "get_fermi_levels"):
        try:
            levels = getattr(calc, "get_fermi_levels")()
        except (ValueError, TypeError, AttributeError):
            levels = None
        if levels is not None:
            levels = np.asarray(levels, dtype=float)
            if levels.ndim == 0:
                levels = levels.reshape(1)
            if levels.size:
                return levels * Ha

    if hasattr(calc, "wfs") and hasattr(calc.wfs, "fermi_levels"):
        levels = getattr(calc.wfs, "fermi_levels")
        if levels is not None:
            levels = np.asarray(levels, dtype=float)
            if levels.ndim == 0:
                levels = levels.reshape(1)
            if levels.size:
                return levels * Ha

    if hasattr(calc, "get_fermi_level"):
        return np.array([float(getattr(calc, "get_fermi_level")())], dtype=float)

    raise AttributeError("Unable to read Fermi level from GPAW calculator")


def _source_kpoint(wfs, spin, ibz_index):
    if hasattr(wfs, "kpt_qs"):
        return wfs.kpt_qs[ibz_index][spin]
    for kpt in wfs.kpt_u:
        if kpt.s == spin and kpt.q == ibz_index:
            return kpt
    raise KeyError(f"missing GPAW k-point for spin={spin}, ibz_index={ibz_index}")


def _projection_blocks(projections):
    items = list(projections.items())
    return [
        np.asarray(block) for _, block in sorted(items, key=lambda item: int(item[0]))
    ]


def _full_bz_projection_blocks(wfs, bz_index, spin, ibz2bz, source_kpt):
    if ibz2bz is None:
        return _projection_blocks(source_kpt.P_ani)
    projections = _map_projections_in_bz(wfs, bz_index, spin, ibz2bz)
    return _projection_blocks(projections)


def _make_ibz2bz_maps(calc):
    from gpaw.ibz2bz import IBZ2BZMaps

    return IBZ2BZMaps.from_calculator(calc)


def _map_projections_in_bz(wfs, bz_index, spin, ibz2bz):
    from gpaw.wannier90 import get_projections_in_bz

    return get_projections_in_bz(wfs, bz_index, spin, ibz2bz, bcomm=None)


def _collect_hij(calc, site_nproj):
    from gpaw.utilities import unpack_hermitian

    nspin = calc.wfs.nspins
    nsite = len(site_nproj)
    nmax = int(np.max(site_nproj))
    hij = np.zeros((nspin, nsite, nmax, nmax), dtype=complex)
    for atom, dH_sp in calc.hamiltonian.dH_asp.items():
        atom = int(atom)
        nproj = site_nproj[atom]
        for spin in range(nspin):
            hij[spin, atom, :nproj, :nproj] = unpack_hermitian(dH_sp[spin]) * Ha
    return hij


def _collect_delta_xc_paw_xc(calc, site_nproj):
    """Atom-centered XC exchange field in the PAW partial-wave basis.

    Per atom, evaluate the spin splitting of the PAW XC energy derivative matrix
    dE_xc/dD_sp via GPAW ``hamiltonian.xc.calculate_paw_correction``.  For
    LSDA/GGA this is the explicit partial-wave matrix element of the
    exchange-correlation spin field V_xc^up - V_xc^down restricted to the XC
    contribution (paper eq:pseudo-partial-delta).  Hartree, ionic, and scalar
    external PAW terms are spin independent, so for plain collinear DFT this
    coincides with the full dH_asp spin splitting (verified on bcc Fe); it
    cleanly isolates the XC exchange field for +U/SOC/general cases.
    """
    from gpaw.utilities import unpack_hermitian

    nspin = calc.wfs.nspins
    nsite = len(site_nproj)
    nmax = int(np.max(site_nproj))
    delta_xc = np.zeros((nsite, nmax, nmax), dtype=complex)
    if nspin != 2:
        return delta_xc
    xc = calc.hamiltonian.xc
    for atom, D_sp in calc.density.D_asp.items():
        atom = int(atom)
        setup = calc.wfs.setups[atom]
        D_sp = np.asarray(D_sp)
        dEdD_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(setup, D_sp, dEdD_sp)
        delta = (unpack_hermitian(dEdD_sp[0]) - unpack_hermitian(dEdD_sp[1])) * Ha
        nproj = site_nproj[atom]
        delta_xc[atom, :nproj, :nproj] = delta[:nproj, :nproj]
    return delta_xc


def gpaw_calc_to_projector_green_data(calc, atoms=None, metadata=None):
    """Convert a converged GPAW PAW calculation to ProjectorGreenData."""
    if atoms is None:
        atoms = calc.get_atoms()
    kpoints, weights, eigenvalues, occupations, coefficients = _collect_kpoint_data(
        calc
    )
    nspin = calc.wfs.nspins
    fermi_levels = _collect_fermi_levels(calc)
    fermi_spin = None
    if fermi_levels.shape[0] == nspin and nspin > 1:
        fermi_spin = fermi_levels
    elif fermi_levels.shape[0] != 1:
        raise ValueError(
            "unexpected number of GPAW fermi levels: "
            f"{int(fermi_levels.shape[0])}; expected 1 or nspin"
        )
    efermi = float(np.mean(fermi_levels))
    projector_metadata = _setup_projector_metadata(calc.wfs.setups)
    hij = _collect_hij(calc, projector_metadata["site_nproj"])
    delta_xc = _collect_delta_xc_paw_xc(calc, projector_metadata["site_nproj"])
    operator_components = {"delta_xc": delta_xc} if calc.wfs.nspins == 2 else None
    operator_component_metadata = (
        {
            "delta_xc": {
                "units": "eV",
                "definition": (
                    "explicit V_xc^up - V_xc^down PAW partial-wave matrix "
                    "(GPAW xc.calculate_paw_correction spin splitting)"
                ),
                "source": "GPAW hamiltonian.xc.calculate_paw_correction",
                "operator_basis": "paw_partial_wave_channel",
            }
        }
        if calc.wfs.nspins == 2
        else None
    )
    metadata = {} if metadata is None else dict(metadata)
    metadata.update(
        {
            "source": "GPAW PAW projector workflow",
            "projector_basis_type": "paw",
            "source_code": "gpaw",
            "gpaw_mode": str(calc.wfs.mode),
            "gpaw_nbzkpts": int(calc.wfs.kd.nbzkpts),
            "gpaw_nibzkpts": int(calc.wfs.kd.nibzkpts),
            "symmetry_unfolded_to_full_bz": bool(
                calc.wfs.kd.nbzkpts != calc.wfs.kd.nibzkpts
            ),
            "paw_N0_p": projector_metadata["paw_N0_p"],
            "overlap_metric": (
                "GPAW PAW onsite dO_ii correction, block diagonal by site"
            ),
            "population_method": "GPAW PAW occupation projector density with N0_p",
            "magnetic_moment_total": calc.get_magnetic_moment(),
            "magnetic_moments": calc.get_magnetic_moments().tolist(),
        }
    )
    return ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=efermi,
        efermi_spin=fermi_spin,
        projector_site=projector_metadata["projector_site"],
        projector_atom=projector_metadata["projector_atom"],
        cell=atoms.cell.array,
        positions=atoms.positions,
        atomic_numbers=atoms.numbers,
        occupations=occupations,
        projector_l=projector_metadata["projector_l"],
        projector_m=projector_metadata["projector_m"],
        projector_radial=projector_metadata["projector_radial"],
        overlap_metric=projector_metadata["overlap_metric"],
        site_nproj=projector_metadata["site_nproj"],
        site_projector_indices=projector_metadata["site_projector_indices"],
        hij=hij,
        hij_definition="paw_dh_asp_projector_hamiltonian",
        hij_units="eV",
        hij_source="GPAW dH_asp",
        hij_projection="native PAW projector Hamiltonian matrix",
        coefficient_source="gpaw.P_ani",
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_partial_wave_channel",
        overlap_metric_definition=(
            "GPAW PAW onsite dO_ii correction, block diagonal by site"
        ),
        population_metric="GPAW PAW N0_p packed density contraction",
        operator_basis="native_paw_projector_hamiltonian",
        operator_components=operator_components,
        operator_component_metadata=operator_component_metadata,
        metadata=metadata,
    )


def save_gpaw_projector_netcdf(calc, filename, atoms=None, metadata=None):
    """Save a converged GPAW calculation as TB2J projector Green NetCDF."""
    data = gpaw_calc_to_projector_green_data(calc, atoms=atoms, metadata=metadata)
    data.save_netcdf(filename)
    return data


def _R_grid(nmax=1):
    return np.array(
        [
            (i, j, k)
            for i in range(-nmax, nmax + 1)
            for j in range(-nmax, nmax + 1)
            for k in range(-nmax, nmax + 1)
        ],
        dtype=int,
    )


def _R_grid_for_cutoff(data, sites, Rcut=None, Rmax=None):
    if Rmax is None:
        if Rcut is None:
            Rmax = 1
        else:
            cell_lengths = np.linalg.norm(np.asarray(data.cell, dtype=float), axis=1)
            positions = np.asarray(data.positions, dtype=float)
            max_pair = 0.0
            for i in sites:
                for j in sites:
                    max_pair = max(
                        max_pair, float(np.linalg.norm(positions[j] - positions[i]))
                    )
            Rmax = int(np.ceil((float(Rcut) + max_pair) / np.min(cell_lengths)))
    Rpts = _R_grid(nmax=Rmax)
    if Rcut is None:
        return Rpts

    selected = set()
    for R in Rpts:
        for sign in (1, -1):
            Rs = sign * R
            for i in sites:
                for j in sites:
                    vector = Rs @ data.cell + data.positions[j] - data.positions[i]
                    if float(np.linalg.norm(vector)) < float(Rcut):
                        selected.add(tuple(int(x) for x in R))
                        selected.add(tuple(int(-x) for x in R))
                        break
                else:
                    continue
                break
    return np.asarray(sorted(selected), dtype=int)


def _magnetic_sites(
    data, magnetic_elements=None, index_magnetic_atoms=None, threshold=1e-3
):
    if index_magnetic_atoms is not None:
        return [int(site) for site in index_magnetic_atoms]
    if magnetic_elements is not None:
        from ase.data import chemical_symbols

        elements = set(magnetic_elements)
        return [
            i
            for i, number in enumerate(data.atomic_numbers)
            if chemical_symbols[int(number)] in elements
        ]
    moments = data.metadata.get("magnetic_moments")
    if moments is not None:
        return [i for i, moment in enumerate(moments) if abs(moment) > threshold]
    return list(range(len(data.site_nproj)))


def _paw_N0_from_data_or_setups(data):
    if "paw_N0_p" in data.metadata:
        return [np.asarray(values, dtype=float) for values in data.metadata["paw_N0_p"]]

    from gpaw import GPAW, PW, FermiDirac

    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
        pbc=True,
    )
    calc = GPAW(
        mode=PW(200),
        xc=data.metadata.get("xc", "PBE"),
        kpts=(1, 1, 1),
        spinpol=True,
        occupations=FermiDirac(0.05),
        nbands=max(data.nband, len(atoms)),
        symmetry="off",
        txt=None,
    )
    calc.initialize(atoms)
    return [np.asarray(setup.N0_p, dtype=float) for setup in calc.wfs.setups]


def compute_paw_projected_charges_moments(data):
    """Compute PAW local populations from stored occupations and projectors."""
    if data.occupations is None:
        raise ValueError("PAW projected populations require occupations in NetCDF")
    paw_N0_p = _paw_N0_from_data_or_setups(data)
    nsite = len(data.site_nproj)
    density_by_spin = np.zeros((data.nspin, nsite), dtype=float)
    for spin in range(data.nspin):
        rho = np.einsum(
            "k,kb,kbp,kbq->pq",
            data.weights,
            data.occupations[spin],
            data.coefficients[spin].conj(),
            data.coefficients[spin],
            optimize="optimal",
        )
        for site, nproj in enumerate(data.site_nproj):
            indices = data.site_projector_indices[site, :nproj]
            block = rho[np.ix_(indices, indices)]
            density_by_spin[spin, site] = np.real(
                np.dot(_pack_density(block.real), paw_N0_p[site])
            )
    charges = np.sum(density_by_spin, axis=0)
    spinat = np.zeros((nsite, 3), dtype=float)
    if data.nspin >= 2:
        spinat[:, 2] = density_by_spin[0] - density_by_spin[1]
    return charges, spinat, density_by_spin


def compute_projected_charges_moments(data):
    """Compute local populations for projector exchange output metadata."""
    if "paw_N0_p" in data.metadata:
        return compute_paw_projected_charges_moments(data)
    if data.occupations is None:
        raise ValueError("projected populations require occupations")

    nsite = len(data.site_nproj)
    density_by_spin = np.zeros((data.nspin, nsite), dtype=float)
    for spin in range(data.nspin):
        rho = np.einsum(
            "k,kb,kbp,kbq->pq",
            data.weights,
            data.occupations[spin],
            data.coefficients[spin].conj(),
            data.coefficients[spin],
            optimize="optimal",
        )
        for site, nproj in enumerate(data.site_nproj):
            indices = data.site_projector_indices[site, :nproj]
            block = rho[np.ix_(indices, indices)]
            if data.population_metric_matrix is not None:
                metric = data.population_metric_matrix[np.ix_(indices, indices)]
                density_by_spin[spin, site] = float(np.real(np.trace(block @ metric)))
            elif data.overlap_metric is not None:
                metric = data.overlap_metric[np.ix_(indices, indices)]
                density_by_spin[spin, site] = float(np.real(np.trace(block @ metric)))
            else:
                density_by_spin[spin, site] = float(np.real(np.trace(block)))
    charges = np.sum(density_by_spin, axis=0)
    spinat = np.zeros((nsite, 3), dtype=float)
    if data.nspin >= 2:
        spinat[:, 2] = density_by_spin[0] - density_by_spin[1]
    return charges, spinat, density_by_spin


def component_local_operators(data, component_name, sites, source_label="projector"):
    """Return exchange-ready site-local blocks from an operator component."""
    if component_name is None:
        component_name = "delta_total"
    if not data.has_operator_component(component_name):
        raise ValueError(
            f"{source_label} operator component is unavailable: {component_name}"
        )
    metadata = (data.operator_component_metadata or {}).get(component_name, {})
    completeness = metadata.get("completeness")
    exchange_ready = str(metadata.get("exchange_ready", "")).lower()
    if exchange_ready in {"false", "0", "no"}:
        raise ValueError(
            f"{source_label} operator component is not exchange-ready: "
            f"{component_name} exchange_ready={metadata.get('exchange_ready')!r}"
        )
    if completeness not in {
        None,
        "complete",
        "zero_by_symmetry",
    } and exchange_ready not in {"true", "1", "yes"}:
        raise ValueError(
            f"{source_label} operator component is not exchange-ready: "
            f"{component_name} completeness={completeness!r}"
        )
    return {
        int(site): data.get_operator_component(component_name, site=site)
        for site in sites
    }


def compute_projector_exchange_jdict(
    data,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    sites=None,
    local_operators=None,
    operator_component=None,
    overlap_mode=None,
    overlap_rcond=None,
):
    """Compute TB2J-style isotropic exchange dictionary from projector trace."""
    if Rpts is None:
        Rpts = _R_grid(nmax=1)
    Rpts = np.asarray(Rpts, dtype=int)
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    sites = [int(site) for site in sites]
    if operator_component is not None and local_operators is None:
        local_operators = component_local_operators(
            data, operator_component, sites, "projector exchange"
        )
    green = ProjectorGreen(data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond)
    if local_operators is None:
        local_operators = {int(site): green.get_local_operator(site) for site in sites}
    site_to_spin = {site: ispin for ispin, site in enumerate(sites)}
    contour = CFR(nz=nz, T=smearing_eV / kB)
    values = {
        (tuple(int(x) for x in R), i, j): [] for R in Rpts for i in sites for j in sites
    }
    for energy in contour.path:
        if local_operators is not None:
            trace = projector_exchange_trace(
                green,
                Rpts,
                energy=energy,
                local_operators=local_operators,
                sites=sites,
            )
        else:
            trace = projector_exchange_trace(green, Rpts, energy=energy, sites=sites)
        for key in values:
            values[key].append(trace["trace"][key])

    exchange_Jdict = {}
    sign_cache = {}
    for (R, i, j), vals in values.items():
        integrated = contour.integrate_values(np.asarray(vals))
        if i not in sign_cache:
            tr_i = (
                float(np.real(np.trace(local_operators[i]))) if local_operators else 0.0
            )
            sign_cache[i] = 1.0 if tr_i >= 0 else -1.0
        if j not in sign_cache:
            tr_j = (
                float(np.real(np.trace(local_operators[j]))) if local_operators else 0.0
            )
            sign_cache[j] = 1.0 if tr_j >= 0 else -1.0
        sign = sign_cache[i] * sign_cache[j]
        exchange_Jdict[(R, site_to_spin[i], site_to_spin[j])] = (
            float(np.imag(integrated)) / sign
        )
    return exchange_Jdict


def write_projector_exchange_out(
    data,
    path="TB2J_results",
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    description=None,
    population_mode="projector",
    charges=None,
    spinat=None,
    Rcut=None,
    local_operators=None,
    operator_component=None,
    overlap_mode=None,
    overlap_rcond=None,
):
    """Write TB2J exchange.out from projector Green data."""
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
    exchange_Jdict = compute_projector_exchange_jdict(
        data,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        local_operators=local_operators,
        operator_component=operator_component,
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )
    if charges is not None or spinat is not None:
        if charges is None or spinat is None:
            raise ValueError("charges and spinat must be provided together")
        charges = np.asarray(charges, dtype=float)
        spinat = np.asarray(spinat, dtype=float)
        if charges.shape != (len(atoms),):
            raise ValueError("charges must have one value per atom")
        if spinat.shape != (len(atoms), 3):
            raise ValueError("spinat must have shape (natom, 3)")
    elif population_mode == "none":
        charges = np.zeros(len(atoms), dtype=float)
        spinat = np.zeros((len(atoms), 3), dtype=float)
    elif population_mode == "projector":
        charges, spinat, _ = compute_projected_charges_moments(data)
    else:
        raise ValueError(f"unsupported population_mode: {population_mode}")
    index_spin = [-1] * len(atoms)
    for ispin, site in enumerate(sites):
        index_spin[site] = ispin
    spin_to_site = {ispin: site for ispin, site in enumerate(sites)}
    distance_dict = {}
    filtered_exchange_Jdict = {}
    for R, i, j in exchange_Jdict:
        site_i = spin_to_site[i]
        site_j = spin_to_site[j]
        vector = (
            np.asarray(R) @ data.cell
            + atoms.positions[site_j]
            - atoms.positions[site_i]
        )
        distance = float(np.linalg.norm(vector))
        if Rcut is None or distance < float(Rcut):
            distance_dict[(R, i, j)] = (vector, distance)
            filtered_exchange_Jdict[(R, i, j)] = exchange_Jdict[(R, i, j)]
    exchange_Jdict = filtered_exchange_Jdict

    if description is None:
        description = (
            "Projector Green workflow using GPAW PAW projections and native dH_asp "
            "as H_ij. Values are from the controlled projector exchange-like trace, "
            "not yet a production PAW MFT benchmark.\n"
        )
    output = SpinIO(
        atoms=atoms,
        charges=charges,
        spinat=spinat,
        index_spin=index_spin,
        colinear=True,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_Jdict,
        description=description,
    )
    output.write_txt(path=path)
    return Path(path) / "exchange.out", exchange_Jdict


def gen_exchange_projector_netcdf(
    filename,
    output_path="TB2J_results",
    Rmax=None,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    operator_component=None,
):
    """Python interface for projector-NetCDF exchange calculation."""
    data = ProjectorGreenData.load_netcdf(filename)
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    Rpts = _R_grid_for_cutoff(
        data, sites or list(range(len(data.site_nproj))), Rcut, Rmax
    )
    local_operators = None
    description = None
    if operator_component is not None:
        component_name = operator_component
        local_operators = component_local_operators(
            data,
            component_name,
            sites or list(range(len(data.site_nproj))),
            "GPAW NetCDF",
        )
        description = (
            "Projector Green workflow using GPAW PAW projections and operator "
            f"component {component_name} in basis {data.operator_basis}. Values are "
            "from the controlled projector exchange-like trace, not yet a "
            "production PAW MFT benchmark.\n"
        )
    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        Rcut=Rcut,
        local_operators=local_operators,
        description=description,
    )


def gen_exchange_gpaw(
    calc,
    atoms=None,
    output_path="TB2J_results",
    magnetic_elements=None,
    index_magnetic_atoms=None,
    operator_component=None,
    Rmax=None,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    save_netcdf=None,
    metadata=None,
):
    """High-level GPAW projector-Green exchange API.

    Take a *converged* GPAW calculator, build the projector-Green data (which
    collects the explicit pseudo-partial-wave ``delta_xc`` exchange field
    V_xc^up - V_xc^down via ``hamiltonian.xc.calculate_paw_correction``),
    optionally save the NetCDF to ``save_netcdf``, and evaluate the controlled
    projector exchange trace, writing the standard ``exchange.out``.

    Operator selection: by default ``ProjectorGreen.get_local_operator`` prefers
    the ``delta_xc`` component exported here, then ``delta_total``, then the
    ``hij`` spin splitting. Pass ``operator_component`` (e.g. ``"delta_xc"``,
    ``"delta_total"``) to force a specific component.

    Returns ``(exchange_out_path, exchange_Jdict)``.
    """
    data = gpaw_calc_to_projector_green_data(calc, atoms=atoms, metadata=metadata)
    if save_netcdf is not None:
        data.save_netcdf(save_netcdf)
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    Rpts = _R_grid_for_cutoff(
        data, sites or list(range(len(data.site_nproj))), Rcut, Rmax
    )
    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        Rcut=Rcut,
        operator_component=operator_component,
        description=(
            "Projector Green workflow using GPAW PAW projections and operator "
            f"component {operator_component or 'delta_xc (auto)'} in basis "
            f"{data.operator_basis}. Values are from the controlled projector "
            "exchange-like trace, not yet a production PAW MFT benchmark.\n"
        ),
    )
