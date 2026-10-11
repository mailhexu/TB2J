"""Optional unfold-epc consumer: finite-q pencil and both endpoint terms.

unfold_epc is imported only by consumer calls, never by TB2J import itself.
See docs/sympy/q_phased_exchange_contraction.py for every phase and gauge step.
"""

import numpy as np


def load_unfold_epc(path):
    try:
        from unfold_epc.store import TwoIndexStore
    except ModuleNotFoundError as exc:
        if exc.name != "unfold_epc":
            raise
        raise ImportError(
            "install unfold-epc to load its two-index EPC stores"
        ) from exc
    return TwoIndexStore.open(path)


def mixed_green_response(
    gk, gkq, dh, ds, energy, reference_fermi=0.0, *, drop_overlap=False
):
    """Raw physical dH/dS -> dG in a fixed Fermi-shifted model gauge."""
    # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#fixed_gauge_pencil_vertex
    vertex = dh if drop_overlap else dh - (energy + reference_fermi) * ds
    return gk @ vertex @ gkq


def exchange_product_derivative(pi, gij, pj, gji, dpi, dgij, dpj, dgji):
    """All four ordered product-rule terms; dpj already has its cell phase."""
    # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#four_term_exchange_product_rule
    return np.trace(
        dpi @ gij @ pj @ gji
        + pi @ dgij @ pj @ gji
        + pi @ gij @ dpj @ gji
        + pi @ gij @ pj @ dgji
    )


def _onsite(store, q, polarization):
    """Exact zero orbital-cell-separation derivative, not a finite k quadrature."""
    onsite = np.zeros((store.nspin, store.norb, store.norb), complex)
    for (atom, axis), block in store.blocks.items():
        weight = polarization[atom, axis]
        if weight == 0:
            continue
        equal = np.all(block.r1 == block.r2, axis=1)
        row = store.primitive_rows(block)[equal]
        col = store.primitive_columns(block)[equal]
        phase = np.exp(2j * np.pi * (block.r1[equal] @ q)) * weight
        for spin in range(store.nspin):
            np.add.at(onsite[spin], (row, col), phase * block.dH[equal, spin])
    return onsite


def gen_exchange_spin_lattice_unfold_epc(
    tbmodel,
    atoms,
    store_file,
    q,
    polarization,
    *,
    basis=None,
    magnetic_elements=None,
    kmesh=(5, 5, 1),
    emin=-85.0,
    emax=0.0,
    nz=320,
    nproc=1,
    Rcut=None,
    smearing=0.025851999786,
    drop_overlap=False,
    kpts=None,
    kweights=None,
):
    """Return complex O(q) [eV/Ang] keyed by (R, primitive atom i, atom j).

    tbmodel must use the stored fixed reference Fermi gauge (efermi=0),
    primitive cell-periodic Bloch gauge (convention 2), orbital order,
    collinear interleaved spins, and the same
    undisplaced Hamiltonian as the acceptance reference. All bands are kept.
    The returned response retains complex linearity via conjugate q partners.
    """
    from unfold_epc.transform import transform

    from TB2J.spin_lattice_lcao import ExchangePert2Sl, _FullSpectrumGreen

    store = load_unfold_epc(store_file)
    if basis is None:
        basis = getattr(tbmodel, "orbs", None)
    if basis is None:
        raise ValueError("basis labels are required for a model without orbs")
    if magnetic_elements is None:
        magnetic_elements = sorted(set(atoms.get_chemical_symbols()))
    q = np.asarray(q, dtype=float)
    polarization = np.asarray(polarization, dtype=complex)
    if (
        q.shape != (3,)
        or not np.isfinite(q).all()
        or polarization.shape != (store.natom, 3)
        or not np.isfinite(polarization).all()
    ):
        raise ValueError("q must be finite length 3 and polarization finite (natom,3)")
    if (
        store.nspin != 2
        or tbmodel.nbasis != 2 * store.norb
        or len(atoms) != store.natom
    ):
        raise ValueError(
            "consumer requires matching collinear primitive model and store"
        )
    if not getattr(tbmodel, "collinear_interleaved", False):
        raise ValueError("consumer requires a collinear interleaved-spin model")
    if not np.allclose(
        q @ np.asarray(store.supercell_matrix).T,
        np.rint(q @ np.asarray(store.supercell_matrix).T),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("q is not commensurate with the stored supercell")

    class QExchange(ExchangePert2Sl):
        @staticmethod
        def _green_class(**parameters):
            if kpts is not None:
                parameters.pop("kmesh", None)
                parameters["kpts"] = kpts
                parameters["kweights"] = kweights
            return _FullSpectrumGreen(**parameters)

        def set_q(self, wavevector, eigenvector):
            self.q = wavevector
            self.disp = store  # activates inherited endpoint terms, overridden below
            self.J_only = False
            self.density_method = "eigenvector"
            self.response = "q"
            self.basis_is_separated = False
            self._init_pert_dicts()
            self._q_green = _FullSpectrumGreen(
                tbmodel=tbmodel,
                kpts=self.G.kpts + wavevector,
                kweights=self.G.kweights,
                efermi=0.0,
                nproc=nproc,
                initial_emin=emin,
                smearing_width=smearing,
                use_cache=False,
            )
            transformed = [
                transform(store, k, wavevector, polarization=eigenvector)
                for k in self.G.kpts
            ]
            dh = np.asarray([pair[0] for pair in transformed]).transpose(1, 0, 2, 3)
            ds = np.asarray([pair[1] for pair in transformed])
            # Dropping overlap means dropping the WHOLE physical overlap channel,
            # including the -Ef_ref*dS gauge step, not only its z-dependent part.
            self._q_dh = dh if drop_overlap else dh - store.reference_fermi * ds[None]
            self._q_ds = ds
            onsite = _onsite(store, wavevector, eigenvector)
            self._q_onsite = (onsite[0] - onsite[1]) / 2
            self._dPdict = {}
            self._q_Rpts = np.asarray(self.short_Rlist).reshape(-1, 3)
            self._q_phase = (
                np.exp(-2j * np.pi * (self._q_Rpts @ self.G.kpts.T))
                * self.G.kweights[None]
            )
            # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#oriented_forward_bond_phase
            self._q_forward_phase = (
                self._q_phase * np.exp(-2j * np.pi * (self._q_Rpts @ self.q))[:, None]
            )
            # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#oriented_reverse_bond_phase
            self._q_reverse_phase = (
                np.exp(2j * np.pi * (self._q_Rpts @ self.G.kpts.T))
                * self.G.kweights[None]
            )

        def get_dP_iatom(self, iatom, R=(0, 0, 0)):
            key = (iatom, tuple(R))
            if key not in self._dPdict:
                orbital = np.asarray(self.iorb(iatom))[::2] // 2
                reference = self.get_H_atom(iatom)
                splitting_trace = np.trace(
                    (reference[::2, ::2] - reference[1::2, 1::2]) / 2
                ).real
                direction = (
                    np.sign(splitting_trace) if abs(splitting_trace) > 1e-8 else 1.0
                )
                # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#onsite_endpoint_phase
                self._dPdict[key] = (
                    direction
                    * self._q_onsite[np.ix_(orbital, orbital)]
                    * np.exp(-2j * np.pi * np.dot(self.q, R))
                )
            return self._dPdict[key]

        def get_AijR_rhoR(self, energy):
            left = self.G.get_spin_Gk_all(energy)
            right = self._q_green.get_spin_Gk_all(energy)
            # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#mixed_momentum_inverse_derivative
            dg = [
                left[s]
                @ (
                    self._q_dh[s]
                    if drop_overlap
                    else self._q_dh[s] - energy * self._q_ds
                )
                @ right[s]
                for s in (0, 1)
            ]
            points = self._q_Rpts
            phase = self._q_phase
            forward_phase = self._q_forward_phase
            reverse_phase = self._q_reverse_phase
            n = store.norb
            gr = np.zeros((len(points), 2 * n, 2 * n), complex)
            forward = np.zeros_like(gr)
            reverse = np.zeros_like(gr)
            for spin in (0, 1):
                sl = slice(spin, None, 2)
                gr[:, sl, sl] = (
                    phase @ left[spin].reshape(len(self.G.kpts), -1)
                ).reshape(len(points), n, n)
                forward[:, sl, sl] = (
                    forward_phase @ dg[spin].reshape(len(self.G.kpts), -1)
                ).reshape(len(points), n, n)
                reverse[:, sl, sl] = (
                    reverse_phase @ dg[spin].reshape(len(self.G.kpts), -1)
                ).reshape(len(points), n, n)
            a, da, aorb, daorb = self.get_all_A(
                dict(zip(self.short_Rlist, gr)),
                dict(zip(self.short_Rlist, forward)),
                dict(zip(self.short_Rlist, reverse)),
            )
            return a, da, None, aorb, daorb

    parameters = dict(
        tbmodels=tbmodel,
        atoms=atoms,
        basis=basis,
        magnetic_elements=magnetic_elements,
        efermi=0.0,
        kmesh=kmesh,
        emin=emin,
        emax=emax,
        nz=nz,
        nproc=nproc,
        integration_method="semicircle",
        Rcut=Rcut,
        smearing=smearing,
        use_cache=False,
    )
    integrals = []
    for wavevector, eigenvector in ((q, polarization), (-q, polarization.conj())):
        exchanger = QExchange(**parameters)
        exchanger.set_q(wavevector, eigenvector)
        exchanger.calculate_all()
        response = {
            key: value[0, 0] - value[1, 1] - value[2, 2] - value[3, 3]
            for key, value in exchanger.dA_ijR.items()
            if key[0] != (0, 0, 0) or key[1] != key[2]
        }
        integrals.append(response)
        exchanger.finalize()
    if set(integrals[0]) != set(integrals[1]):
        raise ValueError("q partner bond sets differ")
    # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#complex_linear_contour_reduction
    return {
        key: (integrals[0][key] - integrals[1][key].conjugate()) / (2j)
        for key in integrals[0]
    }


def load_unfold_epc_reference(reference_fdf, store_file):
    """Unfold the exact shared reference; no separate primitive SCF is needed.

    Equilibrium translational copies are averaged, not derivative endpoints.
    The physical reference is shifted by the same stored Ef_ref as the vertex.
    """
    from pathlib import Path

    import sisl
    from ase import Atoms
    from netCDF4 import Dataset
    from unfold_epc.collect import read_snapshot
    from unfold_epc.provenance import file_hash

    from TB2J.interfaces.siesta_spin_lattice import _CollinearSpinorModel

    store = load_unfold_epc(store_file)
    path = Path(reference_fdf).resolve()
    label = sisl.get_sile(str(path)).get("SystemLabel", default=path.stem)
    nc_path = path.with_name(str(label) + ".nc")
    expected = (
        store.metadata.get("input_hashes", {})
        .get("scf_outputs", {})
        .get("ref", {})
        .get("netcdf")
    )
    if expected is None or file_hash(nc_path) != expected:
        raise ValueError("reference netCDF does not match store provenance")
    snapshot, geometry = read_snapshot(nc_path)
    if geometry.no != store.no_sc or snapshot.shifted_h.shape[1] != 2:
        raise ValueError("reference orbital/spin dimensions disagree with store")
    row, col, offset = snapshot.keys[:, 0], snapshot.keys[:, 1], snapshot.keys[:, 2:]
    relative = (
        store.orbital_cells[col]
        + offset @ store.supercell_matrix
        - store.orbital_cells[row]
    )
    translations, indices = np.unique(relative, axis=0, return_inverse=True)
    n = store.norb
    h = np.zeros((len(translations), 2 * n, 2 * n), complex)
    s = np.zeros_like(h)
    multiplicity = store.no_sc // n
    primitive_row = store.primitive_orbital[row]
    primitive_col = store.primitive_orbital[col]
    shifted = (
        snapshot.shifted_h
        + (snapshot.fermi - store.reference_fermi) * snapshot.overlap[:, None]
    )
    for spin in (0, 1):
        target = (indices, 2 * primitive_row + spin, 2 * primitive_col + spin)
        # MathDoc: nonorth-spin-lattice/q-phased-exchange-contraction#primitive_reference_copy_average
        np.add.at(h, target, shifted[:, spin] / multiplicity)
        np.add.at(s, target, snapshot.overlap / multiplicity)
    atoms = Atoms(
        symbols=store.metadata["symbols"],
        positions=store.metadata["primitive_positions"],
        cell=store.cell,
        pbc=True,
    )
    counts = store.metadata["orbital_counts"]
    names, orbital_dictionary = [], {}
    symbol_count = {}
    starts = np.concatenate(([0], np.cumsum(counts)))
    for atom, symbol in enumerate(atoms.get_chemical_symbols()):
        symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
        atom_name = f"{symbol}{symbol_count[symbol]}"
        source_row = int(np.flatnonzero(store.primitive_orbital == starts[atom])[0])
        source_atom = int(geometry.o2a(source_row))
        orbitals = geometry.atoms[source_atom].orbitals
        if len(orbitals) != counts[atom]:
            raise ValueError("primitive orbital counts disagree with reference basis")
        orbital_dictionary[atom_name] = list(
            range(2 * int(starts[atom]), 2 * int(starts[atom + 1]))
        )
        for orbital in orbitals:
            for spin in ("up", "down"):
                names.append(f"{atom_name}|{orbital.name()}|{spin}")
    with Dataset(nc_path) as nc:
        electrons = (
            float(np.asarray(nc.variables["Qtot"][:]).reshape(-1)[0]) / multiplicity
        )
    model = _CollinearSpinorModel(
        HR=h,
        SR=s,
        Rlist=translations,
        nbasis=2 * n,
        orbs=names,
        atoms=atoms,
        nspin=1,
        nel=electrons,
        orb_dict=orbital_dictionary,
        orth=False,
    )
    return model
