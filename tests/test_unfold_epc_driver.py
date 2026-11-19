import numpy as np
from ase import Atoms
from scipy.linalg import eigh
from unfold_epc.store import SparseDerivative, TwoIndexStore

from TB2J.interfaces.unfold_epc import gen_exchange_spin_lattice_unfold_epc
from TB2J.spin_lattice_lcao import ExchangePert2Sl


class Model:
    name = "LCAOHamiltonian"
    R2kfactor = 2j * np.pi
    is_orthogonal = False
    collinear_interleaved = True

    def __init__(self, h=None, s=None):
        self.h, self.s = h, s
        self.norb = 2 if h is None else 6
        self.nbasis = 2 * self.norb
        self.nel = 1.0 if h is None else 3.0
        self.atoms = Atoms(
            "Fe" if h is None else "Fe3",
            positions=[[0, 0, 0]] if h is None else [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            cell=np.diag([1 if h is None else 3, 10, 10]),
            pbc=True,
        )

    def HSE_k(self, k, convention=2):
        if self.h is not None:
            h, s = self.h, self.s
        else:
            h = np.zeros((4, 4), complex)
            s = np.eye(4, dtype=complex)
            for spin in (0, 1):
                a, b = spin, spin + 2
                h[a, a] = -0.7 + 0.6 * (1 - 2 * spin) - 0.8 * np.cos(2 * np.pi * k[0])
                h[b, b] = 12 + 0.2 * (1 - 2 * spin)
                h[a, b] = h[b, a] = 1.5
                s[a, b] = s[b, a] = 0.06
        ev = np.empty(self.nbasis)
        vectors = np.zeros_like(h)
        for spin in (0, 1):
            sl = slice(spin, None, 2)
            ev[sl], vectors[sl, sl] = eigh(h[sl, sl], s[sl, sl])
        return h, s, ev, vectors


def test_full_q_driver_matches_independent_supercell_exchange_fd(tmp_path):
    model = Model()
    dh = np.array([[[0.2, 0.4], [0.4, 0.0]], [[-0.2, 0.4], [0.4, 0.0]]])
    ds = np.array([[0.0, 0.02], [0.02, 0.0]])
    r, c = np.indices((2, 2))
    primitive_orbital = np.tile([0, 1], 3)
    cells = np.repeat(np.column_stack([np.arange(3), np.zeros((3, 2), int)]), 2, axis=0)
    block = SparseDerivative(
        r.ravel(),
        c.ravel(),
        np.zeros((4, 3), int),
        np.zeros((4, 3), int),
        dh.reshape(2, -1).T,
        ds.ravel(),
    )
    store = TwoIndexStore(
        np.diag([1, 10, 10]),
        np.diag([3, 1, 1]),
        primitive_orbital,
        cells,
        2,
        1,
        2,
        0.0,
        0.005,
        {(0, 0): block},
        {"input_hashes": {}},
    )
    path = tmp_path / "q-driver-toy.nc"
    store.write(path)
    basis = ["Fe1|s|up", "Fe1|s|down", "Fe1|p|up", "Fe1|p|down"]
    q = np.array([1 / 3, 0, 0])
    result = gen_exchange_spin_lattice_unfold_epc(
        model,
        model.atoms,
        path,
        q,
        [[1, 0, 0]],
        basis=basis,
        magnetic_elements=["Fe"],
        kmesh=(3, 1, 1),
        Rcut=1.1,
        nz=96,
        emin=-4.0,
        smearing=0.1,
    )
    # Independent 3-cell Fourier construction and real displaced J-only calls.
    h0, s0 = np.zeros((12, 12), complex), np.zeros((12, 12), complex)
    for k in (0.0, 1 / 3, 2 / 3):
        h, s, _, _ = model.HSE_k([k, 0, 0])
        u = np.kron(
            np.exp(2j * np.pi * k * np.arange(3))[:, None] / np.sqrt(3), np.eye(4)
        )
        h0 += u @ h @ u.conj().T
        s0 += u @ s @ u.conj().T
    full_dh = np.zeros((4, 4), complex)
    full_ds = np.zeros((4, 4), complex)
    for spin in (0, 1):
        full_dh[spin::2, spin::2] = dh[spin]
        full_ds[spin::2, spin::2] = ds
    basis_sc = [
        f"Fe{atom+1}|{orb}|{spin}"
        for atom in range(3)
        for orb in ("s", "p")
        for spin in ("up", "down")
    ]
    step = 1e-5
    parts = []
    for component in ("cos", "sin"):
        ph = (
            np.cos(2 * np.pi * np.arange(3) / 3)
            if component == "cos"
            else np.sin(2 * np.pi * np.arange(3) / 3)
        )
        shift_h = np.kron(np.diag(ph), full_dh)
        shift_s = np.kron(np.diag(ph), full_ds)
        values = []
        for sign in (1, -1):
            displaced = Model(h0 + sign * step * shift_h, s0 + sign * step * shift_s)
            ex = ExchangePert2Sl(
                tbmodels=displaced,
                atoms=displaced.atoms,
                basis=basis_sc,
                magnetic_elements=["Fe"],
                efermi=0.0,
                kmesh=(1, 1, 1),
                Rcut=1.1,
                nz=96,
                emin=-4.0,
                integration_method="semicircle",
                smearing=0.1,
            )
            ex.set_displacement(None, J_only=True, response="periodic")
            ex.calculate_all()
            values.append(ex.exchange_Jdict[((0, 0, 0), 0, 1)])
            ex.finalize()
        parts.append((values[0] - values[1]) / (2 * step))
    actual = result[((1, 0, 0), 0, 0)]
    expected = parts[0] - 1j * parts[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
