"""Tests for TB2J.symmetrize_J: exact space-group symmetrization of exchange.

The symmetrizer projects the pair tensors (J^iso, DMI, J^ani) and the
single-ion anisotropy onto the symmetry orbits of the stored keys, using the
full spglib operations (rotations and translations), optionally of the
magnetic space group.
"""

import subprocess
import sys
from collections import defaultdict

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.symmetrize_J import (
    TB2JSymmetrizer,
    _bond_image,
    _build_site_maps,
    combine_tensor,
    crystal_symmetry_ops,
    decompose_tensor,
    symmetrize_exchange,
)


def _make_atoms(symbols, cell, scaled_positions):
    return Atoms(symbols, cell=cell, scaled_positions=scaled_positions)


def _distance_dict(atoms, index_spin, keys):
    """Minimal distance dict {(R, i, j): (vec, distance)} for write_txt."""
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    pos = atoms.get_positions()
    dd = {}
    for R, i, j in keys:
        ia, ja = index_spin[i], index_spin[j]
        vec = pos[ja] + np.dot(R, cell) - pos[ia]
        dd[(tuple(R), i, j)] = (vec, float(np.linalg.norm(vec)))
    return dd


def _make_spinio(
    atoms,
    spinat,
    exchange_Jdict,
    dmi_ddict=None,
    Jani_dict=None,
    sia_tensor=None,
    index_spin=None,
):
    natom = len(atoms)
    if index_spin is None:
        index_spin = list(range(natom))
    keys = list(exchange_Jdict)
    if dmi_ddict:
        keys += list(dmi_ddict)
    if Jani_dict:
        keys += list(Jani_dict)
    return SpinIO(
        atoms,
        None if spinat is None else np.asarray(spinat, dtype=float),
        [0] * natom,
        list(index_spin),
        colinear=True,
        distance_dict=_distance_dict(atoms, index_spin, keys),
        exchange_Jdict=dict(exchange_Jdict),
        dmi_ddict=None if dmi_ddict is None else dict(dmi_ddict),
        Jani_dict=None if Jani_dict is None else dict(Jani_dict),
        sia_tensor=None if sia_tensor is None else dict(sia_tensor),
    )


def _brute_force_orbit_means(atoms, values, symprec=1e-5):
    """Independent orbit computation: union-find over symmetry bond images.

    values: dict with atom-level keys (ia, ja, R) -> scalar.
    Returns (expected dict key -> orbit mean, partition as frozenset of frozensets).
    """
    ops = crystal_symmetry_ops(atoms, symprec=symprec)
    xfrac = np.asarray(atoms.get_scaled_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    smaps = _build_site_maps(xfrac, cell, ops, symprec)
    keys = sorted(values)
    keyset = set(keys)

    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for W, smap in zip(ops.rotations, smaps):
        for k in keys:
            img = _bond_image(W, smap, xfrac, *k)
            if img in keyset:
                union(k, img)

    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    expected = {}
    for members in groups.values():
        m = float(np.mean([values[k] for k in members]))
        for k in members:
            expected[k] = m
    partition = frozenset(frozenset(members) for members in groups.values())
    return expected, partition


# ---------------------------------------------------------------------------
# decomposition round trip


def test_decomposition_roundtrip_random():
    rng = np.random.default_rng(42)
    for _ in range(10):
        G = rng.normal(size=(3, 3))
        Jiso, D, Jani = decompose_tensor(G)
        assert np.allclose(combine_tensor(Jiso, D, Jani), G, atol=1e-13)
        # J^ani is symmetric and traceless, D comes from the skew part only
        assert np.allclose(Jani, Jani.T, atol=1e-14)
        assert abs(np.trace(Jani)) < 1e-13


def test_orbit_average_reproduces_gammabar_exactly():
    """In P1 the only operation is the identity; the symmetrized tensor must
    be exactly the reversal-symmetrized average of the two stored keys, and
    the decomposition must reproduce it (zero_tol=0: no snapping)."""
    cell = [[3.0, 0.0, 0.0], [1.3, 3.6, 0.2], [0.4, 0.5, 3.9]]
    # two atoms in generic positions: the structure (unlike the bare lattice)
    # has no inversion, spglib finds P1 with a single operation
    atoms = _make_atoms("Fe2", cell, [[0.11, 0.13, 0.17], [0.31, 0.29, 0.41]])
    rng = np.random.default_rng(7)
    M1 = rng.normal(size=(3, 3))
    Jani1 = 0.5 * (M1 + M1.T)
    G1 = combine_tensor(1.0, np.array([0.1, 0.2, 0.3]), Jani1)
    M2 = rng.normal(size=(3, 3))
    Jani2 = 0.5 * (M2 + M2.T)
    G2 = combine_tensor(1.4, np.array([-0.05, -0.15, -0.2]), Jani2)
    exc = _make_spinio(
        atoms,
        [[0, 0, 2.0], [0, 0, 2.0]],
        {((1, 0, 0), 0, 0): 1.0, ((-1, 0, 0), 0, 0): 1.4},
        dmi_ddict={
            ((1, 0, 0), 0, 0): np.array([0.1, 0.2, 0.3]),
            ((-1, 0, 0), 0, 0): np.array([-0.05, -0.15, -0.2]),
        },
        Jani_dict={((1, 0, 0), 0, 0): Jani1, ((-1, 0, 0), 0, 0): Jani2},
    )
    sym = TB2JSymmetrizer(exc, verbose=False, zero_tol=0.0)
    sym.symmetrize_J()
    # symmetry images of the stored keys are absent from the dataset, so the
    # two orbits are singletons: each key keeps its reversal-symmetrized
    # tensor, and the two are exact transposes of each other
    expected = {
        ((1, 0, 0), 0, 0): 0.5 * (G1 + G2.T),
        ((-1, 0, 0), 0, 0): 0.5 * (G2 + G1.T),
    }
    for key, Gbar in expected.items():
        Jiso, D, Jani = decompose_tensor(Gbar)
        assert sym.new_exc.exchange_Jdict[key] == pytest.approx(Jiso, abs=1e-14)
        assert np.allclose(sym.new_exc.dmi_ddict[key], D, atol=1e-14)
        assert np.allclose(sym.new_exc.Jani_dict[key], Jani, atol=1e-14)
        # exact reproduction of the average tensor by the three channels
        assert np.allclose(
            combine_tensor(
                sym.new_exc.exchange_Jdict[key],
                sym.new_exc.dmi_ddict[key],
                sym.new_exc.Jani_dict[key],
            ),
            Gbar,
            atol=1e-15,
        )
    # the two keys are exactly reversal partners
    assert np.allclose(
        sym.new_exc.dmi_ddict[((-1, 0, 0), 0, 0)],
        -sym.new_exc.dmi_ddict[((1, 0, 0), 0, 0)],
        atol=1e-15,
    )


# ---------------------------------------------------------------------------
# isotropic exchange: orbits and averaging


def test_equivalent_scalars_averaged_machine_precision_bcc():
    """bcc cell (Im-3m): noisy but symmetry-equivalent J's are averaged to the
    orbit mean; the orbit partition matches an independent brute-force
    application of the operations."""
    atoms = _make_atoms("Fe2", np.eye(3) * 2.8, [[0, 0, 0], [0.5, 0.5, 0.5]])
    rng = np.random.default_rng(1)
    # self bonds along the 6 cubic axes: one orbit under Im-3m
    Rself = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    # NN body-diagonal bonds and their reversals (all 8, so the stored set is
    # closed under the space group, as a real bond-cutoff dataset is)
    Rnn = [
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
        (-1, -1, -1),
    ]
    values = {}
    Jdict = {}
    for R in Rself:
        v = 2.0 + rng.normal(scale=0.05)
        values[(0, 0, tuple(R))] = v
        Jdict[(tuple(R), 0, 0)] = v
    for R in Rnn:
        v = 1.5 + rng.normal(scale=0.05)
        values[(0, 1, tuple(R))] = v
        values[(1, 0, tuple(-np.array(R)))] = v
        Jdict[(tuple(R), 0, 1)] = v
        Jdict[(tuple(-np.array(R)), 1, 0)] = v

    exc = _make_spinio(atoms, [[0, 0, 2.5], [0, 0, 2.5]], Jdict)
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()

    expected, partition = _brute_force_orbit_means(atoms, values)
    # engine partition, recovered from the (identical) assigned values
    by_value = defaultdict(set)
    for (R, i, j), val in sym.new_exc.exchange_Jdict.items():
        by_value[val].add((i, j, tuple(R)))
    engine_partition = frozenset(frozenset(s) for s in by_value.values())
    assert engine_partition == partition

    for key, val in expected.items():
        (ia, ja, R) = key
        got = sym.new_exc.exchange_Jdict[(R, ia, ja)]
        # machine precision: identical floats within an orbit
        assert got == pytest.approx(val, abs=1e-12)
    # all members of the self-bond orbit share exactly the same float
    vself = [sym.new_exc.exchange_Jdict[(R, 0, 0)] for R in Rself]
    assert len(set(vself)) == 1
    vnn = [sym.new_exc.exchange_Jdict[(R, 0, 1)] for R in Rnn]
    assert len(set(vnn)) == 1


def test_accidental_degeneracy_not_merged_triclinic():
    """Two bonds of identical length and identical site tags in a triclinic
    cell are not symmetry related and must NOT be merged (regression against
    the old distance-based heuristic)."""
    # |a| = |b| = 4.0 exactly, but the cell is fully triclinic: P1, one op.
    ay, az = 1.3, 0.9
    by = np.sqrt(16.0 - ay**2 - az**2)
    cell = [[4.0, 0.0, 0.0], [ay, by, az], [0.7, 0.5, 3.6]]
    atoms = _make_atoms("Fe", cell, [[0.0, 0.0, 0.0]])
    Jdict = {
        ((1, 0, 0), 0, 0): 2.0,
        ((-1, 0, 0), 0, 0): 2.1,
        ((0, 1, 0), 0, 0): 2.5,
        ((0, -1, 0), 0, 0): 2.6,
    }
    exc = _make_spinio(atoms, [[0, 0, 2.0]], Jdict)
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert ne.exchange_Jdict[((1, 0, 0), 0, 0)] == pytest.approx(2.05, abs=1e-12)
    assert ne.exchange_Jdict[((-1, 0, 0), 0, 0)] == pytest.approx(2.05, abs=1e-12)
    assert ne.exchange_Jdict[((0, 1, 0), 0, 0)] == pytest.approx(2.55, abs=1e-12)
    assert ne.exchange_Jdict[((0, -1, 0), 0, 0)] == pytest.approx(2.55, abs=1e-12)
    assert abs(
        ne.exchange_Jdict[((1, 0, 0), 0, 0)] - ne.exchange_Jdict[((0, 1, 0), 0, 0)]
    ) == pytest.approx(0.5, abs=1e-12)


def test_centering_translation_joins_orbits():
    """C-centered cell: the operation with t = (1/2, 1/2, 0) relates pairs that
    no t=0 operation relates; their orbits must be joined."""
    atoms = _make_atoms("Fe2", np.diag([3.0, 3.0, 4.0]), [[0, 0, 0], [0.5, 0.5, 0.0]])
    Jdict = {
        ((0, 0, 1), 0, 0): 1.0,
        ((0, 0, -1), 0, 0): 1.0,
        ((0, 0, 1), 1, 1): 1.2,
        ((0, 0, -1), 1, 1): 1.2,
    }
    exc = _make_spinio(atoms, [[0, 0, 2.0], [0, 0, 2.0]], Jdict)
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert ne.exchange_Jdict[((0, 0, 1), 0, 0)] == pytest.approx(1.1, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, -1), 0, 0)] == pytest.approx(1.1, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, 1), 1, 1)] == pytest.approx(1.1, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, -1), 1, 1)] == pytest.approx(1.1, abs=1e-12)


# ---------------------------------------------------------------------------
# DMI


def test_dmi_rotated_by_fourfold_operation():
    """P4 cell (C4 present, mirrors broken by satellite atoms): a fourfold
    rotation about z relates four directed bonds. The in-plane DMI components
    must be rotated into each other (averaging to zero) while the z component
    survives."""
    # two atoms on the rotation axis + four satellite atoms off-axis, so the
    # space group is P4 with proper rotations only
    atoms = _make_atoms(
        "Fe6",
        np.diag([3.0, 3.0, 3.7]),
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.31, 0.17, 0.6],
            [0.83, 0.31, 0.6],
            [0.69, 0.83, 0.6],
            [0.17, 0.69, 0.6],
        ],
    )
    C4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    D0 = np.array([0.1, 0.2, 0.3])
    Rs = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    Jdict, Ddict = {}, {}
    for k, R in enumerate(Rs):
        D = np.linalg.matrix_power(C4, k) @ D0
        Jdict[(R, 0, 1)] = 1.5
        Jdict[(tuple(-np.array(R)), 1, 0)] = 1.5
        Ddict[(R, 0, 1)] = D
        Ddict[(tuple(-np.array(R)), 1, 0)] = -D
    exc = _make_spinio(
        atoms,
        [[0, 0, 2.0]] * 6,
        Jdict,
        dmi_ddict=Ddict,
    )
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc
    for R in Rs:
        Rrev = tuple(-np.array(R))
        assert np.allclose(ne.dmi_ddict[(R, 0, 1)], [0.0, 0.0, 0.3], atol=1e-12)
        assert np.all(ne.dmi_ddict[(R, 0, 1)][:2] == 0.0)
        assert np.allclose(ne.dmi_ddict[(Rrev, 1, 0)], [0.0, 0.0, -0.3], atol=1e-12)
        assert ne.exchange_Jdict[(R, 0, 1)] == pytest.approx(1.5, abs=1e-12)
    # reversal identity of the output
    for R in Rs:
        Rrev = tuple(-np.array(R))
        assert np.allclose(
            ne.dmi_ddict[(Rrev, 1, 0)], -ne.dmi_ddict[(R, 0, 1)], atol=1e-14
        )


def test_inversion_symmetric_bond_D_exactly_zero():
    """Orthorhombic Pmmm: the ±x bonds form one orbit closed under inversion;
    any stored DMI must be driven to exactly zero."""
    atoms = _make_atoms("Fe", np.diag([4.0, 4.1, 5.0]), [[0, 0, 0]])
    Jdict = {((1, 0, 0), 0, 0): 2.0, ((-1, 0, 0), 0, 0): 2.0}
    Ddict = {
        ((1, 0, 0), 0, 0): np.array([0.3, 0.0, 0.0]),
        ((-1, 0, 0), 0, 0): np.array([-0.3, 0.0, 0.0]),
    }
    exc = _make_spinio(atoms, [[0, 0, 2.0]], Jdict, dmi_ddict=Ddict)
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert np.all(ne.dmi_ddict[((1, 0, 0), 0, 0)] == 0.0)
    assert np.all(ne.dmi_ddict[((-1, 0, 0), 0, 0)] == 0.0)
    assert ne.exchange_Jdict[((1, 0, 0), 0, 0)] == pytest.approx(2.0, abs=1e-14)


def test_reversal_consistency_enforced_from_noisy_data():
    """Stored pairs that violate Gamma_ji(-R) = Gamma_ij(R)^T are restored to
    exact reversal consistency by the symmetrization."""
    cell = [[3.0, 0.0, 0.0], [1.3, 3.6, 0.2], [0.4, 0.5, 3.9]]
    atoms = _make_atoms("Fe2", cell, [[0.11, 0.13, 0.17], [0.31, 0.29, 0.41]])
    rng = np.random.default_rng(11)
    M = rng.normal(size=(3, 3))
    Jani1 = 0.5 * (M + M.T)
    Jani2 = Jani1 + 0.5 * ((rng.normal(size=(3, 3))) + (rng.normal(size=(3, 3))).T)
    exc = _make_spinio(
        atoms,
        [[0, 0, 2.0], [0, 0, 2.0]],
        {((1, 0, 0), 0, 0): 1.0, ((-1, 0, 0), 0, 0): 1.3},
        dmi_ddict={
            ((1, 0, 0), 0, 0): np.array([0.1, 0.2, 0.3]),
            ((-1, 0, 0), 0, 0): np.array([-0.05, -0.25, -0.4]),
        },
        Jani_dict={((1, 0, 0), 0, 0): Jani1, ((-1, 0, 0), 0, 0): Jani2},
    )
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert np.allclose(
        ne.dmi_ddict[((-1, 0, 0), 0, 0)],
        -ne.dmi_ddict[((1, 0, 0), 0, 0)],
        atol=1e-14,
    )
    assert np.allclose(
        ne.Jani_dict[((-1, 0, 0), 0, 0)], ne.Jani_dict[((1, 0, 0), 0, 0)].T, atol=1e-14
    )
    assert ne.exchange_Jdict[((-1, 0, 0), 0, 0)] == pytest.approx(
        ne.exchange_Jdict[((1, 0, 0), 0, 0)], abs=1e-14
    )


# ---------------------------------------------------------------------------
# magnetic space group


def _afm_cell():
    atoms = _make_atoms(
        "Fe2O2",
        np.diag([3.0, 3.0, 3.2]),
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0, 0, 0.5]],
    )
    return atoms


def _afm_exchange(noise_rng):
    """NN Fe-Fe body-diagonal bonds with reversal-consistent D and a
    symmetry-allowed Jani."""
    Rs = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]
    Jani = np.diag([0.2, 0.2, -0.4])
    Jdict, Ddict, Janidict = {}, {}, {}
    for R in Rs:
        Rrev = tuple(-np.array(R))
        Jf = 1.5 + noise_rng.normal(scale=0.02)
        Jb = 1.5 + noise_rng.normal(scale=0.02)
        Jdict[(R, 0, 1)] = Jf
        Jdict[(Rrev, 1, 0)] = Jb
        Ddict[(R, 0, 1)] = np.array([0.05, 0.02, 0.3]) + noise_rng.normal(size=3) * 0.01
        Ddict[(Rrev, 1, 0)] = -Ddict[(R, 0, 1)]
    for key in Jdict:
        Janidict[key] = Jani
    return Jdict, Ddict, Janidict


def test_magnetic_primed_half_translation_afrm(capsys):
    """Collinear AFM cell (|m| equal on both sublattices): the magnetic space
    group contains the primed half-translation. Primed operations use the same
    tensor action, so the DMI on the sublattice-exchanging bond is killed
    while J^ani is preserved."""
    atoms = _afm_cell()
    spinat = [[0, 0, 2.5], [0, 0, -2.5], [0, 0, 0], [0, 0, 0]]
    Jdict, Ddict, Janidict = _afm_exchange(np.random.default_rng(21))
    exc = _make_spinio(atoms, spinat, Jdict, dmi_ddict=Ddict, Jani_dict=Janidict)
    sym = TB2JSymmetrizer(exc, verbose=True, magnetic=True)
    sym.symmetrize_J()
    out = capsys.readouterr().out
    assert "magnetic space group" in out
    assert "with time reversal" in out
    ne = sym.new_exc
    Rs = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]
    for R in Rs:
        Rrev = tuple(-np.array(R))
        assert np.all(ne.dmi_ddict[(R, 0, 1)] == 0.0)
        assert np.all(ne.dmi_ddict[(Rrev, 1, 0)] == 0.0)
        assert np.allclose(
            ne.Jani_dict[(R, 0, 1)], np.diag([0.2, 0.2, -0.4]), atol=1e-12
        )
        assert np.allclose(
            ne.Jani_dict[(Rrev, 1, 0)], np.diag([0.2, 0.2, -0.4]), atol=1e-12
        )
    jvals = set(ne.exchange_Jdict.values())
    assert len(jvals) == 1  # all eight keys merged into one orbit
    meanJ = np.mean(list(exc.exchange_Jdict.values()))
    assert jvals.pop() == pytest.approx(meanJ, abs=1e-12)

    # For an equal-|m| collinear AFM the spatial parts of the MSG coincide
    # with the crystallographic group, so the crystal path gives the same
    # result (a pinned-contract consequence; see module docstring).
    exc2 = _make_spinio(
        atoms, spinat, dict(Jdict), dmi_ddict=dict(Ddict), Jani_dict=dict(Janidict)
    )
    sym2 = TB2JSymmetrizer(exc2, verbose=False, magnetic=False)
    sym2.symmetrize_J()
    for key, val in sym.new_exc.exchange_Jdict.items():
        assert val == pytest.approx(sym2.new_exc.exchange_Jdict[key], abs=1e-10)
        assert np.allclose(
            sym.new_exc.dmi_ddict[key], sym2.new_exc.dmi_ddict[key], atol=1e-10
        )
        assert np.allclose(
            sym.new_exc.Jani_dict[key], sym2.new_exc.Jani_dict[key], atol=1e-10
        )


def test_magnetic_group_respects_sublattice_moments():
    """Ferrimagnetic cell (|m| different on the two Fe sublattices): the
    crystallographic group wrongly merges the Fe0-Fe0 and Fe1-Fe1 bonds
    through the centering translation, while the magnetic space group keeps
    them apart."""
    atoms = _afm_cell()
    spinat = [[0, 0, 2.5], [0, 0, -2.0], [0, 0, 0], [0, 0, 0]]
    Jdict = {
        ((0, 0, 1), 0, 0): 1.0,
        ((0, 0, -1), 0, 0): 1.0,
        ((0, 0, 1), 1, 1): 2.0,
        ((0, 0, -1), 1, 1): 2.0,
    }
    exc_msg = _make_spinio(atoms, spinat, dict(Jdict))
    sym = TB2JSymmetrizer(exc_msg, verbose=False, magnetic=True)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert ne.exchange_Jdict[((0, 0, 1), 0, 0)] == pytest.approx(1.0, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, -1), 0, 0)] == pytest.approx(1.0, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, 1), 1, 1)] == pytest.approx(2.0, abs=1e-12)
    assert ne.exchange_Jdict[((0, 0, -1), 1, 1)] == pytest.approx(2.0, abs=1e-12)

    exc_sg = _make_spinio(atoms, spinat, dict(Jdict))
    sym2 = TB2JSymmetrizer(exc_sg, verbose=False, magnetic=False)
    sym2.symmetrize_J()
    for key in Jdict:
        assert sym2.new_exc.exchange_Jdict[key] == pytest.approx(1.5, abs=1e-12)


def test_magnetic_requires_spinat():
    atoms = _afm_cell()
    exc = _make_spinio(atoms, None, {((0, 0, 1), 0, 0): 1.0})
    sym = TB2JSymmetrizer(exc, verbose=False, magnetic=True)
    with pytest.raises(ValueError, match="spinat"):
        sym.symmetrize_J()


# ---------------------------------------------------------------------------
# single-ion anisotropy, Jonly, IO


def test_sia_averaged_over_operation_orbit():
    """P4 cell (mirrors broken by satellite atoms): site 1 sits on the
    rotation axis; the x/y components of its SIA tensor must be symmetrized
    to equality while z is preserved, and site 0 is already invariant."""
    atoms = _make_atoms(
        "Fe6",
        np.diag([3.0, 3.0, 3.7]),
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.31, 0.17, 0.6],
            [0.83, 0.31, 0.6],
            [0.69, 0.83, 0.6],
            [0.17, 0.69, 0.6],
        ],
    )
    Jdict = {((1, 0, 0), 0, 1): 1.0, ((-1, 0, 0), 1, 0): 1.0}
    sia = {
        0: np.diag([0.5, 0.5, 0.1]),
        1: np.diag([0.5, 0.6, 0.1]),
    }
    exc = _make_spinio(atoms, [[0, 0, 2.0]] * 6, Jdict, sia_tensor=sia)
    assert exc.has_sia_tensor
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    K0 = sym.new_exc.sia_tensor[0]
    K1 = sym.new_exc.sia_tensor[1]
    # site 0: tensor already symmetric under the site point group
    assert np.allclose(K0, np.diag([0.5, 0.5, 0.1]), atol=1e-12)
    # site 1: x/y equalized by the rotations about z, z preserved
    assert K1[0, 0] == pytest.approx(K1[1, 1], abs=1e-12)
    assert K1[2, 2] == pytest.approx(0.1, abs=1e-12)
    assert np.trace(K1) == pytest.approx(1.2, abs=1e-12)
    assert np.allclose(K1, np.diag([0.55, 0.55, 0.1]), atol=1e-12)


def test_jonly_discards_dmi_jani_sia():
    atoms = _make_atoms("Fe2", np.diag([3.0, 3.0, 3.7]), [[0, 0, 0], [0, 0, 0.3]])
    Jdict = {((1, 0, 0), 0, 1): 1.5, ((-1, 0, 0), 1, 0): 1.5}
    Ddict = {
        ((1, 0, 0), 0, 1): np.array([0.0, 0.0, 0.3]),
        ((-1, 0, 0), 1, 0): np.array([0.0, 0.0, -0.3]),
    }
    sia = {0: np.diag([0.5, 0.5, 0.1]), 1: np.diag([0.5, 0.5, 0.1])}
    exc = _make_spinio(
        atoms, [[0, 0, 2.0], [0, 0, 2.0]], Jdict, dmi_ddict=Ddict, sia_tensor=sia
    )
    sym = TB2JSymmetrizer(exc, verbose=False, Jonly=True)
    sym.symmetrize_J()
    ne = sym.new_exc
    assert ne.dmi_ddict is None
    assert ne.Jani_dict is None
    assert not ne.has_sia_tensor
    assert ne.sia_tensor is None
    assert ne.exchange_Jdict[((1, 0, 0), 0, 1)] == pytest.approx(1.5, abs=1e-12)
    # the original object is untouched
    assert exc.dmi_ddict is not None


def test_write_all_roundtrip(tmp_path):
    atoms = _make_atoms("Fe2", np.diag([3.0, 3.0, 3.7]), [[0, 0, 0], [0, 0, 0.3]])
    Jdict = {((1, 0, 0), 0, 1): 1.5, ((-1, 0, 0), 1, 0): 1.5}
    Ddict = {
        ((1, 0, 0), 0, 1): np.array([0.0, 0.0, 0.3]),
        ((-1, 0, 0), 1, 0): np.array([0.0, 0.0, -0.3]),
    }
    exc = _make_spinio(atoms, [[0, 0, 2.0], [0, 0, 2.0]], Jdict, dmi_ddict=Ddict)
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    out = tmp_path / "symmetrized"
    sym.output(path=str(out))
    loaded = SpinIO.load_pickle(path=str(out))
    assert loaded.exchange_Jdict == sym.new_exc.exchange_Jdict
    for key, val in sym.new_exc.dmi_ddict.items():
        assert np.allclose(loaded.dmi_ddict[key], val)
    assert np.allclose(loaded.atoms.cell.array, atoms.cell.array)


# ---------------------------------------------------------------------------
# symmetrize_exchange (symmetry from a provided target structure)


def test_symmetrize_exchange_with_target_structure():
    """Symmetrizing a distorted cell to its ideal tetragonal parent: bonds
    that become equivalent under the ideal symmetry are averaged."""
    cell = np.diag([3.0, 3.08, 3.7])
    atoms = _make_atoms("Fe2", cell, [[0, 0, 0], [0.002, 0.0, 0.3]])
    # ideal parent: x and y bonds of site 0->1 equivalent under C4
    Rs = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    Jdict, Ddict = {}, {}
    rng = np.random.default_rng(31)
    for R in Rs:
        v = 1.5 + rng.normal(scale=0.05)
        Jdict[(R, 0, 1)] = v
        Jdict[(tuple(-np.array(R)), 1, 0)] = v
        Ddict[(R, 0, 1)] = np.array([0.0, 0.0, 0.3])
        Ddict[(tuple(-np.array(R)), 1, 0)] = np.array([0.0, 0.0, -0.3])
    spinio = _make_spinio(atoms, [[0, 0, 2.0], [0, 0, 2.0]], Jdict, dmi_ddict=Ddict)
    ideal = _make_atoms("Fe2", np.diag([3.0, 3.0, 3.7]), [[0, 0, 0], [0.0, 0.0, 0.3]])
    symmetrize_exchange(spinio, ideal, symprec=1e-2)
    # x and y bonds are now equivalent (one orbit), and the output is
    # reversal-consistent; the orbit mean is near the input mean. Some
    # operations have non-integer bond images in the distorted basis and are
    # skipped, so the mean is not exactly the naive input mean.
    jvals = {R: spinio.exchange_Jdict[(R, 0, 1)] for R in Rs}
    assert len(set(jvals.values())) == 1
    assert jvals[(1, 0, 0)] == pytest.approx(
        float(np.mean(list(Jdict.values()))), abs=5e-3
    )
    for R in Rs:
        Rrev = tuple(-np.array(R))
        assert spinio.exchange_Jdict[(Rrev, 1, 0)] == pytest.approx(
            spinio.exchange_Jdict[(R, 0, 1)], abs=1e-12
        )
    # the ideal parent has vertical mirrors, which force the DMI of these
    # bonds to zero
    assert np.allclose(spinio.dmi_ddict[((1, 0, 0), 0, 1)], 0.0, atol=1e-8)
    assert np.allclose(spinio.dmi_ddict[((-1, 0, 0), 1, 0)], 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# additional coverage: noncollinear MSG, SIA sublattices, zero_tol, monoclinic


def test_magnetic_noncollinear_spinat_axial(capsys):
    """Noncollinear-form spinat (3-vectors, SpinIO colinear=False): the
    moments go to spglib as axial vectors, the primed half-translation is
    still found, and the sublattice-exchanging DMI is killed while J^ani is
    preserved."""
    atoms = _afm_cell()
    spinat = np.array(
        [[0.0, 0.0, 2.5], [0.0, 0.0, -2.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    Jdict, Ddict, Janidict = _afm_exchange(np.random.default_rng(23))
    keys = list(Jdict)
    exc = SpinIO(
        atoms,
        spinat,
        [0] * 4,
        [0, 1, 2, 3],
        colinear=False,
        distance_dict=_distance_dict(atoms, [0, 1, 2, 3], keys),
        exchange_Jdict=Jdict,
        dmi_ddict=Ddict,
        Jani_dict=Janidict,
    )
    sym = TB2JSymmetrizer(exc, verbose=True, magnetic=True)
    sym.symmetrize_J()
    assert "magnetic space group" in capsys.readouterr().out
    ne = sym.new_exc
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]:
        assert np.all(ne.dmi_ddict[(R, 0, 1)] == 0.0)
        assert np.allclose(
            ne.Jani_dict[(R, 0, 1)], np.diag([0.2, 0.2, -0.4]), atol=1e-10
        )


def test_magnetic_sia_respects_sublattice_moments():
    """Ferrimagnetic cell: the crystallographic group merges the Fe0/Fe1 site
    orbits through the centering translation and wrongly averages their SIA
    tensors; the magnetic space group keeps them apart."""
    atoms = _afm_cell()
    spinat = [[0, 0, 2.5], [0, 0, -2.0], [0, 0, 0], [0, 0, 0]]
    Jdict = {((0, 0, 1), 0, 0): 1.0, ((0, 0, -1), 0, 0): 1.0}
    sia = {0: np.diag([0.3, 0.0, -0.3]), 1: np.diag([0.6, 0.0, -0.6])}
    exc = _make_spinio(atoms, spinat, dict(Jdict), sia_tensor=dict(sia))
    sym = TB2JSymmetrizer(exc, verbose=False, magnetic=True)
    sym.symmetrize_J()
    ne = sym.new_exc
    # Fe0 sits on the 4-fold axis: xx == yy is forced within each sublattice,
    # but the two sublattices must stay distinct (no op maps one to the other)
    K0, K1 = ne.sia_tensor[0], ne.sia_tensor[1]
    assert K0[0, 0] == pytest.approx(K0[1, 1], abs=1e-12)
    assert K0[2, 2] == pytest.approx(-0.3, abs=1e-12)
    assert K1[0, 0] == pytest.approx(K1[1, 1], abs=1e-12)
    assert K1[2, 2] == pytest.approx(-0.6, abs=1e-12)
    assert abs(K0[0, 0] - K1[0, 0]) > 0.1

    # crystallographic path merges the sublattices through the centering
    exc2 = _make_spinio(atoms, spinat, dict(Jdict), sia_tensor=dict(sia))
    sym2 = TB2JSymmetrizer(exc2, verbose=False, magnetic=False)
    sym2.symmetrize_J()
    assert np.allclose(
        sym2.new_exc.sia_tensor[0], sym2.new_exc.sia_tensor[1], atol=1e-12
    )


def test_zero_tol_keeps_allowed_and_snaps_forbidden_dmi():
    """P4 cell with D_z allowed: the default zero_tol leaves the surviving
    D_z untouched, while zero_tol=0.5 snaps the whole vector to exact zero
    and leaves the isotropic J untouched."""
    atoms = _make_atoms(
        "Fe6",
        np.diag([3.0, 3.0, 3.7]),
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.31, 0.17, 0.6],
            [0.83, 0.31, 0.6],
            [0.69, 0.83, 0.6],
            [0.17, 0.69, 0.6],
        ],
    )
    Rs = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]

    def build():
        Jdict, Ddict = {}, {}
        for R in Rs:
            Rrev = tuple(-np.array(R))
            Jdict[(R, 0, 1)] = 1.5
            Jdict[(Rrev, 1, 0)] = 1.5
            Ddict[(R, 0, 1)] = np.array([0.1, 0.2, 0.3])
            Ddict[(Rrev, 1, 0)] = -Ddict[(R, 0, 1)]
        return _make_spinio(atoms, [[0, 0, 2.0]] * 6, Jdict, dmi_ddict=Ddict)

    sym = TB2JSymmetrizer(build(), verbose=False)
    sym.symmetrize_J()
    D = sym.new_exc.dmi_ddict[((1, 0, 0), 0, 1)]
    assert np.all(D[:2] == 0.0)
    assert D[2] == pytest.approx(0.3, abs=1e-12)

    sym2 = TB2JSymmetrizer(build(), verbose=False, zero_tol=0.5)
    sym2.symmetrize_J()
    D2 = sym2.new_exc.dmi_ddict[((1, 0, 0), 0, 1)]
    assert np.all(D2 == 0.0)
    assert sym2.new_exc.exchange_Jdict[((1, 0, 0), 0, 1)] == pytest.approx(1.5)


def test_cli_end_to_end_magnetic(tmp_path):
    """Full CLI pass: write a TB2J results directory, run the CLI with
    --magnetic/--zero-tol, reload the output and check the MSG-forced zeros."""
    ind = tmp_path / "in"
    outd = tmp_path / "out"
    atoms = _afm_cell()
    spinat = [[0, 0, 2.5], [0, 0, -2.5], [0, 0, 0], [0, 0, 0]]
    Jdict, Ddict, Janidict = _afm_exchange(np.random.default_rng(31))
    _make_spinio(atoms, spinat, Jdict, dmi_ddict=Ddict, Jani_dict=Janidict).write_all(
        path=str(ind)
    )
    argv = sys.argv
    sys.argv = [
        "TB2J_symmetrize.py",
        "-i",
        str(ind),
        "-o",
        str(outd),
        "--magnetic",
        "--zero-tol",
        "1e-6",
    ]
    try:
        from TB2J.symmetrize_J import symmetrize_J_cli

        symmetrize_J_cli()
    finally:
        sys.argv = argv
    loaded = SpinIO.load_pickle(path=str(outd))
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]:
        assert np.all(loaded.dmi_ddict[(R, 0, 1)] == 0.0)
        assert np.allclose(
            loaded.Jani_dict[(R, 0, 1)], np.diag([0.2, 0.2, -0.4]), atol=1e-10
        )


def test_monoclinic_p21c_tensor_invariance():
    """C2/m monoclinic cell with beta = 100 deg: the cartesian rotation W_c is
    non-diagonal and the operations carry centering translation parts.  With a
    bond-distance-cutoff seed (closed under all operations, as real TB2J data
    is), the symmetrized tensors must map exactly onto each other under every
    operation."""
    beta = np.radians(100.0)
    cell = [
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [5.0 * np.cos(beta), 0.0, 5.0 * np.sin(beta)],
    ]
    atoms = _make_atoms("Fe2", cell, [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    ops = crystal_symmetry_ops(atoms, symprec=1e-5)
    assert len(ops) > 1  # genuine C2/m, not P1

    # bond-cutoff seed set, pair-complete, closed under all operations
    xA = np.zeros(3)
    xB = np.array([0.0, 0.5, 0.5])
    dcut = 3.4
    R01 = sorted(
        tuple(int(v) - 4 for v in R)
        for R in np.ndindex(9, 9, 9)
        if np.linalg.norm(xB + (np.array(R) - 4) - xA) <= dcut
    )
    rng = np.random.default_rng(11)
    Jdict, Ddict, Janidict = {}, {}, {}
    for R in R01:
        Rrev = tuple(-np.array(R))
        Jdict[(R, 0, 1)] = 1.0
        Jdict[(Rrev, 1, 0)] = 1.0
        Ddict[(R, 0, 1)] = rng.normal(size=3)
        Ddict[(Rrev, 1, 0)] = -Ddict[(R, 0, 1)]
        M = rng.normal(size=(3, 3))
        Janidict[(R, 0, 1)] = 0.5 * (M + M.T)
        Janidict[(Rrev, 1, 0)] = Janidict[(R, 0, 1)]
    exc = _make_spinio(
        atoms, [[0, 0, 2.0]] * 2, Jdict, dmi_ddict=Ddict, Jani_dict=Janidict
    )
    sym = TB2JSymmetrizer(exc, verbose=False)
    sym.symmetrize_J()
    ne = sym.new_exc

    xfrac = np.asarray(atoms.get_scaled_positions(), dtype=float)
    cella = np.asarray(atoms.get_cell().array, dtype=float)
    smaps = _build_site_maps(xfrac, cella, ops, 1e-5)
    keyset = {(R, i, j) for (R, i, j) in ne.exchange_Jdict}
    cell_inv = np.linalg.inv(cella)
    nops_closed = 0
    nchecked = 0
    for W, smap in zip(ops.rotations, smaps):
        Wc = cella @ np.asarray(W, dtype=float) @ cell_inv
        images = {}
        closed = True
        for R, i, j in keyset:
            ia, ja = ne.iatom(i), ne.iatom(j)
            ia2, ja2 = int(smap[ia]), int(smap[ja])
            d = xfrac[ja] + np.asarray(R, dtype=float) - xfrac[ia]
            diff = np.asarray(W, dtype=float) @ d - (xfrac[ja2] - xfrac[ia2])
            R2 = np.round(diff)
            if np.max(np.abs(diff - R2)) > 1e-6:
                closed = False
                break
            images[(R, i, j)] = (tuple(int(x) for x in R2), ia2, ja2)
        if not closed:
            continue
        nops_closed += 1
        for (R, i, j), (R2, i2, j2) in images.items():
            G1 = combine_tensor(
                ne.exchange_Jdict[(R, i, j)],
                ne.dmi_ddict[(R, i, j)],
                ne.Jani_dict[(R, i, j)],
            )
            G2 = combine_tensor(
                ne.exchange_Jdict[(R2, i2, j2)],
                ne.dmi_ddict[(R2, i2, j2)],
                ne.Jani_dict[(R2, i2, j2)],
            )
            assert np.allclose(G2, Wc @ G1 @ Wc.T, atol=1e-7)
            nchecked += 1
    assert nops_closed == len(ops)  # cutoff seed is closed under the group
    assert nchecked > len(ops)


# ---------------------------------------------------------------------------
# CLI


def test_cli_help():
    proc = subprocess.run(
        [sys.executable, "-c", "from TB2J.symmetrize_J import symmetrize_J_cli"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run(
        [sys.executable, "-m", "TB2J.symmetrize_J", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--magnetic" in proc.stdout
    assert "--zero-tol" in proc.stdout
    assert "--Jonly" in proc.stdout


if __name__ == "__main__":
    import inspect

    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            if (
                "capsys" in fn.__code__.co_varnames
                or "tmp_path" in fn.__code__.co_varnames
            ):
                print(f"SKIP {name} (needs pytest fixtures)")
                continue
            print(f"RUN {name}")
            fn()
