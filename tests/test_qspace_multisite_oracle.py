"""Independent multi-site oracle (Story 009).

Two inequivalent magnetic sites per cell, asymmetric inter-cell hoppings
(non-centrosymmetric), complete odd 3x1x1 mesh, real gauge, collinear
SOC-free channels.  The real-space route (ExchangeCL2 pair table) is the
independent oracle for the reciprocal-space driver (ExchangeCLQspace):
no production integration machinery is reused across the two routes.

Authority: docs/sympy/qspace_multisite_transform.py

  S1  the q -> R inverse transform of the per-q complex kernel carries the
      PLUS phase exp(+2 pi i R.q) (pinned by get_GR's k2Rfactor = -2 pi i);
      the minus phase yields the transposed pair table T_ji(R).
  S2  the imaginary extraction must be taken after the transform; the
      per-q imaginary part does not commute with the phase transform on
      asymmetric inter-sublattice pairs.

Failing-before record (both defects active): the inter-sublattice entries
of the q-space pair table equal the real-space oracle TRANSPOSED at R != 0
(max deviation 1.6e-3 eV on this fixture); diagonals (R-even entries) were
already exact.
"""

import numpy as np
from ase import Atoms
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian

from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.exchangeCL2 import ExchangeCL2

NZ = 4097


def build_models():
    # Up channel occupied (band below mu = 0), down channel empty; site A
    # and B inequivalent (splittings dA != dB); asymmetric tR != tL.
    ea = eb = 0.10
    tAB, tR, tL = 0.08, 0.05, 0.03
    dA, dB = -0.35, -0.25
    rlist = np.array([(-1, 0, 0), (0, 0, 0), (1, 0, 0)])
    atoms = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[3.0, 12.0, 12.0],
        pbc=[True, False, False],
    )
    models = []
    for sign in (+1.0, -1.0):
        h0 = np.array([[ea + sign * dA, tAB], [tAB, eb + sign * dB]])
        hp = np.array([[0.0, tR], [tL, 0.0]])
        hm = np.array([[0.0, tL], [tR, 0.0]])
        models.append(
            LCAOHamiltonian(
                HR=np.array([hm, h0, hp]),
                SR=np.array([np.zeros((2, 2)), np.eye(2), np.zeros((2, 2))]),
                Rlist=rlist,
                nbasis=2,
                atoms=atoms,
                nspin=1,
                orth=True,
            )
        )
    return models, atoms


def run_pair():
    models, atoms = build_models()
    common = dict(
        tbmodels=tuple(models),
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|s", "Fe2|s"],
        efermi=0.0,
        nproc=1,
    )
    calc_q = ExchangeCLQspace(
        kmesh=[3, 1, 1], Rcut=8.0, nz=NZ, method="semicircle", **common
    )
    calc_r = ExchangeCL2(
        kmesh=[3, 1, 1], Rcut=8.0, nz=NZ, method="semicircle", **common
    )
    calc_q.calculate_all()
    calc_r.calculate_all()
    return calc_q, calc_r


def pair_table(calc):
    ind = list(calc.ind_mag_atoms)
    rlist = sorted(calc.Rlist)
    out = {}
    for r in rlist:
        for i, ia in enumerate(ind):
            for j, ja in enumerate(ind):
                key = (tuple(r), ia, ja)
                if key not in calc.exchange_Jdict:
                    continue
                out[key] = complex(calc.exchange_Jdict[key])
    return out


def test_multisite_qspace_matches_realspace_oracle():
    calc_q, calc_r = run_pair()
    tq = pair_table(calc_q)
    tr = pair_table(calc_r)
    assert set(tq) == set(tr), (sorted(tq), sorted(tr))
    worst = max(abs(tq[k] - tr[k]) for k in tq)
    assert worst < 1e-9, f"q-space pair table differs from real-space oracle: {worst}"


def test_multisite_reciprocity_of_pair_table():
    calc_q, _ = run_pair()
    tq = pair_table(calc_q)
    for (r, i, j), val in tq.items():
        mirror = tq.get((tuple(-np.asarray(r)), j, i))
        assert mirror is not None, f"missing mirrored pair for {(r, i, j)}"
        assert abs(val - mirror) < 1e-9


if __name__ == "__main__":
    test_multisite_qspace_matches_realspace_oracle()
    test_multisite_reciprocity_of_pair_table()
    print("oracle green")
