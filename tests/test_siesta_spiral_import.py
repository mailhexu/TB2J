"""Tests for the SIESTA spiral importer (story 002, epic 001).

Grounded in the derivation pack `docs/sympy/siesta_spiral_pencil.py`:
the importer must assemble the rotating-frame pencil exactly per the
pinned conventions (q/2 phases per spin sector on H and S, no
hermitization on the Hamiltonian path), and must fail closed on
undeclared or inconsistent conventions.

The synthetic fixture mirrors the derivation's ring model: random
nonorthogonal multi-orbital spin-diagonal blocks with the physical
pairing, at an even-m commensurate pitch where the union of folded
pencil eigenvalues equals the supercell ring spectrum.
"""

import numpy as np
import pytest
import scipy.linalg

from TB2J.interfaces.siesta_spiral import (
    BUNDLE_VERSION,
    SpiralBundle,
    SpiralImporter,
    SpiralImportError,
)


def _herm2(rng):
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return (a + a.conj().T) / 2


def _build_ring_model(ncell=6, norb=2, nR=3, seed=20261012):
    rng = np.random.default_rng(seed)
    H_blocks, S_blocks = {}, {}
    for R in range(nR):
        for mu in range(norb):
            for nu in range(norb):
                if R == 0 and mu > nu:
                    continue
                blk = _herm2(rng)
                blk = np.diag([blk[0, 0].real + 0.31, blk[1, 1].real - 0.27])
                H_blocks[(mu, nu, R)] = blk.astype(complex)
                H_blocks[(nu, mu, -R)] = blk.astype(complex)
                val = float(rng.normal())
                S_blocks[(mu, nu, R)] = val
                S_blocks[(nu, mu, -R)] = val
    for mu in range(norb):
        S_blocks[(mu, mu, 0)] += 4.0 * norb
    return H_blocks, S_blocks


def _make_bundle(ncell=6, norb=2, q_num=2, spin_packing="H4=-ImH12"):
    H_blocks, S_blocks = _build_ring_model(norb=norb)
    # stored 4-component form: H1, H2 real; H3 = Re H12, H4 = -Im H12
    stored = {}
    for key, blk in H_blocks.items():
        h12 = blk[0, 1]
        stored[key] = (blk[0, 0].real, blk[1, 1].real, h12.real, -h12.imag)
    return SpiralBundle(
        version=BUNDLE_VERSION,
        norb=norb,
        ncell=ncell,
        q=(2.0 * np.pi * q_num / ncell,),
        spin_packing=spin_packing,
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse=stored,
        overlap=S_blocks,
    )


class TestAssembly:
    def test_folded_union_equals_ring_spectrum(self):
        ncell, norb = 6, 2
        bundle = _make_bundle(ncell=ncell, norb=norb)
        importer = SpiralImporter()
        # ring reference
        n = ncell * norb
        Hr = np.zeros((2 * n, 2 * n), complex)
        Sr = np.zeros((2 * n, 2 * n))
        for c in range(ncell):
            for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
                j = ((c + R) % ncell) * norb + nu
                i = c * norb + mu
                Hr[i, j] += h1
                Hr[i + n, j + n] += h2
                Sv = bundle.overlap[(mu, nu, R)]
                Sr[i, j] += Sv
                Sr[i + n, j + n] += Sv
        ev_ring = np.sort(scipy.linalg.eigh(Hr, Sr, eigvals_only=True))
        ev = []
        for k in 2.0 * np.pi * np.arange(ncell) / ncell:
            Hk, Sk = importer.assemble_pencil(bundle, k)
            assert np.allclose(Hk, Hk.conj().T, atol=1e-12)
            assert np.allclose(Sk, Sk.conj().T, atol=1e-12)
            ev.append(scipy.linalg.eigh(Hk, Sk, eigvals_only=True))
        assert np.max(np.abs(np.sort(np.concatenate(ev)) - ev_ring)) < 1e-10

    def test_q_half_channel_split(self):
        bundle = _make_bundle()
        importer = SpiralImporter()
        k, q = 0.3, bundle.q[0]
        _, Sk = importer.assemble_pencil(bundle, k)

        def S_lab(kk):
            out = np.zeros((2, 2), complex)
            for (mu, nu, R), v in bundle.overlap.items():
                out[mu, nu] += v * np.exp(-1j * kk * R)
            return out

        assert np.allclose(Sk[:2, :2], S_lab(k + q / 2), atol=1e-12)
        assert np.allclose(Sk[2:, 2:], S_lab(k - q / 2), atol=1e-12)


class TestFailClosed:
    def test_unknown_packing_refused(self):
        bundle = _make_bundle(spin_packing="H3=ImH12")
        with pytest.raises(SpiralImportError, match="spin_packing"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_unknown_phase_convention_refused(self):
        bundle = _make_bundle()
        bundle.phase_convention = "full_q"
        with pytest.raises(SpiralImportError, match="phase_convention"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_unversioned_bundle_refused(self):
        bundle = _make_bundle()
        bundle.version = "0.0-unknown"
        with pytest.raises(SpiralImportError, match="version"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_asymmetric_overlap_refused(self):
        bundle = _make_bundle()
        bundle.overlap[(0, 1, 2)] = 5.0
        with pytest.raises(SpiralImportError, match="overlap pairing"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_inconsistent_sparse_diagonal_refused(self):
        bundle = _make_bundle()
        h1, h2, h3, h4 = bundle.sparse[(0, 1, 1)]
        bundle.sparse[(1, 0, -1)] = (h1 + 0.5, h2, h3, h4)
        with pytest.raises(SpiralImportError, match="spin-diagonal components"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_missing_overlap_entry_refused(self):
        bundle = _make_bundle()
        del bundle.overlap[(0, 1, 1)]
        del bundle.overlap[(1, 0, -1)]
        with pytest.raises(SpiralImportError, match="missing overlap"):
            SpiralImporter().assemble_pencil(bundle, 0.0)

    def test_unknown_pitch_sign_refused(self):
        bundle = _make_bundle()
        bundle.pitch_sign = "opposite"
        with pytest.raises(SpiralImportError, match="pitch_sign"):
            SpiralImporter().assemble_pencil(bundle, 0.0)
