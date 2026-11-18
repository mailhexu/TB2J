"""Tests for the ABINIT spinor PAW channel (nspden=4, nspinor=2) — Story 011.

Source-verified conventions (ABINIT libpaw):

* ``m_pawdij.F90`` stores ``paw_ij%dij(cplex_dij*qphase*lmn2_size, ndij)``
  with ``dij(:,1)=up-up``, ``dij(:,2)=dwn-dwn``, ``dij(:,3)=up-dwn``,
  ``dij(:,4)=dwn-up``; the spinor packing is the operator Hermiticity
  ``D^{ss'}_{ij} = conjg(D^{s's}_{ji})`` — transposed site indices across
  the cross-spin blocks, NOT elementwise conjugation of block 3 (the
  elementwise form only coincides for a symmetric up-dwn block).  For
  nspden=4, ``cplex_dij=2``: real/imaginary interleaved along the lmn2 axis.
* ``pawdij_print_dij`` labels the four blocks per atom
  ``Atom # N - Component up-up / dwn-dwn / up-dwn / dwn-up`` and
  ``pawio_print_ij`` prints each block as a full square matrix with a
  ``=== REAL PART:`` and a ``=== IMAGINARY PART:`` section (f9.5 rows),
  filling each block's lower triangle from ``conjg`` of the partner
  component (``idij_sym = 7 - idij``, ``opt_sym=2``), so the printed
  blocks satisfy ``dwn-up == conj(up-dwn).T`` and up-up/dwn-dwn are
  Hermitian (complex for SOC prints).
* The Pauli splitting follows the nspden=4 packing contract: per
  projector pair ``B = ((D3+D4)/2, i(D3-D4)/2, (D1-D2)/2)`` (Hermitian;
  reduces to ``(Re D3, -Im D3, (D1-D2)/2)`` for symmetric ``D3``),
  ``Delta = 2 B.sigma`` (same convention as ``abinao.spinor_export`` for
  the V_xc grid), identity part dropped.

Coefficient convention: RAW dual-projector cprj ``<~p|psi>`` with NO
conjugation anywhere on the seam — the abinao abe4152 k-gauge lesson
(conjugating c equals the ``-k`` projection set and the Im-prescription
kernels are not invariant under it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_paw import (
    assemble_paw_exchange_data,
    build_abinit_paw_snapshot,
)
from TB2J.interfaces.abinit_paw_spinor import (
    build_abinit_paw_spinor_data,
    gen_exchange_abinit_paw_spinor,
    parse_pawprt_dij_spinor,
    pauli_delta_blocks_from_components,
    save_spinor_projected_data,
)
from TB2J.interfaces.gpaw_projector import (
    _R_grid,
    compute_projector_exchange_jdict,
)
from TB2J.interfaces.gpaw_spinor_projector import (
    compute_spinor_projector_exchange,
)
from TB2J.projector_green import ProjectorGreenData

# ---------------------------------------------------------------------------
# Shared synthetic layout: 2 atoms x 2 projector channels, 2 k-points,
# 4 bands per spin — same shape family as the collinear test fixture.
# ---------------------------------------------------------------------------

NATOM = 2
NPROJ_PER_ATOM = 2
NPROJ_TOTAL = NATOM * NPROJ_PER_ATOM
NKPT = 2
NBAND = 4


def _synthetic_collinear_payload():
    """Collinear nsppol=2 cprj: coefficients (nproj_total, nband) per k/spin."""
    rng = np.random.default_rng(42)
    rotation = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    cprj = np.zeros((2, NKPT, NPROJ_TOTAL, NBAND), dtype=complex)
    for spin in range(2):
        for ik in range(NKPT):
            base = np.zeros((NATOM, NPROJ_PER_ATOM, NBAND), dtype=complex)
            for atom in range(NATOM):
                for band in range(NBAND):
                    base[atom, :, band] = rotation[band % NPROJ_PER_ATOM, :]
            base += 0.05 * rng.standard_normal(base.shape)
            cprj[spin, ik] = base.reshape(NPROJ_TOTAL, NBAND)
    eigenvalues = np.array(
        [
            [[-2.0, -1.0, 1.0, 2.0], [-1.5, -0.5, 0.5, 1.5]],
            [[-1.8, -0.8, 1.2, 2.2], [-1.3, -0.3, 0.7, 1.7]],
        ]
    )
    delta_ij = {
        0: np.array([[0.50, 0.05], [0.05, 0.30]]),
        1: np.array([[0.40, 0.03], [0.03, 0.20]]),
    }
    return cprj, eigenvalues, delta_ij


def _synthetic_spinor_operator(delta_ij):
    """Delta(s) = [[D, 0], [0, -D]] per site: the collinear Delta x sigma_z."""
    operator = np.zeros((NATOM, NPROJ_PER_ATOM, NPROJ_PER_ATOM, 2, 2), dtype=complex)
    for atom, delta in delta_ij.items():
        operator[atom, :, :, 0, 0] = delta
        operator[atom, :, :, 1, 1] = -delta
    return operator


# ---------------------------------------------------------------------------
# nspden=4 pawprt log parser
# ---------------------------------------------------------------------------


def _format_matrix(matrix, fmt="{:9.5f}"):
    return ["".join(f" {fmt.format(v)}" for v in row) for row in matrix]


def _write_spinor_pawprt_log(
    path: Path,
    components_by_atom: dict[int, dict[str, np.ndarray]],
    unit: str = "hartree",
) -> Path:
    """Write a pawprt Dij block in the nspden=4 (ndij=4) print layout."""
    lines = [f" Total pseudopotential strength Dij ({unit}):"]
    order = ("up-up", "dwn-dwn", "up-dwn", "dwn-up")
    for atom in sorted(components_by_atom):
        for name in order:
            matrix = np.asarray(components_by_atom[atom][name])
            lines.append(f" Atom #{atom + 1:3d} - Component {name}")
            lines.append("    === REAL PART:")
            lines.extend(_format_matrix(matrix.real))
            lines.append("    === IMAGINARY PART:")
            lines.extend(_format_matrix(matrix.imag))
    lines.append(" ...(more output)...")
    path.write_text("\n".join(lines))
    return path


def _reference_components(delta_matrices):
    """Ground-truth four-component Dij set for collinear Delta matrices.

    Delta = D1 - D2 per projector element with D2 = 0 and a GENERAL
    (non-symmetric) complex up-dwn block, packed with the m_pawdij spinor
    Hermiticity ``D^{ss'}_{ij} = conjg(D^{s's}_{ji})``:
    ``dwn-up = conj(up-dwn).T`` — NOT ``conj(up-dwn)`` elementwise (the two
    coincide only for a symmetric up-dwn block, which would make fixtures
    unable to detect a transposed/conjugated packing swap).
    """
    out = {}
    for atom, delta in enumerate(delta_matrices):
        comp = np.asarray(delta, dtype=complex)
        d3 = np.full_like(comp, 0.02 + 0.01j)
        if comp.shape[0] > 1:
            d3[0, 1] = -0.004 + 0.003j
            d3[1, 0] = 0.005 - 0.002j
        out[atom] = {
            "up-up": comp,
            "dwn-dwn": np.zeros_like(comp),
            "up-dwn": d3,
            "dwn-up": np.conj(d3.T),
        }
    return out


class TestSpinorDijLogParser:
    def test_parses_four_component_blocks(self, tmp_path):
        delta = [
            np.array([[0.5, 0.05], [0.05, 0.3]]),
            np.array([[0.4, 0.03], [0.03, 0.2]]),
        ]
        reference = _reference_components(delta)
        log = _write_spinor_pawprt_log(tmp_path / "run.abo", reference, unit="hartree")

        parsed, unit = parse_pawprt_dij_spinor(log)

        assert unit == "hartree"
        assert sorted(parsed) == [0, 1]
        for atom in (0, 1):
            for name in ("up-up", "dwn-dwn", "up-dwn", "dwn-up"):
                np.testing.assert_allclose(
                    parsed[atom][name], reference[atom][name], atol=2e-5
                )

    def test_auto_unit_prefers_ev_section(self, tmp_path):
        delta = [np.array([[0.5, 0.05], [0.05, 0.3]])]
        reference = _reference_components(delta)
        hartree = _write_spinor_pawprt_log(
            tmp_path / "ha.abo", reference, unit="hartree"
        ).read_text()
        ev_reference = {
            atom: {name: mat * 27.211386245988 for name, mat in comp.items()}
            for atom, comp in reference.items()
        }
        ev = _write_spinor_pawprt_log(
            tmp_path / "ev.abo", ev_reference, unit="eV"
        ).read_text()
        log = tmp_path / "both.abo"
        log.write_text(hartree + "\n" + ev)

        parsed, unit = parse_pawprt_dij_spinor(log)

        assert unit == "eV"
        np.testing.assert_allclose(
            parsed[0]["up-up"], ev_reference[0]["up-up"], atol=2e-5
        )

    def test_dwn_up_packing_is_conj_transpose_of_up_dwn(self, tmp_path):
        """Pin the m_pawdij spinor packing D^{ss'}_ij = conjg(D^{s's}_ji).

        The print fills each cross-spin block's lower triangle from conjg of
        the partner component (idij_sym = 7 - idij, pawio_print_ij
        opt_sym=2), so the parsed full matrices must satisfy
        ``dwn-up = conj(up-dwn).T``.  The previously pinned elementwise
        ``conj(up-dwn)`` relation is WRONG — it coincides with the transpose
        only for symmetric up-dwn blocks, and the real Ni fr_spinor log
        (antisymmetric up-dwn) fails it with max deviation 2*max|D3|.
        """
        reference = _reference_components([np.eye(2) * 0.5])
        log = _write_spinor_pawprt_log(tmp_path / "run.abo", reference)
        parsed, unit = parse_pawprt_dij_spinor(log)
        # The fixture packs the true (transposed) relation: accepted.
        blocks, _ = pauli_delta_blocks_from_components(parsed, unit)
        np.testing.assert_allclose(
            blocks[0, :, :, 1, 0], 2 * np.conj(parsed[0]["up-dwn"].T)
        )

        # Elementwise conj — the previously pinned, wrong relation — must be
        # rejected: the fixture's up-dwn block is non-symmetric, so
        # conj(up-dwn) != conj(up-dwn).T.
        bad = dict(parsed[0])
        bad["dwn-up"] = np.conj(parsed[0]["up-dwn"])
        with pytest.raises(ValueError, match=r"dwn-up block is not conj\(up-dwn\)\.T"):
            pauli_delta_blocks_from_components({0: bad}, unit)

        # A dwn-up block with the transposed partner on the WRONG side
        # (missing conjugation) is equally rejected.
        bad2 = dict(parsed[0])
        bad2["dwn-up"] = parsed[0]["up-dwn"].T
        with pytest.raises(ValueError, match=r"dwn-up block is not conj\(up-dwn\)\.T"):
            pauli_delta_blocks_from_components({0: bad2}, unit)

    def test_rejects_collinear_log(self, tmp_path):
        log = tmp_path / "colo.abo"
        log.write_text(
            " Total pseudopotential strength Dij (hartree):\n"
            " Atom #  1 - Spin component 1\n"
            "  0.50000  0.00000\n"
            "  0.00000  0.30000\n"
        )
        with pytest.raises(ValueError, match="no nspden=4 Dij block"):
            parse_pawprt_dij_spinor(log)

    def test_rejects_truncated_block(self, tmp_path):
        reference = _reference_components([np.eye(2) * 0.5])
        log = _write_spinor_pawprt_log(tmp_path / "trunc.abo", reference)
        lines = log.read_text().splitlines()
        # Drop the imaginary part of the up-dwn block.
        idx = next(
            i
            for i, line in enumerate(lines)
            if "IMAGINARY" in line
            and i > next(j for j, l in enumerate(lines) if "up-dwn" in l)
        )
        del lines[idx : idx + 2]
        log.write_text("\n".join(lines))
        with pytest.raises(ValueError):
            parse_pawprt_dij_spinor(log)


# ---------------------------------------------------------------------------
# Pauli decomposition (source-verified packing)
# ---------------------------------------------------------------------------


class TestPauliDeltaDecomposition:
    def test_delta_equals_two_b_sigma(self):
        d1 = np.array([[0.5, 0.05], [0.05, 0.3]], dtype=complex)
        d3 = np.full((2, 2), 0.02 + 0.01j)
        d3[0, 1] = -0.004 + 0.003j  # general (non-symmetric) up-dwn block
        d3[1, 0] = 0.005 - 0.002j
        comps = {
            0: {
                "up-up": d1,
                "dwn-dwn": np.zeros((2, 2), dtype=complex),
                "up-dwn": d3,
                "dwn-up": np.conj(d3.T),
            }
        }

        blocks, unit = pauli_delta_blocks_from_components(comps, unit="hartree")

        assert unit == "hartree"
        block = blocks[0]
        np.testing.assert_allclose(block[..., 0, 0], d1)
        np.testing.assert_allclose(block[..., 0, 1], 2 * d3)
        np.testing.assert_allclose(block[..., 1, 0], 2 * np.conj(d3.T))
        np.testing.assert_allclose(block[..., 1, 1], -d1)
        # Hermitian in (projector, spin) up to print-level noise.
        dense = block.transpose(2, 0, 3, 1).reshape(4, 4)
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)

    def test_accepts_hermitian_complex_onsite_blocks(self):
        """SOC prints carry antisymmetric imaginary parts in up-up/dwn-dwn.

        The blocks are Hermitian, NOT real (regression: the real Ni
        fr_spinor log has max |Im up-up| = 2.7e-2 hartree; the retired
        "must be real" check rejected such logs before the packing check).
        """
        rng = np.random.default_rng(3)

        def complex_hermitian(scale):
            re = rng.normal(size=(2, 2)) * scale
            im = rng.normal(size=(2, 2)) * scale
            return (re + re.T) + 1j * (im - im.T)

        d1 = complex_hermitian(0.5)
        d2 = complex_hermitian(0.3)
        d3 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        comps = {
            0: {
                "up-up": d1,
                "dwn-dwn": d2,
                "up-dwn": d3,
                "dwn-up": np.conj(d3.T),
            }
        }

        blocks, _ = pauli_delta_blocks_from_components(comps, unit="hartree")

        np.testing.assert_allclose(blocks[0, :, :, 0, 0], d1 - d2)
        np.testing.assert_allclose(blocks[0, :, :, 0, 1], 2 * d3)
        np.testing.assert_allclose(blocks[0, :, :, 1, 0], 2 * np.conj(d3.T))
        np.testing.assert_allclose(blocks[0, :, :, 1, 1], d2 - d1)

    def test_by_sigma_form_matches_component_form(self):
        """Delta = 2(Bx sx + By sy + Bz sz) per projector pair, with
        Bx = (D3+D4)/2, By = i(D3-D4)/2 (Hermitian under the packing;
        = (Re D3, -Im D3) when D3 is symmetric) and Bz = (D1-D2)/2."""
        rng = np.random.default_rng(7)

        def hermitian(scale):
            a = rng.normal(size=(3, 3)) * scale
            return a + a.T

        d1 = hermitian(1.0)
        d2 = hermitian(0.3)
        d3 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        d4 = np.conj(d3.T)
        comps = {
            0: {
                "up-up": d1.astype(complex),
                "dwn-dwn": d2.astype(complex),
                "up-dwn": d3,
                "dwn-up": d4,
            }
        }
        sigmas = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        bx = 0.5 * (d3 + d4)
        by = 0.5j * (d3 - d4)
        bz = 0.5 * (d1 - d2)
        np.testing.assert_allclose(bx, bx.conj().T, atol=1e-14)
        np.testing.assert_allclose(by, by.conj().T, atol=1e-14)
        expected = 2.0 * (
            bx[..., None, None] * sigmas[0]
            + by[..., None, None] * sigmas[1]
            + bz[..., None, None] * sigmas[2]
        )

        blocks, _ = pauli_delta_blocks_from_components(comps, unit="eV")
        np.testing.assert_allclose(blocks[0], expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Spinor snapshot builder + collinear identity (interleave trick)
# ---------------------------------------------------------------------------


def _build_spinor_data():
    cprj, eigenvalues, delta_ij = _synthetic_collinear_payload()
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kweights = np.array([0.5, 0.5])
    # Interleave: spinor band b is pure-up (collinear up band b), band
    # NBAND + b is pure-down.  Coefficients stay RAW (no conjugation).
    coefficients = np.zeros((1, NKPT, 2 * NBAND, 2, NPROJ_TOTAL), dtype=complex)
    coefficients[0, :, :NBAND, 0, :] = cprj[0].swapaxes(1, 2)  # (nk, nband, nproj)
    coefficients[0, :, NBAND:, 1, :] = cprj[1].swapaxes(1, 2)
    spin_eigenvalues = np.concatenate([eigenvalues[0], eigenvalues[1]], axis=1)[None]
    return build_abinit_paw_spinor_data(
        coefficients=coefficients,
        eigenvalues=spin_eigenvalues,
        kweights=kweights,
        kpoints=kpoints,
        efermi=0.0,
        spinor_operator=_synthetic_spinor_operator(delta_ij),
        site_nproj=np.full(NATOM, NPROJ_PER_ATOM, dtype=int),
        cell=2.5 * np.eye(3),
        positions=np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]),
        atomic_numbers=np.array([26, 26], dtype=int),
    )


class TestSpinorCollinearIdentity:
    def test_builds_valid_spinor_data(self):
        data = _build_spinor_data()
        assert isinstance(data, ProjectorGreenData)
        assert data.nspinor == 2
        assert data.coefficients.shape == (1, NKPT, 2 * NBAND, 2, NPROJ_TOTAL)
        data.validate(exchange_ready=True)

    def test_interleaved_spinor_reproduces_collinear_j_exactly(self):
        """Spinor kernel on interleaved collinear data == collinear kernel.

        Circularity pitfall (documented, by design of this test): both sides
        are collinear-derived, so this pins cross-kernel algebraic identity —
        NOT the k-gauge convention.  Only genuine nspinor=2 WFK data can
        validate the gauge (abinao abe4152); the convention itself is pinned
        by test_snapshot_coefficients_are_raw_cprj below.
        """
        coeff, eigenvalues, delta_ij = _synthetic_collinear_payload()
        collinear = assemble_paw_exchange_data(
            cprj_per_kpt=[[coeff[0, ik], coeff[1, ik]] for ik in range(NKPT)],
            delta_ij=delta_ij,
            eigenvalues=eigenvalues,
            kweights=np.array([0.5, 0.5]),
            kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            efermi=0.0,
            natom=NATOM,
            nproj_per_atom=NPROJ_PER_ATOM,
            delta_unit="eV",
        )
        collinear_jdict = compute_projector_exchange_jdict(
            collinear,
            Rpts=_R_grid(nmax=1),
            nz=8,
            smearing_eV=0.1,
            sites=[0, 1],
        )
        assert collinear_jdict

        spinor = _build_spinor_data()
        spinor_exchange = compute_spinor_projector_exchange(
            spinor,
            Rpts=_R_grid(nmax=1),
            nz=8,
            smearing_eV=0.1,
            sites=[0, 1],
        )
        spinor_jdict = {
            key: entry["J_leg"][0, 0] for key, entry in spinor_exchange.items()
        }
        assert set(spinor_jdict) == set(collinear_jdict)
        for key in collinear_jdict:
            np.testing.assert_allclose(
                spinor_jdict[key],
                collinear_jdict[key],
                rtol=1e-8,
                atol=1e-12,
                err_msg=f"spinor/collinear mismatch for {key}",
            )
            # Non-trivial comparison: the exchange must not be identically 0.
            assert abs(collinear_jdict[key]) > 1e-8


# ---------------------------------------------------------------------------
# Raw-convention pins (abe4152 k-gauge lesson)
# ---------------------------------------------------------------------------


class TestRawCoefficientConvention:
    def test_snapshot_coefficients_are_raw_cprj(self):
        cprj = np.array([[1 + 2j, 3 - 1j], [0.5 + 0.5j, -1 + 0j]])
        layout = _fake_layout(1)
        snapshot = build_abinit_paw_snapshot(
            cprj_per_kpt=[[cprj]],
            delta_ij={0: np.eye(2)},
            eigenvalues=np.array([[[-1.0, -0.5]]]),
            kweights=np.array([1.0]),
            kpoints=np.zeros((1, 3)),
            efermi=0.0,
            site_layout=layout,
            cell=np.eye(3),
            positions=np.zeros((1, 3)),
            atomic_numbers=np.array([26]),
            delta_unit="hartree",
            provenance={},
        )
        np.testing.assert_allclose(
            snapshot.coefficients[0, 0], cprj.T, err_msg="cprj must be stored raw"
        )

    def test_conjugated_storage_leaves_collinear_contour_j_invariant(self):
        """Collinear contour J is EXACTLY invariant under global conj(c).

        Why this matters (documented, abe4152 lesson): the collinear pairing
        Tr[D_i g_up,ij(R) D_j g_dn,ji(-R)] carries opposite phase arguments,
        so conjugating the coefficient set conjugates the trace at E* and the
        Im-prescription contour integral reproduces itself identically.  This
        invariance is why the TB2J-side conjugation (commit 42f4dfc) went
        unnoticed — and it is NOT a license to conjugate: the spinor kernel
        on genuine nspinor=2 WFK data is NOT invariant (abinao abe4152:
        conjugated cprj gave bond-asymmetric 58.5/-0.35 meV garbage vs
        +1.518 meV raw on the CrI3 soc_off WFK), and cross-kernel consistency
        (the identity test above) requires the same raw convention on both
        sides of the seam.
        """
        coeff, eigenvalues, delta_ij = _synthetic_collinear_payload()
        common = dict(
            delta_ij=delta_ij,
            eigenvalues=eigenvalues,
            kweights=np.array([0.5, 0.5]),
            kpoints=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            efermi=0.0,
            natom=NATOM,
            nproj_per_atom=NPROJ_PER_ATOM,
            delta_unit="eV",
        )
        raw = assemble_paw_exchange_data(
            cprj_per_kpt=[[coeff[0, ik], coeff[1, ik]] for ik in range(NKPT)],
            **common,
        )
        conjugated = assemble_paw_exchange_data(
            cprj_per_kpt=[
                [coeff[0, ik].conj(), coeff[1, ik].conj()] for ik in range(NKPT)
            ],
            **common,
        )
        raw_j = compute_projector_exchange_jdict(
            raw, Rpts=_R_grid(nmax=1), nz=8, smearing_eV=0.1, sites=[0, 1]
        )
        conj_j = compute_projector_exchange_jdict(
            conjugated,
            Rpts=_R_grid(nmax=1),
            nz=8,
            smearing_eV=0.1,
            sites=[0, 1],
        )
        assert set(raw_j) == set(conj_j)
        for key in raw_j:
            np.testing.assert_allclose(conj_j[key], raw_j[key], rtol=1e-10, atol=1e-12)
            assert abs(raw_j[key]) > 1e-8


def _fake_layout(natom):
    from TB2J.paw_projector import (
        PawProjectorChannel,
        PawSiteLayout,
    )

    layouts = []
    for site in range(natom):
        channels = tuple(
            PawProjectorChannel(l=0, m=0, radial=radial, label=f"n0l0m0r{radial}")
            for radial in range(NPROJ_PER_ATOM)
        )
        layouts.append(
            PawSiteLayout(
                source_site=site,
                species="Fe",
                atomic_number=26,
                projector_slice=slice(
                    site * NPROJ_PER_ATOM, (site + 1) * NPROJ_PER_ATOM
                ),
                channels=channels,
                setup_hash="0" * 64,
            )
        )
    return tuple(layouts)


# ---------------------------------------------------------------------------
# End-to-end gen path on synthetic spinor projected data + nspden=4 log
# ---------------------------------------------------------------------------


class TestGenExchangeAbinitPawSpinor:
    def _write_projected(self, tmp_path):
        cprj, eigenvalues, _ = _synthetic_collinear_payload()
        coefficients = np.zeros((1, NKPT, 2 * NBAND, 2, NPROJ_TOTAL), dtype=complex)
        coefficients[0, :, :NBAND, 0, :] = cprj[0].swapaxes(1, 2)
        coefficients[0, :, NBAND:, 1, :] = cprj[1].swapaxes(1, 2)
        spin_eigenvalues = np.concatenate([eigenvalues[0], eigenvalues[1]], axis=1)
        return save_spinor_projected_data(
            tmp_path / "spinor_projected.pkl",
            coefficients=coefficients,
            eigenvalues=spin_eigenvalues,
            kweights=np.array([0.5, 0.5]),
            kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            efermi=0.0,
            site_nproj=np.full(NATOM, NPROJ_PER_ATOM, dtype=int),
            cell=2.5 * np.eye(3),
            positions=np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]),
            atomic_numbers=np.array([26, 26], dtype=int),
        )

    def test_end_to_end_with_log(self, tmp_path):
        _, _, delta_ij = _synthetic_collinear_payload()
        # nspden=4 Dij log whose Pauli Delta reproduces the collinear fixture:
        # D1 = delta, D2 = 0, D3 = 0 (then Delta_00 = delta, Delta_01 = 0).
        reference = _reference_components(list(delta_ij.values()))
        for atom in reference:
            reference[atom]["up-dwn"] = np.zeros_like(reference[atom]["up-dwn"])
            reference[atom]["dwn-up"] = np.zeros_like(reference[atom]["dwn-up"])
        log = _write_spinor_pawprt_log(tmp_path / "run.abo", reference, unit="hartree")
        projected = self._write_projected(tmp_path)

        with pytest.raises(
            ValueError, match="three independent x/y/z reference datasets"
        ):
            gen_exchange_abinit_paw_spinor(
                projected_data_path=str(projected),
                log_path=str(log),
                output_path=str(tmp_path / "out"),
                nz=8,
                smearing_eV=0.1,
                index_magnetic_atoms=[0, 1],
            )
        assert not (tmp_path / "out" / "TB2J.pickle").exists()

    def test_requires_data_source(self, tmp_path):
        with pytest.raises(ValueError, match="projected_data_path or wfk_path"):
            gen_exchange_abinit_paw_spinor(log_path=str(tmp_path / "x.abo"))

    def test_requires_delta_source(self, tmp_path):
        projected = self._write_projected(tmp_path)
        with pytest.raises(ValueError, match="log_path or delta_ij"):
            gen_exchange_abinit_paw_spinor(
                projected_data_path=str(projected),
                output_path=str(tmp_path / "out"),
            )
