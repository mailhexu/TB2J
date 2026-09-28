"""VASP v7 spinor native reader tests (story 009)."""

import numpy as np
import pytest

from TB2J.interfaces.vasp_native import read_vasp_native_spinor
from TB2J.projector_green import ProjectorGreen, spinor_tangent_trace

MAGIC = 20260812


def _w(arr, dtype):
    return np.asanyarray(arr).ravel(order="F").astype(dtype).tobytes()


def _random_su2(rng):
    z = rng.normal(size=4)
    z /= np.linalg.norm(z)
    a, b, c, d = z
    return np.array(
        [[a + 1j * b, c + 1j * d], [-c + 1j * d, a - 1j * b]], dtype=complex
    )


def _make_v7_file(path, rng, nkpt_bz=4, nions=2, nproj_per_ion=2, nband=3):
    nproj = nproj_per_ion * nions
    nprod_stream = 2 * nproj
    nkpt_ibz = 2
    lmdim = nproj_per_ion
    lmax = 2  # two s channels (1+1 projectors)
    ntyp = 1
    lattice = np.eye(3) * 4.0
    posion = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]])
    ion_offset = np.array([0, nproj_per_ion])
    ion_nproj = np.array([nproj_per_ion] * nions)
    ion_ityp = np.array([1, 1])
    lmmax_typ = np.array([nproj_per_ion])
    lps_typ = np.array([[0], [0]])
    qtot_typ = np.zeros((lmax, lmax, ntyp))
    zval = np.array([8.0])
    labels = np.array([b"Fe"])

    # expansion plan: parents alternate, unit rssymop varying
    bz_kpoints = np.array(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]], dtype=float
    )
    parent = np.array([0, 0, 1, 1])
    source_spin = np.zeros((1, nkpt_bz), dtype=np.int32)
    conjugate = np.zeros((1, nkpt_bz), dtype=np.int32)
    rssymop = np.stack([_random_su2(rng) for _ in range(nkpt_bz)])
    actions = np.zeros((2 * nproj, 2 * nproj, 1, nkpt_bz), dtype=complex)
    for k in range(nkpt_bz):
        spin_part = rssymop[k].conj().T
        actions[:, :, 0, k] = np.kron(spin_part, np.eye(nproj))
    symop = np.ones(nkpt_bz, dtype=np.int32)
    spinflip = np.zeros(nkpt_bz, dtype=np.int32)

    vkpt_ibz = bz_kpoints[:nkpt_ibz]
    wtkpt = np.full(nkpt_ibz, 1.0 / nkpt_ibz)
    efermi = 0.3
    celtot = rng.normal(size=(nband, nkpt_ibz, 1))
    fertot = np.zeros((nband, nkpt_ibz, 1))
    fertot[:2] = 1.0
    cproj = np.zeros((nprod_stream, nband, nkpt_ibz, 1), dtype=complex)
    cproj[: 2 * nproj] = rng.normal(
        size=(2 * nproj, nband, nkpt_ibz, 1)
    ) + 1j * rng.normal(size=(2 * nproj, nband, nkpt_ibz, 1))
    # Raw VASP CDIJ stream: SPINOR representation (D_uu, D_ud, D_du, D_dd)
    # with US_FLIP's factor 1/2 (D_uu = (C00+Cz)/2, ...).
    cdij_raw = np.zeros((lmdim, lmdim, nions, 4), dtype=complex)
    cdij_converted = np.zeros((lmdim, lmdim, nions, 4), dtype=complex)
    for ion in range(nions):
        c00 = np.diag([0.2, 0.1]).astype(complex)
        cz = np.diag([0.5, 0.25]).astype(complex) * (1 if ion == 0 else -1)
        cdij_raw[:, :, ion, 0] = 0.5 * (c00 + cz)
        cdij_raw[:, :, ion, 3] = 0.5 * (c00 - cz)
        cdij_converted[:, :, ion, 0] = c00 + cz
        cdij_converted[:, :, ion, 3] = c00 - cz

    buf = b""

    def i4(x):
        return _w(x, "<i4")

    buf += i4(MAGIC) + i4(7) + i4(2)  # magic, version, nrspinors
    buf += i4(1) + i4(4) + i4(nkpt_ibz) + i4(nband)  # nspin, ncdij, nkpt, nband
    buf += i4(nprod_stream) + i4(nions) + i4(ntyp) + i4(lmdim) + i4(lmax)
    buf += _w(lattice, "<f8") + _w(posion.T, "<f8")
    buf += i4(ion_offset) + i4(ion_nproj) + i4(ion_ityp) + i4(lmmax_typ)
    buf += i4(lps_typ) + _w(qtot_typ, "<f8") + _w(zval, "<f8")
    buf += labels.tobytes()
    # plan
    buf += i4(1) + i4(nkpt_bz)
    buf += _w(bz_kpoints.T, "<f8")
    buf += i4(parent) + i4(source_spin) + i4(conjugate)
    buf += _w(actions, "<c16")
    buf += i4(symop) + i4(spinflip)
    buf += _w(np.transpose(rssymop, (1, 2, 0)), "<c16")
    # spectral
    buf += _w(vkpt_ibz.T, "<f8") + _w(wtkpt, "<f8") + _w(efermi, "<f8")
    buf += _w(celtot, "<f8") + _w(fertot, "<f8")
    buf += _w(cproj, "<c16")
    buf += _w(cdij_raw, "<c16") + _w(cdij_converted, "<c16")
    path.write_bytes(buf)

    expected = {}
    for k in range(nkpt_bz):
        src = cproj[: 2 * nproj, :, parent[k], 0]
        expected[k] = actions[:, :, 0, k] @ src
    return expected, posion, bz_kpoints


def test_v7_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    path = tmp_path / "tb2j_native.bin"
    expected, posion, bz_kpoints = _make_v7_file(path, rng)
    data = read_vasp_native_spinor(path)

    assert data.nspinor == 2
    assert data.coefficients.shape == (1, 4, 3, 2, 4)
    assert data.validate(exchange_ready=True)
    assert data.spinor_operator.shape == (2, 2, 2, 2, 2)

    # Bloch-corrected expectation
    for k in range(4):
        exp = expected[k]  # (2*nproj, nband)
        exp = exp[:4].T, exp[4:].T
        for s in range(2):
            for ion, (off, n) in enumerate([(0, 2), (2, 2)]):
                phase = np.exp(2j * np.pi * (bz_kpoints[k] @ posion[ion]))
                exp[s][:, off : off + n] *= phase
            np.testing.assert_allclose(
                data.coefficients[0, k, :, s, :], exp[s], atol=1e-10
            )

    # Hermitian spinor operator in the Delta convention: identity dropped,
    # zz elements are +/- the full spin splitting (D_uu - D_dd = cz).
    block = data.spinor_operator[0]
    np.testing.assert_allclose(block, block.transpose(1, 0, 3, 2).conj(), atol=1e-12)
    np.testing.assert_allclose(
        data.spinor_operator[0, :, :, 0, 0] + data.spinor_operator[0, :, :, 1, 1],
        np.zeros((2, 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.spinor_operator[0, :, :, 0, 0],
        np.diag([0.5, 0.25]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        data.spinor_operator[1, :, :, 0, 0],
        np.diag([-0.5, -0.25]),
        atol=1e-12,
    )


def test_v7_kernel_consumes(tmp_path):
    rng = np.random.default_rng(12)
    path = tmp_path / "tb2j_native.bin"
    _make_v7_file(path, rng)
    data = read_vasp_native_spinor(path)
    green = ProjectorGreen(data)
    Rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    result = spinor_tangent_trace(green, Rpts, energy=0.05)
    K = result["K_ijR"][((0, 0, 0), 0, 1)]
    np.testing.assert_allclose(K, result["K_ijR"][((0, 0, 0), 1, 0)].T, atol=1e-12)
    assert np.isfinite(K).all()


def test_v7_rejects_v6(tmp_path):
    # craft a v6 file by patching header: version 6 without nrspinors will
    # desync; simplest: write an empty-ish file with wrong magic
    path = tmp_path / "bad.bin"
    path.write_bytes(_w([MAGIC, 6], "<i4") + b"\0" * 64)
    with pytest.raises(ValueError, match="version 7"):
        read_vasp_native_spinor(path)
