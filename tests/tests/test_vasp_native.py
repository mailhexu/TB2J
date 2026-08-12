"""Tests for the VASP native PAW export reader."""

import numpy as np
import pytest

from TB2J.interfaces.vasp_native import (
    HARTREE_TO_EV,
    VaspIbzExpansionPlan,
    _read_v6_expansion_plan,
    read_vasp_native,
)


def _write_test_native(path, version=4, lps_typ=None, nprod_padding=0):
    """Write a minimal VASP stream in the Fortran array order."""
    nspin = 2
    ncdij = 2
    nkpt = 8  # 2x2x2 mesh
    nband = 4
    nprod = 4 + nprod_padding  # 2 ions × 2 channels, potentially padded by VASP
    nions = 2
    ntyp = 1
    lmdim_max = 2

    kpts_3d = np.array(
        [
            [i * 0.5, j * 0.5, k * 0.5]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )
    wts = np.ones(nkpt) / nkpt

    with open(path, "wb") as f:

        def wi(value):
            f.write(np.array(value, dtype="<i4").tobytes())

        def wr(value):
            f.write(np.asarray(value, dtype="<f8").tobytes(order="F"))

        wi(20260812)
        wi(version)
        wi(nspin)
        wi(ncdij)
        wi(nkpt)
        wi(nband)
        wi(nprod)
        wi(nions)
        wi(ntyp)
        wi(lmdim_max)
        if version == 5:
            lps_typ = (0, 0) if lps_typ is None else tuple(lps_typ)
            wi(len(lps_typ))
        wr(np.eye(3) * 5.0)
        wr(np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T)
        wi([0, 2])
        wi([2, 2])
        wi([1, 1])
        wi([2])
        if version == 5:
            wi(lps_typ)
            qtot = np.zeros((len(lps_typ), len(lps_typ), 1), dtype=float)
            qtot[:2, :2, 0] = [[0.8, 0.1], [0.1, 0.6]]
            wr(qtot)
        wr([26.0])
        f.write(b"Fe")
        wr(kpts_3d.T)
        wr(wts)
        wr(5.0)

        eigenvalues = np.arange(nband * nkpt * nspin, dtype=float).reshape(
            nband, nkpt, nspin, order="F"
        )
        wr(eigenvalues)
        wr(np.full((nband, nkpt, nspin), 0.5, dtype=float))

        cproj = (
            np.arange(nprod * nband * nkpt * nspin, dtype=float)
            .reshape(nprod, nband, nkpt, nspin, order="F")
            .astype(np.complex128)
        )
        cproj += 1j * (cproj + 1)
        f.write(cproj.astype("<c16").tobytes(order="F"))

        cdij = np.zeros(
            (lmdim_max, lmdim_max, nions, ncdij), dtype=np.complex128, order="F"
        )
        for spin in range(ncdij):
            for ion in range(nions):
                cdij[0, 0, ion, spin] = 1.0 + spin * 0.5
                cdij[1, 1, ion, spin] = 2.0 + spin * 0.5
                cdij[0, 1, ion, spin] = 0.3
                cdij[1, 0, ion, spin] = 0.3
        f.write(cdij.astype("<c16").tobytes(order="F"))

    return path


def test_read_vasp_native_decodes_fortran_layout(tmp_path):
    snapshot = read_vasp_native(_write_test_native(tmp_path / "test.bin"))

    assert snapshot.kpoints.shape == (8, 3)
    assert np.allclose(snapshot.kpoints[1], [0.0, 0.0, 0.5])
    assert snapshot.eigenvalues.shape == (2, 8, 4)
    assert snapshot.eigenvalues[1, 7, 3] == 63.0
    assert snapshot.coefficients.shape == (2, 8, 4, 4)
    assert snapshot.coefficients.dtype == complex
    assert snapshot.population_metric_matrix is None
    assert snapshot.coefficients[1, 7, 3, 3] == pytest.approx(256 - 255j)
    assert len(snapshot.site_layout) == 2
    assert snapshot.atomic_numbers.tolist() == [26, 26]
    assert snapshot.operators.policy == "authoritative_total"
    assert snapshot.provenance["source_code"] == "vasp"


def test_read_vasp_native_preserves_site_spin_difference(tmp_path):
    snapshot = read_vasp_native(_write_test_native(tmp_path / "test.bin"))

    for operator in snapshot.operators.components[0].values:
        assert np.allclose(operator, np.eye(2) * (-0.5 / HARTREE_TO_EV))


def test_read_vasp_native_v5_builds_qtot_population_metric(tmp_path):
    snapshot = read_vasp_native(_write_test_native(tmp_path / "test.bin", version=5))

    expected = np.zeros((4, 4))
    expected[:2, :2] = [[0.8, 0.1], [0.1, 0.6]]
    expected[2:, 2:] = [[0.8, 0.1], [0.1, 0.6]]
    np.testing.assert_allclose(snapshot.population_metric_matrix, expected)
    assert snapshot.population_metric == "VASP PAW AE partial-wave overlap QTOT"
    assert [
        (channel.l, channel.radial) for channel in snapshot.site_layout[0].channels
    ] == [
        (0, 0),
        (0, 1),
    ]


def test_read_vasp_native_v5_discards_padded_cproj_rows(tmp_path):
    snapshot = read_vasp_native(
        _write_test_native(
            tmp_path / "test.bin",
            version=5,
            lps_typ=(0, 0, -1),
            nprod_padding=2,
        )
    )

    assert snapshot.coefficients.shape == (2, 8, 4, 4)
    assert [
        (layout.projector_slice.start, layout.projector_slice.stop)
        for layout in snapshot.site_layout
    ] == [(0, 2), (2, 4)]
    assert [
        (channel.l, channel.radial) for channel in snapshot.site_layout[0].channels
    ] == [
        (0, 0),
        (0, 1),
    ]
    expected_metric = np.kron(
        np.eye(2), np.array([[0.8, 0.1], [0.1, 0.6]], dtype=float)
    )
    np.testing.assert_allclose(snapshot.population_metric_matrix, expected_metric)
    for operator in snapshot.operators.components[0].values:
        np.testing.assert_allclose(operator, -0.5 * np.eye(2) / HARTREE_TO_EV)


def test_read_vasp_native_rejects_stream_shorter_than_ion_layout(tmp_path):
    with pytest.raises(ValueError, match="shorter than its ion projector layout"):
        read_vasp_native(
            _write_test_native(
                tmp_path / "test.bin",
                version=5,
                lps_typ=(0, 0, -1),
                nprod_padding=-1,
            )
        )


def test_read_vasp_native_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(np.array(999, dtype="<i4").tobytes())

    with pytest.raises(ValueError, match="invalid magic"):
        read_vasp_native(path)


# ---------------------------------------------------------------------------
# V6 IBZ expansion plan decoder tests
# ---------------------------------------------------------------------------


def _write_v6_expansion_section(
    f,
    nspin,
    nproj,
    nkpt_bz,
    nkpt_ibz,
    *,
    bad_kpoint_mode=False,
    bad_parent=False,
    bad_spin=False,
    conjugate_flags=None,
    spinflip_flags=None,
    action_factory=None,
):
    """Write a v6 IBZ expansion section to an open binary file."""

    def wi(value):
        f.write(np.array(value, dtype="<i4").tobytes())

    def wr(value):
        f.write(np.asarray(value, dtype="<f8").tobytes(order="F"))

    def wc(value):
        f.write(np.asarray(value, dtype=complex).astype("<c16").tobytes(order="F"))

    kpt_mode = 999 if bad_kpoint_mode else 1
    wi(kpt_mode)
    wi(nkpt_bz)

    bz_kpts = np.array(
        [
            [i * 0.5, j * 0.5, k * 0.5]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )[:nkpt_bz]
    wr(bz_kpts.T)

    if bad_parent:
        parent = np.full(nkpt_bz, nkpt_ibz + 5, dtype="<i4")
    else:
        parent = np.zeros(nkpt_bz, dtype="<i4")
    f.write(parent.tobytes())

    if bad_spin:
        src_spin = np.full((nspin, nkpt_bz), nspin + 5, dtype="<i4")
    else:
        src_spin = np.tile(np.arange(nspin, dtype="<i4")[:, None], (1, nkpt_bz))
    f.write(np.asarray(src_spin, dtype="<i4").tobytes(order="F"))

    if conjugate_flags is None:
        conjugate_flags = np.zeros((nspin, nkpt_bz), dtype="<i4")
    f.write(np.asarray(conjugate_flags, dtype="<i4").tobytes(order="F"))

    if action_factory is not None:
        actions = action_factory(nproj, nspin, nkpt_bz)
    else:
        actions = np.zeros(
            (nproj, nproj, nspin, nkpt_bz), dtype=np.complex128
        )
        for s in range(nspin):
            for k in range(nkpt_bz):
                actions[:, :, s, k] = np.eye(nproj, dtype=np.complex128)
    wc(actions)

    wi(np.zeros(nkpt_bz, dtype="<i4"))

    if spinflip_flags is None:
        spinflip_flags = np.zeros(nkpt_bz, dtype="<i4")
    f.write(spinflip_flags.tobytes())


def _write_test_native_v6(path, *, nprod_padding=0, **kwargs):
    """Write a minimal VASP v6 IBZ stream for testing."""
    nspin = 2
    ncdij = 2
    nkpt_ibz = 2
    nband = 4
    nprod_stream = 4 + nprod_padding
    nprod_physical = 4
    nions = 2
    ntyp = 1
    lmdim_max = 2
    lmax_max = 2
    nkpt_bz = 8

    ibz_kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    ibz_wts = np.array([0.5, 0.5])

    with open(path, "wb") as f:

        def wi(value):
            f.write(np.array(value, dtype="<i4").tobytes())

        def wr(value):
            f.write(np.asarray(value, dtype="<f8").tobytes(order="F"))

        def wc(value):
            f.write(np.asarray(value, dtype=complex).astype("<c16").tobytes(order="F"))

        wi(20260812)
        wi(6)
        wi(nspin)
        wi(ncdij)
        wi(nkpt_ibz)
        wi(nband)
        wi(nprod_stream)
        wi(nions)
        wi(ntyp)
        wi(lmdim_max)
        wi(lmax_max)
        wr(np.eye(3) * 5.0)
        wr(np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T)
        wi([0, 2])
        wi([2, 2])
        wi([1, 1])
        wi([2])
        wi([0, 0])
        qtot = np.zeros((lmax_max, lmax_max, ntyp), dtype=float)
        qtot[:2, :2, 0] = [[0.8, 0.1], [0.1, 0.6]]
        wr(qtot)
        wr([26.0])
        f.write(b"Fe")

        _write_v6_expansion_section(
            f, nspin, nprod_physical, nkpt_bz, nkpt_ibz, **kwargs
        )

        wr(ibz_kpts.T)
        wr(ibz_wts)
        wr(5.0)
        eigenvalues = np.arange(
            nband * nkpt_ibz * nspin, dtype=float
        ).reshape(nband, nkpt_ibz, nspin, order="F")
        wr(eigenvalues)
        wr(np.full((nband, nkpt_ibz, nspin), 0.5, dtype=float))

        cproj = (
            np.arange(nprod_stream * nband * nkpt_ibz * nspin, dtype=float)
            .reshape(nprod_stream, nband, nkpt_ibz, nspin, order="F")
            .astype(np.complex128)
        )
        cproj += 1j * (cproj + 1)
        wc(cproj)

        cdij = np.zeros(
            (lmdim_max, lmdim_max, nions, ncdij),
            dtype=np.complex128,
            order="F",
        )
        for spin in range(ncdij):
            for ion in range(nions):
                cdij[0, 0, ion, spin] = 1.0 + spin * 0.5
                cdij[1, 1, ion, spin] = 2.0 + spin * 0.5
                cdij[0, 1, ion, spin] = 0.3
                cdij[1, 0, ion, spin] = 0.3
        wc(cdij)

    return path


def test_v6_read_expansion_plan_identity(tmp_path):
    """Identity expansion plan decodes with exact documented shapes."""
    path = tmp_path / "plan.bin"
    nspin, nproj, nkpt_bz, nkpt_ibz = 2, 4, 8, 2
    with open(path, "wb") as f:
        _write_v6_expansion_section(f, nspin, nproj, nkpt_bz, nkpt_ibz)
    with open(path, "rb") as f:
        plan = _read_v6_expansion_plan(f, nspin, nproj, nkpt_ibz)

    assert isinstance(plan, VaspIbzExpansionPlan)
    assert plan.kpoint_storage_mode == 1
    np.testing.assert_allclose(
        plan.bz_kpoints[1], [0.0, 0.0, 0.5]
    )
    assert plan.source_spin.shape == (nspin, nkpt_bz)
    assert plan.source_spin.dtype == np.intp
    assert plan.conjugate.shape == (nspin, nkpt_bz)
    assert plan.conjugate.dtype == bool
    assert plan.projector_actions.shape == (nspin, nkpt_bz, nproj, nproj)
    assert plan.projector_actions.dtype == complex
    assert plan.symmetry_operation.shape == (nkpt_bz,)
    assert plan.spinflip.shape == (nkpt_bz,)
    assert plan.spinflip.dtype == bool
    np.testing.assert_array_equal(
        plan.source_spin, np.tile(np.arange(nspin)[:, None], (1, nkpt_bz))
    )
    assert not np.any(plan.conjugate)
    assert not np.any(plan.spinflip)
    for s in range(nspin):
        for k in range(nkpt_bz):
            np.testing.assert_allclose(
                plan.projector_actions[s, k], np.eye(nproj, dtype=complex)
            )


def test_v6_read_expansion_plan_conjugate_and_spinflip(tmp_path):
    """Conjugation and spinflip flags decode as booleans."""
    path = tmp_path / "plan.bin"
    nspin, nproj, nkpt_bz, nkpt_ibz = 2, 4, 8, 2
    conj_flags = np.zeros((nspin, nkpt_bz), dtype="<i4")
    conj_flags[1, 3] = 1
    conj_flags[0, 5] = 1
    spinflip_flags = np.zeros(nkpt_bz, dtype="<i4")
    spinflip_flags[2] = 1
    with open(path, "wb") as f:
        _write_v6_expansion_section(
            f,
            nspin,
            nproj,
            nkpt_bz,
            nkpt_ibz,
            conjugate_flags=conj_flags,
            spinflip_flags=spinflip_flags,
        )
    with open(path, "rb") as f:
        plan = _read_v6_expansion_plan(f, nspin, nproj, nkpt_ibz)

    assert plan.conjugate[1, 3]
    assert plan.conjugate[0, 5]
    assert not plan.conjugate[0, 0]
    assert plan.spinflip[2]
    assert not plan.spinflip[0]


def test_v6_read_expansion_plan_custom_action(tmp_path):
    """Non-identity action values survive the Fortran round-trip."""
    path = tmp_path / "plan.bin"
    nspin, nproj, nkpt_bz, nkpt_ibz = 2, 2, 4, 2

    def custom_actions(nproj, nspin, nkpt_bz):
        acts = np.zeros(
            (nproj, nproj, nspin, nkpt_bz), dtype=np.complex128
        )
        acts[0, 0, 0, 0] = 0.5 + 0.5j
        acts[1, 0, 0, 0] = -0.5 + 0.5j
        acts[0, 1, 0, 0] = 0.5 - 0.5j
        acts[1, 1, 0, 0] = 0.5 + 0.5j
        return acts

    with open(path, "wb") as f:
        _write_v6_expansion_section(
            f, nspin, nproj, nkpt_bz, nkpt_ibz, action_factory=custom_actions
        )
    with open(path, "rb") as f:
        plan = _read_v6_expansion_plan(f, nspin, nproj, nkpt_ibz)

    expected = np.array(
        [[0.5 + 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, 0.5 + 0.5j]]
    )
    np.testing.assert_allclose(plan.projector_actions[0, 0], expected)


def test_v6_read_expansion_plan_rejects_bad_kpoint_mode(tmp_path):
    path = tmp_path / "plan.bin"
    with open(path, "wb") as f:
        _write_v6_expansion_section(
            f, 2, 4, 8, 2, bad_kpoint_mode=True
        )
    with open(path, "rb") as f:
        with pytest.raises(ValueError, match="kpoint_storage_mode"):
            _read_v6_expansion_plan(f, 2, 4, 2)


def test_v6_read_expansion_plan_rejects_bad_parent(tmp_path):
    path = tmp_path / "plan.bin"
    with open(path, "wb") as f:
        _write_v6_expansion_section(
            f, 2, 4, 8, 2, bad_parent=True
        )
    with open(path, "rb") as f:
        with pytest.raises(ValueError, match="parent_ibz"):
            _read_v6_expansion_plan(f, 2, 4, 2)


def test_v6_read_expansion_plan_rejects_bad_spin(tmp_path):
    path = tmp_path / "plan.bin"
    with open(path, "wb") as f:
        _write_v6_expansion_section(
            f, 2, 4, 8, 2, bad_spin=True
        )
    with open(path, "rb") as f:
        with pytest.raises(ValueError, match="source_spin"):
            _read_v6_expansion_plan(f, 2, 4, 2)


def test_read_vasp_native_v6_raises_not_implemented(tmp_path):
    """v6 decoding succeeds; expansion deferred to Story 003."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        read_vasp_native(_write_test_native_v6(tmp_path / "test_v6.bin"))


def test_read_vasp_native_v6_padded_stream_uses_physical_proj_count(tmp_path):
    """NPROD padding consumed for CPROJ alignment but actions use ion layout."""
    path = _write_test_native_v6(tmp_path / "test_v6_pad.bin", nprod_padding=2)
    # If _read_v6_expansion_plan received the padded nprod (6) instead of
    # the physical nprod (4), the action reshape would fail or frombuffer
    # would read past the section boundary.  NotImplementedError proves the
    # plan decoded with physical dimensions.
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        read_vasp_native(path)


def test_read_vasp_native_v6_rejects_truncated_header(tmp_path):
    path = tmp_path / "trunc.bin"
    path.write_bytes(np.array([20260812, 6], dtype="<i4").tobytes())
    with pytest.raises(Exception):
        read_vasp_native(path)
