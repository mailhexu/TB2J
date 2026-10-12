"""Recorded SIESTA spin-spiral pencils and independent producer oracles.

MathDoc: docs/sympy/siesta_spiral_pencil.py [A, C, D]. Geometry and
momenta are Cartesian Bohr and inverse Bohr. sisl's orbital transpose
and spin-component conversion define the imported sparse orientation;
its HSX Fermi shift is undone using the HSX header, not rounded EIG data.
The entire stored pattern is retained, including zero-overlap entries.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

from .siesta_spiral import BUNDLE_VERSION, SpiralBundle, SpiralImporter

KB_EV = 8.617333262e-5
EIGENVALUE_TOL_EV = 1e-5
ELECTRON_COUNT_TOL = 1e-4
HERMITICITY_TOL = 1e-8
FIXTURE_VERSION = "1.0"


class OracleMismatch(RuntimeError):
    """The reference or imported pencil cannot certify producer agreement."""


@dataclass(frozen=True)
class OracleReport:
    passed: bool
    detail: str
    max_dev: float = 0.0
    deviation: float = 0.0


def load_fixture(path):
    """Read a versioned, non-pickled bundle and its recorded reference."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["fixture_version"]) != FIXTURE_VERSION:
            raise OracleMismatch("unsupported spiral fixture version")
        reference = {key: data[key].copy() for key in ("kp", "kp_weight", "eig")}
        for key in ("ef", "electron_count", "temperature"):
            reference[key] = float(data[key])
        sparse, overlap = {}, {}
        mu, nu, isc, h, s = (data[key] for key in ("mu", "nu", "isc", "h", "s"))
        if not (len(mu) == len(nu) == len(isc) == len(h) == len(s)):
            raise OracleMismatch("inconsistent fixture sparse array lengths")
        for idx in range(len(mu)):
            R = tuple(int(v) for v in data["sc_off"][isc[idx]])
            key = (int(mu[idx]), int(nu[idx]), R)
            if key in sparse:
                raise OracleMismatch(f"duplicate fixture sparse key {key}")
            sparse[key] = tuple(float(v) for v in h[idx])
            overlap[key] = float(s[idx])
        bundle = SpiralBundle(
            version=str(data["bundle_version"]),
            norb=int(data["norb"]),
            ncell=len(data["sc_off"]),
            q=tuple(float(v) for v in data["q_cart"]),
            spin_packing=str(data["spin_packing"]),
            phase_convention=str(data["phase_convention"]),
            hermitization=str(data["hermitization"]),
            fermi_shift=str(data["fermi_shift"]),
            pitch_sign=str(data["pitch_sign"]),
            sparse=sparse,
            overlap=overlap,
            metadata={
                "source": str(data["source_label"]),
                "ef_ev": float(data["ef_hsx"]),
            },
            cell=tuple(map(tuple, data["cell"])),
            orbital_positions=tuple(map(tuple, data["orbital_positions"])),
        )
    bundle.validate()
    return bundle, reference


def _pencil_eigenvalues(importer, bundle, kvec):
    H, S = importer.assemble_pencil(bundle, kvec)
    for name, matrix in (("H", H), ("S", S)):
        if not np.isfinite(matrix).all():
            raise OracleMismatch(f"nonfinite {name} pencil")
        defect = float(np.max(np.abs(matrix - matrix.conj().T)))
        if defect > HERMITICITY_TOL:
            raise OracleMismatch(
                f"{name} pencil is not Hermitian (defect {defect:.3e})"
            )
    try:
        return eigh(H, S, eigvals_only=True)
    except np.linalg.LinAlgError as exc:
        raise OracleMismatch(
            f"generalized pencil cannot be diagonalized: {exc}"
        ) from exc


def _reference_kpoints(reference):
    kp = np.asarray(reference["kp"], dtype=float)
    if kp.ndim != 2 or kp.shape[1] != 3 or len(kp) == 0 or not np.isfinite(kp).all():
        raise OracleMismatch("reference requires nonempty finite Cartesian k-points")
    return kp


def _failure(detail, strict):
    if strict:
        raise OracleMismatch(detail)
    return OracleReport(False, detail, max_dev=np.inf, deviation=np.inf)


def run_eigenvalue_oracle(bundle, reference, *, tol=EIGENVALUE_TOL_EV, strict=False):
    """Compare every recorded finite band; never repair a corrupt pencil."""
    try:
        kp = _reference_kpoints(reference)
        eig = np.asarray(reference["eig"], dtype=float)
        if (
            eig.ndim != 2
            or len(eig) != len(kp)
            or np.isnan(eig).any()
            or np.isneginf(eig).any()
        ):
            raise OracleMismatch("inconsistent or nonfinite eigenvalue reference")
        if not np.isfinite(tol) or tol <= 0:
            raise OracleMismatch("eigenvalue tolerance must be finite and positive")
    except OracleMismatch as exc:
        return _failure(str(exc), strict)
    importer = SpiralImporter()
    max_dev, worst = 0.0, 0
    for ik, kvec in enumerate(kp):
        try:
            w = _pencil_eigenvalues(importer, bundle, kvec)
            row = eig[ik]
            finite = row[np.isfinite(row)]
            if not len(finite) or len(row) != len(w):
                raise OracleMismatch(
                    "missing reference bands or pencil dimension mismatch"
                )
            if not np.isfinite(row[: len(finite)]).all() or np.any(np.diff(finite) < 0):
                raise OracleMismatch(
                    "reference must contain sorted finite bands then uncomputed +Infinity"
                )
            dev = float(np.max(np.abs(w[: len(finite)] - finite)))
        except OracleMismatch as exc:
            return _failure(f"k-point {ik}: {exc}", strict)
        if dev > max_dev:
            max_dev, worst = dev, ik
    passed = max_dev <= tol
    detail = (
        f"eigenvalue oracle {'PASS' if passed else 'FAIL'}: max deviation "
        f"{max_dev:.3e} eV (tolerance {tol:.0e}) at k-point {worst}"
    )
    if strict and not passed:
        raise OracleMismatch(detail)
    return OracleReport(passed, detail, max_dev=max_dev)


def run_electron_count_oracle(
    bundle, reference, *, tol=ELECTRON_COUNT_TOL, strict=False
):
    """Spinor occupations have unit multiplicity; retain the producer's weights."""
    try:
        kp = _reference_kpoints(reference)
        weights = np.asarray(reference["kp_weight"], dtype=float)
        ef, temperature, expected = (
            float(reference[key]) for key in ("ef", "temperature", "electron_count")
        )
        if (
            weights.shape != (len(kp),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise OracleMismatch("invalid producer k-point weights (must sum to one)")
        if (
            not np.isfinite([ef, temperature, expected, tol]).all()
            or temperature <= 0
            or tol <= 0
        ):
            raise OracleMismatch("invalid occupation reference or tolerance")
    except OracleMismatch as exc:
        return _failure(str(exc), strict)
    total = 0.0
    importer = SpiralImporter()
    for ik, kvec in enumerate(kp):
        try:
            w = _pencil_eigenvalues(importer, bundle, kvec)
        except OracleMismatch as exc:
            return _failure(f"k-point {ik}: {exc}", strict)
        total += weights[ik] * np.sum(expit((ef - w) / (KB_EV * temperature)))
    deviation = float(total - expected)
    passed = abs(deviation) <= tol
    detail = (
        f"electron-count oracle {'PASS' if passed else 'FAIL'}: computed "
        f"{total:.9f}, recorded {expected:.9f} (deviation {deviation:+.3e}, tolerance {tol:.0e})"
    )
    if strict and not passed:
        raise OracleMismatch(detail)
    return OracleReport(passed, detail, deviation=deviation)


def make_fixture_from_siesta(
    hsx_path, eig_path, kp_path, q_frac, electron_count, temperature, label
):
    """Reduce a version-1/2 noncollinear HSX without discarding any stored H.

    sisl's public reader converts CSC orientation and spin packing. Keep
    that orientation together with its reversed orbital displacements;
    do not independently undo the imaginary-component conversion.
    """
    import sisl
    from sisl.unit.siesta import unit_convert

    sile = sisl.get_sile(str(hsx_path))
    if sile.version not in (1, 2):
        raise OracleMismatch("HSX version lacks certified geometry and Fermi metadata")
    Hsp = sile.read_hamiltonian(dtype=np.float64)
    if not Hsp.spin.is_noncolinear:
        raise OracleMismatch(
            "fixture requires the four-component noncollinear HSX packing"
        )
    ef_hsx = float(sile.read_fermi_level())
    if not np.isfinite(ef_hsx):
        raise OracleMismatch("HSX Fermi level is missing or nonfinite")
    Hsp.shift(ef_hsx)
    bohr_ang = unit_convert("Bohr", "Ang")
    cell = Hsp.lattice.cell / bohr_ang
    g = Hsp.geometry
    orbital_positions = g.xyz[g.o2a(np.arange(g.no))] / bohr_ang
    no = Hsp.no
    sc_off = Hsp.lattice.sc_off
    dims = [Hsp.tocsr(d) for d in range(4)]
    Sx = sile.read_overlap().tocsr(0)
    eig, ef = read_siesta_eig(eig_path)
    kp, kp_weight = read_siesta_kp(kp_path)
    q_frac = np.asarray(q_frac, dtype=float)
    if q_frac.shape != (3,) or not np.isfinite(q_frac).all():
        raise OracleMismatch("pitch must be a finite reciprocal-lattice 3-vector")
    q_cart = q_frac @ (2 * np.pi * np.linalg.inv(cell).T)
    mu, nu, isc, h, s = [], [], [], [], []
    # All CSR structural entries, not nonzero(S): H can be nonzero where S is zero.
    for i in range(no):
        for col in dims[0].indices[dims[0].indptr[i] : dims[0].indptr[i + 1]]:
            mu.append(i)
            nu.append(int(col % no))
            isc.append(int(col // no))
            h.append([d[i, col] for d in dims])
            s.append(Sx[i, col])
    return {
        "fixture_version": FIXTURE_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "spin_packing": "H4=-ImH12",
        "phase_convention": "q_half_per_spin_sector",
        "hermitization": "none_hamiltonian_path",
        "fermi_shift": "none",
        "pitch_sign": "qphs_up_exp_minus",
        "norb": no,
        "mu": np.asarray(mu, dtype=np.int32),
        "nu": np.asarray(nu, dtype=np.int32),
        "isc": np.asarray(isc, dtype=np.int32),
        "sc_off": np.asarray(sc_off, dtype=np.int32),
        "h": np.asarray(h, dtype=np.float64),
        "s": np.asarray(s, dtype=np.float64),
        "cell": cell,
        "orbital_positions": orbital_positions,
        "q_frac": q_frac,
        "q_cart": q_cart,
        "ef": ef,
        "ef_hsx": ef_hsx,
        "eig": eig,
        "kp": kp,
        "kp_weight": kp_weight,
        "electron_count": electron_count,
        "temperature": temperature,
        "source_label": label,
    }


def read_siesta_eig(path):
    """Read noncollinear EIG: band count, spin=4, k count, indexed band rows."""
    with open(path) as fh:
        ef = float(fh.readline())
        header = fh.readline().split()
        neig, spin, nk = map(int, header)
        if spin != 4 or neig <= 0 or nk <= 0 or not np.isfinite(ef):
            raise OracleMismatch("invalid noncollinear EIG header")
        eig = np.full((nk, neig), np.inf)
        for ik in range(nk):
            head = fh.readline().split()
            if not head or int(head[0]) != ik + 1:
                raise OracleMismatch(f"EIG missing or reordered k-point {ik + 1}")
            vals = [float(t) for t in head[1:]]
            while len(vals) < neig:
                line = fh.readline()
                if not line:
                    raise OracleMismatch(
                        f"EIG truncated at k-point {ik + 1}: {len(vals)}/{neig} bands"
                    )
                vals.extend(float(t) for t in line.split())
            if len(vals) != neig:
                raise OracleMismatch(f"EIG extra bands at k-point {ik + 1}")
            eig[ik] = vals
    return eig, ef


def read_siesta_kp(path):
    """Read Cartesian inverse-Bohr KP and weights, without renormalization."""
    with open(path) as fh:
        nk = int(fh.readline())
        kp, w = [], []
        for ik in range(nk):
            tokens = fh.readline().split()
            if len(tokens) != 5 or int(tokens[0]) != ik + 1:
                raise OracleMismatch(f"KP missing or reordered k-point {ik + 1}")
            kp.append([float(v) for v in tokens[1:4]])
            w.append(float(tokens[4]))
    return np.asarray(kp), np.asarray(w)
