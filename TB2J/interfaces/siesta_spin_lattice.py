"""SIESTA FC derivatives in a fixed reference-energy gauge."""

from pathlib import Path

import numpy as np
from HamiltonIO.siesta.dhsdr import read_dHSdR
from HamiltonIO.siesta.sisl_wrapper import SiestaHamiltonian, SislParser
from netCDF4 import Dataset
from scipy.linalg import eigh
from sisl.unit.siesta import unit_convert

from TB2J.spin_lattice_lcao import (
    LocalizedDisplacementDerivative,
    gen_exchange_spin_lattice_lcao,
)


class _CollinearSpinorModel(SiestaHamiltonian):
    """Exact spin-block eigenpairs without a redundant doubled eigensolve."""

    collinear_interleaved = True

    def HSE_k(self, kpt, convention=2):
        h, s = self.gen_ham(kpt, convention=convention)
        values = np.empty(self.nbasis)
        vectors = np.zeros_like(h, dtype=complex)
        for spin in (0, 1):
            sl = slice(spin, None, 2)
            values[sl], vectors[sl, sl] = eigh(h[sl, sl], s[sl, sl])
        return h, s, values, vectors


def load_siesta_spin_lattice_model(fdf_fname, reference_fermi=None):
    """Return an interleaved model and raw E_F (eV).

    sisl loads ``H - E_F(file) S``. If ``reference_fermi`` is provided,
    undo only the moving part of that shift: the model then contains
    ``H - E_F(reference) S``. Use the same value for displaced FD runs.
    The exchange driver's Fermi energy is zero in this fixed gauge.
    """
    parser = SislParser(fdf_fname=str(fdf_fname), ispin="merge")
    if not parser.ham.spin.is_colinear:
        raise ValueError("SIESTA spin-lattice currently requires collinear spin")
    with Dataset(parser.get_nc_path(parser.fdf)) as nc:
        fermi = float(nc.variables["Ef"][0]) * unit_convert("Ry", "eV")
    parsed = parser.get_model()
    model = _CollinearSpinorModel(
        HR=parsed.HR,
        SR=parsed.SR,
        Rlist=parsed.Rlist,
        nbasis=parsed.nbasis,
        orbs=parsed.orbs,
        atoms=parsed.atoms,
        nspin=1,
        nel=parsed.nel,
        orb_dict=parsed.orb_dict,
        orth=False,
    )
    if reference_fermi is not None:
        model.HR[...] += (fermi - reference_fermi) * model.SR
    return model, fermi


def gen_exchange_spin_lattice_siesta(
    fdf_fname,
    dhsdr_fname,
    atom=1,
    axis=1,
    response="periodic",
    output_path="TB2J_results",
    **params,
):
    """Compute one SIESTA FC displacement mode and serialize J and dJ/dtau.

    ``atom`` and ``axis`` follow SIESTA's 1-based indexing. ``periodic``
    sums translated home-cell vertices and matches a periodic displaced
    SCF finite difference. ``localized`` retains the home-cell convention
    and is not the same observable as that finite difference. Results
    use eV and eV/Angstrom in the SpinIO pickle (multiply by 1000 for meV).
    All bands are retained when differentiating the pencil inverse.
    """
    model, raw_fermi = load_siesta_spin_lattice_model(fdf_fname)
    blocks = list(read_dHSdR(dhsdr_fname, displacements=[(atom, axis)]))
    if len(blocks) != 1:
        raise ValueError(f"No unique dHSdR block for atom {atom}, axis {axis}")
    disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(
        blocks[0],
        atoms=model.atoms,
        reference_fermi=raw_fermi,
    )
    params.setdefault(
        "magnetic_elements", list(dict.fromkeys(model.atoms.get_chemical_symbols()))
    )
    params.setdefault("nz", 320)
    params.setdefault("emin", -12.0)
    params.setdefault("smearing", 0.0258519997864)
    if params.pop("efermi", 0.0) != 0.0:
        raise ValueError("SIESTA model is reference-Fermi-shifted; efermi must be zero")
    return gen_exchange_spin_lattice_lcao(
        tbmodel=model,
        atoms=model.atoms,
        disp=disp,
        basis=model.orbs,
        efermi=0.0,
        response=response,
        output_path=str(Path(output_path)),
        **params,
    )
