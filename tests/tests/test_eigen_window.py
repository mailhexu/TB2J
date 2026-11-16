"""The upper energy limit of the eigenstates kept in TBGreen is configurable.

TBGreen drops bands lying entirely above ``efermi + eigen_emax`` after the
diagonalization. The default (5.1 eV) keeps the previous behaviour;
``eigen_emax=np.inf`` keeps all states. The value is also exposed as the
``--eigen_emax`` command-line option.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from TB2J.exchange_params import add_exchange_args_to_parser, parser_argument_to_dict
from TB2J.green import TBGreen

_EVALS = np.linspace(-9.0, 12.0, 15)  # flat bands, Fermi energy 0


class _FlatBandModel:
    """Orthogonal model with k-independent eigenvalues ``_EVALS``."""

    is_orthogonal = True
    R2kfactor = 2j * np.pi
    norb = len(_EVALS)
    nbasis = len(_EVALS)
    nel = 6

    def HSE_k(self, kpt, convention=2):
        H = np.diag(_EVALS).astype(complex)
        return H, None, _EVALS.copy(), np.eye(self.nbasis, dtype=complex)


@pytest.mark.parametrize(
    "kwargs, nkept",
    [
        ({}, int(np.sum(_EVALS < 5.1))),
        ({"eigen_emax": 1.0}, int(np.sum(_EVALS < 1.0))),
        ({"eigen_emax": np.inf}, len(_EVALS)),
    ],
)
def test_eigen_emax(kwargs, nkept):
    G = TBGreen(tbmodel=_FlatBandModel(), kmesh=[2, 1, 1], efermi=0.0, **kwargs)
    assert G.evals.shape == (2, nkept)
    assert G.evecs.shape == (2, len(_EVALS), nkept)
    np.testing.assert_allclose(G.evals[0], _EVALS[:nkept])


def test_eigen_emax_command_line():
    parser = add_exchange_args_to_parser(argparse.ArgumentParser())
    assert parser_argument_to_dict(parser.parse_args([]))["eigen_emax"] == 5.1
    args = parser.parse_args(["--eigen_emax", "inf"])
    assert parser_argument_to_dict(args)["eigen_emax"] == np.inf
