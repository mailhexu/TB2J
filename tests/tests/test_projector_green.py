import numpy as np

from TB2J.projector_green import ProjectorGreen, ProjectorGreenData


def test_green_uses_bra_ket_coefficient_order():
    coefficient = np.array([[1.0 + 2.0j, 3.0 - 4.0j]])
    data = ProjectorGreenData(
        kpoints=np.zeros((1, 3)),
        weights=np.ones(1),
        eigenvalues=np.zeros((1, 1, 1)),
        coefficients=coefficient[None, None],
        efermi=0.0,
        projector_site=np.array([0, 1]),
        projector_atom=np.array([0, 1]),
    )

    got = ProjectorGreen(data).get_Gk(0, 1.0j)
    expected = np.outer(coefficient[0].conj(), coefficient[0]) / 1.0j
    np.testing.assert_allclose(got, expected)
