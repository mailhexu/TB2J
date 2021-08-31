
TB2J_ref="""
% The implementation of TB2J:

@article{he2021tb2j,
title = {{TB2J}: A python package for computing magnetic interaction parameters},
journal = {Computer Physics Communications},
volume = {264},
pages = {107938},
year = {2021},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2021.107938},
url = {https://www.sciencedirect.com/science/article/pii/S0010465521000679},
author = {Xu He and Nicole Helbig and Matthieu J. Verstraete and Eric Bousquet},
}


"""


LKAG_ref="""
% The original idea of the magnetic force theorem method: 

@article{LKAG,
title = {Local spin density functional approach to the theory of exchange interactions in ferromagnetic metals and alloys},
journal = {Journal of Magnetism and Magnetic Materials},
volume = {67},
number = {1},
pages = {65-74},
year = {1987},
issn = {0304-8853},
doi = {https://doi.org/10.1016/0304-8853(87)90721-9},
url = {https://www.sciencedirect.com/science/article/pii/0304885387907219},
author = {A.I. Liechtenstein and M.I. Katsnelson and V.P. Antropov and V.A. Gubanov},
}


"""


Wannier_LKAG_ref="""
% Uisng Wannier function Hamiltonian:

@article{PhysRevB.91.224405,
  title = {Calculation of exchange constants of the Heisenberg model in plane-wave-based methods using the Green's function approach},
  author = {Korotin, Dm. M. and Mazurenko, V. V. and Anisimov, V. I. and Streltsov, S. V.},
  journal = {Phys. Rev. B},
  volume = {91},
  issue = {22},
  pages = {224405},
  numpages = {7},
  year = {2015},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.91.224405},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.91.224405}
}


"""


NonCollinear_ref="""
% Extension to DMI and anisotropic exchange:

@article{Antropov1997,
  title={Exchange interactions in magnets},
  author={Antropov, VP and Katsnelson, MI and Liechtenstein, AI},
  journal={Physica B: Condensed Matter},
  volume={237},
  pages={336--340},
  year={1997},
  publisher={Elsevier}
}


"""


downfolding_ref="""
% Downfolding method to include the ligand contribution:

@article{PhysRevB.103.104428,
  title = {Exchange interactions and magnetic force theorem},
  author = {Solovyev, I. V.},
  journal = {Phys. Rev. B},
  volume = {103},
  issue = {10},
  pages = {104428},
  numpages = {14},
  year = {2021},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.103.104428},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.103.104428}
}


"""


def print_bib():
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%     We recommend to cite the following references:       %") 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(TB2J_ref)

    print(LKAG_ref)
    print(Wannier_LKAG_ref)
    print(NonCollinear_ref)
    print(downfolding_ref)


if __name__=="__main__":
    print_bib()



