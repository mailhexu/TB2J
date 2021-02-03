#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "0.4.1"

long_description = """TB2J is a Python package aimed to compute automatically the magnetic interactions (superexchange  and Dzyaloshinskii-Moriya) between atoms of magnetic crystals from DFT Hamiltonian based on Wannier functions or Linear combination of atomic orbitals. It uses the Green's function method and take the local rigid spin rotation as a perturbation. The package can take the output from Wannier90, which is interfaced with many density functional theory codes or from codes based on localised orbitals. A minimal user input is needed, which allows for an easily integration into a high-throughput workflows. """

setup(
    name='TB2J',
    version=__version__,
    description=
    'TB2J: First principle to Heisenberg exchange J using tight-binding Green function method',
    long_description=long_description,
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='BSD-2-clause',
    packages=find_packages(),
    scripts=[
        'scripts/wann2J.py', 'scripts/siesta2J.py', 'scripts/TB2J_rotate.py',
        'scripts/TB2J_merge.py', 'scripts/TB2J_magnon.py'
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'ase', 'progressbar', 'pathos'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
    ],
    python_requires='>=3.6',
)
