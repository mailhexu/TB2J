import numpy as np
from ase.io import read
from ase.atoms import Atoms
from collections import defaultdict
import math

def parse_xyz(fname):
    atoms=read(fname)
    symbols=atoms.get_chemical_symbols()
    pos=atoms.get_positions()
    wann_pos=[]
    atoms_pos=[]
    atoms_symbols=[]
    for s, x in zip(symbols, pos):
        if s=='X':
            wann_pos.append(x)
        else:
            atoms_symbols.append(s)
            atoms_pos.append(x)
    return np.array(wann_pos), atoms_symbols, np.array(atoms_pos)

def parse_ham(fname='wannier90_hr.dat',cutoff=None):
    """
    wannier90 hr file phaser.

    :param cutoff: the energy cutoff.  None | number | list (of Emin, Emax).
    """
    with open(fname,'r') as myfile:
        lines=myfile.readlines()
    n_wann=int(lines[1].strip())
    n_R=int(lines[2].strip())

    # The lines of degeneracy of each R point. 15 per line.
    nline=int(math.ceil(n_R/15.0))
    dlist=[]
    for i in range(3,3+nline):
        d=map(float,lines[i].strip().split())
        dlist+=d
    H_mnR=defaultdict(lambda: np.zeros((n_wann, n_wann), dtype=complex))
    for i in range(3+nline,3+nline+n_wann**2*n_R):
        t=lines[i].strip().split()
        R=tuple(map(int,t[:3]))
        m,n=map(int,t[3:5])
        m=m-1
        n=n-1
        H_real,H_imag=map(float, t[5:])
        val=H_real+1j*H_imag
        if (m==n and np.linalg.norm(R)<0.001): #onsite
            H_mnR[R][m, n]=val/2.0
        elif cutoff is not None:
            if abs(val)>cutoff:
                H_mnR[R][m, n]=val/2.0
        else:
           H_mnR[R][m, n]=val/2.0
    return n_wann, H_mnR

