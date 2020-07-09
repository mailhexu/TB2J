import numpy as np
from ase.atoms import Atoms
from TB2J.utils import symbol_number
from collections import defaultdict
from scipy.linalg import eigh

class SislWrapper():
    def __init__(self, sisl_hamiltonian, spin=None):
        self.ham = sisl_hamiltonian
        # k2Rfactor : H(k) = \int_R H(R) * e^(k2Rfactor * k.R)
        self.R2kfactor = -2.0j*np.pi   # 
        if spin=='up':
            spin=0
        elif spin=='down':
            spin=1
        if spin not in [ None, 0, 1, 'merge']:
            raise ValueError("spin should be None/0/1, but is %s"%spin)
        self.spin=spin
        self.orbs=[]
        self.orb_dict=defaultdict(lambda:[])
        g=self.ham._geometry
        _atoms=self.ham._geometry._atoms
        atomic_numbers=[]
        self.positions=g.xyz
        self.cell=np.array(g.sc.cell)
        for ia, a in enumerate(_atoms):
            atomic_numbers.append(a.Z)
        self.atoms=Atoms(numbers=atomic_numbers, cell=self.cell, positions=self.positions)
        sdict=list(symbol_number(self.atoms).keys())
        if self.ham.spin.is_colinear and (self.spin in [0,1]):
            for ia, a in enumerate(_atoms):
                symnum=sdict[ia]
                try:
                    orb_names=[f"{symnum}|{x.name()}|up" for x in a.orbital]
                except:
                    orb_names=[f"{symnum}|{x.name()}|up" for x in a.orbitals]
                self.orbs+=orb_names
                self.orb_dict[ia]+=orb_names
            self.norb = len(self.orbs)
            self.nbasis=self.norb
        elif self.ham.spin.is_spinorbit or self.spin=='merge':
            for spin in {'up', 'down'}:
                for ia, a in enumerate(_atoms):
                    symnum=sdict[ia]
                    orb_names=[]
                    try:
                        for x in a.orbital:  # sisl < v0.10
                            orb_names.append(f"{symnum}|{x.name()}|{spin}")
                    except:
                        for x in a.orbitals:  # sisl >= v0.10
                            orb_names.append(f"{symnum}|{x.name()}|{spin}")
                    self.orbs+=orb_names
                    self.orb_dict[ia]+=orb_names
            #print(self.orb_dict)
            self.norb=len(self.orbs)//2
            #print(f"Norb: {self.norb}")
            self.nbasis= len(self.orbs)
        else:
            raise ValueError("The hamiltonian should be either spin-orbit or colinear")

    def view_info(self):
        print(self.orb_dict)
        print(self.atoms)

    def solve(self, k, convention=2):
        if convention==1:
            gauge='r'
        elif convention==2:
            gauge='R'
        if self.spin in [0,1]:
            evals, evecs = self.ham.eigh(k=k, spin=self.spin, eigvals_only=False, gauge=gauge)
        elif self.spin is None:
            evals, evecs = self.ham.eigh(k=k, eigvals_only=False, gauge=gauge)
            # reorder the first (basis) dimension so that it is 1up,1down, 2up, 2down...
            evecs=np.vstack([evecs[::2, :], evecs[1::2,:]])
        elif self.spin=='merge':
            evals0, evecs0 = self.ham.eigh(k=k, spin=0, eigvals_only=False, gauge=gauge)
            evals1, evecs1 = self.ham.eigh(k=k, spin=1, eigvals_only=False, gauge=gauge)
            evals=np.zeros(self.nbasis, dtype=float)
            evecs=np.zeros((self.nbasis, self.nbasis), dtype=complex)
            evals[:self.norb]=evals0
            evals[self.norb:]=evals1
            evecs[:self.norb, :self.norb]=evecs0
            evecs[self.norb:, self.norb:]=evecs1
        return evals, evecs


    def Hk(self, k, convention=2):
        if convention==1:
            gauge='r'
        elif convention==2:
            gauge='R'
        if self.spin is None:
            H=self.ham.Hk(k, gauge=gauge,format='dense')
            H=np.vstack([H[::2,:], H[1::2,:]])
            H=np.hstack([H[:,::2], H[:,1::2]])
        elif self.spin in [0,1]:
            H=self.ham.Hk(k, spin=self.spin, gauge=gauge, format='dense')
        elif self.spin == 'merge':
            H=np.zeros((self.nbasis, self.nbasis), dtype='complex')
            H[:self.norb, :self.norb]=self.ham.Hk(k, spin=0, gauge=gauge, format='dense')
            H[self.norb:, self.norb:]=self.ham.Hk(k, spin=1, gauge=gauge, format='dense')
        return H

    def eigen(self, k, convention=2):
        return self.solve(k)

    def gen_ham(self, k, convention=2):
        return self.Hk(k, convention=convention)

    def Sk(self, k, convention=2):
        if convention==1:
            gauge='r'
        elif convention==2:
            gauge='R'
        S0=self.ham.Sk(k, gauge='R', format='dense')
        #print(f"shape:{S0.shape}")
        #print(f"{self.nbasis}")
        if self.spin is None:
            S=np.vstack([S0[::2,:], S0[1::2,:]])
            S=np.hstack([S[:,::2], S[:,1::2]])
            #S=np.zeros((self.nbasis, self.nbasis), dtype='complex')
            #S[:self.norb,:self.norb]=S0
            #S[self.norb:, self.norb:]=S0
            #S=np.zeros((self.nbasis, self.nbasis), dtype='complex')
            #S[:self.nbasis//2,:self.norb//2]=S0
            #S[self.norb//2:, self.norb//2:]=S0
        elif self.spin in [0,1]:
            S=S0
        elif self.spin=='merge':
            S=np.zeros((self.nbasis, self.nbasis), dtype='complex')
            S[:self.norb,:self.norb]=S0
            S[self.norb:, self.norb:]=S0
        return S

    def solve_all(self, kpts, orth=True):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            if orth:
                S=self.Sk(k)
                Smh=Lowdin(S)
                H=self.gen_ham(k)
                Horth=Smh.T.conj() @ H @ Smh
                evalue, evec = eigh(Horth)
            else:
                evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs, dtype=complex, order='C')

    def HS_and_eigen(self, kpts, convention=2):
        evals = []
        evecs = []
        H=[]
        S=[]
        for ik, k in enumerate(kpts):
            Hk = self.Hk(k, convention=convention)
            Sk = self.Sk(k, convention=convention)
            H.append(self.Hk(k, convention=convention))
            S.append(self.Sk(k, convention=convention))
            evalue, evec = self.solve(k, convention=convention)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(H), np.array(S), np.array(evals, dtype=float), np.array(evecs, dtype=complex, order='C')


    def get_fermi_level(self):
        return 0.0



#def test():
#    fdf = sisl.get_sile('/home/hexu/projects/learn_siesta/SMO_Wannier/siesta.fdf')
#    H = fdf.read_hamiltonian(order='nc',dim=2)
#    print(H._spin._is_polarized)
#    print(H.__dict__.keys())
#    print(H._geometry.__dict__.keys())
#    print(H._geometry.xyz)
#    print(H._geometry.sc)
#    H._geometry.sc.cell
#    orb=H._geometry._atoms[0].orbital[0]
#    print(orb.name())
#    s=SislWrapper(H)
#    s.view_info()
#
#import sisl
#fdf = sisl.get_sile('/home/hexu/projects/learn_siesta/SMO_Wannier/siesta.fdf')
#H = fdf.read_hamiltonian(order='nc',dim=2)
#if __name__=='__main__':
#    test()
