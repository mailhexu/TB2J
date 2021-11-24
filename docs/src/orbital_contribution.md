### Decompose the exchange into orbital contributions.

The exchange J between two atoms can be decomposed into the sum of all the pairs of orbitals. For two atoms  with m and n orbitals respectively, the decomposition can be written as m by n matrix. 

Here is an example:


Since version 0.6.4, the name of the orbitals are written in the Orbital contribution section. 
With the Wannier90 input, the names of these orbitals are not known, thus are named as orb_1, orb_2, etc. One can search the Wannier initial projectors to see what are these orbitals. 

Here is an example of cubic SrMnO3, 

```
=======================================================================================
Orbitals used in decomposition:
The name of the orbitals for the decomposition:
Mn1 : ['orb_1', 'orb_2', 'orb_3', 'orb_4', 'orb_5']

=======================================================================================
Exchange:
    i      j          R        J_iso(meV)          vector          distance(A)
----------------------------------------------------------------------------------------
   Mn1   Mn1   (  0,   0,   1) -7.1027   ( 0.000,  0.000,  3.810)  3.810
J_iso: -7.1027
Orbital contributions:
 [[ 3.184 -0.    -0.     0.    -0.   ]
 [-0.    -5.06   0.    -0.     0.   ]
 [-0.     0.    -5.06  -0.     0.   ]
 [ 0.    -0.    -0.    -0.121 -0.   ]
 [-0.     0.     0.     0.    -0.047]]

```

One can see that the contribution from the (orb_1, orb_1) pair is 3.184, and that from (orb_2, orb_2) is -5.06. Searching the wannier90 input, we can find that the orb1 and orb_2 are the dz2 and dxz orbitals, respectively. 


For Siesta input, the names are known, and printed in the "orbitals use in decomposition" section. They are not the exactly the name of the siesta basis (like 3dxyZ1). For example, with a double-zeta basis set, the a 3dxy orbital might be splitted into 3dxyZ1 and 3dxyZ2. The contribution of them are summed up to make it more concise. Often, the contribution from some orbitals are negligible. In the siesta2J.py command, it's possible to specify the orbitals to be considered in the decompostion. For example, if only the 3dxy contribution of Cr is needed, one can write

```
siesta2J.py --elements Fe_3d  ....

```

If both the 3d and 4s are to be considered, one can write:

```
siesta2J.py --elements Fe_3d_4s  ....

```

For example, the bcc Fe, when only the 3d orbitals are turned on, we get: 

```
==========================================================================================
Orbitals used in decomposition:
The name of the orbitals for the decomposition:
Fe1 : ('3dxy', '3dyz', '3dz2', '3dxz', '3dx2-y2')

==========================================================================================
Exchange:
    i      j          R        J_iso(meV)          vector          distance(A)
----------------------------------------------------------------------------------------
   Fe1   Fe1   ( -1,   0,   0) 17.6873   (-2.467,  0.000,  0.000)  2.467
J_iso: 17.6873
Orbital contributions:
 [[11.462 -0.     0.297 -0.     0.096]
 [-0.     3.69  -0.    -0.214 -0.   ]
 [-0.163 -0.     3.215 -0.    -3.262]
 [-0.     0.396 -0.    11.451 -0.   ]
 [-0.055  0.    -3.263  0.    -4.248]]

```
where we can find in the $x$ direction, the dxy and dxz  orbitals contribute mostly to the ferromagnetic interaction, whereas the 3dx2-y2 contribution is antiferromagnetic. 

Note that the option for selection of orbitals is not available in the Wannier90 interface, as the informations for labelling the orbitals are not included in the Wannier90 output (sometimes it is even not possible to do so as the Wannier functions are not necessarily atomic-orbital like). 


#### Decomposition in non-collinear mode
In collinear mode, the orbital decomposition of the isotropic exchange, DMI, and anisotropic exchange is turned off by default as there are a lot of terms and boast the size of the output files for large systems. 
It could be turned on with the --orb_decomposition option. The orbital decomposition will be written into another file called exchange_orb_decomposition.txt. In this file, the orbital decompositions for the isotropic exchange, the three DMI vector elements (Dx, Dy, Dz) and the nine anisotropic exchange matrix elements, will be outputed as m by n matrices for each element. 
