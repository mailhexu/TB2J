import os
from collections import defaultdict

import numpy as np
from ase.dft.kpoints import monkhorst_pack

from TB2J.io_exchange import SpinIO
from TB2J.Jtensor import decompose_J_tensor
from TB2J.mryasov import mryasov_downfold


def ind_to_indn(ind, n=3):
    """index to index. e.g. 1, 2, 4 to 1*3, 1*3+1, 1*3+2, ... 4*3, 4*3+1, 4*3+2
    :param ind: the input indices
    :returns: ind3, the output indices
    """
    indn = np.repeat(ind, n) * n
    for i in range(n):
        indn[i::n] += i
    return indn


class JR_model:
    def __init__(self, JR, Rlist):
        self.JR = JR
        self.Rlist = Rlist
        self.nR = len(Rlist)

    def get_Jq(self, q):
        Jq = np.zeros(self.JR[0].shape, dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2.0j * np.pi * np.dot(q, R))
            Jq += self.JR[iR] * phase
        return Jq


class JDownfolder:
    def __init__(self, JR, Rlist, iM, iL, qmesh, iso_only=False):
        self.nxyz = 1 if iso_only else 3
        self.model = JR_model(JR, Rlist)
        self.Rlist = Rlist
        self.nR = len(Rlist)
        self.nM = len(iM)
        self.nL = len(iL)
        self.nsite = self.nM + self.nL
        self.iM = iM
        self.iL = iL
        self.qmesh = qmesh
        self.qpts = monkhorst_pack(qmesh)
        self.nqpt = len(self.qpts)
        self.nMn = self.nM * self.nxyz
        self.nLn = self.nL * self.nxyz
        self.iso_only = iso_only

    def get_JR(self):
        JR_downfolded = np.zeros((self.nR, self.nMn, self.nMn), dtype=float)
        Jq_downfolded = np.zeros((self.nqpt, self.nMn, self.nMn), dtype=complex)
        self.iMn = ind_to_indn(self.iM, n=self.nxyz)
        self.iLn = ind_to_indn(self.iL, n=self.nxyz)
        for iq, q in enumerate(self.qpts):
            Jq = self.model.get_Jq(q)
            Jq_downfolded[iq] = self.downfold_oneq(Jq)
            for iR, R in enumerate(self.Rlist):
                phase = np.exp(-2.0j * np.pi * np.dot(q, R))
                JR_downfolded[iR] += np.real(Jq_downfolded[iq] * phase / self.nqpt)
        return JR_downfolded, self.Rlist

    def downfold_oneq(self, J):
        JMM = J[np.ix_(self.iMn, self.iMn)]
        JLL = J[np.ix_(self.iLn, self.iLn)]
        JLM = J[np.ix_(self.iLn, self.iMn)]
        JML = J[np.ix_(self.iMn, self.iLn)]
        Jn = JMM - JML @ np.linalg.inv(JLL) @ JLM
        return Jn


class PWFDownfolder:
    def __init__(self, JR, Rlist, iM, iL, qmesh, atoms=None, iso_only=False, **kwargs):
        from lawaf.interfaces.magnon.magnon_downfolder import (
            MagnonDownfolder,
            MagnonWrapper,
        )

        model = MagnonWrapper(JR, Rlist, atoms, align_evecs=not iso_only)
        wann = MagnonDownfolder(model)
        # Downfold the band structure.
        index_basis = []
        self.nxyz = 1 if iso_only else 3
        for i in iM:
            index_basis += list(range(i * self.nxyz, i * self.nxyz + self.nxyz))
        params = dict(
            method="projected",
            # method="maxprojected",
            kmesh=qmesh,
            nwann=len(index_basis),
            selected_basis=index_basis,
            # anchors={(0, 0, 0): (-1, -2, -3, -4)},
            # anchors={(0, 0, 0): ()},
            # weight_func="Gauss",
            # weight_func_params=(-0.143, 0.04),
            use_proj=True,
            enhance_Amn=0.0,
        )
        params.update(kwargs)
        wann.set_parameters(**params)
        print("begin downfold")
        ewf = wann.downfold()
        # ewf.save_hr_pickle("downfolded_JR.pickle")

        # Plot the band structure.
        wann.plot_band_fitting(
            # kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
            #                   [0.5, 0.5, 0], [0, 0, 0],
            #                   [.5, .5, .5]]),
            # knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            cell=model.atoms.cell,
            supercell_matrix=None,
            npoints=100,
            efermi=None,
            erange=None,
            fullband_color="blue",
            downfolded_band_color="green",
            marker="o",
            ax=None,
            savefig="downfold_band.png",
            show=False,
        )
        self.JR_downfolded = ewf.HwannR
        self.Rlist = ewf.Rlist

    def get_JR(self):
        return self.JR_downfolded, self.Rlist


class JDownfolder_pickle:
    def __init__(
        self,
        inpath,
        metals,
        ligands,
        outpath,
        qmesh=[7, 7, 7],
        iso_only=False,
        method="lowdin",
        **kwargs,
    ):
        if method.lower() == "mryasov" and os.path.lexists(outpath):
            if not os.path.isdir(outpath) or os.listdir(outpath):
                raise FileExistsError(
                    "Mryasov output must be a new or empty directory."
                )
        self.exc = SpinIO.load_pickle(path=inpath, fname="TB2J.pickle")

        self.iso_only = (self.exc.dmi_ddict is None) or iso_only
        self.metals = metals
        self.ligands = ligands
        self.outpath = outpath
        self.method = method
        print("Using method:", self.method)

        # read atomic structure
        self.atoms = self.exc.atoms
        self.nspin = self.exc.nspin
        self.qmesh = qmesh
        self.natom = len(self.atoms)
        self.Rcut = None
        self._build_atom_index()
        if method.lower() == "mryasov":
            self._mryasov_to_exchange(**kwargs)
            return
        self._prepare_distance()
        Jd, Rlist = self._downfold(**kwargs)
        self._Jd_to_exchange(Jd, Rlist)

    def _mryasov_to_exchange(self, dressing=1.0):
        if set(self.metals) & set(self.ligands):
            raise ValueError("Metal and ligand element sets must be disjoint.")
        if not self.iM or not self.iL or min(self.iM + self.iL) < 0:
            raise ValueError(
                "Include all selected metals and ligands in the input exchange."
            )

        moments = np.asarray(self.exc.spinat, dtype=float)
        metal = moments[self.ind_mag_atoms]
        ligand = moments[[self.exc.ind_atoms[i] for i in self.iL]]
        metal_norm = np.linalg.norm(metal, axis=1)
        ligand_norm = np.linalg.norm(ligand, axis=1)
        if not np.isfinite(metal).all() or not np.isfinite(ligand).all():
            raise ValueError("Selected reference moments must be finite.")
        if np.any(metal_norm <= 1e-10) or np.any(ligand_norm <= 1e-10):
            raise ValueError(
                "Mryasov normalization requires nonzero metal and ligand moments."
            )
        axis = metal[0] / metal_norm[0]
        if not np.allclose(metal / metal_norm[:, None], axis, rtol=0, atol=1e-6):
            raise ValueError("Mryasov requires parallel metal reference moments.")
        if not np.allclose(
            np.cross(ligand / ligand_norm[:, None], axis), 0.0, rtol=0, atol=1e-6
        ):
            raise ValueError("Mryasov requires collinear ligand reference moments.")

        exchange = mryasov_downfold(
            self.exc.exchange_Jdict, self.iM, self.iL, dressing=dressing
        )
        if any(not np.isfinite(float(j) * 1e3) for j in exchange.values()):
            raise ValueError("Downfolded exchange is not representable in meV.")
        self._prepare_index_spin()
        remap = {old: new for new, old in enumerate(self.iM)}
        # TomASD writes one block per consecutive spin pair.
        self.Jdict = {
            (R, remap[i], remap[j]): exchange[R, i, j]
            for R, i, j in sorted(exchange, key=lambda key: (key[1], key[2], key[0]))
        }
        # SpinIO infers the saved spin inventory from bond keys.
        connected = {i for _, a, b in self.Jdict for i in (a, b)}
        for i in range(len(self.iM)):
            if i not in connected:
                self.Jdict[((0, 0, 0), i, i)] = 0.0
        positions = self.atoms.positions[self.ind_mag_atoms]
        cell = self.atoms.cell
        if not np.isfinite(self.atoms.positions).all() or not np.isfinite(cell).all():
            raise ValueError("Mryasov output requires a finite structure.")
        self.distance_dict = {}
        for key in self.Jdict:
            R, i, j = key
            vec = R @ cell + positions[j] - positions[i]
            distance = np.linalg.norm(vec)
            if not np.isfinite(distance):
                raise ValueError(f"Non-finite downfolded bond distance: {key}")
            self.distance_dict[key] = (vec, distance)
        self.iso_only = True
        self.DMIdict = self.Janidict = None
        self._write_exchange(
            colinear=not np.any(moments[:, :2]),
            description=f"Mryasov isotropic downfolding (lambda={dressing:g})",
        )

    def _build_atom_index(self):
        self.magnetic_elements = self.metals
        self.iM = []
        self.iL = []
        self.ind_mag_atoms = []
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                self.iM.append(self.exc.index_spin[i])
                self.ind_mag_atoms.append(i)
            elif sym in self.ligands:
                self.iL.append(self.exc.index_spin[i])

        self.nM = len(self.iM)
        self.nL = len(self.iL)
        self.nsite = self.nM + self.nL

    def _downfold(self, **kwargs):
        if self.iso_only:
            JR2 = self.exc.get_full_Jtensor_for_Rlist(
                order="ij", asr=True, DMI=False, Jani=False, SIA=False
            )
            # magmoms = self.exc.magmoms
            # magmoms of magnetic atoms (metal + ligand)
            # mmagmoms = magmoms[self.iM + self.iL]
            # sqrt_magmoms = np.sqrt(np.abs(mmagmoms))
            # magmoms_mat = np.outer(sqrt_magmoms, sqrt_magmoms)
            # JR2 = JR2 / magmoms_mat
        else:
            JR2 = self.exc.get_full_Jtensor_for_Rlist(order="i3j3_2D", asr=True)

        if self.method.lower() == "lowdin":
            d = JDownfolder(
                JR2,
                self.exc.Rlist,
                iM=self.iM,
                iL=self.iL,
                qmesh=self.qmesh,
                iso_only=self.iso_only,
            )
            Jd, Rlist = d.get_JR()
        else:
            d = PWFDownfolder(
                JR2,
                self.exc.Rlist,
                iM=self.iM,
                iL=self.iL,
                qmesh=self.qmesh,
                atoms=self.atoms,
                iso_only=self.iso_only,
                **kwargs,
            )
            Jd, Rlist = d.get_JR()
            # metal_sqrt_magmoms = np.sqrt(np.abs(self.exc.magmoms[self.iM]))
            # Jd = Jd * np.outer(metal_sqrt_magmoms, metal_sqrt_magmoms)
        return Jd, Rlist

    def _Jd_to_exchange(self, Jd, Rlist):
        self._prepare_distance()
        self._prepare_index_spin()
        self.Jdict = {}
        if self.iso_only:
            self.DMIdict = None
            self.Janidict = None
        else:
            self.DMIdict = {}
            self.Janidict = {}

        for iR, R in enumerate(Rlist):
            for i, ispin in enumerate(self.index_spin):
                for j, jspin in enumerate(self.index_spin):
                    if ispin >= 0 and jspin >= 0:
                        if not (tuple(R) == (0, 0, 0) and ispin == jspin):
                            if self.iso_only:
                                J = Jd[iR, ispin, jspin]
                                self.DMIdict = None
                                self.Janidict = None
                                self.Jdict[(tuple(R), ispin, jspin)] = J.real
                            else:
                                J33 = Jd[
                                    iR,
                                    ispin * 3 : ispin * 3 + 3,
                                    jspin * 3 : jspin * 3 + 3,
                                ]
                                J, DMI, Jani = decompose_J_tensor(J33)
                                self.Jdict[(tuple(R), ispin, jspin)] = J.real
                                self.DMIdict[(tuple(R), ispin, jspin)] = DMI.real
                                self.Janidict[(tuple(R), ispin, jspin)] = Jani.real

        self._write_exchange(colinear=self.iso_only)

    def _write_exchange(self, *, colinear, description=None):
        io = SpinIO(
            atoms=self.atoms,
            spinat=self.exc.spinat,
            charges=self.exc.charges,
            index_spin=self.index_spin,
            colinear=colinear,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.Jdict,
            dmi_ddict=self.DMIdict,
            Jani_dict=self.Janidict,
            description=description,
        )

        io.write_all(self.outpath)

    def _prepare_distance(self):
        self.distance_dict = {}
        self.short_Rlist = []
        self.R_ijatom_dict = defaultdict(lambda: [])
        ind_matoms = self.ind_mag_atoms
        for ispin, iatom in enumerate(ind_matoms):
            for jspin, jatom in enumerate(ind_matoms):
                for R in self.exc.Rlist:
                    pos_i = self.atoms.get_positions()[iatom]
                    pos_jR = self.atoms.get_positions()[jatom] + np.dot(
                        R, self.atoms.get_cell()
                    )
                    vec = pos_jR - pos_i
                    distance = np.sqrt(np.sum(vec**2))
                    if self.Rcut is None or distance < self.Rcut:
                        self.distance_dict[(tuple(R), ispin, jspin)] = (vec, distance)
                        self.R_ijatom_dict[tuple(R)].append((iatom, jatom))
        self.short_Rlist = list(self.R_ijatom_dict.keys())

    def _prepare_index_spin(self):
        # index_spin: index in spin hamiltonian of atom. starts from 1. -1 means not considered.
        ind_matoms = []
        self.index_spin = []
        ispin = 0
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            if sym in self.magnetic_elements:
                ind_matoms.append(i)
                self.index_spin.append(ispin)
                ispin += 1
            else:
                self.index_spin.append(-1)


def test():
    # pass
    # inpath = "/home/hexu/projects/NiCl2/vasp_inputs/TB2J_results"
    inpath = "/home/hexu/projects/TB2J_example/CrI3/TB2J_results"
    # inpath = "/home/hexu/projects/TB2J_projects/NiCl2/TB2J_NiCl/TB2J_results"
    _fname = os.path.join(inpath, "TB2J.pickle")
    _p = JDownfolder_pickle(
        inpath=inpath, metals=["Ni"], ligands=["Cl"], outpath="TB2J_results_downfolded"
    )


if __name__ == "__main__":
    test()
