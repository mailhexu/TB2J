import xml.etree.ElementTree as ET
from ase.atoms import Atoms
import numpy as np
from ase.units import Bohr, eV, J
from .constants import gyromagnetic_ratio
from ase.data import atomic_masses


class BaseSpinModelParser(object):
    """
    SpinModelParser: a general model for spin model file parser.
    """

    def __init__(self, fname):
        self._fname = fname
        self.atoms = None
        self.damping_factors = []
        self.gyro_ratios = []
        self.index_spin = []
        self.cell = None
        self.zions = []
        self.masses = []
        self.positions = []
        self.spinat = []
        self._exchange = {}
        self._dmi = {}
        self._bilinear = {}
        self._parse(fname)
        self.lattice = Atoms(
            positions=self.positions, masses=self.masses, cell=self.cell
        )

    def _parse(self, fname):
        raise NotImplementedError("parse function not implemented yet")

    def get_atoms(self):
        return self.atoms

    def _spin_property(self, prop):
        return [prop[i] for i in range(len(self.index_spin)) if self.index_spin[i] > 0]

    @property
    def spin_positions(self):
        return np.array(self._spin_property(self.positions), dtype="float")

    @property
    def spin_zions(self):
        return np.array(self._spin_property(self.zions), dtype="int")

    @property
    def spin_spinat(self):
        return np.array(self._spin_property(self.spinat), dtype="float")

    @property
    def spin_damping_factors(self):
        return np.array(self._spin_property(self.damping_factors), dtype="float")

    @property
    def spin_gyro_ratios(self):
        return np.array(self._spin_property(self.gyro_ratios), dtype="float")

    def get_index_spin(self):
        return self.index_spin

    def exchange(self, isotropic=True):
        if isotropic:
            iso_jdict = {}
            for key, val in self._exchange.items():
                iso_jdict[key] = val[0]
            return iso_jdict
        else:
            return self._exchange

    @property
    def dmi(self):
        return self._dmi

    @property
    def bilinear(self):
        return self._bilinear

    @property
    def has_exchange(self):
        return bool(len(self._exchange))

    @property
    def has_dmi(self):
        return bool(len(self._dmi))

    @property
    def has_bilinear(self):
        return bool(len(self._bilinear))


class SpinXmlWriter(object):
    def _write(self, model, fname):
        root = ET.Element("System_definition")
        unitcell = model.cell.reshape((3, 3)) / Bohr
        uctext = "\t\n".join(["\t".join(["%.5e" % x for x in ui]) for ui in unitcell])
        uc = ET.SubElement(root, "unitcell", units="bohrradius")
        uc.text = uctext
        nspin = len(model.zion)
        model._map_to_magnetic_only()
        id_spin = [-1] * nspin
        counter = 0
        for i in np.array(model.magsites, dtype=int):
            counter += 1
            id_spin[int(i)] = counter
        for i, z in enumerate(model.zion):
            atom = ET.SubElement(
                root,
                "atom",
                damping_factor="%.5e" % (1.0),
                gyroratio="%.5e" % (model.gyro_ratio[i] / gyromagnetic_ratio),
                mass="%.5e" % atomic_masses[z],
                index_spin="%d" % id_spin[i],
                massunits="atomicmassunit",
            )
            pos = ET.SubElement(atom, "position", units="bohrradius")
            pos.text = "%.5e\t%.5e\t%.5e" % tuple(model.xcart[i] / Bohr)
            spinat = ET.SubElement(atom, "spinat")
            spinat.text = "%.5e\t%.5e\t%.5e" % tuple(model.spinat[i])

        if model.has_exchange:
            exc = ET.SubElement(root, "spin_exchange_list", units="eV")
            ET.SubElement(exc, "nterms").text = "%s" % (len(model.exchange_Jdict))
            for key, val in model.exchange_Jdict.items():
                exc_term = ET.SubElement(exc, "spin_exchange_term", units="eV")
                ET.SubElement(exc_term, "ijR").text = "%d %d %d %d %d" % (
                    key[0] + 1,
                    key[1] + 1,
                    key[2][0],
                    key[2][1],
                    key[2][2],
                )
                # ET.SubElement(exc_term,
                #              "data").text = "%.5e \t %.5e \t %.5e" % (
                #                  val * J / eV, val * J / eV, val * J / eV)

                try:  # if val is a iterable
                    ET.SubElement(exc_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                        val[0] * J / eV,
                        val[1] * J / eV,
                        val[2] * J / eV,
                    )
                except Exception:
                    ET.SubElement(exc_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                        val * J / eV,
                        val * J / eV,
                        val * J / eV,
                    )
        if model.has_dmi:
            dmi = ET.SubElement(root, "spin_DMI_list", units="eV")
            ET.SubElement(dmi, "nterms").text = "%d" % len(model.dmi_ddict)
            for key, val in model.dmi_ddict.items():
                dmi_term = ET.SubElement(dmi, "spin_DMI_term")
                ET.SubElement(dmi_term, "ijR").text = "%d %d %d %d %d" % (
                    key[0] + 1,
                    key[1] + 1,
                    key[2][0],
                    key[2][1],
                    key[2][2],
                )
                ET.SubElement(dmi_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                    val[0] * J / eV,
                    val[1] * J / eV,
                    val[2] * J / eV,
                )

        if model.has_uniaxial_anistropy:
            uni = ET.SubElement(root, "spin_uniaxial_SIA_list", units="eV")
            ET.SubElement(uni, "nterms").text = "%d" % len(model.k1)
            for i, k1 in enumerate(model.k1):
                uni_term = ET.SubElement(uni, "spin_uniaxial_SIA_term")
                ET.SubElement(uni_term, "i").text = "%d " % (i + 1)
                ET.SubElement(uni_term, "amplitude").text = "%.5e" % (k1 * J / eV)
                ET.SubElement(
                    uni_term, "direction"
                ).text = "%.5e \t %.5e \t %.5e " % tuple(model.k1dir[i])

        if model.has_bilinear:
            bi = ET.SubElement()
            bilinear = ET.SubElement(root, "spin_bilinear_list", units="eV")
            ET.SubElement(bilinear, "nterms").text = "%d" % len(model.bilinear_J_dict)
            for key, val in model.bilinear_Jdict.items():
                bilinear_term = ET.SubElement(bilinear, "spin_bilinear_term")
                ET.SubElement(bilinear_term, "ijR").text = "%d %d %d %d %d" % (
                    key[0] + 1,
                    key[1] + 1,
                    key[2][0],
                    key[2][1],
                    key[2][2],
                )
                ET.SubElement(bilinear_term, "data").text = "\t".join(
                    ["%.5e" % (x * J / eV) for x in val]
                )

        tree = ET.ElementTree(root)
        tree.write(fname)


class SpinXmlParser(BaseSpinModelParser):
    """
    parser to spin xml file.
    """

    def _parse(self, fname):
        tree = ET.parse(fname)
        root = tree.getroot()

        self.atoms = Atoms()
        uc = root.find("unit_cell")
        cell = np.array([float(x) for x in uc.text.strip().split()])
        self.cell = np.reshape(cell, (3, 3)) * Bohr

        for a in root.findall("atom"):
            if "damping_factor" in a.attrib:
                self.damping_factors.append(float(a.attrib["damping_factor"]))
            else:
                print("warning: damping factor not found")
                self.damping_factors.append(0.0)

            if "gyroratio" in a.attrib:
                self.gyro_ratios.append(float(a.attrib["gyroratio"]))
            else:
                print("warning: gyroratio not found")
                self.gyro_ratios.append(0.0)

            if "index_spin" in a.attrib:
                self.index_spin.append(int(a.attrib["index_spin"]))
            else:
                print("warning: index_spin not found")
                self.index_spin.append(-1)

            if "mass" in a.attrib:
                self.masses.append(float(a.attrib["mass"]))
            else:
                print("warning: mass not found")
                self.masses.append(1)

            if "zion" in a.attrib:
                self.zions.append(float(a.attrib["zion"]))
            else:
                # print("warning: zion not found")
                self.zions.append(1)

            p = a.find("position")
            if p is not None:
                position = np.array(tuple(map(float, p.text.strip().split()))) * Bohr
                self.positions.append(position)
            else:
                raise Exception("position not found for one atom")

            p = a.find("spinat")
            if p is not None:
                spin = np.array(tuple(map(float, p.text.strip().split())))
                self.spinat.append(spin)
            else:
                raise Exception("spinat not found for one atom")

        exch = root.find("spin_exchange_list")
        n_exch = int(exch.find("nterms").text)

        for exc in exch.findall("spin_exchange_term"):
            ijR = [int(x) for x in exc.find("ijR").text.strip().split()]
            i, j, R0, R1, R2 = ijR
            val = [float(x) for x in exc.find("data").text.strip().split()]
            self._exchange[(i - 1, j - 1, (R0, R1, R2))] = np.array(val) * eV / J
        assert (
            len(self._exchange) == n_exch
        ), "Number of exchange terms different from nterms in xml file"

        dmi = root.find("spin_DMI_list")
        if dmi is not None:
            n_dmi = int(dmi.find("nterms").text)
            for dm in dmi.findall("spin_DMI_term"):
                ijR = [int(x) for x in dm.find("ijR").text.strip().split()]
                i, j, R0, R1, R2 = ijR
                val = [float(x) for x in dm.find("data").text.strip().split()]
                self._dmi[(i - 1, j - 1, (R0, R1, R2))] = np.array(val) * eV / J
            assert (
                len(self._dmi) == n_dmi
            ), f"Number of dmi terms {len(self._dmi)} different from nterms in xml file {n_dmi}"

        bil = root.find("spin_bilinear_list")
        if bil is not None:
            n_bil = int(bil.find("nterms").text)
            for bi in bil.findall("spin_bilinear_term"):
                ijR = [int(x) for x in bi.find("ijR").text.strip().split()]
                i, j, R0, R1, R2 = ijR
                val = [float(x) for x in bi.find("data").text.strip().split()]
                self._bilinear[(i - 1, j - 1, (R0, R1, R2))] = (
                    np.array(val).reshape((3, 3)) * eV / J
                )
            assert (
                len(self._bilinear) == n_bil
            ), f"Number of bilinear terms {len(self._bil)} different from nterms in xml file {n_bil}"
