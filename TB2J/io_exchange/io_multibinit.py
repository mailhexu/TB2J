import xml.etree.cElementTree as ET
from xml.dom import minidom
from ase.units import Bohr
import os


def write_multibinit_inp(cls, path="TB2J_results/Multibinit"):
    if not os.path.exists(path):
        os.makedirs(path)
    # files file
    filesfname = os.path.join(path, "mb.files")
    with open(filesfname, "w") as myfile:
        myfile.write(
            """mb.in
mb.out
exchange.xml
"""
        )
    inputfname = os.path.join(path, "mb.in")
    # template for input file
    with open(inputfname, "w") as myfile:
        infile_template = """
prt_model = 0

#--------------------------------------------------------------
#Monte carlo / molecular dynamics
#--------------------------------------------------------------
dynamics =  0    ! disable molecular dynamics

ncell =   16 16 16  ! size of supercell. 
#-------------------------------------------------------------
#Spin dynamics
#------------------------------------------------------------
spin_dynamics=1  ! enable spin dynamics
spin_mag_field= 0.0 0.0 0.0   ! external magnetic field
spin_ntime_pre =10000          ! warming up steps. 
spin_ntime =100000             ! number of steps. 
spin_nctime=100               ! number of time steps between two nc file write
spin_dt=5e-16 s               ! time step. 
spin_init_state = 1           ! random initial spin

spin_temperature=0.0

spin_var_temperature=1        ! switch on variable temperature calculation
spin_temperature_start=0      ! starting point of temperature
spin_temperature_end=500      ! ending point of temperature. 
spin_temperature_nstep=6      ! number of temperature steps.

spin_sia_add = 1              ! add a single ion anistropy (SIA) term?
spin_sia_k1amp = 1e-6         ! amplitude of SIA (in Ha), how large should be used?
spin_sia_k1dir = 0.0 0.0 1.0  ! direction of SIA

spin_calc_thermo_obs = 1      ! calculate thermodynamics related observables
        """
        myfile.write(infile_template)


def write_xml(cls, fname):
    root = ET.Element("System_definition")
    try:
        unitcell = cls.atoms.get_cell().reshape((3, 3)) / Bohr
    except Exception:  # ase>3.18 cell is not a array.
        unitcell = cls.atoms.get_cell().array.reshape((3, 3)) / Bohr
    uctext = "\t\n".join(["\t".join(["%.5e" % x for x in ui]) for ui in unitcell])
    uc = ET.SubElement(root, "unit_cell", units="bohrradius")
    uc.text = uctext
    # ET.SubElement(
    #    root, "unit_cell", units="bohrradius").text = "\t".join(
    #        list(
    #            "%.5e" % x
    #            for x in np.array(cls.atoms.get_cell()).flatten() / Bohr))
    natom = len(cls.atoms)
    # cls._map_to_magnetic_only()
    # id_spin = [-1] * natom
    # counter = 0
    # for i in np.array(cls.magsites, dtype=int):
    #    counter += 1
    #    id_spin[int(i)] = counter

    for i in range(natom):
        atom = ET.SubElement(
            root,
            "atom",
            mass="%.5e" % cls.atoms.get_masses()[i],
            massunits="atomicmassunit",
            damping_factor="%.5f"
            % cls.damping[i],  # TODO remove this. damping factor is not local.
            gyroratio="%.5e" % cls.gyro_ratio[i],
            index_spin="%d" % (cls.index_spin[i] + 1),  # +1 in fortran
        )
        pos = ET.SubElement(atom, "position", units="bohrradius")
        pos.text = "%.5e\t%.5e\t%.5e" % tuple(cls.atoms.get_positions()[i] / Bohr)
        spinat = ET.SubElement(atom, "spinat")
        spinat.text = "%.5e\t%.5e\t%.5e" % tuple(cls.spinat[i])

    if cls.has_exchange:
        exc = ET.SubElement(root, "spin_exchange_list", units="eV")
        ET.SubElement(exc, "nterms").text = "%s" % (len(cls.exchange_Jdict))
        for key, val in cls.exchange_Jdict.items():
            R, i, j = key
            exc_term = ET.SubElement(exc, "spin_exchange_term")
            ET.SubElement(exc_term, "ijR").text = "%d %d %d %d %d" % (
                i + 1,
                j + 1,
                R[0],
                R[1],
                R[2],
            )
            try:
                ET.SubElement(exc_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                    val[0],
                    val[1],
                    val[2],
                )
            except Exception:
                ET.SubElement(exc_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                    val,
                    val,
                    val,
                )
    if cls.has_dmi:
        dmi = ET.SubElement(root, "spin_DMI_list", units="eV")
        ET.SubElement(dmi, "nterms").text = "%d" % len(cls.dmi_ddict)
        for key, val in cls.dmi_ddict.items():
            R, i, j = key
            dmi_term = ET.SubElement(dmi, "spin_DMI_term")
            ET.SubElement(dmi_term, "ijR").text = "%d %d %d %d %d" % (
                i + 1,
                j + 1,
                R[0],
                R[1],
                R[2],
            )
            ET.SubElement(dmi_term, "data").text = "%.5e \t %.5e \t %.5e" % (
                val[0],
                val[1],
                val[2],
            )

    if cls.has_uniaxial_anistropy:
        uni = ET.SubElement(root, "spin_uniaxial_SIA_list", units="eV")
        ET.SubElement(uni, "nterms").text = "%d" % len(cls.k1)
        for i, k1 in enumerate(cls.k1):
            uni_term = ET.SubElement(uni, "spin_uniaxial_SIA_term")
            ET.SubElement(uni_term, "i").text = "%d " % (i + 1)
            ET.SubElement(uni_term, "amplitude").text = "%.5e" % k1
            ET.SubElement(uni_term, "direction").text = "%.5e \t %.5e \t %.5e " % tuple(
                cls.k1dir[i]
            )

    if cls.has_bilinear:
        bilinear = ET.SubElement(root, "spin_bilinear_list", units="eV")
        ET.SubElement(bilinear, "nterms").text = "%d" % len(cls.Jani_dict)
        for key, val in cls.Jani_dict.items():
            bilinear_term = ET.SubElement(bilinear, "spin_bilinear_term")
            ET.SubElement(bilinear_term, "ijR").text = "%d %d %d %d %d" % (
                key[1] + 1,
                key[2] + 1,
                key[0][0],
                key[0][1],
                key[0][2],
            )
            ET.SubElement(bilinear_term, "data").text = "\t".join(
                ["%.5e" % x for x in val.flatten()]
            )
    if cls.description is not None:
        description = ET.SubElement(root, "description")
        description.text = cls.description

    # tree = ET.ElementTree(root)
    # tree.write(fname)
    with open(fname, "w") as myfile:
        myfile.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t"))


def write_multibinit(cls, path):
    write_multibinit_inp(cls, path=path)
    xmlfname = os.path.join(path, "exchange.xml")
    write_xml(cls, xmlfname)
