from TB2J.abacus.abacus_wrapper import AbacusWrapper, AbacusParser
from TB2J.mathutils.rotate_spin import rotate_Matrix_from_z_to_axis


class AbacusSplitSOCWrapper(AbacusWrapper):
    """
    Abacus wrapper with Hamiltonian split to SOC and non-SOC parts
    """

    def __init__(self, **kwargs):
        HR_soc = kwargs.pop("HR_soc", None)
        super().__init__(**kwargs)
        self.HR_soc = HR_soc

    def rotate_Hxc(self, axis):
        """
        Rotate SOC part of Hamiltonian
        """
        for iR, R in enumerate(self.Rlist):
            self.HR[iR] = rotate_Matrix_from_z_to_axis(self.HR[iR], axis)


class AbacusSplitSOCParser:
    """
    Abacus parser with Hamiltonian split to SOC and non-SOC parts
    """

    def __init__(self, outpath_nosoc=None, outpath_soc=None, binary=False):
        self.outpath_nosoc = outpath_nosoc
        self.outpath_soc = outpath_soc
        self.binary = binary
        parser_nosoc = AbacusParser(outpath=outpath_nosoc, binary=binary)
        parser_soc = AbacusParser(outpath=outpath_soc, binary=binary)
