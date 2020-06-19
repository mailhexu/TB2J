import os
import sys
import pickle
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    from TB2J.io_merge import merge
    merge(sys.argv[1], sys.argv[2], sys.argv[3], save=True)


main()
