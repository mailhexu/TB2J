import numpy as np
import warnings
import sys

MAX_EXP_ARGUMENT = np.log(sys.float_info.max)


def fermi(e, mu, width=0.01):
    """
    the fermi function.
     .. math::
        f=\\frac{1}{\exp((e-\mu)/width)+1}

    :param e,mu,width: e,\mu,width
    """
    x = (e - mu) / width
    # disable overflow warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = np.where(x < MAX_EXP_ARGUMENT, 1 / (1.0 + np.exp(x)), 0.0)

    return ret
