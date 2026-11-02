import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from TB2J.interfaces.unfold_epc import load_unfold_epc


def test_tb2j_green_operation_remains_available_without_unfold_epc():
    code = """
import importlib.abc
import sys
class MissingUnfold(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'unfold_epc' or fullname.startswith('unfold_epc.'):
            raise ModuleNotFoundError('optional unfold-epc absent', name=fullname)
sys.meta_path.insert(0, MissingUnfold())
import numpy as np
import TB2J
from TB2J.green import eigen_to_G2
from TB2J.interfaces.unfold_epc import load_unfold_epc
h=np.diag([-1.,1.]); overlap=np.eye(2)
np.testing.assert_allclose(eigen_to_G2(h,overlap,0.,2j),np.diag([1/(1+2j),1/(-1+2j)]))
try:
    load_unfold_epc('not-opened.nc')
except ImportError as exc:
    assert 'unfold-epc' in str(exc)
else:
    raise AssertionError('optional consumer call must explain the missing dependency')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_tb2j_loader_rejects_unknown_store_schema(tmp_path):
    from unfold_epc import TwoIndexStore

    store = TwoIndexStore(
        np.eye(3),
        np.eye(3, dtype=int),
        np.array([0]),
        np.zeros((1, 3), int),
        1,
        1,
        2,
        0.0,
        0.005,
        {},
        {"input_hashes": {}},
    )
    path = tmp_path / "epc.nc"
    store.write(path)
    with Dataset(path, "r+") as nc:
        nc.schema_version = 999
    with pytest.raises(ValueError):
        load_unfold_epc(path)
