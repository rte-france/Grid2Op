# making sure test can be ran from:
# root package directory
# Grid2Op subdirectory
# Grid2Op/tests subdirectory
import sys
import os
import unittest
import numpy as np
from pathlib import Path

test_dir = os.fspath(Path(__file__).parent.absolute())
grid2op_dir = os.fspath(Path(__file__).parent.parent.absolute())
data_dir = os.path.abspath(os.path.join(grid2op_dir, "data"))

sys.path.insert(0, grid2op_dir)

PATH_DATA_TEST = data_dir
PATH_CHRONICS = data_dir
if not os.path.exists(os.path.join(PATH_DATA_TEST, "chronics_with_forecast")):
    raise RuntimeError("Impossible to find the test data folder")
PATH_DATA_TEST_PP = os.path.abspath(os.path.join(PATH_DATA_TEST, "test_PandaPower"))


class HelperTests(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        unittest.TestCase.__init__(self, methodName=methodName)
        self.tolvect = 1e-2
        self.tol_one = 1e-5

    def compare_vect(self, pred, true):
        return np.max(np.abs(pred- true)) <= self.tolvect
