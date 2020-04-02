import os
import sys
import unittest
import numpy as np
import pdb
import warnings
from grid2op.tests.helper_path_test import *
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op import make


class TestLoadingVoltageControl(unittest.TestCase):
    def test_creation_ControlVoltage(self):
        nb_env = 1
        with make("case5_example") as env:
            volt_cont = ControlVoltageFromFile(controler_backend=env.backend, gridobj=env.backend)


if __name__ == "__main__":
    unittest.main()