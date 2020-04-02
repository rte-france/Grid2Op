import os
import sys
import unittest
import numpy as np
import pdb
import warnings
from grid2op.tests.helper_path_test import *
from grid2op.Environment import MultiEnvironment
from grid2op import make


class TestLoadingMultiEnv(unittest.TestCase):
    def test_creation_multienv(self):
        nb_env = 1
        with make("case5_example") as env:
            multi_envs = MultiEnvironment(env=env, nb_env=nb_env)
        multi_envs.close()


if __name__ == "__main__":
    unittest.main()
