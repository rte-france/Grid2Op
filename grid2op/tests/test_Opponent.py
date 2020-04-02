import os
import sys
import unittest
import numpy as np
import pdb
import warnings
from grid2op.tests.helper_path_test import *
from grid2op.Opponent import BaseOpponent
from grid2op import make


class TestLoadingOpp(unittest.TestCase):
    def test_creation_BaseOpponent(self):
        nb_env = 1
        with make("case5_example") as env:
            my_opp = BaseOpponent(action_space=env.action_space)


if __name__ == "__main__":
    unittest.main()