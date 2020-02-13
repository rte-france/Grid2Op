# making some test that the backned is working as expected
import os
import sys
import unittest
import warnings

import numpy as np
import pdb

from helper_path_test import PATH_DATA_TEST_PP, PATH_CHRONICS

from Exceptions import *
from MakeEnv import make, _get_default_aux

import time
# TODO check that _get_default_aux properly catches the exception too


class TestLoadingPredefinedEnv(unittest.TestCase):
    def test_case14_fromfile(self):
        env = make("case14_fromfile")
        obs = env.reset()

    def test_l2rpn_2019(self):
        try:
            env = make("l2rpn_2019")
        except EnvError as e:
            pass

    def test_case5_example(self):
        env = make("case5_example")
        obs = env.reset()

    def test_case14_redisp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("case14_redisp")
            obs = env.reset()

    def test_case14_test(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("case14_test")
            obs = env.reset()


class TestGetDefault(unittest.TestCase):
    def test_give_instance_default(self):
        kwargs = {}
        param = _get_default_aux('param', kwargs, defaultClass=str, defaultClassApp=str,
                                 msg_error="bad stuff", isclass=False)
        assert param == str(), "This should have returned the empty string"

    def test_give_instance_nodefault(self):
        kwargs = {"param": "toto"}
        param = _get_default_aux('param', kwargs, defaultClass=str, defaultClassApp=str,
                                 msg_error="bad stuff", isclass=False)
        assert param == "toto", "This should have returned \"toto\""

    def test_give_class_default(self):
        kwargs = {}
        param = _get_default_aux('param', kwargs, defaultClass=str, defaultClassApp=str,
                                 msg_error="bad stuff", isclass=True)
        assert param == str, "This should have returned the empty string"

    def test_give_class_nodefault(self):
        kwargs = {"param": str}
        param = _get_default_aux('param', kwargs, defaultClass=str, defaultClassApp=str,
                                 msg_error="bad stuff", isclass=True)
        assert param == str, "This should have returned \"toto\""


if __name__ == "__main__":
    unittest.main()