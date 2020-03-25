# making some test that the backned is working as expected
import os
import sys
import unittest
import warnings

import numpy as np
import pdb

from helper_path_test import *

from grid2op.Chronics.Settings_case14_redisp import case14_redisp_TH_LIM
from grid2op.Chronics.Settings_case14_test import case14_test_TH_LIM
from grid2op.Chronics.Settings_case14_realistic import case14_real_TH_LIM

from grid2op.Exceptions import *
from grid2op.MakeEnv import make, make2, _get_default_aux

import time

class TestLoadingPredefinedEnv(unittest.TestCase):
    def test_case14_fromfile(self):
        env = make("case14_fromfile")
        obs = env.reset()

    def test_l2rpn_2019(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = make("l2rpn_2019")
        except EnvError as e:
            pass

    def test_case5_example(self):
        env = make("case5_example")
        obs = env.reset()

    def test_case5_redispatch_available(self):
        with make("case5_example") as env:
            obs = env.reset()
            assert env.redispatching_unit_commitment_availble == True

    def test_case5_can_simulate(self):
        with make("case5_example") as env:
            obs = env.reset()
            sim_obs, reward, done, info = obs.simulate(env.action_space())
            assert sim_obs != obs

    def test_case14_redisp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("case14_redisp")
            obs = env.reset()

    def test_case14redisp_redispatch_available(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_redisp") as env:
                    obs = env.reset()
                    assert env.redispatching_unit_commitment_availble == True

    def test_case14redisp_can_simulate(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_redisp") as env:
                    obs = env.reset()
                    sim_obs, reward, done, info = obs.simulate(env.action_space())
                    assert sim_obs != obs

    def test_case14redisp_test_thermals(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case14_redisp") as env:
                obs = env.reset()
                assert np.all(env._thermal_limit_a == case14_redisp_TH_LIM)

    def test_case14_realistic(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("case14_realistic")
            obs = env.reset()

    def test_case14realistic_redispatch_available(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_realistic") as env:
                    obs = env.reset()
                    assert env.redispatching_unit_commitment_availble == True

    def test_case14realistic_can_simulate(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_realistic") as env:
                    obs = env.reset()
                    sim_obs, reward, done, info = obs.simulate(env.action_space())
                    assert sim_obs != obs

    def test_case14realistic_test_thermals(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case14_realistic") as env:
                obs = env.reset()
                assert np.all(env._thermal_limit_a == case14_real_TH_LIM)

    def test_case14_test(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("case14_test")
            obs = env.reset()

    def test_case14test_redispatch_available(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_test") as env:
                    obs = env.reset()
                    assert env.redispatching_unit_commitment_availble == True

    def test_case14test_can_simulate(self):
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with make("case14_test") as env:
                    obs = env.reset()
                    sim_obs, reward, done, info = obs.simulate(env.action_space())
                    assert sim_obs != obs

    def test_case14test_thermals(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case14_test") as env:
                obs = env.reset()
                assert np.all(env._thermal_limit_a == case14_test_TH_LIM)
            
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

    def test_use_sentinel_arg_raises(self):
        with self.assertRaises(RuntimeError):
            _get_default_aux('param', {}, str, _sentinel=True)

    def test_class_not_instance_of_defaultClassApp_raises(self):
        with self.assertRaises(EnvError):
            kwargs = {"param": int}
            _get_default_aux('param', kwargs, defaultClassApp=str, isclass=False)

    def test_type_is_instance_raises(self):
        with self.assertRaises(EnvError):
            kwargs = {"param": 0}
            _get_default_aux('param', kwargs, defaultClassApp=int, isclass=True)

    def test_type_not_subtype_of_defaultClassApp_raises(self):
        with self.assertRaises(EnvError):
            kwargs = {"param": str}
            _get_default_aux('param', kwargs, defaultClassApp=int, isclass=True)

    def test_default_instance_and_class_raises(self):
        with self.assertRaises(EnvError):
            _get_default_aux('param', {}, str,
                             defaultClass=str, defaultinstance="strinstance",
                             isclass=False)

    def test_default_instance_with_build_kwargs_raises(self):
        with self.assertRaises(EnvError):
            _get_default_aux('param', {}, str,
                             defaultinstance="strinstance", isclass=False,
                             build_kwargs=['s', 't', 'r'])

    def test_no_default_provided_raises(self):
        with self.assertRaises(EnvError):
            _get_default_aux('param', {}, str,
                             defaultinstance=None, defaultClass=None,
                             isclass=False)

    def test_class_with_provided_build_kwargs_raises(self):
        with self.assertRaises(EnvError):
            _get_default_aux('param', {}, str,
                             defaultClass=str,
                             isclass=True, build_kwargs=['s', 't', 'r'])

    def test_class_with_provided_instance_raises(self):
        with self.assertRaises(EnvError):
            _get_default_aux('param', {}, str,
                             defaultClass=str,
                             defaultinstance="strinstance",
                             isclass=True)

if __name__ == "__main__":
    unittest.main()
