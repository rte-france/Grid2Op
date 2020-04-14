# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import sys
import unittest
import warnings
import time
import numpy as np
import pdb

from grid2op.tests.helper_path_test import PATH_CHRONICS, PATH_DATA_TEST_PP
from grid2op.tests.helper_path_test import EXAMPLE_CHRONICSPATH, EXAMPLE_CASEFILE
from grid2op.tests.helper_data_test import case14_redisp_TH_LIM, case14_test_TH_LIM, case14_real_TH_LIM
from grid2op.tests.helper_data_test import case14_redisp_layout, case14_test_layout, case14_real_layout
from grid2op.tests.helper_data_test import L2RPN_2019_dict, L2RPN_2019_layout

from grid2op.Chronics.Settings_L2RPN2019 import ReadPypowNetData

from grid2op.Exceptions import *
from grid2op.MakeEnv import make, make2, _get_default_aux
from grid2op.Environment import Environment
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, Multifolder, ChangeNothing
from grid2op.Chronics import GridStateFromFile, GridStateFromFileWithForecasts, GridValue
from grid2op.Action import BaseAction, TopologyAction, TopologyAndDispatchAction, VoltageOnlyAction
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Reward import FlatReward, BaseReward, L2RPNReward, RedispReward
from grid2op.Rules import BaseRules, AlwaysLegal, DefaultRules
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Opponent import BaseOpponent

# TODO make a test that the defaults are correct for all environment below
# (eg that the env.chronics_handler has
# by default the type given in the "make" function,
# that the backend if of the proper type, that the thermal
# limit are properly set up etc.
# basically, test, for all env, all that is defined there:
# if name_env.lower() == "case14_fromfile":
#    default_grid_path = CASE_14_FILE
#    if chronics_path == '':
#        chronics_path = CHRONICS_MLUTIEPISODE
#
#    defaultinstance_chronics_kwargs = {"chronicsClass": Multifolder, "path": chronics_path,
#                                       "gridvalueClass": GridStateFromFileWithForecasts}
#    default_name_converter = NAMES_CHRONICS_TO_BACKEND
#    default_action_class = TopologyAction
#    default_reward_class = L2RPNReward


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


class TestkwargsName(unittest.TestCase):
    def test_param(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", param=Parameters()) as env:
                obs = env.reset()

    def test_backend(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", backend=PandaPowerBackend()) as env:
                obs = env.reset()

    def test_obsclass(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", observation_class=CompleteObservation) as env:
                obs = env.reset()

    def test_gamerules(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", gamerules_class=AlwaysLegal) as env:
                obs = env.reset()

    def test_chronics_path(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", chronics_path=EXAMPLE_CHRONICSPATH) as env:
                obs = env.reset()

    def test_reward_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", reward_class=FlatReward) as env:
                obs = env.reset()

    def test_action_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", action_class=BaseAction) as env:
                obs = env.reset()

    def test_grid_path(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", grid_path=EXAMPLE_CASEFILE) as env:
                obs = env.reset()

    def test_names_chronics_to_backend(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", names_chronics_to_backend={}) as env:
                obs = env.reset()

    def test_data_feeding_kwargs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dict_ = {"chronicsClass": Multifolder, "path": EXAMPLE_CHRONICSPATH,
                    "gridvalueClass": GridStateFromFileWithForecasts}
            with make("case5_example", data_feeding_kwargs=dict_) as env:
                obs = env.reset()

    def test_chronics_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", chronics_class=Multifolder) as env:
                pass

    def test_volagecontroler_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", volagecontroler_class=ControlVoltageFromFile) as env:
                obs = env.reset()

    def test_other_rewards(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", other_rewards={"test": L2RPNReward}) as env:
                obs = env.reset()

    def test_opponent_action_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", opponent_action_class=BaseAction) as env:
                obs = env.reset()

    def test_opponent_class(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", opponent_class=BaseOpponent) as env:
                obs = env.reset()

    def test_opponent_init_budget(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("case5_example", opponent_init_budget=10) as env:
                obs = env.reset()


class TestMake2Config(unittest.TestCase):
    def test_case5_config(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        with make2(dataset_path) as env:
            # Check config is loaded from config.py
            assert env.rewardClass == L2RPNReward
            assert env.actionClass == TopologyAction
            assert env.observationClass == CompleteObservation
            assert isinstance(env.backend, PandaPowerBackend)
            assert env.legalActClass == DefaultRules
            assert isinstance(env.voltage_controler, ControlVoltageFromFile)
            assert isinstance(env.chronics_handler.real_data, Multifolder)
            assert env.action_space.grid_layout != None
            
    def test_case5_runs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        with make2(dataset_path) as env:
            assert env.redispatching_unit_commitment_availble == True
            obs = env.reset()
            sim_obs, reward, done, info = obs.simulate(env.action_space())
            assert sim_obs != obs

    def test_case14_test_config(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        with make2(dataset_path) as env:
            # Check config is loaded from config.py
            assert env.rewardClass == RedispReward
            assert env.actionClass == TopologyAndDispatchAction
            assert env.observationClass == CompleteObservation
            assert isinstance(env.backend, PandaPowerBackend)
            assert env.legalActClass == DefaultRules
            assert isinstance(env.voltage_controler, ControlVoltageFromFile)
            assert isinstance(env.chronics_handler.real_data, Multifolder)
            assert env.action_space.grid_layout != None
            
    def test_case14_test_runs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        with make2(dataset_path) as env:
            assert env.redispatching_unit_commitment_availble == True
            obs = env.reset()
            sim_obs, reward, done, info = obs.simulate(env.action_space())
            assert sim_obs != obs
            assert np.all(env._thermal_limit_a == case14_test_TH_LIM)

    def test_case14_redisp_config(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_redisp")
        with make2(dataset_path) as env:
            # Check config is loaded from config.py
            assert env.rewardClass == RedispReward
            assert env.actionClass == TopologyAndDispatchAction
            assert env.observationClass == CompleteObservation
            assert isinstance(env.backend, PandaPowerBackend)
            assert env.legalActClass == DefaultRules
            assert isinstance(env.voltage_controler, ControlVoltageFromFile)
            assert isinstance(env.chronics_handler.real_data, Multifolder)
            
    def test_case14_redisp_runs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_redisp")
        with make2(dataset_path) as env:
            assert env.redispatching_unit_commitment_availble == True
            obs = env.reset()
            sim_obs, reward, done, info = obs.simulate(env.action_space())
            assert sim_obs != obs
            assert np.all(env._thermal_limit_a == case14_redisp_TH_LIM)

    def test_l2rpn19_test_config(self):
        dataset_path = os.path.join(PATH_CHRONICS, "l2rpn_2019")
        with make2(dataset_path) as env:
            # Check config is loaded from config.py
            assert env.rewardClass == L2RPNReward
            assert env.actionClass == TopologyAction
            assert env.observationClass == CompleteObservation
            assert isinstance(env.backend, PandaPowerBackend)
            assert env.legalActClass == DefaultRules
            assert isinstance(env.voltage_controler, ControlVoltageFromFile)
            assert isinstance(env.chronics_handler.real_data, Multifolder)
            assert env.action_space.grid_layout != None


class TestMake2ConfigOverride(unittest.TestCase):
    def test_case5_override_reward(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        with make2(dataset_path, reward_class=FlatReward) as env:
            assert env.rewardClass == FlatReward

    def test_case14_test_override_reward(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        with make2(dataset_path, reward_class=FlatReward) as env:
            assert env.rewardClass == FlatReward

    def test_l2rpn19_override_reward(self):
        dataset_path = os.path.join(PATH_CHRONICS, "l2rpn_2019")
        with make2(dataset_path, reward_class=FlatReward) as env:
            assert env.rewardClass == FlatReward

    def test_case5_override_action(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        with make2(dataset_path, action_class=VoltageOnlyAction) as env:
            assert env.actionClass == VoltageOnlyAction

    def test_case14_test_override_action(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        with make2(dataset_path, action_class=VoltageOnlyAction) as env:
            assert env.actionClass == VoltageOnlyAction

    def test_l2rpn19_override_action(self):
        dataset_path = os.path.join(PATH_CHRONICS, "l2rpn_2019")
        with make2(dataset_path, action_class=VoltageOnlyAction) as env:
            assert env.actionClass == VoltageOnlyAction

    def test_case5_override_chronics(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        with make2(dataset_path, chronics_class=ChangeNothing) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)

    def test_case14_test_override_chronics(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        with make2(dataset_path, chronics_class=ChangeNothing) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)

    def test_l2rpn19_override_chronics(self):
        dataset_path = os.path.join(PATH_CHRONICS, "l2rpn_2019")
        with make2(dataset_path, chronics_class=ChangeNothing) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)

    def test_case5_override_feed_kwargs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case5_example")
        chronics_path = os.path.join(dataset_path, "chronics", "0")
        dfk = {
            "chronicsClass": ChangeNothing,
            "path": chronics_path,
            "gridvalueClass": GridStateFromFile
        }
        with make2(dataset_path, data_feeding_kwargs=dfk) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)

    def test_case14_test_override_feed_kwargs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "rte_case14_test")
        chronics_path = os.path.join(dataset_path, "chronics", "0")
        dfk = {
            "chronicsClass": ChangeNothing,
            "path": chronics_path,
            "gridvalueClass": GridStateFromFile
        }
        with make2(dataset_path, data_feeding_kwargs=dfk) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)

    def test_l2rpn19_override_feed_kwargs(self):
        dataset_path = os.path.join(PATH_CHRONICS, "l2rpn_2019")
        chronics_path = os.path.join(dataset_path, "chronics", "0000")
        dfk = {
            "chronicsClass": ChangeNothing,
            "path": chronics_path,
            "gridvalueClass": GridStateFromFile
        }
        with make2(dataset_path, data_feeding_kwargs=dfk) as env:
            assert isinstance(env.chronics_handler.real_data, ChangeNothing)


if __name__ == "__main__":
    unittest.main()
