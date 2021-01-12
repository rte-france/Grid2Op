# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import pdb
import time
import warnings

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, ChangeNothing
from grid2op.Reward import L2RPNReward
from grid2op.MakeEnv import make
from grid2op.Rules import RulesChecker, DefaultRules
from grid2op.Action import *
from grid2op.dtypes import dt_float

import warnings

# TODO check when ambiguous
# TODO check when there is also redispatching


class TestStorageEnv(HelperTests):
    """test the env part of the storage functionality"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True)

    def tearDown(self) -> None:
        self.env.close()

    def test_init_storage_ok(self):
        """
        test the right storage is given at the init of an environment
        it tests the param.INIT_STORAGE_CAPACITY parameter
        """
        obs = self.env.get_obs()
        assert np.all(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one)

        obs = self.env.reset()
        assert np.all(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one)

        param = Parameters()
        param.INIT_STORAGE_CAPACITY = 0.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.reset()
        assert np.all(np.abs(obs.storage_charge) <= self.tol_one)

        param = Parameters()
        param.INIT_STORAGE_CAPACITY = 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.reset()
        assert np.all(np.abs(obs.storage_charge - obs.storage_Emax) <= self.tol_one)

        # now test the reset works
        act = self.env.action_space({"set_storage": [(0, 1), (1, 5)]})
        obs, reward, done, info = self.env.step(act)
        assert np.any(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) > self.tol_one)
        obs = self.env.reset()
        assert np.all(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one)

    def test_action_ok(self):
        """test a storage action is supported (basic test)"""
        act = self.env.action_space({"set_storage": [(0, 1)]})
        str_ = act.__str__()
        real_str = 'This action will:\n\t - NOT change anything to the injections\n\t - NOT perform any ' \
                   'redispatching action\n\t - set the new power produced / absorbed for storage storage_5_0 ' \
                   'to be 1.0 MW\n\t - NOT force any line status\n\t - NOT switch any line status\n\t - NOT ' \
                   'switch anything in the topology\n\t - NOT force any particular bus configuration'
        assert str_ == real_str

    def test_env_storage_ok(self):
        """test i can perform normal storage actions (no trick here) just normal actions"""

        # first test the storage loss
        act = self.env.action_space()
        loss = 1.0 * self.env.storage_loss
        loss /= 12.  # i have 12 steps per hour (ts = (mins), losses are given in MW and capacity in MWh
        for nb_ts in range(5):
            obs, reward, done, info = self.env.step(act)
            assert np.all(np.abs(obs.storage_charge - (0.5 * obs.storage_Emax - (nb_ts + 1) * loss)) <= self.tol_one), \
                   f"wrong value computed for time step {nb_ts}"

        # now modify the storage capacity
        # storage value is [7.4583335, 3.4583333]
        act = self.env.action_space({"set_storage": [(0, 3)]})  # ask the first storage to absorb 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.4583335, 3.4583333] - loss
        # modif adds [3 / 12, 0.]  # 3/12 is because i ask to absorb 3 MW during 5 mins
        # final result is [7.70000017, 3.44999997]
        assert np.all(np.abs(obs.storage_charge - [7.70000017, 3.44999997]) <= self.tol_one)
        assert np.all(np.abs(obs.target_dispatch) <= self.tol_one)  # i did not do any dispatch
        # I asked to absorb 3MW, so dispatch should produce 3MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (+3.)) <= self.tol_one

        # second action (battery produces something)
        act = self.env.action_space({"set_storage": [(1, -5)]})  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.70000017, 3.44999997] - loss
        # modif adds [0., -5/12.]  # -5/12 is because i ask to produce 5 MW during 5 mins
        # final result is [7.69166684, 3.02499997]
        assert np.all(np.abs(obs.storage_charge - [7.69166684, 3.02499997]) <= self.tol_one)
        assert np.all(np.abs(obs.target_dispatch) <= self.tol_one)  # i did not do any dispatch
        # I asked to produce 5MW, so dispatch should produce 5MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (-5.)) <= self.tol_one

        # third i do nothing and make sure everything is reset
        act = self.env.action_space()  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.69166684, 3.02499997]- loss
        assert np.all(np.abs(obs.storage_charge - ([7.68333351, 3.01666664])) <= self.tol_one)
        assert np.all(np.abs(obs.target_dispatch) <= self.tol_one)  # i did not do any dispatch
        # I did not modify the battery, so i should not modify the dispatch
        assert np.abs(np.sum(obs.actual_dispatch) - (0.)) <= self.tol_one

        # third i do nothing and make sure everything is reset
        act = self.env.action_space()  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.68333351, 3.01666664] - loss
        assert np.all(np.abs(obs.storage_charge - ([7.67500018, 3.00833331])) <= self.tol_one)
        assert np.all(np.abs(obs.target_dispatch) <= self.tol_one)  # i did not do any dispatch
        # I did not modify the battery, so i should not modify the dispatch
        assert np.abs(np.sum(obs.actual_dispatch) - (0.)) <= self.tol_one

    def test_activate_storage_loss(self):
        """
        test that the parameters param.ACTIVATE_STORAGE_LOSS properly deactivate the loss in the storage
        units
        """
        # first test the storage loss
        act = self.env.action_space()
        loss = 1.0 * self.env.storage_loss
        loss /= 12.  # i have 12 steps per hour (ts = (mins), losses are given in MW and capacity in MWh
        for nb_ts in range(5):
            obs, reward, done, info = self.env.step(act)
            assert np.all(np.abs(obs.storage_charge - (0.5 * obs.storage_Emax - (nb_ts + 1) * loss)) <= self.tol_one), \
                   f"wrong value computed for time step {nb_ts}"

        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.get_obs()
        assert np.all(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one), "wrong initial capacity"
        for nb_ts in range(5):
            obs, reward, done, info = env.step(act)
            assert np.all(np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one), \
                   f"wrong value computed for time step {nb_ts} (no loss in storage)"

    def test_storage_loss_dont_make_negative(self):
        """
        test that the storage loss dont make negative capacity
        or in other words that loss don't apply when storage are empty
        """
        init_coeff = 0.01
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = True
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.get_obs()
        init_charge = init_coeff * obs.storage_Emax
        loss = 1.0 * env.storage_loss
        loss /= 12.  # i have 12 steps per hour (ts =  5mins, losses are given in MW and capacity in MWh
        act = env.action_space()

        assert np.all(np.abs(obs.storage_charge - init_charge) <= self.tol_one), "wrong initial capacity"
        for nb_ts in range(8):
            obs, reward, done, info = env.step(act)
            assert np.all(np.abs(obs.storage_charge - (init_charge - (nb_ts +1) * loss)) <= self.tol_one), \
                f"wrong value computed for time step {nb_ts} (with loss in storage)"

        # now a loss should 'cap' the second battery
        obs, reward, done, info = env.step(act)
        th_storage = (init_charge - (nb_ts + 1) * loss)
        th_storage[0] -= loss[0]
        th_storage[1] = 0.
        assert np.all(np.abs(obs.storage_charge - th_storage) <= self.tol_one)
        for nb_ts in range(9):
            obs, reward, done, info = env.step(act)
            th_storage[0] -= loss[0]
            assert np.all(np.abs(obs.storage_charge - th_storage) <= self.tol_one), \
                f"capacity error for time step {nb_ts}"

        # all storages are empty
        obs, reward, done, info = env.step(act)
        assert np.all(np.abs(obs.storage_charge) <= self.tol_one), "error battery should be empty - 0"
        obs, reward, done, info = env.step(act)
        assert np.all(np.abs(obs.storage_charge) <= self.tol_one), "error, battery should be empty - 1"
        obs, reward, done, info = env.step(act)
        assert np.all(np.abs(obs.storage_charge) <= self.tol_one), "error, battery should be empty - 2"

    def test_env_storage_ambiguous(self):
        """test i cannot perform normal storage actions (no trick here) just normal actions"""
        pass