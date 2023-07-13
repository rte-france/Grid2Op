# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np
import copy

import grid2op
from grid2op.Action import BaseAction, DontAct, TopologyAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import BaseEnv, Environment
from grid2op.Observation import BaseObservation, CompleteObservation
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.Reward import FlatReward
from grid2op.Rules import AlwaysLegal
from grid2op.Runner import Runner
from grid2op.Chronics import FromOneEpisodeData, Multifolder, GridStateFromFileWithForecasts, GridStateFromFile, ChronicsHandler
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Exceptions import Grid2OpException

import pdb

from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.operator_attention import LinearAttentionBudget


class GameOverTestAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        if observation.current_step == 5:
            return self.action_space({"set_bus": {"loads_id": [(0, -1)]}})
        return self.action_space()


class SpecialChronicsHandler(ChronicsHandler):
    pass

class SpecialMultifolder(Multifolder):
    pass


class SpecialRunnerAddMaintenance(Runner):
    def __init__(self, *args, data_ref, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_ref = copy.deepcopy(data_ref)
        
    def init_env(special_runner_obj) -> BaseEnv:
        res =  super().init_env()
        res.chronics_handler.__class__ = SpecialChronicsHandler
        res.chronics_handler.real_data.__class__ = SpecialMultifolder
        
        def custom_initialize(multifolder_obj, 
                              order_backend_loads,
                              order_backend_prods,
                              order_backend_lines,
                              order_backend_subs,
                              names_chronics_to_backend=None):
            
            # use original implementation
            Multifolder.initialize(multifolder_obj,
                                   order_backend_loads,
                                   order_backend_prods,
                                   order_backend_lines,
                                   order_backend_subs,
                                   names_chronics_to_backend)
            # and then assign maintenance from the data ref
            max_iter = (multifolder_obj._max_iter + 1) if multifolder_obj._max_iter > 0 else 2018
            multifolder_obj.data.maintenance[:, :] = special_runner_obj.data_ref.maintenance[:max_iter, :]
            multifolder_obj.data.maintenance_time[:, :] = special_runner_obj.data_ref.maintenance_time[:max_iter, :]
            multifolder_obj.data.maintenance_duration[:, :] = special_runner_obj.data_ref.maintenance_duration[:max_iter, :]
        type(res.chronics_handler.real_data).initialize = custom_initialize
        return res


class TestTSFromEpisode(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = "l2rpn_idf_2023"  # with maintenance and attacks !
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name)
        self.env.set_id(0)
        self.env.seed(0)
        self.max_iter = 10
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_obs_equal(self, obs1, obs2, error_msg=""):
        assert np.abs(obs1.gen_p - obs2.gen_p).max() <= 1e-5, f"{error_msg}: {np.abs(obs1.gen_p - obs2.gen_p).max()}"
        assert np.array_equal(obs1.load_p, obs2.load_p), f"{error_msg}: {np.abs(obs1.load_p - obs2.load_p).max()}"
        assert np.array_equal(obs1.time_next_maintenance, obs2.time_next_maintenance), f"{error_msg}"
        assert np.array_equal(obs1.duration_next_maintenance, obs2.duration_next_maintenance), f"{error_msg}"
        assert np.abs(obs1.a_or - obs2.a_or).max() <= 1e-3, f"{error_msg}: {np.abs(obs1.a_or - obs2.a_or).max()}"
        
    def test_basic(self):
        """test injection, without opponent nor maintenance"""
        obs = self.env.reset()
        runner = Runner(
            **self.env.get_params_for_runner()
        )
        res = runner.run(nb_episode=1, max_iter=self.max_iter, add_detailed_output=True)
        ep_data = res[0][-1]
        env = grid2op.make(self.env_name,
                           chronics_class=FromOneEpisodeData,
                           data_feeding_kwargs={"ep_data": ep_data})
        obs = env.reset()
        self._aux_obs_equal(obs,  ep_data.observations[0])
        for i in range(10):
            obs, reward, done, info = env.step(env.action_space())
            self._aux_obs_equal(obs,  ep_data.observations[i+1], f"at it. {i}")
        assert done
        with self.assertRaises(Grid2OpException):
            obs, reward, done, info = env.step(env.action_space())
            
        # again :-)
        obs = env.reset()
        self._aux_obs_equal(obs,  ep_data.observations[0], "after reset")
        for i in range(10):
            obs, reward, done, info = env.step(env.action_space())
            self._aux_obs_equal(obs,  ep_data.observations[i+1], f"at it. {i} (after reset)")
        assert done
    
    def test_when_game_over(self):
        """test I can load from a runner that used an agent that games over"""
        obs = self.env.reset()
        runner = Runner(
            **self.env.get_params_for_runner(),
            agentClass=GameOverTestAgent
        )
        res = runner.run(nb_episode=1, max_iter=self.max_iter, add_detailed_output=True)
        ep_data = res[0][-1]
        env = grid2op.make(self.env_name,
                           chronics_class=FromOneEpisodeData,
                           data_feeding_kwargs={"ep_data": ep_data})
        obs = env.reset()
        self._aux_obs_equal(obs,  ep_data.observations[0])
        for i in range(6):
            obs, reward, done, info = env.step(env.action_space())
            self._aux_obs_equal(obs,  ep_data.observations[i+1])
        assert done
    
    def test_maintenance(self):
        """test the maintenance are correct"""
        obs = self.env.reset()
        
        # hack for generating maintenance
        dataref = copy.deepcopy(self.env.chronics_handler.real_data.data)
        dataref.maintenance[2:7, 2] = True
        
        for line_id in range(dataref.n_line):
            dataref.maintenance_time[:, line_id] = dataref.get_maintenance_time_1d(
                dataref.maintenance[:, line_id]
            )
            dataref.maintenance_duration[
                :, line_id
            ] = dataref.get_maintenance_duration_1d(dataref.maintenance[:, line_id])
        env_dict_params = self.env.get_params_for_runner()
        env_dict_params["data_ref"] = dataref
        runner = SpecialRunnerAddMaintenance(
            **env_dict_params
        )
        # now run as usual
        res = runner.run(nb_episode=1, max_iter=self.max_iter, add_detailed_output=True)
        ep_data = res[0][-1]
        assert len(ep_data) == self.max_iter, f"{len(ep_data)} vs {self.max_iter}"
        assert ep_data.observations[2].time_next_maintenance[2] == 0  # check right maintenance is applied
        assert ep_data.observations[2].duration_next_maintenance[2] == 5  # check right maintenance is applied
        env = grid2op.make(self.env_name,
                           chronics_class=FromOneEpisodeData,
                           data_feeding_kwargs={"ep_data": ep_data})
        obs = env.reset()
        self._aux_obs_equal(obs,  ep_data.observations[0], f"after reset")
        for i in range(10):
            obs, reward, done, info = env.step(env.action_space())
            self._aux_obs_equal(obs,  ep_data.observations[i+1], f"at it. {i}")
        assert done
        
        # now with an agent that games over
        runner2 = SpecialRunnerAddMaintenance(
            **env_dict_params,
            agentClass=GameOverTestAgent
        )
        # now run as usual
        res2 = runner2.run(nb_episode=1, max_iter=self.max_iter, add_detailed_output=True)
        ep_data2 = res2[0][-1]
        assert len(ep_data2) == 7, f"{len(ep_data2)} vs 7"
        assert ep_data2.observations[2].time_next_maintenance[2] == 0  # check right maintenance is applied
        assert ep_data2.observations[2].duration_next_maintenance[2] == 5  # check right maintenance is applied
        env = grid2op.make(self.env_name,
                           chronics_class=FromOneEpisodeData,
                           data_feeding_kwargs={"ep_data": ep_data2})
        obs = env.reset()
        self._aux_obs_equal(obs,  ep_data2.observations[0], f"after reset (after game over)")
        for i in range(6):
            obs, reward, done, info = env.step(env.action_space())
            self._aux_obs_equal(obs,  ep_data2.observations[i+1], f"at it. {i} (after game over)")
        assert done
        
        
if __name__ == "__main__":
    unittest.main()