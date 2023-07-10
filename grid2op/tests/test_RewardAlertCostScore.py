# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import tempfile
import grid2op
from grid2op.Reward import _AlertCostScore, _AlertTrustScore
from grid2op.Agent import DoNothingAgent, BaseAgent
from grid2op.tests.helper_path_test import *
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner 
from grid2op.Observation import BaseObservation
from grid2op.Episode import EpisodeData
from grid2op.Parameters import Parameters
from grid2op.Opponent import BaseOpponent, GeometricOpponent
from grid2op.Action import BaseAction, PlayableAction


ALL_ATTACKABLE_LINES= [
            "62_58_180",
            "62_63_160",
            "48_50_136",
            "48_53_141",
            "41_48_131",
            "39_41_121",
            "43_44_125",
            "44_45_126",
            "34_35_110",
            "54_58_154",
        ] 

ATTACKED_LINE = "48_50_136"

def _get_steps_attack(kwargs_opponent, multi=False):
    """computes the steps for which there will be attacks"""
    ts_attack = np.array(kwargs_opponent["steps_attack"])
    res = []
    for i, ts in enumerate(ts_attack):
        if not multi:
            res.append(ts + np.arange(kwargs_opponent["duration"]))
        else:
            res.append(ts + np.arange(kwargs_opponent["duration"][i]))
    return np.unique(np.concatenate(res).flatten())

class AlertAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        if observation.current_step == 2:
            return self.action_space({"raise_alert": [0]})
        return super().act(observation, reward, done)
    
class TestOpponent(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked=[ATTACKED_LINE], duration=10, steps_attack=[0,1]):
        attacked_line = lines_attacked[0]
        self.custom_attack = self.action_space({"set_line_status" : [(l, -1) for l in lines_attacked]})
        self.duration = duration
        self.steps_attack = steps_attack
        self.env = partial_env

    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None
        current_step = self.env.nb_time_step
        if current_step not in self.steps_attack: 
            return None, None
        
        return self.custom_attack, self.duration


class TestAlertCostScore(unittest.TestCase):
    
    def test_specs(self):
        # test function without actual data
        assert _AlertCostScore._penalization_fun(50) == -1.
        assert _AlertCostScore._penalization_fun(80) == 0.
        assert _AlertCostScore._penalization_fun(100) == 1.
    
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        
    def tearDown(self) -> None:
        return super().tearDown()
        
    def test_assistant_reward_value_no_blackout_no_attack_no_alert(self) -> None : 
        """ When no blackout and no attack occur, and no alert is raised we expect a reward of 0
            until the end of the episode where we get the max reward 1.

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with grid2op.make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertCostScore
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, reward, done, info = env.step(env.action_space())
                if done:
                    assert reward == 1.
                else:
                    assert reward == 0.
                    
                    
class TestSimulate(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        self.env = grid2op.make(self.env_nm, test=True, difficulty="1",
                        reward_class=_AlertCostScore)
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_simulate(self):
        obs = self.env.reset()
        simO, simr, simd, simi = obs.simulate(self.env.action_space())
        assert simr == 0.
        assert not simd
        
        go_act = self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}})
        simO, simr, simd, simi = obs.simulate(go_act)
        assert simr == 0., f"{simr} vs 0."
        assert simd
        
    def test_simulated_env(self):
        obs = self.env.reset()
        f_env = obs.get_forecast_env()
        forD = False
        while not forD:
            forO, forR, forD, forI = f_env.step(self.env.action_space())
            assert forR == 0.
            
        f_env = obs.get_forecast_env()
        forD = False
        go_act = self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}})
        while not forD:
            forO, forR, forD, forI = f_env.step(go_act)
            assert forR == 0.
    
    
class TestRunnerAlertCost(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        self.env = grid2op.make(self.env_nm, test=True, difficulty="1",
                        reward_class=_AlertCostScore)
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_dn_agent(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 1. #it got to the end
    
    def test_simagent(self):
        #simulate blackout but act donothing
        obs = self.env.reset()
        
        class SimAgent(BaseAgent):
            def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
                go_act = self.action_space({"set_bus": {"generators_id": [(0, -1)]}})
                simO, simr, simd, simi = obs.simulate(go_act)
                simO, simr, simd, simi = obs.simulate(self.action_space())
                return super().act(observation, reward, done)
            
        runner = Runner(**self.env.get_params_for_runner(),
                        agentClass=SimAgent)
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 1.
        
    def test_episodeData(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0], add_detailed_output=True)
        assert res[0][2] == 1.
        assert res[0][5].rewards[8] == 1.
        
    def test_with_save(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        with tempfile.TemporaryDirectory() as f:
            res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0],
                             path_save=f)
            assert res[0][2] == 1.
            ep0, *_ = EpisodeData.list_episode(f)
            ep = EpisodeData.from_disk(*ep0)
            assert ep.rewards[8] == 1.
 

class TestAlertTrustScore(unittest.TestCase):  
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        
    def tearDown(self) -> None:
        return super().tearDown()
    
    def get_dn(self, env):
        return env.action_space({})
    
    def get_blackout(self, env):
        blackout_action = env.action_space({})
        blackout_action.gen_set_bus = [(0, -1)]
        return blackout_action   
    
    def test_assistant_reward_value_no_blackout_no_attack_no_alert(self) -> None : 
        """ When no blackout and no attack occur, and no alert is raised we expect a reward of 0
            until the end of the episode where we get the max reward 1 as score.

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with grid2op.make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, reward, done, info = env.step(env.action_space())
                if done:
                    assert reward == 1., f"{reward} vs 1."
                else:
                    assert reward == 0., f"{reward} vs 0."
                    
    def test_assistant_reward_value_blackout_attack_no_alert(self) -> None :
        """
        When 1 line is attacked at step 3 and we don't raise any alert
        and a blackout occur at step 4
        we expect a score of -10 at step 4 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with grid2op.make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent, 
                  reward_class=_AlertTrustScore,
                  _add_to_name="_tarvbana"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if step == 3 :
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    # When the blackout occurs, reward is -10 because we didn't raise an attack
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
                    
    def test_assistant_reward_value_blackout_attack_raise_good_alert(self) -> None :
        """
        When 1 line is attacked at step 3 and we do raise an alert
        and a blackout occur at step 4
        we expect a score of 2. at step 4 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with grid2op.make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore,
                  _add_to_name="_tarvbarga"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 2:
                    # I raise the alert (on the right line) just before the opponent attack
                    # opp attack at step = 3, so i = 2
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
                    
                    
class TestRunnerAlertTrust(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        self.env = grid2op.make(self.env_nm, test=True, difficulty="1",
                        reward_class=_AlertTrustScore)
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_dn_agent(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 1. #it got to the end
    
    def test_simagent(self):
        #simulate blackout but act donothing
        obs = self.env.reset()
        
        class SimAgent(BaseAgent):
            def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
                go_act = self.action_space({"set_bus": {"generators_id": [(0, -1)]}})
                simO, simr, simd, simi = obs.simulate(go_act)
                simO, simr, simd, simi = obs.simulate(self.action_space())
                return super().act(observation, reward, done)
            
        runner = Runner(**self.env.get_params_for_runner(),
                        agentClass=SimAgent)
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 1.
        
    def test_episodeData(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0], add_detailed_output=True)
        assert res[0][2] == 1.
        assert res[0][5].rewards[8] == 1.
        
    def test_with_save(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        with tempfile.TemporaryDirectory() as f:
            res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0],
                             path_save=f)
            assert res[0][2] == 1.
            ep0, *_ = EpisodeData.list_episode(f)
            ep = EpisodeData.from_disk(*ep0)
            assert ep.rewards[8] == 1.    
    
if __name__ == "__main__":
    unittest.main()        
