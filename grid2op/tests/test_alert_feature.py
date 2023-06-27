# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import os
import copy
import tempfile
from grid2op.Observation import BaseObservation
from grid2op.tests.helper_path_test import *

from grid2op import make
from grid2op.Reward import AlertReward
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner  # TODO
from grid2op.Opponent import BaseOpponent, GeometricOpponent
from grid2op.Action import BaseAction, PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Episode import EpisodeData


ALL_ATTACKABLE_LINES = [
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


class OpponentForTestAlert(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.env = None  
        self.lines_attacked = None  
        self.custom_attack = None  
        self.attack_duration = None  
        self.attack_steps = None  
        self.attack_id = None  

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        new_obj.env = dict_["partial_env"]    
        new_obj.lines_attacked = copy.deepcopy(self.lines_attacked)
        new_obj.custom_attack = [act.copy() for act in self.custom_attack]
        new_obj.attack_duration = copy.deepcopy(self.attack_duration)
        new_obj.attack_steps = copy.deepcopy(self.attack_steps)
        new_obj.attack_id = copy.deepcopy(self.attack_id)
        return super()._custom_deepcopy_for_copy(new_obj, dict_)
    
    def init(self,
             partial_env,
             lines_attacked=ALL_ATTACKABLE_LINES,
             attack_duration=[],
             attack_steps=[],
             attack_id=[]):
        self.lines_attacked = lines_attacked
        self.custom_attack = [ self.action_space({"set_line_status" : [(l, -1)]}) for l in attack_id]
        self.attack_duration = attack_duration
        self.attack_steps = attack_steps
        self.attack_id = attack_id
        self.env = partial_env
        
    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None
        current_step = self.env.nb_time_step
        if current_step not in self.attack_steps: 
            return None, None
        index = self.attack_steps.index(current_step)
        return self.custom_attack[index], self.attack_duration[index]


# Test alert blackout / tets alert no blackout
class TestAction(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no attack occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        kwargs_opponent = dict(lines_attacked=ALL_ATTACKABLE_LINES)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm,
                            test=True,
                            difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=OpponentForTestAlert, 
                            kwargs_opponent=kwargs_opponent, 
                            reward_class=AlertReward(reward_end_episode_bonus=42),
                            _add_to_name="_tafta")

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_init_default_param(self) -> None : 
        env = self.env
        assert isinstance(env.parameters.ALERT_TIME_WINDOW, np.int32)
        assert env.parameters.ALERT_TIME_WINDOW > 0

        param = env.parameters
        param.init_from_dict({"ALERT_TIME_WINDOW": -1})
        
        negative_value_invalid = False
        try: 
            env.change_parameters(param)
            env.reset()
        except Exception as exc_: 
            negative_value_invalid = True 

        assert negative_value_invalid

        # test observations for this env also
        true_alertable_lines = ALL_ATTACKABLE_LINES
        
        assert isinstance(env.alertable_line_names, list)
        assert sorted(env.alertable_line_names) == sorted(true_alertable_lines)
        assert env.dim_alerts == len(true_alertable_lines)

    def test_raise_alert_action(self) -> None :
        """test i can raise an alert on all attackable lines"""
        env = self.env
        for attackable_line_id in range(env.dim_alerts):
            # raise alert on line number line_id
            act = env.action_space()
            act.raise_alert = [attackable_line_id]
            act_2 = env.action_space({"raise_alert": [attackable_line_id]})
            assert act == act_2, f"error for line {attackable_line_id}"
        
        for attackable_line_id in range(env.dim_alerts): 
            for attackable_line_id2 in range(env.dim_alerts):     
                if attackable_line_id2 == attackable_line_id:
                    continue
                act = env.action_space()
                act.raise_alert = [attackable_line_id, attackable_line_id2]
                act_2 = env.action_space({"raise_alert": [attackable_line_id2, attackable_line_id]})
                assert act == act_2, f"error for line {attackable_line_id}"
                assert act._raise_alert[attackable_line_id]
                assert act._raise_alert[attackable_line_id2]
                assert np.sum(act._raise_alert) == 2
                assert act._modif_alert


# Test alert blackout / tets alert no blackout
class TestObservation(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no attack occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        kwargs_opponent = dict(lines_attacked=ALL_ATTACKABLE_LINES)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm,
                            test=True,
                            difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=OpponentForTestAlert, 
                            kwargs_opponent=kwargs_opponent, 
                            reward_class=AlertReward(reward_end_episode_bonus=42),
                            _add_to_name="_tafto")
        param = self.env.parameters
        param.ALERT_TIME_WINDOW = 2
        self.env.change_parameters(param)
        assert type(self.env).dim_alerts == len(ALL_ATTACKABLE_LINES)

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_obs_init(self, obs):
        assert np.all(obs.last_alert == False)
        assert np.all(obs.time_since_last_alert == -1)
        assert np.all(obs.time_since_last_attack == -1)
        assert np.all(obs.alert_duration == 0)
        assert obs.total_number_of_alert == 0
        assert np.all(obs.was_alert_used_after_attack == False)
        
    def test_init_observation(self) -> None :    
        obs : BaseObservation = self.env.reset()
        self._aux_obs_init(obs)
    
    def _aux_alert_0(self, obs):
        assert obs.last_alert[0]
        assert obs.last_alert.sum() == 1
        assert obs.time_since_last_alert[0] == 0
        assert np.all(obs.time_since_last_alert[1:] == -1)
        assert np.all(obs.time_since_last_attack == -1)
        assert np.all(obs.alert_duration[1:] == 0)
        assert obs.alert_duration[0] == 1
        assert obs.total_number_of_alert == 1 
        assert np.all(obs.was_alert_used_after_attack == False)
    
    def test_after_action(self):
        obs : BaseObservation = self.env.reset()
        
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        self._aux_alert_0(obs)
        
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0, 1]}))
        assert np.all(obs.last_alert[:2])
        assert obs.last_alert.sum() == 2
        assert obs.time_since_last_alert[0] == 0
        assert obs.time_since_last_alert[1] == 0
        assert np.all(obs.time_since_last_alert[2:] == -1)
        assert np.all(obs.time_since_last_attack == -1)
        assert obs.alert_duration[0] == 2
        assert obs.alert_duration[1] == 1
        assert np.all(obs.alert_duration[2:] == 0)
        assert obs.total_number_of_alert == 3
        assert np.all(obs.was_alert_used_after_attack == False)
        
        obs : BaseObservation = self.env.reset()
        self._aux_obs_init(obs)
        
    def test_illegal_action(self):
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0],
                                                                       "set_bus": {"lines_or_id": [(0, 2), (19, 2)]}}))
        assert info["is_illegal"]
        self._aux_alert_0(obs)
    
    def test_ambiguous_action_nonalert(self):
        obs : BaseObservation = self.env.reset()
        act = self.env.action_space({"raise_alert": [0]})
        
        # now create an ambiguous action that is accepted by grid2op (which is not that easy)
        act._set_topo_vect[0] = 2  # ambiguous because the flag has not been modified !
        obs, reward, done, info = self.env.step(act)
        assert info["is_ambiguous"]
        self._aux_alert_0(obs)
    
    def test_ambiguous_action_alert(self):
        obs : BaseObservation = self.env.reset()
        act = self.env.action_space({"raise_alert": [0]})
        
        # now create an ambiguous action that is accepted by grid2op (which is not that easy)
        act._modif_alert = False  # ambiguous because the flag has not been modified !
        obs, reward, done, info = self.env.step(act)
        assert info["is_ambiguous"]
        self._aux_obs_init(obs)
        
    def test_env_cpy(self):
        obs : BaseObservation = self.env.reset()
        env_cpy = self.env.copy()
        
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        self._aux_alert_0(obs)
        
        obs_cpy, *_ = env_cpy.step(self.env.action_space())
        self._aux_obs_init(obs_cpy)

    def test_when_attacks(self):
        obs : BaseObservation = self.env.reset()
        
        # tell the opponent to make an attack
        attack_id = np.where(self.env.name_line == ALL_ATTACKABLE_LINES[0])[0][0]
        opp = self.env._oppSpace.opponent
        opp.custom_attack = [opp.action_space({"set_line_status" : [(l, -1)]}) for l in [attack_id]]
        opp.attack_duration = [1]
        opp.attack_steps = [2]
        opp.attack_id = [attack_id]
        
        # 1 no game over
        # 1a alert on the wrong line, the was_alert_used_after_attack[0] should be 1
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [1]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert reward == 1., f"{reward} vs 1."
        assert obs.was_alert_used_after_attack[1] == 0
        assert obs.was_alert_used_after_attack[0] == 1  # used (even if I did not sent an alarm, which by default means 'no alarm')
        assert np.all(obs.was_alert_used_after_attack[2:] == 0)
        # 1b alert on the right line (which is wrong), the was_alert_used_after_attack[0] should be -1
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert reward == -1., f"{reward} vs -1."
        assert obs.was_alert_used_after_attack[0] == -1
        assert np.all(obs.was_alert_used_after_attack[1:] == 0)
        
        # 2 game over
        # 2a no alert on the line
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [1]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}}))
        assert done
        assert reward == -10., f"{reward} vs -10."
        assert obs.was_alert_used_after_attack[1] == 0
        assert obs.was_alert_used_after_attack[0] == -1, f"{obs.was_alert_used_after_attack[0]} vs -1"  # used (even if I did not sent an alarm, which by default means 'no alarm')
        assert np.all(obs.was_alert_used_after_attack[2:] == 0)
        
        # 2b alert on the line (game over in window)
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}}))
        assert done
        assert reward == 2., f"{reward} vs 2."
        assert obs.was_alert_used_after_attack[1] == 0
        assert obs.was_alert_used_after_attack[0] == 1, f"{obs.was_alert_used_after_attack[0]} vs 1"
        assert np.all(obs.was_alert_used_after_attack[2:] == 0)
        
        # 2c alert on the line (game over still in window)
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}}))
        assert done
        assert reward == 2., f"{reward} vs 2."
        assert obs.was_alert_used_after_attack[1] == 0
        assert obs.was_alert_used_after_attack[0] == 1, f"{obs.was_alert_used_after_attack[0]} vs 1"
        assert np.all(obs.was_alert_used_after_attack[2:] == 0)

        # 2c alert on the line (game over out of the window)
        obs : BaseObservation = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"raise_alert": [0]}))
        assert info["opponent_attack_line"] is not None
        assert info["opponent_attack_line"][attack_id]
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"generators_id": [(0, -1)]}}))
        assert done
        assert reward == 0., f"{reward} vs 2."
        assert obs.was_alert_used_after_attack[1] == 0
        assert obs.was_alert_used_after_attack[0] == 0, f"{obs.was_alert_used_after_attack[0]} vs 0"
        assert np.all(obs.was_alert_used_after_attack[2:] == 0)

# TODO test the update_obs_after_reward in the runner !

# TODO test "as_dict" and "as_json"
        
if __name__ == "__main__":
    unittest.main()
