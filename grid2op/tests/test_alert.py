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
import tempfile
from grid2op.tests.helper_path_test import *

from grid2op.operator_attention import LinearAttentionBudgetByLine
from grid2op import make
from grid2op.Reward import AlertReward
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData
from grid2op.Opponent import BaseOpponent, GeometricOpponent, GeometricOpponentMultiArea
from grid2op.Action import PlayableAction

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
        

    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None

        if not observation.current_step in self.steps_attack: 
            return None, None 
        
        return self.custom_attack, self.duration

class TestOpponentMultiLines(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked=[ATTACKED_LINE], duration=[10,10], steps_attack=[0,1]):
        attacked_line = lines_attacked[0]
        self.custom_attack = [ self.action_space({"set_line_status" : [(l, -1)]}) for l in lines_attacked]
        self.duration = duration
        self.steps_attack = steps_attack
        

    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None

        if not observation.current_step in self.steps_attack: 
            return None, None 
        
        index = self.steps_attack.index(observation.current_step)

        return self.custom_attack[index], self.duration[index]

# Test alert blackout / tets alert no blackout
class TestAlertNoBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no attack occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=10, 
                                   steps_attack=[0,10])
            self.env = make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent, 
                            reward_class=AlertReward(reward_end_episode_bonus=42))

        self.env.seed(0)
        self.env.reset()
        self.do_nothing = self.env.action_space({})

    def tearDown(self) -> None:
        self.env.close()

    def test_init_default_param(self) -> None : 
        env = make(self.env_nm, test=True, difficulty="1")
        assert isinstance(env.parameters.ALERT_TIME_WINDOW, np.int32)
        assert isinstance(env._attention_budget, LinearAttentionBudgetByLine)
        assert env._opponent_class == GeometricOpponent
        assert env.parameters.ALERT_TIME_WINDOW > 0

        param = env.parameters
        param.init_from_dict({
            "ALERT_TIME_WINDOW": -1, 
            "ASSISTANT_WARNING_TYPE": "BY_LINE"})
        
        negative_value_invalid = False
        try: 
            env.change_parameters(param)
            env.reset()
        except : 
            negative_value_invalid = True 

        assert negative_value_invalid

        # test observations for this env also
        true_alertable_lines = ALL_ATTACKABLE_LINES
        
        assert isinstance(env.alertable_line_names, list)
        assert sorted(env.alertable_line_names) == sorted(true_alertable_lines)
        assert env.dim_alerts == len(true_alertable_lines)


    def test_init_observation(self) -> None :
        true_alertable_lines = [ATTACKED_LINE]
        
        assert isinstance(self.env.alertable_line_names, list)
        assert sorted(self.env.alertable_line_names) == sorted(true_alertable_lines)
        assert self.env.dim_alerts == len(true_alertable_lines)


    def test_raise_alert_action(self) -> None :
        attackable_line_id = 0
        # raise alert on line number line_id
        act = self.env.action_space()
        act.raise_alert = [attackable_line_id]

        act_2 = self.env.action_space({"raise_alert": [attackable_line_id]})
        
        assert act == act_2 

    def test_assistant_reward_value_no_blackout_no_attack_no_alert(self) -> None : 
        """ When no blackout and no attack occur, and no alert is raised we expect a reward of 0
            until the end of the episode where we have a bonus (here artificially 42)

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, reward, done, info = env.step(self.do_nothing)
                if env._oppSpace.last_attack is None : 
                    if i == env.max_episode_duration()-1: 
                        assert reward == 42
                    else : 
                        assert reward == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done
    
    def test_assistant_reward_value_no_blackout_no_attack_alert(self) -> None : 
        """ When an alert is raised while no attack / nor blackout occur, we expect a reward of 0
            until the end of the episode where we have a bonus (here artificially 42)

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            attackable_line_id=0
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)

                if env._oppSpace.last_attack is None : 
                    if i == env.max_episode_duration()-1: 
                        assert reward == 42
                    else : 
                        assert reward == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done


# If attack 
    def test_assistant_reward_value_no_blackout_attack_no_alert(self) -> None :
        """ When we don't raise an alert for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[1])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                obs, reward, done, info = env.step(act)
                if i == 1 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 2 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert(self) -> None :
        """When an alert occur at step 2, we raise an alert is at step 1 
            We expect a reward -1 at step 3 (with a window size of 2)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == -1
                elif i == env.max_episode_duration()-1 : 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_late(self) -> None :
        """ When we raise an alert too late for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 2 :
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1 : 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_early(self)-> None :
        """ When we raise an alert too early for an attack but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 0 :
                    # An alert is raised at step 0
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1 : 
                    assert reward == 1
                else : 
                    assert reward == 0

    # 2 ligne attaquées 
    def test_assistant_reward_value_no_blackout_2_attack_same_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at the same time 
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[1])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                obs, reward, done, info = env.step(act)
                if i == 1 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 2 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_2_attack_same_time_1_alert(self) -> None :
        """ When we raise only 1 alert for 2 attacks at the same time 
            but no blackout occur, we expect a reward of 0
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == 0
                elif i == env.max_episode_duration()-1 : 
                    assert reward == 1
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_2_attack_same_time_2_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time 
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_ids = [0, 1]
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": attackable_line_ids})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == -1
                elif i == env.max_episode_duration()-1 : 
                    assert reward == 1
                else : 
                    assert reward == 0


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at two times  
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[1, 2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponentMultiLines, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                obs, reward, done, info = env.step(act)
                if i in [1,2] : 
                    assert env._oppSpace.last_attack is not None
                elif i == 2 : 
                    assert reward == 1
                elif i == 3 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0
        
    def test_assistant_reward_value_no_blackout_2_attack_diff_time_2_alert(self) -> None :
        """ When we raise 2 alert for 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponentMultiLines, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": [0]})
                elif i == 2 : 
                    act = self.env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)

                if i in [2,3] : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == -1
                elif i == 4 : 
                    assert reward == -1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_first_attack(self) -> None :
        """ When we raise 1 alert on the first attack while we have 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and 1 step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponentMultiLines, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                if i == 1 :
                    act = self.env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)

                if i in [2,3] : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == -1
                elif i == 4 : 
                    assert reward == 1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_second_attack(self) -> None :
        """ When we raise 1 alert on the second attack while we have 2 attacks at two times  
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2) and 1 step 4
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponentMultiLines, 
                            kwargs_opponent=kwargs_opponent,
                            reward_class=AlertReward(reward_end_episode_bonus=42)
            ) as env : 
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                act = self.do_nothing
                if i == 2 : 
                    act = self.env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)

                if i in [2,3] : 
                    assert env._oppSpace.last_attack is not None
                elif i == 3 : 
                    assert reward == 1
                elif i == 4 : 
                    assert reward == -1
                elif i == env.max_episode_duration()-1: 
                        assert reward == 42
                else : 
                    assert reward == 0

class TestAlertBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no blackout occur"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=10, 
                                   steps_attack=[0,10])
            self.env = make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent)
        self.env.seed(0)
        self.env.reset()
        self.do_nothing = self.env.action_space({})

    def tearDown(self) -> None:
        self.env.close()
# Cas avec blackout 1 ligne attaquée
# return -10
    def test_assistant_reward_value_blackout_attack_no_alert(self) -> None :
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 0 : 
                    # Disconnect line 48_53_141 to produce blackout
                    disconnected_lines_idx = [(env.name_line.tolist().index(l), -1) for l in ["48_53_141"]]
                    act = env.action_space({"set_line_status": disconnected_lines_idx})
                obs, reward, done, info = env.step(act)
                
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                    assert info["opponent_attack_line"] is not None # Equivalent to above
                    assert info["opponent_attack_line"][136] # Equivalent to above 
                
                if i == 4: 
                    # When the blackout occurs, reward is -10 because we didn't raise an attack
                    assert reward == -10
                else : 
                    assert reward == 0

                if done: 
                    break


# return 2
    def test_assistant_reward_value_blackout_attack_raise_good_alert(self) -> None :
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 0 : 
                    # Disconnect line 48_53_141 to produce blackout
                    disconnected_lines_idx = [(env.name_line.tolist().index(l), -1) for l in ["48_53_141"]]
                    act = env.action_space({"set_line_status": disconnected_lines_idx})
                elif i == 1:
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                
                if i == 4: 
                    assert reward == 2
                else : 
                    assert reward == 0

                if done: 
                    break


# return -10
    def test_assistant_reward_value_blackout_attack_raise_alert_just_before_blackout(self) -> None :
        """
        We test that if we raise an alert "too late" i.e. after an attack, the reward is -10.
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10
            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 0 : 
                    disconnected_lines_idx = [(env.name_line.tolist().index(l), -1) for l in ["48_53_141"]]
                    act = env.action_space({"set_line_status": disconnected_lines_idx})
                elif i == 3:
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None
                if i == 3 : 
                    assert reward == 1
                elif i == 4: 
                    assert reward == 2
                else : 
                    assert reward == 0


# return -10
    def test_assistant_reward_value_blackout_attack_raise_alert_too_early(self) -> None :
        """
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10
            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 0 : 
                    disconnected_lines_idx = [(env.name_line.tolist().index(l), -1) for l in ["48_53_141"]]
                    act = env.action_space({"set_line_status": disconnected_lines_idx})
                elif i == 3:
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                if i == 2 : 
                    assert env._oppSpace.last_attack is not None

                assert reward == 0


# return 2
    def  test_assistant_reward_value_blackout_2_lines_same_step_in_window_good_alerts(self) -> None :
        pass

# return 2
    def  test_assistant_reward_value_blackout_2_lines_different_step_in_window_good_alerts(self) -> None : 
        pass

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_simulaneous_only_1_alert(self) -> None:
        """
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE] + ["48_53_141"], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm, test=True, difficulty="1", 
                            opponent_attack_cooldown=0, 
                            opponent_attack_duration=99999, 
                            opponent_budget_per_ts=1000, 
                            opponent_init_budget=10000., 
                            opponent_action_class=PlayableAction, 
                            opponent_class=TestOpponent, 
                            kwargs_opponent=kwargs_opponent
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10
            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step_blackout = env.max_episode_duration()

            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.do_nothing
                if i == 3:
                    act = self.env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)

                if i == 2 : 
                    assert env._oppSpace.last_attack is not None

                if i == step_blackout : 
                    assert reward == -4
                else : 
                    assert reward == 0

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_first_attacked_line(self) -> None:
        pass

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_second_attacked_line(self) -> None:
        pass

# return 2 
    def test_assistant_reward_value_blackout_2_lines_attacked_different_1_in_window_1_good_alert(self) -> None:
        pass

# return 0 
    def test_assistant_reward_value_blackout_no_attack_alert(self) -> None :
        pass

# return 0 
    def test_assistant_reward_value_blackout_no_attack_no_alert(self) -> None :
        pass

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_alert(self) -> None :
        pass 

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_no_alert(self) -> None :
        pass 

# TODO : test des actions ambigues  
# Action ambigue : par exemple alert sur la ligne (nb_lignes)+1 
# Aller voir la doc : file:///home/crochepierrelau/Documents/Git/Grid2Op/documentation/html/action.html#illegal-vs-ambiguous

# TODO : test de l'attention budget