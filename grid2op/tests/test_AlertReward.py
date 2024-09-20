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
from grid2op.Observation import BaseObservation
from grid2op.tests.helper_path_test import *

from grid2op import make
from grid2op.Reward import AlertReward
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner  # TODO
from grid2op.Action import BaseAction, PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Episode import EpisodeData
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from _aux_opponent_for_test_alerts import (_get_steps_attack,
                                           TestOpponent,
                                           TestOpponentMultiLines)

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

DEFAULT_ALERT_REWARD_PARAMS = dict(reward_min_no_blackout=-1.0,
                                   reward_min_blackout=-10.0, 
                                   reward_max_no_blackout=1.0,
                                   reward_max_blackout=2.0,
                                   reward_end_episode_bonus=42.0)


# Test alert blackout / tets alert no blackout
class TestAlertNoBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when no attack occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )

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
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, reward, done, info = env.step(env.action_space())
                if info["opponent_attack_line"] is None : 
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
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            attackable_line_id=0
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1

                if info["opponent_attack_line"] is None : 
                    if step == env.max_episode_duration(): 
                        assert reward == 42
                    else : 
                        assert reward == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done
# If attack 
    def test_assistant_reward_value_no_blackout_attack_no_alert(self) -> None :
        """ When we don't raise an alert for an attack (at step 1)
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnbana"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert(self) -> None :
        """When an alert occur at step 2, we raise an alert at step 1 
            We expect a reward -1 at step 3 (with a window size of 2)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnba"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                    assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_late(self) -> None :
        """ When we raise an alert too late for an attack (at step 2) but no blackout occur, 
            we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnbaatl"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 2 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                    assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_attack_alert_too_early(self)-> None :
        """ When we raise an alert too early for an attack (at step 2)
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnbaate"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 0 :
                    # An alert is raised at step 0
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4: 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                    assert reward == 42
                else : 
                    assert reward == 0

    # 2 ligne attaquées 
    def test_assistant_reward_value_no_blackout_2_attack_same_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at the same time (step 1)
            but no blackout occur, we expect a reward of 1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2astna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0
    
    def test_assistant_reward_value_no_blackout_2_attack_same_time_1_alert(self) -> None :
        """ When we raise only 1 alert for 2 attacks at the same time (step 2)
            but no blackout occur, we expect a reward of 0
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2ast1a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 0
                elif step == env.max_episode_duration(): 
                    assert reward == 42
                else : 
                    assert reward == 0

    def test_assistant_reward_value_no_blackout_2_attack_same_time_2_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time (step 2)
            but no blackout occur, we expect a reward of -1
            at step 3 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2ast2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_ids = [0, 1]
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": attackable_line_ids})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                    assert reward == 42
                else : 
                    assert reward == 0


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at two times resp. (steps 1 and 2)
            but no blackout occur, we expect a reward of 1
            at step 3 and 4 (here with a window size of 2)
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[1, 2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2dtna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True) : 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 3 : 
                    assert reward == 1
                elif step == 4 : 
                    assert reward == 1
                elif step == env.max_episode_duration(): 
                        assert reward == 42
                else : 
                    assert reward == 0
        
    def test_assistant_reward_value_no_blackout_2_attack_diff_time_2_alert(self) -> None :
        """ When we raise 2 alert for 2 attacks at two times (step 2 and 3)  
            but no blackout occur, we expect a reward of -1
            at step 4 (here with a window size of 2) and step 5
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2dt2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [0]})
                elif step == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4: 
                    assert reward == -1, f"error for step {step}: {reward} instead of -1."
                elif step == 5: 
                    assert reward == -1
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} instead of -1."
                else : 
                    assert reward == 0, f"error for step {step}: {reward} instead of 0."

    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_first_attack(self) -> None :
        """ When we raise 1 alert on the first attack while we have 2 attacks at two times (steps 2 and 3)
            but no blackout occur, we expect a reward of -1
            at step 4 (here with a window size of 2) and 1 at step 5
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2dtafa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == -1, f"error for step {step}: {reward} vs -1"
                elif step == 5 : 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"


    def test_assistant_reward_value_no_blackout_2_attack_diff_time_alert_second_attack(self) -> None :
        """ When we raise 1 alert on the second attack while we have 2 attacks at two times (steps 2 and 3)
            but no blackout occur, we expect a reward of 1
            at step 4 (here with a window size of 2) and -1 step 5
            otherwise 0 at other time steps
            until the end of the episode where we have a bonus (here artificially 42)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvnb2dtasa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"
                elif step == 5 : 
                    assert reward == -1, f"error for step {step}: {reward} vs -1"
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"


    def test_raise_illicit_alert(self) -> None:
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()
            assert type(env).dim_alerts == 10, f"dim_alerts: {type(env).dim_alerts} instead of 10"
            attackable_line_id = 10
            try : 
                act = env.action_space({"raise_alert": [attackable_line_id]})
            except Grid2OpException as exc_ : 
                assert exc_.args[0] == ('Impossible to modify the alert with your input. Please consult the '
                                        'documentation. The error was:\n"Grid2OpException IllegalAction '
                                        '"Impossible to change a raise alert id 10 because there are only '
                                        '10 on the grid (and in python id starts at 0)""')


class TestAlertBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when a blackout occur"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
    
    def get_dn(self, env):
        return env.action_space({})

    def get_blackout(self, env):
        blackout_action = env.action_space({})
        blackout_action.gen_set_bus = [(0, -1)]
        return blackout_action

# Cas avec blackout 1 ligne attaquée
# return -10
    def test_assistant_reward_value_blackout_attack_no_alert(self) -> None :
        """
        When 1 line is attacked at step 3 and we don't raise any alert
        and a blackout occur at step 4
        we expect a reward of -10 at step 4 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent, 
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
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
# return 2
    def test_assistant_reward_value_blackout_attack_raise_good_alert(self) -> None :
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
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

# return -10
    def test_assistant_reward_value_blackout_attack_raise_alert_just_before_blackout(self) -> None :
        """
        When 1 line is attacked at step 3 and we raise 1 alert  too late
        we expect a reward of -10 at step 4 when the blackout occur 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvbarajbb"
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
                elif i == 1:
                    # opponent attack at step 3, so when i = 2
                    # i raise the alert BEFORE that (so when i = 1)
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"
                
    def test_assistant_reward_value_blackout_attack_raise_alert_too_early(self) -> None :
        """
        When 1 line is attacked at step 3 and we raise 1 alert  too early
        we expect a reward of -10 at step 4 when the blackout occur 
        """
        # return -10
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvbarate"
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
                elif i == 1:
                    # opp attacks at step = 3, so i = 2, I raise an alert just before
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -10, f"error for step {step}: {reward} vs -10"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return 2
    def  test_assistant_reward_value_blackout_2_lines_same_step_in_window_good_alerts(self) -> None :
        """
        When 2 lines are attacked simustaneously at step 2 and we raise 2 alert 
        we expect a reward of 2 at the blackout (step 4)
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[3, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvb2lssiwga"
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
                    # attack at step 3, so when i = 2 (which is the right time to send an alert)
                    act = env.action_space({"raise_alert": [0,1]})
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert done
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_simulaneous_only_1_alert(self) -> None:
        """
        When 2 lines are attacked simustaneously at step 2 and we raise only 1 alert 
        we expect a reward of -4 when the blackout occur at step 4 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[3, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  
                  _add_to_name="_tarvb2laso1a"
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
                    # attack at step 3, so i = 2, which is the 
                    # right time to send an alert
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 4: 
                    assert reward == -4, f"error for step {step}: {reward} vs -4"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return 2
    def  test_assistant_reward_value_blackout_2_lines_different_step_in_window_good_alerts(self) -> None : 
        """
        When 2 lines are attacked at different steps 3 and 4 and we raise 2  alert 
        we expect a reward of 2 when the blackout occur at step 5 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvb2ldsiwga"
            ) as env : 
            env.seed(0)
            obs = env.reset()
            step = 0            
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 2 :
                    # opp attack "line 0" at step 3 so i = 2 => good alert
                    act = env.action_space({"raise_alert": [0]})
                elif i == 3 : 
                    # opp attack "line 1" at step 4 so i = 3 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 4 : 
                    # trigger blackout
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 5 : 
                    assert done  # blackout
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                    break
                elif step == env.max_episode_duration(): 
                    assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_first_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 3 and 4 and we raise 1 alert on the first attack
        we expect a reward of -4 on blackout at step 4 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvb2ladsiwo1aofal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):  
                act = self.get_dn(env)
                if i == 2 :
                    # opp attack "line 0" at step 3 so i = 2 => good alert
                    act = env.action_space({"raise_alert": [0]})
                elif i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1  # i = step - 1 at this stage
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 4 : 
                    assert done  # blackout
                    assert reward == -4, f"error for step {step}: {reward} vs -4"  # bug in Laure too
                    break
                elif step == env.max_episode_duration(): 
                        assert reward == 42, f"error for step {step}: {reward} vs 42"
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return -4
    def test_assistant_reward_value_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_second_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 2 and 3 and we raise 1 alert on the second attack
        we expect a reward of -4 at step 5 when the blackout occur 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvb2ladsiwo1aosal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 :
                    # opp attack "line 1" at step 4 so i = 3 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 4 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if step == 5 : 
                    assert reward == -4., f"error for step {step}: {reward} vs -4"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return 2 
    def test_assistant_reward_value_blackout_2_lines_attacked_different_1_in_window_1_good_alert(self) -> None:
        """
        When 2 lines are attacked at different steps 3 and 6 and we raise 1 alert on the second attack
        we expect a reward of 1 at step 5 and 2 at step 7 when the blackout happen 
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[3, 6])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                  _add_to_name="_tarvb2lad1iw1ga"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 5 :
                    # opp attack "line 1" at step 6 so i = 5 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 6 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                step += 1

                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if step == 5: 
                    assert reward == 1, f"error for step {step}: {reward} vs 1"  # no blackout here
                elif step == 7 : 
                    assert reward == 2, f"error for step {step}: {reward} vs 2"
                    assert done
                    break
                else : 
                    assert reward == 0, f"error for step {step}: {reward} vs 0"

# return 0 
    def test_assistant_reward_value_blackout_no_attack_alert(self) -> None :

        """Even if there is a blackout, an we raise an alert
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_no_attack_no_alert(self) -> None :
        """Even if there is a blackout, an we don't raise an alert
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too early
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i in [0, 1, 2]:
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_reward_value_blackout_attack_before_window_no_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too late
           we expect a reward of 0 because there is no attack"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 4:
                    # we never go here ...
                    act = env.action_space({"raise_alert": [0]})
                obs, reward, done, info = env.step(act)
                
                if info["opponent_attack_line"] is None : 
                    assert reward == 0.
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done


class TestSimulate(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        self.env = make(self.env_nm, test=True, difficulty="1",
                        reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS))
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
    
    
class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
        self.env = make(self.env_nm, test=True, difficulty="1",
                        reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS))
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_dn_agent(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 42
    
    def test_simagent(self):
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
        assert res[0][2] == 42
        
    def test_episodeData(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0], add_detailed_output=True)
        assert res[0][2] == 42
        assert res[0][5].rewards[8] == 42
        
    def test_with_save(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner())
        with tempfile.TemporaryDirectory() as f:
            res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0],
                             path_save=f)
            assert res[0][2] == 42
            ep0, *_ = EpisodeData.list_episode(f)
            ep = EpisodeData.from_disk(*ep0)
            assert ep.rewards[8] == 42
            
    def test_with_opp(self):
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        env = make(self.env_nm,
                   test=True,
                   difficulty="1", 
                   opponent_attack_cooldown=0, 
                   opponent_attack_duration=99999, 
                   opponent_budget_per_ts=1000, 
                   opponent_init_budget=10000., 
                   opponent_action_class=PlayableAction, 
                   opponent_class=TestOpponent, 
                   kwargs_opponent=kwargs_opponent,
                   reward_class=AlertReward(**DEFAULT_ALERT_REWARD_PARAMS),
                   _add_to_name = "_test_with_opp")
        # without alert
        runner = Runner(**env.get_params_for_runner())
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 43, f"{res[0][2]} vs 43"
        
        class AlertAgent(BaseAgent):
            def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
                if observation.current_step == 2:
                    return self.action_space({"raise_alert": [0]})
                return super().act(observation, reward, done)
            
        # with a wrong alert
        runner = Runner(**env.get_params_for_runner(), agentClass=AlertAgent)
        res = runner.run(nb_episode=1, episode_id=[0], max_iter=10, env_seeds=[0])
        assert res[0][2] == 41, f"{res[0][2]} vs 41"
        

if __name__ == "__main__":
    unittest.main()