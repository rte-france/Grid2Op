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
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,
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
        
        index = self.steps_attack.index(current_step)

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
                            _add_to_name="_tio")

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

    def test_init_observation(self) -> None :    
        obs : BaseObservation = self.env.reset()
        import pdb
        pdb.set_trace()
        assert obs.last_alert
        
        # TODO => make it vect
        assert obs.was_alert_used_after_attack is False
        
        # TODO => make it vect
        assert obs.time_since_last_alert
        
        # TODO create the "duration" (0 for no alert, and then += 1 per time alerted)
        
        # TODO => make it vect
        assert obs.is_alert_illegal

    def test_raise_alert_action(self) -> None :
        """test i can raise an alert on all attackable lines"""
        env = self.env
        for attackable_line_id in range(env.dim_alerts):
            # raise alert on line number line_id
            act = env.action_space()
            act.raise_alert = [attackable_line_id]
            act_2 = env.action_space({"raise_alert": [attackable_line_id]})
            assert act == act_2, f"error for line {attackable_line_id}"
                    
# TODO test that even if an action is illegal, the "alert" part is not
# replace by "do nothing"

                    
if __name__ == "__main__":
    unittest.main()
