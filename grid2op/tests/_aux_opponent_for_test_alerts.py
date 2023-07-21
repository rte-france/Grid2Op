# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy
from grid2op.Opponent import BaseOpponent


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
             lines_attacked,
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
    
    
class TestOpponent(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked, duration=10, steps_attack=[0,1]):
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


class TestOpponentMultiLines(BaseOpponent): 
    """An opponent that can select the line attack, the time and duration of the attack."""
    
    def __init__(self, action_space):
        super().__init__(action_space)
        self.custom_attack = None
        self.duration = None
        self.steps_attack = None

    def init(self, partial_env,  lines_attacked, duration=[10,10], steps_attack=[0,1]):
        attacked_line = lines_attacked[0]
        self.custom_attack = [ self.action_space({"set_line_status" : [(l, -1)]}) for l in lines_attacked]
        self.duration = duration
        self.steps_attack = steps_attack
        self.env = partial_env
        
    def attack(self, observation, agent_action, env_action, budget, previous_fails): 
        if observation is None:
            return None, None

        current_step = self.env.nb_time_step
        if current_step not in self.steps_attack: 
            return None, None
        
        index = self.steps_attack.index(current_step)

        return self.custom_attack[index], self.duration[index]
