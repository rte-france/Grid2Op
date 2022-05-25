# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from dataclasses import dataclass
from numpy.random import shuffle

def random_order(agents : list, *args, **kwargs):
    """Returns the random order

    Args:
        agents (list): agents' names in the env

    Returns:
        list [int]: the execution order of agents
    """
    return shuffle(agents)

class AgentSelector:
    """
    Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    """
    #TODO
    
    def __init__(self, agents : list, agent_order_fn = lambda x : x, *args, **kwargs):
        self.agents = agents
        self.agent_order_fn = agent_order_fn
        self.reinit(*args, **kwargs)

    def reinit(self, *args, **kwargs):
        self.agent_order = self.agent_order_fn(self.agents, *args, **kwargs)
        self._current_agent = 0
        self.selected_agent = self.agent_order[0]

    def reset(self, *args, **kwargs):
        self.reinit(*args, **kwargs)
        return self.next()
    
    def get_order(self, reinit = False, *args, **kwargs):
        order = self.agent_order.copy() 
        if reinit :
            self.reinit(*args, **kwargs)
        return order

    def next(self):
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self):
        """
        Does not work as expected if you change the order
        """
        return self.selected_agent == self.agent_order[-1]

    def is_first(self):
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other):
        if not isinstance(other, AgentSelector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )
        
        
   
#@dataclass     
#class ActionDomain :
#    gen_id : list
#    load_id : list
#    line_id : list
#    substaion_id : list
#    storage_id : list
#    shunt_id : list
#
#@dataclass 
#class ObservationDomain :
#    storage_theta : list
#    gen_theta : list
#    load_theta : list
#    theta_ex : list
#    theta_or : list
#    line_status : list
#    topo_vect : list
#    timestep_overflow : list
#    rho : list
#    gen_p : list
#    gen_q : list
#    gen_v : list
#    load_p : list
#    load_q : list
#    load_v : list
#    p_or : list
#    q_or : list
#    v_or : list
#    a_or : list
#    p_ex : list
#    q_ex : list
#    v_ex : list
#    a_ex : list
#    rho : list
#    time_before_cooldown_line : list
#    time_before_cooldown_sub : list
#    time_next_maintenance : list
#    duration_next_maintenance : list
#    target_dispatch : list
#    actual_dispatch : list
#    _shunt_p : list
#    _shunt_q : list
#    _shunt_v : list
#    _shunt_bus : list
#    # storage
#    storage_charge : list
#    storage_power_target : list
#    storage_power : list
#    # curtailment
#    gen_p_before_curtail : list
#    curtailment : list
#    curtailment_limit : list
#    curtailment_limit_effective : list
#    curtailment_mw : list
#    # attention budget
#    time_since_last_alarm : list
#    last_alarm : list
#    # gen up / down
#    gen_margin_up : list
#    gen_margin_down : list
#    # Observable by default
#    year : bool = True
#    month : bool = True
#    day : bool = True
#    hour_of_day : bool = True
#    minute_of_hour : bool = True
#    day_of_week : bool = True
#    support_theta : bool = True
#    current_step : bool = True
#    max_step : bool = True
#    max_step : bool = True
#    time_since_last_alarm : bool = True
#    is_alarm_illegal : bool = True
#    attention_budget : bool = True
#    was_alarm_used_after_game_over : bool = True
#    
