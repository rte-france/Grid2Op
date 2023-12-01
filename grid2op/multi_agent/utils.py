# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from numpy.random import shuffle


def random_order(agents : list, *args, **kwargs):
    """Returns the random order

    Args:
        agents (list): agents' names in the env

    Returns:
        list [int]: the execution order of agents
    """
    # TODO BEN: remove that and use the space_prng of the Environment
    return shuffle(agents)


class AgentSelector:
    """
    Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    """
    #TODO
    
    def __init__(self, agents : list, agent_order_fn, *args, **kwargs):
        raise NotImplementedError("This function is not implemented at the moment.")
        self.agents = agents
        if agent_order_fn is None:
            agent_order_fn = lambda x : x
        self.agent_order_fn = agent_order_fn
        self.reinit(*args, **kwargs)

    def reinit(self, *args, **kwargs):
        self.agent_order = self.agent_order_fn(self.agents, *args, **kwargs)
        self._current_agent = 0
        self.selected_agent = self.agent_order[0]

    def reset(self, *args, **kwargs):
        self.reinit(*args, **kwargs)
        return self.next()
    
    def get_order(self, new_order = False, *args, **kwargs):
        order = self.agent_order.copy() 
        if new_order :
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
        