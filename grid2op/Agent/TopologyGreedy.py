# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Agent.GreedyAgent import GreedyAgent


class TopologyGreedy(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to reconfigure the substations connectivity.

    It will choose among:

      - doing nothing
      - changing the topology of one substation.

    To choose, it will simulate the outcome of all actions, and then chose the action leading to the best rewards.

    """
    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)
        self.li_actions = None

    def _get_tested_action(self, observation):
        if self.li_actions is None:
            res = [self.action_space({})]  # add the do nothing
            # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
            # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
            res += self.action_space.get_all_unitary_topologies_set(self.action_space)
            self.li_actions = res
        return self.li_actions

