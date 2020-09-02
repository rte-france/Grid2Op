# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Agent.GreedyAgent import GreedyAgent


class PowerLineSwitch(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to disconnect powerlines.

    It will choose among:

      - doing nothing
      - changing the status of one powerline

    which action that will maximize the simulated reward. All powerlines are tested at each steps. This means
    that if `n` is the number of powerline on the grid, at each steps this actions will perform `n` +1
    calls to "simulate" (one to do nothing and one that change the status of each powerline)

    """

    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)

    def _get_tested_action(self, observation):
        res = [self.action_space({})]  # add the do nothing
        for i in range(self.action_space.n_line):
            tmp = np.full(self.action_space.n_line, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.action_space({"change_line_status": tmp})
            if not observation.line_status[i]:
                # so the action consisted in reconnecting the powerline
                # i need to say on which bus (always on bus 1 for this type of agent)
                action = action.update({"set_bus": {"lines_or_id": [(i, 1)], "lines_ex_id": [(i, 1)]}})
            res.append(action)
        return res
