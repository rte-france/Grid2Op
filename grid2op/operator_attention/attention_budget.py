# Copyright (c) 2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Exceptions import NotEnoughAttentionBudget


class LinearAttentionBudget:
    def __init__(self):
        self._max_budget = None
        self._budget_per_ts = None
        self._alarm_cost = None
        self._current_budget = None
        self._last_alarm_raised = -1

    def init(self, partial_env, max_budget, budget_per_ts, alarm_cost, **kwargs):
        self._max_budget = max_budget
        self._budget_per_ts = budget_per_ts
        self._alarm_cost = alarm_cost
        self._current_budget = max_budget

    def reset(self):
        """
        called each time the scenario is over by the environment

        Returns
        -------

        """
        self._current_budget = self.max_budget
        self._last_alarm_raised = -1

    def register_action(self, env, action):
        """
        INTERNAL

        Called at each step to update the budget according to the action played

        Parameters
        ----------
        env
        action

        Returns
        -------

        """
        if action.dim_alarms == 0:
            # this feature is not supported (grid2op <= 1.6.0) or is not activated
            return None

        if action.alarm_raised().size:
            # an alarm has been raised
            self._last_alarm_raised = env.nb_time_step
            if self._current_budget >= self.alarm_cost:
                # i could raise it
                self._current_budget -= self.alarm_cost
            else:
                # not enough budget
                current_budget = self._current_budget
                self._current_budget = 0
                return NotEnoughAttentionBudget(f"You need a budget of {self._alarm_cost} to raise an alarm "
                                                f"but you had only {current_budget}")
        else:
            # no alarm has been raised, budget increases
            self._current_budget = min(self._max_budget, self._budget_per_ts + self._current_budget)
        return None
