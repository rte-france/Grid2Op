# Copyright (c) 2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_float, dt_int
from grid2op.Exceptions import NotEnoughAttentionBudget


class LinearAttentionBudget:
    def __init__(self):
        self._max_budget = None
        self._budget_per_ts = None
        self._alarm_cost = None
        self._current_budget = None
        self._last_alarm_raised = dt_int(-1)
        self._init_budget = None
        self._last_successful_alarm_raised = dt_int(-1)

    def init(self, partial_env, init_budget, max_budget, budget_per_ts, alarm_cost, **kwargs):
        self._max_budget = dt_float(max_budget)
        self._budget_per_ts = dt_float(budget_per_ts)
        self._alarm_cost = dt_float(alarm_cost)
        self._init_budget = dt_float(init_budget)
        self.reset()

    def reset(self):
        """
        called each time the scenario is over by the environment

        Returns
        -------

        """
        self._current_budget = self._init_budget
        self._last_alarm_raised = dt_int(-1)
        self._last_successful_alarm_raised = dt_int(-1)

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
            if self._current_budget >= self._alarm_cost:
                # i could raise it
                self._current_budget -= self._alarm_cost
                self._last_successful_alarm_raised = env.nb_time_step
            else:
                # not enough budget
                current_budget = self._current_budget
                # self._current_budget = 0
                return NotEnoughAttentionBudget(f"You need a budget of {self._alarm_cost} to raise an alarm "
                                                f"but you had only {current_budget}. Nothing is done.")
        else:
            # no alarm has been raised, budget increases
            self._current_budget = min(self._max_budget, self._budget_per_ts + self._current_budget)
        return None
