# Copyright (c) 2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np


class LinearAttentionBudget:
    def __init__(self, max_budget, budget_per_ts, alarm_cost):
        self.max_budget = max_budget
        self.budget_per_ts = budget_per_ts
        self.alarm_cost = alarm_cost
        self.current_budget = max_budget
        self.last_alarm_raised = -1

    def reset(self):
        """
        called each time the scenario is over by the environment

        Returns
        -------

        """
        self.current_budget = self.max_budget
        self.last_alarm_raised = -1

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
            return None

        if action.alarm_raised().size:
            # an alarm has been raised
            self.last_alarm_raised = env.nb_time_step
            if self.current_budget >= self.alarm_cost:
                # i could raise it
                self.current_budget -= self.alarm_cost
            else:
                # no more budget
                # TODO
                pass
        else:
            self.current_budget = min(self.max_budget, self.budget_per_ts + self.current_budget)
        return None
