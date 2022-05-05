# Copyright (c) 2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
import numpy as np

from grid2op.dtypes import dt_float, dt_int, dt_bool
from grid2op.Exceptions import NotEnoughAttentionBudget


class LinearAttentionBudget:
    def __init__(self):
        self._max_budget = None
        self._budget_per_ts = None
        self._alarm_cost = None
        self._current_budget = None
        self._init_budget = None
        self._time_last_alarm_raised = dt_int(-1)
        self._time_last_successful_alarm_raised = dt_int(-1)
        self._last_alarm_raised = None
        self._last_successful_alarm_raised = None
        self._all_successful_alarms = []

    @property
    def time_last_alarm_raised(self):
        """
        Time step of the last alarm raised (-1 if no alarm has been raised yet)
        Returns
        -------

        """
        return self._time_last_alarm_raised

    @property
    def time_last_successful_alarm_raised(self):
        """time of the last successful alarm raised"""
        return self._time_last_successful_alarm_raised

    @property
    def current_budget(self):
        """current attention budget"""
        return self._current_budget

    @property
    def last_alarm_raised(self):
        """
        for each zone, says:

          - -1 if no alarm have been raised for this zone for the entire episode
          - `k` (with `k>0`) says that the last alarm raised for this zone was at step `k`

        .. note::
            This counts both successful and non successful alarms
        """
        return self._last_alarm_raised

    @property
    def last_successful_alarm_raised(self):
        """
        for each zone, says:

          - -1 if no alarm have been raised for this zone for the entire episode
          - `k` (with `k>0`) says that the last alarm raised for this zone was at step `k`

        .. note::
            This counts only successful alarms

        """
        return self._last_successful_alarm_raised

    def init(
        self, partial_env, init_budget, max_budget, budget_per_ts, alarm_cost, **kwargs
    ):
        self._max_budget = dt_float(max_budget)
        self._budget_per_ts = dt_float(budget_per_ts)
        self._alarm_cost = dt_float(alarm_cost)
        self._init_budget = dt_float(init_budget)
        self._last_alarm_raised = np.empty(partial_env.dim_alarms, dtype=dt_int)
        self._last_successful_alarm_raised = np.empty(
            partial_env.dim_alarms, dtype=dt_int
        )
        self.reset()

    def reset(self):
        """
        called each time the scenario is over by the environment

        Returns
        -------

        """
        self._current_budget = self._init_budget
        self._time_last_alarm_raised = dt_int(-1)
        self._time_last_successful_alarm_raised = dt_int(-1)
        self._last_alarm_raised[:] = -1
        self._last_successful_alarm_raised[:] = -1
        self._all_successful_alarms = []

    def get_state(self):
        """used to retrieve the sate in simulate"""
        res = (
            self._time_last_alarm_raised,
            self._last_alarm_raised,
            self._current_budget,
            self._time_last_successful_alarm_raised,
            self._last_successful_alarm_raised,
            self._all_successful_alarms,
        )
        return res

    def set_state(self, state):
        """used to update the internal state of the budget, for simulate"""
        (
            _time_last_alarm_raised,
            _last_alarm_raised,
            _current_budget,
            _time_last_successful_alarm_raised,
            _last_successful_alarm_raised,
            _all_successful_alarms,
        ) = state

        self._time_last_alarm_raised = _time_last_alarm_raised
        self._last_alarm_raised[:] = _last_alarm_raised
        self._current_budget = _current_budget
        self._time_last_successful_alarm_raised = _time_last_successful_alarm_raised
        self._last_successful_alarm_raised[:] = _last_successful_alarm_raised
        self._all_successful_alarms = copy.copy(_all_successful_alarms)

    def register_action(self, env, action, is_action_illegal, is_action_ambiguous):
        """
        INTERNAL

        Called at each step to update the budget according to the action played

        Parameters
        ----------
        env
        action
        is_action_illegal
        is_action_ambiguous

        Returns
        -------

        """
        if action.dim_alarms == 0 or is_action_illegal or is_action_ambiguous:
            # this feature is not supported (grid2op <= 1.6.0) or is not activated

            # also, if the action is illegal is ambiguous, it is replaced with do nothing, but i don't really
            # want to affect the budget on this case
            return None

        if action.alarm_raised().size:
            # an alarm has been raised
            self._time_last_alarm_raised = env.nb_time_step
            self._last_alarm_raised[action.raise_alarm] = env.nb_time_step
            if self._current_budget >= self._alarm_cost:
                # i could raise it
                self._current_budget -= self._alarm_cost
                self._time_last_successful_alarm_raised = env.nb_time_step
                self._last_successful_alarm_raised[
                    action.raise_alarm
                ] = env.nb_time_step
                self._all_successful_alarms.append(
                    (env.nb_time_step, copy.deepcopy(action.raise_alarm))
                )
            else:
                # not enough budget
                current_budget = self._current_budget
                # self._current_budget = 0
                return NotEnoughAttentionBudget(
                    f"You need a budget of {self._alarm_cost} to raise an alarm "
                    f"but you had only {current_budget}. Nothing is done."
                )
        else:
            # no alarm has been raised, budget increases
            self._current_budget = min(
                self._max_budget, self._budget_per_ts + self._current_budget
            )
        return None
