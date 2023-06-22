# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float, dt_bool, dt_int
from grid2op.Opponent import GeometricOpponentMultiArea


class AlertRewardNew(BaseReward):
    """
    This reward is based on the "alert feature" where the agent is asked to send information about potential line overload issue
    on the grid.

    On this case, when the environment is in a "game over" state (eg it's the end) then the reward is computed
    the following way:

    - if the environment has been successfully manage until the end of the chronics, and no attack occurs then 1.0 is returned
    - if an alarm has been raised and a attack occurs before the end of the chronics, then 2.0 is returned
    - if an alarm has been raised, and no attack occurs then -1.0 is return
    - if no alarm has been raised, and a attack occurs then -10.0 is return


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import AlertReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=AlertReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the AlertReward class


    """

    def __init__(self,
                 logger=None,
                 reward_min_no_blackout=-1.0,
                 reward_min_blackout=-10.0, 
                 reward_max_no_blackout=1.0,
                 reward_max_blackout=2.0,
                 reward_end_episode_bonus=1.0):
        BaseReward.__init__(self, logger=logger)
        
        self.reward_min_no_blackout = dt_float(reward_min_no_blackout)
        self.reward_min_blackout = dt_float(reward_min_blackout)
        self.reward_max_no_blackout = dt_float(reward_max_no_blackout)
        self.reward_max_blackout = dt_float(reward_max_blackout)
        self.reward_end_episode_bonus = dt_float(reward_end_episode_bonus)
        self.reward_no_game_over = dt_float(0.0)
        
        self._reward_range_blackout = (self.reward_max_blackout - self.reward_min_blackout)

        self.total_time_steps = dt_int(0.0)
        self.time_window = None
        
        self._ts_attack : np.ndarray = None
        self._current_id : int = 0
        self._lines_currently_attacked : np.ndarray = None
        self._alert_launched : np.ndarray = None
        self._nrows_array : int = None

    def initialize(self, env: "grid2op.Environment.BaseEnv"):
        self.total_time_steps = env.max_episode_duration()
        self.time_window = env.parameters.ALERT_TIME_WINDOW
        self._nrows_array = self.time_window + 2
        
        # TODO simulate env stuff !
        
        # TODO vectors proper size
        self._ts_attack = np.full((self._nrows_array, type(env).dim_alerts), False, dtype=dt_bool)
        self._alert_launched = np.full((self._nrows_array, type(env).dim_alerts), False, dtype=dt_bool)
        self._current_id = 0
        self._lines_currently_attacked = np.full(type(env).dim_alerts, False, dtype=dt_bool)
        return super().initialize(env)      
    
    def _update_attack(self, env):
        if env.infos["opponent_attack_line"] is None:
            # no attack at this step
            self._lines_currently_attacked[:] = False
            self._ts_attack[self._current_id, :] = False
        else:
            # an attack at this step
            lines_attacked = env.infos["opponent_attack_line"][env.alertable_lines_id]
            # compute the list of lines that are "newly" attacked
            new_lines_attacked = lines_attacked & (~self._lines_currently_attacked)
            # remember the steps where these lines are attacked
            self._ts_attack[self._current_id, new_lines_attacked] = True
            # and now update the state of lines under attack
            self._lines_currently_attacked[:] = False
            self._lines_currently_attacked[lines_attacked] = True
    
    def _update_alert(self, action):
        self._alert_launched[self._current_id, :] = 1 * action.raise_alert
        
    def _update_state(self, env, action):
        self._current_id += 1
        self._current_id %= self._nrows_array
        
        # update attack
        self._update_attack(env)
        
        # update alerts
        self._update_alert(action)
            
    def _compute_score_attack_blackout(self, ts_attack_in_order, indexes_to_look):
        # retrieve the lines that have been attacked in the time window
        ts_ind, line_ind = np.where(ts_attack_in_order)
        line_first_attack, first_ind_line_attacked = np.unique(line_ind, return_index=True)
        ts_first_line_attacked = ts_ind[first_ind_line_attacked]
        # now retrieve the array starting at the correct place
        ts_first_line_attacked_orig = indexes_to_look[ts_first_line_attacked]
        # and now look at the previous step if alerts were send
        prev_ts = (ts_first_line_attacked_orig - 1) % self._nrows_array
        return np.mean(self._alert_launched[prev_ts, line_first_attack]) * self._reward_range_blackout +  self.reward_min_blackout
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # retrieve the alert made by the agent
        res = 0.
        
        if  is_done & (not has_error): 
            # end of episode, no blackout => reward specific for this case
            return self.reward_end_episode_bonus
        
        self._update_state(env, action)
        
        if self.is_in_blackout(has_error, is_done): 
            # I am in blackout, I need to check for attack in the time window
            # if there is no attack, I do nothing
            # indexes_to_look = (np.arange(-self.time_window, 0) + self._current_id) % self._nrows_array
            # indexes_to_look = (np.arange(-(self.time_window + 1), -1) + self._current_id) % self._nrows_array
            indexes_to_look = (np.arange(-self.time_window, 1) + self._current_id) % self._nrows_array  # include current step
            ts_attack_in_order = self._ts_attack[indexes_to_look, :]
            has_attack = np.any(ts_attack_in_order)
            if has_attack:
                # I need to check the alarm for the attacked lines
                res = self._compute_score_attack_blackout(ts_attack_in_order, indexes_to_look)
        else:
            # no blackout: i check the first step in the window before me to see if there is an attack,
            index_window = (self._current_id - self.time_window) % self._nrows_array
            lines_attack = self._ts_attack[index_window, :]
            if np.any(lines_attack):
                # prev_ind = (index_window - 1) % self._nrows_array
                # I don't need the "-1" because the action is already BEFORE the observation in the reward.
                prev_ind = index_window
                alert_send = self._alert_launched[prev_ind, lines_attack]
                res = (self.reward_min_no_blackout - self.reward_max_no_blackout) * np.mean(alert_send) + self.reward_max_no_blackout
        return res
    