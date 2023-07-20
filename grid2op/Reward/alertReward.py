# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from typing import Optional
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float, dt_bool, dt_int


class AlertReward(BaseReward):
    """
    .. note::
        DOC IN PROGRESS !
    
    This reward is based on the "alert feature" where the agent is asked to send information about potential line overload issue
    on the grid after unpredictable powerline disconnection (attack of the opponent). The alerts are assessed once per attack.


    This rewards is computed as followed:
    
    - if an attack occurs and the agent survives `env.parameters.ALERT_TIME_WINDOW` steps then:
      - if the agent sent an alert BEFORE the attack, reward returns `reward_min_no_blackout` (-1 by default)
      - if the agent did not sent an alert BEFORE the attack, reward returns `reward_max_no_blackout` (1 by default)
    - if an attack occurs and the agent "games over" withing `env.parameters.ALERT_TIME_WINDOW` steps then:
      - if the agent sent an alert BEFORE the attack, reward returns `reward_max_blackout` (2 by default)
      - if the agent did not sent an alert BEFORE the attack, reward returns `reward_min_blackout` (-10 by default)
    - whatever the attacks / no attacks / alert / no alert, if the scenario is completed until the end, 
      then agent receive `reward_end_episode_bonus` (1 by default)

    In all other cases, including but not limited to:
    
    - agent games over but there has been no attack within the previous `env.parameters.ALERT_TIME_WINDOW` (12) steps
    - there is no attack 
    
    The reward outputs 0.
    
    This is then a "delayed reward": you receive the reward (in general) `env.parameters.ALERT_TIME_WINDOW` after
    having sent the alert.
    
    This is also a "sparse reward": in the vast majority of cases it's 0. It is only non zero in case of blackout (at
    most once per episode) and each time an attack occurs (and in general there is relatively few attacks)
    
    TODO explain a bit more in the "multi lines attacked"
    
    .. seealso:: :ref:`grid2op-alert-module` section of the doc for more information
    
    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import AlertReward

        # then you create your environment with it:
        # at time of writing, the only env supporting it is "l2rpn_idf_2023"
        NAME_OF_THE_ENVIRONMENT = "l2rpn_idf_2023"  
        
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT, reward_class=AlertReward)
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
        
        self.reward_min_no_blackout : float = dt_float(reward_min_no_blackout)
        self.reward_min_blackout : float = dt_float(reward_min_blackout)
        self.reward_max_no_blackout : float = dt_float(reward_max_no_blackout)
        self.reward_max_blackout : float = dt_float(reward_max_blackout)
        self.reward_end_episode_bonus : float = dt_float(reward_end_episode_bonus)
        self.reward_no_game_over : float = dt_float(0.0)
        
        self._reward_range_blackout : float = (self.reward_max_blackout - self.reward_min_blackout)

        self.total_time_steps : Optional[int] = dt_int(0)
        self.time_window : Optional[int] = None
        
        self._ts_attack : Optional[np.ndarray] = None
        self._current_id : int = 0
        self._lines_currently_attacked : Optional[np.ndarray] = None
        self._alert_launched : Optional[np.ndarray] = None
        self._nrows_array : Optional[int] = None
        
        self._i_am_simulate : bool = False

    
    def initialize(self, env: "grid2op.Environment.BaseEnv"):
        self.total_time_steps = env.max_episode_duration()
        self.time_window = env.parameters.ALERT_TIME_WINDOW
        self._nrows_array = self.time_window + 2
        self._ts_attack = np.full((self._nrows_array, type(env).dim_alerts), False, dtype=dt_bool)
        self._alert_launched = np.full((self._nrows_array, type(env).dim_alerts), False, dtype=dt_bool)
        self._current_id = 0
        self._lines_currently_attacked = np.full(type(env).dim_alerts, False, dtype=dt_bool)
        
        self._i_am_simulate = self.is_simulated_env(env)
        return super().initialize(env)      

    def reset(self, env):
        self._ts_attack[:,:] = False
        self._alert_launched[:,:] = False
        self._current_id = 0
        self._lines_currently_attacked[:] = False
        self._i_am_simulate = self.is_simulated_env(env)
        return super().reset(env)      
    
    def _update_attack(self, env):
        if env.infos["opponent_attack_line"] is None:
            # no attack at this step
            self._lines_currently_attacked[:] = False
            self._ts_attack[self._current_id, :] = False
        else:
            # an attack at this step
            lines_attacked = env.infos["opponent_attack_line"][type(env).alertable_line_ids]
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
        
        # update internal state of the environment
        # (this is updated in case the reward returns non 0)
        env._was_alert_used_after_attack[:] = 0
            
    def _compute_score_attack_blackout(self, env, ts_attack_in_order, indexes_to_look):
        # retrieve the lines that have been attacked in the time window
        ts_ind, line_ind = np.where(ts_attack_in_order)
        line_first_attack, first_ind_line_attacked = np.unique(line_ind, return_index=True)
        ts_first_line_attacked = ts_ind[first_ind_line_attacked]
        # now retrieve the array starting at the correct place
        ts_first_line_attacked_orig = indexes_to_look[ts_first_line_attacked]
        # and now look at the previous step if alerts were send
        # prev_ts = (ts_first_line_attacked_orig - 1) % self._nrows_array
        prev_ts = ts_first_line_attacked_orig
        # update the state of the environment
        env._was_alert_used_after_attack[line_first_attack] = self._alert_launched[prev_ts, line_first_attack] * 2 - 1
        return np.mean(self._alert_launched[prev_ts, line_first_attack]) * self._reward_range_blackout +  self.reward_min_blackout
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # retrieve the alert made by the agent
        res = 0.
        if self._i_am_simulate:
            # does not make sense for simulate
            return res
        
        if  is_done & (not has_error): 
            # end of episode, no blackout => reward specific for this case
            return self.reward_end_episode_bonus
        
        self._update_state(env, action)
        
        if self.is_in_blackout(has_error, is_done): 
            # I am in blackout, I need to check for attack in the time window
            # if there is no attack, I do nothing
            indexes_to_look = (np.arange(-self.time_window, 1) + self._current_id) % self._nrows_array  # include current step (hence the np.arange(..., **1**))
            ts_attack_in_order = self._ts_attack[indexes_to_look, :]
            has_attack = (ts_attack_in_order).any()
            if has_attack:
                # I need to check the alarm for the attacked lines
                res = self._compute_score_attack_blackout(env, ts_attack_in_order, indexes_to_look)
        else:
            # no blackout: i check the first step in the window before me to see if there is an attack,
            index_window = (self._current_id - self.time_window) % self._nrows_array
            lines_attack = self._ts_attack[index_window, :]
            if lines_attack.any():
                # prev_ind = (index_window - 1) % self._nrows_array
                # I don't need the "-1" because the action is already BEFORE the observation in the reward.
                prev_ind = index_window
                alert_send = self._alert_launched[prev_ind, lines_attack]
                # update the state of the environment
                env._was_alert_used_after_attack[lines_attack] = 1 - alert_send * 2
                res = (self.reward_min_no_blackout - self.reward_max_no_blackout) * np.mean(alert_send) + self.reward_max_no_blackout
                self._ts_attack[index_window, :] = False  # attack has been taken into account we "cancel" it
        return res
    