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

def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if (item == vec[i]):
            return i
    return -1

class AlertReward(BaseReward):
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

    def __init__(self, logger=None, reward_min_no_blackout=-1.0, reward_min_blackout=-10.0, 
                 reward_max_no_blackout=1.0, reward_max_blackout=2.0, reward_end_episode_bonus=1.0):
        BaseReward.__init__(self, logger=logger)
        # required if you want to design a custom reward taking into account the
        # alert feature
        self.has_alert_component = True
        self.is_alert_used = False  # required to update it in __call__ !!

        
        self.reward_min_no_blackout = dt_float(reward_min_no_blackout)
        self.reward_min_blackout = dt_float(reward_min_blackout)
        self.reward_max_no_blackout = dt_float(reward_max_no_blackout)
        self.reward_max_blackout = dt_float(reward_max_blackout)
        self.reward_end_episode_bonus = dt_float(reward_end_episode_bonus)
        self.reward_no_game_over = dt_float(0.0)

        self.total_time_steps = dt_float(0.0)
        self.time_window = None

        self._has_attack_in_time_window = False # MaJ à chaque appel du call 
        self._fist_attack_step_in_time_window = None # MaJ à chaque appel du call 
        self._attacks_in_time_window = None # MaJ à chaque appel du call 
        self._has_attack_at_first_time_in_window = None # MaJ à chaque appel du call 
        self._is_first_step_of_attack = None # MaJ à chaque appel du call 

        self._has_line_alerts_in_time_window = False
        self._line_alerts_in_time_window = None # pas de temps en ligne et lignes élec en colonnes 
        
        # index à quel pas de temps on est 
        self.current_index = dt_int(-1)
        self.current_step_first_encountered = dt_int(-1) # à ne mettre à jour que lorsque le current_step passe au suivant (la première fois)
        # Pour ne MaJ qu'un pas de temps 

        self.reward_unit_step = dt_int(-1)
        
        self.last_attacked_lines = None
        self.has_new_attack = False

    def initialize(self, env):
        from grid2op.Environment._ObsEnv import (
            _ObsEnv,
        )  # to avoid circular dependencies

        # This reward is not compatible with simulations
        if isinstance(env, _ObsEnv):
            raise Grid2OpException(
                'Impossible to use the "AlertReward" with a simulation environment ``_ObsEnv``.' 
                'Please make sure "env._has_attention_budget" is set to ``True`` or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        if not env._has_attention_budget:
            raise Grid2OpException(
                'Impossible to use the "AlertReward" with an environment for which this feature '
                'is disabled. Please make sure "env._has_attention_budget" is set to ``True`` or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        if env.parameters.ASSISTANT_WARNING_TYPE != "BY_LINE":
            raise Grid2OpException(
                'Impossible to use the "AlertReward" with an environment for which this feature '
                'is ``ZONAL``. Please make sure "env.parameters.ASSISTANT_WARNING_TYPE" is set to ``BY_LINE`` or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        self.reset(env)

        if isinstance(env._opponent_class, GeometricOpponentMultiArea): 
            raise Grid2OpException("GeometricOpponentMultiArea is not handled by the alert feature")     

        self.total_time_steps = env.max_episode_duration()
        self.time_window = env.parameters.ALERT_TIME_WINDOW

        # Storing attacks in the past time window 
        self._has_attack_in_time_window = np.full(self.time_window, False, dtype=dt_bool)
        self._fist_attack_step_in_time_window = np.full(env.dim_alerts, -1, dtype=dt_int) # time steps of the first attack
        self._attacks_in_time_window = np.full((env.dim_alerts, self.time_window), False, dtype=dt_bool)
        self._has_attack_at_first_time_in_window = np.full(self.time_window, False, dtype=dt_bool)
        self._is_first_step_of_attack = np.full((env.dim_alerts, self.time_window), False, dtype=dt_bool)

        # Storing alert in the past time window
        self._line_alerts_in_time_window = np.full((env.dim_alerts, self.time_window+1), False, dtype=dt_bool)
        self._has_line_alerts_in_time_window = np.full(self.time_window+1, -1, dtype=dt_int)
        
        # Current index 
        self.current_step_first_encountered = dt_int(0)
                
    def _step_update(self, legal_alert_action, action, env): 
        """Update all 

        Args:
            legal_alert_action (Grid2opAction): RL action, where illegal actions are filtered by budget
            env (Grid2opEnv): RL environment

        Raises:
            Grid2OpException: if the opponent is of type GeometricOpponentMultiArea, has it is not handled by the alert feature
            Grid2OpException: Attacks on substations are not handled by the agent
        """
        # Get number of area 
        if isinstance(env._opponent_class, GeometricOpponentMultiArea): 
            self.nb_area = len(self._opponent.list_opponents)
            raise Grid2OpException("GeometricOpponentMultiArea is not handled by the alert feature")


        self._line_alerts_in_time_window[:, :-1] = self._line_alerts_in_time_window[:, 1:]
        self._line_alerts_in_time_window[:, -1] = legal_alert_action

        self._has_line_alerts_in_time_window = np.any(self._line_alerts_in_time_window, axis=0)
        
        
        if env.infos["opponent_attack_line"] is not None:
            # the opponent choose to attack
            lines_attacked = env.infos['opponent_attack_line'].take(env.alertable_lines_id)

            if self.last_attacked_lines.tolist() != lines_attacked.tolist() : 
                self._is_first_step_of_attack[:, :-1] = self._is_first_step_of_attack[:, 1:]
                self._is_first_step_of_attack[:, -1] = lines_attacked
                self.has_new_attack = True

            else : 
                self._is_first_step_of_attack[:, :-1] = self._is_first_step_of_attack[:, 1:]
                self._is_first_step_of_attack[:, -1] = np.full(env._attention_budget._dim_alerts, False, dtype=dt_bool)
                self.has_new_attack = False

            self.last_attacked_lines = lines_attacked

            self.nb_max_concurrent_attacks_in_window = sum(lines_attacked)
            self.reward_unit_step = (self.reward_max_blackout - self.reward_min_blackout) / self.nb_max_concurrent_attacks_in_window

        else:
            lines_attacked = np.full(env._attention_budget._dim_alerts, False, dtype=dt_bool)
            self._is_first_step_of_attack[:, :-1] = self._is_first_step_of_attack[:, 1:]
            self._is_first_step_of_attack[:, -1] = lines_attacked
            self.last_attacked_lines = lines_attacked
        
        

        self._attacks_in_time_window[:, :-1] = self._attacks_in_time_window[:, 1:]
        self._attacks_in_time_window[:, -1] = lines_attacked

        self._has_attack_in_time_window = np.bitwise_or.reduce(self._attacks_in_time_window, axis=1)

        self._fist_attack_step_in_time_window = self._get_fist_attack_step_in_time_window()

        self._has_attack_at_first_time_in_window = self._attacks_in_time_window[:,0]

    def _get_fist_attack_step_in_time_window(self): 
        """Function to get the first time step where an attack occur in the time window. 
           if no attack occur return -1

        Returns:
            first_attack_step_in_window: for reach line return the position of the attack in the time horizon

        Note : 
            return -1 if there is no attack in the time horizon on each line
        """
        # Initialize as if no attack happens
        first_attack_step_in_window = np.apply_along_axis(lambda x : find_first(True, x), 1, self._attacks_in_time_window)
        return first_attack_step_in_window

    def _get_nb_of_alerts_matching_an_attack(self): 
        attacked_lines_first_step =  self._fist_attack_step_in_time_window[self._fist_attack_step_in_time_window>=0]
        alert_on_attacked_lines =  self._line_alerts_in_time_window[self._fist_attack_step_in_time_window>=0]
        
        attacked_lines_with_alert = np.array([alert_on_attacked_lines[line, step] for line, step in enumerate(attacked_lines_first_step)])
        nb_correct_alert = attacked_lines_with_alert.sum()
        return nb_correct_alert


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        legal_alert_action = env._attention_budget._last_alert_action_filtered_by_budget

        if env.nb_time_step == self.current_step_first_encountered+1:
            self.current_step_first_encountered = env.current_obs.current_step
            self._step_update(legal_alert_action, action, env)

        score = self.reward_min

        if  is_done & (not has_error): 
            # end of episode 
            score = self.reward_end_episode_bonus
        elif self.is_in_blackout(has_error, is_done): 
            # If blackout
            # Is the blackout caused by an attack ? 
            # Has there been an attack in the last ``ALERT_TIME_WINDOW`` time steps ? 
            if self._has_attack_in_time_window.any() :
                
                nb_correct_alerts = self._get_nb_of_alerts_matching_an_attack()
                score = self.reward_min_blackout + self.reward_unit_step * nb_correct_alerts
                # If we correctly predict 2 out of 3 we get - 2
                # If we correctly predict  1 out of 3 we get - 6
                # If we correctly predict  0 out of 3 we get - 10 

        else: 
            # If there is no blackout 
            if self._has_attack_at_first_time_in_window.any() & self._is_first_step_of_attack.any():
                # If there is an attack

                alerts_with_attack_at_first_time_in_window =self._line_alerts_in_time_window[:,0][self._has_attack_at_first_time_in_window]
                # As there is no blackout, we do not want to raise any alert 
                if alerts_with_attack_at_first_time_in_window.any():
                    nb_alerts_for_attacks = alerts_with_attack_at_first_time_in_window.sum()
                    # If there is an alert raised for one of the attacked line, we give the minimal reward
                    nb_of_attacks = self._has_attack_at_first_time_in_window.sum()
                    score = self.reward_max_no_blackout - (nb_alerts_for_attacks/nb_of_attacks) * (self.reward_max_no_blackout  - self.reward_min_no_blackout)
                else : 
                    # If we don't raise any alert, we are happy with it, so we give the maximal reward value                    
                    score = self.reward_max_no_blackout
 
        return score
