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

from numba import jit 
@jit(nopython=True)
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

    def __init__(self, logger=None, reward_min_no_blackout=-1.0, reward_min_blackout=-4.0, 
                 reward_max_no_blackout=1.0, reward_max_blackout=2.0):
        BaseReward.__init__(self, logger=logger)
        # required if you want to design a custom reward taking into account the
        # alarm feature
        self.has_alert_component = True
        self.is_alert_used = False  # required to update it in __call__ !!

        
        self.reward_min_no_blackout = dt_float(reward_min_no_blackout)
        self.reward_min_blackout = dt_float(reward_min_blackout)
        self.reward_max_no_blackout = dt_float(reward_max_no_blackout)
        self.reward_max_blackout = dt_float(reward_max_blackout)
        self.reward_no_game_over = dt_float(0.0)

        self.total_time_steps = dt_float(0.0)
        self.time_window = None

        self._has_attack_in_time_window = False # MaJ à chaque appel du call 
        self._fist_attack_step_in_time_window = None # MaJ à chaque appel du call 

        self._has_line_alerts_in_time_window = False
        self._line_alerts_in_time_window = None # pas de temps en ligne et lignes élec en colonnes 
        
        # index à quel pas de temps on est 
        self.current_index = dt_int(-1)
        self.current_step_first_encountered = dt_int(-1) # à ne mettre à jour que lorsque le current_step passe au suivant (la première fois)
        # Pour ne MaJ qu'une ligne 

        # SImulate : MAJ le current index qu'en fonction du "current_step"
        # Vecteurs de taille time_window =
        self.reward_unit_step = (self.reward_max_blackout - self.reward_min_blackout) / self.nb_max_simultaneous_attacks 
        self.nb_area = None

    def initialize(self, env):
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

        self.nb_max_simultaneous_attacks = 1 
        if isinstance(env._opponent_class, GeometricOpponentMultiArea): 
            raise Grid2OpException("GeometricOpponentMultiArea is not handled by the alert feature")
            # TODO : self.nb_max_simultaneous_attacks = len(self._opponent.list_opponents) # equal number of areas    def reset(self, env):

        self.total_time_steps = env.max_episode_duration()
        self.time_window = env.parameters.ALERT_TIME_WINDOW

        # Storing attacks in the past time window 
        self._has_attack_in_time_window = np.full(self.time_window, False, dtype=dt_bool)
        self._fist_attack_step_in_time_window = np.full(env.dim_alerts, -1, dtype=dt_int) # time steps of the first attack
        self._attacks_in_time_window = np.full((env.dim_alerts, self.time_window), False, dtype=dt_bool)
        self._has_attack_at_first_time_in_window = np.full(self.time_window, False, dtype=dt_bool)

        # Storing alert in the past time window
        self._line_alerts_in_time_window = np.full((env.dim_alerts, self.time_window+1), False, dtype=dt_bool)
        self._has_line_alerts_in_time_window = np.full(self.time_window+1, -1, dtype=dt_int)
        
        # Current index 
        self.current_step_first_encountered = dt_int(0)
                
    def _step_update(self, legal_alert_action, env): 
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
        
        
        if env._oppSpace.last_attack is not None:
            # the opponent choose to attack
            # i update the "cooldown" on these things
            lines_attacked, subs_attacked = legal_alert_action.get_topological_impact()

            if sum(subs_attacked) > 0 : 
                raise Grid2OpException("Attacks on substations are not handled by the agent")
        else:
            lines_attacked = np.full(env._attention_budget._dim_alerts, False, dtype=dt_bool)

        # TODO valider avec Benjamin: 
        self._attacks_in_time_window[:, :-1] = self._attacks_in_time_window[:, 1:]
        self._attacks_in_time_window[:, -1] = lines_attacked

        # For debug 
        self._attacks_in_time_window[5, 0] = True
        self._attacks_in_time_window[4, 0] = True

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


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        legal_alert_action = env._attention_budget._last_alert_action_filtered_by_budget

        # TODO Gérer le "simulate"
        if env.nb_time_step == self.current_step_first_encountered+1:
            self.current_step_first_encountered = env.current_obs.current_step
            self._step_update(legal_alert_action, env)

        score = self.reward_min
        # If blackout 

        # TODO remove for debug 
        has_error = True
        is_done = True

        if self.is_in_blackout(has_error, is_done): 
            # Is the blackout caused by an attack ? 
            # Has there been an attack in the last ``ALERT_TIME_WINDOW`` time steps ? 
            if self._has_attack_in_time_window.any() :
                attacked_lines_first_step =  self._fist_attack_step_in_time_window[self._fist_attack_step_in_time_window>=0]
                alert_on_attacked_lines =  self._line_alerts_in_time_window[self._fist_attack_step_in_time_window>=0]
                
                attacked_lines_with_alert = np.take(alert_on_attacked_lines, attacked_lines_first_step)
                nb_correct_alert = attacked_lines_with_alert.sum()

                score = self.reward_min_blackout + self.reward_unit_step * nb_correct_alert  
                # SI on en prédit 2/3 on aurait - 2
                # SI on en prédit 1/3 on aurait - 6
                # SI on en prédit 0/3 on aurait - 10 
        else: 
            
            # If there is an attack
            if self._has_attack_at_first_time_in_window.any() :

                # As there is no blackout, we do not want to raise any alert 
                
                # TODO remove to test behaviour
                #self._line_alerts_in_time_window[:,0] = np.array([False, False, False, False, True, False, False, False, False,False])

                if self._line_alerts_in_time_window[:,0][self._has_attack_at_first_time_in_window].any():
                    # If there is an alert raised for one of the attacked line, we give the minimal reward
                    score = self.reward_min_no_blackout 

                else : 
                    # If we don't raise any alert, we are happy with it, so we give the maximal reward value                    
                    score = self.reward_max_no_blackout
 
        # TODO Gérer la simulation env.infos['opponent_attack_line'] vérifier si c'est le premier temps 
        return score
