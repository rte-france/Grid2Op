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
        self._fist_attack_in_time_window = None # MaJ à chaque appel du call 

        self._has_line_alerts_in_time_window = False
        self._line_alerts_in_time_window = None # pas de temps en ligne et lignes élec en colonnes 
        
        # index à quel pas de temps on est 
        self.current_index = dt_int(-1)
        self.current_step_first_encountered = dt_int(-1) # à ne mettre à jour que lorsque le current_step passe au suivant (la première fois)
        # Pour ne MaJ qu'une ligne 

        # SImulate : MAJ le current index qu'en fonction du "current_step"
        # Vecteurs de taille time_window =
        self.unitary_reward_step = None
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


    def reset(self, env):
        self.total_time_steps = env.max_episode_duration()
        self.time_window = env.parameters.ALERT_TIME_WINDOW

        # Storing attacks in the past time window 
        self._has_attack_in_time_window = np.full(self.time_window, False, dtype=dt_bool)
        self._fist_attack_in_time_window = np.full(env.dim_alerts, False, dtype=dt_int) # time steps of the first attack
        self._attacks_in_time_window = np.full((env.dim_alerts, self.time_window), False, dtype=dt_bool)
        self._has_attack_at_first_time_in_window = np.full(self.time_window, False, dtype=dt_bool)

        # Storing alert in the past time window
        self._line_alerts_in_time_window = np.full((env.dim_alerts, self.time_window+1), False, dtype=dt_bool)
        self._has_line_alerts_in_time_window = np.full(self.time_window+1, -1, dtype=dt_int)
        
        # Current index 
        self.current_step_first_encountered = dt_int(-1)
        # self.current_index = dt_int(-1)

        self.unitary_reward_step = dt_float(4.0)
        self.nb_area
        
    def _step_update(self, legal_alert_action, env): 
        if env.nb_time_step == self.current_step_first_encountered+1:
            self.current_step_first_encountered = env.current_obs.current_step
            # self.current_index += 1 

        self._line_alerts_in_time_window[self.current_index % (self.time_window+1)] = legal_alert_action
        self._has_line_alerts_in_time_window = np.any(self._line_alerts_in_time_window, axis=1)
        
        # TODO voir avec Benjamin: 
        self._attacks_in_time_window[:-1] = self._attacks_in_time_window[1:]
        self._attacks_in_time_window[-1] = 
        self._fist_attack_in_time_window = 

        self._has_attack_in_time_window = np.where(self._attacks_in_time_window).any()
        self._has_attack_at_first_time_in_window = np.where(self._attacks_in_time_window[un_autre_index % self.time_window])

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        legal_alert_action = env._attention_budget._last_alert_action_filtered_by_budget
        self._step_update(legal_alert_action, env)

        score = self.reward_min
        # If blackout 
        if self.is_in_blackout(has_error, is_done): 
            # Is the blackout caused by an attack ? 
            # Has there been an attack in the last ``ALERT_TIME_WINDOW`` time steps ? 
            if self._has_attack_in_time_window.any() : 
                attacked_lines = np.where(self._fist_attack_in_time_window)[0]
                if self._has_line_alerts_in_time_window[attacked_lines].any():
                    nb_correct_alert = ().sum()

                    score = self.reward_max_blackout + nb_correct_alert * self.unitary_reward_step /self.nb_area ?
                    # SI on en prédit 2/3 on aurait - 2
                    # SI on en prédit 1/3 on aurait - 6
                    # SI on en prédit 1/3 on aurait - 10 
                    # paramètre sur le step entre X/nb_zone_l'opponent
                else : 
                    score = self.reward_min_blackout # ok 
        else: 
            if self._has_attack_at_first_time_in_window.any() : 
                attacked_lines = np.where(self._attacks_in_time_window[:,0])[0]
                if self._has_line_alerts_in_time_window[:,0][attacked_lines].any():
                    score = self.reward_min_no_blackout # ok
                else : 
                    score = self.reward_max_no_blackout # ok

        # env._oppSpace.last_attack # None lorsque pas d'attack et Action type 
        # Attention il faut récupérer l'info de 
        # action.get_topological_impact  return lines_attached, subs_attacked ? 
        # env.infos['opponent_attack_line'] vérifier si c'est le premier temps 
        # Update attack info at the end ? 
        return score
