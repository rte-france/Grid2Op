# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward import AlertReward
from grid2op.dtypes import dt_float

class _AlertTrustScore(AlertReward):
    """

    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be **MAXIMIZED**,
            as it is a negative! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this reward is based on the "alert feature" 
    where the agent is asked to send information about potential line overload issue on the grid after unpredictable powerline
    disconnection (attack of the opponent).
    The alerts are assessed once per attack and cumulated over the episode. In this scheme, this "reward" computed the assistant "score",
    which assesses how well the agent is aware of its capacities to deal with a situation during an episode.
    It should not be used to train an agent.
    
    """

    def __init__(self,
                 logger=None,
                 reward_min_no_blackout=-1.0,
                 reward_min_blackout=-50,
                 reward_max_no_blackout=0.0,
                 reward_max_blackout=0.0,
                 reward_end_episode_bonus=0.0,
                 min_score=-3):

        super().__init__(logger,
                 reward_min_no_blackout,
                 reward_min_blackout, 
                 reward_max_no_blackout,
                 reward_max_blackout,
                 reward_end_episode_bonus)
        
        self.min_score = dt_float(min_score)
        self.max_score = dt_float(1.0)
        
    def initialize(self, env):
        self._is_simul_env = self.is_simulated_env(env)
        #SURVIVOR_TIMESTEPS=env._al
            
        self.cumulated_reward = 0
        #KPIs
        self.total_nb_attacks = 0
        self.nb_last_attacks = 0 #attacks in the AlertTimeWindow before done

        #TODO
        #self.total_nb_alerts = 0
        #self.alert_attack_no_blackout = 0
        #self.alert_attack_blackout = 0
        #self.no_alert_attack_no_blackout = 0
        #self.no_alert_attack_blackout = 0
        #self.blackout_encountered = False
        return super().initialize(env)
            
    def reset(self, env):
        super().reset(env)
        self.cumulated_reward = 0
        #KPIs
        self.total_nb_attacks = 0
        self.nb_last_attacks = 0

        # TODO
        #self.total_nb_alerts = 0
        #self.alert_attack_no_blackout = 0
        #self.alert_attack_blackout = 0
        #self.no_alert_attack_no_blackout = 0
        #self.no_alert_attack_blackout = 0
        #self.blackout_encountered = False
            
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        score_ep = 0.
        if self._is_simul_env:
            return score_ep

        self.blackout_encountered = self.is_in_blackout(has_error, is_done)
        
        score_ep = 0.
        if self._is_simul_env:
            return score_ep
            
        res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        self.cumulated_reward += res

        lines_attacked = env._time_since_last_attack == 0
        self.total_nb_attacks += lines_attacked.sum()

        # TODO
        #lines_alerted_beforeattack = np.equal(env._time_since_last_alert, env._time_since_last_attack + 1) and lines_attacked
        #self.total_nb_alerts += np.sum(lines_alerted_beforeattack)
            
        if not is_done:

            # TODO
            #lines_attacked_no_blackout = env._time_since_last_attack == SURVIVOR_TIMESTEPS
#
            #self.alert_attack_no_blackout += np.sum(lines_alerted_beforeattack[lines_attacked_no_blackout])
            #self.no_alert_attack_no_blackout += np.sum(~lines_alerted_beforeattack[lines_attacked_no_blackout])
            
            return score_ep
            
        else:
            self.nb_last_attacks=np.sum(self._ts_attack)
            cm_reward_min_ep, cm_reward_max_ep = self._compute_min_max_reward(self.total_nb_attacks,self.nb_last_attacks)
            score_ep = self._normalisation_fun(self.cumulated_reward, cm_reward_min_ep, cm_reward_max_ep,self.min_score,self.max_score)

            # TODO
            #if self.blackout_encountered:
            #    lines_attacked_dangerzone = (env._time_since_last_attack >= 0) * (env._time_since_last_attack < SURVIVOR_TIMESTEPS)
            #
            #    self.alert_attack_blackout += 1. * any(lines_alerted_beforeattack[lines_attacked_dangerzone])
            #    self.no_alert_attack_blackout += 1. * any(~lines_alerted_beforeattack[lines_attacked_dangerzone])
            #else :
            #    lines_attacked_no_blackout = env._time_since_last_attack > 0
            #
            #    self.alert_attack_no_blackout += np.sum(lines_alerted_beforeattack[lines_attacked_no_blackout])
            #    self.no_alert_attack_no_blackout += np.sum(~lines_alerted_beforeattack[lines_attacked_no_blackout])
            
            return score_ep
        
    @staticmethod
    def _normalisation_fun(cm_reward, cm_reward_min_ep, cm_reward_max_ep,min_score,max_score,tol=1e-5):
        standardized_score = np.round((cm_reward - cm_reward_min_ep) / (cm_reward_max_ep - cm_reward_min_ep +tol),4)
        #in case cm_reward_min_ep=cm_reward_max_ep=0, score is 0.0
        if(cm_reward_min_ep==cm_reward_max_ep):
            score_ep = 0.
        else:
            score_ep = min_score + (max_score - min_score) * standardized_score
        return score_ep
    
    def _compute_min_max_reward(self, nb_attacks,nb_last_attacks):

        if (nb_attacks==0 and self.blackout_encountered):
            cm_reward_min_ep = 0.
            cm_reward_max_ep= 0.
        elif(self.blackout_encountered):
            if(nb_last_attacks==0):
                cm_reward_min_ep = self.reward_min_no_blackout * nb_attacks
                cm_reward_max_ep = self.reward_max_no_blackout * nb_attacks
            elif(nb_last_attacks>=1):
                cm_reward_min_ep = self.reward_min_no_blackout * (nb_attacks - nb_last_attacks) + self.reward_min_blackout
                cm_reward_max_ep = self.reward_max_no_blackout * (nb_attacks - nb_last_attacks) + self.reward_max_blackout
        else:
            cm_reward_min_ep = self.reward_min_no_blackout * nb_attacks
            cm_reward_max_ep = self.reward_max_no_blackout * nb_attacks + self.reward_end_episode_bonus
        
        return cm_reward_min_ep, cm_reward_max_ep
        
