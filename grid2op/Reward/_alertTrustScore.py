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
    The alerts are assessed once per attack. In this scheme, this "reward" computed the assistant"cost score", which penalized the number of alerts
    the assistant have produced during an episode. It should not be used to train an agent.
    
    """
    
    def __init__(self,
                 logger=None,
                 reward_min_no_blackout=-1.0,
                 reward_min_blackout=-10.0, 
                 reward_max_no_blackout=1.0,
                 reward_max_blackout=2.0,
                 reward_end_episode_bonus=1.0):
        
        super().__init__(logger,
                 reward_min_no_blackout,
                 reward_min_blackout, 
                 reward_max_no_blackout,
                 reward_max_blackout,
                 reward_end_episode_bonus)
        
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self.score_min_ep = lambda k: reward_min_no_blackout * (k - 1) + reward_min_blackout
        self.score_max_ep = lambda k: np.max(reward_max_no_blackout * (k - 1) + reward_max_blackout,
                                             reward_max_no_blackout * k + reward_end_episode_bonus)
        
    def initialize(self, env):
        self._is_simul_env = self.is_simulated_env(env)
        if self._is_simul_env:
            return
        self.reset(env)
        return super().initialize(env)
            
    def reset(self, env):
        super().reset(env)
        self.total_nb_attacks = 0
        self.cumulated_reward = 0
            
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        score_ep = 0.
        if self._is_simul_env:
            return score_ep
            
        res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        self.cumulated_reward += res
        self.total_nb_attacks += np.sum(env._time_since_last_attack == 0)
            
        if not is_done:
            return score_ep
        else:
            score_min_ep = self.score_min_ep(self.total_nb_attacks)
            score_max_ep = self.score_max_ep(self.total_nb_attacks)
            score_ep = self._normalisation_fun(self.cumulated_reward, score_min_ep, score_max_ep)
                
            return score_ep
        
    @staticmethod
    def _normalisation_fun(score, score_min_ep, score_max_ep):
        standardized_score = (score - score_min_ep) / (score_max_ep - score_min_ep)
        score_ep = standardized_score * 2. - 1.
        return score_ep
         
