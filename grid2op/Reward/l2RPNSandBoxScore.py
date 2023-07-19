# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class L2RPNSandBoxScore(BaseReward):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be **MINIMIZED**,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the "grid operation cost". It should not be used to train an agent.

    The "reward" the closest to this score is given by the :class:`RedispReward` class.

    """

    def __init__(self,
                 alpha_redisp=1.0,
                 alpha_loss=1.0,
                 alpha_storage=1.0,
                 alpha_curtailment=1.0,
                 reward_max=1000.,
                 logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(1.0)  # carefull here between min and max...
        self.reward_max = dt_float(reward_max)
        self.alpha_redisp = dt_float(alpha_redisp)
        self.alpha_loss = dt_float(alpha_loss)
        self.alpha_storage = dt_float(alpha_storage)
        self.alpha_curtailment = dt_float(alpha_curtailment)

    def initialize(self, env):
        # TODO compute reward max! 
        return super().initialize(env)
    
    def _get_load_p(self, env):
        load_p, *_ = env.backend.loads_info()
        return load_p
    
    def _get_gen_p(self, env):
        gen_p, *_ = env.backend.generators_info()
        return gen_p
    
    def _get_losses(self, env, gen_p, load_p):
        return (gen_p.sum(dtype=dt_float) - load_p.sum(dtype=dt_float)) * env.delta_time_seconds / 3600.0
    
    def _get_marginal_cost(self, env):
        gen_activeprod_t = env._gen_activeprod_t
        p_t = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.0]).astype(dt_float)  
        # price is per MWh be sure to convert the MW (of losses and generation) to MWh before multiplying by the cost 
        return p_t
    
    def _get_redisp_cost(self, env, p_t):
        actual_dispatch = env._actual_dispatch
        c_redispatching = (
            np.abs(actual_dispatch).sum() * p_t * env.delta_time_seconds / 3600.0
        )
        return c_redispatching
    
    def _get_curtail_cost(self, env, p_t):
        curtailment_mw = -env._sum_curtailment_mw  # curtailment is always negative in the env 
        c_curtailment = (
            curtailment_mw * p_t * env.delta_time_seconds / 3600.0
        )
        return c_curtailment

    def _get_loss_cost(self, env, p_t):
        gen_p = self._get_gen_p(env)
        load_p = self._get_load_p(env)
        losses = self._get_losses(env, gen_p, load_p)
        c_loss = losses * p_t
        return c_loss
        
    def _get_storage_cost(self, env, p_t):
        c_storage = np.abs(env._storage_power).sum() * p_t * env.delta_time_seconds / 3600.0
        return c_storage
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            # DO SOMETHING IN THIS CASE
            return self.reward_min

        # compute the marginal cost
        p_t = self._get_marginal_cost(env)
        
        # redispatching amount
        c_redispatching = self._get_redisp_cost(env, p_t)
        
        # curtailment amount
        c_curtailment = self._get_curtail_cost(env, p_t)
        
        # cost of losses
        c_loss = self._get_loss_cost(env, p_t)
        
        # storage units
        c_storage = self._get_storage_cost(env, p_t)
        
        # total "operationnal cost"
        c_operations = dt_float(self.alpha_loss * c_loss + 
                                self.alpha_redisp * c_redispatching + 
                                self.alpha_storage * c_storage + 
                                self.alpha_curtailment * c_curtailment)
        return c_operations
