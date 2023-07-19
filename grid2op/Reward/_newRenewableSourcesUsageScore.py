# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float

class _NewRenewableSourcesUsageScore(BaseReward):
    """

    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This score needs to be **MAXIMIZED**,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the "low carbon score", meaning here how much of the new renewable energy sources capacities have been called.
    It should not be used to train an agent.
    It has been designed to be defined in the continuous domain [50,100] with outputs values between[-1,1]
    
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self.gen_res_p_curtailed_list = None
        self.gen_res_p_before_curtail_list = None
        self._is_simul_env = False
        
    def initialize(self, env):
        self.reset(env)
        
    def reset(self, env):
        self._is_simul_env = is_simulated_env(env)
        if self._is_simul_env:
            return
        
        self.gen_res_p_curtailed_list = np.zeros(env.chronics_handler.max_timestep() + 1)
        self.gen_res_p_before_curtail_list = np.zeros(env.chronics_handler.max_timestep() + 1)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        if self._is_simul_env:
            return dt_float(0.)
        
        if not is_done:
            gen_nres_p_effective, gen_nres_p_before_curtail = self._get_total_nres_usage(env)
            self.gen_res_p_curtailed_list[env.nb_time_step] = gen_nres_p_effective
            self.gen_res_p_before_curtail_list[env.nb_time_step] = gen_nres_p_before_curtail
            return dt_float(0.)
        else:
            ratio_nres_usage = 100 * self.gen_res_p_curtailed_list[1:].sum() / self.gen_res_p_before_curtail_list[1:].sum()
            return self._surlinear_func_curtailment(ratio_nres_usage)
            
    @staticmethod
    def _get_total_nres_usage(env):
        nres_mask = env.gen_renewable
        gen_p, *_ = env.backend.generators_info()
        gen_nres_p_before_curtail = env._gen_before_curtailment[nres_mask].sum()
        gen_nres_p_effective = gen_p[nres_mask].sum()
        
        return gen_nres_p_effective, gen_nres_p_before_curtail
    
    @staticmethod
    def _surlinear_func_curtailment(x, center=80, eps=1e-6):
        x = np.fmax(x, eps)  #  to avoid log(0)...
        f_surlinear = lambda x: x * np.log(x)
        f_centralized = lambda x : f_surlinear(x) - f_surlinear(center)
        f_standardizer= lambda x : np.ones_like(x) * f_centralized(100) * (x >= center) - np.ones_like(x) * f_centralized(50) * (x < center)
                
        return f_centralized(x) / f_standardizer(x)
    

#to wait before PR Laure 
def is_simulated_env(env):

    # to prevent cyclical import
    from grid2op.Environment._ObsEnv import _ObsEnv
    from grid2op.Environment._forecast_env import _ForecastEnv

    # This reward is not compatible with simulations
    return isinstance(env, (_ObsEnv, _ForecastEnv))

