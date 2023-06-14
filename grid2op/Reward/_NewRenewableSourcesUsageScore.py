# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.BaseReward import BaseReward

class _NewRenewableSourcesUsageScore(BaseReward):
    """

    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be **MINIMIZED**,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the "low carbon score", meaning here how much of the new renewable energy sources capacities have been called.
    It should not be used to train an agent.
    
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.gen_res_p_list = []
        self.gen_res_p_before_curtail_list = []
        
    def __initialize__(self, env):
        self.reset(env)
        
    def reset(self):
        self.gen_res_p_list = []
        self.gen_res_p_before_curtail_list = []
        
    def __call__(self, env, obs, is_done):
        gen_nres_p, gen_nres_p_before_curtail = _get_total_nres_usage(env, obs)
        self.gen_res_p_list.append(gen_nres_p)
        self.gen_res_p_before_curtail_list.append(gen_nres_p_before_curtail)
        
        if is_done:
            ratio_nres_usage = 100 * np.sum(self.gen_res_p_list) / np.sum(self.gen_res_p_before_curtail_list)
            return ratio_nres_usage * np.log(ratio_nres_usage)
        
        
    @staticmethod
    def _get_total_nres_usage(env, obs):
        gen_type = env.gen_type
        nres_mask = [any(is_nres) for is_nres in zip(gen_type=="wind", gen_type=="solar")]
        
        gen_nres_p = np.sum(obs.gen_p[nres_mask])
        gen_nres_p_before_curtail = np.sum(obs.gen_p_before_curtail[nres_mask])
        
        return gen_nres_p, gen_nres_p_before_curtail
    
