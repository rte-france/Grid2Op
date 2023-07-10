# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward._newRenewableSourcesUsageScore import _NewRenewableSourcesUsageScore
from grid2op.dtypes import dt_float
from grid2op.Exceptions import Grid2OpException
        
class _AlertCostScore(BaseReward):
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
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self._is_simul_env = False
        self.total_nb_alertes_possible = None
        self.total_nb_alerts = None
        
    def initialize(self, env):

        if not env.dim_alerts > 0:
            raise Grid2OpException(
                'Impossible to use the "_AlertCostScore" with an environment for which the Assistant feature '
                'is disabled. Please make sure "env.dim_alerts" is > 0 or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        self.reset(env)
        
    def reset(self, env):
        self._is_simul_env = self.is_simulated_env(env)
        if self._is_simul_env:
            return
        
        self.total_nb_alertes_possible = (env.chronics_handler.max_timestep() + 1) * (env.dim_alerts)
        self.total_nb_alerts = 0
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if self._is_simul_env:
            return dt_float(0.)
        
        if is_done:
            ratio_nb_alerts = 100 * ( 1 - self.total_nb_alerts / self.total_nb_alertes_possible)
            return self._penalization_fun(ratio_nb_alerts)
        else:
            self.total_nb_alerts = env._total_number_of_alert
            return dt_float(0.)
        
    @staticmethod
    def _penalization_fun(x, center=80):
        return _NewRenewableSourcesUsageScore._surlinear_func_curtailment(x=x, center=center)
        

    
