# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Reward.l2RPNSandBoxScore import L2RPNSandBoxScore
from grid2op.dtypes import dt_float


class L2RPNWCCI2022ScoreFun(L2RPNSandBoxScore):
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
                 storage_cost=10.,  # € / MWh
                 alpha_redisp=1.0,
                 alpha_loss=1.0,
                 alpha_storage=1.0,
                 alpha_curtailment=1.0,
                 reward_max=1000.,
                 logger=None):
        super().__init__(alpha_redisp, alpha_loss, alpha_storage, alpha_curtailment, reward_max, logger)
        self.storage_cost = dt_float(storage_cost)
        
    def _get_storage_cost(self, env, p_t):
        """storage cost is a flat 10 € / MWh instead of depending on the marginal cost"""
        c_storage = np.abs(env._storage_power).sum() * self.storage_cost * env.delta_time_seconds / 3600.0
        return c_storage    
