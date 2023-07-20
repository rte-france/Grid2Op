# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Reward.baseReward import BaseReward

class _AssistantConfidenceScore(BaseReward):
    """

    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be **MINIMIZED**,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the assistant "confidence score", which evaluates how confident an agent was in its actions for handling
    unforeseen line l disconnection events prior to occurring.
    It should not be used to train an agent.
    
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def initialize(self, env):
        self.reset(env)
        
    def __call__(self, env, obs, is_done):
        return 0.
        
class _AssistantCostScore(BaseReward):
    """

    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be **MINIMIZED**,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the assistant"cost score", which penalized the number of alarm the assistant have produced.
    It should not be used to train an agent.
    
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def initialize(self, env):
        self.reset(env)
        
    def __call__(self, env, obs, is_done):
        return 0.
