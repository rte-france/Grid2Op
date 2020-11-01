# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from gym.spaces import Tuple, MultiBinary, MultiDiscrete, Discrete

from grid2op.dtypes import dt_int
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter


class MultiToTupleConverter(BaseGymAttrConverter):
    """
    Some framework, for example ray[rllib] do not support MultiBinary nor MultiDiscrete gym
    action space. Appanrently this is not going to change in a near
    future (see https://github.com/ray-project/ray/issues/1519).

    We choose to encode some variable using `MultiBinary` variable in grid2op. This allows for easy
    manipulation of them if using these frameworks.

    MultiBinary are encoded with gym Tuple of gym Discrete variables.
    """
    def __init__(self, init_space):
        if not isinstance(init_space, (MultiBinary, MultiDiscrete)):
            raise RuntimeError("Impossible to convert a gym space of type {} to a Tuple (it should be of "
                               "type space.MultiBinary or space.MultiDiscrete)"
                               "".format(type(init_space)))
        self.size = init_space.n
        li = []
        for i in range(self.size):
            tmp_sz = 2
            if isinstance(init_space, MultiDiscrete):
                tmp_sz = init_space.nvec[i]
            li.append(Discrete(tmp_sz))
        BaseGymAttrConverter.__init__(self,
                                      space=Tuple(li),
                                      )

    def gym_to_g2op(self, gym_object):
        return np.array(gym_object).astype(dt_int)

    def g2op_to_gym(self, g2op_object):
        return tuple(g2op_object.astype(dt_int))
