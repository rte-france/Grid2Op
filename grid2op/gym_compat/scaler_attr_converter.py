# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from gym.spaces import Box

from grid2op.dtypes import dt_float
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter


class ScalerAttrConverter(BaseGymAttrConverter):
    """
    This is a scaler that transforms a initial gym space `init_space` into its scale version.

    It can be use to scale the observation by substracting the mean and dividing by the variance for
    example.

    """
    def __init__(self, substract, divide, init_space, dtype=None):
        self._substract = np.array(substract)
        self._divide = np.array(divide)
        self.dtype = dtype if dtype is not None else dt_float
        if not isinstance(init_space, Box):
            raise RuntimeError("Impossible to scale a converter if this one is not from type space.Box")
        tmp_space = copy.deepcopy(init_space)
        tmp_space.low = self.scale(tmp_space.low)
        tmp_space.high = self.scale(tmp_space.high)
        if dtype is not None:
            tmp_space.dtype = dtype
            tmp_space.low = tmp_space.low.astype(self.dtype)
            tmp_space.high = tmp_space.high.astype(self.dtype)
        BaseGymAttrConverter.__init__(self,
                                      g2op_to_gym=self.scale,
                                      gym_to_g2op=self.unscale,
                                      space=tmp_space)
        self.dtype = self.my_space.dtype
        self._substract = self._substract.astype(self.dtype)
        self._divide = self._divide.astype(self.dtype)

    def scale(self, vect):
        tmp = vect.astype(self.dtype)
        tmp = (tmp - self._substract) / self._divide
        return tmp

    def unscale(self, vect):
        tmp = vect * self._divide + self._substract
        return tmp
