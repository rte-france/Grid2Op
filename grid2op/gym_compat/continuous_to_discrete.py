# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from gym.spaces import Box, MultiDiscrete

from grid2op.dtypes import dt_int
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter


class ContinuousToDiscreteConverter(BaseGymAttrConverter):
    """
    Some RL algorithms are particularly suited for dealing with discrete action space or observation space.

    This "AttributeConverter" is responsible to convert continuous space to discrete space. The way it does
    it is by using bins. It uses `np.linspace` to compute the bins.

    We recommend using an odd number of bins (eg 3, 7 or 9 for example).

    Examples
    --------
    If `nb_bins` is 3 and  the original input space is [-10, 10], then the split is the following:

    - 0 encodes all numbers in [-10, -3.33)
    - 1 encodes all numbers in  [-3.33, 3.33)
    - 2 encode all numbers in [3.33, 10.]

    And reciprocally, this action with :

    - 0 is understand as -5.0 (middle of the interval -10 / 0)
    - 1 is understand as 0.0 (middle of the interval represented by -10 / 10)
    - 2 is understand as 5.0 (middle of the interval represented by 0 / 10)

    If `nb_bins` is 5 and  the original input space is [-10, 10], then the split is the following:

    - 0 encodes all numbers in [-10, -6)
    - 1 encodes all numbers in  [-6, -2)
    - 2 encode all numbers in [-2, 2)
    - 3 encode all numbers in [2, 6)
    - 3 encode all numbers in [6, 10]

    And reciprocally, this action with :

    - 0 is understand as -6.6666...
    - 1 is understand as -3.333...
    - 2 is understand as 0.
    - 3 is understand as 3.333...
    - 4 is understand as 6.6666...

    """
    def __init__(self, init_space, nb_bins):
        if not isinstance(init_space, Box):
            raise RuntimeError("Impossible to convert a gym space of type {} to a discrete space"
                               " (it should be of "
                               "type space.Box)"
                               "".format(type(init_space)))
        if nb_bins < 2:
            raise RuntimeError("This do not work with less that 1 bin (if you want to ignored some part "
                               "of the action_space or observation_space please use the "
                               "\"gym_space.ignore_attr\" or \"gym_space.keep_only_attr\"")

        min_ = init_space.low
        max_ = init_space.high
        self._ignored = min_ == max_  # which component are ignored
        self._res = min_
        self._values = np.linspace(min_, max_, num=nb_bins+2)
        self._values = self._values[1:-1, :]  # the values that will be used when using #gym_to_glop

        # TODO there might a cleaner approach here
        self._bins_size = np.linspace(min_, max_, num=2*nb_bins+1)
        self._bins_size = self._bins_size[2:-1:2, :]  # the values defining the "cuts"

        self._gen_idx = np.arange(self._bins_size.shape[-1])
        n_bins = np.ones(min_.shape[0]) * nb_bins
        n_bins[self._ignored] = 1  # if min and max are equal, i don't want to have multiple variable
        BaseGymAttrConverter.__init__(self,
                                      space=MultiDiscrete(n_bins),
                                      )

    def gym_to_g2op(self, gym_object):
        return copy.deepcopy(self._values[gym_object, self._gen_idx])

    def g2op_to_gym(self, g2op_object):
        mask = self._bins_size >= g2op_object
        mask = 1 - mask
        res = np.sum(mask, axis=0)
        res[self._ignored] = 0
        return res
