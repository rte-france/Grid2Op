# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from gym.spaces import Tuple, MultiBinary, MultiDiscrete, Discrete

from grid2op.dtypes import dt_int
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter


class FixedTuple(Tuple):
    """I simply overload the "seed" function because the default one behaves
    really really poorly
    see issue https://github.com/openai/gym/issues/2166
    """

    def seed(self, seed=None):
        """Seed the PRNG of this space.
        see issue https://github.com/openai/gym/issues/2166
        of openAI gym
        """
        seeds = super(Tuple, self).seed(seed)
        sub_seeds = seeds
        max_ = np.iinfo(int).max
        for i, space in enumerate(self.spaces):
            sub_seed = self.np_random.randint(max_)
            sub_seeds.append(space.seed(sub_seed))
        return sub_seeds


class MultiToTupleConverter(BaseGymAttrConverter):
    """
    Some framework, for example ray[rllib] do not support MultiBinary nor MultiDiscrete gym
    action space. Apparently this is not going to change in a near
    future (see https://github.com/ray-project/ray/issues/1519).

    We choose to encode some variable using `MultiBinary` variable in grid2op. This allows for easy
    manipulation of them if using these frameworks.

    MultiBinary are encoded with gym Tuple of gym Discrete variables.

    TODO add code example
    """
    def __init__(self, init_space=None):
        self.size = None
        BaseGymAttrConverter.__init__(self,
                                      space=None)
        if init_space is not None:
            self.initialize_space(init_space)

        self.previous_fun = self._previous_fun
        self.after_fun = self._after_fun

    def _previous_fun(self, x):
        return x

    def _after_fun(self, x):
        return x

    def initialize_space(self, init_space):
        if isinstance(init_space, (MultiBinary, MultiDiscrete)):
            pass
        elif isinstance(init_space, BaseGymAttrConverter):
            self.previous_fun = init_space.g2op_to_gym
            self.after_fun = init_space.gym_to_g2op
            if isinstance(init_space.my_space, (MultiBinary, MultiDiscrete)):
                init_space = init_space.my_space
            else:
                raise RuntimeError("Bad converter used. It should be of type MultiBinary or MultiDiscrete")
        else:
            raise RuntimeError("Impossible to convert a gym space of type {} to a Tuple (it should be of "
                               "type space.MultiBinary or space.MultiDiscrete)"
                               "".format(type(init_space)))
        if isinstance(init_space, MultiBinary):
            self.size = init_space.n
        else:
            # then it's a MultiDiscrete
            self.size = init_space.nvec.shape[0]
        li = []
        for i in range(self.size):
            tmp_sz = 2
            if isinstance(init_space, MultiDiscrete):
                tmp_sz = init_space.nvec[i]
            li.append(Discrete(tmp_sz))
        self.base_initialize(space=FixedTuple(li),
                             g2op_to_gym=None,
                             gym_to_g2op=None)

    def gym_to_g2op(self, gym_object):
        tmp = np.array(gym_object).astype(dt_int)
        return self.after_fun(tmp)

    def g2op_to_gym(self, g2op_object):
        tmp = self.previous_fun(g2op_object)  # TODO
        return tuple(tmp.astype(dt_int))
        
    def close(self):
        pass
