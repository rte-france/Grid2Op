# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from gym.spaces import Space


class BaseGymAttrConverter(object):
    def __init__(self, space=None, gym_to_g2op=None, g2op_to_gym=None):
        self.__is_init_super = False  # is the "super" class initialized, do not modify in child class

        self._is_init_space = False  # is the instance initialized
        self._my_gym_to_g2op = None
        self._my_g2op_to_gym = None
        self.my_space = None

        if self.my_space is not None:
            self.base_initialize(space, gym_to_g2op, g2op_to_gym)

    def base_initialize(self, space, gym_to_g2op, g2op_to_gym):
        if self.__is_init_super:
            return
        self._my_gym_to_g2op = gym_to_g2op
        self._my_g2op_to_gym = g2op_to_gym
        self.my_space = space
        self._is_init_space = True
        self.__is_init_super = True

    def is_init_space(self):
        return self._is_init_space

    def initialize_space(self, space):
        if self._is_init_space:
            return
        if not isinstance(space, Space):
            raise RuntimeError("Impossible to scale a converter if this one is not from type space.Space")
        self.my_space = space
        self._is_init_space = True

    def gym_to_g2op(self, gym_object):
        if self._my_gym_to_g2op is None:
            raise NotImplementedError("Unable to convert gym object to grid2op object with this converter")
        return self._my_gym_to_g2op(gym_object)

    def g2op_to_gym(self, g2op_object):
        if self._my_g2op_to_gym is None:
            raise NotImplementedError("Unable to convert grid2op object to gym object with this converter")
        return self._my_g2op_to_gym(g2op_object)
