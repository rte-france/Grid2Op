# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


class BaseGymAttrConverter(object):
    def __init__(self, space, gym_to_g2op=None, g2op_to_gym=None):
        self._my_gym_to_g2op = gym_to_g2op
        self._my_g2op_to_gym = g2op_to_gym
        self.my_space = space

    def gym_to_g2op(self, gym_object):
        if self._my_gym_to_g2op is None:
            raise NotImplementedError("Unable to convert gym object to grid2op object with this converter")
        return self._my_gym_to_g2op(gym_object)

    def g2op_to_gym(self, g2op_object):
        if self._my_g2op_to_gym is None:
            raise NotImplementedError("Unable to convert grid2op object to gym object with this converter")
        return self._my_g2op_to_gym(g2op_object)
