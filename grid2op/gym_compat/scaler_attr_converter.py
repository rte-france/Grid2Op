# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np

from grid2op.dtypes import dt_float
from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE


class __AuxScalerAttrConverter:
    """
    This is a scaler that transforms a initial gym space `init_space` into its scale version.

    It can be use to scale the observation by substracting the mean and dividing by the variance for
    example.

    TODO work in progress !

    Need help if you can :-)

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`ScalerAttrConverter` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`ScalerAttrConverterGymnasium`), otherwise it will
          inherit from gym (and will be exactly :class:`ScalerAttrConverterLegacyGym`)
        - :class:`ScalerAttrConverterGymnasium` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`ScalerAttrConverterLegacyGym` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    """

    def __init__(self, substract, divide, dtype=None, init_space=None):
        super().__init__(
            g2op_to_gym=None, gym_to_g2op=None, space=None
        )  # super should be from type BaseGymAttrConverter
        self._substract = np.array(substract)
        self._divide = np.array(divide)
        self.dtype = dtype if dtype is not None else dt_float
        if init_space is not None:
            self.initialize_space(init_space)

    def initialize_space(self, init_space):
        if self._is_init_space:
            return
        if not isinstance(init_space, type(self)._BoxType):
            raise RuntimeError(
                "Impossible to scale a converter if this one is not from type space.Box"
            )

        tmp_space = copy.deepcopy(init_space)
        # properly change the low / high value
        low_tmp = self.scale(tmp_space.low)
        high_tmp = self.scale(tmp_space.high)
        low_ = np.minimum(high_tmp, low_tmp)
        high_ = np.maximum(high_tmp, low_tmp)
        tmp_space.low[:] = low_
        tmp_space.high[:] = high_

        if self.dtype is not None:
            tmp_space.dtype = np.dtype(self.dtype)
            tmp_space.low = tmp_space.low.astype(self.dtype)
            tmp_space.high = tmp_space.high.astype(self.dtype)
        self.base_initialize(
            space=tmp_space, g2op_to_gym=self.scale, gym_to_g2op=self.unscale
        )
        self.dtype = self.my_space.dtype
        self._substract = self._substract.astype(self.dtype)
        self._divide = self._divide.astype(self.dtype)
        self._is_init_space = True

    def scale(self, vect):
        tmp = vect.astype(self.dtype)
        tmp = (tmp - self._substract) / self._divide
        return tmp

    def unscale(self, vect):
        tmp = vect * self._divide + self._substract
        return tmp

    def close(self):
        pass


if GYM_AVAILABLE:
    from gym.spaces import Box as LegacyGymBox
    from grid2op.gym_compat.base_gym_attr_converter import BaseLegacyGymAttrConverter
    ScalerAttrConverterLegacyGym = type("ScalerAttrConverterLegacyGym",
                                        (__AuxScalerAttrConverter, BaseLegacyGymAttrConverter, ),
                                        {"_gymnasium": False,
                                         "_BoxType": LegacyGymBox,
                                         "__module__": __name__})
    ScalerAttrConverterLegacyGym.__doc__ = __AuxScalerAttrConverter.__doc__
    ScalerAttrConverter = ScalerAttrConverterLegacyGym
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Box
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    ScalerAttrConverterGymnasium = type("ScalerAttrConverterGymnasium",
                                     (__AuxScalerAttrConverter, BaseGymnasiumAttrConverter, ),
                                     {"_gymnasium": True,
                                      "_BoxType": Box,
                                      "__module__": __name__})
    ScalerAttrConverterGymnasium.__doc__ = __AuxScalerAttrConverter.__doc__
    ScalerAttrConverter = ScalerAttrConverterGymnasium
