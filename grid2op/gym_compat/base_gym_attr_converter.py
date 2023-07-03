# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE
from grid2op.gym_compat.utils import check_gym_version


class __AuxBaseGymAttrConverter(object):
    """
    TODO work in progress !

    Need help if you can :-)
    
    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`BaseGymAttrConverter` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`BaseGymnasiumAttrConverter`), otherwise it will
          inherit from gym (and will be exactly :class:`BaseLegacyGymAttrConverter`)
        - :class:`BaseGymnasiumAttrConverter` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`BaseLegacyGymAttrConverter` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    """

    def __init__(self, space=None, gym_to_g2op=None, g2op_to_gym=None):
        check_gym_version(type(self)._gymnasium)
        self.__is_init_super = (
            False  # is the "super" class initialized, do not modify in child class
        )

        self._is_init_space = False  # is the instance initialized
        self._my_gym_to_g2op = None
        self._my_g2op_to_gym = None
        self.my_space = None

        if space is not None:
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
        if not isinstance(space, type(self)._SpaceType):
            raise RuntimeError(
                "Impossible to scale a converter if this one is not from type space.Space"
            )
        self.my_space = space
        self._is_init_space = True

    def gym_to_g2op(self, gym_object):
        """
        Convert a gym object to a grid2op object

        Parameters
        ----------
        gym_object:
            An object (action or observation) represented as a gym "ordered dictionary"

        Returns
        -------
        The same object, represented as a grid2op.Action.BaseAction or grid2op.Observation.BaseObservation.

        """
        if self._my_gym_to_g2op is None:
            raise NotImplementedError(
                "Unable to convert gym object to grid2op object with this converter"
            )
        return self._my_gym_to_g2op(gym_object)

    def g2op_to_gym(self, g2op_object):
        """
        Convert a gym object to a grid2op object

        Parameters
        ----------
        g2op_object:
            An object (action or observation) represented as a grid2op.Action.BaseAction or
            grid2op.Observation.BaseObservation

        Returns
        -------
        The same object, represented as a gym "ordered dictionary"

        """
        if self._my_g2op_to_gym is None:
            raise NotImplementedError(
                "Unable to convert grid2op object to gym object with this converter"
            )
        return self._my_g2op_to_gym(g2op_object)


if GYM_AVAILABLE:
    from gym.spaces import Space as LegGymSpace
    BaseLegacyGymAttrConverter = type("BaseLegacyGymAttrConverter",
                                     (__AuxBaseGymAttrConverter, ),
                                     {"_SpaceType": LegGymSpace, 
                                      "_gymnasium": False,
                                      "__module__": __name__})
    BaseLegacyGymAttrConverter.__doc__ = __AuxBaseGymAttrConverter.__doc__
    BaseGymAttrConverter = BaseLegacyGymAttrConverter
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Space
    BaseGymnasiumAttrConverter = type("BaseGymnasiumAttrConverter",
                                     (__AuxBaseGymAttrConverter, ),
                                     {"_SpaceType": Space, 
                                      "_gymnasium": True,
                                      "__module__": __name__})
    BaseGymnasiumAttrConverter.__doc__ = __AuxBaseGymAttrConverter.__doc__
    BaseGymAttrConverter = BaseGymnasiumAttrConverter
