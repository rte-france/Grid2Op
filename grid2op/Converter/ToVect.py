# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from collections import OrderedDict

import numpy as np
from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_float, dt_int


class ToVect(Converter):
    """
    This converters allows to manipulate the vector representation of the actions and observations.

    In this converter:

    - `encoded_act` are numpy ndarray
    - `transformed_obs` are numpy ndarray
      (read more about these concepts by looking at the documentation of :class:`grid2op.Converter.Converters`)

    It is convertible to a gym representation (like the original action space) in the form of a spaces.Box
    representing a continuous action space (even though most component are probably discrete).
    Note that if converted to a gym space, it is unlikely the method "sample" will yield to valid results.
    Most of the time it should generate Ambiguous action that will not be handled by grid2op.

    **NB** the conversion to a gym space should be done thanks to the :class:`grid2op.Converter.GymActionSpace`.
    """

    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.init_action_space = action_space
        self.__class__ = ToVect.init_grid(action_space)
        self.do_nothing_vect = action_space({}).to_vect()

        # for gym conversion
        self.__gym_action_space = None
        self.__dict_space = None
        self.__order_gym = None
        self.__dtypes_gym = None
        self.__shapes_gym = None

        self.__order_gym_2_me = None
        self.__order_me_2_gym = None

    def convert_obs(self, obs):
        """
        This converter will match the observation to a vector, using the
        :func:`grid2op.Observation.BaseObservation.to_vect`
        function.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The observation, that will be processed into a numpy ndarray vector.

        Returns
        -------
        transformed_obs: ``numpy.ndarray``
            The vector representation of the action.

        """
        return obs.to_vect()

    def convert_act(self, encoded_act):
        """
        In this converter `encoded_act` is a numpy ndarray. This function transforms it back to a valid action.

        Parameters
        ----------
        encoded_act: ``numpy.ndarray``
            The action, represented as a vector

        Returns
        -------
        regular_act: :class:`grid2op.Action.Action`
            The corresponding action transformed with the :func:`grid2op.Action.BaseAction.from_vect`.

        """
        res = self.__call__({})
        res.from_vect(encoded_act, check_legit=False)
        return res

    def _init_gym_converter(self, cls_gym):
        if self.__gym_action_space is None:
            # i do that not to duplicate the code of the low / high bounds
            gym_action_space = cls_gym(self.init_action_space)
            low = tuple()
            high = tuple()
            order_gym = []
            dtypes = []
            shapes = []
            sizes = []
            prev = 0
            for k, v in gym_action_space.spaces.items():
                order_gym.append(k)
                dtypes.append(v.dtype)
                if isinstance(v, cls_gym._MultiBinaryType):
                    low += tuple([0 for _ in range(v.n)])
                    high += tuple([1 for _ in range(v.n)])
                    my_size = v.n
                elif isinstance(v, cls_gym._BoxType):
                    low += tuple(v.low)
                    high += tuple(v.high)
                    my_size = v.low.shape[0]
                else:
                    raise RuntimeError(
                        "Impossible to convert this converter to gym. Type {} of data "
                        "encountered while only MultiBinary and Box are supported for now."
                    )
                shapes.append(my_size)
                sizes.append(np.arange(my_size) + prev)
                prev += my_size
            self.__gym_action_space = gym_action_space
            my_type = cls_gym._BoxType(low=np.array(low), high=np.array(high), dtype=dt_float)

            order_me = []
            _order_gym_2_me = np.zeros(my_type.shape[0], dtype=dt_int) - 1
            _order_me_2_gym = np.zeros(my_type.shape[0], dtype=dt_int) - 1
            for el in type(self.init_action_space).attr_list_vect:
                order_me.append(cls_gym.keys_grid2op_2_human[el])

            prev = 0
            order_gym = list(gym_action_space.spaces.keys())
            for id_me, nm_attr in enumerate(order_me):
                id_gym = order_gym.index(nm_attr)
                index_me = np.arange(shapes[id_gym]) + prev
                _order_gym_2_me[sizes[id_gym]] = index_me
                _order_me_2_gym[index_me] = sizes[id_gym]
                # self.__order_gym_2_me[this_gym_ind] = sizes[id_me]
                prev += shapes[id_gym]
            self.__order_gym_2_me = _order_gym_2_me
            self.__order_me_2_gym = _order_me_2_gym
            self.__dict_space = {"action": my_type}
            self.__order_gym = order_gym
            self.__dtypes_gym = dtypes
            self.__shapes_gym = shapes

    def get_gym_dict(self, cls_gym):
        """
        Convert this action space int a "gym" action space represented by a dictionary (spaces.Dict)
        This dictionary counts only one keys which is "action" and inside this action is the
        
        cls_gym represents either :class:`grid2op.gym_compat.LegacyGymActionSpace` or
        :class:`grid2op.gym_compat.GymnasiumActionSpace`
        """
        self._init_gym_converter(cls_gym)
        return self.__dict_space

    def convert_action_from_gym(self, gymlike_action):
        """
        Convert a gym-like action (ie a Ordered dictionary with one key being only "action") to an
        action compatible with this converter (in this case a vectorized action).
        """
        vect = gymlike_action["action"]
        return vect[self.__order_gym_2_me]

    def convert_action_to_gym(self, action):
        """
        Convert a an action of this converter (ie a numpy array) into an action that is usable with
        an open ai gym (ie a Ordered dictionary with one key being only "action")
        """
        res = OrderedDict({"action": action[self.__order_me_2_gym]})
        return res
