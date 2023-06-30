# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from collections import OrderedDict
import numpy as np
import copy

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.gym_compat.utils import check_gym_version, sample_seed
from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE


class __AuxBaseGymSpaceConverter:
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        Used as a base class to convert grid2op state to gym state (wrapper for some useful function
        for both the action space and the observation space).

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`_BaseGymSpaceConverter` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`_BaseGymnasiumSpaceConverter`), otherwise it will
          inherit from gym (and will be exactly :class:`_BaseGymLegacySpaceConverter`)
        - :class:`_BaseGymnasiumSpaceConverter` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`_BaseGymLegacySpaceConverter` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    """

    def __init__(self, dict_gym_space, dict_variables=None):
        check_gym_version(True)  # TODO GYMNASIUM
        type(self)._DictType.__init__(self, dict_gym_space)
        self._keys_encoding = {}
        if dict_variables is not None:
            for k, v in dict_variables.items():
                self._keys_encoding[k] = v
        self.__func = {}

    @classmethod
    def _generic_gym_space(cls, dt, sh, low=None, high=None):
        if dt == dt_int:
            if low is None:
                low = np.iinfo(dt).min
            if high is None:
                high = np.iinfo(dt).max
        else:
            low = -np.inf
            high = +np.inf
        shape = (sh,)
        my_type = cls._BoxType(
            low=dt.type(low), high=dt.type(high), shape=shape, dtype=dt
        )
        return my_type

    @classmethod
    def _boolean_type(cls, sh):
        return cls._MultiBinaryType(n=sh)

    @staticmethod
    def _simplifykeys_for_timestamps(key):
        """some keys are encoded to be returned as scalar, i need to transform them."""
        res = (
            (key == "year")
            or (key == "month")
            or (key == "day")
            or (key == "hour_of_day")
            or (key == "minute_of_hour")
            or (key == "day_of_week")
        )
        res = (
            res
            or (key == "is_alarm_illegal")
            or (key == "was_alarm_used_after_game_over")
        )
        return res

    @staticmethod
    def _extract_obj_grid2op(vect, dtype, key):
        if len(vect) == 1 and _BaseGymSpaceConverter._simplifykeys_for_timestamps(key):
            res = vect[0]
            # convert the types for json serializable
            # this is not automatically done by gym...
            if dtype == dt_int or dtype == np.int64 or dtype == np.int32:
                res = int(res)
            elif dtype == dt_float or dtype == np.float64 or dtype == np.float32:
                res = float(res)
            elif dtype == dt_bool:
                res = bool(res)
        else:
            res = vect
        return res

    def _base_to_gym(self, keys, obj, dtypes, converter=None):
        """convert the obj (grid2op object) into a gym observation / action space"""
        res = OrderedDict()
        for k in keys:
            if k in self.__func:
                obj_json_cleaned = self.__func[k](obj)
            else:
                if converter is not None:
                    # for converting the names between internal names and "human readable names"
                    conv_k = converter[k]
                else:
                    conv_k = k

                obj_raw = obj._get_array_from_attr_name(conv_k)
                if conv_k in self._keys_encoding:
                    if self._keys_encoding[conv_k] is None:
                        # keys is deactivated
                        continue
                    elif isinstance(self._keys_encoding[conv_k], type(self)._SpaceType):
                        obj_json_cleaned = getattr(obj, conv_k)
                    else:
                        # i need to process the "function" part in the keys
                        obj_json_cleaned = self._keys_encoding[conv_k].g2op_to_gym(
                            obj_raw
                        )
                else:
                    obj_json_cleaned = self._extract_obj_grid2op(obj_raw, dtypes[k], k)
            res[k] = obj_json_cleaned
        return res

    def add_key(self, key_name, function, return_type):
        """

        Allows to add arbitrary function to the representation, as a gym environment of
        the action space of the observation space.

        TODO
        **NB** this key is not used when converted back to grid2Op object, as of now we don't recommend to
        use it for the action space !

        See the example for more information.

        Parameters
        ----------
        key_name:
            The name you want to get

        function:
            A function that takes as input

        return_type

        Returns
        -------


        Examples
        ---------
        In the example below, we explain how to add the "connectivity_matrix" as part of the observation space
        (when converted to gym). The new key "connectivity matrix" will be added to the gym observation.

        .. code-block:: python


            # create a grid2op environment
            import grid2op
            env_name = "l2rpn_case14_sandbox"
            env_glop = grid2op.make(env_name)

            # convert it to gym
            import gym
            import numpy as np
            from grid2op.gym_compat import GymEnv
            env_gym = GymEnv(env_glop)

            # default gym environment, the connectivity matrix is not computed
            obs_gym = env_gym.reset()
            print(f"Is the connectivity matrix part of the observation in gym: {'connectivity_matrix' in obs_gym}")

            # add the "connectivity matrix" as part of the observation in gym
            from gym.spaces import Box
            shape_ = (env_glop.dim_topo, env_glop.dim_topo)
            env_gym.observation_space.add_key("connectivity_matrix",
                                              lambda obs: obs.connectivity_matrix(),
                                              Box(shape=shape_,
                                                  low=np.zeros(shape_),
                                                  high=np.ones(shape_),
                                                )
                                              )

            # we highly recommend to "reset" the environment after setting up the observation space

            obs_gym = env_gym.reset()
            print(f"Is the connectivity matrix part of the observation in gym: {'connectivity_matrix' in obs_gym}")
        """

        self.spaces[key_name] = return_type
        self.__func[key_name] = function

    def get_dict_encoding(self):
        """
        TODO examples and description

        Returns
        -------

        """
        return copy.deepcopy(self._keys_encoding)

    def reencode_space(self, key, func):
        """
        TODO examples and description

        Returns
        -------

        """
        raise NotImplementedError(
            "This should be implemented in the GymActionSpace and GymObservationSpace"
        )

    def reenc(self, key, fun):
        """
        shorthand for :func:`GymObservationSpace.reencode_space` or
        :func:`GymActionSpace.reencode_space`
        """
        return self.reencode_space(key, fun)

    def keep_only_attr(self, attr_names):
        """
        keep only a certain part of the observation
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names]

        dict_ = self.spaces.keys()
        res = self
        for k in dict_:
            if k not in attr_names:
                res = res.reencode_space(k, None)
        return res

    def ignore_attr(self, attr_names):
        """
        ignore some attribute names from the space
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        res = self
        for k in self.spaces.keys():
            if k in attr_names:
                res = res.reencode_space(k, None)
        return res

    def seed(self, seed=None):
        """Seed the PRNG of this space.
        see issue https://github.com/openai/gym/issues/2166
        of openAI gym
        """
        seeds = super(type(self)._DictType, self).seed(seed)
        sub_seeds = seeds
        max_ = np.iinfo(dt_int).max
        for i, space_key in enumerate(sorted(self.spaces.keys())):
            sub_seed = sample_seed(max_, self.np_random)
            sub_seeds.append(self.spaces[space_key].seed(sub_seed))
        return sub_seeds

    def close(self):
        pass


if GYM_AVAILABLE:
    from gym.spaces import Discrete, Box, Dict, Space, MultiBinary, Tuple
    _BaseGymLegacySpaceConverter = type("_BaseGymLegacySpaceConverter",
                                        (__AuxBaseGymSpaceConverter, Dict, ),
                                        {"_DiscreteType": Discrete,
                                         "_BoxType": Box,
                                         "_DictType": Dict,
                                         "_SpaceType": Space, 
                                         "_MultiBinaryType": MultiBinary, 
                                         "_TupleType": Tuple, 
                                         "_gymnasium": False})
    _BaseGymLegacySpaceConverter.__doc__ = __AuxBaseGymSpaceConverter.__doc__
    _BaseGymSpaceConverter = _BaseGymLegacySpaceConverter
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Discrete, Box, Dict, Space, MultiBinary, Tuple
    _BaseGymnasiumSpaceConverter = type("_BaseGymnasiumSpaceConverter",
                                        (__AuxBaseGymSpaceConverter, Dict, ),
                                        {"_DiscreteType": Discrete,
                                         "_BoxType": Box,
                                         "_DictType": Dict,
                                         "_SpaceType": Space, 
                                         "_MultiBinaryType": MultiBinary, 
                                         "_TupleType": Tuple, 
                                         "_gymnasium": True})
    _BaseGymnasiumSpaceConverter.__doc__ = __AuxBaseGymSpaceConverter.__doc__
    _BaseGymSpaceConverter = _BaseGymnasiumSpaceConverter
