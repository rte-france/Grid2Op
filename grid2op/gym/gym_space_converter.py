# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from gym import spaces
import numpy as np
import copy

from grid2op.Converter.Converters import Converter
from grid2op.gym.base_gym_attr_onverter import BaseGymAttrConverter
from grid2op.Environment import Environment
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int, dt_bool, dt_float


class _BaseGymSpaceConverter(spaces.Dict):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        Used as a base class to convert grid2op state to gym state (wrapper for some useful function
        for both the action space and the observation space).

    """
    def __init__(self, dict_gym_space, dict_variables=None):
        spaces.Dict.__init__(self, dict_gym_space)
        self._keys_encoding = {}
        if dict_variables is not None:
            for k, v in dict_variables.items():
                self._keys_encoding[k] = v

    @staticmethod
    def _generic_gym_space(dt, sh, low=None, high=None):
        if low is None:
            low = np.iinfo(dt).min
        if high is None:
            high = np.iinfo(dt).max
        shape = (sh,)
        my_type = spaces.Box(low=dt.type(low), high=dt.type(high), shape=shape, dtype=dt)
        return my_type

    @staticmethod
    def _boolean_type(sh):
        return spaces.MultiBinary(n=sh)

    @staticmethod
    def _extract_obj_grid2op(vect, dtype):
        if len(vect) == 1:
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
        res = spaces.dict.OrderedDict()
        for k in keys:
            conv_k = k
            if converter is not None:
                # for converting the names between internal names and "human readable names"
                conv_k = converter[k]

            obj_raw = obj._get_array_from_attr_name(conv_k)
            if k in self._keys_encoding:
                if self._keys_encoding[k] is None:
                    # keys is deactivated
                    continue
                else:
                    # i need to process the "function" part in the keys
                    obj_json_cleaned = self._keys_encoding[k].g2op_to_gym(obj_raw)
            else:
                obj_json_cleaned = self._extract_obj_grid2op(obj_raw, dtypes[k])
            res[k] = obj_json_cleaned
        return res

    def get_dict_encoding(self):
        return copy.deepcopy(self._keys_encoding)

    def reenc(self, key, fun):
        """
        shorthand for :func:`GymObservationSpace.reencode_space` or
        :func:`GymActionSpace.reencode_space`
        """
        return self.reenc(key, fun)

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

    def ignore_obs_attr(self, attr_names):
        """
        ignore some attribute names from the space
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        res = self
        for k in self.spaces.keys():
            if k in attr_names:
                res = res.reencode_observation_space(k, None)
        return res


class GymObservationSpace(_BaseGymSpaceConverter):
    # deals with the observation space (rather easy)
    """
    This class allows to transform the observation space into a gym space.

    Gym space will be a :class:`gym.spaces.Dict` with the keys being the different attributes
    of the grid2op observation. All attributes are used.

    Note that gym space converted with this class should be seeded independently. It is NOT seeded
    when calling :func:`grid2op.Environment.Environment.seed`.

    Examples
    --------
    Converting an observation space is fairly straightforward:

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymObservationSpace
        env = grid2op.make()

        gym_observation_space = GymObservationSpace(env.observation_space)
        # and now gym_observation_space is a `gym.spaces.Dict` representing the observation space

        # you can "convert" the grid2op observation to / from this space with:

        grid2op_obs = env.reset()
        same_gym_obs = gym_observation_space.to_gym(grid2op_obs)

        # the conversion from gym_obs to grid2op obs is feasible, but i don't imagine
        # a situation where it is useful. And especially, you will not be able to
        # use "obs.simulate" for the observation converted back from this gym action.

    """
    def __init__(self, env, dict_variables=None):
        self._init_env = env
        self.initial_obs_space = self._init_env.observation_space
        dict_ = {}  # will represent the gym.Dict space
        if dict_variables is None:
            dict_variables = {}
        self._fill_dict_obs_space(dict_,
                                  env.observation_space,
                                  env.parameters,
                                  env._oppSpace,
                                  dict_variables)
        _BaseGymSpaceConverter.__init__(self, dict_, dict_variables=dict_variables)

    def reencode_space(self, key, fun):
        """
        This function is used  to reencode the observation space. For example, it can be used to scale
        the observation into values close to 0., it can also be used to encode continuous variables into
        discrete variables or the other way around etc.

        Basically, it's a tool that lets you define your own observation space (there is the same for
        the action space)

        Parameters
        ----------
        key: ``str``
            Which part of the observation space you want to study

        fun: :class:`BaseGymAttrConverter`
            Put `None` to deactive the feature (it will be hided from the observation space)
            It can also be a `BaseGymAttrConverter`. See the example for more information.

        Returns
        -------
        self:
            The current instance, to be able to chain these calls

        Notes
        ------
        It modifies the observation space. We highly recommend to set it up at the beginning of your script
        and not to modify it afterwards

        'fun' should be deep copiable (meaning that if `copy.deepcopy(fun)` is called, then it does not crash

        If an attribute has been ignored, for example by :func`GymEnv.keep_only_obs_attr`
        or and is now present here, it will be re added in the final observation
        """

        my_dict = self.get_dict_encoding()
        if fun is not None and not isinstance(fun, BaseGymAttrConverter):
            raise RuntimeError("Impossible to initialize a converter with a function of type {}".format(type(fun)))
        my_dict[key] = fun
        res = GymObservationSpace(self._init_env, my_dict)
        return res

    def _fill_dict_obs_space(self, dict_, observation_space, env_params, opponent_space,
                             dict_variables={}):
        for attr_nm, sh, dt in zip(observation_space.attr_list_vect,
                                   observation_space.shape,
                                   observation_space.dtype):
            my_type = None
            shape = (sh,)
            if attr_nm in dict_variables:
                # case where the user specified a dedicated encoding
                if dict_variables[attr_nm] is None:
                    # none is by default to disable this feature
                    continue
                my_type = dict_variables[attr_nm].my_space

            elif dt == dt_int:
                # discrete observation space
                if attr_nm == "year":
                    my_type = spaces.Discrete(n=2100)
                elif attr_nm == "month":
                    my_type = spaces.Discrete(n=13)
                elif attr_nm == "day":
                    my_type = spaces.Discrete(n=32)
                elif attr_nm == "hour_of_day":
                    my_type = spaces.Discrete(n=24)
                elif attr_nm == "minute_of_hour":
                    my_type = spaces.Discrete(n=60)
                elif attr_nm == "day_of_week":
                    my_type = spaces.Discrete(n=8)
                elif attr_nm == "topo_vect":
                    my_type = spaces.Box(low=-1, high=2, shape=shape, dtype=dt)
                elif attr_nm == "time_before_cooldown_line":
                    my_type = spaces.Box(low=0,
                                         high=max(env_params.NB_TIMESTEP_COOLDOWN_LINE,
                                                  env_params.NB_TIMESTEP_RECONNECTION,
                                                  opponent_space.attack_cooldown
                                                  ),
                                         shape=shape,
                                         dtype=dt)
                elif attr_nm == "time_before_cooldown_sub":
                    my_type = spaces.Box(low=0,
                                         high=env_params.NB_TIMESTEP_COOLDOWN_SUB,
                                         shape=shape,
                                         dtype=dt)
                elif attr_nm == "duration_next_maintenance" or attr_nm == "time_next_maintenance":
                    # can be -1 if no maintenance, otherwise always positive
                    my_type = self._generic_gym_space(dt, sh, low=-1)

            elif dt == dt_bool:
                # boolean observation space
                my_type = self._boolean_type(sh)
            else:
                # continuous observation space
                low = float("-inf")
                high = float("inf")
                shape = (sh,)
                SpaceType = spaces.Box
                if attr_nm == "prod_p":
                    low = observation_space.gen_pmin
                    high = observation_space.gen_pmax*1.2  # because of the slack bus... # TODO
                    shape = None
                elif attr_nm == "prod_v" or attr_nm == "load_v" or attr_nm == "v_or" or attr_nm == "v_ex":
                    # voltages can't be negative
                    low = 0.
                elif attr_nm == "a_or" or attr_nm == "a_ex" or attr_nm == "rho":
                    # amps can't be negative
                    low = 0.
                elif attr_nm == "target_dispatch" or attr_nm == "actual_dispatch":
                    # TODO check that to be sure
                    low = np.minimum(observation_space.gen_pmin,
                                     -observation_space.gen_pmax)
                    high = np.maximum(-observation_space.gen_pmin,
                                      +observation_space.gen_pmax)
                my_type = SpaceType(low=low, high=high, shape=shape, dtype=dt)

            if my_type is None:
                # if nothing has been found in the specific cases above
                my_type = self._generic_gym_space(dt, sh)
            dict_[attr_nm] = my_type

    def from_gym(self, gymlike_observation: spaces.dict.OrderedDict) -> BaseObservation:
        """
        This function convert the gym-like representation of an observation to a grid2op observation.

        Parameters
        ----------
        gymlike_observation: :class:`gym.spaces.dict.OrderedDict`
            The observation represented as a gym ordered dict

        Returns
        -------
        grid2oplike_observation: :class:`grid2op.Observation.BaseObservation`
            The corresponding grid2op observation
        """
        res = self.initial_obs_space.get_empty_observation()
        for k, v in gymlike_observation.items():
            res._assign_attr_from_name(k, v)
        return res

    def to_gym(self, grid2op_observation: BaseObservation) -> spaces.dict.OrderedDict:
        """
        Convert a grid2op observation into a gym ordered dict.

        Parameters
        ----------
        grid2op_observation: :class:`grid2op.Observation.BaseObservation`
            The observation represented as a grid2op observation

        Returns
        -------
        gymlike_observation: :class:`gym.spaces.dict.OrderedDict`
           The corresponding gym ordered dict

        """
        return self._base_to_gym(self.spaces.keys(),
                                 grid2op_observation,
                                 dtypes={k: self.spaces[k].dtype for k in self.spaces}
                                 )


class GymActionSpace(_BaseGymSpaceConverter):
    """
    This class enables the conversion of the action space into a gym "space".

    Resulting action space will be a :class:`gym.spaces.Dict`.

    **NB** it is NOT recommended to use the sample of the gym action space. Please use the sampling (
    if availabe) of the original action space instead [if not available this means there is no
    implemented way to generate reliable random action]

    **Note** that gym space converted with this class should be seeded independently. It is NOT seeded
    when calling :func:`grid2op.Environment.Environment.seed`.

    Examples
    --------
    Converting an action space is fairly straightforward, though the resulting gym action space
    will depend on the original encoding of the action space.

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymActionSpace
        env = grid2op.make()

        gym_action_space = GymActionSpace(env.action_space)
        # and now gym_action_space is a `gym.spaces.Dict` representing the action space.
        # you can convert action to / from this space to grid2op the following way

        grid2op_act = env.action_space(...)
        gym_act = gym_action_space.to_gym(grid2op_act)

        # and the opposite conversion is also possible:
        gym_act = ... # whatever you decide to do
        grid2op_act = gym_action_space.from_gym(gym_act)

    **NB** you can use this `GymActionSpace` to  represent action into the gym format even if these actions
    comes from another converter, such as :class`IdToAct` or `ToVect` in this case, to get back a grid2op
    action you NEED to convert back the action from this converter. Here is a complete example
    on this (more advanced) usecase:

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymActionSpace, IdToAct
        env = grid2op.make()

        converted_action_space = IdToAct(env.action_space)
        gym_action_space = GymActionSpace(converted_action_space)

        # and now gym_action_space is a `gym.spaces.Dict` representing the action space.
        # you can convert action to / from this space to grid2op the following way

        converter_act = ... # whatever action you want
        gym_act = gym_action_space.to_gym(converter_act)

        # and the opposite conversion is also possible:
        gym_act = ... # whatever you decide to do
        converter_act = gym_action_space.from_gym(gym_act)

        # note that this converter act only makes sense for the converter. It cannot
        # be digest by grid2op directly. So you need to also convert it to grid2op
        grid2op_act = IdToAct.convert_act(converter_act)


    """
    # deals with the action space (it depends how it's encoded...)
    keys_grid2op_2_human = {"prod_p": "prod_p",
                            "prod_v": "prod_v",
                            "load_p": "load_p",
                            "load_q": "load_q",
                            "_redispatch": "redispatch",
                            "_set_line_status": "set_line_status",
                            "_switch_line_status": "change_line_status",
                            "_set_topo_vect": "set_bus",
                            "_change_bus_vect": "change_bus",
                            "_hazards": "hazards",
                            "_maintenance": "maintenance",
                            }
    keys_human_2_grid2op = {v: k for k, v in keys_grid2op_2_human.items()}

    def __init__(self, action_space, dict_variables=None):
        """
        note: for consistency with GymObservationSpace, "action_space" here can be an environment or
        an action space or a converter
        """
        if dict_variables is None:
            dict_variables = {}
        if isinstance(action_space, Environment):
            # action_space is an environment
            initial_act_space = action_space.action_space
        elif isinstance(action_space, ActionSpace):
            initial_act_space = action_space

        dict_ = {}
        # TODO Make sure it works well !
        if isinstance(initial_act_space, Converter):
            # a converter allows to ... convert the data so they have specific gym space
            dict_ = initial_act_space.get_gym_dict()
            self.__is_converter = True
        else:
            self._fill_dict_act_space(dict_, initial_act_space, dict_variables)
            dict_ = self._fix_dict_keys(dict_)
            self.__is_converter = False

        _BaseGymSpaceConverter.__init__(self, dict_, dict_variables)

    def reencode_space(self, key, fun):
        """
        This function is used  to reencode the action space. For example, it can be used to scale
        the observation into values close to 0., it can also be used to encode continuous variables into
        discrete variables or the other way around etc.

        Basically, it's a tool that lets you define your own observation space (there is the same for
        the action space)

        Parameters
        ----------
        key: ``str``
            Which part of the observation space you want to study

        fun: :class:`BaseGymAttrConverter`
            Put `None` to deactivate the feature (it will be hided from the observation space)
            It can also be a `BaseGymAttrConverter`. See the example for more information.

        Returns
        -------
        self:
            The current instance, to be able to chain these calls

        Notes
        ------
        It modifies the observation space. We highly recommend to set it up at the beginning of your script
        and not to modify it afterwards

        'fun' should be deep copiable (meaning that if `copy.deepcopy(fun)` is called, then it does not crash

        If an attribute has been ignored, for example by :func`GymEnv.keep_only_obs_attr`
        or and is now present here, it will be re added in the final observation
        """
        # TODO
        raise NotImplementedError("TODOOOOO")
        my_dict = self.get_dict_encoding()
        if fun is not None and not isinstance(fun, BaseGymAttrConverter):
            raise RuntimeError("Impossible to initialize a converter with a function of type {}".format(type(fun)))
        my_dict[key] = fun
        res = GymActionSpace(self._init_env, my_dict)
        return res

    def _fill_dict_act_space(self, dict_, action_space, dict_variables):
        # TODO what about dict_variables !!!
        for attr_nm, sh, dt in zip(action_space.attr_list_vect,
                                   action_space.shape,
                                   action_space.dtype):
            my_type = None
            shape = (sh,)
            if dt == dt_int:
                # discrete action space
                if attr_nm == "_set_line_status":
                    my_type = spaces.Box(low=-1,
                                         high=1,
                                         shape=shape,
                                         dtype=dt)
                elif attr_nm == "_set_topo_vect":
                    my_type = spaces.Box(low=-1,
                                         high=2,
                                         shape=shape,
                                         dtype=dt)
            elif dt == dt_bool:
                # boolean observation space
                my_type = self._boolean_type(sh)
                # case for all "change" action and maintenance / hazards
            else:
                # continuous observation space
                low = float("-inf")
                high = float("inf")
                shape = (sh,)
                SpaceType = spaces.Box

                if attr_nm == "prod_p":
                    low = action_space.gen_pmin
                    high = action_space.gen_pmax
                    shape = None
                elif attr_nm == "prod_v":
                    # voltages can't be negative
                    low = 0.
                elif attr_nm == "_redispatch":
                    # redispatch
                    low = -action_space.gen_max_ramp_down
                    high = action_space.gen_max_ramp_up
                my_type = SpaceType(low=low, high=high, shape=shape, dtype=dt)

            if my_type is None:
                # if nothing has been found in the specific cases above
                my_type = self._generic_gym_space(dt, sh)

            dict_[attr_nm] = my_type

    def _fix_dict_keys(self, dict_: dict) -> dict:
        res = {}
        for k, v in dict_.items():
            res[self.keys_grid2op_2_human[k]] = v
        return res

    def from_gym(self, gymlike_action: spaces.dict.OrderedDict) -> object:
        """
        Transform a gym-like action (such as the output of "sample()") into a grid2op action

        Parameters
        ----------
        gymlike_action: :class:`gym.spaces.dict.OrderedDict`
            The action, represented as a gym action (ordered dict)

        Returns
        -------
        An action that can be understood by the given action_space (either a grid2Op action if the
        original action space was used, or a Converter)

        """
        if self.__is_converter:
            # case where the action space comes from a converter, in this case the converter takes the
            # delegation to convert the action to openai gym
            res = self.initial_act_space.convert_action_from_gym(gymlike_action)
        else:
            # case where the action space is a "simple" action space
            res = self.initial_act_space()
            for k, v in gymlike_action.items():
                res._assign_attr_from_name(self.keys_human_2_grid2op[k], v)
        return res

    def to_gym(self, action: object) -> spaces.dict.OrderedDict:
        """
        Transform an action (non gym) into an action compatible with the gym Space.

        Parameters
        ----------
        action:
            The action (coming from grid2op or understandable by the converter)

        Returns
        -------
        gym_action:
            The same action converted as a OrderedDict (default used by gym in case of action space
            being Dict)

        """
        if self.__is_converter:
            gym_action = self.initial_act_space.convert_action_to_gym(action)
        else:
            # in that case action should be an instance of grid2op BaseAction
            assert isinstance(action, BaseAction), "impossible to convert an action not coming from grid2op"
            gym_action = self._base_to_gym(self.spaces.keys(),
                                           action,
                                           dtypes={k: self.spaces[k].dtype for k in self.spaces},
                                           converter=self.keys_human_2_grid2op
                                           )
        return gym_action
