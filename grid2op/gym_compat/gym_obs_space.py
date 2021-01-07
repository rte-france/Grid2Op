# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from gym import spaces


from grid2op.gym_compat.gym_space_converter import _BaseGymSpaceConverter
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int, dt_bool
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter


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