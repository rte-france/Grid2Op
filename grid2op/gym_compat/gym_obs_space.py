# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
import numpy as np

from grid2op.Environment import (
    Environment,
    MultiMixEnvironment,
    BaseMultiProcessEnvironment,
)
from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE
if GYMNASIUM_AVAILABLE:
    from gymnasium import spaces  # only used for type hints
elif GYM_AVAILABLE:
    from gym import spaces
    
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.gym_compat.utils import _compute_extra_power_for_losses


class __AuxGymObservationSpace:
    """
    TODO explain gym / gymnasium
    
    This class allows to transform the observation space into a gym space.

    Gym space will be a :class:`gym.spaces.Dict` with the keys being the different attributes
    of the grid2op observation. All attributes are used.

    Note that gym space converted with this class should be seeded independently. It is NOT seeded
    when calling :func:`grid2op.Environment.Environment.seed`.

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`GymObservationSpace` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`GymnasiumObservationSpace`), otherwise it will
          inherit from gym (and will be exactly :class:`LegacyGymObservationSpace`)
        - :class:`GymnasiumObservationSpace` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`LegacyGymObservationSpace` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information

    Examples
    --------
    Converting an observation space is fairly straightforward:

    .. code-block:: python

        import grid2op
        from grid2op.Converter import GymObservationSpace
        env = grid2op.make("l2rpn_case14_sandbox")

        gym_observation_space = GymObservationSpace(env.observation_space)
        # and now gym_observation_space is a `gym.spaces.Dict` representing the observation space

        # you can "convert" the grid2op observation to / from this space with:

        grid2op_obs = env.reset()
        same_gym_obs = gym_observation_space.to_gym(grid2op_obs)

        # the conversion from gym_obs to grid2op obs is feasible, but i don't imagine
        # a situation where it is useful. And especially, you will not be able to
        # use "obs.simulate" for the observation converted back from this gym action.

    Notes
    -----
    The range of the values for "gen_p" / "prod_p" are not strictly `env.gen_pmin` and `env.gen_pmax`.
    This is due to the "approximation" when some redispatching is performed (the precision of the
    algorithm that computes the actual dispatch from the information it receives) and also because
    sometimes the losses of the grid are really different that the one anticipated in the "chronics" (yes
    `env.gen_pmin` and `env.gen_pmax` are not always ensured in grid2op)

    """
    ALLOWED_ENV_CLS = (Environment, MultiMixEnvironment, BaseMultiProcessEnvironment)
    def __init__(self, env, dict_variables=None):
        if not isinstance(
            env, type(self).ALLOWED_ENV_CLS + (type(self), )
        ):
            raise RuntimeError(
                "GymActionSpace must be created with an Environment of an ActionSpace (or a Converter)"
            )

        # self._init_env = env
        if isinstance(env, type(self).ALLOWED_ENV_CLS):
            init_env_cls = type(env)
            if init_env_cls._CLS_DICT_EXTENDED is None:
                # make sure the _CLS_DICT_EXTENDED exists
                tmp_ = {}
                init_env_cls._make_cls_dict_extended(init_env_cls, res=tmp_, as_list=False, copy_=False, _topo_vect_only=False)
            self.init_env_cls_dict = init_env_cls._CLS_DICT_EXTENDED.copy()
            # retrieve an empty observation an disable the forecast feature
            self.initial_obs = env.observation_space.get_empty_observation()
            self.initial_obs._obs_env = None
            self.initial_obs._ptr_kwargs_env = None
            
            self._tol_poly = env.observation_space.obs_env._tol_poly
            self._env_params = env.parameters
            self._opp_attack_max_duration = env._oppSpace.attack_max_duration
        elif isinstance(env, type(self)):
            self.init_env_cls_dict = env.init_env_cls_dict.copy()
            
            # retrieve an empty observation an disable the forecast feature
            self.initial_obs = env.initial_obs
            
            self._tol_poly = env._tol_poly
            self._env_params = env._env_params
            self._opp_attack_max_duration = env._opp_attack_max_duration
        else:
            raise RuntimeError("Unknown way to build a gym observation space")
        
        dict_ = {}  # will represent the gym.Dict space
        
        if dict_variables is None:
            # get the extra variables in the gym space I want to get
            dict_variables = {
                "thermal_limit":
                    type(self)._BoxType(
                        low=0.,
                        high=np.inf,
                        shape=(self.init_env_cls_dict["n_line"], ),
                        dtype=dt_float,
                    ),
                "theta_or":
                     type(self)._BoxType(
                        low=-180.,
                        high=180.,
                        shape=(self.init_env_cls_dict["n_line"], ),
                        dtype=dt_float,
                    ),
                "theta_ex":
                     type(self)._BoxType(
                        low=-180.,
                        high=180.,
                        shape=(self.init_env_cls_dict["n_line"], ),
                        dtype=dt_float,
                    ),
                "load_theta":
                     type(self)._BoxType(
                        low=-180.,
                        high=180.,
                        shape=(self.init_env_cls_dict["n_load"], ),
                        dtype=dt_float,
                    ),
                "gen_theta":
                     type(self)._BoxType(
                        low=-180.,
                        high=180.,
                        shape=(self.init_env_cls_dict["n_gen"], ),
                        dtype=dt_float,
                    )
                }
            if self.init_env_cls_dict["n_storage"]:
                dict_variables["storage_theta"] = type(self)._BoxType(
                        low=-180.,
                        high=180.,
                        shape=(self.init_env_cls_dict["n_storage"], ),
                        dtype=dt_float,
                    )
                
        self._fill_dict_obs_space(
            dict_, dict_variables
        )
        super().__init__(dict_, dict_variables=dict_variables) # super should point to _BaseGymSpaceConverter

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
        if fun is not None and not isinstance(fun, type(self)._BaseGymAttrConverterType):
            raise RuntimeError(
                "Impossible to initialize a converter with a function of type {}".format(
                    type(fun)
                )
            )

        if fun is not None and not fun.is_init_space():
            if key in my_dict:
                fun.initialize_space(my_dict[key])
            elif key in self.spaces:
                fun.initialize_space(self.spaces[key])
            else:
                raise RuntimeError(
                    f"Impossible to find key {key} in your observation space"
                )
        my_dict[key] = fun
        res = type(self)(self, my_dict)
        return res

    def _fill_dict_obs_space(
        self, dict_, dict_variables={}
    ):
        for attr_nm in dict_variables:
            # case where the user specified a dedicated encoding
            if dict_variables[attr_nm] is None:
                # none is by default to disable this feature
                continue
            if isinstance(dict_variables[attr_nm], type(self)._SpaceType):
                if hasattr(self.initial_obs, attr_nm):
                    # add it only if attribute exists in the observation
                    dict_[attr_nm] = dict_variables[attr_nm]
            else:
                dict_[attr_nm] = dict_variables[attr_nm].my_space
                
        # by default consider all attributes that are vectorized    
        for attr_nm, sh, dt in zip(
            type(self.initial_obs).attr_list_vect,
            self.initial_obs.shapes(),
            self.initial_obs.dtypes(),
        ):
            if sh == 0:
                # do not add "empty" (=0 dimension) arrays to gym otherwise it crashes
                continue
            
            if (attr_nm in dict_ or 
                (attr_nm in dict_variables and dict_variables[attr_nm] is None)):
                # variable already treated somewhere
                continue
            
            my_type = None
            shape = (sh,)
            if dt == dt_int:
                # discrete observation space
                if attr_nm == "year":
                    my_type = type(self)._DiscreteType(n=2100)
                elif attr_nm == "month":
                    my_type = type(self)._DiscreteType(n=13)
                elif attr_nm == "day":
                    my_type = type(self)._DiscreteType(n=32)
                elif attr_nm == "hour_of_day":
                    my_type = type(self)._DiscreteType(n=24)
                elif attr_nm == "minute_of_hour":
                    my_type = type(self)._DiscreteType(n=60)
                elif attr_nm == "day_of_week":
                    my_type = type(self)._DiscreteType(n=8)
                elif attr_nm == "topo_vect":
                    my_type = type(self)._BoxType(low=-1,
                                                  high=self.init_env_cls_dict["n_busbar_per_sub"],
                                                  shape=shape, dtype=dt)
                elif attr_nm == "time_before_cooldown_line":
                    my_type = type(self)._BoxType(
                        low=0,
                        high=max(
                            self._env_params.NB_TIMESTEP_COOLDOWN_LINE,
                            self._env_params.NB_TIMESTEP_RECONNECTION,
                            self._opp_attack_max_duration,
                        ),
                        shape=shape,
                        dtype=dt,
                    )
                elif attr_nm == "time_before_cooldown_sub":
                    my_type = type(self)._BoxType(
                        low=0,
                        high=self._env_params.NB_TIMESTEP_COOLDOWN_SUB,
                        shape=shape,
                        dtype=dt,
                    )
                elif (
                    attr_nm == "duration_next_maintenance"
                    or attr_nm == "time_next_maintenance"
                ):
                    # can be -1 if no maintenance, otherwise always positive
                    my_type = self._generic_gym_space(dt, sh, low=-1)
                elif attr_nm == "time_since_last_alarm":
                    # can be -1 if no maintenance, otherwise always positive
                    my_type = self._generic_gym_space(dt, 1, low=-1)
                elif attr_nm == "last_alarm":
                    # can be -1 if no maintenance, otherwise always positive
                    my_type = self._generic_gym_space(dt, sh, low=-1)
                elif attr_nm == "last_alert":
                    # can be -1 if no maintenance, otherwise always positive
                    my_type = self._generic_gym_space(dt, sh, low=-1)
                elif attr_nm == "was_alert_used_after_attack":
                    # can be -1 or >= 0
                    my_type = self._generic_gym_space(dt, sh, low=-1, high=1)
                elif attr_nm == "total_number_of_alert":
                    my_type = self._generic_gym_space(dt, sh, low=0)
                elif (attr_nm == "time_since_last_attack" or 
                      attr_nm == "time_since_last_alert"):
                    my_type = self._generic_gym_space(dt, sh, low=-1)
                elif attr_nm == "attack_under_alert":
                    my_type = self._generic_gym_space(dt, sh, low=-1, high=1)
                elif attr_nm == "alert_duration":
                    my_type = self._generic_gym_space(dt, sh, low=0)
                    
            elif dt == dt_bool:
                # boolean observation space
                if sh > 1:
                    my_type = self._boolean_type(sh)
                else:
                    my_type = type(self)._DiscreteType(n=2)
            else:
                # continuous observation space
                low = float("-inf")
                high = float("inf")
                shape = (sh,)
                SpaceType = type(self)._BoxType
                if attr_nm == "gen_p" or attr_nm == "gen_p_before_curtail":
                    low = copy.deepcopy(self.init_env_cls_dict["gen_pmin"])
                    high = copy.deepcopy(self.init_env_cls_dict["gen_pmax"])
                    shape = None

                    # for redispatching
                    low -= self._tol_poly
                    high += self._tol_poly

                    # for "power losses" that are not properly computed in the original data
                    extra_for_losses = _compute_extra_power_for_losses(
                        self.init_env_cls_dict
                    )
                    low -= extra_for_losses
                    high += extra_for_losses

                elif (
                    attr_nm == "gen_v"
                    or attr_nm == "load_v"
                    or attr_nm == "v_or"
                    or attr_nm == "v_ex"
                ):
                    # voltages can't be negative
                    low = 0.0
                elif attr_nm == "a_or" or attr_nm == "a_ex" or attr_nm == "rho":
                    # amps can't be negative
                    low = 0.0
                elif attr_nm == "target_dispatch" or attr_nm == "actual_dispatch":
                    # TODO check that to be sure
                    low = np.minimum(
                        self.init_env_cls_dict["gen_pmin"], -self.init_env_cls_dict["gen_pmax"]
                    )
                    high = np.maximum(
                        -self.init_env_cls_dict["gen_pmin"], +self.init_env_cls_dict["gen_pmax"]
                    )
                elif attr_nm == "storage_power" or attr_nm == "storage_power_target":
                    low = -self.init_env_cls_dict["storage_max_p_prod"]
                    high = self.init_env_cls_dict["storage_max_p_absorb"]
                elif attr_nm == "storage_charge":
                    low = np.zeros(self.init_env_cls_dict["n_storage"], dtype=dt_float)
                    high = self.init_env_cls_dict["storage_Emax"]
                elif (
                    attr_nm == "curtailment"
                    or attr_nm == "curtailment_limit"
                    or attr_nm == "curtailment_limit_effective"
                ):
                    low = 0.0
                    high = 1.0
                elif attr_nm == "attention_budget":
                    low = 0.0
                    high = np.inf
                elif attr_nm == "delta_time":
                    low = 0.0
                    high = np.inf
                elif attr_nm == "gen_margin_up":
                    low = 0.0
                    high = self.init_env_cls_dict["gen_max_ramp_up"]
                elif attr_nm == "gen_margin_down":
                    low = 0.0
                    high = self.init_env_cls_dict["gen_max_ramp_down"]
                    
                # curtailment, curtailment_limit, gen_p_before_curtail
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
        res = self.initial_obs.copy()
        for k, v in gymlike_observation.items():
            try:
                res._assign_attr_from_name(k, v)
            except ValueError as exc_:
                warnings.warn(f"Cannot set attribute \"{k}\" in grid2op. "
                              f"This key is ignored.")
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
        return self._base_to_gym(
            self.spaces.keys(),
            grid2op_observation,
            dtypes={k: self.spaces[k].dtype for k in self.spaces},
        )

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment


if GYM_AVAILABLE:
    from gym.spaces import (Discrete as LegGymDiscrete,
                            Box as LegGymBox,
                            Dict as LegGymDict,
                            Space as LegGymSpace,
                            MultiBinary as LegGymMultiBinary,
                            Tuple as LegGymTuple)
    from grid2op.gym_compat.gym_space_converter import _BaseLegacyGymSpaceConverter
    from grid2op.gym_compat.base_gym_attr_converter import BaseLegacyGymAttrConverter
    LegacyGymObservationSpace = type("LegacyGymObservationSpace",
                                     (__AuxGymObservationSpace, _BaseLegacyGymSpaceConverter, ),
                                     {"_DiscreteType": LegGymDiscrete,
                                      "_BoxType": LegGymBox,
                                      "_DictType": LegGymDict,
                                      "_SpaceType": LegGymSpace, 
                                      "_MultiBinaryType": LegGymMultiBinary, 
                                      "_TupleType": LegGymTuple, 
                                      "_BaseGymAttrConverterType": BaseLegacyGymAttrConverter,
                                      "_gymnasium": False,
                                      "__module__": __name__})
    LegacyGymObservationSpace.__doc__ = __AuxGymObservationSpace.__doc__
    GymObservationSpace = LegacyGymObservationSpace
    GymObservationSpace.__doc__ = __AuxGymObservationSpace.__doc__
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Discrete, Box, Dict, Space, MultiBinary, Tuple
    from grid2op.gym_compat.gym_space_converter import _BaseGymnasiumSpaceConverter
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    GymnasiumObservationSpace = type("GymnasiumObservationSpace",
                                     (__AuxGymObservationSpace, _BaseGymnasiumSpaceConverter, ),
                                     {"_DiscreteType": Discrete,
                                      "_BoxType": Box,
                                      "_DictType": Dict,
                                      "_SpaceType": Space, 
                                      "_MultiBinaryType": MultiBinary, 
                                      "_TupleType": Tuple, 
                                      "_BaseGymAttrConverterType": BaseGymnasiumAttrConverter,
                                      "_gymnasium": True,
                                      "__module__": __name__})
    GymnasiumObservationSpace.__doc__ = __AuxGymObservationSpace.__doc__
    GymObservationSpace = GymnasiumObservationSpace
    GymObservationSpace.__doc__ = __AuxGymObservationSpace.__doc__
    