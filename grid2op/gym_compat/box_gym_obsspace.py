# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple
import copy
import warnings
import numpy as np

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Observation import ObservationSpace
from grid2op.Exceptions import Grid2OpException

from grid2op.gym_compat.utils import (_compute_extra_power_for_losses,
                                      GYM_AVAILABLE,
                                      GYMNASIUM_AVAILABLE,
                                      check_gym_version)

ALL_ATTR_OBS = (
    "year",
    "month",
    "day",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week",
    "gen_p",
    "gen_p_before_curtail",
    "gen_q",
    "gen_v",
    "gen_margin_up",
    "gen_margin_down",
    "load_p",
    "load_q",
    "load_v",
    "p_or",
    "q_or",
    "v_or",
    "a_or",
    "p_ex",
    "q_ex",
    "v_ex",
    "a_ex",
    "rho",
    "line_status",
    "timestep_overflow",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "time_next_maintenance",
    "duration_next_maintenance",
    "target_dispatch",
    "actual_dispatch",
    "storage_charge",
    "storage_power_target",
    "storage_power",
    "curtailment",
    "curtailment_limit",
    "curtailment_limit_effective",
    "thermal_limit",
    "is_alarm_illegal",
    "time_since_last_alarm",
    "last_alarm",
    "attention_budget",
    "was_alarm_used_after_game_over",
    "max_step",
    "active_alert",
    "attack_under_alert",
    "time_since_last_alert",
    "alert_duration",
    "total_number_of_alert",
    "time_since_last_attack",
    "was_alert_used_after_attack",
    "theta_or",
    "theta_ex",
    "load_theta",
    "gen_theta",
)

# TODO add the alarm stuff
# TODO add the time step
# TODO add the is_illegal and co there


class __AuxBoxGymObsSpace:
    """
    This class allows to convert a grid2op observation space into a gym "Box" which is
    a regular Box in R^d.

    It also allows to customize which part of the observation you want to use and offer capacity to
    center / reduce the data or to use more complex function from the observation.
    
    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`BoxGymObsSpace` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`BoxGymnasiumObsSpace`), otherwise it will
          inherit from gym (and will be exactly :class:`BoxLegacyGymObsSpace`)
        - :class:`BoxGymnasiumObsSpace` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`BoxLegacyGymObsSpace` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
    
    .. note::
        A gymnasium Box is encoded as a numpy array.
        
    Examples
    --------
    If you simply want to use it you can do:

    .. code-block:: python

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv, BoxGymObsSpace
        gym_env = GymEnv(env)

        gym_env.observation_space = BoxGymObsSpace(env.observation_space)

    In this case it will extract all the features in all the observation (a detailed list is given
    in the documentation at :ref:`observation_module`.

    You can select the attribute you want to keep, for example:

    .. code-block:: python

        gym_env.observation_space = BoxGymObsSpace(env.observation_space,
                                                   attr_to_keep=['load_p', "gen_p", "rho])

    You can also apply some basic transformation to the attribute of the observation before building
    the resulting gym observation (which in this case is a vector). This can be done with:

    .. code-block:: python

        gym_env.observation_space = BoxGymObsSpace(env.observation_space,
                                                   attr_to_keep=['load_p', "gen_p", "rho"],
                                                   divide={"gen_p": env.gen_pmax},
                                                   substract={"gen_p": 0.5 * env.gen_pmax})

    In the above example, the resulting "gen_p" part of the vector will be given by the following
    formula: `gym_obs = (grid2op_obs - substract) / divide`.

    Hint: you can use: divide being the standard deviation and subtract being the average of the attribute
    on a few episodes for example. This can be done with :class:`grid2op.utils.EpisodeStatistics` for example.

    Finally, you can also modify more the attribute of the observation and add it to your box. This
    can be done rather easily with the "functs" argument like:

    .. code-block:: python

        gym_env.observation_space = BoxGymObsSpace(env.observation_space,
                                                   attr_to_keep=["connectivity_matrix", "log_load"],
                                                   functs={"connectivity_matrix":
                                                              (lambda grid2opobs: grid2opobs.connectivity_matrix().flatten(),
                                                               0., 1.0, None, None),
                                                           "log_load":
                                                            (lambda grid2opobs: np.log(grid2opobs.load_p),
                                                            None, 10., None, None)
                                                        }
                                                   )

    In this case, "functs" should be a dictionary, the "keys" should be string (keys should also be
    present in the `attr_to_keep` list) and the values should count 5 elements
    (callable, low, high, shape, dtype) with:

    - `callable` a function taking as input a grid2op observation and returning a numpy array
    - `low` (optional) (put None if you don't want to specify it, defaults to `-np.inf`) the lowest value
      your numpy array can take. It can be a single number or an array with the same shape
      as the return value of your function.
    - `high` (optional) (put None if you don't want to specify it, defaults to `np.inf`) the highest value
      your numpy array can take. It can be a single number or an array with the same shape
      as the return value of your function.
    - `shape` (optional) (put None if you don't want to specify it) the shape of the return value
      of your function. It should be a tuple (and not a single number). By default it is computed
      with by applying your function to an observation.
    - `dtype` (optional, put None if you don't want to change it, defaults to np.float32) the type of
      the numpy array as output of your function.

    Notes
    -----
    The range of the values for "gen_p" / "prod_p" are not strictly `env.gen_pmin` and `env.gen_pmax`.
    This is due to the "approximation" when some redispatching is performed (the precision of the
    algorithm that computes the actual dispatch from the information it receives) and also because
    sometimes the losses of the grid are really different that the one anticipated in the "chronics" (yes
    env.gen_pmin and env.gen_pmax are not always ensured in grid2op)

    """

    def __init__(
        self,
        grid2op_observation_space,
        attr_to_keep=ALL_ATTR_OBS,
        subtract=None,
        divide=None,
        functs=None,
    ):
        check_gym_version(type(self)._gymnasium)
        if not isinstance(grid2op_observation_space, ObservationSpace):
            raise RuntimeError(
                f"Impossible to create a BoxGymObsSpace without providing a "
                f"grid2op observation. You provided {type(grid2op_observation_space)}"
                f'as the "grid2op_observation_space" attribute.'
            )
        self._attr_to_keep = sorted(attr_to_keep)

        ob_sp = grid2op_observation_space
        tol_redisp = (
            ob_sp.obs_env._tol_poly
        )  # add to gen_p otherwise ... well it can crash
        extra_for_losses = _compute_extra_power_for_losses(ob_sp)
        
        self._dict_properties = {
            "year": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 2200,
                (1,),
                dt_int,
            ),
            "month": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 12,
                (1,),
                dt_int,
            ),
            "day": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 31,
                (1,),
                dt_int,
            ),
            "hour_of_day": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 24,
                (1,),
                dt_int,
            ),
            "minute_of_hour": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 60,
                (1,),
                dt_int,
            ),
            "day_of_week": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + 7,
                (1,),
                dt_int,
            ),
            "current_step": (
                np.zeros(1, dtype=dt_int),
                np.zeros(1, dtype=dt_int) + np.iinfo(dt_int).max,
                (1,),
                dt_int,
            ),
            "gen_p": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float)
                - tol_redisp
                - extra_for_losses,
                ob_sp.gen_pmax + tol_redisp + extra_for_losses,
                (ob_sp.n_gen,),
                dt_float,
            ),
            "gen_q": (
                np.full(shape=(ob_sp.n_gen,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_gen,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "gen_v": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_gen,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "gen_margin_up": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                1.0 * ob_sp.gen_max_ramp_up,
                (ob_sp.n_gen,),
                dt_float,
            ),
            "gen_margin_down": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                1.0 * ob_sp.gen_max_ramp_down,
                (ob_sp.n_gen,),
                dt_float,
            ),
            "gen_theta": (
                np.full(shape=(ob_sp.n_gen,), fill_value=-180., dtype=dt_float),
                np.full(shape=(ob_sp.n_gen,), fill_value=180., dtype=dt_float),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "load_p": (
                np.full(shape=(ob_sp.n_load,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_load,), fill_value=+np.inf, dtype=dt_float),
                (ob_sp.n_load,),
                dt_float,
            ),
            "load_q": (
                np.full(shape=(ob_sp.n_load,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_load,), fill_value=+np.inf, dtype=dt_float),
                (ob_sp.n_load,),
                dt_float,
            ),
            "load_v": (
                np.full(shape=(ob_sp.n_load,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_load,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_load,),
                dt_float,
            ),
            "load_theta": (
                np.full(shape=(ob_sp.n_load,), fill_value=-180., dtype=dt_float),
                np.full(shape=(ob_sp.n_load,), fill_value=180., dtype=dt_float),
                (ob_sp.n_load,),
                dt_float,
            ),
            "p_or": (
                np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "q_or": (
                np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "a_or": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "v_or": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "theta_or": (
                np.full(shape=(ob_sp.n_line,), fill_value=-180., dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=180., dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "p_ex": (
                np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "q_ex": (
                np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "a_ex": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "v_ex": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "theta_ex": (
                np.full(shape=(ob_sp.n_line,), fill_value=-180., dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=180., dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "rho": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "line_status": (
                np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                np.full(shape=(ob_sp.n_line,), fill_value=1, dtype=dt_int),
                (ob_sp.n_line,),
                dt_int,
            ),
            "timestep_overflow": (
                np.full(
                    shape=(ob_sp.n_line,), fill_value=np.iinfo(dt_int).min, dtype=dt_int
                ),
                np.full(
                    shape=(ob_sp.n_line,), fill_value=np.iinfo(dt_int).max, dtype=dt_int
                ),
                (ob_sp.n_line,),
                dt_int,
            ),
            "topo_vect": (
                np.full(shape=(ob_sp.dim_topo,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_topo,), fill_value=2, dtype=dt_int),
                (ob_sp.dim_topo,),
                dt_int,
            ),
            "time_before_cooldown_line": (
                np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                np.full(
                    shape=(ob_sp.n_line,), fill_value=np.iinfo(dt_int).max, dtype=dt_int
                ),
                (ob_sp.n_line,),
                dt_int,
            ),
            "time_before_cooldown_sub": (
                np.full(shape=(ob_sp.n_sub,), fill_value=0, dtype=dt_int),
                np.full(
                    shape=(ob_sp.n_sub,), fill_value=np.iinfo(dt_int).max, dtype=dt_int
                ),
                (ob_sp.n_sub,),
                dt_int,
            ),
            "time_next_maintenance": (
                np.full(shape=(ob_sp.n_line,), fill_value=-1, dtype=dt_int),
                np.full(
                    shape=(ob_sp.n_line,), fill_value=np.iinfo(dt_int).max, dtype=dt_int
                ),
                (ob_sp.n_line,),
                dt_int,
            ),
            "duration_next_maintenance": (
                np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                np.full(
                    shape=(ob_sp.n_line,), fill_value=np.iinfo(dt_int).max, dtype=dt_int
                ),
                (ob_sp.n_line,),
                dt_int,
            ),
            "target_dispatch": (
                np.minimum(ob_sp.gen_pmin, -ob_sp.gen_pmax),
                np.maximum(-ob_sp.gen_pmin, +ob_sp.gen_pmax),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "actual_dispatch": (
                np.minimum(ob_sp.gen_pmin, -ob_sp.gen_pmax),
                np.maximum(-ob_sp.gen_pmin, +ob_sp.gen_pmax),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "storage_charge": (
                np.full(shape=(ob_sp.n_storage,), fill_value=0, dtype=dt_float),
                1.0 * ob_sp.storage_Emax,
                (ob_sp.n_storage,),
                dt_float,
            ),
            "storage_power_target": (
                -1.0 * ob_sp.storage_max_p_prod,
                1.0 * ob_sp.storage_max_p_absorb,
                (ob_sp.n_storage,),
                dt_float,
            ),
            "storage_power": (
                -1.0 * ob_sp.storage_max_p_prod,
                1.0 * ob_sp.storage_max_p_absorb,
                (ob_sp.n_storage,),
                dt_float,
            ),
            "storage_theta": (
                np.full(shape=(ob_sp.n_storage,), fill_value=-180., dtype=dt_float),
                np.full(shape=(ob_sp.n_storage,), fill_value=180., dtype=dt_float),
                (ob_sp.n_storage,),
                dt_float,
            ),
            "curtailment": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_gen,), fill_value=1.0, dtype=dt_float),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "curtailment_limit": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_gen,), fill_value=1.0, dtype=dt_float),
                (ob_sp.n_gen,),
                dt_float,
            ),
            "curtailment_mw": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                1.0 * ob_sp.gen_pmax,
                (ob_sp.n_gen,),
                dt_float,
            ),
            "curtailment_limit_mw": (
                np.full(shape=(ob_sp.n_gen,), fill_value=0.0, dtype=dt_float),
                1.0 * ob_sp.gen_pmax,
                (ob_sp.n_gen,),
                dt_float,
            ),
            "thermal_limit": (
                np.full(shape=(ob_sp.n_line,), fill_value=0.0, dtype=dt_float),
                np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                (ob_sp.n_line,),
                dt_float,
            ),
            "is_alarm_illegal": (
                np.full(shape=(1,), fill_value=False, dtype=dt_bool),
                np.full(shape=(1,), fill_value=True, dtype=dt_bool),
                (1,),
                dt_bool,
            ),
            "time_since_last_alarm": (
                np.full(shape=(1,), fill_value=-1, dtype=dt_int),
                np.full(shape=(1,), fill_value=np.iinfo(dt_int).max, dtype=dt_int),
                (1,),
                dt_int,
            ),
            "last_alarm": (
                np.full(shape=(ob_sp.dim_alarms,), fill_value=-1, dtype=dt_int),
                np.full(
                    shape=(ob_sp.dim_alarms,),
                    fill_value=np.iinfo(dt_int).max,
                    dtype=dt_int,
                ),
                (ob_sp.dim_alarms,),
                dt_int,
            ),
            "attention_budget": (
                np.full(shape=(1,), fill_value=-1, dtype=dt_float),
                np.full(shape=(1,), fill_value=np.inf, dtype=dt_float),
                (1,),
                dt_float,
            ),
            "was_alarm_used_after_game_over": (
                np.full(shape=(1,), fill_value=False, dtype=dt_bool),
                np.full(shape=(1,), fill_value=True, dtype=dt_bool),
                (1,),
                dt_bool,
            ),
            "delta_time": (
                np.full(shape=(1,), fill_value=0, dtype=dt_float),
                np.full(shape=(1,), fill_value=np.inf, dtype=dt_float),
                (1,),
                dt_float,
            ),
            # alert stuff
            "active_alert": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=False, dtype=dt_bool),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=True, dtype=dt_bool),
                (ob_sp.dim_alerts,),
                dt_bool,
            ),
            "time_since_last_alert": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=np.iinfo(dt_int).max, dtype=dt_int),
                (ob_sp.dim_alerts,),
                dt_int,
            ),
            "alert_duration": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=np.iinfo(dt_int).max, dtype=dt_int),
                (ob_sp.dim_alerts,),
                dt_int,
            ),
            "total_number_of_alert": (
                np.full(shape=(1 if ob_sp.dim_alerts else 0,), fill_value=-1, dtype=dt_int),
                np.full(shape=(1 if ob_sp.dim_alerts else 0,), fill_value=np.iinfo(dt_int).max, dtype=dt_int),
                (1 if ob_sp.dim_alerts else 0,),
                dt_int,
            ),
            "time_since_last_attack": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=np.iinfo(dt_int).max, dtype=dt_int),
                (ob_sp.dim_alerts,),
                dt_int,
            ),
            "was_alert_used_after_attack": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=1, dtype=dt_int),
                (ob_sp.dim_alerts,),
                dt_int,
            ),
            "attack_under_alert": (
                np.full(shape=(ob_sp.dim_alerts,), fill_value=-1, dtype=dt_int),
                np.full(shape=(ob_sp.dim_alerts,), fill_value=1, dtype=dt_int),
                (ob_sp.dim_alerts,),
                dt_int,
            ),
        }
        self._dict_properties["max_step"] = copy.deepcopy(self._dict_properties["current_step"])
        self._dict_properties["delta_time"] = copy.deepcopy(self._dict_properties["current_step"])
        self._dict_properties["prod_p"] = copy.deepcopy(self._dict_properties["gen_p"])
        self._dict_properties["prod_q"] = copy.deepcopy(self._dict_properties["gen_q"])
        self._dict_properties["prod_v"] = copy.deepcopy(self._dict_properties["gen_v"])
        self._dict_properties["gen_p_before_curtail"] = copy.deepcopy(self._dict_properties["gen_p"])
        self._dict_properties["curtailment_limit_effective"] = copy.deepcopy(self._dict_properties[
            "curtailment_limit"
        ])
        
        if functs is None:
            functs = {}
            
        for key in functs.keys():
            if key not in self._attr_to_keep:
                raise RuntimeError(
                    f'The key {key} is present in the "functs" dictionary but not in the '
                    f'"attr_to_keep". This is not consistent: either ignore this function, '
                    f'in that case remove "{key}" from "functs" or you want to add '
                    f'something to your observation, in that case add it to "attr_to_keep"'
                )

        if subtract is None:
            subtract = {}
        self._subtract = subtract.copy()
        if divide is None:
            divide = {}
        self._divide = divide.copy()

        # handle the "functional" part
        self._template_obs = ob_sp._template_obj.copy()
        self.__func = {}

        self._dims = None
        low, high, shape, dtype = self._get_info(functs)

        # initialize the base container
        type(self)._BoxType.__init__(self, low=low, high=high, shape=shape, dtype=dtype)
        
        # convert data in `_subtract` and `_divide` to the right type
        self._fix_value_sub_div(self._subtract, functs)
        self._fix_value_sub_div(self._divide, functs)

    def _get_shape(self, el, functs):
        if el in functs:
            callable_, low_, high_, shape_, dtype_ = functs[el]
        elif el in self._dict_properties:
            # el is an attribute of an observation, for example "load_q" or "topo_vect"
            low_, high_, shape_, dtype_ = self._dict_properties[el]
        return shape_
    
    def _fix_value_sub_div(self, dict_, functs):
        """dict_ is either self._subtract or self._divide"""
        keys = list(dict_.keys())
        for k in keys:
            v = dict_[k]
            if isinstance(v, (list, tuple)):
                v = np.array(v).astype(self.dtype)
            else:
                shape = self._get_shape(k, functs)
                v = np.full(shape, fill_value=v, dtype=self.dtype)
            dict_[k] = v
        
    def _get_info(self, functs):
        low = None
        high = None
        shape = None
        dtype = None
        self._dims = []
        for el in self._attr_to_keep:
            if el in functs:
                # the attribute name "el" has been put in the functs
                try:
                    callable_, low_, high_, shape_, dtype_ = functs[el]
                except Exception as exc_:
                    raise RuntimeError(
                        f'When using keyword argument "functs" you need to provide something '
                        f"like: (callable_, low_, high_, shape_, dtype_) for each key. "
                        f'There was an error with "{el}".'
                        f"The error was:\n {exc_}"
                    )

                try:
                    tmp = callable_(self._template_obs.copy())
                except Exception as exc_:
                    raise RuntimeError(
                        f'Error for the function your provided with key "{el}" (using the'
                        f'"functs" dictionary) '
                        f"The error was :\n {exc_}"
                    )
                if not isinstance(tmp, np.ndarray):
                    raise RuntimeError(
                        f'The result of the function you provided as part of the "functs"'
                        f"dictionary for key {el}"
                        f"do not return a numpy array. This is not supported."
                    )
                self.__func[el] = callable_
                if dtype_ is None:
                    dtype_ = dt_float
                if shape_ is None:
                    shape_ = tmp.shape

                if not isinstance(shape_, tuple):
                    raise RuntimeError(
                        "You need to provide a tuple as a shape of the output of your data"
                    )

                if low_ is None:
                    low_ = np.full(shape_, fill_value=-np.inf, dtype=dtype_)
                elif isinstance(low_, float):
                    low_ = np.full(shape_, fill_value=low_, dtype=dtype_)

                if high_ is None:
                    high_ = np.full(shape_, fill_value=np.inf, dtype=dtype_)
                elif isinstance(high_, float):
                    high_ = np.full(shape_, fill_value=high_, dtype=dtype_)

                if ((tmp < low_) | (tmp > high_)).any():
                    raise RuntimeError(
                        f"Wrong value for low / high in the functs argument for key {el}. Please"
                        f"fix the low_ / high_ in the tuple ( callable_, low_, high_, shape_, dtype_)."
                    )

            elif el in self._dict_properties:
                # el is an attribute of an observation, for example "load_q" or "topo_vect"
                low_, high_, shape_, dtype_ = self._dict_properties[el]
            else:
                li_keys = "\n\t-".join(
                    sorted(list(self._dict_properties.keys()) + list(self.__func.keys()))
                )
                raise RuntimeError(
                    f'Unknown observation attributes "{el}". Supported attributes are: '
                    f"\n{li_keys}"
                )

            # handle the data type
            if dtype is None:
                dtype = dtype_
            else:
                if dtype_ == dt_float:
                    # promote whatever to float anyway
                    dtype = dt_float
                elif dtype_ == dt_int and dtype == dt_bool:
                    # promote bool to int
                    dtype = dt_int

            # handle the shape
            if shape is None:
                shape = shape_
            else:
                shape = (shape[0] + shape_[0],)

            # handle low / high
            if el in self._subtract:
                low_ =  1.0 * low_.astype(dtype)
                high_ =  1.0 * high_.astype(dtype)
                low_ -= self._subtract[el]
                high_ -= self._subtract[el]
                
            if el in self._divide:
                low_ =  1.0 * low_.astype(dtype)
                high_ =  1.0 * high_.astype(dtype)
                low_ /= self._divide[el]
                high_ /= self._divide[el]
            if low is None:
                low = low_
                high = high_
            else:
                low = np.concatenate((low.astype(dtype), low_.astype(dtype))).astype(
                    dtype
                )
                high = np.concatenate((high.astype(dtype), high_.astype(dtype))).astype(
                    dtype
                )

            # remember where this need to be stored
            self._dims.append(shape[0])

        return low, high, shape, dtype

    def _handle_attribute(self, grid2op_observation, attr_nm):
        res = getattr(grid2op_observation, attr_nm).astype(self.dtype)
        if attr_nm in self._subtract:
            res -= self._subtract[attr_nm]
        if attr_nm in self._divide:
            res /= self._divide[attr_nm]
        return res

    def to_gym(self, grid2op_observation):
        """
        This is the function that is called to transform a grid2Op observation, sent by the grid2op environment
        and convert it to a numpy array (an element of a gym Box)

        Parameters
        ----------
        grid2op_observation:
            The grid2op observation (as a grid2op object)

        Returns
        -------
        res: :class:`numpy.ndarray`
            A numpy array compatible with the openAI gym Box that represents the action space.

        """
        res = np.empty(shape=self.shape, dtype=self.dtype)
        prev = 0
        for attr_nm, where_to_put in zip(self._attr_to_keep, self._dims):
            if attr_nm in self.__func:
                tmp = self.__func[attr_nm](grid2op_observation)
            elif hasattr(grid2op_observation, attr_nm):
                tmp = self._handle_attribute(grid2op_observation, attr_nm)
            else:
                raise RuntimeError(f'Unknown attribute "{attr_nm}".')
            res[prev:where_to_put] = tmp
            prev = where_to_put
        return res

    def close(self):
        pass

    def get_indexes(self, key: str) -> Tuple[int, int]:
        """Allows to retrieve the indexes of the gym action that
        are concerned by the attribute name `key` given in input.

        .. versionadded:: 1.9.3
        
        .. warning::
            Copy paste from box_gym_act_space, need refacto !
            
        Parameters
        ----------
        key : str
            the attribute name (*eg* "set_storage" or "redispatch")

        Returns
        -------
        Tuple[int, int]
            _description_

        Examples
        --------
        
        You can use it like:
        
        .. code-block:: python
        
            gym_env = ... # an environment with a BoxActSpace
            
            act = np.zeros(gym_env.action_space.shape)
            key = "redispatch"  # "redispatch", "curtail", "set_storage"
            start_, end_ = gym_env.action_space.get_indexes(key)
            act[start_:end_] = np.random.uniform(high=1, low=-1, size=env.gen_redispatchable.sum())
            # act only modifies the redispatch with the input given (here a uniform redispatching between -1 and 1)
            
        """
        error_msg =(f"Impossible to use the grid2op action property \"{key}\""
                    f"with this action space.")
        if key not in self._attr_to_keep:
            raise Grid2OpException(error_msg)
        prev = 0
        for attr_nm, where_to_put in zip(
            self._attr_to_keep, self._dims
        ):
            if attr_nm == key:
                return prev, where_to_put
            prev = where_to_put
        raise Grid2OpException(error_msg)
    
    def normalize_attr(self, attr_nm: str):
        """
        This function normalizes the part of the space
        that corresponds to the attribute `attr_nm`.

        The normalization consists in having a vector between 0. and 1.
        It is achieved by:
        
        - dividing by the range (high - low)
        - adding the minimum value (low).

        .. note::
            It only affects continuous attribute. No error / warnings are
            raised if you attempt to use it on a discrete attribute.

        .. warning::
            This normalization relies on the `high` and `low` attribute. It cannot be done if
            the attribute is not bounded (for example when its maximum limit is `np.inf`). A warning
            is raised in this case.
            
        Parameters
        ----------
        attr_nm : `str`
            The name of the attribute to normalize
        """
        if attr_nm in self._divide or attr_nm in self._subtract:
            raise Grid2OpException(
                f"Cannot normalize attribute \"{attr_nm}\" that you already "
                f"modified with either `divide` or `subtract` (observation space)."
            )
        prev = 0
        if self.dtype != dt_float:
            raise Grid2OpException(
                "Cannot normalize attribute with a observation "
                "space that is not float !"
            )
        for attr_tmp, where_to_put in zip(self._attr_to_keep, self._dims):
            if attr_tmp == attr_nm:
                curr_high = 1.0 * self.high[prev:where_to_put]
                curr_low = 1.0 * self.low[prev:where_to_put]
                finite_high = np.isfinite(curr_high)
                finite_low = np.isfinite(curr_high)
                both_finite = finite_high & finite_low
                both_finite &= curr_high > curr_low

                if (~both_finite).any():
                    warnings.warn(f"The normalization of attribute \"{attr_nm}\" cannot be performed entirely as "
                                  f"there are some non finite value, or `high == `low` "
                                  f"for some components.")
                    
                self._divide[attr_nm] = np.ones(curr_high.shape, dtype=self.dtype)
                self._subtract[attr_nm] = np.zeros(curr_high.shape, dtype=self.dtype)

                self._divide[attr_nm][both_finite] = (
                    curr_high[both_finite] - curr_low[both_finite]
                )
                self._subtract[attr_nm][both_finite] += curr_low[both_finite]

                self.high[prev:where_to_put][both_finite] = 1.0
                self.low[prev:where_to_put][both_finite] = 0.0
                break
            prev = where_to_put


if GYM_AVAILABLE:
    from gym.spaces import Box as LegGymBox
    from grid2op.gym_compat.base_gym_attr_converter import BaseLegacyGymAttrConverter
    BoxLegacyGymObsSpace = type("BoxLegacyGymObsSpace",
                                (__AuxBoxGymObsSpace, LegGymBox, ),
                                {"_gymnasium": False,
                                 "_BaseGymAttrConverterType": BaseLegacyGymAttrConverter,
                                 "_BoxType": LegGymBox,
                                 "__module__": __name__})
    BoxLegacyGymObsSpace.__doc__ = __AuxBoxGymObsSpace.__doc__
    BoxGymObsSpace = BoxLegacyGymObsSpace
    BoxGymObsSpace.__doc__ = __AuxBoxGymObsSpace.__doc__
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Box
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    BoxGymnasiumObsSpace = type("BoxGymnasiumObsSpace",
                                (__AuxBoxGymObsSpace, Box, ),
                                {"_gymnasium": True,
                                 "_BaseGymAttrConverterType": BaseGymnasiumAttrConverter,
                                 "_BoxType": Box,
                                 "__module__": __name__})
    BoxGymnasiumObsSpace.__doc__ = __AuxBoxGymObsSpace.__doc__
    BoxGymObsSpace = BoxGymnasiumObsSpace
    BoxGymObsSpace.__doc__ = __AuxBoxGymObsSpace.__doc__
