# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from typing import Tuple, Union, List
from grid2op.Environment.environment import Environment
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Exceptions import EnvError
from grid2op.dtypes import dt_bool, dt_float, dt_int


class MaskedEnvironment(Environment):  # TODO heritage ou alors on met un truc de base
    """This class is the grid2op implementation of a "maked" environment: lines not in the 
    `lines_of_interest` mask will NOT be deactivated by the environment is the flow is too high
    (or moderately high for too long.)
    
    .. warning::
        This class might not behave normally if used with TimeOutEnvironment, MultiEnv, MultiMixEnv etc.
    
    .. warning::
        At time of writing, the behaviour of "obs.simulate" is not modified
    """  
    # some kind of infinity value
    # NB we multiply np.finfo(dt_float).max by a small number (1e-7) to avoid overflow
    # indeed, _hard_overflow_threshold is multiply by the flow on the lines
    INF_VAL_THM_LIM = 1e-7 * np.finfo(dt_float).max  
    
    # some kind of infinity value
    INF_VAL_TS_OVERFLOW_ALLOW = np.iinfo(dt_int).max - 1  
    
    def __init__(self,
                 grid2op_env: Union[Environment, dict],
                 lines_of_interest):
        
        self._lines_of_interest = self._make_lines_of_interest(lines_of_interest)
        if isinstance(grid2op_env, Environment):
            super().__init__(**grid2op_env.get_kwargs())
        elif isinstance(grid2op_env, dict):
            super().__init__(**grid2op_env)
        else:
            raise EnvError(f"For MaskedEnvironment you need to provide "
                           f"either an Environment or a dict "
                           f"for grid2op_env. You provided: {type(grid2op_env)}")
        
    def _make_lines_of_interest(self, lines_of_interest):
        # NB is called BEFORE the env has been created...
        if isinstance(lines_of_interest, np.ndarray):
            # if lines_of_interest.size() != type(self).n_line:
                # raise EnvError("Impossible to init A masked environment when the number of lines "
                            #    "of the mask do not match the number of lines on the grid.")
            res = lines_of_interest.astype(dt_bool)
            if res.sum() == 0:
                raise EnvError("You cannot use MaskedEnvironment and masking all "
                               "the grid. If you don't want to simulate powerline "
                               "disconnection when they are game over, please "
                               "set params.NO_OVERFLOW_DISCONNECT=True (see doc)")
        else:
            raise EnvError("Format of lines_of_interest is not understood. "
                           "Please provide a vector of the size of the "
                           "number of lines on the grid.")
        return res
    
    def _reset_vectors_and_timings(self):
        super()._reset_vectors_and_timings()
        self._hard_overflow_threshold[~self._lines_of_interest] = type(self).INF_VAL_THM_LIM
        self._nb_timestep_overflow_allowed[~self._lines_of_interest] = type(self).INF_VAL_TS_OVERFLOW_ALLOW

    def get_kwargs(self, with_backend=True, with_chronics_handler=True):
        res = {}
        res["lines_of_interest"] = copy.deepcopy(self._lines_of_interest)
        res["grid2op_env"] = super().get_kwargs(with_backend, with_chronics_handler)
        return res

    def get_params_for_runner(self):
        res = super().get_params_for_runner()
        res["envClass"] = MaskedEnvironment
        res["other_env_kwargs"] = {"lines_of_interest": copy.deepcopy(self._lines_of_interest)}
        return res

    def _custom_deepcopy_for_copy(self, new_obj):
        super()._custom_deepcopy_for_copy(new_obj)
        new_obj._lines_of_interest = copy.deepcopy(self._lines_of_interest)
    
    @classmethod
    def init_obj_from_kwargs(cls,
                             other_env_kwargs,
                             init_env_path,
                             init_grid_path,
                             chronics_handler,
                             backend,
                             parameters,
                             name,
                             names_chronics_to_backend,
                             actionClass,
                             observationClass,
                             rewardClass,
                             legalActClass,
                             voltagecontrolerClass,
                             other_rewards,
                             opponent_space_type,
                             opponent_action_class,
                             opponent_class,
                             opponent_init_budget,
                             opponent_budget_per_ts,
                             opponent_budget_class,
                             opponent_attack_duration,
                             opponent_attack_cooldown,
                             kwargs_opponent,
                             with_forecast,
                             attention_budget_cls,
                             kwargs_attention_budget,
                             has_attention_budget,
                             logger,
                             kwargs_observation,
                             observation_bk_class,
                             observation_bk_kwargs,
                             _raw_backend_class,
                             _read_from_local_dir):
        res = MaskedEnvironment(grid2op_env={"init_env_path": init_env_path,
                                             "init_grid_path": init_grid_path,
                                             "chronics_handler": chronics_handler,
                                             "backend": backend,
                                             "parameters": parameters,
                                             "name": name,
                                             "names_chronics_to_backend": names_chronics_to_backend,
                                             "actionClass": actionClass,
                                             "observationClass": observationClass,
                                             "rewardClass": rewardClass,
                                             "legalActClass": legalActClass,
                                             "voltagecontrolerClass": voltagecontrolerClass,
                                             "other_rewards": other_rewards,
                                             "opponent_space_type": opponent_space_type,
                                             "opponent_action_class": opponent_action_class,
                                             "opponent_class": opponent_class,
                                             "opponent_init_budget": opponent_init_budget,
                                             "opponent_budget_per_ts": opponent_budget_per_ts,
                                             "opponent_budget_class": opponent_budget_class,
                                             "opponent_attack_duration": opponent_attack_duration,
                                             "opponent_attack_cooldown": opponent_attack_cooldown,
                                             "kwargs_opponent": kwargs_opponent,
                                             "with_forecast": with_forecast,
                                             "attention_budget_cls": attention_budget_cls,
                                             "kwargs_attention_budget": kwargs_attention_budget,
                                             "has_attention_budget": has_attention_budget,
                                             "logger": logger,
                                             "kwargs_observation": kwargs_observation,
                                             "observation_bk_class": observation_bk_class,
                                             "observation_bk_kwargs": observation_bk_kwargs,
                                             "_raw_backend_class": _raw_backend_class,
                                             "_read_from_local_dir": _read_from_local_dir},
                                  **other_env_kwargs)
        return res
