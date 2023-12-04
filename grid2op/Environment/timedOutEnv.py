# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
from math import floor
from typing import Tuple, Union, List
from grid2op.Environment.environment import Environment
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Exceptions import EnvError


class TimedOutEnvironment(Environment):  # TODO heritage ou alors on met un truc de base
    """This class is the grid2op implementation of a "timed out environment" entity in the RL framework.

    This class is very similar to the
    standard environment. They only differ in the behaivour 
    of the `step` function. 
    
    For more information, see the documentation of 
    :func:`TimedOutEnvironment.step` for 
    
    Attributes
    ----------

    name: ``str``
        The name of the environment

    time_out_ms: ``int``
        maximum duration before performing a do_nothing action and updating to the next time_step.
    
    action_space: :class:`grid2op.Action.ActionSpace`
        Another name for :attr:`Environment.helper_action_player` for gym compatibility.

    observation_space:  :class:`grid2op.Observation.ObservationSpace`
        Another name for :attr:`Environment.helper_observation` for gym compatibility.

    reward_range: ``(float, float)``
        The range of the reward function

    metadata: ``dict``
        For gym compatibility, do not use

    spec: ``None``
        For Gym compatibility, do not use

    _viewer: ``object``
        Used to display the powergrid. Currently properly supported.

    """  
    CAN_SKIP_TS = True  # some steps can be more than one time steps
    def __init__(self,
                 grid2op_env: Union[Environment, dict],
                 time_out_ms: int=1e3) -> None:
        if time_out_ms <= 0.:
            raise EnvError(f"For TimedOutEnvironment you need to provide "
                           f"a time_out_ms > 0 (currently {time_out_ms})")
        self.time_out_ms = float(time_out_ms)  # in ms
        self.__last_act_send = time.perf_counter()
        self.__last_act_received = self.__last_act_send
        self._nb_dn_last = 0
        self._is_init_dn = False
        if isinstance(grid2op_env, Environment):
            super().__init__(**grid2op_env.get_kwargs())
        elif isinstance(grid2op_env, dict):
            super().__init__(**grid2op_env)
        else:
            raise EnvError(f"For TimedOutEnvironment you need to provide "
                           f"either an Environment or a dict "
                           f"for grid2op_env. You provided: {type(grid2op_env)}")
        self._is_init_dn = True
        self._res_skipped = []
        self._opp_attacks = []

    def step(self, action: BaseAction) -> Tuple[BaseObservation, float, bool, dict]:      
        """This function allows to pass to the 
        next step for the action.
        
        Provided the action the agent wants to do, it will 
        perform the action on the grid and resturn the typical
        "observation, reward, done, info" tuple.
        
        Compared to :func:`BaseEnvironment.step` this function
        will emulate the "time that passes" supposing that the duration
        between each step should be `time_out_ms`. Indeed, in reality,
        there is only 5 mins to take an action between two grid states
        separated from 5 mins.
        
        More precisely:
        
        If your agent takes less than `time_out_ms` to chose its action
        then this function behaves normally.
        
        If your agent takes between `time_out_ms` and `2 x time_out_ms` 
        to provide an action then
        a "do nothing" action is performed and then the provided
        action is performed. 
        
        If your agent takes between `2 x time_out_ms` and `3 x time_out_ms`
        to provide an action, then 2 "do nothing" actions are
        performed before your action.
        
        .. note::
            It is possible that the environment "fails" before 
            the action of the agent is implemented on the grid.

        Parameters
        ----------
        action : `grid2op.Action.BaseAction`
            The action the agent wish to perform.

        Returns
        -------
        Tuple[BaseObservation, float, bool, dict]
            _description_
        """
        self.__last_act_received = time.perf_counter()
        self._res_skipped = []
        self._opp_attacks = []
        
        # do the "do nothing" actions
        self._nb_dn_last = 0
        if self._is_init_dn:
            nb_dn = floor(1000. * (self.__last_act_received  - self.__last_act_send) / (self.time_out_ms))
        else:
            nb_dn = 0
        do_nothing_action = self.action_space()
        for _ in range(nb_dn):
            obs, reward, done, info = super().step(do_nothing_action)
            self._nb_dn_last += 1
            self._opp_attacks.append(self._oppSpace.last_attack)
            if done:
                info["nb_do_nothing"] = nb_dn
                info["nb_do_nothing_made"] = self._nb_dn_last
                info["action_performed"] = False
                info["last_act_received"] = self.__last_act_received
                info["last_act_send"] = self.__last_act_send
                return obs, reward, done, info
            self._res_skipped.append((obs, reward, done, info))
        
        # now do the action
        obs, reward, done, info = super().step(action)
        self._opp_attacks.append(self._oppSpace.last_attack)
        info["nb_do_nothing"] = nb_dn
        info["nb_do_nothing_made"] = self._nb_dn_last
        info["action_performed"] = True
        info["last_act_received"] = self.__last_act_received
        info["last_act_send"] = self.__last_act_send
        self.__last_act_send = time.perf_counter()
        return obs, reward, done, info

    def steps(self, action) -> Tuple[List[Tuple[BaseObservation, float, bool, dict]],
                                     List[BaseAction]]:
        tmp = self.step(action)            
        res = []
        for el in self._res_skipped:
            res.append(el)
        res.append(tmp)
        return res, self._opp_attacks
    
    def get_kwargs(self, with_backend=True, with_chronics_handler=True):
        res = {}
        res["time_out_ms"] = self.time_out_ms
        res["grid2op_env"] = super().get_kwargs(with_backend, with_chronics_handler)
        return res

    def get_params_for_runner(self):
        res = super().get_params_for_runner()
        res["envClass"] = TimedOutEnvironment
        res["other_env_kwargs"] = {"time_out_ms": self.time_out_ms}
        return res
    
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
        res = TimedOutEnvironment(grid2op_env={"init_env_path": init_env_path,
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
            
    def reset(self) -> BaseObservation:
        """Reset the environment.

        Returns
        -------
        BaseObservation
            The first observation of the new episode.
            
        """
        self.__last_act_send = time.perf_counter()
        self.__last_act_received = self.__last_act_send
        self._is_init_dn = False
        res = super().reset()
        self.__last_act_send = time.perf_counter()
        self._is_init_dn = True
        return res
    
    def _custom_deepcopy_for_copy(self, new_obj):
        super()._custom_deepcopy_for_copy(new_obj)
        new_obj.__last_act_send = time.perf_counter()
        new_obj.__last_act_received = new_obj.__last_act_send
        new_obj._is_init_dn = self._is_init_dn
        new_obj.time_out_ms = self.time_out_ms
