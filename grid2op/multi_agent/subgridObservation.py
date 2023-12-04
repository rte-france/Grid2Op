# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np

from typing import Type
from grid2op.dtypes import dt_float, dt_int, dt_bool
from grid2op.Exceptions import SimulateError
from grid2op.Observation import ObservationSpace, CompleteObservation, BaseObservation
from grid2op.multi_agent.subGridObjects import SubGridObjects
from grid2op.multi_agent.ma_typing import ActionProfile


class SubGridObservation(SubGridObjects, CompleteObservation):
    # TODO NOT TESTED in general
    attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
    
    attr_list_vect.append("interco_p")
    attr_list_vect.append("interco_q")
    attr_list_vect.append("interco_a")
    attr_list_vect.append("interco_v")
    attr_list_vect.append("interco_theta")
    attr_list_vect.append("interco_rho")
    attr_list_vect.append("interco_status")
    attr_list_vect.append("time_before_cooldown_interco")
    attr_list_vect.append("duration_next_maintenance_interco")
    attr_list_vect.append("time_next_maintenance_interco")
    attr_list_vect.append("timestep_overflow_interco")
    attr_list_set = set(attr_list_vect)
    
    attr_list_set = set(attr_list_vect)
    def __init__(self,
                 is_complete_obs=False,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None):
        SubGridObjects.__init__(self)
        CompleteObservation.__init__(self, obs_env, action_helper, random_prng)
        self._is_complete_obs = is_complete_obs
        
        self.interco_p = np.empty(shape=self.n_interco, dtype=dt_float)
        self.interco_q = np.empty(shape=self.n_interco, dtype=dt_float)
        self.interco_a = np.empty(shape=self.n_interco, dtype=dt_float)
        self.interco_v = np.empty(shape=self.n_interco, dtype=dt_float)
        self.interco_theta = np.empty(shape=self.n_interco, dtype=dt_float)
        self.interco_rho = np.empty(shape=self.n_interco, dtype=dt_float)
        
        self.interco_status = np.empty(shape=self.n_interco, dtype=dt_bool)
        
        self.time_before_cooldown_interco = np.empty(shape=self.n_interco, dtype=dt_int)
        self.duration_next_maintenance_interco = np.empty(shape=self.n_interco, dtype=dt_int)
        self.time_next_maintenance_interco = np.empty(shape=self.n_interco, dtype=dt_int)
        self.timestep_overflow_interco = np.empty(shape=self.n_interco, dtype=dt_int)
    
    def simulate(self, actions: ActionProfile, time_step: int=1):
        """
        
        .. warning::
            It does not use the same logic as the environment to "avoid conflict" between the agents.
            
            This function takes the agent (in alphabetical order) and apply their local action successively.
            
            Then it applies the action of the agent that has the observation space (to make sure it has "the last word").

        Parameters
        ----------
        actions : ActionProfile
            List of actions you want to "simulate". This should be a dictionnary with the keys being the name
            of the agents, and the values the corresponding "local action". It should contain at least the 
            "local action" of the agent concerned by the observation. If no other actions are sent, "do nothing" is assumed.
             
        time_step : int, optional
            see the documentation of the observation.simulate, by default 1

        Returns
        -------
        It returns the complete observation after combination of the actions, the global reward, the "done" flag and a centralized "info".

        Raises
        ------
        SimulateError
            It is raised when the observatino is not "complete".
            
        """
        # TODO not tested intensively
        if not self._is_complete_obs:
            raise SimulateError("Impossible to forecast the impact of a local action if you "
                                "have the complete information about the grid. "
                                "Your agent can only observe a small part of the grid.")
            
        # 1) convert the actions to a global action
        my_cls = type(self)
        action = self.action_helper()
        for agent_nm in sorted(actions.keys()):
            if agent_nm == my_cls.agent_name:
                continue
            local_act  = actions[agent_nm]
            action += local_act.to_global(self.action_helper)
        action += actions[my_cls.agent_name].to_global(self.action_helper)
        # 2) simulate with the global action
        complete_obs, global_reward, done, info = super().simulate(action, time_step)
        # 3) return the result of the simulation
        return complete_obs, global_reward, done, info
    
    def reset(self):
        # TODO not tested
        super().reset()
        self.interco_p[:] = np.NaN
        self.interco_q[:] = np.NaN
        self.interco_a[:] = np.NaN
        self.interco_v[:] = np.NaN
        self.interco_theta[:] = np.NaN
        self.interco_rho[:] = np.NaN
        
        self.interco_status[:] = False
        
        self.time_before_cooldown_interco[:] = -1
        self.duration_next_maintenance_interco[:] = -1
        self.time_next_maintenance_interco[:] = -1
        self.timestep_overflow_interco[:] = 0
        
    def update(self, env, complete_obs, with_forecast=True):
        # env: centralized env
        # complete_obs: the complete observation at this step
        
        # TODO not tested
        
        if self._is_complete_obs:
            CompleteObservation.update(self, env, with_forecast)
        else:
            self._reset_matrices()
            self.reset()
            self._update_from_complete_obs(complete_obs)
            
    @classmethod
    def _aux_get_vect(cls, vect_line_or, vect_line_ex):    
        res = copy.deepcopy(vect_line_or[cls.mask_interco])
        res[~cls.interco_is_origin] = vect_line_ex[cls.mask_interco][~cls.interco_is_origin]
        return res
    
    def _read_attr_from_complete(self, complete_obs: CompleteObservation):   
        # TODO not tested  
        my_cls = type(self)
           
        self._is_done = complete_obs._is_done

        # calendar data
        self.year = complete_obs.year
        self.month = complete_obs.month
        self.day = complete_obs.day
        self.hour_of_day = complete_obs.hour_of_day
        self.minute_of_hour = complete_obs.minute_of_hour
        self.day_of_week = complete_obs.day_of_week

        self.timestep_overflow[:] = complete_obs.timestep_overflow[my_cls.mask_line]

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status[:] = complete_obs.line_status[my_cls.mask_line]

        # topological vector
        self.topo_vect[:] = complete_obs.topo_vect[my_cls.mask_orig_pos_topo_vect]

        # generators information
        self.gen_p[:] = complete_obs.gen_p[my_cls.mask_gen]
        self.gen_q[:] = complete_obs.gen_q[my_cls.mask_gen]
        self.gen_v[:] = complete_obs.gen_v[my_cls.mask_gen]
        self.gen_margin_up[:] = complete_obs.gen_margin_up[my_cls.mask_gen]
        self.gen_margin_down[:] = complete_obs.gen_margin_down[my_cls.mask_gen]

        # loads information
        self.load_p[:] = complete_obs.load_p[my_cls.mask_load]
        self.load_q[:] = complete_obs.load_q[my_cls.mask_load]
        self.load_v[:] = complete_obs.load_v[my_cls.mask_load]
        
        # lines origin information
        self.p_or[:] = complete_obs.p_or[my_cls.mask_line]
        self.q_or[:] =  complete_obs.q_or[my_cls.mask_line]
        self.v_or[:] =  complete_obs.v_or[my_cls.mask_line]
        self.a_or[:] =  complete_obs.a_or[my_cls.mask_line]
        # lines extremity information
        self.p_ex[:] =  complete_obs.p_ex[my_cls.mask_line]
        self.q_ex[:] =  complete_obs.q_ex[my_cls.mask_line]
        self.v_ex[:] =  complete_obs.v_ex[my_cls.mask_line]
        self.a_ex[:] =  complete_obs.a_ex[my_cls.mask_line]
        # lines relative flows
        self.rho[:] =  complete_obs.rho[my_cls.mask_line]

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line[:] = complete_obs.time_before_cooldown_line[my_cls.mask_line]
        self.time_before_cooldown_sub[:] = complete_obs.time_before_cooldown_sub[my_cls.mask_sub]
        self.time_next_maintenance[:] = complete_obs.time_next_maintenance[my_cls.mask_line]
        self.duration_next_maintenance[:] = complete_obs.duration_next_maintenance[my_cls.mask_line]

        # redispatching
        self.target_dispatch[:] = complete_obs.target_dispatch[my_cls.mask_gen]
        self.actual_dispatch[:] = complete_obs.actual_dispatch[my_cls.mask_gen]

        # storage unit
        self.storage_charge[:] = complete_obs.storage_charge[my_cls.mask_storage]
        self.storage_power_target[:] = complete_obs.storage_power_target[my_cls.mask_storage]
        self.storage_power[:] = complete_obs.storage_power[my_cls.mask_storage]

        # attention budget
        if my_cls.dim_alarms:
            raise NotImplementedError("Partial observation cannot handle raising alarm at the moment")
            # TODO
            self.is_alarm_illegal[:] = np.ones(shape=1, dtype=dt_bool)
            self.time_since_last_alarm[:] = np.empty(shape=1, dtype=dt_int)
            self.last_alarm[:] = np.empty(shape=self.dim_alarms, dtype=dt_int)
            self.attention_budget[:] = np.empty(shape=1, dtype=dt_float)
            self.was_alarm_used_after_game_over[:] = np.zeros(shape=1, dtype=dt_bool)

        # to save some computation time
        self._connectivity_matrix_ = None
        self._bus_connectivity_matrix_ = None
        self._dictionnarized = None
        self._vectorized = None

        # for shunt (these are not stored!)
        if my_cls.shunts_data_available:
            self._shunt_p[:] = complete_obs._shunt_p[my_cls.mask_shunt]
            self._shunt_q[:] = complete_obs._shunt_q[my_cls.mask_shunt]
            self._shunt_v[:] = complete_obs._shunt_v[my_cls.mask_shunt]
            self._shunt_bus[:] = complete_obs._shunt_bus[my_cls.mask_shunt]

        self._thermal_limit[:] = complete_obs._thermal_limit[my_cls.mask_line]

        self.gen_p_before_curtail[:] = complete_obs.gen_p_before_curtail[my_cls.mask_gen]
        self.curtailment[:] = complete_obs.curtailment[my_cls.mask_gen]
        self.curtailment_limit[:] = complete_obs.curtailment_limit[my_cls.mask_gen]
        self.curtailment_limit_effective[:] = complete_obs.curtailment_limit_effective[my_cls.mask_gen]

        # the "theta" (voltage angle, in degree)
        if complete_obs.support_theta:
            self.theta_or[:] = complete_obs.theta_or[my_cls.mask_line]
            self.theta_ex[:] = complete_obs.theta_ex[my_cls.mask_line]
            self.load_theta[:] = complete_obs.load_theta[my_cls.mask_load]
            self.gen_theta[:] = complete_obs.gen_theta[my_cls.mask_gen]
            self.storage_theta[:] = complete_obs.storage_theta[my_cls.mask_storage]

        # counter
        self.current_step = complete_obs.current_step
        self.max_step = complete_obs.max_step
        self.delta_time = complete_obs.delta_time

        
    def _update_from_complete_obs(self, complete_obs: CompleteObservation):
        # TODO read the correct data from the observation in this case
        # and update the "interco" data
        
        my_cls = type(self)
        
        # TODO not tested
        self._read_attr_from_complete(complete_obs)
        
        # TODO not tested
        self.interco_p[:] = my_cls._aux_get_vect(complete_obs.p_or, complete_obs.p_ex)
        self.interco_q[:] = my_cls._aux_get_vect(complete_obs.q_or, complete_obs.q_ex)
        self.interco_a[:] = my_cls._aux_get_vect(complete_obs.a_or, complete_obs.a_ex)
        self.interco_v[:] = my_cls._aux_get_vect(complete_obs.v_or, complete_obs.v_ex)
        if self.support_theta:
            self.interco_theta[:] = my_cls._aux_get_vect(complete_obs.theta_or, complete_obs.theta_ex)
        self.interco_rho[:] = complete_obs.rho[my_cls.mask_interco]
        
        self.interco_status[:] = complete_obs.line_status[my_cls.mask_interco]
        
        self.time_before_cooldown_interco[:] = complete_obs.time_before_cooldown_line[my_cls.mask_interco]
        self.duration_next_maintenance_interco[:] = complete_obs.duration_next_maintenance[my_cls.mask_interco]
        self.time_next_maintenance_interco[:] = complete_obs.time_next_maintenance[my_cls.mask_interco]
        self.timestep_overflow_interco[:] = complete_obs.timestep_overflow[my_cls.mask_interco]
        
    def _aux_set_game_over_thermal_limit(self, env=None):
        """set the thermal limit when game over.
        
        Needs to be overriden in the SubObservationClass for multi agent"""
        if env is not None:
            self._thermal_limit[:] = 1.0 * env._thermal_limit_a[type(self).mask_line]  # function get_thermal_limit() "crashes" (because env is game over)
        else:
            self._thermal_limit[:] = 0. 
        
class SubGridObservationSpace(SubGridObjects, ObservationSpace):
    # modeling choice : one observation space per "sub agent"
    
    # TODO NOT TESTED
    def __init__(
        self,
        full_gridobj,  # full grid, used for "simulate"
        ma_env,
        local_gridobj,  # gridobject for the observation space of this agent
        is_complete_obs=True, # whether the observation is complete or not
        rewardClass=None,
        observationClass : Type[SubGridObservation]=SubGridObservation,  # should be a local observation
        actionClass=None,  # Complete action, used internally for simulate
        with_forecast=True,
        kwargs_observation=None,
        logger=None,
    ):
        SubGridObjects.__init__(self)
        ObservationSpace.__init__(self,
                                  full_gridobj,
                                  ma_env._cent_env,  # pass the centralized env for the observation space, maybe ?
                                  rewardClass,
                                  CompleteObservation,
                                  actionClass,
                                  with_forecast,
                                  kwargs_observation,
                                  logger)
        self.local_obs_cls : Type[SubGridObservation] = observationClass.init_grid(local_gridobj)
        self._is_complete_obs : bool = is_complete_obs
    
    def __call__(self, ma_env, cent_obs, _update_state=True):
        # cent obs is the centralized observation
        # ma_env is the multi agent env
        if self.with_forecast:
            # update the backend for the "simulate"
            self.obs_env.update_grid(ma_env._cent_env)
        
        # self.__nb_simulate_called_this_step = 0 
        self._ObservationSpace__nb_simulate_called_this_step = 0

        res = self.local_obs_cls(
            self._is_complete_obs,
            obs_env=self.obs_env if self.obs_env.is_valid() else None,
            action_helper=self.action_helper_env,
            random_prng=self.space_prng,
            **self._ptr_kwargs_observation
        )
        
        if _update_state:
            res.update(ma_env._cent_env,
                       cent_obs,
                       with_forecast=self.with_forecast)
        return res