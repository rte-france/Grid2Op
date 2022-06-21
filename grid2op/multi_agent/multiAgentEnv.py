# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
from typing import Optional, Tuple, Union
import warnings
import numpy as np
import re
from grid2op.Environment.Environment import Environment

import grid2op
from grid2op.Exceptions.EnvExceptions import EnvError
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Observation.observationSpace import ObservationSpace
from grid2op.dtypes import dt_bool, dt_int
from grid2op.Action import ActionSpace
from grid2op.multi_agent.subGridObjects import SubGridObjects
from grid2op.operator_attention import LinearAttentionBudget
from grid2op.multi_agent.utils import AgentSelector, random_order  
from grid2op.multi_agent.ma_typing import ActionProfile, AgentID, MADict
from grid2op.multi_agent.multi_agentExceptions import *

import pdb


class MultiAgentEnv :
    def __init__(self,
                 env : Environment,
                 action_domains : MADict,
                 observation_domains : MADict = None,
                 agent_order_fn = lambda x : x,
                 illegal_action_pen : float = 0.,
                 ambiguous_action_pen : float = 0.,
                 _add_to_name: Optional[str] = None,
                 ):
        """Multi-agent Grid2Op POSG (Partially Observable Stochastic Game) environment
        This class transforms the given classical Grid2Op environment into a 
        multi-agent environment.

        Args:
            * env (Environment): The Grid2Op classical environment
            * action_domains (MADict):
                - keys : agents' names 
                - values : list of substations' (or lines) id under the control of the agent. Note that these ids imply
                    also generators and storage units in charge of the agent.
            * observation_domains (MADict): 
                - keys : agents' names 
                - values : list of attributes of a Grid2Op observation that the agent can observe. It is represented by 
                    the ObservationDomain class.
            * agent_order_fn (_type_, optional): _description_. Defaults to lambdax:x.
            * illegal_action_pen (float, optional): The penalization received by an agent in case of an illegal action.  
                                                    Defaults to 0..
            * ambiguous_action_pen (float, optional): The penalization received by an agent in case of an ambiguous action. 
                                                        Defaults to 0..
                                                        
        Attributes:
            * _cent_env : The Grid2Op classical environment
            
            * _action_domains (MADict): a dictionary with agents' names as keys and different masks as values.
                These masks are used to filter elements of the env in which the agent in key is in charge.
                
            * _observation_domains (MADict): a dictionary with agents' names as keys and different masks as values.
                These masks are used to filter elements of the env in which the agent in key can observe.
                
            * illegal_action_pen : The penalization received by an agent in case of an illegal action.
            
            * ambiguous_action_pen : The penalization received by an agent in case of an ambiguous action.
            
            * rewards (MADict) : 
                - keys : agents' names
                - values : agent's reward
            
            * _cumulative_rewards (MADict) :
                - keys : agents' names
                - values : agent's cumulative reward in the episode
                
            * observation (MADict) :
                - keys : agents' names
                - values : agent's observation
            
            * agent_order : 
                Gives the order (priority) of agents. If there’s a conflict, the action of the agent with higher
                priority (smaller index in the list) is kept.
                
            * _subgrids (MADIct):
                - keys : agents’ names
                - values : created subgrids of type SubGridObjects by _build_subgrids method

                
            * action_spaces (MADict) :
                - keys: agents’ names
                - values : action spaces created from _action_domains and _subgrids
            
            * action_spaces (MADict) :
                - keys: agents’ names
                - values : observation spaces created from _observation_domains and _subgrids
                      
 
        """
        # added to the class name, if you want to build multiple ma environment with the same underlying environment
        self._add_to_name = _add_to_name  
        
        self._cent_env : Environment = env
            
        
        self._verify_domains(action_domains)
        self._action_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in action_domains.items()}
        self.num_agents = len(action_domains)
        
        if observation_domains is not None:
            # user specified an observation domain
            self._is_global_obs : bool = True
            if action_domains.keys() != observation_domains.keys():
                raise("action_domains and observation_domains must have the same agents' names !")
            self._observation_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in observation_domains.items()}
            self._verify_domains(observation_domains)
        else:
            # no observation domain specified, so I assume it's a domain
            # with full observability
            self._is_global_obs : bool = False
            self._observation_domains = None
        
        self.agents = list(action_domains.keys())
        self.num_agents = len(self.agents)
        
        self.illegal_action_pen = illegal_action_pen
        self.ambiguous_action_pen = ambiguous_action_pen
        self.rewards = dict(
            zip(
                self.agents, [0. for _ in range(self.num_agents)]
            )
        )
        self._cumulative_rewards = dict(
            zip(
                self.agents, [0. for _ in range(self.num_agents)]
            )
        )
        self.observations = dict(
            zip(
                self.agents, [None for _ in range(self.num_agents)]
            )
        )
        self.agent_order = AgentSelector(self.agents, agent_order_fn)
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self._build_subgrids()
        self._build_action_spaces()
        self._build_observation_spaces()
        
        
    def reset(self) -> MADict:
        self._cent_env.reset()
        self._update_observations(_update_state=False)
        return self.observations
    
    def step(self, action : ActionProfile) -> Tuple[MADict, MADict, MADict, MADict]:
    # TODO have a more usefull typing !
    # this is close to useless here
        """_summary_#TODO

        Args:
            action (dict): _description_
        """
        
        # We create the global action
        global_action = self._cent_env.action_space({})
        self.rewards = dict(
            zip(
                self.agents, [0. for _ in range(self.num_agents)]
            )
        )
        self.info = dict(
            zip(
                self.agents, [{} for _ in range(self.num_agents)]
            )
        )
        
        order = self.select_agent.get_order(reinit=True)
        for agent in order:
            # TODO agent is an "agent_id" or an "agent_name" not clear
            converted_action = action[agent] # TODO should be converted into grid2op action
            proposed_action = global_action + converted_action
            
            # TODO are you sure it's possible here ? Are you sure it's what you want to do ?
            # How did you define "illegal" for a partial action ? I'm not sure that's how
            # it's implemented.
            is_legal, reason = self._cent_env._game_rules(action=proposed_action, env=self._cent_env)
            if not is_legal:
                # action is replace by do nothing
                # action = self._action_space({})
                #init_disp = 1.0 * proposed_action._redispatch  # dispatching action
                #action_storage_power = (
                #    1.0 * action._storage_power
                #)  # battery information
                #is_illegal = True
                self.rewards[agent] -= self.illegal_action_pen
                self.info[agent]['is_illegal_local'] = True

            ambiguous, except_tmp = proposed_action.is_ambiguous()
            # TODO can you think of other type of "ambiguous" actions maybe ?
            if ambiguous:
                ## action is replace by do nothing
                #action = self._action_space({})
                #init_disp = 1.0 * action._redispatch  # dispatching action
                #action_storage_power = (
                #    1.0 * action._storage_power
                #)  # battery information
                #is_ambiguous = True
                #except_.append(except_tmp)
                self.rewards[agent] -= self.ambiguous_action_pen
                self.info[agent]['is_ambiguous_local'] = True
                
            if is_legal and not ambiguous :
                global_action = proposed_action

            #if self._has_attention_budget:
            #    # this feature is implemented, so i do it
            #    reason_alarm_illegal = self._attention_budget.register_action(
            #        self, action, is_illegal, is_ambiguous
            #    )
            #    self._is_alarm_illegal = reason_alarm_illegal is not None
                
        observation, reward, done, info = self.env.step(global_action)
        # update agents' observation, reward, done, info
        for agent in self.agents:
            self.rewards[agent] += reward
            self.dones[agent] = done
            self.info[agent].update(info)
            
        self._update_observations(_update_state=False)
        
        return self.observations, self.rewards, self.dones, self.info 
    
    #def agent_iter(self, max_iter=2 ** 63):
    #    """Agent iterator : it interacts with agents
    #    Take the global observation, reward, 
    
    #    Args:
    #        max_iter (int): Maximum iteration. Defaults to 2**63.
    #    """
    #    #TODO
    #    pass
    
    def _build_subgrids(self):
        """_summary_ #TODO
        """
        
        self._subgrids_cls = {
            'action' : dict(),
            'observation' : dict()
        }
        for agent_nm in self.agents : 
            self._build_agent_domain(agent_nm, self._action_domains[agent_nm])
            subgridobj = self._build_subgrid_obj_from_domain(self._action_domains[agent_nm])
            extra_name = agent_nm
            if self._add_to_name is not None:
                extra_name += f"{self._add_to_name}"
                
            #TODO init grid does not work when the grid is not connected
            self._subgrids_cls['action'][agent_nm] = SubGridObjects.init_grid(gridobj=subgridobj, extra_name=extra_name)
            self._subgrids_cls['action'][agent_nm].shunt_to_subid = subgridobj.shunt_to_subid.copy()
            self._subgrids_cls['action'][agent_nm].grid_objects_types = subgridobj.grid_objects_types.copy()
            
            if self._observation_domains is not None:
                self._build_agent_domain(agent_nm, self._observation_domains[agent_nm])
                subgridobj = self._build_subgrid_obj_from_domain(self._observation_domains[agent_nm])
                self._subgrids_cls['observation'][agent_nm] = SubGridObjects.init_grid(gridobj=subgridobj, extra_name=extra_name)
                self._subgrids_cls['observation'][agent_nm].grid_objects_types = subgridobj.grid_objects_types
        
        
    def _build_agent_domain(self, agent_nm, domain):
        """_summary_#TODO

        Args:
            domain (_type_): _description_
        """        
        is_sub_in = np.full(self._cent_env.n_sub, fill_value=False, 
                            dtype=dt_bool)
        is_sub_in[domain['sub_id']] = True
        domain['mask_sub'] = is_sub_in
        
        domain['mask_load'] = np.isin(
            self._cent_env.load_to_subid, domain['sub_id'] 
        )
        domain['mask_gen'] = np.isin(
            self._cent_env.gen_to_subid, domain['sub_id'] 
        )
        domain['mask_storage'] = np.isin(
            self._cent_env.storage_to_subid, domain['sub_id'] 
        )
        domain['mask_shunt'] = np.isin(
            self._cent_env.shunt_to_subid, domain['sub_id'] 
        )
        domain['mask_line_or'] = np.isin(
            self._cent_env.line_or_to_subid, domain['sub_id']
        )
        domain['mask_line_ex'] = np.isin(
            self._cent_env.line_ex_to_subid, domain['sub_id'] 
        )
        domain['mask_interco'] = domain['mask_line_or']^domain['mask_line_ex']
        domain['interco_is_origin'] = domain['mask_line_or'][domain['mask_interco']]
        domain['mask_line_or'] = domain['mask_line_or']&domain['mask_line_ex']
        domain['mask_line_ex'] = domain['mask_line_or'].copy()
        domain["agent_name"] = agent_nm
    
    def _relabel_subid(self, mask, new_label, id_full_grid):
        tmp_ = id_full_grid[mask]
        return new_label[tmp_]
    
    
    def _build_subgrid_obj_from_domain(self, domain):
        cent_env_cls = type(self._cent_env)
        tmp_subgrid = SubGridObjects()
        tmp_cls = type(tmp_subgrid)
        tmp_cls.sub_orig_ids = copy.deepcopy(domain['sub_id'])
        tmp_cls.mask_load = copy.deepcopy(domain['mask_load'])
        tmp_cls.mask_gen = copy.deepcopy(domain['mask_gen'])
        tmp_cls.mask_storage = copy.deepcopy(domain['mask_storage'])
        tmp_cls.mask_shunt = copy.deepcopy(domain['mask_shunt'])
        tmp_cls.mask_line_or = copy.deepcopy(domain['mask_line_or'])
        tmp_cls.mask_line_ex = copy.deepcopy(domain['mask_line_ex'])
        tmp_cls.agent_name = copy.deepcopy(domain["agent_name"])
        tmp_cls.mask_sub = copy.deepcopy(domain["mask_sub"])
        tmp_cls.mask_interco = copy.deepcopy(domain["mask_interco"])
        tmp_cls.interco_is_origin = copy.deepcopy(domain["interco_is_origin"])
        
        tmp_cls.glop_version = cent_env_cls.glop_version
        tmp_cls._PATH_ENV = cent_env_cls._PATH_ENV

        # name of the objects
        tmp_cls.env_name = cent_env_cls.env_name
        tmp_cls.name_load = cent_env_cls.name_load[
            tmp_cls.mask_load
        ]
        tmp_cls.name_gen = cent_env_cls.name_gen[
            tmp_cls.mask_gen
        ]
        tmp_cls.name_line = cent_env_cls.name_line[
            tmp_cls.mask_line_or & tmp_cls.mask_line_ex
        ]
        tmp_cls.name_sub = cent_env_cls.name_sub[
            tmp_cls.mask_sub
        ]
        tmp_cls.name_storage = cent_env_cls.name_storage[
            tmp_cls.mask_storage
        ]
        tmp_cls.name_shunt = cent_env_cls.name_shunt[
            tmp_cls.mask_shunt
        ]
        
        n_col_grid_objects_types = 5
        tmp_cls.n_gen = len(tmp_cls.name_gen)
        tmp_cls.n_load = len(tmp_cls.name_load)
        tmp_cls.n_line = len(tmp_cls.name_line)
        tmp_cls.n_sub = len(tmp_cls.name_sub)
        tmp_cls.n_shunt = len(tmp_cls.name_shunt)
        if tmp_cls.n_shunt > 0:
            n_col_grid_objects_types += 1
            #n_line_grid_objects_types += tmp_cls.n_shunt #TODO should I add shunt ?
        tmp_cls.n_storage = len(tmp_cls.name_storage)
        if tmp_cls.n_storage > 0:
            n_col_grid_objects_types += 1
        tmp_cls.n_interco = len(tmp_cls.interco_is_origin)
        if tmp_cls.n_interco > 0:
            n_col_grid_objects_types += 1
        

        tmp_cls.sub_info = cent_env_cls.sub_info[
            tmp_cls.sub_orig_ids
        ]
        
        #TODO correct ?
        tmp_cls.dim_topo = 2*tmp_cls.n_line + tmp_cls.n_interco + \
            tmp_cls.n_load + tmp_cls.n_gen + tmp_cls.n_storage

        # to which substation is connected each element 
        # this is a bit "tricky" because I have to
        # re label the substation in the subgrid
        tmp_cls.load_to_subid = np.zeros(tmp_cls.n_load, dtype=dt_int)
        tmp_cls.gen_to_subid = np.zeros(tmp_cls.n_gen, dtype=dt_int)
        tmp_cls.line_or_to_subid = np.zeros(tmp_cls.n_line, dtype=dt_int)
        tmp_cls.line_ex_to_subid = np.zeros(tmp_cls.n_line, dtype=dt_int)
        tmp_cls.storage_to_subid = np.zeros(tmp_cls.n_storage, dtype=dt_int)
        tmp_cls.shunt_to_subid = np.zeros(tmp_cls.n_shunt, dtype=dt_int)
        
        # new_label[orig_grid_sub_id] = new_grid_sub_id
        new_label = np.zeros(cent_env_cls.n_sub, dtype=dt_int) - 1
        new_label[tmp_cls.sub_orig_ids] = np.arange(tmp_cls.n_sub)
        
        tmp_cls.load_to_subid  = self._relabel_subid(tmp_cls.mask_load,
                                                     new_label,
                                                     cent_env_cls.load_to_subid
        )
        tmp_cls.gen_to_subid  = self._relabel_subid(tmp_cls.mask_gen,
                                                     new_label,
                                                     cent_env_cls.gen_to_subid
        )
        tmp_cls.shunt_to_subid  = self._relabel_subid(tmp_cls.mask_shunt,
                                                      new_label,
                                                      cent_env_cls.shunt_to_subid
        )
        tmp_cls.storage_to_subid  = self._relabel_subid(tmp_cls.mask_storage,
                                                        new_label,
                                                        cent_env_cls.storage_to_subid
        )
        tmp_cls.line_or_to_subid = self._relabel_subid(tmp_cls.mask_line_or,
                                                        new_label,
                                                        cent_env_cls.line_or_to_subid
        )
        tmp_cls.line_ex_to_subid = self._relabel_subid(tmp_cls.mask_line_ex,
                                                        new_label,
                                                        cent_env_cls.line_ex_to_subid
        )
        
        
        tmp_cls.interco_to_subid = new_label[
            np.array([
                cent_env_cls.line_or_to_subid[tmp_cls.mask_interco][i] if tmp_cls.interco_is_origin[i] 
                else cent_env_cls.line_ex_to_subid[tmp_cls.mask_interco][i] for i in range(tmp_cls.n_interco)
            ])
        ]
        tmp_cls.interco_to_lineid = np.arange(cent_env_cls.n_line)[tmp_cls.mask_interco]
        tmp_cls.name_interco = np.array([
            f'interco_{i}_line_{tmp_cls.interco_to_lineid[i]}' for i in range(len(tmp_cls.interco_is_origin))
        ])

        # which index has this element in the substation vector 
        tmp_cls.load_to_sub_pos = cent_env_cls.load_to_sub_pos[tmp_cls.mask_load]
        tmp_cls.gen_to_sub_pos = cent_env_cls.gen_to_sub_pos[tmp_cls.mask_gen]
        tmp_cls.storage_to_sub_pos = cent_env_cls.storage_to_sub_pos[tmp_cls.mask_storage]
        tmp_cls.line_or_to_sub_pos = cent_env_cls.line_or_to_sub_pos[tmp_cls.mask_line_or]
        tmp_cls.line_ex_to_sub_pos = cent_env_cls.line_ex_to_sub_pos[tmp_cls.mask_line_ex]
        #tmp_cls.shunt_to_sub_pos = cent_env_cls.shunt_to_sub_pos[tmp_cls.mask_shunt]
        tmp_cls.interco_to_sub_pos = np.array([
            cent_env_cls.line_or_to_sub_pos[tmp_cls.mask_interco][i] if tmp_cls.interco_is_origin[i] 
            else cent_env_cls.line_ex_to_sub_pos[tmp_cls.mask_interco][i] for i in range(tmp_cls.n_interco)
        ])

        # #TODO
        tmp_cls.grid_objects_types = -np.ones((tmp_cls.dim_topo, n_col_grid_objects_types))
        
        # # which index has this element in the topology vector
        # # "convenient" way to retrieve information of the grid
        # # to which substation each element of the topovect is connected
        sub_info_cum_sum = np.cumsum(tmp_cls.sub_info)
        tmp_cls.load_pos_topo_vect = np.where(tmp_cls.load_to_subid == 0, 
            tmp_cls.load_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.load_to_subid-1] + tmp_cls.load_to_sub_pos
        )
        tmp_cls.grid_objects_types[tmp_cls.load_pos_topo_vect, 0] = tmp_cls.load_to_subid
        tmp_cls.grid_objects_types[tmp_cls.load_pos_topo_vect, 1] = np.arange(len(tmp_cls.load_pos_topo_vect))
        
        tmp_cls.gen_pos_topo_vect = np.where(tmp_cls.gen_to_subid == 0, 
            tmp_cls.gen_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.gen_to_subid-1] + tmp_cls.gen_to_sub_pos
        )
        tmp_cls.grid_objects_types[tmp_cls.gen_pos_topo_vect, 0] = tmp_cls.gen_to_subid
        tmp_cls.grid_objects_types[tmp_cls.gen_pos_topo_vect, 2] = np.arange(len(tmp_cls.gen_pos_topo_vect))
        
        tmp_cls.line_or_pos_topo_vect = np.where(tmp_cls.line_or_to_subid == 0, 
            tmp_cls.line_or_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.line_or_to_subid-1] + tmp_cls.line_or_to_sub_pos
        )
        tmp_cls.grid_objects_types[tmp_cls.line_or_pos_topo_vect, 0] = tmp_cls.line_or_to_subid
        tmp_cls.grid_objects_types[tmp_cls.line_or_pos_topo_vect, 3] = np.arange(len(tmp_cls.line_or_pos_topo_vect))
        
        tmp_cls.line_ex_pos_topo_vect = np.where(tmp_cls.line_ex_to_subid == 0, 
            tmp_cls.line_ex_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.line_ex_to_subid-1] + tmp_cls.line_ex_to_sub_pos
        )
        tmp_cls.grid_objects_types[tmp_cls.line_ex_pos_topo_vect, 0] = tmp_cls.line_ex_to_subid
        tmp_cls.grid_objects_types[tmp_cls.line_ex_pos_topo_vect, 4] = np.arange(len(tmp_cls.line_ex_pos_topo_vect))
        
        tmp_cls.storage_pos_topo_vect = np.where(tmp_cls.storage_to_subid == 0, 
            tmp_cls.storage_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.storage_to_subid-1] + tmp_cls.storage_to_sub_pos
        )
        last_col = 4
        if tmp_cls.n_storage >0:
            last_col += 1
            tmp_cls.grid_objects_types[tmp_cls.storage_pos_topo_vect, 0] = tmp_cls.storage_to_subid
            tmp_cls.grid_objects_types[tmp_cls.storage_pos_topo_vect, last_col] = np.arange(len(tmp_cls.storage_pos_topo_vect))
        
        tmp_cls.interco_pos_topo_vect = np.where(tmp_cls.interco_to_subid == 0, 
            tmp_cls.interco_to_sub_pos, 
            sub_info_cum_sum[tmp_cls.interco_to_subid-1] + tmp_cls.interco_to_sub_pos
        )
        if tmp_cls.n_interco >0:
            last_col += 1
            tmp_cls.grid_objects_types[tmp_cls.interco_pos_topo_vect, 0] = tmp_cls.interco_to_subid
            tmp_cls.grid_objects_types[tmp_cls.interco_pos_topo_vect, last_col] = np.arange(len(tmp_cls.interco_pos_topo_vect))
        
    
         #
         #tmp_cls_action._topo_vect_to_sub = None
        #tmp_cls._compute_pos_big_topo_cls()



        # redispatch data, not available in all environment
        tmp_cls.redispatching_unit_commitment_availble = False
        tmp_cls.gen_type = cent_env_cls.gen_type[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_pmin = cent_env_cls.gen_pmin[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_pmax = cent_env_cls.gen_pmax[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_redispatchable = cent_env_cls.gen_redispatchable[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_max_ramp_up = cent_env_cls.gen_max_ramp_up[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_max_ramp_down = cent_env_cls.gen_max_ramp_down[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_min_uptime = cent_env_cls.gen_min_uptime[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_min_downtime = cent_env_cls.gen_min_downtime[
            tmp_cls.mask_gen
        ]
        tmp_cls.gen_cost_per_MW = cent_env_cls.gen_cost_per_MW[
            tmp_cls.mask_gen
        ]  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
        tmp_cls.gen_startup_cost = cent_env_cls.gen_startup_cost[
            tmp_cls.mask_gen
        ]  # start cost (in currency)
        tmp_cls.gen_shutdown_cost = cent_env_cls.gen_shutdown_cost[
            tmp_cls.mask_gen
        ]  # shutdown cost (in currency)
        tmp_cls.gen_renewable = cent_env_cls.gen_renewable[
            tmp_cls.mask_gen
        ]

        # storage unit static data 
        tmp_cls.storage_type = cent_env_cls.storage_type[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_Emax = cent_env_cls.storage_Emax[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_Emin = cent_env_cls.storage_Emin[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_max_p_prod = cent_env_cls.storage_max_p_prod[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_max_p_absorb = cent_env_cls.storage_max_p_absorb[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_marginal_cost = cent_env_cls.storage_marginal_cost[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_loss = cent_env_cls.storage_loss[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_charging_efficiency = cent_env_cls.storage_charging_efficiency[
            tmp_cls.mask_storage
        ]
        tmp_cls.storage_discharging_efficiency = cent_env_cls.storage_discharging_efficiency[
            tmp_cls.mask_storage
        ]

        # grid layout
        tmp_cls.grid_layout = None if cent_env_cls.grid_layout is None else {
            k : cent_env_cls.grid_layout[k]
            for k in tmp_cls.name_sub
        }

        # shunt data, not available in every backend 
        tmp_cls.shunts_data_available = cent_env_cls.shunts_data_available
        # you should first check that and then fill all the shunts related attributes !!!!!
        # TODO BEN

        # # alarms #TODO
        # tmp_cls.dim_alarms = 0
        # tmp_cls.alarms_area_names = []
        # tmp_cls.alarms_lines_area = {}
        # tmp_cls.alarms_area_lines = []
        
        return tmp_cls
        
    
    def _build_observation_spaces(self):
        """Build observation spaces from given domains for each agent

        Args:
            global_obs (bool, optional): The agent can observe the whole grid. Defaults to True.

        Raises:
            NotImplementedError: Local observations are not implemented yet
        """
        
        if self._observation_domains is None:
            for agent in self.agents:
                self.observation_spaces[agent] = self._cent_env.observation_space.copy()
        else:
            raise NotImplementedError("Local observations are not available yet !")
        #TODO Local observations
        #for agent in self.agents: 
        #    _cls_agent_action_space = ObservationSpace.init_grid(gridobj=self._subgrids_cls['observation'][agent], extra_name=agent)
        #    self.observation_spaces[agent] = _cls_agent_action_space(
        #        gridobj = self._subgrids_cls['observation'][agent],
        #        env = self._cent_env,
        #        rewardClass=self._cent_env._rewardClass,
        #        observationClass=self._cent_env._observationClass,
        #        actionClass=self._cent_env._actionClass,
        #        #TODO following parameters
        #        with_forecast=True,
        #        kwargs_observation=None,
        #    )
    
    def _build_action_spaces(self):
        """Build action spaces from given domains for each agent
        The action class is the same as 
        """
        for agent in self.agents: 
            _cls_agent_action_space = ActionSpace.init_grid(gridobj=self._subgrids_cls['action'][agent], extra_name=agent)
            self.action_spaces[agent] = _cls_agent_action_space(
                gridobj=self._subgrids_cls['action'][agent],
                actionClass=self._cent_env._actionClass,
                legal_action=self._cent_env._game_rules.legal_action,
            )
    
    
    def _update_observations(self, _update_state = True):
        """Update agents' observations from the global observation given by the self._cent_env

        Args:
            observation (BaseObservation): _description_
        """
        #for agent in self.agents:
        #    self.observations[agent] = observation.copy()
        
        if self._cent_env.__closed:
            raise EnvError("This environment is closed. You cannot use it anymore.")
        if not self.__is_init:
            raise EnvError(
                "This environment is not initialized. You cannot retrieve its observation."
            )
        
        for agent in self.agents:
            self.observations[agent] = self.observation_spaces[agent](
                env=self._cent_env, _update_state=_update_state
            )
    
    def _verify_domains(self, domains : MADict) :
        """It verifies if substation ids are valid

        Args:
            domains (dict): 
                - key : agents' names
                - value : list of substation ids
        """
        sum_subs = 0
        for agent in domains.keys():
            if not (isinstance(domains[agent], list)
                    or isinstance(domains[agent], set)
                    or isinstance(domains[agent], np.ndarray)
            ):
                raise DomainException(f"Agent id {agent} : The domain must be a list or a set of substation indices")
            
            if len(domains[agent]) == 0:
                raise DomainException(f"Agent id {agent} : The domain is empty !")
            
            for i in range(len(domains[agent])) : 
                try:
                    domains[agent][i] = int(domains[agent][i])
                except Exception as e:
                    raise DomainException(f"Agent id {agent} : The id must be of type int or convertible to an int. Type {type(domains[agent][i])} is not valid")
                if domains[agent][i] < 0 or domains[agent][i] > self._cent_env.n_sub:
                    raise DomainException(f"Agent id {agent} : The substation's id must be between 0 and {len(self._cent_env.name_sub)-1}, but {domains[agent][i]} has been given")
            sum_subs += len(domains[agent])

        if sum_subs != self._cent_env.n_sub:
            raise DomainException(f"The sum of sub id lists' length must be equal to _cent_env.n_sub = {self._cent_env.n_sub} but is {sum_subs}")
    
    
    def observation_space(self, agent : AgentID):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_space(self, agent : AgentID):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]
    
    def observe(self, agent : AgentID):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        # observations are updated in reset and step methods
        return self.observations[agent]

