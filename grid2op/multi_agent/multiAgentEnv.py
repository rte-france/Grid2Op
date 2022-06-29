# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
from typing import Optional, Tuple
import warnings
import numpy as np
from grid2op.Environment.Environment import Environment

import grid2op
from grid2op.Exceptions.EnvExceptions import EnvError
from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.RandomObject import RandomObject
from grid2op.dtypes import dt_bool, dt_int
from grid2op.Action import ActionSpace
from grid2op.multi_agent.subGridObjects import SubGridObjects
from grid2op.Action import BaseAction

from grid2op.multi_agent.ma_typing import ActionProfile, AgentID, LocalAction, LocalActionSpace, LocalObservation, LocalObservationSpace, MADict
from grid2op.multi_agent.multi_agentExceptions import *
from grid2op.multi_agent.subgridAction import SubGridAction, SubGridActionSpace

import pdb


class MultiAgentEnv(RandomObject):
    def __init__(self,
                 env : Environment,
                 action_domains : MADict,
                 observation_domains : MADict = None,
                 agent_order_fn = lambda x : x, #TODO BEN
                 illegal_action_pen : float = 0.,
                 ambiguous_action_pen : float = 0.,
                 copy_env = True,
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
                    These domains must be non empty and constitute a partition, i.e., the union is the set of all substation ids 
                    and any intersection between two domains is empty.
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
        
        if copy_env:
            self._cent_env : Environment = env.copy()
        else:
            self._cent_env : Environment = env
        
        self._verify_domains(action_domains)
        self._action_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in action_domains.items()}
        self.num_agents = len(action_domains)
        
        if observation_domains is not None:
            # user specified an observation domain
            self._is_global_obs : bool = False
            if action_domains.keys() != observation_domains.keys():
                raise("action_domains and observation_domains must have the same agents' names !")
            self._observation_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in observation_domains.items()}
            self._verify_domains(observation_domains)
        else:
            # no observation domain specified, so I assume it's a domain
            # with full observability
            self._is_global_obs : bool = True
            self._observation_domains = None
        
        self.agents = sorted(list(action_domains.keys()))
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
        self.agent_order = self.agents.copy()
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self._build_subgrids()
        self._build_action_spaces()
        self._build_observation_spaces()
        
        
    def reset(self) -> MADict:
        # TODO : done, need tests
        self._cent_env.reset()
        self._update_observations(_update_state=False)
        return self.observations
    

    def _handle_illegal_action(self, reason):
        
        for a in self.agents:
            self.info[a]['action_is_illegal'] = True
            self.info[a]['reason_illegal'] = reason

    def _handle_ambiguous_action(self, except_tmp):
        
        for a in self.agents:
            self.info[a]['is_ambiguous'] = True
            self.info[a]['ambiguous_except_tmp'] = except_tmp

    def _build_global_action(self, action : ActionProfile, order : list):
        
        self.global_action = self._cent_env.action_space({})
        proposed_action = self.global_action.copy()
        
        for agent in order:
            proposed_action += self._local_action_to_global(agent, action[agent])

        is_legal, reason = self._cent_env._game_rules(action=proposed_action, env=self._cent_env)
        if not is_legal:
            self._handle_illegal_action(reason)
            
        ambiguous, except_tmp = proposed_action.is_ambiguous()
        if ambiguous:
            self._handle_ambiguous_action(except_tmp)
            
        if is_legal and not ambiguous :
            # If the proposed action is valid, we adopt it
            #Otherwise, the global action stays unchanged
            self.global_action = proposed_action.copy()
            
    def step(self, action : ActionProfile) -> Tuple[MADict, MADict, MADict, MADict]:
        """_summary_

        Parameters
        ----------
        action : ActionProfile
            _description_

        Returns
        -------
        Tuple[MADict, MADict, MADict, MADict]
            _description_
        """
        
        order = self.agent_order
        self._build_global_action(action, order)

        self._cent_observation, reward, done, info = self._cent_env.step(self.global_action)
        
        self._dispatch_reward_done_info(reward, done, info)

        self._update_observations()

        return self.observations, self.rewards, self.done, self.info 

    
    def _build_subgrids(self):
        self._subgrids_cls = {
            'action' : dict(),
            'observation' : dict()
        }
        for agent_nm in self.agents : 
            # action space
            self._build_subgrid_masks(agent_nm, self._action_domains[agent_nm])
            subgridcls = self._build_subgrid_cls_from_domain(self._action_domains[agent_nm])
            self._subgrids_cls['action'][agent_nm] = subgridcls 
            
            # observation space
            if self._observation_domains is not None:
                self._build_subgrid_masks(agent_nm, self._observation_domains[agent_nm])
                subgridcls = self._build_subgrid_cls_from_domain(self._observation_domains[agent_nm])
                self._subgrids_cls['observation'][agent_nm] = subgridcls
        
    def _build_subgrid_masks(self, agent_nm, domain):
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
        domain['mask_interco'] = domain['mask_line_or'] ^ domain['mask_line_ex']
        domain['interco_is_origin'] = domain['mask_line_or'][domain['mask_interco']]
        domain['mask_line_or'] = domain['mask_line_or'] & domain['mask_line_ex']
        domain['mask_line_ex'] = domain['mask_line_or'].copy()
        domain["agent_name"] = agent_nm
    
    def _relabel_subid(self, mask, new_label, id_full_grid):
        tmp_ = id_full_grid[mask]
        return new_label[tmp_]
    
    def seed(self, seed):
        # TODO
        raise NotImplementedError()
    
    def _local_action_to_global(self, agent : AgentID, local_action : LocalAction) -> BaseAction :
        # TODO
        # Empty global action
        converted_action = self._cent_env.action_space({})
        subgrid_type = type(local_action)
        
        if local_action._modif_set_bus:
            converted_action._modif_set_bus = True
            converted_action._set_topo_vect[subgrid_type.mask_orig_pos_topo_vect] = local_action._set_topo_vect
        
        if local_action._modif_change_bus:
            converted_action._modif_change_bus = True
            converted_action._change_bus_vect[subgrid_type.mask_orig_pos_topo_vect] = local_action._change_bus_vect
            
        if local_action._modif_set_status:
            converted_action._modif_set_status = True
            converted_action._set_line_status[subgrid_type.line_orig_ids] = local_action._set_line_status
            
        if local_action._modif_change_status:
            converted_action._modif_change_status = True
            converted_action._switch_line_status[subgrid_type.line_orig_ids] = local_action._switch_line_status
        
        if local_action._modif_redispatch:
            converted_action._modif_redispatch = True
            converted_action._redispatch[subgrid_type.gen_orig_ids] = local_action._redispatch
        
        if local_action._modif_storage:
            converted_action._modif_storage = True
            converted_action._storage_power[subgrid_type.gen_orig_ids] = local_action._storage_power
        
        if local_action._modif_curtailment:
            converted_action._modif_curtailment = True
            converted_action._curtail[subgrid_type.gen_orig_ids] = local_action._curtail
        
        # V0
        # TODO set_bus done tested
        # TODO change_bus done tested
        # TODO redispatch done, tested
        # TODO curtail done, tested
        # TODO change_line_status done tested
        # TODO set_line_status done tested
        # TODO set_storage done 
        
        # V inf
        # injection
        # hazards
        # maintenance
        # alarm
        
        return converted_action
    
    def _build_subgrid_cls_from_domain(self, domain):                
        cent_env_cls = type(self._cent_env)
        tmp_subgrid = SubGridObjects()
        
        tmp_subgrid.agent_name = copy.deepcopy(domain["agent_name"])
        
        tmp_subgrid.sub_orig_ids = copy.deepcopy(np.sort(domain['sub_id']))
        tmp_subgrid.mask_sub = copy.deepcopy(domain["mask_sub"])
        
        tmp_subgrid.mask_load = copy.deepcopy(domain['mask_load'])
        tmp_subgrid.mask_gen = copy.deepcopy(domain['mask_gen'])
        tmp_subgrid.mask_storage = copy.deepcopy(domain['mask_storage'])
        tmp_subgrid.mask_line_or = copy.deepcopy(domain['mask_line_or'])
        tmp_subgrid.mask_line_ex = copy.deepcopy(domain['mask_line_ex'])
        tmp_subgrid.mask_interco = copy.deepcopy(domain["mask_interco"])
        tmp_subgrid.mask_shunt = copy.deepcopy(domain["mask_shunt"])
        tmp_subgrid.interco_is_origin = copy.deepcopy(domain["interco_is_origin"])
        
        tmp_subgrid.load_orig_ids = np.where(domain['mask_load'])[0]
        tmp_subgrid.gen_orig_ids = np.where(domain['mask_gen'])[0]
        tmp_subgrid.storage_orig_ids = np.where(domain['mask_storage'])[0]
        tmp_subgrid.line_orig_ids = np.where(domain['mask_line_or'])[0]
        
        tmp_subgrid.glop_version = cent_env_cls.glop_version
        tmp_subgrid._PATH_ENV = cent_env_cls._PATH_ENV

        # name of the objects
        tmp_subgrid.env_name = cent_env_cls.env_name
        tmp_subgrid.name_load = cent_env_cls.name_load[
            tmp_subgrid.mask_load
        ]
        tmp_subgrid.name_gen = cent_env_cls.name_gen[
            tmp_subgrid.mask_gen
        ]
        tmp_subgrid.name_line = cent_env_cls.name_line[
            tmp_subgrid.mask_line_or & tmp_subgrid.mask_line_ex
        ]
        tmp_subgrid.name_sub = cent_env_cls.name_sub[
            tmp_subgrid.mask_sub
        ]
        tmp_subgrid.name_storage = cent_env_cls.name_storage[
            tmp_subgrid.mask_storage
        ]
        
        tmp_subgrid.n_gen = len(tmp_subgrid.name_gen)
        tmp_subgrid.n_load = len(tmp_subgrid.name_load)
        tmp_subgrid.n_line = len(tmp_subgrid.name_line)
        tmp_subgrid.n_sub = len(tmp_subgrid.name_sub)
        tmp_subgrid.n_storage = len(tmp_subgrid.name_storage)
        tmp_subgrid.n_interco = tmp_subgrid.mask_interco.sum()

        tmp_subgrid.sub_info = cent_env_cls.sub_info[
            tmp_subgrid.sub_orig_ids
        ]
        
        tmp_subgrid.dim_topo = 2*tmp_subgrid.n_line + tmp_subgrid.n_interco + \
            tmp_subgrid.n_load + tmp_subgrid.n_gen + tmp_subgrid.n_storage

        # to which substation is connected each element 
        # this is a bit "tricky" because I have to
        # re label the substation in the subgrid
        tmp_subgrid.load_to_subid = np.zeros(tmp_subgrid.n_load, dtype=dt_int)
        tmp_subgrid.gen_to_subid = np.zeros(tmp_subgrid.n_gen, dtype=dt_int)
        tmp_subgrid.line_or_to_subid = np.zeros(tmp_subgrid.n_line, dtype=dt_int)
        tmp_subgrid.line_ex_to_subid = np.zeros(tmp_subgrid.n_line, dtype=dt_int)
        tmp_subgrid.storage_to_subid = np.zeros(tmp_subgrid.n_storage, dtype=dt_int)
        
        # new_label[orig_grid_sub_id] = new_grid_sub_id
        new_label = np.zeros(cent_env_cls.n_sub, dtype=dt_int) - 1
        new_label[tmp_subgrid.sub_orig_ids] = np.arange(tmp_subgrid.n_sub)
        
        tmp_subgrid.load_to_subid  = self._relabel_subid(tmp_subgrid.mask_load,
                                                     new_label,
                                                     cent_env_cls.load_to_subid
        )
        tmp_subgrid.gen_to_subid  = self._relabel_subid(tmp_subgrid.mask_gen,
                                                     new_label,
                                                     cent_env_cls.gen_to_subid
        )
        tmp_subgrid.storage_to_subid  = self._relabel_subid(tmp_subgrid.mask_storage,
                                                        new_label,
                                                        cent_env_cls.storage_to_subid
        )
        tmp_subgrid.line_or_to_subid = self._relabel_subid(tmp_subgrid.mask_line_or,
                                                        new_label,
                                                        cent_env_cls.line_or_to_subid
        )
        tmp_subgrid.line_ex_to_subid = self._relabel_subid(tmp_subgrid.mask_line_ex,
                                                        new_label,
                                                        cent_env_cls.line_ex_to_subid
        )
        
        interco_to_subid = np.zeros(cent_env_cls.n_line, dtype=dt_int) - 1
        tmp_ = np.zeros(tmp_subgrid.n_interco, dtype=dt_int) - 1
        tmp_[tmp_subgrid.interco_is_origin] = cent_env_cls.line_or_to_subid[tmp_subgrid.mask_interco][tmp_subgrid.interco_is_origin]
        tmp_[~tmp_subgrid.interco_is_origin] = cent_env_cls.line_ex_to_subid[tmp_subgrid.mask_interco][~tmp_subgrid.interco_is_origin]
        interco_to_subid[tmp_subgrid.mask_interco] = tmp_
        tmp_subgrid.interco_to_subid = self._relabel_subid(tmp_subgrid.mask_interco,
                                                           new_label,
                                                           interco_to_subid)
        
        tmp_subgrid.interco_to_lineid = np.arange(cent_env_cls.n_line)[tmp_subgrid.mask_interco]
        tmp_subgrid.name_interco = np.array([
            f'interco_{i}_line_{tmp_subgrid.interco_to_lineid[i]}' for i in range(len(tmp_subgrid.interco_is_origin))
        ])

        # which index has this element in the substation vector 
        tmp_subgrid.load_to_sub_pos = cent_env_cls.load_to_sub_pos[tmp_subgrid.mask_load]
        tmp_subgrid.gen_to_sub_pos = cent_env_cls.gen_to_sub_pos[tmp_subgrid.mask_gen]
        tmp_subgrid.storage_to_sub_pos = cent_env_cls.storage_to_sub_pos[tmp_subgrid.mask_storage]
        tmp_subgrid.line_or_to_sub_pos = cent_env_cls.line_or_to_sub_pos[tmp_subgrid.mask_line_or]
        tmp_subgrid.line_ex_to_sub_pos = cent_env_cls.line_ex_to_sub_pos[tmp_subgrid.mask_line_ex]
        
        # Depending on whether the interco is a line_or or a line_ex,
        # we take the corresponding sub_pos in cent_env 
        tmp_ = np.zeros(tmp_subgrid.n_interco, dtype=dt_int) - 1
        tmp_[tmp_subgrid.interco_is_origin] = cent_env_cls.line_or_to_sub_pos[tmp_subgrid.mask_interco][tmp_subgrid.interco_is_origin]
        tmp_[~tmp_subgrid.interco_is_origin] = cent_env_cls.line_ex_to_sub_pos[tmp_subgrid.mask_interco][~tmp_subgrid.interco_is_origin]
        tmp_subgrid.interco_to_sub_pos = tmp_

        # redispatch data, not available in all environment
        tmp_subgrid.redispatching_unit_commitment_availble = cent_env_cls.redispatching_unit_commitment_availble
        if tmp_subgrid.redispatching_unit_commitment_availble:
            tmp_subgrid.gen_type = cent_env_cls.gen_type[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_pmin = cent_env_cls.gen_pmin[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_pmax = cent_env_cls.gen_pmax[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_redispatchable = cent_env_cls.gen_redispatchable[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_max_ramp_up = cent_env_cls.gen_max_ramp_up[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_max_ramp_down = cent_env_cls.gen_max_ramp_down[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_min_uptime = cent_env_cls.gen_min_uptime[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_min_downtime = cent_env_cls.gen_min_downtime[
                tmp_subgrid.mask_gen
            ]
            tmp_subgrid.gen_cost_per_MW = cent_env_cls.gen_cost_per_MW[
                tmp_subgrid.mask_gen
            ]  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
            tmp_subgrid.gen_startup_cost = cent_env_cls.gen_startup_cost[
                tmp_subgrid.mask_gen
            ]  # start cost (in currency)
            tmp_subgrid.gen_shutdown_cost = cent_env_cls.gen_shutdown_cost[
                tmp_subgrid.mask_gen
            ]  # shutdown cost (in currency)
            tmp_subgrid.gen_renewable = cent_env_cls.gen_renewable[
                tmp_subgrid.mask_gen
            ]

        # shunt data, not available in every backend 
        tmp_subgrid.shunts_data_available = cent_env_cls.shunts_data_available
        if tmp_subgrid.shunts_data_available:
            tmp_subgrid.mask_shunt = copy.deepcopy(domain['mask_shunt'])
            tmp_subgrid.name_shunt = cent_env_cls.name_shunt[
                tmp_subgrid.mask_shunt
            ]
            tmp_subgrid.shunt_to_subid  = self._relabel_subid(tmp_subgrid.mask_shunt,
                                                      new_label,
                                                      cent_env_cls.shunt_to_subid
            )
            tmp_subgrid.n_shunt = len(tmp_subgrid.name_shunt)
            tmp_subgrid.shunt_orig_ids = np.where(domain['mask_shunt'])[0]

        # storage unit static data 
        tmp_subgrid.storage_type = cent_env_cls.storage_type[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_Emax = cent_env_cls.storage_Emax[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_Emin = cent_env_cls.storage_Emin[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_max_p_prod = cent_env_cls.storage_max_p_prod[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_max_p_absorb = cent_env_cls.storage_max_p_absorb[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_marginal_cost = cent_env_cls.storage_marginal_cost[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_loss = cent_env_cls.storage_loss[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_charging_efficiency = cent_env_cls.storage_charging_efficiency[
            tmp_subgrid.mask_storage
        ]
        tmp_subgrid.storage_discharging_efficiency = cent_env_cls.storage_discharging_efficiency[
            tmp_subgrid.mask_storage
        ]

        # grid layout
        tmp_subgrid.grid_layout = None if cent_env_cls.grid_layout is None else {
            k : cent_env_cls.grid_layout[k]
            for k in tmp_subgrid.name_sub
        }

        # alarms
        tmp_subgrid.dim_alarms = 0
        tmp_subgrid.alarms_area_names = []
        tmp_subgrid.alarms_lines_area = {}
        tmp_subgrid.alarms_area_lines = []
        if cent_env_cls.dim_alarms != 0:
            warnings.warn("Alarms are not yet handled by the \"multi agent\" environment. They have been deactivated")
        
        # mask to the original pos topo vect
        tmp_subgrid.mask_orig_pos_topo_vect = np.full(cent_env_cls.dim_topo, fill_value=False, dtype=dt_bool)
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.load_pos_topo_vect[tmp_subgrid.mask_load]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.gen_pos_topo_vect[tmp_subgrid.mask_gen]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.storage_pos_topo_vect[tmp_subgrid.mask_storage]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.line_or_pos_topo_vect[tmp_subgrid.mask_line_or]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.line_ex_pos_topo_vect[tmp_subgrid.mask_line_ex]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.line_or_pos_topo_vect[tmp_subgrid.mask_interco][tmp_subgrid.interco_is_origin]] = True
        tmp_subgrid.mask_orig_pos_topo_vect[cent_env_cls.line_ex_pos_topo_vect[tmp_subgrid.mask_interco][~tmp_subgrid.interco_is_origin]] = True
        
        extra_name = self._get_cls_name_from_domain(domain)
        res_cls = SubGridObjects.init_grid(gridobj=tmp_subgrid, extra_name=extra_name)

        # make sure the class is consistent
        res_cls.assert_grid_correct_cls()
        return res_cls
    
    def _get_cls_name_from_domain(self, domain=None, agent_name=None):
        if domain is not None and agent_name is None:
            extra_name = domain["agent_name"]
        elif agent_name is not None and domain is None:
            extra_name = agent_name
        elif agent_name is None and domain is None:
            raise EnvError("give at least agent_name or domain")
        else:
            raise EnvError("give exactly one of agent_name or domain")
        
        if self._add_to_name is not None:
            extra_name += f"{self._add_to_name}"
        return extra_name
    
    def _build_observation_spaces(self):
        """Build observation spaces from given domains for each agent

        Args:
            global_obs (bool, optional): The agent can observe the whole grid. Defaults to True.

        Raises:
            NotImplementedError: Local observations are not implemented yet
        """
        
        if self._is_global_obs:
            # in case of global observation, I simply copy the observation space
            for agent in self.agents:
                self.observation_spaces[agent] = self._cent_env.observation_space.copy()
        else:
            raise NotImplementedError("Local observations are not available yet !")
    
    def _build_action_spaces(self):
        """Build action spaces from given domains for each agent
        The action class is the same as 
        """
        # TODO may be coded, tests not done
        for agent_nm in self.agents: 
            this_subgrid = self._subgrids_cls['action'][agent_nm]
            extra_name = self._get_cls_name_from_domain(agent_name=agent_nm)
            _cls_agent_action_space = SubGridActionSpace.init_grid(gridobj=this_subgrid, extra_name=extra_name)
            self.action_spaces[agent_nm] = _cls_agent_action_space(
                gridobj=this_subgrid,
                agent_name=extra_name,
                # actionClass=self._cent_env._actionClass_orig,  # TODO later: have a specific action class for MAEnv
                actionClass=SubGridAction,  # TODO later: have a specific action class for MAEnv
                legal_action=self._cent_env._game_rules.legal_action,  # TODO later: probably not no... But we'll see when we do the rules
            )
    
    def _update_observations(self, _update_state = True):
        """Update agents' observations from the global observation given by the self._cent_env

        Args:
            observation (BaseObservation): _description_
        """
        #for agent in self.agents:
        #    self.observations[agent] = observation.copy()
        
        #TODO BEN: check that cent env is initialized and not closed
        #TODO BEN: update self.observations
        
        raise NotImplementedError()
    
    def _verify_domains(self, domains : MADict) -> None:
        """It verifies if substation ids are valid

        Args:
            domains (dict): 
                - key : agents' names
                - value : list of substation ids convertible to int (allowed types are list, set and np.ndarray)
                These domains must be non empty and constitute a partition, i.e., the union is the set of all substation ids 
                in the global env and any intersection between two domains is empty.
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
    
    def observation_space(self, agent : AgentID)-> LocalObservationSpace:
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_space(self, agent : AgentID) -> LocalActionSpace:
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]
    
    def observe(self, agent : AgentID) -> LocalObservation:
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        # observations are updated in reset and step methods
        return self.observations[agent]
