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
from grid2op.Observation.observationSpace import ObservationSpace
from grid2op.dtypes import dt_bool, dt_int
from grid2op.Action import ActionSpace
from grid2op.multi_agent.subGridObjects import SubGridObjects
from grid2op.operator_attention import LinearAttentionBudget
from grid2op.multi_agent.utils import AgentSelector, random_order  
from grid2op.multi_agent.ma_typing import ActionProfile, MADict
from grid2op.multi_agent.multi_agentExceptions import *

import pdb


class MultiAgentEnv :
    def __init__(self,
                 env : Environment,
                 observation_domains : MADict,
                 action_domains : MADict,
                 agent_order_fn = lambda x : x,
                 illegal_action_pen : float = 0.,
                 ambiguous_action_pen : float = 0.,
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
        
        self._cent_env = env
        
        if action_domains.keys() != observation_domains.keys():
            raise("action_domains and observation_domains must have the same agents' names !")
        
        self._verify_domains(action_domains)
        self._verify_domains(observation_domains)
        
        self.num_agents = len(action_domains)
        
        self._action_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in action_domains.items()}
        self._observation_domains = {k: {"sub_id": copy.deepcopy(v)} for k,v in observation_domains.items()}
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
        
        self._build_subgrids()
        self._build_action_spaces()
        self._build_observation_spaces()
        
        
    def reset(self) -> MADict:
        observation = self._cent_env.reset()
        self.observations = self._update_observations(observation)
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
            
        self.observations = self._update_observations(observation)
        
        return self.observations, self.rewards, self.dones, self.info 
    
    def last(self):
        #TODO
        pass
    
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
            self._build_subgrid_from_domain(agent_nm, self._action_domains[agent_nm])
            subgrid_obj = self._build_subgrid_obj_from_domain(self._action_domains[agent_nm])
            self._subgrids_cls['action'][agent_nm] = subgrid_obj
            
            self._build_subgrid_from_domain(agent_nm, self._observation_domains[agent_nm])
            subgrid_obj = self._build_subgrid_obj_from_domain(self._observation_domains[agent_nm])
            self._subgrids_cls['observation'][agent_nm] = subgrid_obj
        
        
    def _build_subgrid_from_domain(self, agent_nm, domain):
        # TODO: rename the function ! (do not forget the tests)
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
        domain["agent_name"] = agent_nm
    
    def _relabel_subid(self, mask, new_label, id_full_grid):
        tmp_ = id_full_grid[mask]
        return new_label[tmp_]
    
    def _build_subgrid_obj_from_domain(self, domain):
        cent_env_cls = type(self._cent_env)
        tmp_cls = SubGridObjects()
        tmp_cls.sub_orig_ids = copy.deepcopy(domain['sub_id'])
        tmp_cls.mask_load = copy.deepcopy(domain['mask_load'])
        tmp_cls.mask_gen = copy.deepcopy(domain['mask_gen'])
        tmp_cls.mask_storage = copy.deepcopy(domain['mask_storage'])
        tmp_cls.mask_shunt = copy.deepcopy(domain['mask_shunt'])
        tmp_cls.mask_line_or = copy.deepcopy(domain['mask_line_or'])
        tmp_cls.mask_line_ex = copy.deepcopy(domain['mask_line_ex'])
        tmp_cls.agent_name = copy.deepcopy(domain["agent_name"])
        tmp_cls.mask_sub = copy.deepcopy(domain["mask_sub"])
        
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
            tmp_cls.mask_line_or | tmp_cls.mask_line_ex
        ]
        tmp_cls.name_sub = cent_env_cls.name_sub[
            tmp_cls.mask_sub
        ]
        tmp_cls.name_storage = cent_env_cls.name_storage[
            tmp_cls.mask_storage
        ]

        tmp_cls.n_gen = len(tmp_cls.name_gen)
        tmp_cls.n_load = len(tmp_cls.name_load)
        tmp_cls.n_line = len(tmp_cls.name_line)
        #tmp_cls.n_line_or = np.count_nonzero(tmp_cls.mask_line_or)
        #tmp_cls.n_line_ex = np.count_nonzero(tmp_cls.mask_line_ex)
        tmp_cls.n_sub = len(tmp_cls.name_sub)
        tmp_cls.n_storage = len(tmp_cls.name_storage)

        tmp_cls.sub_info = cent_env_cls.sub_info[
            tmp_cls.sub_orig_ids
        ]
        tmp_cls.dim_topo = tmp_cls.n_line_ex + tmp_cls.n_line_or + \
            tmp_cls.n_load + tmp_cls.n_gen + tmp_cls.n_storage

        # to which substation is connected each element 
        # this is a bit "tricky" because I have to
        # re label the substation in the subgrid
        tmp_cls.load_to_subid = np.zeros(tmp_cls.n_load, dtype=dt_int)
        tmp_cls.gen_to_subid = np.zeros(tmp_cls.n_gen, dtype=dt_int)
        tmp_cls.line_or_to_subid = np.zeros(tmp_cls.n_line, dtype=dt_int)
        tmp_cls.line_ex_to_subid = np.zeros(tmp_cls.n_line, dtype=dt_int)
        tmp_cls.storage_to_subid = np.zeros(tmp_cls.n_storage, dtype=dt_int)
        
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
        
        # TODO: when we'll thought about it more in depth 
        # tmp_cls.line_to_subid  = self._relabel_subid(tmp_cls.mask_line,
        #                                              new_label,
        #                                              self._cent_env.line_to_subid
        #                                              )

        # which index has this element in the substation vector  #TODO
        tmp_cls.load_to_sub_pos = cent_env_cls.load_to_sub_pos[tmp_cls.mask_load]
        tmp_cls.gen_to_sub_pos = cent_env_cls.gen_to_sub_pos[tmp_cls.mask_gen]
        tmp_cls.storage_to_sub_pos = cent_env_cls.storage_to_sub_pos[tmp_cls.mask_storage]
        # tmp_cls.line_or_to_sub_pos = None
        # tmp_cls.line_ex_to_sub_pos = None

        
        # # which index has this element in the topology vector
        # # "convenient" way to retrieve information of the grid
        # # to which substation each element of the topovect is connected
        # #tmp_cls_action.load_pos_topo_vect = None
        # #tmp_cls_action.gen_pos_topo_vect = None
        # #tmp_cls_action.line_or_pos_topo_vect = None
        # #tmp_cls_action.line_ex_pos_topo_vect = None
        # #tmp_cls_action.storage_pos_topo_vect = None

        # #
        # #tmp_cls_action.grid_objects_types = None
        # #
        # #tmp_cls_action._topo_vect_to_sub = None
        # tmp_cls._compute_pos_big_topo_cls()



        # # redispatch data, not available in all environment
        # tmp_cls.redispatching_unit_commitment_availble = False
        # tmp_cls.gen_type = self._cent_env.gen_type[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_pmin = self._cent_env.gen_pmin[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_pmax = self._cent_env.gen_pmax[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_redispatchable = self._cent_env.gen_redispatchable[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_max_ramp_up = self._cent_env.gen_max_ramp_up[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_max_ramp_down = self._cent_env.gen_max_ramp_down[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_min_uptime = self._cent_env.gen_min_uptime[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_min_downtime = self._cent_env.gen_min_downtime[
        #     tmp_cls.mask_gen
        # ]
        # tmp_cls.gen_cost_per_MW = self._cent_env.gen_cost_per_MW[
        #     tmp_cls.mask_gen
        # ]  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
        # tmp_cls.gen_startup_cost = self._cent_env.gen_startup_cost[
        #     tmp_cls.mask_gen
        # ]  # start cost (in currency)
        # tmp_cls.gen_shutdown_cost = self._cent_env.gen_shutdown_cost[
        #     tmp_cls.mask_gen
        # ]  # shutdown cost (in currency)
        # tmp_cls.gen_renewable = self._cent_env.gen_renewable[
        #     tmp_cls.mask_gen
        # ]

        # # storage unit static data 
        # tmp_cls.storage_type = self._cent_env.storage_type[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_Emax = self._cent_env.storage_Emax[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_Emin = self._cent_env.storage_Emin[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_max_p_prod = self._cent_env.storage_max_p_prod[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_max_p_absorb = self._cent_env.storage_max_p_absorb[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_marginal_cost = self._cent_env.storage_marginal_cost[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_loss = self._cent_env.storage_loss[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_charging_efficiency = self._cent_env.storage_charging_efficiency[
        #     tmp_cls.mask_storage
        # ]
        # tmp_cls.storage_discharging_efficiency = self._cent_env.storage_discharging_efficiency[
        #     tmp_cls.mask_storage
        # ]

        # # grid layout
        # tmp_cls.grid_layout = None if self._cent_env.grid_layout is None else {
        #     k : self._cent_env.grid_layout[k]
        #     for k in tmp_cls.name_sub
        # }

        # # shunt data, not available in every backend 
        # tmp_cls.shunts_data_available = self._cent_env.shunts_data_available[
        #     tmp_cls.mask_shunt
        # ]
        # tmp_cls.n_shunt = self._cent_env.n_shunt[
        #     tmp_cls.mask_shunt
        # ]
        # tmp_cls.name_shunt = self._cent_env.name_shunt[
        #     tmp_cls.mask_shunt
        # ]
        # tmp_cls.shunt_to_subid = self._cent_env.shunt_to_subid[
        #     tmp_cls.mask_shunt
        # ]

        # # alarms #TODO
        # tmp_cls.dim_alarms = 0
        # tmp_cls.alarms_area_names = []
        # tmp_cls.alarms_lines_area = {}
        # tmp_cls.alarms_area_lines = []
        
        return tmp_cls
        
    
    def _build_observation_spaces(self):
        """Build observation spaces from given domains for each agent

        Args:
            observation_domains (_type_): _description_
        """
        #TODO (Is it good ??)
        for agent in self.agents: 
            _cls_agent_action_space = ObservationSpace.init_grid(gridobj=self._subgrids_cls['observation'][agent], extra_name=agent)
            self.observation_spaces[agent] = _cls_agent_action_space(
                gridobj = self._subgrids_cls['observation'][agent],
                env = self._cent_env,
                rewardClass=self._cent_env._rewardClass,
                observationClass=self._cent_env._observationClass,
                actionClass=self._cent_env._actionClass,
                #TODO following parameters
                with_forecast=True,
                kwargs_observation=None,
            )
    
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
    
    
    def _update_observations(self, observation):
        """Update agents' observations from the global observation given by the self._cent_env

        Args:
            observation (_type_): _description_
        """
        #TODO
        pass
    
    def _verify_domains(self, domains : MADict) :
        """It verifies if substation ids are valid

        Args:
            domains (dict): 
                - key : agents' names
                - value : list of substation ids
        """
        for agent in domains.keys():
            if not isinstance(domains[agent], Union[list, np.ndarray, set]) \
                and not isinstance(domains[agent], set):
                raise DomainException("The domain must be a list or a set of substation indices")
            for sub_id in domains[agent] : 
                if not isinstance(sub_id, Union[int, dt_int, np.int32, np.int64]) :
                    raise DomainException(f"The id must be of type int. Type {type(sub_id)} is not valid")
                if sub_id < 0 or sub_id > len(self._cent_env.name_sub) :
                    raise DomainException(f"The substation's id must be between 0 and {len(self._cent_env.name_sub)-1}, but {sub_id} has been given")
    
    
    
    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_spaces(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        # observations are updated in reset and step methods
        return self.observations[agent]





#########################################################################
#########################################################################
# AEC
#------------------------------------------------------------------------

class AECIterable:
    def __init__(self, env, max_iter):
        self.env = env
        self.max_iter = max_iter

    def __iter__(self):
        return AECIterator(self.env, self.max_iter)


class AECIterator:
    def __init__(self, env, max_iter):
        self.env = env
        self.iters_til_term = max_iter

    def __next__(self):
        if not self.env.agents or self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        return self.env.agent_selection


class MultiAgentEnvAEC:
    """
    The AECEnv steps agents one at a time. If you are unsure if you
    have implemented a AECEnv correctly, try running the `api_test` documented in
    the Developer documentation on the website.
    """

    def __init__(self,
                 env_name,
                 agents_names,
                 env_test : bool = False,
                 forbidden_action_pen : float = 1.,
                 ) :
        
        self.env = grid2op.make(env_name, test = env_test)
        self.agent = [str(name) for name in agents_names]
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

    def step(self, action):
        """
        Accepts and add the action of the current agent_selection
        in the environment, automatically switches control to the next agent.
        """
        raise NotImplementedError

    def reset(self, seed=None):
        """
        Resets the environment to a starting state.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Reseeds the environment (making the resulting environment deterministic).
        """
        raise NotImplementedError(
            "Calling seed externally is deprecated; call reset(seed=seed) instead"
        )

    def observe(self, agent):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        # observations are updated in reset and step methods
        return self.observations[agent]

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        """
        raise NotImplementedError

    def state(self):
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def close(self):
        """
        Closes the rendering window, subprocesses, network connections, or any other resources
        that should be released.
        """
        pass

    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        warnings.warn(
            "Your environment should override the observation_space function. Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        warnings.warn(
            "Your environment should override the action_space function. Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def _dones_step_first(self):
        """
        Makes .agent_selection point to first done agent. Stores old value of agent_selection
        so that _was_done_step can restore the variable after the done agent steps.
        """
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        return self.agent_selection

    def _clear_rewards(self):
        """
        clears all items in .rewards
        """
        for agent in self.rewards:
            self.rewards[agent] = 0.

    def _accumulate_rewards(self):
        """
        adds .rewards dictionary to ._cumulative_rewards dictionary. Typically
        called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def agent_iter(self, max_iter=2 ** 63):
        """
        yields the current agent (self.agent_selection) when used in a loop where you step() each iteration.
        """
        return AECIterable(self, max_iter)

    def last(self, observe=True):
        """
        returns observation, cumulative reward, done, info   for the current agent (specified by self.agent_selection)
        """
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.dones[agent],
            self.infos[agent],
        )

    def _was_done_step(self, action):
        """
        Helper function that performs step() for done agents.

        Does the following:

        1. Removes done agent from .agents, .dones, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is done, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Highly recommended to use at the beginning of step as follows:

        def step(self, action):
            if self.dones[self.agent_selection]:
                self._was_done_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError("when an agent is done, the only valid action is None")

        # removes done agent
        agent = self.agent_selection
        assert self.dones[
            agent
        ], "an agent that was not done as attempted to be removed"
        del self.dones[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def __str__(self):
        """
        returns a name which looks like: "space_invaders_v1"
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self):
        return self




class ParallelEnv:
    """
    The Parallel environment steps every live agent at once. If you are unsure if you
    have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
    the Developer documentation on the website.
    """

    def reset(self, seed=None):
        """
        resets the environment and returns a dictionary of observations (keyed by the agent name)
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Reseeds the environment (making it deterministic).
        """
        raise NotImplementedError(
            "Calling seed externally is deprecated; call reset(seed=seed) instead"
        )

    def step(self, actions):
        """
        receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary,
        and info dictionary, where each dictionary is keyed by the agent.
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the rendering window.
        """
        pass

    def state(self):
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        warnings.warn(
            "Your environment should override the observation_space function. Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        warnings.warn(
            "Your environment should override the action_space function. Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def __str__(self):
        """
        returns a name which looks like: "space_invaders_v1" by default
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self):
        return self
    
