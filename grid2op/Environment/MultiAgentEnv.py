# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import copy
import warnings
import numpy as np
import re

import grid2op
from grid2op.dtypes import dt_float, dt_bool, dt_int
from grid2op.Action import ActionSpace, BaseAction, TopologyAction, DontAct, CompleteAction
from grid2op.Exceptions import *
from grid2op.Observation import CompleteObservation, ObservationSpace, BaseObservation
from grid2op.Reward import FlatReward, RewardHelper, BaseReward
from grid2op.Rules import RulesChecker, AlwaysLegal, BaseRules
from grid2op.Backend import Backend
from grid2op.Chronics import ChronicsHandler
from grid2op.VoltageControler import ControlVoltageFromFile, BaseVoltageController
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.operator_attention import LinearAttentionBudget
from grid2op.utils.multi_agent_utils import AgentSelector, random_order

from grid2op.Backend import PandaPowerBackend




class MultiAgentEnv :
    def __init__(self,
                 env_name,
                 agents_names,
                 env_test : bool = False,
                 agent_order_fn = random_order
                 ) :
        
        self.env = grid2op.make(env_name, test = env_test)
        self.agents = [name for name in agents_names]
        self.rewards = dict(
            zip(
                self.agents, [0. for _ in range(len(self.agents))]
            )
        )
        self.select_agent = AgentSelector(len(self.agents), agent_order_fn)
        
    def reset(self):
        observation = self.env.reset()
        #TODO make observations for all the agents
        
    def step(self, action):
        #TODO
        pass 
    
    def last(self):
        #TODO
        pass



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
    
