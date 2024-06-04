# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""example with local observation and local actions"""

import warnings
import numpy as np
import copy

from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAEnv
from ray.rllib.policy.policy import PolicySpec, Policy
    
import grid2op
from grid2op.Action import PlayableAction
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace

from lightsim2grid import LightSimBackend
from grid2op.gym_compat.utils import ALL_ATTR_FOR_DISCRETE
from grid2op.multi_agent import ClusterUtils

ENV_NAME = "l2rpn_case14_sandbox"
DO_NOTHING_EPISODES = -1  # 200

env_for_cls = grid2op.make(ENV_NAME,
                           action_class=PlayableAction,
                           backend=LightSimBackend())


# Get ACTION_DOMAINS by clustering the substations
ACTION_DOMAINS = ClusterUtils.cluster_substations(env_for_cls)

# Get OBSERVATION_DOMAINS by clustering the substations
OBSERVATION_DOMAINS = ClusterUtils.cluster_substations(env_for_cls)

# wrapper for gym env
class MAEnvWrapper(MAEnv):
    def __init__(self, env_config=None):
        super().__init__()
        if env_config is None:
            env_config = {}

        env = grid2op.make(ENV_NAME,
                           action_class=PlayableAction,
                           backend=LightSimBackend())  

        action_domains = copy.deepcopy(ACTION_DOMAINS)
        if "action_domains" in env_config:
            action_domains = env_config["action_domains"]
        observation_domains = copy.deepcopy(OBSERVATION_DOMAINS)
        if "observation_domains" in env_config:
            observation_domains = env_config["observation_domains"]
        self.ma_env = MultiAgentEnv(env,
                                    action_domains,
                                    observation_domains)
        
        self._agent_ids = set(self.ma_env.agents)
        self.ma_env.seed(0)
        self._agent_ids = self.ma_env.agents
        
        # see the grid2op doc on how to customize the observation space
        # with the grid2op / gym interface.
        self._gym_env = GymEnv(env)
        self._gym_env.observation_space.close()
        obs_attr_to_keep = ["gen_p", "rho"]
        if "obs_attr_to_keep" in env_config:
            obs_attr_to_keep = copy.deepcopy(env_config["obs_attr_to_keep"])
        self._gym_env.observation_space = BoxGymObsSpace(env.observation_space,
                                                         attr_to_keep=obs_attr_to_keep,
                                                         replace_nan_by_0=True  # replace Nan by 0.
                                                         )
        
        # we did not experiment yet with the "partially observable" setting
        # so for now we suppose all agents see the same observation
        # which is the full grid                                
        self._aux_observation_space = {
            agent_id : BoxGymObsSpace(self.ma_env.observation_spaces[agent_id],
                                      attr_to_keep=obs_attr_to_keep,
                                      replace_nan_by_0=True  # replace Nan by 0.
                                      )
            for agent_id in self.ma_env.agents
        }
        # to avoid "weird" pickle issues
        self.observation_space = {
            agent_id : Box(low=self._aux_observation_space[agent_id].low,
                           high=self._aux_observation_space[agent_id].high,
                           dtype=self._aux_observation_space[agent_id].dtype)
            for agent_id in self.ma_env.agents
        }
                
        # we represent the action as discrete action for now. 
        # It should work to encode then differently using the 
        # gym_compat module for example
        act_attr_to_keep = ALL_ATTR_FOR_DISCRETE
        if "act_attr_to_keep" in env_config:
            act_attr_to_keep = copy.deepcopy(env_config["act_attr_to_keep"])
            
        self._conv_action_space = {
            agent_id : DiscreteActSpace(self.ma_env.action_spaces[agent_id], attr_to_keep=act_attr_to_keep)
            for agent_id in self.ma_env.agents
        }
        
        # to avoid "weird" pickle issues
        self.action_space = {
            agent_id : Discrete(n=self.ma_env.action_spaces[agent_id].n)
            for agent_id in self.ma_env.agents
        }
        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        # reset the underlying multi agent environment
        obs = self.ma_env.reset()
        
        return self._format_obs(obs), {}
        
    def seed(self, seed):
        return self.ma_env.seed(seed)
        
    
    def _format_obs(self, grid2op_obs):
        # grid2op_obs is a dictionnary, representing a "multi agent grid2op action"
        
        # convert the observation to a gym one
        
        # return the proper dictionnary
        return {
            agent_id : self._aux_observation_space[agent_id].to_gym(grid2op_obs[agent_id])
            for agent_id in self.ma_env.agents
        }
        
    def step(self, actions):       
        # convert the action to grid2op
        if actions:
            grid2op_act = {
                agent_id : self._conv_action_space[agent_id].from_gym(actions[agent_id])
                for agent_id in self.ma_env.agents
            }
        else:
            grid2op_act = {
                agent_id : self._conv_action_space[agent_id].from_gym(0)
                for agent_id in self.ma_env.agents
            }
            
        # just to retrieve the first agent id...
        first_agent_id = next(iter(self.ma_env.agents))
        
        # do a step in the underlying multi agent environment
        obs, r, done, info = self.ma_env.step(grid2op_act)
        
        # all agents have the same flag "done"
        done['__all__'] = done[first_agent_id]
        
        # now retrieve the observation in the proper form
        gym_obs =  self._format_obs(obs)
        
        # ignored for now
        info = {}
        truncateds = {k: False for k in self.ma_env.agents}
        truncateds['__all__'] = truncateds[first_agent_id]
        return gym_obs, r, done, truncateds, info


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id


if __name__ == "__main__":
    import ray
    # from ray.rllib.agents.ppo import ppo
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    import json
    import os
    import shutil
    
    ray_ma_env = MAEnvWrapper()
    
    checkpoint_root = "./ma_ppo_test_2ndsetting"
    
    # Where checkpoints are written:
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

    # Where some data will be written and used by Tensorboard below:
    ray_results = f'{os.getenv("HOME")}/ray_results/'
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))
    
    # #Configs (see ray's doc for more information)
    SELECT_ENV = MAEnvWrapper                            # Specifies the OpenAI Gym environment for Cart Pole
    N_ITER = 1000                                     # Number of training runs.

    # config = ppo.DEFAULT_CONFIG.copy()              # PPO's default configuration. See the next code cell.
    # config["log_level"] = "WARN"                    # Suppress too many messages, but try "INFO" to see what can be printed.

    # # Other settings we might adjust:
    # config["num_workers"] = 1                       # Use > 1 for using more CPU cores, including over a cluster
    # config["num_sgd_iter"] = 10                     # Number of SGD (stochastic gradient descent) iterations per training minibatch.
    #                                                 # I.e., for each minibatch of data, do this many passes over it to train. 
    # config["sgd_minibatch_size"] = 64              # The amount of data records per minibatch
    # config["model"]["fcnet_hiddens"] = [100, 50]    #
    # config["num_cpus_per_worker"] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed
    # config["vf_clip_param"] = 100

    # # multi agent specific config
    # config["multiagent"] = {
    #     "policies" : {
    #         "agent_0" : PolicySpec(
    #             action_space=ray_ma_env.action_space["agent_0"],
    #             observation_space=ray_ma_env.observation_space["agent_0"],
    #         ),
    #         "agent_1" : PolicySpec(
    #             action_space=ray_ma_env.action_space["agent_1"],
    #             observation_space=ray_ma_env.observation_space["agent_1"],
    #         )
    #         },
    #     "policy_mapping_fn": policy_mapping_fn,
    #     "policies_to_train": ["agent_0", "agent_1"],
    # }

    # #Trainer
    # agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    
    
    # see ray doc for this...
    # syntax changes every ray major version apparently...
    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)
   
    # multi agent parts
    config.multi_agent(policies={
        "agent_0" : PolicySpec(
            action_space=ray_ma_env.action_space["agent_0"],
            observation_space=ray_ma_env.observation_space["agent_0"]
        ),
        "agent_1" : PolicySpec(
            action_space=ray_ma_env.action_space["agent_1"],
            observation_space=ray_ma_env.observation_space["agent_1"],
        )
        }, 
                    policy_mapping_fn = policy_mapping_fn, 
                    policies_to_train= ["agent_0", "agent_1"])
         
    #Trainer
    agent = PPO(config=config, env=SELECT_ENV)

    results = []
    episode_data = []
    episode_json = []

    for n in range(N_ITER):
        result = agent.train()
        results.append(result)
        
        episode = {'n': n, 
                   'episode_reward_min': result['episode_reward_min'], 
                   'episode_reward_mean': result['episode_reward_mean'], 
                   'episode_reward_max': result['episode_reward_max'],  
                   'episode_len_mean': result['episode_len_mean']
                  }
        
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = agent.save(checkpoint_root)
        
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')

        with open(f'{ray_results}/rewards.json', 'w') as outfile:
            json.dump(episode_json, outfile)
