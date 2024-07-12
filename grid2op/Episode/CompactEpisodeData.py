# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# Addition by Xavier Weiss (@DEUCE1957)
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import json
import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Observation import ObservationSpace

from pathlib import Path as p


class CompactEpisodeData():

    """
    This module provides a compact way to serialize/deserialize one episode of a Reinforcement Learning (RL) run.
    This enables episodes to be replayed, so we can understand the behaviour of the agent. 
    It is compatible with :class:`EpisodeData` through the "to_dict()" method.

    If enabled when using the :class:`Runner`, the :class:`CompactEpisodeData`
    will save the information in a structured and compact way.
    For each unique environment it will store a folder with:
      - "dict_action_space.json"
      - "dict_attack_space.json"
      - "dict_env_modification.json"
      - "dict_observation_space.json"
    Then for each episode it stores a single compressed Numpy archive (.npz) file, identified by the chronics ID (e.g. "003").
    Inside this archive we find:
      - "actions": actions taken by the :class:`grid2op.BaseAgent.BaseAgent`, each row of this numpy 2d-array is a vector representation of the action
        taken by the agent at a particular timestep.
      - "env_actions": represents the modification of the powergrid by the environment, these modification usually concern hazards, maintenance, as well as modification of the generators production
        setpoint or the loads consumption.
      - "attacks": actions taken by any opponent present in the RL environment, stored similary to "actions".
      - "observations": observations of the class :class:`grid2op.BaseObservation.BaseObservation made by the :class:`grid2op.Agent.BaseAgent` after taking an action, stored as a numpy 2d-array
        where each row corresponds to a vector representation of the observation at that timestep. Note this includes the initial timestep, hence this array is 1 row longer than (e.g.) the actionss.
      - "rewards": reward received by the :class:`grid2op.Agent.BaseAgent from the :class:`grid2op.Environment` at timestep 't', represented as 1d-array.
      - "other_rewards": any other rewards logged by the :class:`grid2op.Environment` (but not necessarily passed to the agent), represented as a 2d-array.
      - "disc_lines": gives which lines have been disconnected during the simulation at each time step. The same convention as for "rewards" has been adopted. This means that the powerlines are
        disconnected when the :class:`grid2op.Agent.BaseAgent` takes the :class:`grid2op.BaseAction` at timestep 't`.
      - "times": gives some information about the processor time spent (in seconds), mainly the time taken by
        :class:`grid2op.Agent.BaseAgent` (and especially its method :func:`grid2op.BaseAgent.act`) and amount of time
        spent in the :class:`grid2op.Environment.Environment`
    All of the above can be read back from disk.

    Inputs
    ----------
    environment: :class:`grid2op.Environment`
        The environment we are running, contains most of the metadata required to store the episode.
    obs: :class:`grid2op.Observation`
        The initial observation of the environment in the current episode. Used to store the first observation.
    
    Examples
    --------
    Here is an example on how to use the :class:`CompactEpisodeData` class outside of the :class:`grid2op.Runner.Runner`.

    .. code-block:: python
        from pathlib import Path as p
        from grid2op.Agent import DoNothingAgent
        env = grid2op.make(""rte_case14_realistic")
        obs = env.reset()
        ep_id = env.chronics_handler.get_name()
        data_dir = p.cwd() # CHANGE THIS TO DESIRED LOCATION ON DISK
        agent = DoNothingAgent(env.action_space)
        reward = 0.0
        episode_store = CompactEpisodeData(env, obs)
        for t in range(env.max_episode_duration()):
            start = time.perf_counter()
            act = agent.act(obs, reward)
            obs, reward, done, info = env.step(act)
            duration = time.perf_counter() - start
            episode_store.update(t, env, act, obs, reward, duration, info)
        # Store Episode Data to file (compactly)
        episode_store.to_disk()
        # Load Episode Data from disk by referring to the specific episode ID
        episode_store.from_disk(ep_id)
    """
        
    def __init__(self, env, obs, exp_dir, ep_id:str=None):
        """
        Creates Dictionary of Numpy Arrays for storing the details of a Grid2Op Episode (actions, observations, etc.).
        Pre-allocating the arrays like this is more efficient than appending to a mutable datatype (like a list).
        For the initial timestep, an extra observation is stored (the initial state of the Environment). 

        Args:
            env (grid2op.Environment): Current Grid2Op Environment, used to grab static attributes.
            obs (grid2op.Observation): Initial Observation (before agent is active)
            exp_dir (pathlib.Path): Where experiment data is stored
            ep_id (str | None): If provided tries to load previously stored episode from disk.
        
        Returns:
            dict<str:obj>: Contains all data to fully represent what happens in an episode
        """
        if exp_dir is not None:
            self.exp_dir = p(exp_dir)
        else:
            self.exp_dir = None
        self.array_names = ("actions", "env_actions", "attacks", "observations", "rewards", "other_rewards", "disc_lines", "times")
        self.space_names = ("observation_space", "action_space", "attack_space", "env_modification_space")
        if ep_id is None:
            self.ep_id = env.chronics_handler.get_name()
            max_no_of_timesteps = int(env.max_episode_duration())
            
            # Numpy Arrays
            self.actions = np.full((max_no_of_timesteps, env.action_space.n), fill_value=np.NaN, dtype=np.float16)
            self.env_actions = np.full((max_no_of_timesteps, env._helper_action_env.n), fill_value=np.NaN, dtype=np.float32)
            self.attacks = np.full((max_no_of_timesteps, env._opponent_action_space.n), fill_value=0.0, dtype=np.float32)
            self.observations = np.full((max_no_of_timesteps + 1, len(obs.to_vect())),fill_value=np.NaN,dtype=np.float32)
            self.rewards = np.full(max_no_of_timesteps, fill_value=np.NaN, dtype=np.float32)
            self.other_reward_names = list(sorted(env.other_rewards.keys()))
            self.other_rewards = np.full((max_no_of_timesteps, len(self.other_reward_names)), fill_value=np.NaN, dtype=np.float32)
            self.disc_lines = np.full((max_no_of_timesteps, env.backend.n_line), fill_value=np.NaN, dtype=np.bool_)
            self.times = np.full(max_no_of_timesteps, fill_value=np.NaN, dtype=np.float32)
            
            self.disc_lines_templ = np.full((1, env.backend.n_line), fill_value=False, dtype=np.bool_)
            # AttackTempl: Not used, kept for comptabiility with EpisodeData
            self.attack_templ = np.full((1, env._oppSpace.action_space.size()), fill_value=0.0, dtype=np.float32)

            self.legal = np.full(max_no_of_timesteps, fill_value=True, dtype=np.bool_)
            self.ambiguous = np.full(max_no_of_timesteps, fill_value=False, dtype=np.bool_)
            self.n_cols = env.action_space.n + env._helper_action_env.n + len(obs.to_vect()) + env.backend.n_line + env._oppSpace.action_space.size() + 6

            # Store First Observation
            self.observations[0] = obs.to_vect()
            self.game_over_timestep = max_no_of_timesteps

            # JSON-serializable Objects
            self.observation_space=env.observation_space
            self.action_space=env.action_space
            self.attack_space=env._opponent_action_space
            self.env_modification_space=env._helper_action_env
            
            # Special JSON-Serializable Object: Episode MetaData
            self.meta = dict(
                chronics_path = self.ep_id, 
                chronics_max_timestep = max_no_of_timesteps,
                game_over_timestep = self.game_over_timestep,
                other_reward_names = self.other_reward_names,
                grid_path = env._init_grid_path,
                backend_type = type(env.backend).__name__,
                env_type = type(env).__name__,
                env_seed = (env.seed_used.item() if env.seed_used.ndim == 0 else list(env.seed_used)) if isinstance(env.seed_used, np.ndarray) else env.seed_used,
                agent_seed = self.action_space.seed_used,
                nb_timestep_played = 0,
                cumulative_reward = 0.0,
            )
        elif exp_dir is not None:
            self.load_metadata(ep_id)
            self.load_spaces()
            self.load_arrays(ep_id)
        
    def update(self, t:int, env, action,
               obs, reward:float, done:bool, duration:float, info):
        """
        Update the arrays in the Episode Store for each step of the environment.
        Args:
            t (int): Current time step
            env (grid2op.Environment): State of Environment
            action (grid2op.Action): Action agent took on the Environment
            obs (grid2op.Observation): Observed result of action on Environment
            reward (float): Numeric reward returned by Environment for the given action
            duration (float): Time in seconds needed to choose and execute the action
            info (dict<str:np.array>): Dictionary containing information on legality and ambiguity of action
        """
        self.actions[t - 1] = action.to_vect()
        self.env_actions[t - 1] = env._env_modification.to_vect()
        self.observations[t] = obs.to_vect()
        opp_attack = env._oppSpace.last_attack
        if opp_attack is not None:
            self.attacks[t - 1] = opp_attack.to_vect()
        self.rewards[t - 1] = reward
        if "disc_lines" in info:
            arr = info["disc_lines"]
            if arr is not None:
                self.disc_lines[t - 1] = arr
            else:
                self.disc_lines[t - 1] = self.disc_lines_templ
        if "rewards" in info:
            for i, other_reward_name in enumerate(self.other_reward_names):
                self.other_rewards[t-1, i] = info["rewards"][other_reward_name]
        self.times[t - 1] = duration
        self.legal[t - 1] = not info["is_illegal"]
        self.ambiguous[t - 1] = info["is_ambiguous"]
        if done:
            self.game_over_timestep = t
        # Update metadata
        self.meta.update(
            nb_timestep_played = t,
            cumulative_reward = self.meta["cumulative_reward"] + float(reward),
        )
        return self.meta["cumulative_reward"]
    
    def asdict(self):
        """
        Return the Episode Store as a dictionary.
        Compatible with Grid2Op's internal EpisodeData format as keyword arguments.
        """
        # Other rewards in Grid2op's internal Episode Data is a list of dictionaries, so we convert to that format
        other_rewards = [{other_reward_name:float(self.other_rewards[t, i]) for i, other_reward_name in enumerate(self.other_reward_names)} for t in range(len(self.times))]
        return dict(actions=self.actions, env_actions=self.env_actions,
                    observations=self.observations,
                    rewards=self.rewards, 
                    other_rewards=other_rewards,
                    disc_lines=self.disc_lines, times=self.times,
                    disc_lines_templ=self.disc_lines_templ, attack_templ=self.attack_templ,
                    attack=self.attacks, legal=self.legal, ambiguous=self.ambiguous,
                    observation_space=self.observation_space, action_space=self.action_space,
                    attack_space=self.attack_space, helper_action_env=self.env_modification_space)
    
    def store_metadata(self):
        """
        Store this Episode's meta data to disk.
        """
        with open(self.exp_dir / f"{self.ep_id}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=4, sort_keys=True)
    
    def load_metadata(self, ep_id:str):
        """
        Load metadata from a specific Episode.
        """
        with open(self.exp_dir / f"{ep_id}_metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            self.other_reward_names = self.meta["other_reward_names"]
            self.game_over_timestep = self.meta["game_over_timestep"]
    
    def store_spaces(self):
        """
        Store the Observation, Action, Environment and Opponent spaces to disk.
        """
        for space_name in self.space_names:
            with open(self.exp_dir / f"dict_{space_name}.json", "w", encoding="utf-8") as f:
                json.dump(getattr(self, space_name).cls_to_dict(), f, indent=4, sort_keys=True)

    def load_spaces(self):
        """
        Load the Observation, Action, Environment and Opponent spaces from disk
        """
        for space_name in self.space_names:
            with open(self.exp_dir / f"dict_{space_name}.json", "r", encoding="utf-8") as f:
                if space_name == "observation_space":
                    setattr(self, space_name, ObservationSpace.from_dict(json.load(f)))
                else:
                    setattr(self, space_name, ActionSpace.from_dict(json.load(f)))
    
    def store_arrays(self):
        """
        Store compressed versions of the Actions, Observations, Rewards, Attacks and other metadata 
        to disk as a compressed numpy archive (single file per episode).
        """
        np.savez_compressed(self.exp_dir / f"{self.ep_id}.npz", **{array_name: getattr(self, array_name) for array_name in self.array_names})

    def load_arrays(self, ep_id:str):
        """
        Load Actions, Observations, Rewards, Attacks and other metadata from disk
        for a specific Episode ID (identified by Chronics name)
        """
        arrays = np.load(self.exp_dir / f"{ep_id}.npz")
        for array_name in self.array_names:
            setattr(self, array_name, arrays[array_name])
        self.ep_id = ep_id
    
    def to_disk(self):
        """
        Store this EpisodeStore object instance to disk (as .json and .npz files)
        """
        if self.exp_dir is not None:
            # Store Episode metadata
            self.store_metadata()
            # Store Spaces (values are static, so only save once per experiment)
            if len([f for f in self.exp_dir.glob("*.json")]) != 4:
                self.store_spaces()
            # Store Arrays as Compressed Numpy archive
            self.store_arrays()

    @classmethod
    def from_disk(cls, path, ep_id:str):
        """
        Load EpisodeStore data from disk for a specific episode.
        """
        return cls(env=None, obs=None, exp_dir=p(path), ep_id=ep_id)

    @staticmethod
    def list_episode(path):
        """
        From a given path, extracts the episodes that can be loaded

        Parameters
        ----------
        path: ``str``
            The path where to look for data coming from "episode"

        Returns
        -------
        res: ``list``
            A list of possible episodes. Each element of this list is a tuple: (full_path, episode_name)
        """
        return [(str(full_path), full_path.stem) for full_path in path.glob("*.npz")]
    
    def __len__(self):
        return self.game_over_timestep
    
    def make_serializable(self):
        """
        INTERNAL

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by he runner to serialize properly an episode

        Called in the _aux_run_one_episode (one of the Runner auxilliary function) to make 
        sure the EpisodeData can be sent back to the main process withtout issue (otherwise
        there is a complain about the _ObsEnv)
        """
        from grid2op.Episode.EpisodeData import EpisodeData
        EpisodeData._aux_make_obs_space_serializable(self)
