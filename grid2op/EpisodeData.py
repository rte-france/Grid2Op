"""
This module provides a way to serialize on disk et deserialize one run episode along with some 
methods and utilities to ease its manipulation.

If enabled when usign the :class:`Runner`, the :class:`EpisodeData` will save the information in a structured way. For each episode there will be a folder
with:

  - "episode_meta.json" that represents some meta information about:

    - "backend_type": the name of the :class:`grid2op.Backend` class used
    - "chronics_max_timestep": the **maximum** number of timestep for the chronics used
    - "chronics_path": the path where the temporal data (chronics) are located
    - "env_type": the name of the :class:`grid2op.Environment` class used.
    - "grid_path": the path where the powergrid has been loaded from

  - "episode_times.json": gives some information about the total time spend in multiple part of the runner, mainly the
    :class:`grid2op.Agent` (and especially its method :func:`grid2op.Agent.act`) and amount of time spent in the
    :class:`grid2op.Environment`

  - "_parameters.json": is a representation as json of a the :class:`grid2op.Parameters.Parameters` used for this episode
  - "rewards.npy" is a numpy 1d array giving the rewards at each time step. We adopted the convention that the stored
    reward at index `i` is the one observed by the agent at time `i` and **NOT** the reward sent by the
    :class:`grid2op.Environment` after the action has been implemented.
  - "exec_times.npy" is a numpy 1d array giving the execution time of each time step of the episode
  - "actions.npy" gives the actions that has been taken by the :class:`grid2op.Agent.Agent`. At row `i` of "actions.npy" is a
    vectorized representation of the action performed by the agent at timestep `i` *ie.* **after** having observed
    the observation present at row `i` of "observation.npy" and the reward showed in row `i` of "rewards.npy".
  - "disc_lines.npy" gives which lines have been disconnected during the simulation of the cascading failure at each
    time step. The same convention as for "rewards.npy" has been adopted. This means that the powerlines are
    disconnected when the :class:`grid2op.Agent` takes the :class:`grid2op.Action` at time step `i`.
  - "observations.npy" is a numpy 2d array reprensenting the :class:`grid2op.Observation.Observation` at the disposal of the
    :class:`grid2op.Agent` when he took his action.
  - "env_modifications.npy" is a 2d numpy array representing the modification of the powergrid from the environment.
    these modification usually concerns the hazards, maintenance, as well as modification of the generators production
    setpoint or the loads consumption.

All of the above should allow to read back, and better understand the behaviour of some :class:`grid2op.Agent.Agent`, even
though such utility functions have not been coded yet.
"""

import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd

try:
    from .Exceptions import Grid2OpException, AmbiguousAction
    from .Utils import ActionSpace, ObservationSpace
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException, AmbiguousAction
    from Utils import ActionSpace, ObservationSpace


class EpisodeData:
    ACTION_SPACE = "dict_action_space.json"
    OBS_SPACE = "dict_observation_space.json"
    ENV_MODIF_SPACE = "dict_env_modification_space.json"
    PARAMS = "_parameters.json"
    META = "episode_meta.json"
    TIMES = "episode_times.json"

    AG_EXEC_TIMES = "agent_exec_times.npy"
    ACTIONS = "actions.npy"
    ENV_ACTIONS = "env_modifications.npy"
    OBSERVATIONS = "observations.npy"
    LINES_FAILURES = "disc_lines_cascading_failure.npy"
    REWARDS = "rewards.npy"

    def __init__(self, actions=None, env_actions=None, observations=None, rewards=None,
                 disc_lines=None, times=None,
                 params=None, meta=None, episode_times=None,
                 observation_space=None, action_space=None,
                 helper_action_env=None, path_save=None, disc_lines_templ=None,
                 logger=None, name=str(1), get_dataframes=None):

        self.actions = CollectionWrapper(actions, action_space, "actions")
        self.observations = CollectionWrapper(observations, observation_space,
                                              "observations")

        self.env_actions = CollectionWrapper(env_actions, helper_action_env,
                                             "env_actions")
        self.observation_space = observation_space
        self.helper_action_env = helper_action_env
        self.rewards = rewards
        self.disc_lines = disc_lines
        self.times = times
        self.params = params
        self.meta = meta
        self.episode_times = episode_times
        self.name = name
        self.disc_lines_templ = disc_lines_templ
        self.logger = logger
        self.serialize = False
        self.load_names = action_space.name_load
        self.n_loads = len(self.load_names)
        self.prod_names = action_space.name_gen
        self.n_prods = len(self.prod_names)
        self.line_names = action_space.name_line
        self.n_lines = len(self.line_names)
        self.name_sub = action_space.name_sub

        if path_save is not None:
            self.agent_path = os.path.abspath(path_save)
            self.episode_path = os.path.join(self.agent_path, name)
            self.serialize = True
            if not os.path.exists(self.agent_path):
                os.mkdir(self.agent_path)
                self.logger.info(
                    "Creating path \"{}\" to save the runner".format(self.agent_path))

            act_space_path = os.path.join(
                self.agent_path, EpisodeData.ACTION_SPACE)
            obs_space_path = os.path.join(
                self.agent_path, EpisodeData.OBS_SPACE)
            env_modif_space_path = os.path.join(
                self.agent_path, EpisodeData.ENV_MODIF_SPACE)

            if not os.path.exists(act_space_path):
                dict_action_space = action_space.to_dict()
                with open(act_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_action_space, fp=f,
                              indent=4, sort_keys=True)
            if not os.path.exists(obs_space_path):
                dict_observation_space = observation_space.to_dict()
                with open(obs_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_observation_space,
                              fp=f, indent=4, sort_keys=True)
            if not os.path.exists(env_modif_space_path):
                dict_helper_action_env = helper_action_env.to_dict()
                with open(env_modif_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_helper_action_env, fp=f,
                              indent=4, sort_keys=True)

            if not os.path.exists(self.episode_path):
                os.mkdir(self.episode_path)
                logger.info(
                    "Creating path \"{}\" to save the episode {}".format(self.episode_path, self.name))

    def __len__(self):
        return self.meta["chronics_max_timestep"]

    @classmethod
    def from_disk(cls, agent_path, name=str(1)):

        if agent_path is None:
            # TODO: proper exception
            raise Grid2OpException("A path to an episode should be provided")

        episode_path = os.path.abspath(os.path.join(agent_path, name))

        try:
            with open(os.path.join(episode_path, EpisodeData.PARAMS)) as f:
                _parameters = json.load(fp=f)
            with open(os.path.join(episode_path, EpisodeData.META)) as f:
                episode_meta = json.load(fp=f)
            with open(os.path.join(episode_path, EpisodeData.TIMES)) as f:
                episode_times = json.load(fp=f)

            times = np.load(os.path.join(
                episode_path, EpisodeData.AG_EXEC_TIMES))
            actions = np.load(os.path.join(episode_path, EpisodeData.ACTIONS))
            env_actions = np.load(os.path.join(
                episode_path, EpisodeData.ENV_ACTIONS))
            observations = np.load(os.path.join(
                episode_path, EpisodeData.OBSERVATIONS))
            disc_lines = np.load(os.path.join(
                episode_path, EpisodeData.LINES_FAILURES))
            rewards = np.load(os.path.join(episode_path, EpisodeData.REWARDS))
        except FileNotFoundError as ex:
            raise Grid2OpException(f"EpisodeData file not found \n {str(ex)}")

        observation_space = ObservationSpace.from_dict(
            os.path.join(agent_path, EpisodeData.OBS_SPACE))
        action_space = ActionSpace.from_dict(
            os.path.join(agent_path, EpisodeData.ACTION_SPACE))
        helper_action_env = ActionSpace.from_dict(
            os.path.join(agent_path, EpisodeData.ENV_MODIF_SPACE))

        return cls(actions, env_actions, observations, rewards, disc_lines,
                   times, _parameters, episode_meta, episode_times,
                   observation_space, action_space, helper_action_env,
                   agent_path, name=name, get_dataframes=True)

    def set_parameters(self, env):

        if self.serialize:
            self.parameters = env.parameters.to_dict()

    def set_meta(self, env, time_step, cum_reward):
        if self.serialize:
            self.meta = {}
            self.meta["chronics_path"] = "{}".format(
                env.chronics_handler.get_id())
            self.meta["chronics_max_timestep"] = "{}".format(
                env.chronics_handler.max_timestep())
            self.meta["grid_path"] = "{}".format(env.init_grid_path)
            self.meta["backend_type"] = "{}".format(
                type(env.backend).__name__)
            self.meta["env_type"] = "{}".format(type(env).__name__)
            self.meta["nb_timestep_played"] = time_step
            self.meta["cumulative_reward"] = cum_reward

    def incr_store(self, efficient_storing, time_step, time_step_duration,
                   reward, env_act, act, obs, info):

        if self.serialize:
            self.actions.update(time_step, act.to_vect(), efficient_storing)
            self.env_actions.update(
                time_step, env_act.to_vect(), efficient_storing)
            self.observations.update(
                time_step + 1, obs.to_vect(), efficient_storing)
            if efficient_storing:
                # efficient way of writing
                self.times[time_step - 1] = time_step_duration
                self.rewards[time_step - 1] = reward
                if "disc_lines" in info:
                    arr = info["disc_lines"]
                    if arr is not None:
                        self.disc_lines[time_step - 1, :] = arr
                    else:
                        self.disc_lines[time_step - 1,
                                        :] = self.disc_lines_templ
            else:
                # completely inefficient way of writing
                self.times = np.concatenate(
                    (self.times, (time_step_duration,)))
                self.rewards = np.concatenate((self.rewards, (reward,)))
                if "disc_lines" in info:
                    arr = info["disc_lines"]
                    if arr is not None:
                        self.disc_lines = np.concatenate(
                            (self.disc_lines, arr))
                    else:
                        self.disc_lines = np.concatenate(
                            (self.disc_lines, self.disc_lines_templ))

    def set_episode_times(self, env, time_act, beg_, end_):
        if self.serialize:
            self.episode_times = {}
            self.episode_times["Env"] = {}
            self.episode_times["Env"]["total"] = float(
                env._time_apply_act + env._time_powerflow + env._time_extract_obs)
            self.episode_times["Env"]["apply_act"] = float(env._time_apply_act)
            self.episode_times["Env"]["powerflow_computation"] = float(
                env._time_powerflow)
            self.episode_times["Env"]["observation_computation"] = float(
                env._time_extract_obs)
            self.episode_times["Agent"] = {}
            self.episode_times["Agent"]["total"] = float(time_act)
            self.episode_times["total"] = float(end_ - beg_)

    def to_disk(self):
        if self.serialize:
            parameters_path = os.path.join(
                self.episode_path, EpisodeData.PARAMS)
            with open(parameters_path, "w") as f:
                json.dump(obj=self.parameters, fp=f, indent=4, sort_keys=True)

            meta_path = os.path.join(self.episode_path, EpisodeData.META)
            with open(meta_path, "w") as f:
                json.dump(obj=self.meta, fp=f, indent=4, sort_keys=True)

            episode_times_path = os.path.join(
                self.episode_path, EpisodeData.TIMES)
            with open(episode_times_path, "w") as f:
                json.dump(obj=self.episode_times, fp=f,
                          indent=4, sort_keys=True)

            np.save(os.path.join(self.episode_path, EpisodeData.AG_EXEC_TIMES),
                    self.times)
            self.actions.save(
                os.path.join(self.episode_path, EpisodeData.ACTIONS))
            self.env_actions.save(
                os.path.join(self.episode_path, EpisodeData.ENV_ACTIONS))
            self.observations.save(
                os.path.join(self.episode_path, EpisodeData.OBSERVATIONS))
            np.save(os.path.join(
                self.episode_path, EpisodeData.LINES_FAILURES), self.disc_lines)
            np.save(os.path.join(self.episode_path,
                                 EpisodeData.REWARDS), self.rewards)


class CollectionWrapper:
    """
    A wrapping class to add some behaviors (iterability, item access, update, save)
    to grid2op object collections (Action and Observation classes essentially).

    Attributes
    ----------
    collection: ``type``
        The collection to wrap.
    helper:
        The helper object used to access elements of the collection through a 
        `from_vect` method.
    collection_name: ``str``
        The name of the collection.
    elem_name: ``str``
        The name of one element of the collection.
    i: ``int``
        Integer used for iteration.
    _game_over: ``int``
        The time step at which the game_over occurs. None if there is no game_over
    objects:
        The collection of objects built with the `from_vect` method

    Methods
    -------
    update(time_step, values, efficient_storage)
        update the collection with new `values` for a given `time_step`.
    save(path)
        save the collection to disk using `path` as the path to the file to write in.
    Raises
    ------
    Grid2OpException
        If the helper function has no from_vect method.
        If trying to access an element outside of the collection
    """

    def __init__(self, collection, helper, collection_name):
        self.collection = collection
        if not hasattr(helper, "from_vect"):
            raise Grid2OpException(f"Object {helper} must implement a "
                                   f"from_vect methode.")
        self.helper = helper
        self.collection_name = collection_name
        self.elem_name = self.collection_name[:-1]
        self.i = 0
        self._game_over = None
        self.objects = []
        for i, elem in enumerate(self.collection):
            try:
                self.objects.append(
                    self.helper.from_vect(self.collection[i, :]))
            except AmbiguousAction:
                self._game_over = i
                break

    def __len__(self):
        if self._game_over is None:
            return self.collection.shape[0]
        else:
            return self._game_over

    def __getitem__(self, i):
        if isinstance(i, slice) or i < len(self):
            return self.objects[i]
        else:
            raise Grid2OpException(
                f"Trying to reach {self.elem_name} {i + 1} but "
                f"there are only {len(self)} {self.collection_name}.")

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i = self.i + 1
        if self.i < len(self) + 1:
            return self.objects[self.i - 1]
        else:
            raise StopIteration

    def update(self, time_step, values, efficient_storage):
        if efficient_storage:
            self.collection[time_step - 1, :] = values
        else:
            self.collection = np.concatenate((self.collection, values))

    def save(self, path):
        np.save(path, self.collection)
