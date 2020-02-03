"""
This module provides a way to serialize on disk et deserialize one run episode along with some 
methods and utilities to ease its manipulation.

If enabled when usign the :class:`Runner`, the :class:`Episode` will save the information in a structured way. For each episode there will be a folder
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
from tqdm import tqdm

try:
    from .Exceptions import Grid2OpException, AmbiguousAction
    from .Utils import ActionSpace, ObservationSpace
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException, AmbiguousAction
    from Utils import ActionSpace, ObservationSpace


class Episode:
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
                 logger=None, indx=0, get_dataframes=None):

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
        self.indx = indx
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

        if get_dataframes is not None:
            print("computing df")
            beg = time.time()
            print("Environment")
            self.load, self.production, self.rho, self.action_data, self.action_data_table, self.computed_reward, self.flow_and_voltage_line = self._make_df_from_data()
            print("Hazards-Maintenances")
            self.hazards, self.maintenances = self._env_actions_as_df()
            self.timestamps = sorted(self.load.timestamp.dropna().unique())
            self.timesteps = sorted(self.load.timestep.unique())
            end = time.time()
            print(f"end computing df: {end - beg}")
        if path_save is not None:
            self.agent_path = os.path.abspath(path_save)
            self.episode_path = os.path.join(self.agent_path, str(indx))
            self.serialize = True
            if not os.path.exists(self.agent_path):
                os.mkdir(self.agent_path)
                self.logger.info(
                    "Creating path \"{}\" to save the runner".format(self.agent_path))

            act_space_path = os.path.join(
                self.agent_path, Episode.ACTION_SPACE)
            obs_space_path = os.path.join(
                self.agent_path, Episode.OBS_SPACE)
            env_modif_space_path = os.path.join(
                self.agent_path, Episode.ENV_MODIF_SPACE)

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
                    "Creating path \"{}\" to save the episode {}".format(self.episode_path, self.indx))

    def _make_df_from_data(self):
        load_size = (len(self.observations)-1) * len(self.observations[0].load_p)
        prod_size = (len(self.observations)-1) * len(self.observations[0].prod_p)
        rho_size = (len(self.observations)-1) * len(self.observations[0].rho)
        action_size = len(self.actions)
        reward_size = len(self.rewards)
        flow_voltage_size = (len(self.observations)-1)
        cols = ["timestep", "timestamp", "equipement_id", "equipment_name",
                "value"]
        load_data = pd.DataFrame(index=range(load_size), columns=cols)
        production = pd.DataFrame(index=range(prod_size), columns=cols)
        rho = pd.DataFrame(index=range(rho_size), columns=[
            'time', "timestamp", 'equipment', 'value'])
        action_data = pd.DataFrame(index=range(action_size),
                                   columns=['timestep', 'timestamp', 'timestep_reward', 'action_line', 'action_subs',
                                            'set_line', 'switch_line', 'set_topo', 'change_bus', 'distance'])
        action_data_table = pd.DataFrame(index=range(action_size),
                                         columns=['timestep', 'timestamp', 'timestep_reward', 'action_line',
                                                  'action_subs',
                                                  'line_action', 'sub_name', 'objets_changed', 'distance'])
        computed_rewards = pd.DataFrame(index=range(reward_size), columns=[
            'timestep', 'rewards', 'cum_rewards'])
        cols = pd.MultiIndex.from_product([['or', 'ex'], ['active', 'reactive', 'current', 'voltage'], self.line_names])
        flow_voltage_line_table = pd.DataFrame(index=range(flow_voltage_size), columns=cols)
        topo_list = []
        bus_list = []
        for (time_step, (obs, act)) in tqdm(enumerate(zip(self.observations[1:], self.actions)),
                                            total=len(self.env_actions)):
            if obs is None:# obs.game_over:
                continue
            time_stamp = self.timestamp(obs)
            for equipment_id, load_p in enumerate(obs.load_p):
                pos = time_step * self.n_loads + equipment_id
                load_data.loc[pos, :] = [
                    time_step, time_stamp, equipment_id,
                    self.load_names[equipment_id], load_p]
            for equipment_id, prod_p in enumerate(obs.prod_p):
                pos = time_step * self.n_prods + equipment_id
                production.loc[pos, :] = [
                    time_step, time_stamp, equipment_id,
                    self.prod_names[equipment_id], prod_p]
            for equipment, rho_t in enumerate(obs.rho):
                pos = time_step * len(obs.rho) + equipment
                rho.loc[pos, :] = [time_step, time_stamp, equipment, rho_t]
            for line, subs in zip(range(act.n_line), range(len(act.sub_info))):
                pos = time_step
                action_data.loc[pos, :] = [time_step, time_stamp, self.rewards[time_step],
                                           np.sum(act._switch_line_status), np.sum(
                        act._change_bus_vect),
                                           act._set_line_status.flatten().astype(np.float),
                                           act._switch_line_status.flatten().astype(np.float),
                                           act._set_topo_vect.flatten().astype(np.float),
                                           act._change_bus_vect.flatten().astype(np.float),
                                           self.get_distance_from_obs(obs)]
                line_action = ""
                open_status = np.where(act._set_line_status == 1)
                close_status = np.where(act._set_line_status == -1)
                switch_line = np.where(act._switch_line_status == True)
                if len(open_status[0]) == 1:
                    line_action = "open " + str(self.line_names[open_status[0]])
                if len(close_status[0]) == 1:
                    line_action = "close " + str(self.line_names[close_status[0]])
                if len(switch_line[0]) == 1:
                    line_action = "switch " + str(self.line_names[switch_line[0]])
                sub_action = self.get_sub_action(act, obs)
                object_changed_set = self.get_object_changed(act._set_topo_vect, topo_list)
                if object_changed_set is not None:
                    object_changed = object_changed_set
                else:
                    object_changed = self.get_object_changed(act._change_bus_vect, bus_list)
                action_data_table.loc[pos, :] = [time_step, time_stamp, self.rewards[time_step],
                                                 np.sum(act._switch_line_status), np.sum(act._change_bus_vect),
                                                 line_action,
                                                 sub_action,
                                                 object_changed,
                                                 self.get_distance_from_obs(obs)]

            computed_rewards.loc[time_step, :] = [time_stamp,
                                                  self.rewards[time_step],
                                                  self.rewards.cumsum(axis=0)[time_step]]

            flow_voltage_line_table.loc[time_step, :] = np.array([
                obs.p_ex,
                obs.q_ex,
                obs.a_ex,
                obs.v_ex,
                obs.p_or,
                obs.q_or,
                obs.a_or,
                obs.v_or
            ]).flatten()


        load_data["value"] = load_data["value"].astype(float)
        production["value"] = production["value"].astype(float)
        rho["value"] = rho["value"].astype(float)
        return load_data, production, rho, action_data, action_data_table, computed_rewards, flow_voltage_line_table

    def get_object_changed(self, vect, list_topo):
        if np.count_nonzero(vect) is 0:
            return None
        for idx, topo_array in enumerate(list_topo):
            if not np.array_equal(topo_array, vect):
                return idx
        # if we havnt found the vect...
        list_topo.append(vect)
        return len(list_topo) - 1

    def get_sub_action(self, act, obs):
        for sub in range(len(obs.sub_info)):
            effect = act.effect_on(substation_id=sub)
            if np.any(effect["change_bus"] is True):
                return self.name_sub[sub]
            if np.any(effect["set_bus"] is 1) or np.any(effect["set_bus"] is -1):
                return self.name_sub[sub]
        return None

    def get_distance_from_obs(self, obs):
        return len(obs.topo_vect) - np.count_nonzero(obs.topo_vect == 1)

    def _env_actions_as_df(self):
        hazards_size = (len(self.observations)-1) * self.n_lines
        cols = ["timestep", "timestamp", "line_id", "line_name", "value"]
        hazards = pd.DataFrame(index=range(hazards_size), columns=cols)
        maintenances = hazards.copy()
        for (time_step, env_act) in tqdm(enumerate(self.env_actions), total=len(self.env_actions)):
            if env_act is None:
                continue
            time_stamp = self.timestamp(self.observations[time_step])
            iter_haz_maint = zip(env_act._hazards, env_act._maintenance)
            for line_id, (haz, maint) in enumerate(iter_haz_maint):
                pos = time_step * self.n_lines + line_id
                hazards.loc[pos, :] = [
                    time_step, time_stamp, line_id, self.line_names[line_id],
                    int(haz)
                ]
                maintenances.loc[pos, :] = [
                    time_step, time_stamp, line_id, self.line_names[line_id],
                    int(maint)
                ]
        hazards["value"] = hazards["value"].fillna(0).astype(int)
        maintenances["value"] = maintenances["value"].fillna(0).astype(int)
        return hazards, maintenances

    @staticmethod
    def timestamp(obs):
        return dt.datetime(obs.year, obs.month, obs.day, obs.hour_of_day,
                           obs.minute_of_hour)

    @classmethod
    def fromdisk(cls, agent_path, indx=0):

        if agent_path is None:
            # TODO: proper exception
            raise Grid2OpException("A path to an episode should be provided")

        episode_path = os.path.abspath(os.path.join(agent_path, str(indx)))

        try:
            with open(os.path.join(episode_path, Episode.PARAMS)) as f:
                _parameters = json.load(fp=f)
            with open(os.path.join(episode_path, Episode.META)) as f:
                episode_meta = json.load(fp=f)
            with open(os.path.join(episode_path, Episode.TIMES)) as f:
                episode_times = json.load(fp=f)

            times = np.load(os.path.join(
                episode_path, Episode.AG_EXEC_TIMES))
            actions = np.load(os.path.join(episode_path, Episode.ACTIONS))
            env_actions = np.load(os.path.join(
                episode_path, Episode.ENV_ACTIONS))
            observations = np.load(os.path.join(
                episode_path, Episode.OBSERVATIONS))
            disc_lines = np.load(os.path.join(
                episode_path, Episode.LINES_FAILURES))
            rewards = np.load(os.path.join(episode_path, Episode.REWARDS))
        except FileNotFoundError as ex:
            raise Grid2OpException(f"Episode file not found \n {str(ex)}")

        observation_space = ObservationSpace.from_dict(
            os.path.join(agent_path, Episode.OBS_SPACE))
        action_space = ActionSpace.from_dict(
            os.path.join(agent_path, Episode.ACTION_SPACE))
        helper_action_env = ActionSpace.from_dict(
            os.path.join(agent_path, Episode.ENV_MODIF_SPACE))

        return cls(actions, env_actions, observations, rewards, disc_lines,
                   times, _parameters, episode_meta, episode_times,
                   observation_space, action_space, helper_action_env,
                   agent_path, indx=indx, get_dataframes=True)

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

    def todisk(self):
        if self.serialize:
            parameters_path = os.path.join(
                self.episode_path, "_parameters.json")
            with open(parameters_path, "w") as f:
                json.dump(obj=self.parameters, fp=f, indent=4, sort_keys=True)

            meta_path = os.path.join(self.episode_path, "episode_meta.json")
            with open(meta_path, "w") as f:
                json.dump(obj=self.meta, fp=f, indent=4, sort_keys=True)

            episode_times_path = os.path.join(
                self.episode_path, "episode_times.json")
            with open(episode_times_path, "w") as f:
                json.dump(obj=self.episode_times, fp=f,
                          indent=4, sort_keys=True)

            np.save(os.path.join(self.episode_path, Episode.AG_EXEC_TIMES),
                    self.times)
            self.actions.save(
                os.path.join(self.episode_path, Episode.ACTIONS))
            self.env_actions.save(
                os.path.join(self.episode_path, Episode.ENV_ACTIONS))
            self.observations.save(
                os.path.join(self.episode_path, Episode.OBSERVATIONS))
            np.save(os.path.join(
                self.episode_path, Episode.LINES_FAILURES), self.disc_lines)
            np.save(os.path.join(self.episode_path,
                                 Episode.REWARDS), self.rewards)


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
    game_over: ``int``
        The time step at which the game_over occurs. -1 if there is no game_over
    
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

    def __len__(self):
        return self.collection.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            sliced_collection = CollectionWrapper(
                self.collection[i], self.helper, self.collection_name)
            return sliced_collection
        if i < len(self):
            return self.helper.from_vect(self.collection[i, :])
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
            if self.game_over == -1 or self.game_over > self.i:
                return self[self.i - 1]
        else:
            raise StopIteration

    @property
    def game_over(self):
        if self._game_over is None:
            try:
                for i in range(len(self)):
                    self.helper.from_vect(self.collection[i, :])
                    continue
                else:
                    self._game_over = -1
            except AmbiguousAction:
                self._game_over = i
        return self._game_over
                

    def update(self, time_step, values, efficient_storage):
        if efficient_storage:
            self.collection[time_step - 1, :] = values
        else:
            self.collection = np.concatenate((self.collection, values))

    def save(self, path):
        np.save(path, self.collection)
