# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import os

import numpy as np

from grid2op.Exceptions import Grid2OpException, EnvError
from grid2op.Action import ActionSpace
from grid2op.Observation import ObservationSpace

# TODO refacto the "save / load" logic. For now save is in the CollectionWrapper and load in the EpisodeData


class EpisodeData:
    """
    .. warning:: The attributes of this class are not up to date.
        TODO be consistent with the real behaviour now.

    This module provides a way to serialize on disk et deserialize one run episode along with some
    methods and utilities to ease its manipulation.

    If enabled when usign the :class:`Runner`, the :class:`EpisodeData`
    will save the information in a structured way. For each episode there will be a folder with:

      - "episode_meta.json" that represents some meta information about:

        - "agent_seed": the seed used to seed the agent (if any)
        - "backend_type": the name of the :class:`grid2op.Backend.Backend` class used
        - "chronics_max_timestep": the **maximum** number of timestep for the chronics used
        - "chronics_path": the path where the time dependant data (chronics) are located
        - "cumulative_reward": the cumulative reward over all the episode
        - "env_seed": the seed used to seed the environment (if any)
        - "env_type": the name of the :class:`grid2op.Environment` class used.
        - "grid_path": the path where the powergrid has been loaded from
        - "nb_timestep_played": number of time step the agent has succesfully managed

      - "episode_times.json": gives some information about the total time spend in multiple part of the runner, mainly the
        :class:`grid2op.Agent.BaseAgent` (and especially its method :func:`grid2op.BaseAgent.act`) and amount of time
        spent in the :class:`grid2op.Environment.Environment`
      - "_parameters.json": is a representation as json of a the :class:`grid2op.Parameters.Parameters` used for this episode
      - "rewards.npz" is a numpy 1d array giving the rewards at each time step. We adopted the convention that the stored
        reward at index `i` is the one observed by the agent at time `i` and **NOT** the reward sent by the
        :class:`grid2op.Environment` after the action has been implemented.
      - "exec_times.npy" is a numpy 1d array giving the execution time of each time step of the episode
      - "actions.npy" gives the actions that has been taken by the :class:`grid2op.BaseAgent.BaseAgent`. At row `i` of
        "actions.npy" is a
        vectorized representation of the action performed by the agent at timestep `i` *ie.* **after** having observed
        the observation present at row `i` of "observation.npy" and the reward showed in row `i` of "rewards.npy".
      - "disc_lines.npy" gives which lines have been disconnected during the simulation of the cascading failure at each
        time step. The same convention as for "rewards.npy" has been adopted. This means that the powerlines are
        disconnected when the :class:`grid2op.Agent.BaseAgent` takes the :class:`grid2op.BaseAction` at time step `i`.
      - "observations.npy" is a numpy 2d array representing the :class:`grid2op.BaseObservation.BaseObservation` at the
        disposal of the
        :class:`grid2op.Agent.BaseAgent` when he took his action.
      - "env_modifications.npy" is a 2d numpy array representing the modification of the powergrid from the environment.
        these modification usually concerns the hazards, maintenance, as well as modification of the generators production
        setpoint or the loads consumption.

    All of the above should allow to read back, and better understand the behaviour of some
    :class:`grid2op.Agent.BaseAgent`, even though such utility functions have not been coded yet.

    Attributes
    ----------
    actions: ``type``
        Stores the Agent actions as a collection of :class:`grid2op.BaseAction`.
        The collection is stored the utility class :class:`grid2op.Episode.CollectionWrapper`.
    observations: ``type``
        Stores the Observations as a collection of :class:`grid2op.BaseObservation`.
        The collection is stored the utility class :class:`grid2op.Episode.CollectionWrapper`.
    env_actions: ``type``
        Stores the Environment actions as a collection of :class:`grid2op.BaseAction`.
        The collection is stored the utility class :class:`grid2op.Episode.CollectionWrapper`.
    attacks: ``type``
        Stores the Opponent actions as a collection of :class:`grid2op.BaseAction`.
        The collection is stored the utility class :class:`grid2op.Episode.CollectionWrapper`.

    Examples
    --------
    Here is an example on how to save the action your agent was doing by the :class:`grid2op.Runner.Runner` of grid2op.

    .. code-block:: python

        import grid2op
        from grid2op.Runner import Runner

        # I create an environment
        env = grid2op.make("rte_case5_example", test=True)

        # I create the runner
        runner = Runner(**env.get_params_for_runner())

        # I start the runner and save the results in "/I/SAVED/RESULTS/THERE"
        # I start the evaluation on 2 different episode
        res = runner.run(path_save="/I/SAVED/RESULTS/THERE", nb_episode=2)

    And now i can reload the data easily with the EpisodeData class:

    .. code-block:: python

        import grid2op
        from grid2op.Episode import EpisodeData
        # I study only the first episode saved, because... why not
        li_episode = EpisodeData.list_episode(path_save)
        full_path, episode_studied = li_episode[0]
        this_episode = EpisodeData.from_disk(path_agent, episode_studied)

        # now the episode is loaded, and you can easily iterate through the observation, the actions etc.
        for act in this_episode.actions:
            print(act)

        for i, obs in enumerate(this_episode.observations):
            print("At step {} the active productions were {}".format(i, obs.prod_p))

    """
    ACTION_SPACE = "dict_action_space.json"
    OBS_SPACE = "dict_observation_space.json"
    ENV_MODIF_SPACE = "dict_env_modification_space.json"
    ATTACK_SPACE = "dict_attack_space.json"  # action space of the attack (this is NOT the OpponentSpace) this is the "opponent action space"

    PARAMS = "_parameters.json"
    META = "episode_meta.json"
    TIMES = "episode_times.json"
    OTHER_REWARDS = "other_rewards.json"
    AG_EXEC_TIMES = "agent_exec_times.npz"
    ACTIONS = "actions.npz"
    ENV_ACTIONS = "env_modifications.npz"
    OBSERVATIONS = "observations.npz"
    LINES_FAILURES = "disc_lines_cascading_failure.npz"
    ATTACK = "opponent_attack.npz"
    REWARDS = "rewards.npz"

    ATTR_EPISODE = [PARAMS, META, TIMES, OTHER_REWARDS, AG_EXEC_TIMES, ACTIONS,
                    ENV_ACTIONS, OBSERVATIONS, LINES_FAILURES, ATTACK, REWARDS]

    def __init__(self,
                 actions=None,
                 env_actions=None,
                 observations=None,
                 rewards=None,
                 disc_lines=None,
                 times=None,
                 params=None,
                 meta=None,
                 episode_times=None,
                 observation_space=None,
                 action_space=None,
                 helper_action_env=None,
                 attack_space=None,
                 path_save=None,
                 disc_lines_templ=None,
                 attack_templ=None,
                 attack=None,
                 logger=None,
                 name="EpisodeData",
                 get_dataframes=None,
                 other_rewards=[]):
        self.parameters = None

        self.actions = CollectionWrapper(actions,
                                         action_space,
                                         "actions",
                                         check_legit=False)
        self.observations = CollectionWrapper(observations,
                                              observation_space,
                                              "observations")
        self.env_actions = CollectionWrapper(env_actions,
                                             helper_action_env,
                                             "env_actions",
                                             check_legit=False)

        self.attacks = CollectionWrapper(attack,
                                         attack_space,
                                         "attacks")

        self.meta = meta
        # gives a unique game over for everyone
        # TODO this needs testing!
        action_go = self.actions._game_over
        obs_go = self.observations._game_over
        env_go = self.env_actions._game_over
        real_go = action_go
        if self.meta is not None:
            # when initialized by the runner, meta is None
            if "nb_timestep_played" in self.meta:
                real_go = int(self.meta["nb_timestep_played"])
        if real_go is None:
            real_go = action_go
        else:
            if action_go is not None:
                real_go = min(action_go, real_go)
        if real_go is None:
            real_go = obs_go
        else:
            if obs_go is not None:
                real_go = min(obs_go, real_go)
        if real_go is None:
            real_go = env_go
        else:
            if env_go is not None:
                real_go = min(env_go, real_go)
        if real_go is not None:
            # there is a real game over, i assign the proper value for each collection
            self.actions._game_over = real_go
            self.observations._game_over = real_go + 1
            self.env_actions._game_over = real_go

        self.other_rewards = other_rewards
        self.observation_space = observation_space
        self.rewards = rewards
        self.disc_lines = disc_lines
        self.times = times
        self.params = params
        self.episode_times = episode_times
        self.name = name
        self.disc_lines_templ = disc_lines_templ

        self.attack_templ = attack_templ

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
                try:
                    os.mkdir(self.agent_path)
                    self.logger.info(
                        "Creating path \"{}\" to save the runner".format(self.agent_path))
                except FileExistsError:
                    pass

            act_space_path = os.path.join(
                self.agent_path, EpisodeData.ACTION_SPACE)
            obs_space_path = os.path.join(
                self.agent_path, EpisodeData.OBS_SPACE)
            env_modif_space_path = os.path.join(
                self.agent_path, EpisodeData.ENV_MODIF_SPACE)
            attack_space_path = os.path.join(
                self.agent_path, EpisodeData.ATTACK_SPACE)

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
            if not os.path.exists(attack_space_path):
                dict_attack_space = attack_space.to_dict()
                with open(attack_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_attack_space, fp=f,
                              indent=4, sort_keys=True)

            if not os.path.exists(self.episode_path):
                os.mkdir(self.episode_path)
                logger.info(
                    "Creating path \"{}\" to save the episode {}".format(self.episode_path, self.name))

    @staticmethod
    def list_episode(path_agent):
        """
        From a given path where a runner is supposed to have run, it extracts the subdirectories that can
        store values from an episode.

        Parameters
        ----------
        path_agent: ``str``
            The path where to look for data coming from "episode"

        Returns
        -------
        res: ``list``
            A list of possible episodes. Each element of this list is a tuple: (full_path, episode_name)

        Examples
        --------

        .. code-block:: python

            import grid2op
            import os
            import numpy as np
            from grid2op.Runner import Runner
            from grid2op.Episode import EpisodeData

            ################
            # INTRO
            # create a runner
            env = grid2op.make()
            # see the documentation of the Runner if you want to change the agent.
            # in this case it will be "do nothing"
            runner = Runner(**env.get_params_for_runner())

            # execute it a given number of chronics
            nb_episode = 2
            path_save = "i_saved_the_runner_here"
            res = runner.run(nb_episode=nb_episode, path_save=path_save)

            # END INTRO
            ##################

            li_episode = EpisodeData.list_episode(path_save)
            # and now you can iterate through it:
            for full_episode_path, episode_name in li_episode:
                this_episode = EpisodeData.from_disk(path_agent, episode_name)
                # you can do something with it now

        """
        res = []
        li_subfiles = list(os.listdir(path_agent))
        for el in sorted(li_subfiles):
            # loop through the files that stores the agent's logs
            this_dir = os.path.join(path_agent, el)
            if not os.path.isdir(this_dir):
                # it cannot be the result of an episode if it is not a directory.
                continue
            ok_ = True
            for file_that_should_be in EpisodeData.ATTR_EPISODE:
                if not os.path.exists(os.path.join(this_dir, file_that_should_be)):
                    # one file is missing
                    ok_ = False
                    break
            if ok_:
                res.append((os.path.abspath(path_agent), el))
        return res

    def reboot(self):
        """
        Do as if the data just got read from the hard drive (loop again from the
        initial observation and action)
        """
        self.actions.reboot()
        self.observations.reboot()
        self.env_actions.reboot()

    def go_to(self, index):
        self.actions.go_to(index)
        self.observations.go_to(index+1)
        self.env_actions.go_to(index)

    def get_actions(self):
        return self.actions.collection

    def get_observations(self):
        return self.observations.collection

    def __len__(self):
        return int(self.meta["chronics_max_timestep"])

    @classmethod
    def from_disk(cls, agent_path, name="1"):
        """
        This function allows you to reload an episode stored using the runner.

        See the example at the definition of the class for more information on how to use it.

        Parameters
        ----------
        agent_path: ``str``
            Path pass at the "runner.run" method

        name: ``str``
            The name of the episode you want to reload.

        Returns
        -------
        res:
            The data loaded properly in memory.
        """
        if agent_path is None:
            raise Grid2OpException("A path to an episode should be provided, please call \"from_disk\" with "
                                   "\"agent_path other\" than None")
        episode_path = os.path.abspath(os.path.join(agent_path, name))

        try:
            with open(os.path.join(episode_path, EpisodeData.PARAMS)) as f:
                _parameters = json.load(fp=f)
            with open(os.path.join(episode_path, EpisodeData.META)) as f:
                episode_meta = json.load(fp=f)
            with open(os.path.join(episode_path, EpisodeData.TIMES)) as f:
                episode_times = json.load(fp=f)
            with open(os.path.join(episode_path, EpisodeData.OTHER_REWARDS)) as f:
                other_rewards = json.load(fp=f)

            times = np.load(os.path.join(
                episode_path, EpisodeData.AG_EXEC_TIMES))["data"]
            actions = np.load(os.path.join(episode_path, EpisodeData.ACTIONS))["data"]
            env_actions = np.load(os.path.join(
                episode_path, EpisodeData.ENV_ACTIONS))["data"]
            observations = np.load(os.path.join(
                episode_path, EpisodeData.OBSERVATIONS))["data"]
            disc_lines = np.load(os.path.join(
                episode_path, EpisodeData.LINES_FAILURES))["data"]
            attack = np.load(os.path.join(
                episode_path, EpisodeData.ATTACK))["data"]
            rewards = np.load(os.path.join(episode_path, EpisodeData.REWARDS))["data"]

        except FileNotFoundError as ex:
            raise Grid2OpException(f"EpisodeData file not found \n {str(ex)}")

        observation_space = ObservationSpace.from_dict(
            os.path.join(agent_path, EpisodeData.OBS_SPACE))
        action_space = ActionSpace.from_dict(
            os.path.join(agent_path, EpisodeData.ACTION_SPACE))
        helper_action_env = ActionSpace.from_dict(
            os.path.join(agent_path, EpisodeData.ENV_MODIF_SPACE))
        attack_space = ActionSpace.from_dict(
            os.path.join(agent_path, EpisodeData.ATTACK_SPACE))
        return cls(actions=actions,
                   env_actions=env_actions,
                   observations=observations,
                   rewards=rewards,
                   disc_lines=disc_lines,
                   times=times,
                   params=_parameters,
                   meta=episode_meta,
                   episode_times=episode_times,
                   observation_space=observation_space,
                   action_space=action_space,
                   helper_action_env=helper_action_env,
                   path_save=None,  # No save when reading
                   attack=attack,
                   attack_space=attack_space,
                   name=name,
                   get_dataframes=True,
                   other_rewards=other_rewards)

    def set_parameters(self, env):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by the Runner to serialize properly an episode

        TODO

        Parameters
        ----------
        env

        Returns
        -------

        """
        if self.serialize:
            self.parameters = env.parameters.to_dict()

    def set_meta(self, env, time_step, cum_reward, env_seed, agent_seed):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by he runner to serialize properly an episode

        TODO

        Parameters
        ----------
        env
        time_step
        cum_reward
        env_seed
        agent_seed

        Returns
        -------

        """
        if self.serialize:
            self.meta = {}
            self.meta["chronics_path"] = "{}".format(
                env.chronics_handler.get_id())
            self.meta["chronics_max_timestep"] = "{}".format(
                env.chronics_handler.max_timestep())
            self.meta["grid_path"] = "{}".format(env._init_grid_path)
            self.meta["backend_type"] = "{}".format(
                type(env.backend).__name__)
            self.meta["env_type"] = "{}".format(type(env).__name__)
            self.meta["nb_timestep_played"] = time_step
            self.meta["cumulative_reward"] = cum_reward
            if env_seed is None:
                self.meta["env_seed"] = env_seed
            else:
                self.meta["env_seed"] = int(env_seed)
            if agent_seed is None:
                self.meta["agent_seed"] = agent_seed
            else:
                self.meta["agent_seed"] = int(agent_seed)

    def incr_store(self, efficient_storing, time_step, time_step_duration,
                   reward, env_act, act, obs, opp_attack, info):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by he runner to serialize properly an episode

        TODO

        Parameters
        ----------
        efficient_storing
        time_step
        time_step_duration
        reward
        env_act
        act
        obs
        opp_attack
        info

        Returns
        -------

        """

        if self.serialize:
            self.actions.update(time_step, act.to_vect(), efficient_storing)
            self.env_actions.update(
                time_step, env_act.to_vect(), efficient_storing)
            self.observations.update(
                time_step + 1, obs.to_vect(), efficient_storing)
            if opp_attack is not None:
                self.attacks.update(
                    time_step, opp_attack.to_vect(), efficient_storing)
            else:
                if efficient_storing:
                    self.attacks.collection[time_step - 1, :] = 0.
                else:
                    self.attack = np.concatenate(
                        (self.attack, self.attack_templ))

            if efficient_storing:
                # efficient way of writing
                self.times[time_step - 1] = time_step_duration
                self.rewards[time_step - 1] = reward
                if "disc_lines" in info:
                    arr = info["disc_lines"]
                    if arr is not None:
                        self.disc_lines[time_step - 1, :] = arr
                    else:
                        self.disc_lines[time_step - 1, :] = self.disc_lines_templ
            else:
                # completely inefficient way of writing
                self.times = np.concatenate(
                    (self.times, (time_step_duration,)))
                self.rewards = np.concatenate((self.rewards, (reward,)))
                if "disc_lines" in info:
                    arr = info["disc_lines"]
                    if arr is not None:
                        self.disc_lines = np.concatenate(
                            (self.disc_lines, arr.reshape(1, -1)))
                    else:
                        self.disc_lines = np.concatenate(
                            (self.disc_lines, self.disc_lines_templ))

            if "rewards" in info:
                self.other_rewards.append({k: self._convert_to_float(v) for k, v in info["rewards"].items()})
            # TODO add is_illegal and is_ambiguous flags!

    def _convert_to_float(self, el):
        try:
            res = float(el)
        except Exception as e:
            res = -float('inf')
        return res

    def set_episode_times(self, env, time_act, beg_, end_):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by he runner to serialize properly an episode

         TODO

        Parameters
        ----------
        env
        time_act
        beg_
        end_

        Returns
        -------

        """
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
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used by he runner to serialize properly an episode

         TODO

        Returns
        -------

        """
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

            episode_other_rewards_path = os.path.join(
                self.episode_path, EpisodeData.OTHER_REWARDS)
            with open(episode_other_rewards_path, "w") as f:
                json.dump(obj=self.other_rewards, fp=f,
                          indent=4, sort_keys=True)

            np.savez_compressed(os.path.join(self.episode_path, EpisodeData.AG_EXEC_TIMES),
                    data=self.times)
            self.actions.save(
                os.path.join(self.episode_path, EpisodeData.ACTIONS))
            self.env_actions.save(
                os.path.join(self.episode_path, EpisodeData.ENV_ACTIONS))
            self.observations.save(
                os.path.join(self.episode_path, EpisodeData.OBSERVATIONS))
            self.attacks.save(
                os.path.join(os.path.join(self.episode_path,
                                          EpisodeData.ATTACK)))
            np.savez_compressed(os.path.join(
                self.episode_path, EpisodeData.LINES_FAILURES), data=self.disc_lines)
            np.savez_compressed(os.path.join(self.episode_path,
                                 EpisodeData.REWARDS), data=self.rewards)


class CollectionWrapper:
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        Utility to make the interaction with stored actions and stored observations more pythonic

    A wrapping class to add some behaviors (iterability, item access, update, save)
    to grid2op object collections (:class:`grid2op.Action.BaseAction` and :class:`grid2op.Observation.BaseObservation`
    classes essentially).

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
    :class:`grid2op.Exceptions.Grid2OpException`
        If the helper function has no from_vect method.
        If trying to access an element outside of the collection

    """

    def __init__(self, collection, helper, collection_name, check_legit=True):
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
                collection_obj = self.helper.from_vect(self.collection[i, :], check_legit=check_legit)
                self.objects.append(collection_obj)
            except EnvError as exc_:
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
            self.collection = np.concatenate((self.collection, values.reshape(1, -1)))

    def save(self, path):
        np.savez_compressed(path, data=self.collection)  # do not change keyword arguments

    def reboot(self):
        self.i = 0

    def go_to(self, index):
        if index >= len(self):
            raise Grid2OpException("index too long for collection {}".format(self.collection_name))
        self.i = index


if __name__ == "__main__":
    pass
