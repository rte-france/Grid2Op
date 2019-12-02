import json
import os

import numpy as np

try:
    from .Exceptions import Grid2OpException
    from .Utils import ActionSpace, ObservationSpace
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException
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
                 logger=None, indx=0):

        self.actions = CollectionWrapper(actions, action_space, "actions")
        self.observations = CollectionWrapper(observations, observation_space,
                                              "observations")

        self.env_actions = CollectionWrapper(env_actions, helper_action_env,
                                             "env_actions")
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
                   observation_space, action_space,
                   helper_action_env,
                   agent_path, indx=indx)

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
            if efficient_storing:
                # efficient way of writing
                self.times[time_step-1] = time_step_duration
                self.rewards[time_step-1] = reward
                self.actions[time_step-1, :] = act.to_vect()
                self.env_actions[time_step-1, :] = env_act.to_vect()
                self.observations[time_step-1, :] = obs.to_vect()
                if "disc_lines" in info:
                    arr = info["disc_lines"]
                    if arr is not None:
                        self.disc_lines[time_step-1, :] = arr
                    else:
                        self.disc_lines[time_step - 1,
                                        :] = self.disc_lines_templ
            else:
                # completely inefficient way of writing
                self.times = np.concatenate(
                    (self.times, (time_step_duration, )))
                self.rewards = np.concatenate((self.rewards, (reward, )))
                self.actions = np.concatenate((self.actions, act.to_vect()))
                self.env_actions = np.concatenate(
                    (self.actions, env_act.to_vect()))
                self.observations = np.concatenate(
                    (self.observations, obs.to_vect()))
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
                env._time_apply_act+env._time_powerflow+env._time_extract_obs)
            self.episode_times["Env"]["apply_act"] = float(env._time_apply_act)
            self.episode_times["Env"]["powerflow_computation"] = float(
                env._time_powerflow)
            self.episode_times["Env"]["observation_computation"] = float(
                env._time_extract_obs)
            self.episode_times["Agent"] = {}
            self.episode_times["Agent"]["total"] = float(time_act)
            self.episode_times["total"] = float(end_-beg_)

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
            np.save(os.path.join(self.episode_path, Episode.ACTIONS),
                    self.actions)
            np.save(os.path.join(self.episode_path, Episode.ENV_ACTIONS),
                    self.env_actions)
            np.save(os.path.join(self.episode_path, Episode.OBSERVATIONS),
                    self.observations)
            np.save(os.path.join(
                self.episode_path, Episode.LINES_FAILURES), self.disc_lines)
            np.save(os.path.join(self.episode_path,
                                 Episode.REWARDS), self.rewards)


class CollectionWrapper:
    def __init__(self, collection, helper, collection_name):
        self.collection = collection
        if not hasattr(helper, "from_vect"):
            raise Grid2OpException(f"Object {helper} must implement a "
                                   f"from_vect methode.")
        self.helper = helper
        self.collection_name = collection_name
        self.elem_name = self.collection_name[:-1]
        self.i = 0

    def __len__(self):
        return self.collection.shape[0]

    def __getitem__(self, i):
        if i < len(self):
            return self.helper.from_vect(self.collection[i, :])
        else:
            raise Grid2OpException(
                f"Trying to reach {self.elem_name} {i+1} but "
                f"there are only {len(self)} {self.collection_name}.")

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i = self.i + 1
        if self.i < len(self) + 1:
            return self[self.i-1]
        else:
            raise StopIteration
