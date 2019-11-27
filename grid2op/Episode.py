import json
import os

import numpy as np

try:
    from .Exceptions import Grid2OpException
    from .Utils import ActionSpace, ObservationSpace
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException
    from Utils import ActionSpace, ObservationSpace


class Episode(object):
    def __init__(self, actions=None, observations=None, rewards=None,
                 disc_lines=None, times=None, params=None, meta=None,
                 episode_times=None, observation_space=None, action_space=None,
                 path_save=None, disc_lines_templ=None, logger=None, indx=0):

        self.actions = actions
        self.observations = observations
        self.rewards = rewards
        self.disc_lines = disc_lines
        self.times = times
        self.params = params
        self.meta = meta
        self.episode_times = episode_times
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_path = os.path.abspath(path_save)
        self.indx = indx
        self.episode_path = os.path.join(self.agent_path, str(indx))
        self.disc_lines_templ = disc_lines_templ
        self.logger = logger
        self.serialize = False

        if self.agent_path is not None:
            self.serialize = True
            if not os.path.exists(self.agent_path):
                os.mkdir(self.agent_path)
                self.logger.info(
                    "Creating path \"{}\" to save the runner".format(self.agent_path))

            act_space_path = os.path.join(self.agent_path,
                                          "dict_action_space.json")
            obs_space_path = os.path.join(self.agent_path,
                                          "dict_observation_space.json")

            if not os.path.exists(act_space_path):
                dict_action_space = self.action_space.to_dict()
                with open(act_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_action_space, fp=f,
                              indent=4, sort_keys=True)
                dict_observation_space = self.observation_space.to_dict()
                with open(obs_space_path, "w", encoding='utf8') as f:
                    json.dump(obj=dict_observation_space,
                              fp=f, indent=4, sort_keys=True)

            if not os.path.exists(self.episode_path):
                os.mkdir(self.episode_path)
                logger.info(
                    "Creating path \"{}\" to save the episode {}".format(self.episode_path, self.indx))

    def get_action(self, i):
        return self.action_space.from_vect(self.actions[i, :])

    def get_observation(self, i):
        return self.observation_space.from_vect(self.observations[i, :])

    @classmethod
    def fromdisk(cls, agent_path, indx=0):

        if agent_path is None:
            # TODO: proper exception
            raise Grid2OpException("A path to an episode should be provided")

        episode_path = os.path.abspath(os.path.join(agent_path, str(indx)))

        try:
            with open(os.path.join(episode_path, "_parameters.json")) as f:
                _parameters = json.load(fp=f)
            with open(os.path.join(episode_path, "episode_meta.json")) as f:
                episode_meta = json.load(fp=f)
            with open(os.path.join(episode_path, "episode_times.json")) as f:
                episode_times = json.load(fp=f)

            times = np.load(os.path.join(
                episode_path, "agent_exec_times.npy"))
            actions = np.load(os.path.join(episode_path, "actions.npy"))
            observations = np.load(os.path.join(
                episode_path, "observations.npy"))
            disc_lines = np.load(os.path.join(
                episode_path, "disc_lines_cascading_failure.npy"))
            rewards = np.load(os.path.join(episode_path, "rewards.npy"))
        except FileNotFoundError as ex:
            raise Grid2OpException(f"Episode file not found \n {str(ex)}")

        observation_space = ObservationSpace.from_dict(
            os.path.join(agent_path, "dict_observation_space.json"))
        action_space = ActionSpace.from_dict(
            os.path.join(agent_path, "dict_action_space.json"))

        return cls(actions, observations, rewards, disc_lines,
                   times, _parameters, episode_meta, episode_times,
                   observation_space, action_space, agent_path, indx=indx)

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
                   reward, act, obs, info):
        if self.serialize:
            if efficient_storing:
                # efficient way of writing
                self.times[time_step-1] = time_step_duration
                self.rewards[time_step-1] = reward
                self.actions[time_step-1, :] = act.to_vect()
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

            np.save(os.path.join(self.episode_path, "agent_exec_times.npy"),
                    self.times)
            np.save(os.path.join(self.episode_path, "actions.npy"),
                    self.actions)
            np.save(os.path.join(self.episode_path, "observations.npy"),
                    self.observations)
            np.save(os.path.join(
                self.episode_path, "disc_lines_cascading_failure.npy"), self.disc_lines)
            np.save(os.path.join(self.episode_path, "rewards.npy"), self.rewards)
