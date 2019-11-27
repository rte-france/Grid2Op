import json
import os

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Utils import ActionSpace, ObservationSpace


class Episode(object):
    def __init__(self, actions, observations, rewards, disc_lines,
                 exec_times, params, meta, times, observation_space,
                 action_space):

        self.actions = actions
        self.observations = observations
        self.rewards = rewards
        self.disc_lines = disc_lines
        self.exec_times = exec_times
        self.params = params
        self.meta = meta
        self.times = times
        self.observation_space = observation_space
        self.action_space = action_space

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

            exec_times = np.load(os.path.join(
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
                   exec_times, _parameters, episode_meta, episode_times,
                   observation_space, action_space)

    def todisk(self):
        pass
