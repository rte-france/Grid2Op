import json
import os

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner


class Episode(object):
    def __init__(self, runner, actions, observations, rewards, disc_lines,
                 exec_times, params, meta, times):

        self.runner = runner
        self.actions = actions
        self.observations = observations
        self.rewards = rewards
        self.disc_lines = disc_lines
        self.exec_times = exec_times
        self.params = params
        self.meta = meta
        self.times = times

    def get_action(self, i):
        return self.runner.env.action_space.from_vect(self.actions[i, :])

    def get_observation(self, i):
        return self.runner.env.observation_space.from_vect(self.observations[i, :])

    @classmethod
    def fromdisk(cls, path, indx=0, grid_path=None, chronics_path=None):

        if path is None:
            # TODO: proper exception
            raise Grid2OpException("A path to an episode should be provided")

        path = os.path.abspath(path)
        try:
            with open(os.path.join(path, "_parameters.json")) as f:
                _parameters = json.load(fp=f)
            with open(os.path.join(path, "episode_meta.json")) as f:
                episode_meta = json.load(fp=f)
            with open(os.path.join(path, "episode_times.json")) as f:
                episode_times = json.load(fp=f)

            exec_times = np.load(os.path.join(path, "agent_exec_times.npy"))
            actions = np.load(os.path.join(path, "actions.npy"))
            observations = np.load(os.path.join(path, "observations.npy"))
            disc_lines = np.load(os.path.join(
                path, "disc_lines_cascading_failure.npy"))
            rewards = np.load(os.path.join(path, "rewards.npy"))
        except FileNotFoundError as ex:
            raise Grid2OpException(f"Result file not found \n {str(ex)}")

        if chronics_path is None:
            chronics_path = episode_meta["chronics_path"]
        if grid_path is None:
            grid_path = episode_meta["grid_path"]

        runner = Runner(grid_path,
                        chronics_path,
                        parameters_path=os.path.join(path, "_parameters.json"))

        runner.init_env()

        return cls(runner, actions, observations, rewards, disc_lines,
                   exec_times, _parameters, episode_meta, episode_times)

    def todisk(self):
        pass
