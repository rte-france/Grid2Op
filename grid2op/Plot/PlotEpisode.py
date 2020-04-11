# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np
from datetime import datetime

from grid2op.EpisodeData import EpisodeData
from grid2op.Exceptions import Grid2OpException
from grid2op.Plot.PlotPyGame import PlotPyGame
from grid2op.Exceptions.PlotExceptions import PyGameQuit

try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    can_plot = True
except Exception as e:
    can_plot = False
    pass

try:
    from array2gif import write_gif
    can_save_gif = True
except:
    import warnings
    warnings.warn("The final gif will not be saved as \"array2gif\" package cannot be imported.")
    can_save_gif = False


class PlotEpisode(object):
    def __init__(self, agent_path):
        if not os.path.exists(agent_path):
            raise Grid2OpException("Nothing is found at \"{}\" where an agent path should have been.".format(agent_path))
        self.agent_path = agent_path
        self.episode_data = None

    def plot_episode(self, episode_id, max_fps=10, gif_name=None):
        path_ep = os.path.join(self.agent_path, episode_id)
        if not os.path.exists(path_ep):
            raise Grid2OpException("No episode is found at \"{}\" where the episode should have been.".format(path_ep))
        if gif_name is None:
            if not can_save_gif:
                raise Grid2OpException("Impossible to load the \"array2gif\" package. It is not possible to save a gif.")

        self.episode_data = EpisodeData.from_disk(agent_path=self.agent_path, name=episode_id)
        plot_runner = PlotPyGame(self.episode_data.observation_space,
                                 timestep_duration_seconds=1./max_fps)
        nb_timestep_played = int(self.episode_data.meta["nb_timestep_played"])
        all_obs = [el for el in self.episode_data.observations]
        all_reward = [el for el in self.episode_data.rewards]
        if gif_name is not None:
            total_array = np.zeros((nb_timestep_played, plot_runner.video_width, plot_runner.video_height, 3))
        for i, (obs, reward) in enumerate(zip(all_obs, all_reward)):
            timestamp = datetime(year=obs.year,
                                 month=obs.month,
                                 day=obs.day,
                                 hour=obs.hour_of_day,
                                 minute=obs.minute_of_hour)
            try:
                plot_runner.plot_obs(observation=obs,
                                     reward=reward,
                                     timestamp=timestamp,
                                     done=i == nb_timestep_played-1)
                array_ = pygame.surfarray.array3d(plot_runner.screen)
                if gif_name is not None:
                    total_array[i,:,:,:] = array_
            except PyGameQuit:
                break
        if gif_name is None:
            write_gif(total_array, gif_name, fps=max_fps)
