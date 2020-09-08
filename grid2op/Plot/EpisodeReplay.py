# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import sys
import numpy as np
from datetime import datetime
import warnings

from grid2op.Episode import EpisodeData
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
    # from array2gif import write_gif
    import imageio
    import imageio_ffmpeg
    can_save_gif = True
except:
    can_save_gif = False


class EpisodeReplay(object):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

        Prefer using the class `grid2op.Episode.EpisodeReplay`

    This class allows to see visually what an agent has done during an episode. It uses for now the "PlotPygame" as the
    method to plot the different states of the system. It reads directly data from the runner.

    It can be used the following manner.

    .. code-block:: python

        import grid2op
        agent_class = grid2op.Agent.DoNothingAgent  # change that for studying other agent
        env = grid2op.make()  # make the default environment
        runner = grid2op.Runner.Runner(**env.get_params_for_runner(), agentClass=agent_class)
        path_log = "agent_log"  # where the runner will output the standardized data when running the agent.
        res = runner.run(nb_episode=1, path_save=path_log)

        # and when it's done, you can visualize it this way:
        episode_replay = EpisodeReplay(agent_path=path_log)
        episode_id = res[0][1]
        episode_replay.plot_episode(episode_id, max_fps=10)

        # you can pause by clicking the "space" key
        # At any time, you can quit by pressing the "esc" key or the "exit" button of the window.

    Attributes
    ----------
    agent_path: ``str``
        The path were the log of the agent are stored. It is recommended to use a :class:`grid2op.Runner.Runner`
        to save tha log of the agent.

    episode_data: :class:`grid2op.EpisodeData.EpisodeData`, optional
        The last data of the episode inspected.
    """
    def __init__(self, agent_path):
        warnings.warn("This whole class has been deprecated. Use `grid2op.PlotGrid module instead`",
                      category=DeprecationWarning)

        if not os.path.exists(agent_path):
            raise Grid2OpException("Nothing is found at \"{}\" where an agent path should have been.".format(agent_path))
        self.agent_path = agent_path
        self.episode_data = None

        if not can_save_gif:
            warnings.warn("The final video will not be saved as \"imageio\" and \"imageio_ffmpeg\" packages cannot be "
                          "imported. Please try \"{} -m pip install imageio imageio-ffmpeg\"".format(sys.executable))

    def replay_episode(self, episode_id, max_fps=10, video_name=None, display=True):
        """
        .. warning:: /!\\\\ This class is deprecated /!\\\\

            Prefer using the class `grid2op.Episode.EpisodeReplay`

        When called, this function will start the display of the episode in a "mini movie" format.

        Parameters
        ----------
        episode_id: ``str``
            ID of the episode to replay

        max_fps: ``int``
            Maximum "frame per second". When it's low, you will have more time to look at each frame, but the episode
            will last longer. When it's high, episode will be faster, but frames will stay less time on the screen.

        video_name: ``str``
            In beta mode for now. This allows to save the "video" of the episode in a gif or a mp4 for example.

        Returns
        -------

        """
        path_ep = os.path.join(self.agent_path, episode_id)
        if not os.path.exists(path_ep):
            raise Grid2OpException("No episode is found at \"{}\" where the episode should have been.".format(path_ep))
        if video_name is None:
            if not can_save_gif:
                raise Grid2OpException("The final video cannot be saved as \"imageio\" and \"imageio_ffmpeg\" "
                                       "packages cannot be imported. Please try "
                                       "\"{} -m pip install imageio imageio-ffmpeg\"".format(sys.executable))

        self.episode_data = EpisodeData.from_disk(agent_path=self.agent_path, name=episode_id)
        plot_runner = PlotPyGame(self.episode_data.observation_space,
                                 timestep_duration_seconds=1./max_fps)
        nb_timestep_played = int(self.episode_data.meta["nb_timestep_played"])
        all_obs = [el for el in self.episode_data.observations]
        all_reward = [el for el in self.episode_data.rewards]
        if video_name is not None:
            total_array = np.zeros((nb_timestep_played+1, plot_runner.video_width, plot_runner.video_height, 3),
                                   dtype=np.uint8)

        if display is False:
            plot_runner.deactivate_display()

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
                if video_name is not None:
                    total_array[i, :, :, :] = array_.astype(np.uint8)
            except PyGameQuit:
                break

        if video_name is not None:
            imageio.mimwrite(video_name, np.swapaxes(total_array, 1,2), fps=max_fps)
        plot_runner.close()
