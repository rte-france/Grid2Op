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
import warnings
import time
import imageio

from grid2op.Exceptions import Grid2OpException
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Exceptions.PlotExceptions import PyGameQuit
from grid2op.Episode.EpisodeData import EpisodeData

class EpisodeReplay(object):
    """
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
        if not os.path.exists(agent_path):
            raise Grid2OpException("Nothing is found at \"{}\" where an agent path should have been.".format(agent_path))
        self.agent_path = agent_path
        self.episode_data = None

    def replay_episode(self, episode_id, fps=2.0, gif_name=None, display=True):
        """
        When called, this function will start the display of the episode in a "mini movie" format.

        Parameters
        ----------
        episode_id: ``str``
            ID of the episode to replay

        fps: ``float``
            Frames per second. When it's low, you will have more time to look at each frame, but the episode
            will last longer. When it's high, episode will be faster, but frames will stay less time on the screen.

        gif_name: ``str``
            If provided, a .gif file is saved in the episode folder with the name :gif_name:. 
            The .gif extension is appened by this function
        """
        # Check args
        path_ep = os.path.join(self.agent_path, episode_id)
        if not os.path.exists(path_ep):
            raise Grid2OpException("No episode is found at \"{}\".".format(path_ep))

        # Load episode observations
        self.episode_data = EpisodeData.from_disk(agent_path=self.agent_path, name=episode_id)
        all_obs = [el for el in self.episode_data.observations]

        # Create a plotter
        plot_runner = PlotMatplot(self.episode_data.observation_space)

        # Some vars for gif export if enabled
        frames = []
        gif_path = None
        if gif_name is not None:
            gif_path = os.path.join(path_ep, gif_name + ".gif")

        # Render loop
        figure = None
        time_per_frame = 1.0 / fps
        for obs in all_obs:
            # Get a timestamp for current frame
            start_time = time.time()

            # Render the observation
            fig = plot_runner.plot_obs(observation=obs, figure=figure, redraw=True)
            if figure is None and display:
                fig.show()
            else:
                fig.canvas.draw()

            # Store figure for re-use
            figure = fig
            # Save pixel array if needed
            if gif_name is not None:
                frames.append(plot_runner.convert_figure_to_numpy_HWC(figure))

            # Get the timestamp after frame is rendered
            end_time = time.time()
            delta_time = end_time - start_time

            # Cap fps for display mode
            if display:
                wait_time = time_per_frame - delta_time
                if wait_time > 0.0:
                    time.sleep(wait_time)

        # Export all frames as gif if enabled
        if gif_name is not None:
            imageio.mimwrite(gif_path, frames, fps=fps)
            # Try to compress
            try:
                from pygifsicle import optimize
                optimize(gif_path)
            except:
                warn_msg = "Failed to optimize .GIF size, but gif is still saved:\n" \
                           "Install dependencies to reduce size by ~3 folds\n" \
                           "apt-get install gifsicle && pip3 install pygifsicle"
                warnings.warn(warn_msg)

def replay_cli():    
    import argparse
    parser = argparse.ArgumentParser(description="EpisodeReplay")
    parser.add_argument("--agent_path", required=True, type=str)
    parser.add_argument("--episode_id", required=True, type=str)
    parser.add_argument("--display", required=False, default=False, action="store_true")
    parser.add_argument("--fps", required=False, default=2.0, type=float)
    parser.add_argument("--gif_name", required=False, default=None, type=str)
    args = parser.parse_args()
    er = EpisodeReplay(args.agent_path)
    er.replay_episode(args.episode_id,
                      fps=args.fps,
                      gif_name=args.gif_name,
                      display=args.display)

# Dev / Test by running this file
if __name__ == "__main__":
    replay_cli()
