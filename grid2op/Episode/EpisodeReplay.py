# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import warnings
import time
import imageio

import argparse

from grid2op.Exceptions import Grid2OpException
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Episode.EpisodeData import EpisodeData


class EpisodeReplay(object):
    """
    This class allows to see visually what an agent has done during an episode. It uses for now the "PlotMatplot" as the
    method to plot the different states of the system. It reads directly data from the runner.

    Examples
    --------

    It can be used the following manner.

    .. code-block:: python

        import grid2op
        agent_class = grid2op.Agent.DoNothingAgent  # change that for studying other agent
        env = grid2op.make("l2rpn_case14_sandbox")  # make the default environment
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
        The last data of the episode inspected.replay_cli
    """

    def __init__(self, agent_path):
        if not os.path.exists(agent_path):
            raise Grid2OpException(
                'Nothing is found at "{}" where an agent path should have been.'.format(
                    agent_path
                )
            )
        self.agent_path = agent_path
        self.episode_data = None

    def replay_episode(
        self,
        episode_id,
        fps=2.0,
        gif_name=None,
        display=True,
        start_step=0,
        end_step=-1,
        line_info="rho",
        load_info="p",
        gen_info="p",
        resolution=(1280, 720),
    ):
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
            The .gif extension is happened by this function

        start_step: ``int``
            Default to 0. The step at which to start generating the gif

        end_step: ``int``
            Default to -1. The step at which to stop generating the gif.
            Set to -1 to specify no limit

        load_info: ``str``
            Defaults to "p". What kind of values to show on loads.
            Can be oneof `["p", "v", None]`

        gen_info: ``str``
            Defaults to "p". What kind of values to show on generators.
            Can be oneof `["p", "v", None]`

        line_info: ``str``
            Defaults to "rho". What kind of values to show on lines.
            Can be oneof `["rho", "a", "p", "v", None]`

        resolution: ``tuple``
            Defaults to (1280, 720). The resolution to use for the gif.
        """
        # Check args
        path_ep = os.path.join(self.agent_path, episode_id)
        if not os.path.exists(path_ep):
            raise Grid2OpException('No episode is found at "{}".'.format(path_ep))

        # Load episode observations
        self.episode_data = EpisodeData.from_disk(
            agent_path=self.agent_path, name=episode_id
        )
        all_obs = [el for el in self.episode_data.observations]
        # Create a plotter
        width, height = resolution
        plot_runner = PlotMatplot(
            self.episode_data.observation_space,
            width=width,
            height=height,
            load_name=False,
            gen_name=False,
        )

        # Some vars for gif export if enabled
        frames = []
        gif_path = None
        if gif_name is not None:
            gif_path = os.path.join(path_ep, gif_name + ".gif")

        # Render loop
        figure = None
        time_per_frame = 1.0 / fps
        for step, obs in enumerate(all_obs):
            # Skip up to start_step
            if step < start_step:
                continue
            # Terminate if reached end_step
            if end_step > 0 and step >= end_step:
                break
            # Get a timestamp for current frame
            start_time = time.perf_counter()

            # Render the observation
            fig = plot_runner.plot_obs(
                observation=obs,
                line_info=line_info,
                gen_info=gen_info,
                load_info=load_info,
                figure=figure,
                redraw=True,
            )
            if figure is None and display:
                fig.show()
            elif display:
                fig.canvas.draw()

            # Store figure for re-use
            figure = fig
            # Save pixel array if needed
            if gif_name is not None:
                frames.append(plot_runner.convert_figure_to_numpy_HWC(figure))

            # Get the timestamp after frame is rendered
            end_time = time.perf_counter()
            delta_time = end_time - start_time

            # Cap fps for display mode
            if display:
                wait_time = time_per_frame - delta_time
                if wait_time > 0.0:
                    time.sleep(wait_time)

        # Export all frames as gif if enabled
        if gif_name is not None and len(frames) > 0:
            try:
                imageio.mimwrite(gif_path, frames, fps=fps)
                # Try to compress
                try:
                    from pygifsicle import optimize

                    optimize(gif_path, options=["-w", "--no-conserve-memory"])
                except:
                    warn_msg = (
                        "Failed to optimize .GIF size, but gif is still saved:\n"
                        "Install dependencies to reduce size by ~3 folds\n"
                        "apt-get install gifsicle && pip3 install pygifsicle"
                    )
                    warnings.warn(warn_msg)
            except Exception as e:
                warnings.warn("Impossible to save gif with error :\n{}".format(e))


def episode_replay_cli():
    parser = argparse.ArgumentParser(description="EpisodeReplay")
    parser.add_argument("--agent_path", required=True, type=str)
    parser.add_argument("--episode_id", required=True, type=str)
    parser.add_argument("--display", required=False, default=False, action="store_true")
    parser.add_argument("--fps", required=False, default=2.0, type=float)
    parser.add_argument("--gif_name", required=False, default=None, type=str)
    parser.add_argument("--gif_start", required=False, default=0, type=int)
    parser.add_argument("--gif_end", required=False, default=-1, type=int)
    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = episode_replay_cli()
    er = EpisodeReplay(args.agent_path)
    er.replay_episode(
        args.episode_id,
        fps=args.fps,
        gif_name=args.gif_name,
        start_step=args.gif_start,
        end_step=args.gif_end,
        display=args.display,
    )


# Dev / Test by running this file
if __name__ == "__main__":
    args = episode_replay_cli()
    main(args)
