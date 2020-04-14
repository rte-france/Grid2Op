.. currentmodule:: grid2op.Plot

.. |randomagent| image:: img/random_agent.gif

Grid2Op Plotting capabilities (beta)
=====================================

Objectives
-----------
This section is a work in progress. Layout, coloring, display etc. will get better in a near future.

This module contrains all the plotting utilities of grid2op. These utilities can be used in different manners to serve
different purposes:

- :class:`PlotPygame` uses the library "pygame" to display a powergrid, this should mainly be used for the
  renderer for example and is particularly suited for looking at the overall dynamic without too much
  care taken for each step.
- :class:`PlotMatplotlib` allows a more in depth study, with a better overall layout. It uses the well-known
  matplotlib library to represent the powergrid on the screen.
- :class:`PlotPlotly` uses plotly library to represent the graph. As opposed to the others, plotly allows dynamic
  modifications such as zoom in / out. This makes this class particularly suited for in depth study of some
  powergrid state.


The class :class:`PlotPygame` is also used by :class:`EpisodeReplay` that allows to look at the action taken by
the agent pretty easily, and allows easy saving into gif or mp4 format.

.. code-block:: python3

    import os
    import warnings
    import grid2op
    from grid2op.Plot import EpisodeReplay
    from grid2op.Agent import GreedyAgent, RandomAgent
    from grid2op.Runner import Runner
    from tqdm import tqdm

    path_agents = "agent_pseudo_random"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make("case14_realistic")

    class CustomRandom(RandomAgent):
        """
        This agent takes 1 random action every 10 time steps.
        """
        def __init__(self, action_space):
            RandomAgent.__init__(self, action_space)
            self.i = 0

        def my_act(self, transformed_observation, reward, done=False):
            if self.i % 10 != 0:
                res = 0
            else:
                res = self.action_space.sample()
            self.i += 1
            return res

    # execute this agent on 1 scenario, saving the results
    runner = Runner(**env.get_params_for_runner(), agentClass=CustomRandom)
    path_agent = os.path.join(path_agents, "RandomAgent")
    res = runner.run(nb_episode=1, path_save=path_agent, pbar=tqdm)
    # and now reload it and display the "movie" of this scenario
    plot_epi = EpisodeReplay(path_agent)
    plot_epi.replay_episode(res[0][1], max_fps=2, video_name="random_agent.gif")


An possible output will look like this:

|randomagent|


TODO add images for other layout and grid2viz too

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Plot
    :members:
    :special-members:
    :autosummary:

.. include:: final.rst
