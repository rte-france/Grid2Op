import os
from grid2op.Plot import PlotEpisode
path_agents = "getting_started/study_agent_getting_started"
max_iter = 30

path_agent = os.path.join(path_agents, "PowerLineSwitch")
plot_epi = PlotEpisode(path_agent)
plot_epi.plot_episode("001", max_fps=5)