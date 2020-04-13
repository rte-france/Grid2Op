import os
from grid2op.Plot import EpisodeReplay
path_agents = "getting_started/study_agent_getting_started"
max_iter = 30

path_agent = os.path.join(path_agents, "PowerLineSwitch")
plot_epi = EpisodeReplay(path_agent)
plot_epi.replay_episode("001", max_fps=5, video_name="test.mp4")
# plot_epi.plot_episode("001", max_fps=5, video_name="test.gif")
