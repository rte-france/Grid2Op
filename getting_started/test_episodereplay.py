import os
import warnings
import grid2op
from grid2op.Plot import EpisodeReplay
from grid2op.Agent import GreedyAgent, RandomAgent
from grid2op.Runner import Runner
from tqdm import tqdm

path_agents = "getting_started/study_agent_getting_started"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    env = grid2op.make("case14_realistic")


class CustomRandom(RandomAgent):
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


runner = Runner(**env.get_params_for_runner(), agentClass=CustomRandom)
path_agent = os.path.join(path_agents, "RandomAgent")
res = runner.run(nb_episode=2, path_save=path_agent, pbar=tqdm)

ep_replay = EpisodeReplay(agent_path=path_agent)
for _, chron_name, cum_reward, nb_time_step, max_ts in res:
    ep_replay.replay_episode(chron_name,
                             video_name=os.path.join(path_agent, chron_name, "epidose.gif"),
                             display=False)
if False:
    plot_epi = EpisodeReplay(path_agent)
    #plot_epi.replay_episode("001", max_fps=5, video_name="test.mp4")
    plot_epi.replay_episode(res[0][1], max_fps=2, video_name="random_agent.gif")
