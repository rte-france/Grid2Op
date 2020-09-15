import unittest
import warnings
import grid2op
from grid2op.Agent import DeltaRedispatchRandomAgent
from grid2op.Runner import Runner
from grid2op import make
from grid2op.Episode import EpisodeData
import os
import numpy as np
import tempfile

class Issue126Tester(unittest.TestCase):

  def test_issue_126(self):
    with tempfile.TemporaryDirectory() as tmpdirname:
      #run redispatch agent on one scenario for 100 timesteps
      dataset = "rte_case14_realistic"
      nb_episode=1
      nb_timesteps=100

      with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = make(dataset, test=True)
        agent = DeltaRedispatchRandomAgent(env.action_space)
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        nb_episode=1
        res = runner.run(nb_episode=nb_episode,
                         path_save=tmpdirname,
                         nb_process=1,
                         max_iter=nb_timesteps,
                         env_seeds=[0],
                         agent_seeds=[0],
                         pbar=False)

        episode_data = EpisodeData.from_disk(tmpdirname, '000')

        assert len(episode_data.actions.objects) == nb_timesteps
        assert len(episode_data.observations.objects) == (nb_timesteps + 1)
        assert len(episode_data.actions) == nb_timesteps
        assert len(episode_data.observations) == (nb_timesteps + 1)

if __name__ == "__main__":
    unittest.main()
