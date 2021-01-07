# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

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
import pdb


class Issue126Tester(unittest.TestCase):
    def test_issue_126(self):
        # run redispatch agent on one scenario for 100 timesteps
        dataset = "rte_case14_realistic"
        nb_episode = 1
        nb_timesteps = 100
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(dataset, test=True)

        agent = DeltaRedispatchRandomAgent(env.action_space)
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=agent)
        with tempfile.TemporaryDirectory() as tmpdirname:
            res = runner.run(nb_episode=nb_episode,
                             path_save=tmpdirname,
                             nb_process=1,
                             max_iter=nb_timesteps,
                             env_seeds=[0],
                             agent_seeds=[0],
                             pbar=False)
            episode_data = EpisodeData.from_disk(tmpdirname, res[0][1])
            
        assert len(episode_data.actions.objects) - nb_timesteps == 0, "wrong number of actions {}".format(len(episode_data.actions.objects))
        assert len(episode_data.actions) - nb_timesteps == 0, "wrong number of actions {}".format(len(episode_data.actions))
        assert len(episode_data.observations.objects) - (nb_timesteps + 1) == 0, "wrong number of observations: {}".format(len(episode_data.observations.objects))
        assert len(episode_data.observations) - (nb_timesteps + 1) == 0, "wrong number of observations {}".format( len(episode_data.observations))


if __name__ == "__main__":
    unittest.main()
