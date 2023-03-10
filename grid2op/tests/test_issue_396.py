# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import os
import pdb

import tempfile
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeReplay, EpisodeData


class Issue396Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("rte_case5_example", test=True)
        self.env.seed(0)
        self.env.set_id(0)
    
    def test_gif(self):
        with tempfile.TemporaryDirectory() as path:
            runner = Runner(**self.env.get_params_for_runner())
            _ = runner.run(path_save=path,
                           nb_episode=1,
                           nb_process=1,
                           max_iter=10,
                            )
            li_ep = EpisodeData.list_episode(path)
            ep_replay = EpisodeReplay(li_ep[0][0])
            ep_replay.replay_episode(episode_id=li_ep[0][1],
                                     gif_name=li_ep[0][1],
                                     display=False)


if __name__ == "__main__":
    unittest.main()
