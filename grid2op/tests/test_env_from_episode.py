# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import grid2op
from grid2op.Runner import Runner


class TestTSFromEpisode(unittest.TestCase):
    def setUp(self) -> None:
        env_name = "l2rpn_idf_2023"  # with maintenance and attacks !
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_name)
        return super().setUp()
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_basic(self):
        obs = self.env.reset()
        runner = Runner(
            **self.env.get_params_for_runner(),
            agentClass=None
        )

        # test that the right seeds are assigned to the agent
        res = runner.run(nb_episode=1, max_iter=self.max_iter, add_detailed_output=True)
        ep_stat = res[-1]
        
if __name__ == "__main__":
    unittest.main()