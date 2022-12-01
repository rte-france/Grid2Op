# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings

from grid2op.Runner import Runner


class Issue367Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022", test=True)
        self.env.set_id(0) 
        obs = self.env.reset()

    def test_action(self):
        gen_id = 2

        # Step 11
        act = self.env.action_space()
        act.redispatch = [(gen_id, -3.5)]
        obs, reward, done, info = self.env.step(act)
        assert obs.target_dispatch[gen_id] == -3.5

        # Step 12
        act = self.env.action_space()
        act.redispatch = [(gen_id, +3.5)]
        obs, reward, done, info = self.env.step(act)
        assert obs.target_dispatch[gen_id] == 0.

        # Step 13
        act = self.env.action_space()
        act.redispatch = [(gen_id, +1.5)]
        obs, reward, done, info = self.env.step(act)
        assert obs.target_dispatch[gen_id] == 1.5

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
        

if __name__ == "__main__":
    unittest.main()
