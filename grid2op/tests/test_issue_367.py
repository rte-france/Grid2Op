# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import grid2op
import unittest
import warnings

from grid2op.tests.helper_path_test import *


class Issue367Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "test_issue_367"), test=True, _add_to_name=type(self).__name__)
        self.env.set_id(0) 
        self.env.seed(0)
        param = self.env.parameters
        param.NO_OVERFLOW_DISCONNECTION = True
        self.env.change_parameters(param)
        obs = self.env.reset()

    def test_action(self):
        gen_id = 2

        dn = self.env.action_space()
        for i in range(7):
            obs, reward, done, info = self.env.step(dn)
            if done:
                raise RuntimeError(f"done at step {i}")
            
        assert obs.target_dispatch[gen_id] == 0.0, f"should be 0., but is {obs.target_dispatch[gen_id]:.2f}"
        
        act = self.env.action_space()
        act.redispatch = [(gen_id, -3.5)]
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info['is_dispatching_illegal']
        assert obs.target_dispatch[gen_id] == -3.5, f"should be -3.5, but is {obs.target_dispatch[gen_id]:.2f}"

        act = self.env.action_space()
        act.redispatch = [(gen_id, +3.5)]
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert obs.target_dispatch[gen_id] == 0., f"should be 0., but is {obs.target_dispatch[gen_id]:.2f}"

        act = self.env.action_space()
        act.redispatch = [(gen_id, +1.5)]
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not info['is_dispatching_illegal']
        assert abs(obs.target_dispatch[gen_id] - 1.5) <= 1e-2, f"should be 1.5, but is {obs.target_dispatch[gen_id]:.2f}"

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    

if __name__ == "__main__":
    unittest.main()
