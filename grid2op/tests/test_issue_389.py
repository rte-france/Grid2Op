# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import numpy as np
import pdb

    
import grid2op
import numpy as np


class Issue389Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)

    def test_issue(self):
        act = self.env.action_space({"set_bus":{ "substations_id": [(4, (2, 1, 2))]}})
        obs, reward, done, info = self.env.step(act)
        act = self.env.action_space({"set_bus":{ "lines_or_id": [(7, -1)]}})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert not np.isnan(obs.theta_ex[-1])
        G = obs.get_energy_graph()
        assert not np.isnan(G.nodes[4]["theta"])
        assert G.edges[(0, 4)]["theta_or"] == G.nodes[0]["theta"]
        assert G.edges[(0, 4)]["theta_ex"] == G.nodes[4]["theta"]

        assert G.edges[(0, 4)]["v_or"] == G.nodes[0]["v"]
        assert G.edges[(0, 4)]["v_ex"] == G.nodes[4]["v"]

if __name__ == "__main__":
    unittest.main()
